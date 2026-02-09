import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandlerGMSH
from mpi4py import MPI
import dolfinx as do
import torch as to
import numpy as np
from petsc4py import PETSc
import Utils as utils
from MaterialProps import *
from HeatEquation import HeatDiffusion
from MomentumEquation import LinearMomentumPrimal, LinearMomentumMixed, LinearMomentumMixedStar
import HeatBC as heatBC
import MomentumBC as momBC
from CharacteristicLength import ModelML
from OutputHandler import SaveFields
from Simulators import Simulator_TM, Simulator_M
from TimeHandler import TimeController, TimeControllerParabolic
import time

MPa = 1e6
GPa = 1e9
hour = 60*60
day = 24*hour

def get_geometry_parameters(path_to_grid):
	f = open(os.path.join(path_to_grid, "geom.geo"), "r")
	data = f.readlines()
	ovb_thickness = float(data[10][len("ovb_thickness = "):-2])
	salt_thickness = float(data[11][len("salt_thickness = "):-2])
	hanging_wall = float(data[12][len("hanging_wall = "):-2])
	return ovb_thickness, salt_thickness, hanging_wall


class LinearMomentumPrimalMod(LinearMomentumPrimal):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C.x.array[:] = to.flatten(self.mat.C)
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_th = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_th.x.array[:] = to.flatten(self.mat.elems_th[0].eps_th)
		except:
			pass

class LinearMomentumMixedMod(LinearMomentumMixed):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_th = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		self.eps_th.x.array[:] = to.flatten(self.mat.elems_th[0].eps_th)

class LinearMomentumMixedModStar(LinearMomentumMixedStar):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_th = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		# self.eps_th.x.array[:] = to.flatten(self.mat.elems_th[0].eps_th)





def run_case(formulation: str, beta: float = 0.0) -> None:
	comm = MPI.COMM_WORLD
	comm.Barrier()
	if MPI.COMM_WORLD.rank == 0:
	    start_time = MPI.Wtime()

	# Read grid
	grid_path = os.path.join("grids", "grid_0")
	grid = GridHandlerGMSH("geom", grid_path)

	# Read gas P and T
	gas_data = utils.read_json("gas_P_T.json")
	p_gas = -np.array(gas_data["Pressure"])
	T_gas = np.array(gas_data["Temperature"])
	time_sim = np.array(gas_data["Time"])

	# Define output folder
	if formulation == "mixed_star":
		output_folder = os.path.join("output", f"case_iE", f"{formulation}_beta_{beta}")
	else:
		output_folder = os.path.join("output", f"case_iE", f"{formulation}")

	# Define momentum equation
	theta = 0.0
	if formulation == "primal":
		mom_eq = LinearMomentumPrimalMod(grid, theta)
	elif formulation == "mixed":
		mom_eq = LinearMomentumMixedMod(grid, theta)
	elif formulation == "mixed_stab":
		mom_eq = LinearMomentumMixedMod(grid, theta)
		model_h = ModelML()
		h = model_h.compute_mesh_h(grid.mesh)
		mom_eq.set_stabilization_h(h)
	elif formulation == "mixed_star":
		mom_eq = LinearMomentumMixedModStar(grid, theta)
		model_h = ModelML()
		h = model_h.compute_mesh_h(grid.mesh)
		mom_eq.set_stabilization_h(beta*h)
	else:
		raise Exception(f"Formulation {formulation} not supported.")

	# Define solver
	mom_solver = PETSc.KSP().create(grid.mesh.comm)
	mom_solver.setType("gmres")
	mom_solver.getPC().setType("asm")
	mom_solver.setTolerances(rtol=1e-12, max_it=100)
	mom_eq.set_solver(mom_solver)

	# Define material properties
	mat = Material(mom_eq.n_elems)

	# Extract region indices
	ind_salt = grid.region_indices["Salt"]
	ind_ovb = grid.region_indices["Overburden"]

	# Set material density
	salt_density = 2200
	ovb_density = 2800
	gas_density = 0.082
	rho = to.zeros(mom_eq.n_elems, dtype=to.float64)
	rho[ind_salt] = salt_density
	rho[ind_ovb] = ovb_density
	mat.set_density(rho)

	# Constitutive model
	E0 = to.zeros(mom_eq.n_elems)
	E0[ind_salt] = 102*GPa
	E0[ind_ovb] = 180*GPa
	nu0 = 0.3*to.ones(mom_eq.n_elems)
	spring_0 = Spring(E0, nu0, "spring")

	# Create Kelvin-Voigt viscoelastic element
	eta = 105e11*to.ones(mom_eq.n_elems)
	E1 = 10*utils.GPa*to.ones(mom_eq.n_elems)
	nu1 = 0.32*to.ones(mom_eq.n_elems)
	kelvin = Viscoelastic(eta, E1, nu1, "kelvin")

	# Create creep
	A = to.zeros(mom_eq.n_elems)
	A[ind_salt] = 1.9e-21
	A[ind_ovb] = 0.0
	Q = 51600*to.ones(mom_eq.n_elems)
	n = 3.0*to.ones(mom_eq.n_elems)
	creep_ds = DislocationCreep(A, Q, n, "creep_ds")

	# Create pressure-solution creep
	A = to.zeros(mom_eq.n_elems)
	A[ind_salt] = 1.29e-19
	A[ind_ovb] = 0.0
	Q = 72819*to.ones(mom_eq.n_elems)
	d = 0.01*to.ones(mom_eq.n_elems)
	creep_ps = PressureSolutionCreep(A, d, Q, "creep_ps")

	# Thermo-elastic element
	# alpha = 120e-6*to.ones(mom_eq.n_elems)
	alpha = to.zeros(mom_eq.n_elems)
	alpha[ind_salt] = 4e-5
	alpha[ind_ovb] = 0.0
	thermo = Thermoelastic(alpha, "thermo")

	# Create constitutive model
	mat.add_to_elastic(spring_0)
	mat.add_to_non_elastic(kelvin)
	mat.add_to_non_elastic(creep_ds)
	mat.add_to_non_elastic(creep_ps)
	mat.add_to_thermoelastic(thermo)

	# Set constitutive model
	mom_eq.set_material(mat)

	# Set body forces
	g = -9.81
	g_vec = [0.0, 0.0, g]
	mom_eq.build_body_force(g_vec)

	# Set initial temperature field
	T_SURFACE = 20 + 273
	km = 1000
	dTdZ = 27/km
	T_field_fun = lambda x,y,z: T_SURFACE - dTdZ*z
	T0_field = utils.create_field_elems(grid, T_field_fun)
	mom_eq.set_T0(T0_field)
	mom_eq.set_T(T0_field)

	# Time settings for equilibrium stage
	tc_eq = TimeControllerParabolic(final_time=50, initial_time=0.0, n_time_steps=20, time_unit="day")

	# Boundary conditions
	time_values = [tc_eq.t_initial,  tc_eq.t_final]
	nt = len(time_values)

	bc_west_salt = momBC.DirichletBC(boundary_name="West_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
	bc_west_ovb = momBC.DirichletBC(boundary_name = "West_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

	bc_east_salt = momBC.DirichletBC(boundary_name="East_salt", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
	bc_east_ovb = momBC.DirichletBC(boundary_name = "East_ovb", component=0, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

	bc_bottom = momBC.DirichletBC(boundary_name="Bottom", component=2, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

	bc_south_salt = momBC.DirichletBC(boundary_name="South_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
	bc_south_ovb = momBC.DirichletBC(boundary_name="South_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

	bc_north_salt = momBC.DirichletBC(boundary_name="North_salt", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])
	bc_north_ovb = momBC.DirichletBC(boundary_name="North_ovb", component=1, values=[0.0, 0.0], time_values=[0.0, tc_eq.t_final])

	# Extract geometry dimensions
	Lx = grid.Lx
	Ly = grid.Ly
	Lz = grid.Lz
	z_surface = 0.0

	g = 9.81
	ovb_thickness, salt_thickness, hanging_wall = get_geometry_parameters(grid_path)
	cavern_roof = ovb_thickness + hanging_wall
	p_roof = 0 + salt_density*g*hanging_wall + ovb_density*g*ovb_thickness

	if MPI.COMM_WORLD.rank == 0:
		print(cavern_roof)
		print(p_roof/MPa)
		print(0.8*p_roof/MPa)
		print(0.2*p_roof/MPa)
		print(p_gas[0])

	# Pressure at the top of the salt layer (bottom of overburden)
	p_top = ovb_density*g*ovb_thickness

	bc_top = momBC.NeumannBC(boundary_name = "Top",
						direction = 2,
						density = 0.0,
						ref_pos = z_surface,
						values = [0*MPa, 0*MPa],
						time_values = time_values,
						g = g_vec[2])

	bc_cavern = momBC.NeumannBC(boundary_name = "Cavern",
						direction = 2,
						density = gas_density,
						ref_pos = cavern_roof,
						values = [p_gas[0], p_gas[0]],
						time_values = time_values,
						g = g_vec[2])

	bc_equilibrium = momBC.BcHandler(mom_eq)
	bc_equilibrium.add_boundary_condition(bc_west_salt)
	bc_equilibrium.add_boundary_condition(bc_west_ovb)
	bc_equilibrium.add_boundary_condition(bc_east_salt)
	bc_equilibrium.add_boundary_condition(bc_east_ovb)
	bc_equilibrium.add_boundary_condition(bc_bottom)
	bc_equilibrium.add_boundary_condition(bc_south_salt)
	bc_equilibrium.add_boundary_condition(bc_south_ovb)
	bc_equilibrium.add_boundary_condition(bc_north_salt)
	bc_equilibrium.add_boundary_condition(bc_north_ovb)
	bc_equilibrium.add_boundary_condition(bc_top)
	bc_equilibrium.add_boundary_condition(bc_cavern)


	# Set boundary conditions
	mom_eq.set_boundary_conditions(bc_equilibrium)

	# Equilibrium output folder
	ouput_folder_equilibrium = os.path.join(output_folder, "equilibrium")

	# Print output folder
	if MPI.COMM_WORLD.rank == 0:
		print(ouput_folder_equilibrium)

	# Create output handlers
	output_mom = SaveFields(mom_eq)
	output_mom.set_output_folder(ouput_folder_equilibrium)
	output_mom.add_output_field("u", "Displacement (m)")
	output_mom.add_output_field("p_elems", "Mean stress (Pa)")
	output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
	outputs = [output_mom]

	# Define simulator
	sim = Simulator_M(mom_eq, tc_eq, outputs, True)
	sim.run()

	# Print time
	if MPI.COMM_WORLD.rank == 0:
		end_time = MPI.Wtime()
		elaspsed_time = end_time - start_time
		formatted_time = time.strftime("%H:%M:%S", time.gmtime(elaspsed_time))
		print(f"Time: {formatted_time} ({elaspsed_time} seconds)\n")





	# Time settings for operation stage
	tc_op = TimeController(time_step=0.5, final_time=100, initial_time=0.0, time_unit="day")

	# Define heat diffusion equation
	heat_eq = HeatDiffusion(grid)

	# Define solver
	solver_heat = PETSc.KSP().create(grid.mesh.comm)
	solver_heat.setType("cg")
	solver_heat.getPC().setType("asm")
	solver_heat.setTolerances(rtol=1e-12, max_it=100)
	heat_eq.set_solver(solver_heat)

	# Set specific heat capacity
	cp = 800*to.ones(heat_eq.n_elems, dtype=to.float64)
	mat.set_specific_heat_capacity(cp)

	# Set thermal conductivity
	k = 5.5*to.ones(heat_eq.n_elems, dtype=to.float64)
	mat.set_thermal_conductivity(k)

	# Set material properties to heat_equation
	heat_eq.set_material(mat)

	# Set initial temperature
	T0_field_nodes = utils.create_field_nodes(grid, T_field_fun)
	heat_eq.set_initial_T(T0_field_nodes)

	# Define boundary conditions for heat diffusion
	time_values = [tc_op.t_initial, tc_op.t_final]
	nt = len(time_values)

	bc_handler = heatBC.BcHandler(heat_eq)

	bc_top = heatBC.DirichletBC("Top", nt*[T_SURFACE], time_values)
	bc_handler.add_boundary_condition(bc_top)

	bc_bottom = heatBC.NeumannBC("Bottom", nt*[dTdZ], time_values)
	bc_handler.add_boundary_condition(bc_bottom)

	h_conv = 5.0
	bc_cavern = heatBC.RobinBC("Cavern", T_gas, h_conv, time_sim)
	bc_handler.add_boundary_condition(bc_cavern)

	heat_eq.set_boundary_conditions(bc_handler)





	# Set operation stage settings for momentum equation

	# Boundary conditions
	bc_bottom = momBC.DirichletBC("Bottom", 2, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_west_salt = momBC.DirichletBC("West_salt", 0, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_west_ovb = momBC.DirichletBC("West_ovb", 0, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_east_salt = momBC.DirichletBC("East_salt", 0, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_east_ovb = momBC.DirichletBC("East_ovb", 0, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_south_salt = momBC.DirichletBC("South_salt", 1, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_south_ovb = momBC.DirichletBC("South_ovb", 1, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_north_salt = momBC.DirichletBC("North_salt", 1, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_north_ovb = momBC.DirichletBC("North_ovb", 1, [0.0, 0.0], [0.0, tc_op.t_final])
	bc_cavern = momBC.NeumannBC("Cavern", 2, gas_density, -cavern_roof, p_gas, time_sim, g_vec[2])

	bc_operation = momBC.BcHandler(mom_eq)
	bc_operation.add_boundary_condition(bc_west_salt)
	bc_operation.add_boundary_condition(bc_west_ovb)
	bc_operation.add_boundary_condition(bc_bottom)
	bc_operation.add_boundary_condition(bc_south_salt)
	bc_operation.add_boundary_condition(bc_south_ovb)
	bc_operation.add_boundary_condition(bc_east_salt)
	bc_operation.add_boundary_condition(bc_east_ovb)
	bc_operation.add_boundary_condition(bc_north_salt)
	bc_operation.add_boundary_condition(bc_north_ovb)
	bc_operation.add_boundary_condition(bc_cavern)

	# Set boundary conditions
	mom_eq.set_boundary_conditions(bc_operation)

	# Define output folder
	output_folder_operation = os.path.join(output_folder, "operation")

	# Print output folder
	if MPI.COMM_WORLD.rank == 0:
		print(output_folder_operation)

	# Create output handlers
	output_mom = SaveFields(mom_eq, skip=2)
	output_mom.set_output_folder(output_folder_operation)
	output_mom.add_output_field("u", "Displacement (m)")
	output_mom.add_output_field("p_elems", "Mean stress (Pa)")
	output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
	output_mom.add_output_field("p_nodes", "Mean stress (Pa)")
	output_mom.add_output_field("q_nodes", "Von Mises stress (Pa)")
	output_mom.add_output_field("eps_cr", "Creep strain (-)")
	# output_mom.add_output_field("eps_th", "Thermal strain (-)")
	output_mom.add_output_field("eps_tot", "Total strain (-)")

	output_heat = SaveFields(heat_eq, skip=2)
	output_heat.set_output_folder(output_folder_operation)
	output_heat.add_output_field("T", "Temperature (K)")

	outputs = [output_mom, output_heat]

	# Define simulator
	sim = Simulator_TM(mom_eq, heat_eq, tc_op, outputs, False)
	sim.run()

	# Print time
	if MPI.COMM_WORLD.rank == 0:
		end_time = MPI.Wtime()
		elaspsed_time = end_time - start_time
		formatted_time = time.strftime("%H:%M:%S", time.gmtime(elaspsed_time))
		print(f"Time: {formatted_time} ({elaspsed_time} seconds)\n")

def main():
	run_case(formulation="primal")
	run_case(formulation="mixed")
	run_case(formulation="mixed_stab")
	run_case(formulation="mixed_star", beta=1.0)


if __name__ == '__main__':
	main()