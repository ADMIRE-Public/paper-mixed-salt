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
from MomentumEquation import LinearMomentumPrimal, LinearMomentumMixedStar, LinearMomentumMixed
from CharacteristicLength import ModelML
import HeatBC as heatBC
import MomentumBC as momBC
from OutputHandler import SaveFields
from Simulators import Simulator_M
from TimeHandler import TimeController, TimeControllerParabolic, TimeControllerFvp, TimeControllerEpsilonRate
import time

class LinearMomentumPrimalMod(LinearMomentumPrimal):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C.x.array[:] = to.flatten(self.mat.C)
		self.eps_ve = do.fem.Function(self.DG0_3x3)
		self.eps_ds = do.fem.Function(self.DG0_3x3)
		self.eps_ps = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)
		self.Fvp = do.fem.Function(self.DG0_1)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_ds.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_ps.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[3].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[3].Fvp
		except:
			pass

class LinearMomentumMixedMod(LinearMomentumMixed):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C_tilde.x.array[:] = self.mat.C_tilde.flatten()
		self.C_tilde_inv.x.array[:] = self.mat.C_tilde_inv.flatten()
		self.K.x.array[:] = self.mat.K
		self.E.x.array[:] = self.mat.E
		self.Fvp = do.fem.Function(self.DG0_1)
		self.eps_ve = do.fem.Function(self.DG0_3x3)
		self.eps_ds = do.fem.Function(self.DG0_3x3)
		self.eps_ps = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_ds.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_ps.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[3].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[3].Fvp
		except:
			pass

class LinearMomentumMixedModStar(LinearMomentumMixedStar):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C_tilde.x.array[:] = self.mat.C_tilde.flatten()
		self.C_tilde_inv.x.array[:] = self.mat.C_tilde_inv.flatten()
		self.K.x.array[:] = self.mat.K
		self.E.x.array[:] = self.mat.E
		self.Fvp = do.fem.Function(self.DG0_1)
		self.eps_ve = do.fem.Function(self.DG0_3x3)
		self.eps_ds = do.fem.Function(self.DG0_3x3)
		self.eps_ps = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_ds.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_ps.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[3].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[3].Fvp
		except:
			pass



def run_case(formulation):
	comm = MPI.COMM_WORLD
	comm.Barrier()
	if MPI.COMM_WORLD.rank == 0:
	    start_time = MPI.Wtime()

	# Read grid
	grid_path = os.path.join("grids", "cube_0")
	grid = GridHandlerGMSH("geom", grid_path)

	# Define output folder
	output_folder = os.path.join("output", "case_0", formulation)

	# Time settings for equilibrium stage
	dt_min = 0.01
	dt_max = 0.2
	t_final = 38
	t_control = TimeControllerEpsilonRate(dt_min=dt_min, dt_max=dt_max, dt_init=dt_max, final_time=t_final, initial_time=0, time_unit="day")

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
		mom_eq.set_stabilization_h(h)
	else:
		raise Exception(f"Formulation {formulation} not supported.")


	mom_solver = PETSc.KSP().create(grid.mesh.comm)
	mom_solver.setType("bicg")
	mom_solver.getPC().setType("asm")
	mom_solver.setTolerances(rtol=1e-12, max_it=100)
	mom_eq.set_solver(mom_solver)

	# Build material properties
	mat = Material(mom_eq.n_elems)

	# Set density
	rho = 0.0*to.ones(mom_eq.n_elems)
	mat.set_density(rho)

	# Create elastic element
	E = 102e9*to.ones(mom_eq.n_elems)
	nu = 0.3*to.ones(mom_eq.n_elems)
	spring_0 = Spring(E, nu, "spring")

	# Create Kelvin-Voigt viscoelastic element
	E = 10e9*to.ones(mom_eq.n_elems)
	nu = 0.32*to.ones(mom_eq.n_elems)
	eta = 105e11*to.ones(mom_eq.n_elems)
	kelvin = Viscoelastic(eta, E, nu, "kelvin")

	# Create dislocation creep element
	A = 1.1e-21*to.ones(mom_eq.n_elems)
	Q = 51600*to.ones(mom_eq.n_elems)
	n = 3.0*to.ones(mom_eq.n_elems)
	creep_ds = DislocationCreep(A, Q, n, "creep")

	# Create pressure-solution creep
	A = 1.29e-19*to.ones(mom_eq.n_elems)
	Q = 72819*to.ones(mom_eq.n_elems)
	d = 0.01*to.ones(mom_eq.n_elems)
	creep_ps = PressureSolutionCreep(A, d, Q, "creep_ps")

	# Create Desai's viscoplastic model
	mu_1 = 5.3665857009859815e-11*to.ones(mom_eq.n_elems)
	N_1 = 3.1*to.ones(mom_eq.n_elems)
	n = 3.0*to.ones(mom_eq.n_elems)
	a_1 = 1.965018496922832e-05*to.ones(mom_eq.n_elems)
	eta = 0.8275682807874163*to.ones(mom_eq.n_elems)
	beta_1 = 0.0048*to.ones(mom_eq.n_elems)
	beta = 0.995*to.ones(mom_eq.n_elems)
	m = -0.5*to.ones(mom_eq.n_elems)
	gamma = 0.095*to.ones(mom_eq.n_elems)
	alpha_0 = 0.0018*to.ones(mom_eq.n_elems)
	sigma_t = 5.0*to.ones(mom_eq.n_elems)
	desai = ViscoplasticDesai(mu_1, N_1, a_1, eta, n, beta_1, beta, m, gamma, sigma_t, alpha_0, "desai")

	# Create constitutive model
	mat.add_to_elastic(spring_0)
	mat.add_to_non_elastic(kelvin)
	mat.add_to_non_elastic(creep_ds)
	mat.add_to_non_elastic(creep_ps)
	mat.add_to_non_elastic(desai)

	# Set constitutive model
	mom_eq.set_material(mat)

	try:
		t_control.set_eq_mom(mom_eq)
	except:
		pass

	try:
		# Set material to dynamic time control
		t_control.set_material(mat)
	except:
		pass

	# Set body forces
	g = -9.81
	g_vec = [0.0, 0.0, g]
	mom_eq.build_body_force(g_vec)

	# Set temperature
	T_elems = 298*to.ones(mom_eq.n_elems)
	mom_eq.set_T0(T_elems)
	mom_eq.set_T(T_elems)

	sA = 13
	sB = 26
	sC = 27.5
	sD = 31
	sE = 33.5

	load_hist = np.array([
		[0.0, 0.0,    0.1],
		[26., sA-0.1,  sA],
		[52., sA-0.1,  sB],
		[148, sA-0.1,  sB],
		[174, sA-0.1,  sA],
		[222, sA-0.1,  sA],
		[251, sA-0.1,  sC],
		[347, sA-0.1,  sC],
		[376, sA-0.1,  sA],
		[424, sA-0.1,  sA],
		[460, sA-0.1,  sD],
		[556, sA-0.1,  sD],
		[592, sA-0.1,  sA],
		[640, sA-0.1,  sA],
		[681, sA-0.1,  sE],
		[777, sA-0.1,  sE],
		[818, sA-0.1,  sA],
		[866, sA-0.1,  sA],
	])
	time_values = load_hist[:,0]*utils.hour
	s3_values = load_hist[:,1]*utils.MPa
	s1_values = load_hist[:,2]*utils.MPa
	nt = len(time_values)

	if MPI.COMM_WORLD.rank == 0:
		load_data = {
			"Time": list(time_values),
			"s1": list(s1_values),
			"s3": list(s3_values)
		}
		if not os.path.exists(output_folder):
			os.makedirs(output_folder, exist_ok=True)
		utils.save_json(load_data, os.path.join(output_folder, "loads.json"))

	# Boundary conditions
	bc_bottom = momBC.DirichletBC(boundary_name = "BOTTOM", 
					 		component = 2,
							values = nt*[0.0],
							time_values = time_values)

	bc_west = momBC.DirichletBC(boundary_name = "WEST", 
					 	  component = 0,
					 	  values = nt*[0.0],
					 	  time_values = time_values)

	bc_south = momBC.DirichletBC(boundary_name = "SOUTH", 
					 	   component = 1,
					 	   values = nt*[0.0],
					 	   time_values = time_values)

	bc_east = momBC.NeumannBC(boundary_name = "EAST",
						direction = 0,
						density = 0.0,
						ref_pos = .0,
						values = s3_values,
						time_values = time_values,
						g = g_vec[2])

	bc_north = momBC.NeumannBC(boundary_name = "NORTH",
						direction = 0,
						density = 0.0,
						ref_pos = .0,
						values = s3_values,
						time_values = time_values,
						g = g_vec[2])

	bc_top = momBC.NeumannBC(boundary_name = "TOP",
						direction = 2,
						density = 0.0,
						ref_pos = .0,
						values = s1_values,
						time_values = time_values,
						g = g_vec[2])


	bc_handler = momBC.BcHandler(mom_eq)
	bc_handler.add_boundary_condition(bc_bottom)
	bc_handler.add_boundary_condition(bc_west)
	bc_handler.add_boundary_condition(bc_south)
	bc_handler.add_boundary_condition(bc_east)
	bc_handler.add_boundary_condition(bc_north)
	bc_handler.add_boundary_condition(bc_top)

	# Set boundary conditions
	mom_eq.set_boundary_conditions(bc_handler)

	# Create output handlers
	output_mom = SaveFields(mom_eq)
	output_mom.set_output_folder(output_folder)
	output_mom.add_output_field("u", "Displacement (m)")
	output_mom.add_output_field("sig", "Stress (Pa)")
	# output_mom.add_output_field("p_elems", "Mean stress (Pa)")
	# output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")
	output_mom.add_output_field("eps_tot", "Total strain (-)")
	output_mom.add_output_field("eps_ve", "Viscoelastic strain (-)")
	output_mom.add_output_field("eps_ds", "Dislocation creep (-)")
	output_mom.add_output_field("eps_ps", "Pressure-solution (-)")
	output_mom.add_output_field("eps_vp", "Viscoplastic strain (-)")
	output_mom.add_output_field("Fvp", "Yield function (-)")

	outputs = [output_mom]

	# Print output folder
	if MPI.COMM_WORLD.rank == 0:
		print(output_folder)

	# Define simulator
	sim = Simulator_M(mom_eq, t_control, [output_mom], True)
	sim.run()

	# Print time
	if MPI.COMM_WORLD.rank == 0:
		end_time = MPI.Wtime()
		elaspsed_time = end_time - start_time
		formatted_time = time.strftime("%H:%M:%S", time.gmtime(elaspsed_time))
		print(f"Time: {formatted_time} ({elaspsed_time} seconds)\n")


def main():
	run_case("primal")
	run_case("mixed")
	run_case("mixed_stab")
	run_case("mixed_star")

if __name__ == '__main__':
	main()

