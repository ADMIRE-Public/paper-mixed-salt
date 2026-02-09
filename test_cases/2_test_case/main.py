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
from TimeHandler import TimeController, TimeControllerParabolic
import time

class LinearMomentumPrimalMod(LinearMomentumPrimal):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)

	def initialize(self) -> None:
		self.C.x.array[:] = to.flatten(self.mat.C)
		self.eps_ve = do.fem.Function(self.DG0_3x3)
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)
		self.Fvp = do.fem.Function(self.DG0_1)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[2].Fvp
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
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[2].Fvp
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
		self.eps_cr = do.fem.Function(self.DG0_3x3)
		self.eps_vp = do.fem.Function(self.DG0_3x3)

	def run_after_solve(self):
		try:
			self.eps_ve.x.array[:] = to.flatten(self.mat.elems_ne[0].eps_ne_k)
		except:
			pass
		try:
			self.eps_cr.x.array[:] = to.flatten(self.mat.elems_ne[1].eps_ne_k)
		except:
			pass
		try:
			self.eps_vp.x.array[:] = to.flatten(self.mat.elems_ne[2].eps_ne_k)
			self.Fvp.x.array[:] = self.mat.elems_ne[2].Fvp
		except:
			pass




def run_case(formulation: str, model: str="SLS", t_unit: str="day") -> None:
	comm = MPI.COMM_WORLD
	comm.Barrier()
	if MPI.COMM_WORLD.rank == 0:
	    start_time = MPI.Wtime()

	# Read grid
	grid_path = os.path.join("grids", "hole_0")
	grid = GridHandlerGMSH("geom", grid_path)

	# Define output folder
	case_folder = "case_1"
	output_folder = os.path.join("output", case_folder, model, formulation)

	# Time settings for equilibrium stage
	t_final = 100.0
	dt = t_final/50
	t_control = TimeController(time_step=dt, final_time=t_final, initial_time=0.0, time_unit=t_unit)


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

	# Define solver
	mom_solver = PETSc.KSP().create(grid.mesh.comm)
	mom_solver.setType("bcgs")
	mom_solver.getPC().setType("asm")
	mom_solver.setTolerances(rtol=1e-12, max_it=100)
	mom_eq.set_solver(mom_solver)

	# Build material properties
	mat = Material(mom_eq.n_elems)

	# Set density
	rho = 0*to.ones(mom_eq.n_elems)
	mat.set_density(rho)

	# Constitutive model
	E = 102e9*to.ones(mom_eq.n_elems)
	nu = 0.3*to.ones(mom_eq.n_elems)
	spring_0 = Spring(E, nu, "spring")
	G = E/(2*(1 + nu))
	K = E/(3*(1 - 2*nu))
	print("%.2e, %.2e"%(float(G[10]), float(K[10])))

	# Create Kelvin-Voigt viscoelastic element
	E = 10e9*to.ones(mom_eq.n_elems)
	nu = 0.32*to.ones(mom_eq.n_elems)
	eta = 105e11*to.ones(mom_eq.n_elems)
	kelvin = Viscoelastic(eta, E, nu, "kelvin")

	# Create dislocation creep element
	A = 1.1e-21*to.ones(mom_eq.n_elems)
	Q = 51600*to.ones(mom_eq.n_elems)
	n = 3.0*to.ones(mom_eq.n_elems)
	creep_0 = DislocationCreep(A, Q, n, "creep")

	# Create constitutive model
	mat.add_to_elastic(spring_0)
	if model == "SLS":
		mat.add_to_non_elastic(kelvin)
	elif model == "Maxwell":
		mat.add_to_non_elastic(creep_0)
	else:
		raise Exception("Model must be SLS or Maxwell.")

	# Set constitutive model
	mom_eq.set_material(mat)

	# Set body forces
	g = -9.81
	g_vec = [0.0, 0.0, g]
	mom_eq.build_body_force(g_vec)

	# Set temperature
	T_elems = 298*to.ones(mom_eq.n_elems)
	mom_eq.set_T0(T_elems)
	mom_eq.set_T(T_elems)

	# Boundary conditions
	time_values = [t_control.t_initial, t_control.t_final]
	nt = len(time_values)
	bc_west = momBC.DirichletBC(boundary_name = "WEST", 
					 	  component = 0,
					 	  values = nt*[0.0],
					 	  time_values = time_values)

	bc_south = momBC.DirichletBC(boundary_name = "SOUTH", 
					 	  component = 1,
					 	  values = nt*[0.0],
					 	  time_values = time_values)

	bc_bottom = momBC.DirichletBC(boundary_name = "BOTTOM", 
					 	  component = 2,
					 	  values = nt*[0.0],
					 	  time_values = time_values)

	bc_top = momBC.DirichletBC(boundary_name = "TOP", 
					 	  component = 2,
					 	  values = nt*[0.0],
					 	  time_values = time_values)

	bc_east = momBC.NeumannBC(boundary_name = "EAST",
						direction = 0,
						density = 0.0,
						ref_pos = .0,
						values = [-10*utils.MPa, -10*utils.MPa],
						time_values = time_values,
						g = g_vec[2])

	bc_handler = momBC.BcHandler(mom_eq)
	bc_handler.add_boundary_condition(bc_west)
	bc_handler.add_boundary_condition(bc_south)
	bc_handler.add_boundary_condition(bc_bottom)
	bc_handler.add_boundary_condition(bc_top)
	bc_handler.add_boundary_condition(bc_east)

	# Set boundary conditions
	mom_eq.set_boundary_conditions(bc_handler)

	# Create output handlers
	output_mom = SaveFields(mom_eq)
	output_mom.set_output_folder(output_folder)
	output_mom.add_output_field("u", "Displacement (m)")
	output_mom.add_output_field("sig", "Stress (Pa)")
	output_mom.add_output_field("p_nodes", "Mean stress (Pa)")
	output_mom.add_output_field("p_elems", "Mean stress (Pa)")
	output_mom.add_output_field("q_nodes", "Von Mises stress (Pa)")
	output_mom.add_output_field("q_elems", "Von Mises stress (Pa)")

	outputs = [output_mom]

	# Print output folder
	if MPI.COMM_WORLD.rank == 0:
		print(output_folder)

	# Define simulator
	sim = Simulator_M(mom_eq, t_control, outputs, True)
	sim.run()

	# Print time
	if MPI.COMM_WORLD.rank == 0:
		end_time = MPI.Wtime()
		elaspsed_time = end_time - start_time
		formatted_time = time.strftime("%H:%M:%S", time.gmtime(elaspsed_time))
		print(f"Time: {formatted_time} ({elaspsed_time} seconds)\n")

def main():
	run_case(formulation="primal", model="SLS", t_unit="minute")
	run_case(formulation="mixed", model="SLS", t_unit="minute")
	run_case(formulation="mixed_stab", model="SLS", t_unit="minute")
	run_case(formulation="mixed_star", model="SLS", t_unit="minute")

	run_case(formulation="primal", model="Maxwell", t_unit="day")
	run_case(formulation="mixed", model="Maxwell", t_unit="day")
	run_case(formulation="mixed_stab", model="Maxwell", t_unit="day")
	run_case(formulation="mixed_star", model="Maxwell", t_unit="day")

if __name__ == '__main__':
	main()