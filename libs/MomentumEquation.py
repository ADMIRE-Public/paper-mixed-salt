from abc import ABC, abstractmethod
import dolfinx as do
import basix
import ufl
from petsc4py import PETSc
import torch as to
from Utils import dotdot2
from CharacteristicLength import ModelML
from MaterialProps import Material
from MomentumBC import BcHandler
import Utils as utils

class LinearMomentum(ABC):
	def __init__(self, grid, theta):
		self.grid = grid
		self.theta = theta

		self.create_function_spaces()
		self.create_ds_dx()

		self.n_elems = self.DG0_1.dofmap.index_map.size_local + len(self.DG0_1.dofmap.index_map.ghosts)
		self.n_nodes = self.CG1_1.dofmap.index_map.size_local + len(self.CG1_1.dofmap.index_map.ghosts)

		self.commom_fields()
		self.create_fenicsx_fields()
		self.create_pytorch_fields()

	def commom_fields(self):
		self.T0 = to.zeros(self.n_elems, dtype=to.float64)
		self.Temp = to.zeros(self.n_elems, dtype=to.float64)
		self.eps_tot = do.fem.Function(self.DG0_3x3)
		self.sig = do.fem.Function(self.DG0_3x3)
		self.u = do.fem.Function(self.CG1_3x1)
		self.q_elems = do.fem.Function(self.DG0_1)
		self.q_nodes = do.fem.Function(self.CG1_1)
		self.p_elems = do.fem.Function(self.DG0_1)
		self.p_nodes = do.fem.Function(self.CG1_1)

	def set_material(self, material : Material):
		self.mat = material
		self.initialize()

	def set_T(self, T):
		self.Temp = T

	def set_T0(self, T0):
		self.T0 = T0

	def set_solver(self, solver : PETSc.KSP):
		self.solver = solver

	def set_boundary_conditions(self, bc : BcHandler):
		self.bc = bc

	def create_function_spaces(self):
		self.CG1_3x1 = do.fem.functionspace(self.grid.mesh, ("Lagrange", 1, (self.grid.domain_dim, )))
		self.DG0_1 = do.fem.functionspace(self.grid.mesh, ("DG", 0))
		self.CG1_1 = do.fem.functionspace(self.grid.mesh, ("Lagrange", 1))
		self.DG0_3x3 = do.fem.functionspace(self.grid.mesh, ("DG", 0, (3, 3)))
		self.DG0_6x6 = do.fem.functionspace(self.grid.mesh, ("DG", 0, (6, 6)))

	def create_ds_dx(self):
		self.ds = ufl.Measure("ds", domain=self.grid.mesh, subdomain_data=self.grid.get_boundaries())
		self.dx = ufl.Measure("dx", domain=self.grid.mesh, subdomain_data=self.grid.get_subdomains())

	def create_normal(self):
		n = ufl.FacetNormal(self.grid.mesh)
		self.normal = ufl.dot(n, self.u_)

	def build_body_force(self, g : list):
		density = do.fem.Function(self.DG0_1)
		density.x.array[:] = self.mat.density
		body_force = density*do.fem.Constant(self.grid.mesh, do.default_scalar_type(tuple(g)))
		self.b_body = ufl.dot(body_force, self.u_)*self.dx

	def compute_q_nodes(self) -> do.fem.Function:
		dev = self.sig - (1/3)*ufl.tr(self.sig)*ufl.Identity(3)
		q_form = ufl.sqrt((3/2)*ufl.inner(dev, dev))
		self.q_nodes = utils.project(q_form, self.CG1_1)

	def compute_q_elems(self) -> do.fem.Function:
		dev = self.sig - (1/3)*ufl.tr(self.sig)*ufl.Identity(3)
		q_form = ufl.sqrt((3/2)*ufl.inner(dev, dev))
		self.q_elems = utils.project(q_form, self.DG0_1)

	def compute_total_strain(self):
		self.eps_tot = utils.project(utils.epsilon(self.u), self.DG0_3x3)
		eps_to = utils.numpy2torch(self.eps_tot.x.array.reshape((self.n_elems, 3, 3)))
		return eps_to

	def compute_eps_th(self):
		eps_th = to.zeros((self.n_elems, 3, 3), dtype=to.float64)
		deltaT = self.Temp - self.T0
		for elem_th in self.mat.elems_th:
			elem_th.compute_eps_th(deltaT)
			eps_th += elem_th.eps_th
		return eps_th

	def compute_eps_ne_k(self, dt):
		eps_ne_k = to.zeros((self.n_elems, 3, 3), dtype=to.float64)
		for elem_ne in self.mat.elems_ne:
			elem_ne.compute_eps_ne_k(dt*self.theta, dt*(1 - self.theta))
			eps_ne_k += elem_ne.eps_ne_k
		return eps_ne_k

	def compute_eps_ne_rate(self, stress, dt):
		for elem_ne in self.mat.elems_ne:
			elem_ne.compute_eps_ne_rate(stress, dt*self.theta, self.Temp, return_eps_ne=False)

	def update_eps_ne_rate_old(self):
		for elem_ne in self.mat.elems_ne:
			elem_ne.update_eps_ne_rate_old()

	def update_eps_ne_old(self, stress, stress_k, dt):
		for elem_ne in self.mat.elems_ne:
			elem_ne.update_eps_ne_old(stress, stress_k, dt*(1-self.theta))

	def increment_internal_variables(self, stress, stress_k, dt):
		for elem_ne in self.mat.elems_ne:
			elem_ne.increment_internal_variables(stress, stress_k, dt)

	def update_internal_variables(self):
		for elem_ne in self.mat.elems_ne:
			elem_ne.update_internal_variables()

	def create_solution_vector(self):
		self.X = do.fem.Function(self.V)

	@abstractmethod
	def compute_CT(self, dt, stress_k):
		pass

	@abstractmethod
	def compute_eps_rhs(self, dt, stress_k, eps_k):
		pass

	@abstractmethod
	def compute_elastic_stress(self, eps_e):
		pass

	@abstractmethod
	def compute_stress(self, eps_tot, eps_rhs, p):
		pass

	@abstractmethod
	def create_fenicsx_fields(self):
		pass

	@abstractmethod
	def create_pytorch_fields(self):
		pass

	@abstractmethod
	def create_trial_test_functions(self):
		pass

	@abstractmethod
	def get_uV(self):
		"""Function space for displacement field"""
		pass

	@abstractmethod
	def initialize(self) -> None:
		pass

	@abstractmethod
	def split_solution(self):
		pass

	@abstractmethod
	def split_solution(self):
		pass

	@abstractmethod
	def compute_p_nodes(self):
		pass

	@abstractmethod
	def solve_elastic_response(self):
		pass

	@abstractmethod
	def solve(self):
		pass





class LinearMomentumPrimal(LinearMomentum):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)
		self.V = self.CG1_3x1
		self.create_trial_test_functions()
		self.create_normal()
		self.create_solution_vector()

	def create_fenicsx_fields(self):
		self.C = do.fem.Function(self.DG0_6x6)
		self.CT = do.fem.Function(self.DG0_6x6)
		self.eps_rhs = do.fem.Function(self.DG0_3x3)

	def create_pytorch_fields(self):
		self.eps_rhs_to = to.zeros((self.n_elems, 3, 3))

	def create_trial_test_functions(self):
		self.du = ufl.TrialFunction(self.V)
		self.u_ = ufl.TestFunction(self.V)

	def get_uV(self):
		return self.V

	def initialize(self) -> None:
		self.C.x.array[:] = to.flatten(self.mat.C)

	def compute_CT(self, stress_k, dt):
		self.mat.compute_G_B(stress_k, dt, self.theta, self.Temp)
		self.mat.compute_CT(dt, self.theta)
		self.CT.x.array[:] = to.flatten(self.mat.CT)

	def compute_elastic_stress(self, eps_e : to.Tensor) -> to.Tensor:
		stress_to = dotdot2(self.mat.C, eps_e)
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_stress(self, eps_tot_to : to.Tensor, *_) -> to.Tensor:
		stress_to = dotdot2(self.mat.CT, eps_tot_to - self.eps_rhs_to)
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_eps_rhs(self, dt : float, stress_k : to.Tensor) -> None:
		eps_ne_k = self.compute_eps_ne_k(dt)
		eps_th = self.compute_eps_th()
		self.eps_rhs_to = eps_ne_k + eps_th - dt*(1 - self.theta)*(self.mat.B + dotdot2(self.mat.G, stress_k))
		self.eps_rhs.x.array[:] = to.flatten(self.eps_rhs_to)

	def compute_moduli(self, stress_to):
		pass

	def solve_elastic_response(self):
		# Build bilinear form
		a = ufl.inner(utils.dotdot(self.C, utils.epsilon(self.du)), utils.epsilon(self.u_))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs))
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

	def split_solution(self):
		self.u = self.X

	def compute_p_nodes(self) -> do.fem.Function:
		self.p_nodes = utils.project(ufl.tr(self.sig)/3, self.CG1_1)

	def compute_p_elems(self) -> do.fem.Function:
		self.p_elems = utils.project(ufl.tr(self.sig)/3, self.DG0_1)

	def solve(self, stress_k_to, t, dt):

		# Compute consistent tangent matrix
		self.compute_CT(stress_k_to, dt)

		# Compute right-hand side epsilon
		self.compute_eps_rhs(dt, stress_k_to)

		# Build bilinear form
		a = ufl.inner(utils.dotdot(self.CT, utils.epsilon(self.du)), utils.epsilon(self.u_))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		b_rhs = ufl.inner(utils.dotdot(self.CT, self.eps_rhs), utils.epsilon(self.u_))*self.dx
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs) + b_rhs)
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

		self.run_after_solve()




class LinearMomentumMixed(LinearMomentum):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)
		Vue = basix.ufl.element("CG", self.grid.mesh.basix_cell(), 1, shape=(3,))  	# displacement finite element
		Vpe = basix.ufl.element("CG", self.grid.mesh.basix_cell(), 1)  				# mean stress finite element
		el_mixed = basix.ufl.mixed_element([Vue, Vpe])
		self.V = do.fem.functionspace(self.grid.mesh, el_mixed)
		self.create_trial_test_functions()
		self.create_normal()
		self.create_solution_vector()

	def set_stabilization_h(self, h : to.Tensor) -> None:
		self.h_cell_2.x.array[:] = h**2

	def create_fenicsx_fields(self):
		self.CT_tilde = do.fem.Function(self.DG0_6x6)
		self.C_tilde = do.fem.Function(self.DG0_6x6)
		self.C_tilde_inv = do.fem.Function(self.DG0_6x6)
		self.T_vol = do.fem.Function(self.DG0_1)
		self.B_vol = do.fem.Function(self.DG0_1)
		self.eps_rhs_tilde = do.fem.Function(self.DG0_3x3)
		self.eps_ne_vol = do.fem.Function(self.DG0_1)
		self.eps_th_vol = do.fem.Function(self.DG0_1)
		self.K = do.fem.Function(self.DG0_1)
		self.E = do.fem.Function(self.DG0_1)
		self.h_cell_2 = do.fem.Function(self.DG0_1)
		self.p_k = do.fem.Function(self.CG1_1)

	def create_pytorch_fields(self):
		self.eps_rhs_to = to.zeros((self.n_elems, 3, 3))
		self.eps_rhs_tilde_to = to.zeros((self.n_elems, 3, 3))

	def create_trial_test_functions(self):
		self.du, self.dp = ufl.TrialFunctions(self.V)
		self.u_, self.p_ = ufl.TestFunctions(self.V)

	def get_uV(self):
		return self.V.sub(0)

	def initialize(self) -> None:
		self.C_tilde.x.array[:] = self.mat.C_tilde.flatten()
		self.C_tilde_inv.x.array[:] = self.mat.C_tilde_inv.flatten()
		self.K.x.array[:] = self.mat.K
		self.E.x.array[:] = self.mat.E

	def compute_CT(self, stress_k, dt):
		self.mat.compute_G_B(stress_k, dt, self.theta, self.Temp)
		self.mat.compute_T_IT()
		self.mat.compute_Bvol_Tvol(stress_k, dt)
		self.mat.compute_Gtilde_Btilde(stress_k, dt)
		self.mat.compute_CT_tilde(dt, self.theta)
		self.CT_tilde.x.array[:] = to.flatten(self.mat.CT_tilde)
		self.T_vol.x.array[:] = self.mat.T_vol
		self.B_vol.x.array[:] = self.mat.B_vol

	def compute_elastic_stress(self, eps_e : to.Tensor) -> to.Tensor:
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_e_tilde = self.compute_eps_tilde(eps_e)
		stress_to = dotdot2(self.mat.C_tilde, eps_e_tilde) + self.p_to[:,None,None]*I
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_stress(self, eps_tot):
		eps_tilde = self.compute_eps_tilde(eps_tot)
		I = to.eye(3).expand(self.n_elems, -1, -1)
		pI = self.p_to[:,None,None]*I
		stress_to = dotdot2(self.mat.CT_tilde, eps_tilde - self.eps_rhs_tilde_to + dotdot2(self.mat.C_tilde_inv, pI))
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_eps_k_ne_vol(self, eps_k_ne):
		eps_k_ne_vol = to.einsum("bii->b", eps_k_ne)
		return eps_k_ne_vol

	def compute_eps_k_tilde(self, eps_k_ne, eps_k_ne_vol):
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_k_ne_tilde = eps_k_ne - (1/3)*eps_k_ne_vol[:,None,None]*I
		return eps_k_ne_tilde

	def compute_eps_tilde(self, eps):
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_vol = to.einsum("bii->b", eps)[:,None,None]
		return eps - (1/3)*eps_vol*I

	def compute_eps_rhs(self, dt, stress_k, eps_k_tilde):
		self.eps_rhs_tilde_to = eps_k_tilde - dt*(1 - self.theta)*(self.mat.B_tilde + dotdot2(self.mat.G_tilde, stress_k))
		self.eps_rhs_tilde.x.array[:] = to.flatten(self.eps_rhs_tilde_to)

	def split_solution(self):
		self.u = self.X.sub(0).collapse()
		self.p_nodes = self.X.sub(1).collapse()

		self.u.x.scatter_forward()
		self.p_nodes.x.scatter_forward()

		# Project mean stress to Gauss points
		self.p_elems = utils.project(self.p_nodes, self.DG0_1)
		self.p_to = utils.numpy2torch(self.p_elems.x.array)

	def compute_p_nodes(self) -> do.fem.Function:
		return self.p_nodes

	def compute_p_elems(self) -> do.fem.Function:
		self.p_elems = utils.project(ufl.tr(self.sig)/3, self.DG0_1)
		return self.p_elems

	def compute_moduli(self, stress_to):
		pass

	def solve_elastic_response(self):
		# Build bilinear form
		eps_u = utils.epsilon(self.du)
		eps_tilde = eps_u - (1/3)*ufl.tr(eps_u)*ufl.Identity(3)
		a = ufl.inner(utils.dotdot(self.C_tilde, eps_tilde + utils.dotdot(self.C_tilde_inv, self.dp*ufl.Identity(3))), utils.epsilon(self.u_))*self.dx
		a += (self.dp*self.p_ - self.K*ufl.tr(utils.epsilon(self.du))*self.p_ )*self.dx
		a += (3*ufl.dot(self.h_cell_2*(self.K/self.E)*ufl.grad(self.dp), ufl.grad(self.p_)))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs))
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

	def solve(self, stress_k_to, t, dt):
		# Compute consistent tangent matrix
		self.compute_CT(stress_k_to, dt)

		# Compute epsilons
		eps_ne_k = self.compute_eps_ne_k(dt)
		eps_ne_k_vol = self.compute_eps_k_ne_vol(eps_ne_k)
		eps_k_tilde_to = self.compute_eps_k_tilde(eps_ne_k, eps_ne_k_vol)

		# Compute right-hand side epsilon
		self.compute_eps_rhs(dt, stress_k_to, eps_k_tilde_to)

		# Compute non-elastic volumetric strain
		self.eps_ne_vol.x.array[:] = eps_ne_k_vol

		# Compute thermoelastic strains
		eps_th_to = self.compute_eps_th()
		eps_th_vol_to = to.einsum("bii->b", eps_th_to)
		self.eps_th_vol.x.array[:] = eps_th_vol_to

		# Build bi-linear form
		phi2 = dt*(1 - self.theta)
		eps_u = utils.epsilon(self.du)
		eps_tilde = eps_u - (1/3)*ufl.tr(eps_u)*ufl.Identity(3)
		a = ufl.inner(utils.dotdot(self.CT_tilde, eps_tilde + utils.dotdot(self.C_tilde_inv, self.dp*ufl.Identity(3))), utils.epsilon(self.u_))*self.dx
		a += ((1 + phi2*self.K*self.T_vol)*self.dp*self.p_ - self.K*ufl.tr(utils.epsilon(self.du))*self.p_ )*self.dx
		a += (3*ufl.dot(self.h_cell_2*(self.K/self.E)*ufl.grad(self.dp), ufl.grad(self.p_)))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		b_u = ufl.inner(utils.dotdot(self.CT_tilde, self.eps_rhs_tilde), utils.epsilon(self.u_))*self.dx
		b_p = self.K*(phi2*(self.T_vol*self.p_k + self.B_vol) - self.eps_ne_vol - self.eps_th_vol)*self.p_*self.dx
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs) + b_u + b_p)
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

		# Update p_k
		self.p_k.x.array[:] = self.p_nodes.x.array
		self.p_k.x.scatter_forward()

		self.run_after_solve()




class LinearMomentumMixedStar(LinearMomentum):
	def __init__(self, grid, theta):
		super().__init__(grid, theta)
		Vue = basix.ufl.element("CG", self.grid.mesh.basix_cell(), 1, shape=(3,))  	# displacement finite element
		Vpe = basix.ufl.element("CG", self.grid.mesh.basix_cell(), 1)  				# mean stress finite element
		el_mixed = basix.ufl.mixed_element([Vue, Vpe])
		self.V = do.fem.functionspace(self.grid.mesh, el_mixed)
		self.create_trial_test_functions()
		self.create_normal()
		self.create_solution_vector()

	def set_stabilization_h(self, h : to.Tensor) -> None:
		self.h_cell_2.x.array[:] = h**2

	def create_fenicsx_fields(self):
		self.CT_tilde = do.fem.Function(self.DG0_6x6)
		self.C_tilde = do.fem.Function(self.DG0_6x6)
		self.C_tilde_inv = do.fem.Function(self.DG0_6x6)
		self.T_vol = do.fem.Function(self.DG0_1)
		self.B_vol = do.fem.Function(self.DG0_1)
		self.eps_rhs_tilde = do.fem.Function(self.DG0_3x3)
		self.eps_ne_vol = do.fem.Function(self.DG0_1)
		self.eps_th_vol = do.fem.Function(self.DG0_1)
		self.K = do.fem.Function(self.DG0_1)
		self.E = do.fem.Function(self.DG0_1)
		self.E_star = do.fem.Function(self.DG0_1)
		self.h_cell_2 = do.fem.Function(self.DG0_1)
		self.p_k = do.fem.Function(self.CG1_1)

	def create_pytorch_fields(self):
		self.eps_rhs_to = to.zeros((self.n_elems, 3, 3))
		self.eps_rhs_tilde_to = to.zeros((self.n_elems, 3, 3))

	def create_trial_test_functions(self):
		self.du, self.dp = ufl.TrialFunctions(self.V)
		self.u_, self.p_ = ufl.TestFunctions(self.V)

	def get_uV(self):
		return self.V.sub(0)

	def initialize(self) -> None:
		self.C_tilde.x.array[:] = self.mat.C_tilde.flatten()
		self.C_tilde_inv.x.array[:] = self.mat.C_tilde_inv.flatten()
		self.K.x.array[:] = self.mat.K
		self.E.x.array[:] = self.mat.E

	def compute_CT(self, stress_k, dt):
		self.mat.compute_G_B(stress_k, dt, self.theta, self.Temp)
		self.mat.compute_T_IT()
		self.mat.compute_Bvol_Tvol(stress_k, dt)
		self.mat.compute_Gtilde_Btilde(stress_k, dt)
		self.mat.compute_CT_tilde(dt, self.theta)
		self.CT_tilde.x.array[:] = to.flatten(self.mat.CT_tilde)
		self.T_vol.x.array[:] = self.mat.T_vol
		self.B_vol.x.array[:] = self.mat.B_vol

	def compute_elastic_stress(self, eps_e : to.Tensor) -> to.Tensor:
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_e_tilde = self.compute_eps_tilde(eps_e)
		stress_to = dotdot2(self.mat.C_tilde, eps_e_tilde) + self.p_to[:,None,None]*I
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_stress(self, eps_tot):
		eps_tilde = self.compute_eps_tilde(eps_tot)
		I = to.eye(3).expand(self.n_elems, -1, -1)
		pI = self.p_to[:,None,None]*I
		stress_to = dotdot2(self.mat.CT_tilde, eps_tilde - self.eps_rhs_tilde_to + dotdot2(self.mat.C_tilde_inv, pI))
		self.sig.x.array[:] = to.flatten(stress_to)
		return stress_to

	def compute_eps_k_ne_vol(self, eps_k_ne):
		eps_k_ne_vol = to.einsum("bii->b", eps_k_ne)
		return eps_k_ne_vol

	def compute_eps_k_tilde(self, eps_k_ne, eps_k_ne_vol):
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_k_ne_tilde = eps_k_ne - (1/3)*eps_k_ne_vol[:,None,None]*I
		return eps_k_ne_tilde

	def compute_eps_tilde(self, eps):
		I = to.eye(3).expand(self.n_elems, -1, -1)
		eps_vol = to.einsum("bii->b", eps)[:,None,None]
		return eps - (1/3)*eps_vol*I

	def compute_eps_rhs(self, dt, stress_k, eps_k_tilde):
		self.eps_rhs_tilde_to = eps_k_tilde - dt*(1 - self.theta)*(self.mat.B_tilde + dotdot2(self.mat.G_tilde, stress_k))
		self.eps_rhs_tilde.x.array[:] = to.flatten(self.eps_rhs_tilde_to)

	def split_solution(self):
		self.u = self.X.sub(0).collapse()
		self.p_nodes = self.X.sub(1).collapse()

		self.u.x.scatter_forward()
		self.p_nodes.x.scatter_forward()

		# Project mean stress to Gauss points
		self.p_elems = utils.project(self.p_nodes, self.DG0_1)
		self.p_to = utils.numpy2torch(self.p_elems.x.array)

	def compute_p_nodes(self) -> do.fem.Function:
		return self.p_nodes

	def compute_p_elems(self) -> do.fem.Function:
		self.p_elems = utils.project(ufl.tr(self.sig)/3, self.DG0_1)
		return self.p_elems

	def solve_elastic_response(self):
		# Build bilinear form
		eps_u = utils.epsilon(self.du)
		eps_tilde = eps_u - (1/3)*ufl.tr(eps_u)*ufl.Identity(3)
		a = ufl.inner(utils.dotdot(self.C_tilde, eps_tilde + utils.dotdot(self.C_tilde_inv, self.dp*ufl.Identity(3))), utils.epsilon(self.u_))*self.dx
		a += (self.dp*self.p_ - self.K*ufl.tr(utils.epsilon(self.du))*self.p_ )*self.dx
		a += (3*ufl.dot(self.h_cell_2*(self.K/self.E)*ufl.grad(self.dp), ufl.grad(self.p_)))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs))
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

	def compute_equivalent_stress(self, stress_to, sigma_v):
		I = to.eye(3, device=stress_to.device, dtype=stress_to.dtype).expand(stress_to.shape[0], 3, 3)
		s = stress_to - sigma_v[:, None, None] * I
		sigma_eq = to.sqrt((3.0/2.0) * (s ** 2).sum(dim=(-2, -1)))
		return sigma_eq

	def compute_equivalent_strain(self, strain_to, eps_v):
		I = to.eye(3, device=strain_to.device, dtype=strain_to.dtype).expand(strain_to.shape[0], 3, 3)
		e = strain_to - (eps_v / 3.0)[:, None, None] * I
		eps_eq = to.sqrt((2.0/3.0) * (e ** 2).sum(dim=(-2, -1)))
		return eps_eq

	def compute_moduli(self, stress_to):
		strain_to = self.compute_total_strain()

		principal_stresses = to.linalg.eigvalsh(stress_to)
		principal_strains = to.linalg.eigvalsh(strain_to)
		sigma_1 = principal_stresses[:,0]
		sigma_2 = principal_stresses[:,1]
		sigma_3 = principal_stresses[:,2]
		epsil_1 = principal_strains[:,0]
		epsil_3 = principal_strains[:,2]
		# E_star = sigma_1/epsil_1
		nu = self.mat.elems_e[0].nu
		E_star_1 = (sigma_1 - nu*(sigma_2 + sigma_3))/epsil_1

		sigma_v = utils.numpy2torch(self.p_elems.x.array)
		eps_v = to.einsum("bii->b", strain_to)
		sigma_eq = self.compute_equivalent_stress(stress_to, sigma_v)
		eps_eq = self.compute_equivalent_strain(strain_to, eps_v)
		K_star = utils.numpy2torch(self.K.x.array)
		G_star = sigma_eq/(3*eps_eq)
		E_star_2 = G_star

		self.E_star.x.array[:] = E_star_1
		

	def solve(self, stress_k_to, t, dt):

		# Compute consistent tangent matrix
		self.compute_CT(stress_k_to, dt)

		# Compute epsilons
		eps_ne_k = self.compute_eps_ne_k(dt)
		eps_ne_k_vol = self.compute_eps_k_ne_vol(eps_ne_k)
		eps_k_tilde_to = self.compute_eps_k_tilde(eps_ne_k, eps_ne_k_vol)

		# Compute right-hand side epsilon
		self.compute_eps_rhs(dt, stress_k_to, eps_k_tilde_to)

		# Compute non-elastic volumetric strain
		self.eps_ne_vol.x.array[:] = eps_ne_k_vol

		# Compute thermoelastic strains
		eps_th_to = self.compute_eps_th()
		eps_th_vol_to = to.einsum("bii->b", eps_th_to)
		self.eps_th_vol.x.array[:] = eps_th_vol_to

		# Build bi-linear form
		phi2 = dt*(1 - self.theta)
		eps_u = utils.epsilon(self.du)
		eps_tilde = eps_u - (1/3)*ufl.tr(eps_u)*ufl.Identity(3)
		a = ufl.inner(utils.dotdot(self.CT_tilde, eps_tilde + utils.dotdot(self.C_tilde_inv, self.dp*ufl.Identity(3))), utils.epsilon(self.u_))*self.dx
		a += ((1 + phi2*self.K*self.T_vol)*self.dp*self.p_ - self.K*ufl.tr(utils.epsilon(self.du))*self.p_ )*self.dx
		a += (3*ufl.dot(self.h_cell_2*(self.K/self.E_star)*ufl.grad(self.dp), ufl.grad(self.p_)))*self.dx
		bilinear_form = do.fem.form(a)
		A = do.fem.petsc.assemble_matrix(bilinear_form, bcs=self.bc.dirichlet_bcs)
		A.assemble()

		# Build linear form
		b_u = ufl.inner(utils.dotdot(self.CT_tilde, self.eps_rhs_tilde), utils.epsilon(self.u_))*self.dx
		b_p = self.K*(phi2*(self.T_vol*self.p_k + self.B_vol) - self.eps_ne_vol - self.eps_th_vol)*self.p_*self.dx
		linear_form = do.fem.form(self.b_body + sum(self.bc.neumann_bcs) + b_u + b_p)
		b = do.fem.petsc.assemble_vector(linear_form)
		do.fem.petsc.apply_lifting(b, [bilinear_form], [self.bc.dirichlet_bcs])
		b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
		do.fem.petsc.set_bc(b, self.bc.dirichlet_bcs)
		b.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

		# Solve linear system
		self.solver.setOperators(A)
		self.solver.solve(b, self.X.x.petsc_vec)
		self.X.x.scatter_forward()
		self.split_solution()

		# Update p_k
		self.p_k.x.array[:] = self.p_nodes.x.array
		self.p_k.x.scatter_forward()

		self.run_after_solve()


