from mpi4py import MPI
from dolfinx.io import gmshio
from dolfinx.mesh import meshtags
from dolfinx import mesh
import numpy as np
import torch as to
import meshio
import os


class GridHandlerGMSH(object):
	def __init__(self, geometry_name, grid_folder):
		self.grid_folder = grid_folder
		self.geometry_name = geometry_name
		self.comm = MPI.COMM_WORLD
		self.rank = self.comm.rank

		self.load_mesh()
		self.build_tags()
		self.load_subdomains()
		self.load_boundaries()
		self.build_box_dimensions()
		self.__extract_grid_data()

	def load_mesh(self):
		self.mesh, self.subdomains, self.boundaries = gmshio.read_from_msh(
													    os.path.join(self.grid_folder, f"{self.geometry_name}.msh"),
													    self.comm,
													    rank=0
													)
		self.domain_dim = self.mesh.topology.dim
		self.boundary_dim = self.domain_dim - 1
		self.n_elems = self.mesh.topology.index_map(self.domain_dim).size_local + len(self.mesh.topology.index_map(self.domain_dim).ghosts)
		self.n_nodes = self.mesh.topology.index_map(0).size_local + len(self.mesh.topology.index_map(0).ghosts)

	def build_tags(self):
		grid = meshio.read(os.path.join(self.grid_folder, self.geometry_name+".msh"))
		self.tags = {1:{}, 2:{}, 3:{}}
		for key, value in grid.field_data.items():
			self.tags[value[1]][key] = value[0]
		self.dolfin_tags = self.tags

	def load_subdomains(self):
		self.subdomain_tags = {}
		for subdomain_name in self.get_subdomain_names():
			self.subdomain_tags[subdomain_name] = []


	def load_boundaries(self):
		self.boundary_tags = {}

		for boundary_name in self.get_boundary_names():
			self.boundary_tags[boundary_name] = []
		
		tag_to_name = {fd: name for name, fd in self.dolfin_tags[2].items()}
		boundary_facets = mesh.exterior_facet_indices(self.mesh.topology)
		for i, facet in zip(boundary_facets, self.boundaries.values):
			boundary_name = tag_to_name[facet]
			self.boundary_tags[boundary_name].append(i)


	def build_box_dimensions(self):
		self.Lx = self.mesh.geometry.x[:,0].max() - self.mesh.geometry.x[:,0].min()
		self.Ly = self.mesh.geometry.x[:,1].max() - self.mesh.geometry.x[:,1].min()
		self.Lz = self.mesh.geometry.x[:,2].max() - self.mesh.geometry.x[:,2].min()

	def get_boundaries(self):
		return self.boundaries

	def get_boundary_tags(self, BOUNDARY_NAME):
		if BOUNDARY_NAME == None:
			return None
		else:
			return self.boundary_tags[BOUNDARY_NAME]

	def get_boundary_tag(self, BOUNDARY_NAME):
		if BOUNDARY_NAME == None:
			return None
		else:
			tag_number = self.dolfin_tags[self.boundary_dim][BOUNDARY_NAME]
			return tag_number

	def get_boundary_names(self):
		boundary_names = list(self.dolfin_tags[self.boundary_dim].keys())
		return boundary_names

	def get_subdomain_tag(self, DOMAIN_NAME):
		tag_number = self.dolfin_tags[self.domain_dim][DOMAIN_NAME]
		return tag_number

	def get_subdomains(self):
		return self.subdomains

	def get_subdomain_names(self):
		subdomain_names = list(self.dolfin_tags[self.domain_dim].keys())
		return subdomain_names

	def __extract_grid_data(self):
		self.region_names = self.get_subdomain_names()
		self.n_regions = len(self.region_names)
		self.region_indices = {}
		self.tags_dict = {}

		for i in range(len(self.region_names)):
			self.region_indices[self.region_names[i]] = []
			tag = self.get_subdomain_tag(self.region_names[i])
			self.tags_dict[tag] = self.region_names[i]

		for cell in range(self.n_elems):
			region_marker = self.subdomains.values[cell]
			self.region_indices[self.tags_dict[region_marker]].append(cell)

	def get_parameter(self, param):
		if type(param) == int or type(param) == float:
			return to.tensor([param for i in range(self.n_elems)])
		elif len(param) == self.n_regions:
			param_to = to.zeros(self.n_elems)
			for i, region in enumerate(self.region_indices.keys()):
				param_to[self.region_indices[region]] = param[i]
			return param_to
		elif len(param) == self.n_elems:
			if type(param) == to.Tensor:
				return param
			else:
				return to.tensor(param)
		else:
			raise Exception("Size of parameter list does not match neither # of elements nor # of regions.")

