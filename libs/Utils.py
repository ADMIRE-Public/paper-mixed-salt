import torch as to
import numpy as np
import dolfinx as do
from dolfinx.fem.petsc import LinearProblem
import ufl
import json

GPa = 1e9
MPa = 1e6
kPa = 1e3
minute = 60
hour = 60*minute
day = 24*hour
year = 365*day

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def save_json(data, file_name):
	with open(file_name, "w") as f:
	    json.dump(data, f, indent=4)

def local_projection_old(tensor, V):
    u = do.fem.Function(V)
    dv = ufl.TrialFunction(V)
    v_ = ufl.TestFunction(V)
    a_proj = ufl.inner(dv, v_)*ufl.dx
    b_proj = ufl.inner(tensor, v_)*ufl.dx
    problem = LinearProblem(a_proj, b_proj, u=u)
    problem.solve()
    return u

def project(tensor_ufl, V):
	tensor_expr = do.fem.Expression(tensor_ufl, V.element.interpolation_points())
	tensor = do.fem.Function(V)
	tensor.interpolate(tensor_expr)
	return tensor

def epsilon(u):
	grad_u = ufl.sym(ufl.grad(u))
	return grad_u

def dotdot(C, eps):
	tensor = voigt2tensor(ufl.dot(C, tensor2voigt(eps)))
	return tensor

def tensor2voigt(e):
	e_voigt = ufl.as_vector([e[0,0], e[1,1], e[2,2], e[0,1], e[0,2], e[1,2]])
	return e_voigt

def voigt2tensor(s):
	s_tensor = ufl.as_matrix([[s[0], s[3], s[4]],
							  [s[3], s[1], s[5]],
							  [s[4], s[5], s[2]]])
	return s_tensor

def numpy2torch(numpy_array):
	torch_array = to.tensor(numpy_array, dtype=to.float64)
	return torch_array

def dotdot2(C_voigt, eps_tensor):
	n_elems = C_voigt.shape[0]
	eps_voigt = to.zeros((n_elems, 6), dtype=to.float64)
	eps_voigt[:,0] = eps_tensor[:,0,0]
	eps_voigt[:,1] = eps_tensor[:,1,1]
	eps_voigt[:,2] = eps_tensor[:,2,2]
	eps_voigt[:,3] = eps_tensor[:,0,1]
	eps_voigt[:,4] = eps_tensor[:,0,2]
	eps_voigt[:,5] = eps_tensor[:,1,2]
	stress_voigt = to.bmm(C_voigt, eps_voigt.unsqueeze(2)).squeeze(2)
	stress_torch = to.zeros_like(eps_tensor, dtype=to.float64)
	stress_torch[:,0,0] = stress_voigt[:,0]
	stress_torch[:,1,1] = stress_voigt[:,1]
	stress_torch[:,2,2] = stress_voigt[:,2]
	stress_torch[:,0,1] = stress_torch[:,1,0] = stress_voigt[:,3]
	stress_torch[:,0,2] = stress_torch[:,2,0] = stress_voigt[:,4]
	stress_torch[:,1,2] = stress_torch[:,2,1] = stress_voigt[:,5]
	return stress_torch

def dotdot3(eps_tensor, C_voigt):
	Q = C_voigt.clone()
	Q[:,[3,4,5]] /= 2
	Q = Q.transpose(1,2)
	Q[:,[3,4,5]] *= 2
	n_elems = Q.shape[0]
	eps_voigt = to.zeros((n_elems, 6), dtype=to.float64)
	eps_voigt[:,0] = eps_tensor[:,0,0]
	eps_voigt[:,1] = eps_tensor[:,1,1]
	eps_voigt[:,2] = eps_tensor[:,2,2]
	eps_voigt[:,3] = eps_tensor[:,0,1]
	eps_voigt[:,4] = eps_tensor[:,0,2]
	eps_voigt[:,5] = eps_tensor[:,1,2]
	stress_voigt = to.bmm(Q, eps_voigt.unsqueeze(2)).squeeze(2)
	stress_torch = to.zeros_like(eps_tensor, dtype=to.float64)
	stress_torch[:,0,0] = stress_voigt[:,0]
	stress_torch[:,1,1] = stress_voigt[:,1]
	stress_torch[:,2,2] = stress_voigt[:,2]
	stress_torch[:,0,1] = stress_torch[:,1,0] = stress_voigt[:,3]
	stress_torch[:,0,2] = stress_torch[:,2,0] = stress_voigt[:,4]
	stress_torch[:,1,2] = stress_torch[:,2,1] = stress_voigt[:,5]
	return stress_torch

def create_field_nodes(grid, fun):
	coordinates = grid.mesh.geometry.x
	field = to.zeros(grid.n_nodes, dtype=to.float64)
	for i, coord in enumerate(coordinates):
		x, y, z = coord
		field[i] = fun(x, y, z)
	return field

def create_field_elems(grid, fun):
	field = to.zeros(grid.n_elems, dtype=to.float64)
	coordinates = grid.mesh.geometry.x
	conn_aux = grid.mesh.topology.connectivity(3, 0)
	conn = conn_aux.array.reshape((grid.n_elems, 4))
	for i in range(grid.n_elems):
		cell_vertices = conn[i]
		x = sum(coordinates[v] for v in cell_vertices) / len(cell_vertices)
		field[i] = fun(x[0], x[1], x[2])
	return field



def extract_cavern_surface_from_grid(grid, boundary_name: str):
    """
    Return (coords_wall, tris_local, wall_ids) for the named boundary.

    - coords_wall : (n_wall, 3) coordinates of wall vertices, in the SAME order as wall_ids
    - tris_local  : (n_tris, 3) triangle connectivity indexing into coords_wall
    - wall_ids    : (n_wall,) global vertex ids of the wall vertices
    """
    mesh = grid.mesh
    tdim = mesh.topology.dim
    fdim = tdim - 1

    # --- facet tags + target id ---
    tag = grid.get_boundary_tag(boundary_name)
    # try common names the Grid may expose
    mt = getattr(grid, "boundaries", None) or getattr(grid, "facet_tags", None)
    if mt is None:
        raise RuntimeError("Grid does not expose facet tags as 'boundaries' or 'facet_tags'.")

    # dolfinx MeshTags: .indices (array of facet ids), .values (array of tag values)
    facets = mt.indices[mt.values == tag]

    # --- facet -> vertex connectivity ---
    mesh.topology.create_connectivity(fdim, 0)
    f2v = mesh.topology.connectivity(fdim, 0)

    tri_global = []
    wall_set = set()
    for f in facets:
        verts = f2v.links(f)        # 3 vertex ids for a triangle facet
        if len(verts) != 3:
            # If your mesh has non-triangular faces, you need to triangulate them;
            # salt cavern cases with tetrahedra produce triangles here.
            continue
        tri_global.append(verts)
        wall_set.update(verts)

    if not tri_global:
        raise RuntimeError(f"No triangular facets found for boundary '{boundary_name}' (tag={tag}).")

    wall_ids = np.array(sorted(wall_set), dtype=np.int64)
    gid2lid = {gid: i for i, gid in enumerate(wall_ids)}

    # map global triangles -> local triangles indexing into coords_wall
    tris_local = np.array([[gid2lid[v] for v in tri] for tri in tri_global], dtype=np.int32)

    # dolfinx coordinates
    coords_all = mesh.geometry.x                     # (n_vertices, 3)
    coords_wall = coords_all[wall_ids]               # (n_wall, 3)

    return coords_wall, tris_local, wall_ids, np.array(tri_global)

def surface_centroid(coordinates, triangles): 
    """Calculate the centroid of a surface defined by triangles."""
    total_area = 0
    weighted_sum = np.zeros(3)
    for tri in triangles:
        p0, p1, p2 = coordinates[tri]
        center = (p0 + p1 + p2) / 3.0
        area = np.linalg.norm(np.cross(p1 - p0, p2 - p0)) / 2.0
        weighted_sum += center * area
        total_area += area
    return weighted_sum / total_area

def orient_triangles_outward(coordinates, triangles, reference_point):
    """Orient triangles so that their normals point outward from a reference point."""
    fixed_triangles = []
    for tri in triangles:
        p0, p1, p2 = coordinates[tri]
        normal = np.cross(p1 - p0, p2 - p0)
        center = (p0 + p1 + p2) / 3.0
        inward = reference_point - center
        if np.dot(normal, inward) > 0:
            fixed_triangles.append([tri[0], tri[2], tri[1]])
        else:
            fixed_triangles.append(tri)
    return np.array(fixed_triangles)

def compute_volume(coordinates, triangles):
    """Calculate the volume of a closed surface defined by triangles using the divergence theorem."""
    centroid= surface_centroid(coordinates, triangles)
    volume = 0
    for triangle in triangles:
        v0 = coordinates[triangle[0]] - centroid
        v1 = coordinates[triangle[1]] - centroid
        v2 = coordinates[triangle[2]] - centroid
        volume += np.dot(v0, np.cross(v1, v2)) / 6.0
    return volume