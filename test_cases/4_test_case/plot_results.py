import meshio as ms
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from Grid import GridHandlerGMSH
from Utils import extract_cavern_surface_from_grid, orient_triangles_outward, compute_volume, save_json, read_json

hour = 60*60
day = 24*hour
MPa = 1e6

def apply_grey_theme(fig, axes, transparent=True, grid_color="0.92", back_color='0.85'):
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.grid(True, color=grid_color)
			ax.set_axisbelow(True)
			ax.spines['bottom'].set_color('black')
			ax.spines['top'].set_color('black')
			ax.spines['right'].set_color('black')
			ax.spines['left'].set_color('black')
			ax.tick_params(axis='x', colors='black', which='both')
			ax.tick_params(axis='y', colors='black', which='both')
			ax.yaxis.label.set_color('black')
			ax.xaxis.label.set_color('black')
			ax.set_facecolor(back_color)

def compute_cell_centroids(cells: np.ndarray, points: np.ndarray) -> np.ndarray:
    n_cells = cells.shape[0]
    centroids = np.zeros((n_cells, 3))
    for i, cell in enumerate(cells):
        p0 = points[cell[0]]
        p1 = points[cell[1]]
        p2 = points[cell[2]]
        p3 = points[cell[3]]
        x = (p0[0] + p1[0] + p2[0] + p3[0])/4
        y = (p0[1] + p1[1] + p2[1] + p3[1])/4
        z = (p0[2] + p1[2] + p2[2] + p3[2])/4
        centroids[i,:] = np.array([x, y, z])
    return centroids


def read_cell_scalar(xdmf_field_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = ms.xdmf.TimeSeriesReader(xdmf_field_path)
    points, cells = reader.read_points_cells()
    n_cells = cells["tetra"].shape[0]
    n_steps = reader.num_steps
    centroids = compute_cell_centroids(cells["tetra"], points)
    scalar_field = np.zeros((n_steps, n_cells))
    time_list = np.zeros(n_steps)
    for k in range(reader.num_steps):
        time, point_data, cell_data = reader.read_data(k)
        time_list[k] = time
        field_name = list(cell_data["tetra"].keys())[0]
        scalar_field[k] = cell_data["tetra"][field_name].flatten()
    return centroids, time_list, scalar_field

def read_node_vector(xdmf_field_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reader = ms.xdmf.TimeSeriesReader(xdmf_field_path)
    points, cells = reader.read_points_cells()
    n_nodes = points.shape[0]
    n_steps = reader.num_steps
    vector_field = np.zeros((n_steps, n_nodes, 3))
    time_list = np.zeros(n_steps)
    for k in range(reader.num_steps):
        time, point_data, cell_data = reader.read_data(k)
        time_list[k] = time
        field_name = list(point_data.keys())[0]
        vector_field[k,:,:] = point_data[field_name]
    return points, time_list, vector_field


def save_volumes(output_path):
	grid = GridHandlerGMSH("geom", os.path.join("grids", "grid_0"))
	coords_wall, tris_conn, wall_ids, _ = extract_cavern_surface_from_grid(grid, "Cavern")
	vol_dict = {
		"times": [],
		"volumes": {}
	}
	# for case in ["primal", "mixed_0.0", "mixed_1.0", "mixed_5.0"]:
	# for case in ["mixed_star_beta_1.0", "mixed_star_beta_2.0"]:
	for case in ["primal", "mixed", "mixed_stab", "mixed_star_beta_1.0", "mixed_star_beta_2.0"]:
		points, time_list, u_field = read_node_vector(os.path.join(output_path, case, "operation", "u", "u.xdmf"))
		tris_conn = orient_triangles_outward(coords_wall, tris_conn, [0.0, 0.0, -900])
		n = len(time_list)
		volumes = []
		for i in range(n):
			wall_xu = coords_wall + u_field[i,wall_ids,:]
			volumes.append(compute_volume(wall_xu, tris_conn))
		vol_dict["volumes"][case] = volumes
	vol_dict["times"] = list(time_list)
	save_json(vol_dict, os.path.join(output_path, "volumes.json"))


def plot_volumes(ax, output_path, case_labels):
	data = read_json(os.path.join(output_path, "volumes.json"))
	time = np.array(data["times"])/day
	print(data["volumes"].keys())
	for case_label in case_labels:
		method, label, color = case_label
		volumes = np.array(data["volumes"][method])
		ax.plot(time, 100*(volumes[0] - volumes)/volumes[0], "-", color=color, label=label)
	ax.legend(loc=0, fancybox=True, shadow=True)
	ax.set_xlabel("Time (days)", size=10, fontname="serif")
	ax.set_ylabel("Volume loss (%)", size=10, fontname="serif")


def plot_stresses(ax1, ax2, output_path, case_labels):
	for case_label in case_labels:
		method, label, color = case_label
		centroids, times, p_elems = read_cell_scalar(os.path.join(output_path, method, "operation", "p_elems", "p_elems.xdmf"))
		centroids, times, q_elems = read_cell_scalar(os.path.join(output_path, method, "operation", "q_elems", "q_elems.xdmf"))

		xc = centroids[:,0]
		yc = centroids[:,1]
		zc = centroids[:,2]

		point = [46.53, 58.35, -962.607]
		point = [45, 0, -885]
		d = np.sqrt(  (xc - point[0])**2
		            + (yc - point[1])**2
		            + (zc - point[2])**2 )
		cell_p = d.argmin()

		p_elem = p_elems[:,cell_p]
		q_elem = q_elems[:,cell_p]

		times /= day
		p_elem /= MPa
		q_elem /= MPa

		ax1.plot(times, p_elem, label=label, color=color)
		ax2.plot(times, q_elem, color=color)
	ax1.set_xlabel("Time (day)", size=10, fontname="serif")
	ax1.set_ylabel("Mean stress (MPa)", size=10, fontname="serif")

	ax2.set_xlabel("Time (day)", size=10, fontname="serif")
	ax2.set_ylabel("Von Mises stress (MPa)", size=10, fontname="serif")



def main_1():
	# Chooose output path
	output_path = os.path.join("output", "case_iE")

	# Calculate and save cavern volumes
	# save_volumes(output_path)

	fig1, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3.5))
	fig1.subplots_adjust(top=0.912, bottom=0.248, left=0.064, right=0.995, hspace=0.35, wspace=0.225)


	case_labels = [
		("primal", "P1", "#001f78"),
		("mixed", r"P1-P1", "#00b5db"),
		("mixed_stab", r"P1-P1 stab $E$", "#f03d14"),
		("mixed_star_beta_1.0", r"P1-P1 stab $E^*$", "#179547"),
	]

	plot_stresses(ax1, ax2, output_path, case_labels)
	plot_volumes(ax3, output_path, case_labels)
	ax3.legend(fancybox=True, shadow=True, ncol=4, bbox_to_anchor=(0.2, -0.2))

	ax1.set_title("(a)", size=12, fontname="serif")
	ax2.set_title("(b)", size=12, fontname="serif")
	ax3.set_title("(c)", size=12, fontname="serif")

	apply_grey_theme(fig1, [ax1, ax2, ax3])


	plt.show()



if __name__ == '__main__':
	main_1()

