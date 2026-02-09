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


def plot_stresses(ax, output_path, method):
	centroids, times, p_elems = read_cell_scalar(os.path.join(output_path, method, "operation", "p_elems", "p_elems.xdmf"))
	centroids, times, q_elems = read_cell_scalar(os.path.join(output_path, method, "operation", "q_elems", "q_elems.xdmf"))

	colors = ["steelblue", "lightcoral"]
	for i, cell_p in enumerate([15212, 15217]):
		p_elem = -p_elems[:,cell_p]
		q_elem = q_elems[:,cell_p]
		ax.plot(p_elem/MPa, q_elem/MPa, label=f"Cell {i+1}", color=colors[i])

	ax.set_xlabel("Mean stress (MPa)", size=10, fontname="serif")
	ax.set_ylabel("Von Mises stress (MPa)", size=10, fontname="serif")



def main_1():
	# Chooose output path
	output_path = os.path.join("output", "case_0")

	fig1, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(12, 3.5))
	fig1.subplots_adjust(top=0.912, bottom=0.248, left=0.064, right=0.995, hspace=0.35, wspace=0.225)

	plot_stresses(ax1, output_path, "primal")
	plot_stresses(ax2, output_path, "mixed")
	plot_stresses(ax3, output_path, "mixed_stab")
	plot_stresses(ax4, output_path, "mixed_star")
	ax3.legend(fancybox=True, shadow=True, ncol=4, bbox_to_anchor=(0.2, -0.2))

	ax1.set_title("(a) P1", size=12, fontname="serif")
	ax2.set_title("(b) P1-P1", size=12, fontname="serif")
	ax3.set_title(r"(c) P1-P1 stab$-E$", size=12, fontname="serif")
	ax4.set_title(r"(d) P1-P1 stab$-E^*$", size=12, fontname="serif")

	apply_grey_theme(fig1, [ax1, ax2, ax3, ax4])


	plt.show()



if __name__ == '__main__':
	main_1()

