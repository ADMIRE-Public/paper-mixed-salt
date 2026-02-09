import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import meshio
import json
from PostProcessingTools import (read_vector_from_points,
								find_mapping,
								read_msh_as_pandas,
								read_xdmf_as_pandas,
								compute_cell_centroids,
								read_scalar_from_cells,
								read_tensor_from_cells)

hour = 60*60
day = 24*hour
MPa = 1e6

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def apply_theme(axis_list):
	for ax in axis_list:
		ax.grid(True, color="0.92")
		ax.set_facecolor("0.85")

def find_closest_point(target_point: list, points: pd.DataFrame) -> int:
	x_p, y_p, z_p = target_point
	d = np.sqrt(  (points["x"].values - x_p)**2
	            + (points["y"].values - y_p)**2
	            + (points["z"].values - z_p)**2 )
	cell_p = d.argmin()
	return cell_p


def plot_q(ax, output_folder, props):
	line_style = props["line_style"]
	formulation = props["formulation"]
	colors = props["color"]

	points_xdmf, cells_xdmf = read_xdmf_as_pandas(os.path.join(output_folder, "eps_ve", "eps_ve.xdmf"))
	mid_cells = compute_cell_centroids(points_xdmf.values, cells_xdmf.values)

	target_point = [1.0, 1.0, 1.0]
	cell_id = find_closest_point(target_point, mid_cells)

	tot_xx, tot_yy, tot_zz, tot_xy, tot_xz, tot_yz = read_tensor_from_cells(os.path.join(output_folder, "eps_tot", "eps_tot.xdmf"))

	eps_1 = -100*tot_zz.iloc[cell_id].values
	eps_3 = -100*tot_xx.iloc[cell_id].values
	q = eps_1 - eps_3

	t = tot_xx.iloc[cell_id].index.values/day

	ax.plot(t, q, line_style, color=colors, label=formulation)
	ax.set_xlabel("Time (days)", size=12, fontname="serif")
	ax.set_ylabel(r"$\varepsilon_1 - \varepsilon_3$ (%)", size=12, fontname="serif")
	ax.legend(loc=0, shadow=True, fancybox=True, prop={"size": 8}, ncol=2)


def plot_Fvp(ax, output_folder, props):
	line_style = props["line_style"]
	formulation = props["formulation"]
	color = props["color"]

	points_xdmf, cells_xdmf = read_xdmf_as_pandas(os.path.join(output_folder, "Fvp", "Fvp.xdmf"))
	mid_cells = compute_cell_centroids(points_xdmf.values, cells_xdmf.values)

	# target_point = [0.5, 0.5, 0.5]
	target_point = [1.0, 1.0, 1.0]
	cell_id = find_closest_point(target_point, mid_cells)

	df_Fvp = read_scalar_from_cells(os.path.join(output_folder, "Fvp", "Fvp.xdmf"))
	Fvp = df_Fvp.iloc[cell_id].values

	t = df_Fvp.iloc[cell_id].index.values/day

	ax.plot(t, Fvp, line_style, color=color, label=formulation)
	ax.plot(t, len(t)*[0], "--", color="black")
	ax.set_xlabel("Time (days)", size=12, fontname="serif")
	ax.set_ylabel("Yield function (-)", size=12, fontname="serif")
	ax.legend(loc=0, shadow=True, fancybox=True, prop={"size": 8}, ncol=1)


def plot_stresses(ax):

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
	time_values = load_hist[:,0]*day
	s3_values = load_hist[:,1]*MPa
	s1_values = load_hist[:,2]*MPa

	ax.plot(time_values/day, s1_values/MPa, "-", color="steelblue", label=r"$\sigma_1$")
	ax.plot(time_values/day, s3_values/MPa, "-", color="lightcoral", label=r"$\sigma_3$")
	ax.set_xlabel("Time (days)", size=12, fontname="serif")
	ax.set_ylabel("Stress (MPa)", size=12, fontname="serif")
	ax.legend(loc=0, shadow=True, fancybox=True, prop={"size": 8}, ncol=1)


def main():
	folder_primal = os.path.join("output", "case_0", "primal")
	folder_mixed = os.path.join("output", "case_0", "mixed")
	folder_stab = os.path.join("output", "case_0", "mixed_stab")
	folder_star = os.path.join("output", "case_0", "mixed_star")

	# Plot loading schedule
	fig, axis = plt.subplots(1, 3, figsize=(10, 3.5))
	fig.subplots_adjust(top=0.926, bottom=0.155, left=0.055, right=0.980, hspace=0.35, wspace=0.312)
	fig.patch.set_alpha(0.0)

	cases = [
		(folder_primal, {"line_style": "-", "formulation": "P1", "color": "#001f78"}),
		(folder_mixed, {"line_style": "-", "formulation": "P1-P1", "color": "#00b5db"}),
		(folder_stab, {"line_style": "-", "formulation": r"P1-P1 stab-E", "color": "#f03d14"}),
		((folder_star, {"line_style": "-", "formulation": r"P1-P1 stab-E$^*$", "color": "#179547"}))
	]

	for case in cases:
		folder, props = case
		plot_q(axis[1], folder, props)
		plot_Fvp(axis[2], folder, props)
		
	plot_stresses(axis[0])
	axis[0].set_title("(a)", size=12, fontname="serif")
	axis[1].set_title("(b)", size=12, fontname="serif")
	axis[2].set_title("(c)", size=12, fontname="serif")



	apply_theme(axis.flatten())

	plt.show()

if __name__ == '__main__':
	main()