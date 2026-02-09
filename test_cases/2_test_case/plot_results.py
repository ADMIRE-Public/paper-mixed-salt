import os
import sys
sys.path.append(os.path.join("..", "..", "libs"))
from PostProcessingTools import read_node_scalar
import numpy as np
import matplotlib.pyplot as plt

MPa = 1e6
minute = 60
hour = 60*minute
day = 24*hour

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

def get_results(results_folder, field_name="p_nodes"):
	points, time_list, p_field = read_node_scalar(os.path.join(results_folder, f"{field_name}", f"{field_name}.xdmf"))

	# Extract indices along the line around the hole
	R = 0.2
	line_idx = np.where((points[:,2] == 0.0) & (points[:,1] <= R) & (points[:,0] <= R))[0]

	# Extract points along line
	line_points = points[line_idx]

	# Extract x coordinates along line
	x0 = line_points[:,0]

	# Extract p along line
	p0 = p_field[:,line_idx]/MPa

	# Sort indices according to increasing x
	sorted_idx = np.argsort(x0)

	# Sort x and T
	line_sorted = line_points[sorted_idx]
	x_sorted = x0[sorted_idx]
	p_sorted = p0[:,sorted_idx]

	# Get angle along line
	thetas = []
	for point in line_sorted:
		theta = np.arctan(point[0]/point[1])
		thetas.append(np.degrees(theta))
	thetas = np.array(thetas)

	return time_list, thetas, p_sorted


def main_ve():
	results_folder = os.path.join("output", "case_1", "SLS")

	time, thetas, p_0 = get_results(os.path.join(results_folder, "mixed"), field_name="p_nodes")
	time, thetas, p_1 = get_results(os.path.join(results_folder, "mixed_stab"), field_name="p_nodes")
	time, thetas, p_5 = get_results(os.path.join(results_folder, "mixed_star"), field_name="p_nodes")
	time, thetas, p = get_results(os.path.join(results_folder, "primal"), field_name="p_nodes")

	# Plot pressure schedule
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
	fig.subplots_adjust(top=0.90, bottom=0.15, left=0.076, right=0.845, hspace=0.35, wspace=0.293)

	time_idx = 0
	# t = time[time_idx]/day
	t = time[time_idx]/minute
	ax1.plot(thetas, p[time_idx,:], linestyle="-", marker="x", markersize=6, fillstyle="none", color="#001f78", label="P1")
	ax1.plot(thetas, p_0[time_idx,:], linestyle="-", marker="v", markersize=6, fillstyle="none", color="#00b5db", label=r"P1-P1 non-stab")
	ax1.plot(thetas, p_1[time_idx,:], linestyle="-", marker="o", markersize=6, fillstyle="none", color="#f03d14", label=r"P1-P1 stab-E")
	ax1.plot(thetas, p_5[time_idx,:], linestyle="-", marker="^", markersize=6, fillstyle="none", color="#179547", label=r"P1-P1 stab-E$^*$")
	ax1.set_xlabel(r"Angle, $\theta$ (deg)", font="serif", size=12)
	ax1.set_ylabel(r"Mean stress (MPa)", font="serif", size=12)
	ax1.set_title(f"t = {round(t,2)} minute(s)", font="serif", size=12)
	ax1.legend(loc=0, fancybox=True, shadow=True, bbox_to_anchor=(2.75, 1.01), ncol=1, prop={"size": 8})

	time_idx = -1
	# t = time[time_idx]/day
	t = time[time_idx]/minute
	ax2.plot(thetas, p[time_idx,:], linestyle="-", marker="x", markersize=6, fillstyle="none", color="#001f78", label="P1")
	ax2.plot(thetas, p_0[time_idx,:], linestyle="-", marker="v", markersize=6, fillstyle="none", color="#00b5db", label=r"P1-P1 non-stab")
	ax2.plot(thetas, p_1[time_idx,:], linestyle="-", marker="o", markersize=6, fillstyle="none", color="#f03d14", label=r"P1-P1 stab-E")
	ax2.plot(thetas, p_5[time_idx,:], linestyle="-", marker="^", markersize=6, fillstyle="none", color="#179547", label=r"P1-P1 stab-E$^*$")
	ax2.set_xlabel(r"Angle, $\theta$ (deg)", font="serif", size=12)
	ax2.set_ylabel(r"Mean stress (MPa)", font="serif", size=12)
	ax2.set_title(f"t = {round(t,2)} minute(s)", font="serif", size=12)
	# ax2.set_ylim(-15, 22)

	apply_grey_theme(fig, [ax1, ax2])

	plt.show()


def main_ds():
	results_folder = os.path.join("output", "case_1", "Maxwell")

	time, thetas, p_0 = get_results(os.path.join(results_folder, "mixed"), field_name="p_nodes")
	time, thetas, p_1 = get_results(os.path.join(results_folder, "mixed_stab"), field_name="p_nodes")
	time, thetas, p_5 = get_results(os.path.join(results_folder, "mixed_star"), field_name="p_nodes")
	time, thetas, p = get_results(os.path.join(results_folder, "primal"), field_name="p_nodes")

	# Plot pressure schedule
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
	fig.subplots_adjust(top=0.90, bottom=0.15, left=0.076, right=0.845, hspace=0.35, wspace=0.293)

	time_idx = 0
	t = time[time_idx]/day
	ax1.plot(thetas, p[time_idx,:], linestyle="-", marker="x", markersize=6, fillstyle="none", color="#001f78", label="P1")
	ax1.plot(thetas, p_0[time_idx,:], linestyle="-", marker="v", markersize=6, fillstyle="none", color="#00b5db", label=r"P1-P1 non-stab")
	ax1.plot(thetas, p_1[time_idx,:], linestyle="-", marker="o", markersize=6, fillstyle="none", color="#f03d14", label=r"P1-P1 stab-E")
	ax1.plot(thetas, p_5[time_idx,:], linestyle="-", marker="^", markersize=6, fillstyle="none", color="#179547", label=r"P1-P1 stab-E$^*$")
	ax1.set_xlabel(r"Angle, $\theta$ (deg)", font="serif", size=12)
	ax1.set_ylabel(r"Mean stress (MPa)", font="serif", size=12)
	ax1.set_title(f"t = {round(t,2)} day(s)", font="serif", size=12)
	# ax1.legend(loc=0, fancybox=True, shadow=True, bbox_to_anchor=(1.6, 1.15), ncol=4, prop={"size": 8})
	ax1.legend(loc=0, fancybox=True, shadow=True, bbox_to_anchor=(2.75, 1.01), ncol=1, prop={"size": 8})

	time_idx = -1
	t = time[time_idx]/day
	ax2.plot(thetas, p[time_idx,:], linestyle="-", marker="x", markersize=6, fillstyle="none", color="#001f78", label="P1")
	ax2.plot(thetas, p_0[time_idx,:], linestyle="-", marker="v", markersize=6, fillstyle="none", color="#00b5db", label=r"P1-P1 non-stab")
	ax2.plot(thetas, p_1[time_idx,:], linestyle="-", marker="o", markersize=6, fillstyle="none", color="#f03d14", label=r"P1-P1 stab-E")
	ax2.plot(thetas, p_5[time_idx,:], linestyle="-", marker="^", markersize=6, fillstyle="none", color="#179547", label=r"P1-P1 stab-E$^*$")
	ax2.set_xlabel(r"Angle, $\theta$ (deg)", font="serif", size=12)
	ax2.set_ylabel(r"Mean stress (MPa)", font="serif", size=12)
	ax2.set_title(f"t = {round(t,2)} day(s)", font="serif", size=12)
	ax2.set_ylim(-15, 22)

	apply_grey_theme(fig, [ax1, ax2])

	plt.show()




if __name__ == '__main__':
	main_ve()
	# main_ds()