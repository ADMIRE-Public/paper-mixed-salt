import numpy as np
import matplotlib.pyplot as plt
import json
import os

minute = 60
hour = 60*minute
day = 24*hour
MPa = 1e6

def apply_grey_theme(fig, axes, transparent=True):
	fig.patch.set_facecolor("#212121ff")
	if transparent:
		fig.patch.set_alpha(0.0)
	for ax in axes:
		if ax != None:
			ax.grid(True, color='0.92')
			ax.set_axisbelow(True)
			ax.spines['bottom'].set_color('black')
			ax.spines['top'].set_color('black')
			ax.spines['right'].set_color('black')
			ax.spines['left'].set_color('black')
			ax.tick_params(axis='x', colors='black', which='both')
			ax.tick_params(axis='y', colors='black', which='both')
			ax.yaxis.label.set_color('black')
			ax.xaxis.label.set_color('black')
			ax.set_facecolor("0.85")

def read_json(file_name):
	with open(file_name, "r") as j_file:
		data = json.load(j_file)
	return data

def save_json(data, file_name):
	with open(file_name, "w") as f:
	    json.dump(data, f, indent=4)


def main():
	data = read_json("gas_P_T.json")

	time = np.array(data["Time"]) / day
	P = -np.array(data["Pressure"]) / MPa
	T = np.array(data["Temperature"])

	idx = np.where(time < 100)
	time = time[idx]
	P = P[idx]
	T = T[idx]
	print(time)



	# Plot pressure schedule
	fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
	fig1.subplots_adjust(top=0.925, bottom=0.166, left=0.076, right=0.973, hspace=0.35, wspace=0.285)

	ax1.plot(time, P, "-", color="steelblue", linewidth=1.5)
	ax1.set_xlabel("Time (days)", size=12, fontname="serif")
	ax1.set_ylabel("Pressure (MPa)", size=12, fontname="serif")

	ax2.plot(time, T, "-", color="lightcoral", linewidth=1.5)
	ax2.set_xlabel("Time (days)", size=12, fontname="serif")
	ax2.set_ylabel("Temperature (K)", size=12, fontname="serif")

	apply_grey_theme(fig1, [ax1, ax2], transparent=True)

	plt.show()


if __name__ == '__main__':
	main()