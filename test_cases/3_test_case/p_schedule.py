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

def get_pressure(schedule, p_0=0):
	time = [0]
	pressure = [p_0]
	t_schedule = schedule[:,[0, 2, 4, 6]]
	for i, dt in enumerate(t_schedule.flatten()):
		time.append(time[i] + dt)

	p_schedule = schedule[:,[1, 3, 5, 7]]
	for i, dp in enumerate(p_schedule.flatten()):
		p_i = pressure[i]
		pressure.append(p_i + dp)

	return np.array(time), -np.array(pressure)

def main_drawdown(p0, N_cycles):
	Δt_up = 1*hour
	ΔP_up = -3*MPa
	Δt_bottom = 1*hour
	ΔP_bottom = 0*MPa
	Δt_down = 1*hour
	ΔP_down = 3*MPa
	Δt_top = 8*hour
	ΔP_top = 0*MPa

	schedule_base = np.array([[Δt_up, ΔP_up, Δt_top, ΔP_top, Δt_down, ΔP_down, Δt_bottom, ΔP_bottom]])

	N_cycles = 10
	rt = 1.0
	rp = 0.1
	np.random.seed(6)
	cycle_base = schedule_base[0]
	schedule = []
	for i in range(N_cycles):
		new_cycle = np.array([
		                     	np.random.uniform((1-rt)*cycle_base[0], (1+rt)*cycle_base[0]),	# Δt_up
		                     	np.random.uniform((1-rp)*cycle_base[1], (1+rp)*cycle_base[1]),	# ΔP_up
		                     	np.random.uniform((1-rt)*cycle_base[2], (1+rt)*cycle_base[2]),	# Δt_top
		                     	np.random.uniform((1-rp)*cycle_base[3], (1+rp)*cycle_base[3]),	# ΔP_top
		                     	np.random.uniform((1-rt)*cycle_base[4], (1+rt)*cycle_base[4]),	# Δt_down
		                     	np.random.uniform((1-rp)*cycle_base[5], (1+rp)*cycle_base[5]),	# ΔP_down
		                     	np.random.uniform((1-rt)*cycle_base[6], (1+rt)*cycle_base[6]),	# Δt_bottom
		                     	np.random.uniform((1-rp)*cycle_base[7], (1+rp)*cycle_base[7])	# ΔP_bottom
		                     ])
		schedule.append(list(new_cycle))
	schedule = np.array(schedule)
	time, pressure = get_pressure(schedule, p0)
	return time, pressure

def pressure_schedule(p0, N_cycles):
	Δt_op = 15*hour
	Δt_idle = 15*hour
	ΔP_op = 3*MPa
	ΔP_idle = 0*MPa

	schedule_base = np.array([[Δt_op, -ΔP_op, Δt_idle, ΔP_idle, Δt_op, ΔP_op, Δt_idle, ΔP_idle]])

	rt = 0.9
	rp = 0.1
	np.random.seed(6)
	cycle_base = schedule_base[0]
	schedule = []
	for i in range(N_cycles):
		new_cycle = np.array([
		                     	np.random.uniform((1-rt)*cycle_base[0], (1+rt)*cycle_base[0]),	# Δt_up
		                     	np.random.uniform((1-rp)*cycle_base[1], (1+rp)*cycle_base[1]),	# ΔP_up
		                     	np.random.uniform((1-rt)*cycle_base[2], (1+rt)*cycle_base[2]),	# Δt_top
		                     	np.random.uniform((1-rp)*cycle_base[3], (1+rp)*cycle_base[3]),	# ΔP_top
		                     	np.random.uniform((1-rt)*cycle_base[4], (1+rt)*cycle_base[4]),	# Δt_down
		                     	np.random.uniform((1-rp)*cycle_base[5], (1+rp)*cycle_base[5]),	# ΔP_down
		                     	np.random.uniform((1-rt)*cycle_base[6], (1+rt)*cycle_base[6]),	# Δt_bottom
		                     	np.random.uniform((1-rp)*cycle_base[7], (1+rp)*cycle_base[7])	# ΔP_bottom
		                     ])
		schedule.append(list(new_cycle))
	schedule = np.array(schedule)
	time, pressure = get_pressure(schedule, p0)
	return time, pressure


def main():

	time, pressure = pressure_schedule(p0=10*MPa, N_cycles=10)

	data = {
		"Time": list(time),
		"Pressure": list(pressure)
	}

	# save_json(data, os.path.join("output", "case_1", "mixed_stab", "operation", "gas_pressure.json"))

	# Plot pressure schedule
	fig1, ax = plt.subplots(1, 1, figsize=(7, 3))
	fig1.subplots_adjust(top=0.985, bottom=0.150, left=0.140, right=0.980, hspace=0.35, wspace=0.225)

	index = 7
	ax.plot(time[:index]/hour, -pressure[:index]/MPa, ".-", color="steelblue", linewidth=1.5)
	ax.plot([-100, 0], [-pressure[0]/MPa, -pressure[0]/MPa], ".-", color="lightcoral", linewidth=1.5)

	ax.set_xlabel("Time (days)", size=12, fontname="serif")
	ax.set_ylabel("Pressure (MPa)", size=12, fontname="serif")
	ax.grid(True)

	apply_grey_theme(fig1, [ax], transparent=True)

	plt.show()


if __name__ == '__main__':
	main()