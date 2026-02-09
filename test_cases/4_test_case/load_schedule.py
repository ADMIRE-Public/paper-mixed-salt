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

def get_data(schedule, p_0=0):
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


def create_cycle(var_base, N_cycles):
	rt = 0.6
	rp = 0.1
	schedule = []
	np.random.seed(6)
	for i in range(N_cycles):
		new_cycle = np.array([
		                     	np.random.uniform((1-rt)*var_base[0], (1+rt)*var_base[0]),	# Δt_up
		                     	np.random.uniform((1-rp)*var_base[1], (1+rp)*var_base[1]),	# ΔP_up
		                     	np.random.uniform((1-rt)*var_base[2], (1+rt)*var_base[2]),	# Δt_top
		                     	np.random.uniform((1-rp)*var_base[3], (1+rp)*var_base[3]),	# ΔP_top
		                     	np.random.uniform((1-rt)*var_base[4], (1+rt)*var_base[4]),	# Δt_down
		                     	np.random.uniform((1-rp)*var_base[5], (1+rp)*var_base[5]),	# ΔP_down
		                     	np.random.uniform((1-rt)*var_base[6], (1+rt)*var_base[6]),	# Δt_bottom
		                     	np.random.uniform((1-rp)*var_base[7], (1+rp)*var_base[7])	# ΔP_bottom
		                     ])
		schedule.append(list(new_cycle))
	return np.array(schedule)

def load_schedule(p0, T0, N_cycles):
	Δt_op = 10*day
	Δt_idle = 10*day
	ΔP_op = 11*MPa
	ΔP_idle = 0*MPa
	ΔT_op = 40
	ΔT_idle = 0

	P_base = np.array([Δt_op, -ΔP_op, Δt_idle, ΔP_idle, Δt_op, ΔP_op, Δt_idle, ΔP_idle])
	T_base = np.array([Δt_op, -ΔT_op, Δt_idle, ΔT_idle, Δt_op, ΔT_op, Δt_idle, ΔT_idle])

	P_schedule = create_cycle(P_base, N_cycles)
	T_schedule = create_cycle(T_base, N_cycles)

	time, P = get_data(P_schedule, p0)
	time, T = get_data(T_schedule, T0)
	return time, P, -T


def main():
	time, P, T = load_schedule(p0=15*MPa, T0=330, N_cycles=10)

	data = {
		"Time": list(time),
		"Pressure": list(P),
		"Temperature": list(T),
	}

	save_json(data, "gas_P_T.json")

	# Plot pressure schedule
	fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 6))
	fig1.subplots_adjust(top=0.925, bottom=0.150, left=0.140, right=0.902, hspace=0.35, wspace=0.225)

	ax1.plot(time/day, -P/MPa, ".-", color="steelblue", linewidth=1.5)
	ax1.set_xlabel("Time (days)", size=12, fontname="serif")
	ax1.set_ylabel("Pressure (MPa)", size=12, fontname="serif")

	ax2.plot(time/day, T, ".-", color="lightcoral", linewidth=1.5)
	ax2.set_xlabel("Time (days)", size=12, fontname="serif")
	ax2.set_ylabel("Temperature (K)", size=12, fontname="serif")

	apply_grey_theme(fig1, [ax1, ax2], transparent=True)

	plt.show()


if __name__ == '__main__':
	main()