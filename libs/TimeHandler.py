from Utils import minute, hour, day, year, numpy2torch
import numpy as np
import torch as to
from mpi4py import MPI

class TimeController():
	def __init__(self, time_step, final_time, initial_time, time_unit="second"):
		self.__decide_time_unit(time_unit)
		self.dt = time_step*self.time_unit
		self.t_final = final_time*self.time_unit
		self.t_initial = initial_time*self.time_unit
		self.t = initial_time*self.time_unit

	def __decide_time_unit(self, time_unit):
		if time_unit == "second":
			self.time_unit = 1
		elif time_unit == "minute":
			self.time_unit = minute
		elif time_unit == "hour":
			self.time_unit = hour
		elif time_unit == "day":
			self.time_unit = day
		elif time_unit == "year":
			self.time_unit = year
		else:
			raise Exception(f"Time unit {time_unit} not supported.")

	def advance_time(self):
		self.t += self.dt

	def keep_looping(self):
		return self.t < self.t_final


class TimeControllerParabolic():
	def __init__(self, final_time, initial_time, n_time_steps, time_unit="second"):
		self.__decide_time_unit(time_unit)
		self.n_time_steps = n_time_steps
		self.t_initial = initial_time*self.time_unit
		self.t_final = final_time*self.time_unit

		self.time_list = self.calculate_varying_times(self.fun_parabolic)
		self.time_step = 0
		self.t = self.time_list[self.time_step]
		self.dt = self.time_list[1] - self.time_list[0]

	def __decide_time_unit(self, time_unit):
		if time_unit == "second":
			self.time_unit = 1
		elif time_unit == "minute":
			self.time_unit = minute
		elif time_unit == "hour":
			self.time_unit = hour
		elif time_unit == "day":
			self.time_unit = day
		elif time_unit == "year":
			self.time_unit = year
		else:
			raise Exception(f"Time unit {time_unit} not supported.")


	def fun_parabolic(self, t_array):
		return t_array**2

	def calculate_varying_times(self, fun):
		t_eq = np.linspace(self.t_initial, self.t_final, self.n_time_steps)
		y = fun(t_eq)
		f_min = np.min(t_eq)
		f_max = np.max(y)
		k = (t_eq.max() - t_eq.min())/(f_max - f_min)
		y = k*(y - f_min) + t_eq.min()
		return y

	def advance_time(self):
		self.time_step += 1
		self.t = self.time_list[self.time_step]
		self.dt = self.time_list[self.time_step] - self.time_list[self.time_step-1]

	def keep_looping(self):
		return self.t < self.t_final



class TimeControllerFvp():
	def __init__(self, dt_min, dt_max, final_time, initial_time, dt_init=None, time_unit="second"):
		self.__decide_time_unit(time_unit)
		self.dt_min = dt_min*self.time_unit
		self.dt_max = dt_max*self.time_unit
		if dt_init is None:
			self.dt = (self.dt_max + self.dt_min)/2
		else:
			self.dt = dt_init
		self.t_final = final_time*self.time_unit
		self.t_initial = initial_time*self.time_unit
		self.t = initial_time*self.time_unit

		self.Fvp_pred = 0.0

	def set_material(self, mat):
		self.mat = mat
		Fvp_old_local = float(self.mat.elems_ne[-1].Fvp.max())
		self.Fvp_old = MPI.COMM_WORLD.allreduce(Fvp_old_local, op=MPI.MAX)

	def __decide_time_unit(self, time_unit):
		if time_unit == "second":
			self.time_unit = 1
		elif time_unit == "minute":
			self.time_unit = minute
		elif time_unit == "hour":
			self.time_unit = hour
		elif time_unit == "day":
			self.time_unit = day
		elif time_unit == "year":
			self.time_unit = year
		else:
			raise Exception(f"Time unit {time_unit} not supported.")

	def advance_time(self):
		Fvp_local_max = float(self.mat.elems_ne[-1].Fvp.max())
		self.Fvp = MPI.COMM_WORLD.allreduce(Fvp_local_max, op=MPI.MAX)

		# self.Fvp = float(self.mat.elems_ne[-1].Fvp.max())

		diff_pred = abs(self.Fvp - self.Fvp_pred)
		
		dFdt = (self.Fvp - self.Fvp_old)/self.dt
		self.Fvp_pred = self.Fvp + dFdt*self.dt

		if MPI.COMM_WORLD.rank == 0:
			print(self.Fvp, dFdt, self.Fvp_pred, self.dt/self.time_unit, diff_pred, "\n")

		self.Fvp_old = self.Fvp

		if dFdt < 0 and self.Fvp_pred < 0:
			self.dt = min(self.dt*1.2, self.dt_max)
		elif dFdt < 0 and self.Fvp_pred >= 0:
			self.dt = min(self.dt*1.1, self.dt_max)
		elif dFdt > 0 and self.Fvp_pred < 0 and diff_pred > 0.2:
			self.dt = max(self.dt*0.9, self.dt_min)
		elif dFdt > 0 and self.Fvp_pred < 0 and diff_pred <= 0.2:
			self.dt = min(self.dt*1.1, self.dt_max)
		elif dFdt > 0 and self.Fvp_pred >= 0 and diff_pred > 0.01:
			self.dt = max(self.dt*0.1, self.dt_min)
		elif dFdt > 0 and self.Fvp_pred >= 0 and diff_pred <= 0.01:
			self.dt = max(self.dt*1.1, self.dt_min)
		else:
			pass

		self.t += self.dt

	def keep_looping(self):
		return self.t < self.t_final



class TimeControllerEpsilonRate():
	def __init__(self, dt_min, dt_max, final_time, initial_time, dt_init=None, time_unit="second"):
		self.__decide_time_unit(time_unit)
		self.dt_min = dt_min*self.time_unit
		self.dt_max = dt_max*self.time_unit
		if dt_init is None:
			# self.dt = (self.dt_max + self.dt_min)/2
			self.dt = self.dt_min
		else:
			self.dt = dt_init*self.time_unit
		self.t_final = final_time*self.time_unit
		self.t_initial = initial_time*self.time_unit
		self.t = initial_time*self.time_unit

		self.mat = None

	def set_material(self, mat):
		self.mat = mat
		Fvp_old_local = float(self.mat.elems_ne[-1].Fvp.max())
		self.Fvp_old = MPI.COMM_WORLD.allreduce(Fvp_old_local, op=MPI.MAX)
		self.Fvp_pred = 0.0

	def set_eq_mom(self, eq_mom):
		self.eq_mom = eq_mom
		self.eps_old = self.eq_mom.eps_tot.x.array
		self.eps_rate_max_old = 0

	def __decide_time_unit(self, time_unit):
		if time_unit == "second":
			self.time_unit = 1
		elif time_unit == "minute":
			self.time_unit = minute
		elif time_unit == "hour":
			self.time_unit = hour
		elif time_unit == "day":
			self.time_unit = day
		elif time_unit == "year":
			self.time_unit = year
		else:
			raise Exception(f"Time unit {time_unit} not supported.")

	def advance_time(self):
		self.eps = self.eq_mom.eps_tot.x.array

		eps_rate = (self.eps - self.eps_old) / self.dt
		eps_rate = numpy2torch(eps_rate.reshape((-1, 3, 3)))
		eps_rate_norm = to.linalg.norm(eps_rate, dim=(1,2))
		eps_rate_max_local = float(eps_rate_norm.max())
		eps_rate_max_global = MPI.COMM_WORLD.allreduce(eps_rate_max_local, op=MPI.MAX)

		eps_acc = (eps_rate_max_global - self.eps_rate_max_old) / self.dt

		if self.mat is not None:
			Fvp_local_max = float(self.mat.elems_ne[-1].Fvp.max())
			self.Fvp = MPI.COMM_WORLD.allreduce(Fvp_local_max, op=MPI.MAX)

			# self.Fvp = float(self.mat.elems_ne[-1].Fvp.max())

			diff_pred = abs(self.Fvp - self.Fvp_pred)
			
			dFdt = (self.Fvp - self.Fvp_old)/self.dt
			self.Fvp_pred = self.Fvp + dFdt*self.dt

			if dFdt < 0 and self.Fvp_pred < 0:
				self.dt = min(self.dt*1.2, self.dt_max)
			elif dFdt < 0 and self.Fvp_pred >= 0:
				self.dt = min(self.dt*1.1, self.dt_max)
			elif dFdt > 0 and self.Fvp_pred < 0 and diff_pred > 0.2:
				self.dt = max(self.dt*0.9, self.dt_min)
			elif dFdt > 0 and self.Fvp_pred < 0 and diff_pred <= 0.2:
				self.dt = min(self.dt*1.1, self.dt_max)
			elif dFdt > 0 and self.Fvp_pred >= 0 and diff_pred > 0.01:
				self.dt = max(self.dt*0.1, self.dt_min)
			elif dFdt > 0 and self.Fvp_pred >= 0 and diff_pred <= 0.01:
				self.dt = min(self.dt*1.1, self.dt_min)
			else:
				pass

			self.Fvp_old = self.Fvp




		if MPI.COMM_WORLD.rank == 0:
			# print(eps_rate_max_global, eps_acc, self.dt/self.time_unit)
			if self.mat is not None:
				print(self.Fvp, dFdt, self.Fvp_pred, self.dt/self.time_unit, diff_pred)
			print()
			# print(self.eq_mom.eps_tot.x.array[0])


		self.t += self.dt

	def keep_looping(self):
		return self.t < self.t_final


