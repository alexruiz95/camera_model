#!/usr/bin/python3

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import fabs, ceil, floor, tan, radians, degrees
import sys # for maxsize
import camera_model as cm


#=================================== MAIN =====================================#

# ESA LIES
#
# ST: 20degx15deg, 2592x1944pix, apparent mag 6 (5) maxd 89 km, mind 50m (coc 4 pix) 29m (coc 7 pix)
# acceptable CoC ~9um
# Near camera: 40degx30deg, 2592x1944px (IR:1024x768px), maxd 40m, mind 2m (20cm if no CoC req)
# f/4, 12mm
# docking: 77.3degx61.9deg, 2592x1944px (IR:1024x768px), maxd 15m, mind 9cm
# f/5.6, 3mm

def final_test():#res = 1944, focal_length = 16, f_number = 2.2, pixel_size = 2.2):
	# camera specs
	FoV = 61.93 #40			# deg
	res = 1944 			# pixels
	focal_length = 3 #12	# mm
	# dof = sys.maxsize 	# DEFINITELY > hyperfocal distance
	f_number = 5.6
	pixel_size = 2.2	# um
	pixel_flag = True

	# distance data in meters
	a = np.arange(0.15, 10, .01)
	b = np.arange(11, 100, .25)
	dist = np.concatenate([a,b])

	acceptCoC = 4 #8		# pixels
	acceptCoC_mm = acceptCoC*pixel_size/cm.MM_2_UM # to mm
	hyperfocal = cm.hyperfocal_dist(focal_length, f_number, acceptCoC_mm)
	dof = hyperfocal/cm.M_2_MM # if focused at hyperfocal
	# dof = 0.3
	near_d = cm.near_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)
	far_d = cm.far_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)

	min_feat_size = .1#0.36
	t_min_d = 1
	min_feat_pix = cm.target_size_in_pix(t_min_d, FoV, res, min_feat_size)

	max_feat_size = .1* cm.M_2_MM#1.8 
	min_target_pix = 10#36
	cover = 0.90

	print("Hyperfocal distance is: ", str(hyperfocal), " mm")
	if (far_d == sys.maxsize):
		far_str = "infinity"
	else:
		far_str = "{:d} m".format(far_d/cm.M_2_MM)
	print("Depth of field spans: [{} m, {}]".format((ceil(near_d)/cm.M_2_MM),
		far_str))
	print("target size in pixels from 30 cm away: ", min_feat_pix)
	print("Max distance for {}m target feat size: {} m".format(
		max_feat_size/cm.M_2_MM, cm.max_target_dist(FoV, res, min_target_pix,
			max_feat_size)/cm.M_2_MM))
	print("Min distance for {}m target feat size: {} cm".format(
		min_feat_size, cm.min_target_dist_with_cover(FoV,
			min_feat_size, cover)*cm.M_2_CM))
	# print("Min distance for {}m target feat size: {} cm".format(
	# 	min_feat_size, cm.min_target_dist(FoV,
	# 		min_feat_size)*cm.M_2_CM))
	print("min FoV is: {:.2f} degrees".format(cm.min_FoV(t_min_d, min_feat_size)))


	if pixel_flag:
		# pixel way
		vget_circle_of_confusion_in_pix =  np.vectorize(cm.get_circle_of_confusion_in_pix, 
			excluded = ['N', 'f', 's_s', 'pixel_size'])
		coc_pix = vget_circle_of_confusion_in_pix(dist, f_number, focal_length, 
			cm.dof2s_s(dof,focal_length), pixel_size)
	else:
		# mm way
		vget_circle_of_confusion =  np.vectorize(cm.get_circle_of_confusion, 
			excluded = ['N', 'f', 's_s'])
		coc = vget_circle_of_confusion(dist, f_number, focal_length, 
			cm.dof2s_s(dof,focal_length))
		# valid_idx = np.where(coc <= acceptCoC_mm)[0]
		# point_idx = valid_idx[coc[valid_idx].argmax()]
		# print("crit dist [m]: ", dist[point_idx])


	# Create plots with pre-defined labels.
	fig, ax = plt.subplots()
	if pixel_flag:
		ax.plot(dist, coc_pix, 'k', label='CoC diameter [pixels]')
		ax.hlines(acceptCoC, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [pixels]')
		plt.ylabel('CoC diameter [pixels]')
	else:
		ax.plot(dist, coc, 'k--', label='CoC diameter [mm]')
		ax.hlines(acceptCoC_mm, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [mm]')
		plt.ylabel('CoC diameter [mm]')


	plt.annotate(['Near distance: ', str(ceil(near_d)/cm.M_2_CM),'cm'],
	         xy=(near_d, acceptCoC), 
	         xycoords='data',
	         xytext=(+0, +0), 
	         textcoords='offset points', 
	         fontsize=10,
	         arrowprops=dict(facecolor='blue'))
	ax.set_ylim([-1,15])
	# plt.xscale('log')
	plt.title('Circle of confusion')
	plt.grid(b=True, which='major', color='k', linestyle='-')
	plt.grid(b=True, which='minor', color='grey', linestyle='--')
	legend = ax.legend(loc='best', shadow=True)
	plt.xlabel('Distance [m]')
	plt.tick_params(axis='x', which='minor')
	# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

	plt.show()
	pass 

def FoV_test():
	# camera specs
	res = 1944#768 			# pixels
	pixel_size = 2.2#14  	# um

	# distance data in meters
	dist = [0.3, 100]
	n_pixels = 10		# pixels required to detect edge
	
	min_feat_size = 0.1
	max_feat_size = 0.1
	# min_feat_pix = cm.target_size_in_pix(dist[0], FoV, res, min_feat_size)

	FoV_list = [44, 31, 24, 17, 12]

	for FoV in FoV_list:
		min_dist = cm.min_target_dist(FoV, res, min_feat_size)
		max_dist = cm.max_target_dist(FoV, res, n_pixels, max_feat_size)

		print(('{}deg: min distance [m]: {}; max distance [m]: {}').format(
			FoV, min_dist, max_dist))
	pass


def f_number_test():
	# camera specs
	FoV = 44 #40			# deg
	res = 1944 			# pixels
	focal_length = 16 #12	# mm
	# dof = sys.maxsize 	# DEFINITELY > hyperfocal distance
	pixel_size = 2.2 	# um
	pixel_flag = True

	# distance data in meters
	a = np.arange(0.15, 20, .01)
	b = np.arange(20, 100, .25)
	dist = np.concatenate([a,b])

	acceptCoC = 4 #8		# pixels
	acceptCoC_mm = acceptCoC*pixel_size/cm.MM_2_UM # to mm
	
	vget_circle_of_confusion_in_pix =  np.vectorize(
		cm.get_circle_of_confusion_in_pix, 
		excluded = ['N', 'f', 's_s', 'pixel_size'])
	vget_circle_of_confusion =  np.vectorize(cm.get_circle_of_confusion, 
		excluded = ['N', 'f', 's_s'])
	

	f_number = [1, 1.2, 1.4, 1.8, 2, 2.2, 2.8, 4, 5.6]
	coc_pix = []
	coc = []

	for i in range(0, len(f_number)):
		hyperfocal = cm.hyperfocal_dist(focal_length, f_number[i], acceptCoC_mm)
		dof = hyperfocal/cm.M_2_MM # if focused at hyperfocal
		# dof = .50
		near_d = cm.near_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)
		far_d = cm.far_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)

		print("{:d}: f/{:.1f}".format(i, f_number[i]))
		print("Hyperfocal distance is: ", hyperfocal.__str__(), " mm")
		if (far_d == sys.maxsize):
			far_str = "infinity"
		else:
			far_str = [(far_d/cm.M_2_MM).__str__(), " m" ]
		print("Depth of field spans: [{} m, {}]".format((ceil(near_d)/cm.M_2_MM),
			far_str))


		if pixel_flag:
			# pixel way
			arr = vget_circle_of_confusion_in_pix(dist, f_number[i], focal_length, 
				cm.dof2s_s(dof,focal_length), pixel_size)
			coc_pix.append(arr)
		else:
			# mm way
			arr = vget_circle_of_confusion(dist, f_number[i], focal_length, 
				cm.dof2s_s(dof,focal_length))
			coc.append(arr)


		pass
	# Create plots with pre-defined labels.
	fig, ax = plt.subplots()
	colors = ['r', 'k', 'b', 'c', 'm', 'g', '#D2691E', 'y', '#556b2f']

	if pixel_flag:
		for i in range(0, len(f_number)):
			ax.plot(dist, coc_pix[i], colors[i], 
				label=('f/{:.1f}').format(f_number[i]))
	else:
		for i in range(0, len(f_number)):
			ax.plot(dist, coc[i], colors[i], 
				label=('f/{:.1f}').format(f_number[i]))

	if pixel_flag:
		ax.hlines(acceptCoC, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [pixels]')
		plt.ylabel('CoC diameter [pixels]')
	else:
		ax.hlines(acceptCoC_mm, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [mm]')
		plt.ylabel('CoC diameter [mm]')
	
	plt.annotate(['Near distance: ', str(ceil(near_d)/cm.M_2_CM),'cm'],
	         xy=(near_d, acceptCoC), 
	         xycoords='data',
	         xytext=(+0, +0), 
	         textcoords='offset points', 
	         fontsize=10,
	         arrowprops=dict(facecolor='blue'))
	ax.set_ylim([-1,15])
	# plt.xscale('log')
	plt.title('Circle of confusion, [variable N]')
	plt.grid(b=True, which='major', color='k', linestyle='-')
	plt.grid(b=True, which='minor', color='grey', linestyle='--')
	legend = ax.legend(loc='best', shadow=True)
	plt.xlabel('Distance [m]')
	plt.tick_params(axis='x', which='minor')
	# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

	plt.show()
	pass 

def focal_test():
	# camera specs
	FoV = 44 #40			# deg
	res = 1944 			# pixels
	# dof = sys.maxsize 	# DEFINITELY > hyperfocal distance
	pixel_size = 2.2 	# um
	pixel_flag = True

	# distance data in meters
	a = np.arange(0.15, 20, .01)
	b = np.arange(20, 100, .25)
	dist = np.concatenate([a,b])

	acceptCoC = 4 #8		# pixels
	acceptCoC_mm = acceptCoC*pixel_size/cm.MM_2_UM # to mm
	
	vget_circle_of_confusion_in_pix =  np.vectorize(
		cm.get_circle_of_confusion_in_pix, 
		excluded = ['N', 'f', 's_s', 'pixel_size'])
	vget_circle_of_confusion =  np.vectorize(cm.get_circle_of_confusion, 
		excluded = ['N', 'f', 's_s'])
	

	f_number = 4.6
	focal_length = [8, 12, 16, 20, 24, 28, 36, 40, 50] #12	# mm
	coc_pix = []
	coc = []

	for i in range(0, len(focal_length)):
		hyperfocal = cm.hyperfocal_dist(focal_length[i], f_number, acceptCoC_mm)
		# dof = hyperfocal/cm.M_2_MM # if focused at hyperfocal
		dof = 0.3
		near_d = cm.near_dist_acceptable(focal_length[i], (dof*cm.M_2_MM), hyperfocal)
		far_d = cm.far_dist_acceptable(focal_length[i], (dof*cm.M_2_MM), hyperfocal)
		
		print("{:d}: {:d}mm".format(i, focal_length[i]))
		print("Hyperfocal distance is: ", hyperfocal.__str__(), " mm")
		if (far_d == sys.maxsize):
			far_str = "infinity"
		else:
			far_str = [(far_d/cm.M_2_MM).__str__(), " m" ]
		print("Depth of field spans: [{} m, {}]".format((ceil(near_d)/cm.M_2_MM),
			far_str))


		if pixel_flag:
			# pixel way
			arr = vget_circle_of_confusion_in_pix(dist, f_number, focal_length[i], 
				cm.dof2s_s(dof,focal_length[i]), pixel_size)
			coc_pix.append(arr)
		else:
			# mm way
			arr = vget_circle_of_confusion(dist, f_number, focal_length[i], 
				cm.dof2s_s(dof,focal_length[i]))
			coc.append(arr)


		pass
	# Create plots with pre-defined labels.
	fig, ax = plt.subplots()
	colors = ['r', 'k', 'b', 'c', 'm', 'g', '#D2691E', 'y', '#556b2f']

	if pixel_flag:
		for i in range(0, len(focal_length)):
			ax.plot(dist, coc_pix[i], colors[i], 
				label=('f/{:.1f}, {:d}mm').format(f_number, focal_length[i]))
	else:
		for i in range(0, len(focal_length)):
			ax.plot(dist, coc[i], colors[i], 
				label=('f/{:.1f}, {:d}mm').format(f_number, focal_length[i]))

	if pixel_flag:
		ax.hlines(acceptCoC, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [pixels]')
		plt.ylabel('CoC diameter [pixels]')
	else:
		ax.hlines(acceptCoC_mm, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [mm]')
		plt.ylabel('CoC diameter [mm]')
	
	plt.annotate(['Near distance: ', str(ceil(near_d)/cm.M_2_CM),'cm'],
	         xy=(near_d, acceptCoC), 
	         xycoords='data',
	         xytext=(+0, +0), 
	         textcoords='offset points', 
	         fontsize=10,
	         arrowprops=dict(facecolor='blue'))
	ax.set_ylim([-1,15])
	# plt.xscale('log')
	plt.title('Circle of confusion, [variable f]')
	plt.grid(b=True, which='major', color='k', linestyle='-')
	plt.grid(b=True, which='minor', color='grey', linestyle='--')
	legend = ax.legend(loc='best', shadow=True)
	plt.xlabel('Distance [m]')
	plt.tick_params(axis='x', which='minor')
	# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

	plt.show()
	pass 

def focal_and_f_test():
	# camera specs
	FoV = 44 #40			# deg
	res = 1944 			# pixels
	# dof = sys.maxsize 	# DEFINITELY > hyperfocal distance
	pixel_size = 2.2 	# um
	pixel_flag = True

	# distance data in meters
	a = np.arange(0.15, 20, .01)
	b = np.arange(20, 100, .25)
	dist = np.concatenate([a,b])

	acceptCoC = 4 #8		# pixels
	acceptCoC_mm = acceptCoC*pixel_size/cm.MM_2_UM # to mm
	
	vget_circle_of_confusion_in_pix =  np.vectorize(
		cm.get_circle_of_confusion_in_pix, 
		excluded = ['N', 'f', 's_s', 'pixel_size'])
	vget_circle_of_confusion =  np.vectorize(cm.get_circle_of_confusion, 
		excluded = ['N', 'f', 's_s'])
	

	f_number = [1, 1.2, 1.4, 1.8, 2, 2.2, 2.8, 4, 5.6]
	focal_length = [8, 12, 16, 20, 24, 28, 36, 40, 50] #12	# mm
	coc_pix = []
	coc = []

	for i in range(0, len(f_number)):
		hyperfocal = cm.hyperfocal_dist(focal_length[i], f_number[i], acceptCoC_mm)
		dof = hyperfocal/cm.M_2_MM # if focused at hyperfocal
		# dof = 1
		near_d = cm.near_dist_acceptable(focal_length[i], (dof*cm.M_2_MM), hyperfocal)
		far_d = cm.far_dist_acceptable(focal_length[i], (dof*cm.M_2_MM), hyperfocal)

		print("Hyperfocal distance is: ", hyperfocal.__str__(), " mm")
		if (far_d == sys.maxsize):
			far_str = "infinity"
		else:
			far_str = [(far_d/cm.M_2_MM).__str__(), " m" ]
		print("Depth of field spans: [{} m, {}]".format((ceil(near_d)/cm.M_2_MM),
			far_str))


		if pixel_flag:
			# pixel way
			arr = vget_circle_of_confusion_in_pix(dist, f_number[i], focal_length[i], 
				cm.dof2s_s(dof,focal_length[i]), pixel_size)
			coc_pix.append(arr)
		else:
			# mm way
			arr = vget_circle_of_confusion(dist, f_number[i], focal_length[i], 
				cm.dof2s_s(dof,focal_length[i]))
			coc.append(arr)


		pass
	# Create plots with pre-defined labels.
	fig, ax = plt.subplots()
	colors = ['r', 'k', 'b', 'c', 'm', 'g', '#D2691E', 'y', '#556b2f']

	if pixel_flag:
		for i in range(0, len(f_number)):
			ax.plot(dist, coc_pix[i], colors[i], 
				label=('f/{:.1f}, {:d}mm').format(f_number[i], focal_length[i]))
	else:
		for i in range(0, len(f_number)):
			ax.plot(dist, coc[i], colors[i], 
				label=('f/{:.1f}, {:d}mm').format(f_number[i], focal_length[i]))

	if pixel_flag:
		ax.hlines(acceptCoC, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [pixels]')
		plt.ylabel('CoC diameter [pixels]')
	else:
		ax.hlines(acceptCoC_mm, -0.5, dist[dist.size-1], colors='r', 
			linestyles='dashdot', label='Acceptable CoC [mm]')
		plt.ylabel('CoC diameter [mm]')
	
	plt.annotate(['Near distance: ', str(ceil(near_d)/cm.M_2_CM),'cm'],
	         xy=(near_d, acceptCoC), 
	         xycoords='data',
	         xytext=(+0, +0), 
	         textcoords='offset points', 
	         fontsize=10,
	         arrowprops=dict(facecolor='blue'))
	ax.set_ylim([-1,15])
	# plt.xscale('log')
	plt.title('Circle of confusion, [variable N, f]')
	plt.grid(b=True, which='major', color='k', linestyle='-')
	plt.grid(b=True, which='minor', color='grey', linestyle='--')
	legend = ax.legend(loc='best', shadow=True)
	plt.xlabel('Distance [m]')
	plt.tick_params(axis='x', which='minor')
	# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

	plt.show()
	pass 

final_test()
# FoV_test()
# f_number_test()
# focal_and_f_test()