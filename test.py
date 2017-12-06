#!/usr/bin/python3

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import fabs, ceil, floor, tan, radians, degrees
import sys # for maxsize
import camera_model as cm


#=================================== MAIN =====================================#

def final_test():
	# camera specs
	FoV = 40 			# deg
	res = 1944 			# pixels
	focal_length = 12 	# mm
	# dof = sys.maxsize 	# DEFINITELY > hyperfocal distance
	f_number = 2
	pixel_size = 2.2 	# um
	pixel_flag = True

	# distance data in meters
	a = np.arange(0.15, 10, .01)
	b = np.arange(11, 100, .25)
	dist = np.concatenate([a,b])

	acceptCoC = 8		# pixels
	acceptCoC_mm = acceptCoC*pixel_size/cm.MM_2_UM # to mm
	hyperfocal = cm.hyperfocal_dist(focal_length, f_number, acceptCoC_mm)
	dof = hyperfocal/cm.M_2_MM # if focused at hyperfocal
	# dof = 1
	near_d = cm.near_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)
	far_d = cm.far_dist_acceptable(focal_length, (dof*cm.M_2_MM), hyperfocal)

	min_feat_size = 0.3
	min_feat_pix = cm.target_size_in_pix(0.3, FoV, res, min_feat_size)

	print("Hyperfocal distance is: ", hyperfocal.__str__(), " mm")
	if (far_d == sys.maxsize):
		far_str = "infinity"
	else:
		far_str = [(far_d/cm.M_2_MM).__str__(), " m" ]
	print("Depth of field spans: [{} cm, {}]".format((ceil(near_d)/cm.M_2_CM),far_str))
	print("target size in pixels from 30 cm away: ", min_feat_pix)


	if pixel_flag:
		# pixel way
		vget_circle_of_confusion_in_pix =  np.vectorize(cm.get_circle_of_confusion_in_pix, 
			excluded = ['N', 'f', 's_s', 'pixel_size'])
		coc_pix = vget_circle_of_confusion_in_pix(dist, f_number, focal_length, 
			cm.dof2s_s(dof,focal_length), pixel_size)
	else:
		# mm way
		vget_circle_of_confusion =  np.vectorize(get_circle_of_confusion, 
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
	plt.xscale('log')
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
	res = 1944 			# pixels
	pixel_size = 2.2 	# um

	# distance data in meters
	dist = [0.3, 100]
	acceptCoC = 8		# pixels
	
	min_feat_size = 0.3
	max_feat_size = 3.7
	min_feat_pix = cm.target_size_in_pix(dist[0], FoV, res, min_feat_size)

	min_dist = cm.min_target_dist(FoV, res, min_feat_size)
	max_dist = cm.max_target_dist(FoV, res, max_feat_size)

	print("Hyperfocal distance is: ", hyperfocal.__str__(), " mm")
	if (far_d == sys.maxsize):
		far_str = "infinity"
	else:
		far_str = [(far_d/cm.M_2_MM).__str__(), " m" ]
	print("Depth of field spans: [{} cm, {}]".format((ceil(near_d)/cm.M_2_CM),far_str))
	print("target size in pixels from 30 cm away: ", min_feat_pix)


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
	plt.xscale('log')
	plt.title('Circle of confusion')
	plt.grid(b=True, which='major', color='k', linestyle='-')
	plt.grid(b=True, which='minor', color='grey', linestyle='--')
	legend = ax.legend(loc='best', shadow=True)
	plt.xlabel('Distance [m]')
	plt.tick_params(axis='x', which='minor')
	# ax.xaxis.set_minor_formatter(FormatStrFormatter('%.1f'))

	plt.show()
	pass

FoV_test()