#!/usr/bin/python3

# 
# camera_model.py
#
# module to characterize cameras
# 

import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from math import fabs, ceil, floor, tan, atan, radians, degrees
import sys # for maxsize

M_2_MM = 1000
MM_2_UM = 1000
M_2_CM = 100
CM_2_MM = 10

def get_circle_of_confusion(s_t, N, f, s_s):
	"""
	return: diameter of CoC in mm
	d_CoC = (1/N)* abs(f-s_s*(s_t-f)/s_t)
	args:
	f: focal_length: in mm 
	N: f_stop: ratio of focal length over the entrance pupil 
	s_s: distance b/w lens and sensor when in focus, in mm
	s_t: distance from target, in m
	"""
	s_t *= M_2_MM
	return (1/N) * fabs(f-s_s*(s_t-f)/s_t)

def get_circle_of_confusion_in_pix(s_t, N, f, s_s, pixel_size = 2.2):
	"""
	return: diameter of CoC in pixels
	d_CoC = (1/N)* abs(f-s_s*(s_t-f)/s_t)
	args:
	f: focal_length: in mm 
	N: f_stop: ratio of focal length over the entrance pupil 
	dof: depth_of_field: focus POINT, distance b/w lens and focused target, in m
	s_t: distance from target, in m
	pixel_size: size of a pixel square on the sensor, measured in um
	"""
	# s_s = dof2s_s(dof, f)
	d_CoC = get_circle_of_confusion(s_t, N, f, s_s)
	return (d_CoC/(pixel_size/MM_2_UM))

def target_size_in_pix(s_t, FoV, res, feat_size):
	"""
	return the apparent size of target feature in pixels
	FoV: field of view, in degrees (one dimension)
	s_t: distance from target, in m
	res: the sensor resolution in pixels (one dimension)
	feat_size: size of target feature in m 
	"""
	FoV_rad = radians(FoV)
	iFoV = FoV_rad/res 		# sliver of FoV per pixel
	x = s_t * tan(iFoV)		# pixel projection s_t metres away
	return ceil(feat_size/x)

def min_target_dist(FoV, feat_size):
	"""
	return the minimum distance where the feature is still in the FoV
	FoV: field of view, in degrees (one dimension)
	res: the sensor resolution in pixels (one dimension)
	feat_size: size of target feature in m 
	"""
	# FoV_rad = radians(FoV)
	# iFoV = FoV_rad/res
	# x = feat_size/res 		# required pixel projection
	# return x/tan(iFoV)
	return (feat_size/2)/tan(radians(FoV)/2)

def min_target_dist_with_cover(FoV, feat_size, cover):
	feat_size = feat_size/cover
	return min_target_dist(FoV, feat_size)

def min_FoV(dist, feat_size):
	return degrees(2*atan(feat_size/(2*dist)))

def max_target_dist(FoV, res, min_n_pixel, feat_size):
	FoV_rad = radians(FoV)
	iFoV = FoV_rad/res
	x = feat_size/min_n_pixel 	# required pixel projection for detection
	return x/tan(iFoV)

def hyperfocal_dist(f, N, c):
	"""
	returns hyperfocal distance in mm based on focal length, f-stop, acceptable 
	CoC
	f: focal length: in mm
	N: f-number
	c: circle of confusion in mm
	"""
	H = ((f**2)/(N*c)) + f
	return H

def near_dist_acceptable(f, focus, H):
	"""
	f: focal length: in mm
	focus in mm
	H: hyperfocal distance in mm
	"""
	if (H+focus-(2*f)) == 0:
		return sys.maxsize # effectively infinite
	return (focus*(H-f))/(H+focus-(2*f))

def far_dist_acceptable(f, focus, H):
	"""radians
	f: focal length: in mm
	focus in mm
	H: hyperfocal distance in mm
	"""
	if (H-focus == 0):
		return sys.maxsize
	return (focus*(H-f))/(H-focus)

def dof2s_s(dof, f):
	"""dof in m"""
	dof *= M_2_MM
	return (f * dof) / (dof - f)

def s_s2dof(s_s, f):
	"""s_s in mm"""
	return ((f * s_s) / (s_s - f))/M_2_MM


