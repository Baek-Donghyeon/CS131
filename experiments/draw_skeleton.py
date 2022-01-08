# A simple visualizing tool for making videos from frames

import h5py
import numpy as np
import cv2 as cv
from cv2 import COLORMAP_PLASMA, VideoWriter, VideoWriter_fourcc


# Vectorized version of world2pixels
def world2pixels(points, width, height, fx, fy):
    pixels = np.zeros((points.shape[0], points.shape[1], 2))
    pixels[:, :, 1] = height / 2 - points[:, :, 1] * fy / points[:, :, 2]
    pixels[:, :, 0] = points[:, :, 0] * fx / points[:, :, 2] + width / 2
    return np.asarray(pixels, dtype=np.uint16)


# Configuration of the video
width = 320
height = 240
FPS = 24
keypoints_num = 15
fx = 285.714
fy = 285.714
linewidth = 3


# Your video is stored at
fourcc = VideoWriter_fourcc(*'MP42')
video = VideoWriter('./res/pose_estimation.avi', fourcc, float(FPS), (width, height))


# Load the data
gt_file = r'./res/test_s3_gt.txt'
pred_file = r'./res/test_res.txt'
name_file = r'./res/test_name.txt'
depth_file = r'./datasets/itop/ITOP_side_test_depth_map.h5'

gt = np.loadtxt(gt_file)
gt = gt.reshape(gt.shape[0], -1, 3)

pred = np.loadtxt(pred_file)
pred = pred.reshape(pred.shape[0], -1, 3)

name = np.loadtxt(name_file)

with h5py.File(depth_file, 'r') as f:
    depthmap = f['data'][:]

print('gt: ', gt.shape)
print('pred: ', pred.shape)
print('name: ', name.shape)
print('depthmap:', depthmap.shape)


# Normalize the depth map
# [min, max] -> [0, 255]
depthmap *= (255/depthmap.max())
depthmap = np.asarray(depthmap, dtype=np.uint8)
FRAME = depthmap.shape[0]
id = 0
pixels = world2pixels(pred, width, height, fx, fy)


# Draw skeleton
for i in range(FRAME):
    
    # Depthmap at ith frame
    frame = cv.applyColorMap(depthmap[i], COLORMAP_PLASMA)
    
    # If ref_points exists, draw skeleton on the depthmap
    if name[id] == i:
        p = pixels[id]
        cv.line(frame, p[0], p[1], (178,102,255), linewidth)
        cv.line(frame, p[1], p[2], (153,153,255), linewidth)
        cv.line(frame, p[2], p[4], (102,102,255), linewidth)
        cv.line(frame, p[4], p[6], (51,51,255), linewidth)
        cv.line(frame, p[1], p[3], (255,153,153), linewidth)
        cv.line(frame, p[3], p[5], (255,102,102), linewidth)
        cv.line(frame, p[5], p[7], (255,51,51), linewidth)
        cv.line(frame, p[1], p[8], (230,0,230), linewidth)
        cv.line(frame, p[8], p[9], (153,255,255), linewidth)
        cv.line(frame, p[9], p[11], (102,255,255), linewidth)
        cv.line(frame, p[11], p[13], (51,255,255), linewidth)
        cv.line(frame, p[8], p[10], (204,255,153), linewidth)
        cv.line(frame, p[10], p[12], (178,255,102), linewidth)
        cv.line(frame, p[12], p[14], (153,255,51), linewidth)
        id += 1
        
    video.write(frame)
video.release()