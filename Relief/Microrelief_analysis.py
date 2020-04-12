import os
from PIL import Image as im
import numpy as np
import math as mt
from progressbar import ProgressBar


def focal_statistics(raster, kernel, no_data, oper_type):
    dem = im.open(raster)
    dem_arr = np.array(dem)
    img_new = dem_arr.copy()
    img_new_arr = np.array(img_new)
    max_row = dem_arr.shape[0] - 1
    max_col = dem_arr.shape[1] - 1
    krl = np.shape(kernel)[0]
    i_max = max_row - 1
    j_max = max_col - 1
    for i in range(i_max):
        for j in range(j_max):
            matrix = dem_arr[i:i + krl, j:j + krl]
            mtx_new = matrix
            for k, l in enumerate(matrix):
                for x, y in enumerate(l):
                    if y == no_data:
                        centre_index = mtx_new.shape[0] // 2
                        mtx_new[k, x] = mtx_new[centre_index, centre_index]
                    no_data_count = np.count_nonzero(mtx_new == no_data)
            if no_data_count <= 3:
                max_idx = krl - 1
                mtx_multiply = np.empty([krl, krl], dtype=float)
                for ind in range(0, max_idx + 1):
                    for jnd in range(0, max_idx + 1):
                        mtx_multiply[ind][jnd] = mtx_new[ind][jnd] * kernel[ind][jnd]
                range_value = []
                for ind in range(0, max_idx + 1):
                    for jnd in range(0, max_idx + 1):
                        if mtx_multiply[ind][jnd] != 0:
                            range_value.append(mtx_multiply[ind][jnd])
                range_txt = np.max(range_value) - np.min(range_value)
                min_txt = np.min(range_value)
                max_txt = np.max(range_value)
                dictionary = {
                    'range': range_txt,
                    'min': min_txt,
                    'max': max_txt
                }
                range_val = dictionary[oper_type]
            img_new_arr[i + 1, j + 1] = range_val
    range_ready = im.fromarray(img_new_arr)
    return range_ready


bar = ProgressBar()
os.chdir('C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\TIF')
raster_dem = 'jasionka_start.tif'
nodata = -32767.0
kernel_W = [[1, 0, 1],
            [1, 1, 1],
            [1, 0, 1]]
kernel_N = [[1, 1, 1],
            [0, 1, 0],
            [1, 1, 1]]
kernel_NW = [[1, 1, 0],
             [1, 1, 1],
             [0, 1, 1]]
kernel_NE = [[0, 1, 1],
             [1, 1, 1],
             [1, 1, 0]]
wheelbase = 2.728
ride_height = 0.19

v2_v1 = focal_statistics(raster_dem, kernel_N, nodata, 'range')
v1 = focal_statistics(raster_dem, kernel_N, nodata, 'min')
v2_v1_arr = np.array(v2_v1)
v1_arr = np.array(v1)
dem = im.open(raster_dem)
dem_arr = np.array(dem)
phi = np.arctan(v2_v1_arr/wheelbase)
x = ride_height/np.cos(phi)
hp = np.fabs(v2_v1_arr/2.0) + v1_arr + x
decision = dem_arr >= hp
analysis_ready = im.fromarray(decision)
analysis_ready.save('Honker_N.tif')
