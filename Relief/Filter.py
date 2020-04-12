import os
from PIL import Image as im
import numpy as np
import math as mt
from progressbar import ProgressBar
from osgeo import gdal
from osgeo.gdalconst import GA_Update


def slope(img, mask, cellsize):
    img = im.open(img)
    img_arr = np.array(img)
    img_new = img.copy()
    img_new_arr = np.array(img_new)
    max_row = img_arr.shape[0] - 1
    max_col = img_arr.shape[1] - 1
    kernel = np.shape(mask)[0]
    global no_data
    no_data = -32767.0
    i = 0
    j = 0
    i_max = max_row - 1
    j_max = max_col - 1
    for i in bar(range(i_max)):
        for j in range(j_max):
            matrix = img_arr[i:i+kernel, j:j+kernel]
            mtx_new = matrix
            for k, l in enumerate(matrix):
                for x, y in enumerate(l):
                    if y == no_data:
                        centre_index = mtx_new.shape[0] // 2
                        mtx_new[k, x] = mtx_new[centre_index, centre_index]
                    no_data_count = np.count_nonzero(mtx_new == no_data)
            if no_data_count <= 3:
                max_idx = kernel - 1
                a = 0
                b = int(max_idx / 2)
                q = np.empty((b, 2))
                multiplication_value = np.sum(mask[0]) + np.sum(mask[max_idx])
                divisor = multiplication_value * cellsize
                mtx_multiply = mtx_new * mask
                for p in range(b):
                    dz_dx = (np.sum(mtx_multiply[:, a]) - np.sum(mtx_multiply[:, max_idx])) / divisor
                    dz_dy = (np.sum(mtx_multiply[a]) - np.sum(mtx_multiply[max_idx])) / divisor
                    q[0][0] = dz_dx
                    q[0][1] = dz_dy
                    sum_dx = np.sum(q[:, 0])
                    sum_dy = np.sum(q[:, 1])
                    rise_run = mt.sqrt(sum_dx ** 2 + sum_dy ** 2)
                    slope_dgr = mt.atan(rise_run) * 57.29578
                    img_new_arr[i + 1, j + 1] = slope_dgr
    slope_ready = im.fromarray(img_new_arr)
    return slope_ready


def assign_nodata(raster):
    rast = gdal.Open(raster, GA_Update)
    rast.GetRasterBand(1).SetNoDataValue(no_data)
    rast = None
    return raster


bar = ProgressBar()
os.chdir('C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\TIF')  # path to the NMT folder
nmt_name = 'jasionka_start.tif'
cellsize = 0.5  # wielkość piksela
mask = [[1, 2, 1],
        [2, 1, 2],
        [1, 2, 1]]

img_news = slope(nmt_name, mask, cellsize)
# assign_nodata(img_news)

# img_news.save('slope_script.tif')
os.system("start /wait cmd /k gdalwarp -t_srs EPSG:32634 slope_script.tif slope_script_UTM.tif")