import richdem as rd
import matplotlib
import matplotlib.pyplot as plt
import os
import time
import PIL

start = time.time()
# Na początku trzeba raster przekonwertować do układu metrycznego, tutaj jest zrobione do UTM34 (gdalwarp)
path = 'C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\LIDAR\\NMT'
dem_path = path + '\\nmt_lidar.tif'
dem_plot = path + '\\L_NMT_N-34-138-B-b_utm.tif'
os.chdir("C:\\Users\\wojo1\\Desktop\\Doktorat\\Microrelief\\Data\\TIF")
# string = os.popen("gdalinfo " + dem_plot).read().rstrip()
# min_value = string.splitlines()
# min_float_value = float(min_value[-2][23:])
# print(os.popen('gdalinfo L_NMT_N-34-138-B-b_utm.tif').read().rstrip())

dem = rd.LoadGDAL(dem_plot)
# plt.imshow(dem, interpolation='bilinear')
# plt.clim(vmin=0, vmax=None)
# plt.colorbar(mappable=None, cax=None)
# plt.show()

slope = rd.TerrainAttribute(dem, attrib='slope_degrees')
rd.rdShow(slope, axes=False, cmap='jet', figsize=(10, 8))
plt.show()
# rd.SaveGDAL("nmt_python.tif", slope)
end = time.time()
print('process time: ', end-start)