import gdal
import numpy as np
import sys
from scipy.signal import savgol_filter

inname = sys.argv[1]
outname = sys.argv[2]

ds = gdal.Open(inname, gdal.GA_ReadOnly)
if not ds:
    print("Could not open file ", inname)

raster_count = ds.RasterCount
x_size = ds.RasterXSize
y_size = ds.RasterYSize

min_channel = 1
max_channel = ds.RasterCount
if len(sys.argv) == 5:
    min_channel = int(sys.argv[3])
    max_channel = int(sys.argv[4])

num_channels = max_channel - min_channel

values = np.empty((raster_count, y_size, x_size), dtype=np.float32)
for i in range(0, num_channels):
    values[i] = np.array(ds.GetRasterBand(i + 1).ReadAsArray()).astype(np.float32)

ds = None

# source: https://gitlab.informatik.uni-osnabrueck.de/figelbri/hyperspace
# normalize data
shape = values.shape
values = values.reshape(values.shape[0], -1)
means = values.mean(axis=0) # mean of every spectrum
mins = values.min(axis=0) # min of every spectrum
#maxs = values.max(axis=0)
vmax = np.amax(values)
values = ((values - mins) / (means - mins))

mins = None
means = None

#values = values * (maxs / 2.)
values = values * vmax / 2.

values = values.reshape(shape)

# source: https://gitlab.informatik.uni-osnabrueck.de/figelbri/hyperspace
# smooth spectral data
values = savgol_filter(values, 11, 2, axis=0)

# write normalized dataset
driver = gdal.GetDriverByName("GTiff")
ds = driver.Create(outname, x_size, y_size, raster_count, gdal.GDT_UInt16)
for i in range(0, num_channels):
    ds.GetRasterBand(i + 1).WriteArray(values[i])

