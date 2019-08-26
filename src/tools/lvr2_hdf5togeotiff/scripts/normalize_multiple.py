import gdal
import numpy as np
import sys
import os

num_sets = len(sys.argv) - 1
inputs = sys.argv[1:]
mins = np.empty((num_sets), dtype=np.float32)
means = np.empty((num_sets), dtype=np.float32)
maxs = np.empty((num_sets), dtype=np.float32)

# determine min, max and mean of each dataset
for i in range(0, num_sets):
    ds = gdal.Open(inputs[i], gdal.GA_ReadOnly)

    raster_count = ds.RasterCount
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    num_channels = raster_count - 1

    values = np.empty((raster_count, y_size, x_size), dtype=np.float32)
    for j in range(0, num_channels):
        values[j] = np.array(ds.GetRasterBand(j + 1).ReadAsArray()).astype(np.float32)

    mins[i] = np.amin(values)
    means[i] = np.mean(values)
    maxs[i] = np.amax(values)

    ds = None
    values = None

# determine global min, max and mean
all_min = np.amin(mins)
print("Calculated minimum value over data sets: ", all_min)
all_mean = np.mean(means)
print("Calculated mean value over data sets: ", all_mean)
all_max = np.amax(maxs)
print("Calculated maximum value over data sets: ", all_max)
denominator = all_mean - all_min

# perform normalization for each dataset
for i in range(0, num_sets):
    ds = gdal.Open(inputs[i], gdal.GA_ReadOnly)

    raster_count = ds.RasterCount
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    num_channels = raster_count - 1

    values = np.empty((raster_count, y_size, x_size), dtype=np.float32)
    for j in range(0, num_channels):
        values[j] = np.array(ds.GetRasterBand(j + 1).ReadAsArray()).astype(np.float32)
    ds = None

    values = ((values - all_min) / denominator)
    values = values * (all_max / 2.)
    values = values.astype(np.int)

    driver = gdal.GetDriverByName("GTiff")
    filename, file_extension = os.path.splitext(inputs[i])
    ds = driver.Create(filename + "_normed" + file_extension, x_size, y_size, raster_count, gdal.GDT_UInt16)
    for j in range(0, num_channels):
        ds.GetRasterBand(j + 1).WriteArray(values[j])

    values = None
    ds = None
