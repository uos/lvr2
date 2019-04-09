import gdal
import numpy as np
import sys

num_sets = len(sys.argv) - 1
inputs = sys.argv[1:]
mins = np.empty(num_sets)
means = np.empty(num_sets)
maxs = np.empty(num_sets)

for i in range(0, num_sets):
    ds = gdal.Open(inputs[i], gdal.GA_ReadOnly)

    raster_count = ds.RasterCount
    x_size = ds.RasterXSize
    y_size = ds.RasterYSize
    num_channels = raster_count - 1

    values = np.empty((raster_count, y_size, x_size), dtype=np.float32)
    for j in range(0, num_channels):
        values[j] = np.array(ds.GetRasterBand(j + 1).ReadAsArray()).astype(np.float32)

    np.append(mins, values.min)
    np.append(means, values.mean)
    np.append(maxs, values.max)

    ds = None
    values = None

all_min = np.amin(mins)
all_mean = np.mean(means)
all_max = np.amax(maxs)
denominator = all_mean - all_min

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

    values /= 255.0
    values = ((values - all_min) / denominator)
    values = (values * 255.0).astype(np.int)

    driver = gdal.GetDriverByName("GTiff")
    ds = driver.Create("normed_" + inputs[i], x_size, y_size, raster_count, gdal.GDT_UInt16)
    for j in range(0, num_channels):
        ds.GetRasterBand(j + 1).WriteArray(values[j])

    values = None
    ds = None
