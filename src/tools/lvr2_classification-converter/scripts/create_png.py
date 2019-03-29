import gdal
import sys
import numpy as np
import random
from matplotlib import cm
from multiprocessing import Process, Manager


# Input: Klassifikation als .tif
# Output: Klassifiziertes Panorama als PNG

def process_subarrays(i, png_values, values, colors):
    print("process started")
    shape = values.shape
    for j in range(shape[1]):
        for k in range(shape[2]):
            png_values[i, j, k] = colors[i, values[0, j, k] - 1]


def distinct_colors(n):
    # source: https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    colors = np.empty(n, 4)
    r = int(random.random() * 256)
    g = int(random.random() * 256)
    b = int(random.random() * 256)
    a = 255
    step = 256 / n
    for i in range(n):
        r += step
        g += step
        b += step
        r = int(r) % 256
        g = int(g) % 256
        b = int(b) % 256
        colors[i] = np.array([r, g, b, a])
    return colors


if __name__ == '__main__':
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

    #values = np.empty((raster_count, y_size, x_size), dtype=np.int)
    #for i in range(min_channel-1, max_channel):
    #    values[i] = np.array(ds.GetRasterBand(i+1).ReadAsArray()).astype(np.int)

    values = np.random.randint(1, 5, (1, 90, 74))

    min_val = np.amin(values)
    max_val = np.amax(values)

    num_classes = max_val - min_val + 1

    color_map = np.array(cm.get_cmap('plasma').colors)
    colors = np.empty((num_classes, 4), dtype=np.int)
    step = int((len(color_map)-1) / (num_classes-1))
    for class_i in range(num_classes):
        for rgb_channel in range(3):
            colors[class_i, rgb_channel] = int(color_map[step * class_i, rgb_channel] * 256)
        # alpha channel
        colors[class_i, 3] = 255

    shape = values.shape

    png_values = np.empty((4, shape[1], shape[2]))
    print(colors.shape, png_values.shape, values.shape)

    for rgb_channel in range(4):
        for y in range(shape[1]):
            for x in range(shape[2]):
                png_values[rgb_channel, y, x] = colors[values[0, y, x] - 1, rgb_channel]


    #png_values = np.empty((num_classes, shape[1], shape[2]))
    #manager = Manager()
    #dict = manager.dict()

    #jobs = []
    #for i in range(num_classes):
    #    p = Process(target=process_subarrays, args=(i, dict, values, colors))
    #    jobs.append(p)
    #    p.start()
    #for proc in jobs:
    #    proc.join()
    #    #for y in range(shape[1]):
    #    #    for z in range(shape[2]):
    #    #        png_values[x, y, z] = colors[x, values[0, y, z] - 1]
    #png_values = np.array(dict)
    #print(png_values.shape)
