import gdal
import sys
import numpy as np
from matplotlib import cm
import cv2

# Takes a one channel classification TIFF file as input and converts it to a colorized PNG file
# Input: Classification as .tif
# Output: Colorized classification panorama as PNG

if len(sys.argv) == 3:
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

    values = np.empty((raster_count, y_size, x_size), dtype=np.int)
    for i in range(min_channel-1, max_channel):
        values[i] = np.array(ds.GetRasterBand(i+1).ReadAsArray()).astype(np.int)
else:

    values = np.random.randint(1, 5, (1, 900, 7480))
    outname = "./colored_classification.png"

min_val = np.amin(values)
max_val = np.amax(values)

num_classes = max_val - min_val + 1

#print("Loading color map...")
color_map = np.array(cm.get_cmap('viridis').colors)
colors = np.empty((num_classes, 4), dtype=np.uint)
step = int((len(color_map)-1) / (num_classes-1))
for class_i in range(num_classes):
    for rgb_channel in range(3):
        colors[class_i, rgb_channel] = int(color_map[step * class_i, rgb_channel] * 256)
    # alpha channel
    colors[class_i, 3] = 255

# demo
#colors[0] = [0, 255, 0, 255]        # greenVeg
#colors[2] = [191, 117, 0, 255]      # Gestein
#colors[1] = [0, 0, 255, 255]        # Wasser
#colors[3] = [102, 102, 102, 255]    # Schatten

print("Assigning data to colorized array...")
shape = values.shape
png_values = np.zeros((shape[1], shape[2], 4), dtype=np.uint)
# OpenCV requires BGRA channel order
for rgba_channel in reversed(range(4)):
    for y in range(shape[1]):
        for x in range(shape[2]):
            png_values[y, x, rgba_channel] = colors[values[0, y, x] - 1, rgba_channel]

print("Writing PNG file...")
if cv2.imwrite(outname, png_values):
    print("Done. Data has been written to", outname)
else:
    print("Could not write PNG file.")
