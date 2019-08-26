# Toolkit for the classification of hyperspectral datasets

## Conversion HDF5 to TIFF
1. Create a HDF5 file of a scan dataset using the lvr2_hdf5tool
2. extract the radiometric data to a GDAL readable TIFF file like so:
   in your build/bin execute `./lvr2_hdf5togeotiff <ipnut path of .h5 file> <output path of .tif file>`

## Post processing 
You can process your radiometric data using the script normalize.py. This will apply a normalization and afterwards a savgol filter to the radiometric data.
The script requires python 3.x and the following python libraries:
+ gdal
+ numpy
+ scipy
Execute the script like so: `python normalize.py <input path of GDAL readable dataset file (e.g. .tif)> <output path of processed .tif file>`

## Conversion of a classification to an rbg file
A classification in the form of another one-channeled TIFF file can be converted to an rgb file using the script create_png.py
The script requires python 3.x and the following python libraries:
+ gdal
+ numpy
+ matplotlib
+ OpenCV (cv2)
Execute the script like so: `python create_png.py <input path of GDAL readable dataset file (e.g. .tif)> <output path of rgb file writeable by OpenCV (e.g. .png)>`