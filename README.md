ABOUT
=====

This software delivers tools to build surface reconstructions from point cloud
data and a simple viewer to display the results. Additionally, the found
surfaces will be classified into predefined categories. The main aim of this
project is to deliver fast and accurate surface extraction algorithms for
robotic applications such as tele operation in unknown environments and
localization.

This software is under permanent development and runs under Linux and MacOS. A
Windows version will be made avialable soon.


COMPILATION
===========

The software is built using cmake. To compile create a build subdirectory.
Switch to this directory and call:

```bash
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

The binaries are compiled to the ./bin directory in the base dir.

You may simply copy this:
```bash
mkdir build; cd build; cmake -DCMAKE_BUILD_TYPE=Release ..; make -j$(nproc); cd bin 
```

REQUIRED LIBRARIES
==================

  + OpenGL
  + glut
  + Qt 4.8 or higher for viewer
  + BOOST
    - Filesystem
    - Program Option
    - System
    - Thread
  + Qt 4.6 or newer (for qviewer)
  + libQGLViewer 2.3.9 or newer (for qviewer)
  + libXi
  + libXmu
  + VTK
  + QVTK for viewer
  + OpenCV 3 or 4

You need to compile libQGLViewer with CONFIG += no_keywords to be compatible 
with Boost. If the version that comes with your Linux distrubution is not
build this way, you have to compile and install the library manually with these 
flags. The sources can be found on the project website: http://www.libqglviewer.com/

```bash
sudo apt install libflann-dev liblz4-dev libgsl-dev libxmu-dev libboost-dev libeigen3-dev libboost-filesystem-dev libboost-program-options-dev libboost-thread-dev libboost-mpi-dev libboost-all-dev freeglut3-dev libvtk6-dev libvtk6-qt-dev libproj-dev cmake build-essential
```
**Note on OpenCV**: You probably need to compile OpenCV from Source. Make sure that the opencv_contrib packages are included in the installation.
```bash
git clone https://github.com/opencv/opencv.git
git clone https://github.com/opencv/opencv_contrib.git
cd opencv
mkdir build
cd build
cmake -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules ..
make -j$(nproc)
sudo make install
```

**Note on CUDA**: You need an NVIDIA Graphics card and the CUDA Toolkit from the official NVIDIA website. There have been problems with the .deb package, so you should probably use the .run file
	not all targets require CUDA

USAGE
=====

Your can experiment with the software using the provided dataset. For a simple
reconstruction call:

```bash
./bin/reconstruct -v 100 dat/scan.pts
```

in the root directory of the project. This will create a file called
“traingle_mesh.ply” which can be displayed using the viewer application:

```bash
./bin/lvr_viewer
```

If you want to use the example data, call for colorizing a point cloud without
color informaion call:

```bash
./colorize -d 100 -c ff0000 example-data/scan-no-color.3d example-data/scan-with-color.pts colored-scan.pts
```

For file conversion of point clouds and meshes use the convertmodel
application:

```bash
./convertmodel examplefile.pts examplefile.ply
```

For more information, build the Doxygen documentation by calling
```bash
make doc
```
in the build directory.
