# About

This software delivers tools to build surface reconstructions from point cloud
data and a simple viewer to display the results. Additionally, the found
surfaces will be classified into predefined categories. The main aim of this
project is to deliver fast and accurate surface extraction algorithms for
robotic applications such as tele operation in unknown environments and
localization.

# Download and Compilation from Source

### Step 0: Get the source code from our Github repository:

https://github.com/uos/lvr2

## Linux (Ubuntu 18.04)

### Step 1: Install all required package dependencies: 

```bash
sudo apt-get install build-essential \
     cmake cmake-curses-gui libflann-dev \
     libgsl-dev libeigen3-dev libopenmpi-dev \
     openmpi-bin opencl-c-headers ocl-icd-opencl-dev \
     libvtk7-dev libvtk7-qt-dev libboost-all-dev \
     freeglut3-dev libhdf5-dev qtbase5-dev \
     qt5-default libqt5opengl5-dev liblz4-dev \
     libopencv-dev libyaml-cpp-dev
```

A C++17 compiler is required, e.g., g++7, gcc7 need bo installed.
If CUDA is installed you also need g++6, see "Optional for NVIDIA
graphics cards users" 


### Step 2: Configure and build from sources:

```bash
mkdir build
cd build
cmake .. && make
```

### Optional for NVIDIA graphics cards users: 

If you want to compile with CUDA support install the latest version of the CUDA toolkit, which you can find on NVIDIAs CUDA download site:

To enable CUDA support, you need to compile the software with a compatible GCC version. We have testet compilation with CUDA 9.1 and GCC 6. To use this compiler for compilation of CUDA generated code, set the `CUDA_HOST_COMPILER` option to `g++-6` is forced to g++-6. Please ensure that this version is installed on your system.
/
### Step 3: Installation

After successful compilation, you will find the generated example tools in the ./bin/ directory. Optionally, you can install the library and header files to your system:

```bash
sudo make install
```


## MacOS

Install the required libraries using [Homebrew](https://brew.sh):

```bash
brew install boost boost-mpi cmake eigen flann gcc glew gsl hdf5 opencv lz4 qt vtk 

mkdir build
cd build
cmake .. && make
```

# Usage

Your can experiment with the software using the provided dataset. For a simple
reconstruction call in yout build directory:

```bash
bin/lvr2_reconstruct ../dat/scan.pts
```

in the root directory of the project. This will create a file called
“triangle_mesh.ply” which can be displayed using the viewer application:

```bash
bin/lvr2_viewer triangle_mesh.ply
```

For more information, build the Doxygen documentation by calling
```bash
make doc
```
in the build directory.
