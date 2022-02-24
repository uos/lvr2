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

## Windows (Experimental)

Experimental windows/vcpkg build. This assumes Visual Studio 2019 with clang is installed.

1.) First install chocolatey to be able to install pkgconfig:

install chocolatey:
https://chocolatey.org/install

2.) Install pkgconfiglite:
```powershell
> choco install pkgconfiglite
```powershell


3.) Install vcpkg in repository:
```powershell
cd lvr2
git clone https://github.com/microsoft/vcpkg
.\vcpkg\bootstrap-vcpkg.bat
```powershell

4.) Install MSMPI from here: https://github.com/microsoft/Microsoft-MPI/releases

5.) Install lvr2 dependencies, this will take a while because the packages are compiled from source:
```powershell
.\vcpkg\vcpkg.exe install boost-accumulators:x64-windows boost-algorithm:x64-windows boost-align:x64-windows boost-any:x64-windows boost-array:x64-windows boost-asio:x64-windows boost-assert:x64-windows boost-assign:x64-windows boost-atomic:x64-windows boost-beast:x64-windows boost-bimap:x64-windows boost-bind:x64-windows boost-build:x64-windows boost-callable-traits:x64-windows boost-chrono:x64-windows boost-circular-buffer:x64-windows boost-compatibility:x64-windows boost-compute:x64-windows boost-concept-check:x64-windows boost-config:x64-windows boost-container-hash:x64-windows boost-container:x64-windows boost-context:x64-windows boost-contract:x64-windows boost-conversion:x64-windows boost-convert:x64-windows boost-core:x64-windows boost-coroutine2:x64-windows boost-coroutine:x64-windows boost-crc:x64-windows boost-date-time:x64-windows boost-detail:x64-windows boost-dll:x64-windows boost-dynamic-bitset:x64-windows boost-endian:x64-windows boost-exception:x64-windows boost-fiber:x64-windows boost-filesystem:x64-windows boost-flyweight:x64-windows boost-foreach:x64-windows boost-format:x64-windows boost-function-types:x64-windows boost-function:x64-windows boost-functional:x64-windows boost-fusion:x64-windows boost-geometry:x64-windows boost-gil:x64-windows boost-graph-parallel:x64-windows boost-graph:x64-windows boost-hana:x64-windows boost-heap:x64-windows boost-histogram:x64-windows boost-hof:x64-windows boost-icl:x64-windows boost-integer:x64-windows boost-interprocess:x64-windows boost-interval:x64-windows boost-intrusive:x64-windows boost-io:x64-windows boost-iostreams:x64-windows boost-iterator:x64-windows boost-json:x64-windows boost-lambda:x64-windows boost-leaf:x64-windows boost-lexical-cast:x64-windows boost-local-function:x64-windows boost-locale:x64-windows boost-lockfree:x64-windows boost-log:x64-windows boost-logic:x64-windows boost-math:x64-windows boost-metaparse:x64-windows boost-modular-build-helper:x64-windows boost-move:x64-windows boost-mp11:x64-windows boost-mpl:x64-windows boost-msm:x64-windows boost-multi-array:x64-windows boost-multi-index:x64-windows boost-multiprecision:x64-windows boost-nowide:x64-windows boost-numeric-conversion:x64-windows boost-odeint:x64-windows boost-optional:x64-windows boost-outcome:x64-windows boost-parameter-python:x64-windows boost-parameter:x64-windows boost-pfr:x64-windows boost-phoenix:x64-windows boost-poly-collection:x64-windows boost-polygon:x64-windows boost-pool:x64-windows boost-predef:x64-windows boost-preprocessor:x64-windows boost-process:x64-windows boost-program-options:x64-windows boost-property-map:x64-windows boost-property-tree:x64-windows boost-proto:x64-windows boost-ptr-container:x64-windows boost-python:x64-windows boost-qvm:x64-windows boost-random:x64-windows boost-range:x64-windows boost-ratio:x64-windows boost-rational:x64-windows boost-regex:x64-windows boost-safe-numerics:x64-windows boost-scope-exit:x64-windows boost-serialization:x64-windows boost-signals2:x64-windows boost-smart-ptr:x64-windows boost-sort:x64-windows boost-spirit:x64-windows boost-stacktrace:x64-windows boost-statechart:x64-windows boost-static-assert:x64-windows boost-static-string:x64-windows boost-stl-interfaces:x64-windows boost-system:x64-windows boost-test:x64-windows boost-thread:x64-windows boost-throw-exception:x64-windows boost-timer:x64-windows boost-tokenizer:x64-windows boost-tti:x64-windows boost-tuple:x64-windows boost-type-erasure:x64-windows boost-type-index:x64-windows boost-type-traits:x64-windows boost-typeof:x64-windows boost-ublas:x64-windows boost-uninstall:x64-windows boost-units:x64-windows boost-unordered:x64-windows boost-utility:x64-windows boost-uuid:x64-windows boost-variant2:x64-windows boost-variant:x64-windows boost-vcpkg-helpers:x64-windows boost-vmd:x64-windows boost-wave:x64-windows boost-winapi:x64-windows boost-xpressive:x64-windows boost-yap:x64-windows boost:x64-windows brotli:x64-windows bzip2:x64-windows cfitsio:x64-windows cgal:x64-windows curl:x64-windows curl[non-http]:x64-windows curl[schannel]:x64-windows curl[ssl]:x64-windows curl[sspi]:x64-windows curl[winssl]:x64-windows double-conversion:x64-windows egl-registry:x64-windows eigen3:x64-windows expat:x64-windows flann:x64-windows freeglut:x64-windows freetype:x64-windows freetype[brotli]:x64-windows freetype[bzip2]:x64-windows freetype[png]:x64-windows freetype[zlib]:x64-windows gdal:x64-windows geos:x64-windows glew:x64-windows gmp:x64-windows gsl:x64-windows harfbuzz:x64-windows hdf5:x64-windows hdf5[cpp]:x64-windows hdf5[szip]:x64-windows hdf5[zlib]:x64-windows highfive:x64-windows icu:x64-windows jasper:x64-windows jsoncpp:x64-windows libffi:x64-windows libgeotiff:x64-windows libharu:x64-windows libharu[notiffsymbols]:x64-windows libiconv:x64-windows libjpeg-turbo:x64-windows libjpeg-turbo:x86-windows liblzma:x64-windows liblzma:x86-windows libogg:x64-windows libpng:x64-windows libpq:x64-windows libpq[openssl]:x64-windows libpq[zlib]:x64-windows libtheora:x64-windows libwebp:x64-windows libwebp[nearlossless]:x64-windows libwebp[simd]:x64-windows libwebp[unicode]:x64-windows libxml2:x64-windows lz4:x64-windows mpfr:x64-windows netcdf-c:x64-windows opencl:x64-windows opencv4:x64-windows opencv4[dnn]:x64-windows opencv4[jpeg]:x64-windows opencv4[png]:x64-windows opencv4[quirc]:x64-windows opencv4[tiff]:x64-windows opencv4[webp]:x64-windows opengl:x64-windows openjpeg:x64-windows openssl:x64-windows pcre2:x64-windows pegtl-2:x64-windows proj4:x64-windows proj4[tiff]:x64-windows protobuf:x64-windows pugixml:x64-windows python3:x64-windows qt5-activeqt:x64-windows qt5-base:x64-windows qt5-declarative:x64-windows qt5-imageformats:x64-windows qt5-multimedia:x64-windows qt5-networkauth:x64-windows qt5-quickcontrols2:x64-windows qt5-svg:x64-windows qt5-tools:x64-windows qt5-xmlpatterns:x64-windows qt5:x64-windows qt5[activeqt]:x64-windows qt5[declarative]:x64-windows qt5[essentials]:x64-windows qt5[imageformats]:x64-windows qt5[multimedia]:x64-windows qt5[networkauth]:x64-windows qt5[quickcontrols2]:x64-windows qt5[svg]:x64-windows qt5[tools]:x64-windows quirc:x64-windows sqlite3:x64-windows sqlite3:x86-windows sqlite3[tool]:x64-windows sqlite3[tool]:x86-windows szip:x64-windows tiff:x64-windows tiff:x86-windows utfcpp:x64-windows vs-yasm:x64-windows vtk:x64-windows vtk[opengl]:x64-windows vtk[qt]:x64-windows xxhash:x64-windows yaml-cpp:x64-windows yasm-tool-helper:x64-windows yasm-tool:x64-windows yasm:x64-windows yasm:x86-windows zlib:x64-windows zlib:x86-windows zstd:x64-windows  --clean-after-build```
Integrate in Visual Studio
```powershell
.\vcpkg\vcpkg.exe integrate install
```

6.) Setup toolchain in Visual Studio in CMakeSettings.json:
i) -> Toolset:
	clang_cl_x64
ii) ->Advanced settings:
 	-> Cmake generator:
		ninja

Known Issues:
Compilation with MSVC compiler is possible but there seem to be issues with openmp.
Cuda currently not working, OpenCl works fine.

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
