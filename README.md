```console
         /\
        /  \               ##          ##      ##    #######         ######
       /    \              ##          ##      ##    ##     ##     ##      ##
      /      \             ##           ##    ##     ##      ##            ##
     /________\            ##           ##    ##     ##     ##            ##
    /\        /\           ##            ##  ##      #######             ##
   /  \      /  \          ##            ##  ##      ##    ##          ##
  /    \    /    \         ##             ####       ##     ##       ##
 /      \  /      \        ##########      ##        ##      ##    ##########
/________\/________\
```

# About

This library delivers tools to build surface reconstructions from point cloud
data. Additionally, the found surfaces will be classified into predefined categories. The main aim of this
project is to deliver fast and accurate algorithms for surface reconstruction with a strong focus on
robotic applications such as autonomous navigation and localization in complex environments.

# Download and Compilation from Source

## Step 0: Get the source code from our Github repository:

https://github.com/uos/lvr2 - develop

## Linux (Ubuntu 18.04, 20.04, 22.04, 24.04)

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

A C++17 compiler is required.

### Optional for NVIDIA graphics cards users

If you want to compile with CUDA support install the latest version of the CUDA toolkit, which you can find on NVIDIAs CUDA download site. To enable CUDA support, you need to compile the software with a compatible GCC version. All compatibilities are listed in `CMakeModules/max_cuda_gcc_version.cmake`.

### Step 2: Configure and build from sources:

```bash
mkdir build
cd build
cmake .. && make
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

You can experiment with the software using the the example dataset `scan.pts` from the `dat` folder. For a simple
reconstruction call in your build directory:

```bash
bin/lvr2_reconstruct ../dat/scan.pts
```

in the root directory of the project. This will create a file called
“triangle_mesh.ply” which can be displayed using a mesh viewer of your choice.

For all possible reconstruction parameters enter

```bash
bin/lvr2_reconstruct --help
```

<details>
<summary>Output:</summary>

```bash
Supported options:
  -x [ --xPos ] arg (=0)                Position of the x-coordinates in the 
                                        input point data (according to screen 
                                        coordinates).
  -y [ --yPos ] arg (=1)                Position of the y-coordinates in the 
                                        input data lines (according to screen 
                                        coordinates).
  -z [ --zPos ] arg (=2)                Position of the z-coordinates in the 
                                        input data lines (according to screen 
                                        coordinates).
  --sx arg (=1)                         Scaling factor for the x coordinates.
  --sy arg (=1)                         Scaling factor for the y coordinates.
  --sz arg (=1)                         Scaling factor for the z coordinates.
  --help                                Produce help message
  --inputFile arg                       Input file name. Supported formats are 
                                        ASCII (.pts, .xyz), .ply and .h5
  --inputSchema arg                     The ScanProjectSchema to use with the 
                                        input file. Options are HDF5, HDF5V2, 
                                        RAW, HYPERLIB, EUROC, RAWPLY, SLAM6D
  --outputDirectory arg (=./)           Directory where the output files are 
                                        placed
  --outputFile arg (=triangle_mesh.ply triangle_mesh.obj )
                                        Output file name. Supported formats are
                                        ASCII (.pts, .xyz) and .ply
  -v [ --voxelsize ] arg (=10)          Voxelsize of grid used for 
                                        reconstruction.
  --noExtrusion                         Do not extend grid. Can be used  to 
                                        avoid artefacts in dense data sets but.
                                        Disabling will possibly create 
                                        additional holes in sparse data sets.
  -i [ --intersections ] arg (=-1)      Number of intersections used for 
                                        reconstruction. If other than -1, 
                                        voxelsize will calculated 
                                        automatically.
  -p [ --pcm ] arg (=LVR2)              Point cloud manager used for point 
                                        handling and normal estimation. Choose 
                                        from {FLANN, STANN, PCL, NABO, LVR2, 
                                        LBVH_CUDA}.
  --nem arg (=0)                        Method for estimating point normals / 
                                        planes. 0: PCA (default), 1: RANSAC, 2:
                                        IPCA ilikebigbits, 3: IPCA exact. Make 
                                        sure the computing device is supporting
                                        the respective method.
  -d [ --decomposition ] arg (=PMC)     Defines the type of decomposition that 
                                        is used for the voxels (Standard 
                                        Marching Cubes (MC), Planar Marching 
                                        Cubes (PMC), Standard Marching Cubes 
                                        with sharp feature detection (SF), Dual
                                        Marching Cubes with an adaptive Octree 
                                        (DMC) or Tetraeder (MT) decomposition. 
                                        Choose from {MC, PMC, MT, SF}
  -o [ --optimizePlanes ]               Shift all triangle vertices of a 
                                        cluster onto their shared plane
  -c [ --clusterPlanes ]                Cluster planar regions based on normal 
                                        threshold, do not shift vertices into 
                                        regression plane.
  --cleanContours arg (=0)              Remove noise artifacts from contours. 
                                        Same values are between 2 and 4
  --planeIterations arg (=3)            Number of iterations for plane 
                                        optimization
  -f [ --fillHoles ] arg (=0)           Maximum size for hole filling
  --rda arg (=0)                        Remove dangling artifacts, i.e. remove 
                                        the clusters with less than n triangles
  --pnt arg (=0.850000024)              (Plane Normal Threshold) Normal 
                                        threshold for plane optimization. 
                                        Default 0.85 equals about 3 degrees.
  --smallRegionThreshold arg (=10)      Threshold for small region removal. If 
                                        0 nothing will be deleted.
  -w [ --writeClassificationResult ]    Write classification results to file 
                                        'clusters.clu'
  -e [ --exportPointNormals ]           Exports original point cloud data 
                                        together with normals into a single 
                                        file called 'pointnormals.ply'
  -g [ --saveGrid ]                     Writes the generated grid to a file 
                                        called 'fastgrid.grid. The result can 
                                        be rendered with qviewer.
  -s [ --saveOriginalData ]             Save the original points and the 
                                        estimated normals together with the 
                                        reconstruction into one file 
                                        ('triangle_mesh.ply')
  --scanPoseFile arg                    ASCII file containing scan positions 
                                        that can be used to flip normals
  --kd arg (=5)                         Number of normals used for distance 
                                        function evaluation
  --ki arg (=10)                        Number of normals used in the normal 
                                        interpolation process
  --kn arg (=10)                        Size of k-neighborhood used for normal 
                                        estimation
  --mp arg (=7)                         Minimum value for plane optimzation
  -t [ --retesselate ]                  Retesselate regions that are in a 
                                        regression plane. Implies 
                                        --optimizePlanes.
  --lft arg (=0.00999999978)            (Line Fusion Threshold) Threshold for 
                                        fusing line segments while tesselating.
  --generateTextures                    Generate textures during finalization.
  --texMinClusterSize arg (=100)        Minimum number of faces of a cluster to
                                        create a texture from
  --texMaxClusterSize arg (=0)          Maximum number of faces of a cluster to
                                        create a texture from (0 = no limit)
  --textureAnalysis                     Enable texture analysis features for 
                                        texture matchung.
  --texelSize arg (=1)                  Texel size that determines texture 
                                        resolution.
  --classifier arg (=GREY)              Classfier object used to color the 
                                        mesh. Possible values: GREY, SIMPSONS, 
                                        JET, HOT, HSV, SHSV, WHITE, BLACK
  -r [ --recalcNormals ]                Always estimate normals, even if given 
                                        in .ply file.
  --threads arg (=16)                   Number of threads
  --sft arg (=0.899999976)              Sharp feature threshold when using 
                                        sharp feature decomposition
  --sct arg (=0.699999988)              Sharp corner threshold when using sharp
                                        feature decomposition
  --reductionRatio arg (=0)             Percentage of faces to remove via 
                                        edge-collapse (0.0 means no reduction, 
                                        1.0 means to remove all faces which can
                                        be removed)
  --tp arg                              Path to texture pack
  --co arg                              Coefficents file for texture matching 
                                        based on statistics
  --nsc arg (=16)                       Number of colors for texture statistics
  --nccv arg (=64)                      Number of colors for texture matching 
                                        based on color information
  --ct arg (=50)                        Coherence threshold for texture 
                                        matching based on color information
  --colt arg (=3.40282347e+38)          Threshold for texture matching based on
                                        colors
  --stat arg (=3.40282347e+38)          Threshold for texture matching based on
                                        statistics
  --feat arg (=3.40282347e+38)          Threshold for texture matching based on
                                        features
  --cro                                 Use texture matching based on cross 
                                        correlation.
  --patt arg (=100)                     Threshold for pattern extraction from 
                                        textures
  --mtv arg (=3)                        Minimum number of votes to consider a 
                                        texture transformation as correct
  --vcfp                                Use color information from pointcloud 
                                        to paint vertices
  --useGPU                              GPU normal estimation
  --flipPoint arg                       Flippoint --flipPoint x y z
  -q [ --texFromImages ]                Foo Bar ............
  --scanPositionIndex arg               List of scan positions to load from a 
                                        scan project
  --minSpectralChannel arg (=0)         Minimum Spectral Channel Index for 
                                        Ranged Texture Generation
  --maxSpectralChannel arg (=0)         Maximum Spectral Channel Index for 
                                        Ranged Texture Generation
  -a [ --projectDir ] arg               Foo Bar ............
  --transformScanPosition               Transform the scan with the 
                                        scanpositions pose when using 
                                        --scanPositionIndex
  --outputMeshName arg (=default)       The name of the saved mesh
  --inputMeshName arg                   The name of the mesh to load from the 
                                        file
  --inputMeshFile arg                   The file to load the mesh from
  --reduceScan arg (=0)                 Use Octree reduction algorithm with the
                                        given gridsize when after loading the 
                                        scans
  --reduceScanMinPoints arg (=1)        The number of points an octree voxel 
                                        has to contain to be considered 
                                        occupied
```

</details>


## Docs
For more information, build the Doxygen documentation by calling
```bash
make doc
```
in the build directory.


# Installation

After successful compilation, you will find the generated example tools in the ./bin/ directory. Optionally, you can install the library and header files to your system:

```bash
sudo make install
```

## Use in your own CMake project

After installation, you can include the lvr2 project in your own CMake project as follows:

```cmake
find_package(LVR2 REQUIRED)
add_definitions(${LVR2_DEFINITIONS})
include_directories(${LVR2_INCLUDE_DIRS})

add_executable(my_own_exec my_own_code.cpp)

target_link_libraries(my_own_exec
  ${LVR2_LIBRARIES}
)
```


# Citation

Please reference the following papers when using the lvr2 library in your scientific work.

```bib
@inproceedings{wiemann2018,
  author={Wiemann, Thomas and Mitschke, Isaak and Mock, Alexander and Hertzberg, Joachim},
  booktitle={2018 Second IEEE International Conference on Robotic Computing (IRC)}, 
  title={{Surface Reconstruction from Arbitrarily Large Point Clouds}}, 
  year={2018},
  pages={278-281},
  doi={10.1109/IRC.2018.00059}}
```


## ROS build

You can simply download this library and compile it inside your ROS workspace. The following ROS distributions are supported:

|  Version   |  Supported Distributions    |
|:-----------|:----------------------------|
| ROS 1 | [![noetic](https://github.com/uos/lvr2/actions/workflows/ros-noetic.yml/badge.svg)](https://github.com/uos/lvr2/actions/workflows/ros-noetic.yml) |
| ROS 2 | [![humble](https://github.com/uos/lvr2/actions/workflows/ros-humble.yml/badge.svg)](https://github.com/uos/lvr2/actions/workflows/ros-humble.yml) [![iron](https://github.com/uos/lvr2/actions/workflows/ros-iron.yml/badge.svg)](https://github.com/uos/lvr2/actions/workflows/ros-iron.yml) [![jazzy](https://github.com/uos/lvr2/actions/workflows/ros-jazzy.yml/badge.svg)](https://github.com/uos/lvr2/actions/workflows/ros-jazzy.yml) |