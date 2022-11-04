/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * Options.cpp
 *
 * @date Nov 21, 2010
 * @author Thomas Wiemann
 * @author Malte Hillmann
 */

#include "Options.hpp"

#include "lvr2/config/lvropenmp.hpp"
#include "lvr2/algorithm/HLODTree.hpp"

namespace lvr2
{
// definitions necessary to use custom types with boost::program_options

std::istream& operator>>(std::istream& in, LSROutput& output)
{
    std::string token;
    in >> token;
    std::transform(token.begin(), token.end(), token.begin(), ::tolower);
    if (token == "bigmesh" || token == "mesh")
        output = LSROutput::BigMesh;
    else if (token == "chunks" || token == "chunksply")
        output = LSROutput::ChunksPly;
    else if (token == "chunkshdf5" || token == "hdf5")
        output = LSROutput::ChunksHdf5;
    else if (token == "3dtiles" || token == "tiles3d" || token == "tiles")
#ifdef LVR2_USE_3DTILES
        output = LSROutput::Tiles3d;
#else
        throw boost::program_options::error("If you want to use " + token + ", please call cmake with  -DWITH_3DTILES=ON  ");
#endif
    else
        in.setstate(std::ios_base::failbit);
    return in;
}
std::ostream& operator<<(std::ostream& out, LSROutput output)
{
    switch (output)
    {
    case LSROutput::BigMesh:    out << "bigMesh";    break;
    case LSROutput::ChunksPly:  out << "chunksPly";  break;
    case LSROutput::ChunksHdf5: out << "chunksHdf5"; break;
#ifdef LVR2_USE_3DTILES
    case LSROutput::Tiles3d:    out << "3dTiles";    break;
#endif
    }
    return out;
}
} // namespace lvr2

namespace std
{
template<typename T>
std::ostream& operator<<(std::ostream &os, const std::vector<T> &vec) 
{
      os << "[";
    for (size_t i = 0; i < vec.size(); ++i)
    {
        os << vec[i];
        if (i != vec.size() - 1)
            os << ", ";
    }
    os << "]";
    return os; 
}
} // namespace std

namespace LargeScaleOptions
{

using namespace boost::program_options;

Options::Options(int argc, char** argv) : BaseOption(argc, argv)
{
    // Create option descriptions

    // convert output from std::unordered_set to vector for boost::program_options
    std::vector<lvr2::LSROutput> output(m_options.output.begin(), m_options.output.end());

    bool noExtrude = false;
    bool noMergeChunkBorders = false;

    m_descr.add_options()
    ("help", bool_switch(&m_help),
     "Produce this help message")

    ("inputFile", value<fs::path>(&m_inputFile),
     "Input file or folder name.")

    ("voxelSizes,v", value<std::vector<float>>(&m_options.voxelSizes)->multitoken()->default_value(m_options.voxelSizes),
     "Voxelsize of grid used for reconstruction. multitoken option: it is possible to enter more then one voxelsize")

    ("partMethod", value<uint>(&m_options.partMethod)->default_value(m_options.partMethod),
     "Option to change the partition-process to a gridbase partition (0 = kd-Tree; 1 = VGrid)")

    ("chunkSize,c", value<float>(&m_options.bgVoxelSize)->default_value(m_options.bgVoxelSize),
     "Set the chunksize for the virtual grid.")

    ("noExtrude,E", bool_switch(&noExtrude),
     "Do not extend grid. Can be used to avoid artifacts in dense data sets but. Disabling will possibly create additional holes in sparse data sets.")

    ("useRansac", bool_switch(&m_options.useRansac),
     "Set this flag for RANSAC based normal estimation.")

    ("cleanContours", value<uint>(&m_options.cleanContours)->default_value(m_options.cleanContours),
     "Remove noise artifacts from contours. Same values are between 2 and 4.")

    ("fillHoles,f", value<uint>(&m_options.fillHoles)->default_value(m_options.fillHoles),
     "Maximum size for hole filling.")

    ("optimizePlanes,o", bool_switch(&m_options.optimizePlanes),
     "Shift all triangle vertices of a cluster onto their shared plane.")

    ("planeNormalThreshold,p", value<float>(&m_options.planeNormalThreshold)->default_value(m_options.planeNormalThreshold),
     "Normal threshold for plane optimization. Default 0.85 equals about 3 degrees.")

    ("planeIterations", value<uint>(&m_options.planeIterations)->default_value(m_options.planeIterations),
     "Number of iterations for plane optimization.")

    ("removeDanglingArtifacts,r", value<uint>(&m_options.removeDanglingArtifacts)->default_value(m_options.removeDanglingArtifacts),
     "Remove dangling artifacts, i.e. remove not connected surfaces smaller than n.")

    ("smallRegionThreshold", value<uint>(&m_options.smallRegionThreshold)->default_value(m_options.smallRegionThreshold),
     "Threshold for small region removal. If 0 nothing will be deleted.")

    ("kd", value<uint>(&m_options.kd)->default_value(m_options.kd),
     "Number of normals used for distance function evaluation")

    ("ki", value<uint>(&m_options.ki)->default_value(m_options.ki),
     "Number of normals used in the normal interpolation process")

    ("kn", value<uint>(&m_options.kn)->default_value(m_options.kn),
     "Size of k-neighborhood used for normal estimation.")

    ("minPlaneSize,m", value<uint>(&m_options.minPlaneSize)->default_value(m_options.minPlaneSize),
     "Minimum value for plane optimzation.")

    ("retesselate,t", bool_switch(&m_options.retesselate),
     "Retesselate regions that are in a regression plane. Implies --optimizePlanes.")

    ("lineFusionThreshold,l", value<float>(&m_options.lineFusionThreshold)->default_value(m_options.lineFusionThreshold),
     "Threshold for fusing line segments while tesselating.")

    ("threads", value<int>(&m_numThreads)->default_value(m_numThreads),
     "Number of threads to use. -1 for all available threads.")

    ("nodeSize,",value<uint>(&m_options.nodeSize)->default_value(m_options.nodeSize),
     "Max. Number of Points in a leaf. Only used if --partMethod is set to 0 (kd-Tree).")

    ("useGPU", bool_switch(&m_options.useGPU),
     "Use GPU for normal estimation.")

    ("useGPUDistances", bool_switch(&m_options.useGPUDistances),
     "Use GPU for signed distance computation. Implies --useGPU.")

    ("flipPoint", value<std::vector<float>>()->multitoken()->default_value(m_options.flipPoint),
     "Flippoint, used for GPU normal calculation, multitoken option: use it like this: --flipPoint x y z")

    ("output,O", value<std::vector<lvr2::LSROutput>>(&output)->multitoken()->default_value(output),
     "What to generate with the reconstruction. Supports multiple options. See below for details.")

    ("outputDir", value<fs::path>(&m_options.outputDir),
     "Output directory for generated files. Defaults to \"./<current date>/\".")

    ("noOverlapMerge", bool_switch(&noMergeChunkBorders),
     "Do not merge chunk borders. Merging chunk borders prevents gaps in chunked outputs, but takes a lot longer. "
     "Use this option if you only care about the bigGrid and/or want to save time.")

    ("scale", value<float>(&m_options.scale)->default_value(m_options.scale),
     "Scaling factor, applied to all input points")

    ("3dTilesCompress", bool_switch(&m_options.tiles3dCompress),
     "When generating 3D tiles: Compress Meshes with Draco compression.\n"
     "This will significantly reduce filesize and improve loading times when remotely viewing the tiles "
     "over a slow connection, but greatly increase loading times for local viewing.")

    ("3dTilesMemUsage", value<lvr2::AllowedMemoryUsage>(&m_options.tiles3dMemUsage)->default_value(m_options.tiles3dMemUsage),
     "When generating 3D tiles: How strictly should the algorithm try to save memory.\n"
     "Available Options: 'minimal', 'moderate', 'unbounded' or a number in [0, 2].\n"
     "Less Memory used always means more time required to generate tiles.")

    ;

    try
    {
        setup();

        if (m_help)
        {
            return;
        }

        m_options.extrude = !noExtrude;
        m_options.mergeChunkBorders = !noMergeChunkBorders;

        if (m_numThreads > 0)
        {
            lvr2::OpenMPConfig::setNumThreads(m_numThreads);
        }

        m_options.output.clear();
        for (auto& o : output)
        {
            m_options.output.insert(o);
        }

        if (!m_variables.count("inputFile"))
        {
            throw error("You must specify an input file.");
        }
        if (!fs::exists(m_inputFile))
        {
            throw error("Input file does not exist.");
        }
    }
    catch(const error& e)
    {
        std::cout << e.what() << std::endl;

        std::cout << std::endl;
        std::cout << "See --help for more information." << std::endl;
        m_printed = true;
        return;
    }

    m_options.ensureCorrectness();
}

bool Options::printUsage()
{
    if (m_printed)
    {
        return true;
    }
    if (!m_help)
    {
        return false;
    }
    std::cout << std::endl;
    std::cout << m_descr << std::endl;
    std::cout << std::endl;
    std::cout << R"======(
OUTPUT OPTIONS
    Options --output, -O accept one or more of the following tokens:
        bigMesh | mesh
            Output one big Mesh into "<outputDir>/mesh.ply". Uses A LOT of memory.
        chunksPly | chunks
            Output one mesh per chunk into "<outputDir>/chunks/<x>_<y>_<z>.ply".
        chunksHdf5
            Output one mesh per chunk into "<outputDir>/chunks.h5" in Group "/chunks/<x>_<y>_<z>".
            Meshes in the hdf5 file are stored using a PMPMesh.
            Use PMPMesh(HighFive::Group) to read them.
        3dTiles | tiles3d
            Output a 3D Tiles tileset for rendering to "<outputDir>/mesh.3dtiles".)======";

#ifndef LVR2_USE_3DTILES
    std::cout << R"======(
            To use this option, you have to reconfigure cmake with  -DWITH_3DTILES=ON  .)======";
#endif
    std::cout << R"======(

    Chunk options and 3dTiles require --partMethod 1 (VGrid, the default).
    Multiple Options can be used simultaneously:
        lvr2_largescale_reconstruct -O mesh chunks chunksHdf5 3dTiles
    Generates all possible output (not recommended).

    By default, only the big mesh is generated. This should be disabled for any
    truly large datasets.)======" << std::endl;
    m_printed = true;
    return true;
}

} // namespace LargeScaleOptions
