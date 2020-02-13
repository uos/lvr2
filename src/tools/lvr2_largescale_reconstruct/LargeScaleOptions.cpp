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

/*
 * Options.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "LargeScaleOptions.hpp"

#include <fstream>
#include "lvr2/config/lvropenmp.hpp"

namespace LargeScaleOptions
{

using namespace boost::program_options;

Options::Options(int argc, char** argv) : BaseOption(argc, argv)
{
    // Create option descriptions
    vector<float> flippoint;
    flippoint.push_back(1000000);
    flippoint.push_back(1000000);
    flippoint.push_back(1000000);
    m_descr.add_options()
    ("help", "Produce help message")
    ("inputFile", value<vector<string>>(),"Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
    ("voxelSizes,v",value<vector<float>>(&m_voxelSizes)->multitoken(),"Voxelsize of grid used for reconstruction. multitoken option: it is possible to enter more then one voxelsize")
    ("bgVoxelSize,bgv",value<float>(&m_voxelsizeBG)->default_value(10),"Voxelsize of the bigGrid.")
    ("partMethod",value<int>(&m_partMethod)->default_value(1),"Option to change the partition-process to a gridbase partition (0 = kd-Tree; 1 = VGrid)")
    ("chunkSize",value<int>(&m_chunkSize)->default_value(20),"Set the chunksize for the virtual grid. (default: 20)")
    ("extrude", value<bool>(&m_extrude)->default_value(false), "Do not extend grid. Can be used to avoid artefacts in dense data sets but. Disabling will possibly create additional holes in sparse data sets.")
    ("intersections,i",value<int>(&m_intersections)->default_value(-1),"Number of intersections used for reconstruction. If other than -1, voxelsize will calculated automatically.")
    ("pcm,p",value<string>(&m_pcm)->default_value("FLANN"),"Point cloud manager used for point handling and normal estimation. Choose from {STANN, PCL, NABO}.")
    ("useRansac", "Set this flag for RANSAC based normal estimation.")
    ("decomposition,d",value<string>(&m_pcm)->default_value("PMC"),"Defines the type of decomposition that is used for the voxels (Standard Marching Cubes "
        "(MC), Planar Marching Cubes (PMC), Standard Marching Cubes with sharp feature detection "
        "(SF) or Tetraeder (MT) decomposition. Choose from {MC, PMC, MT, SF}")(
        "optimizePlanes,o", "Shift all triangle vertices of a cluster onto their shared plane")(
        "clusterPlanes,c",
        "Cluster planar regions based on normal threshold, do not shift vertices into regression "
        "plane.")("cleanContours",
                  value<int>(&m_cleanContourIterations)->default_value(0),
                  "Remove noise artifacts from contours. Same values are between 2 and 4")(
        "planeIterations",
        value<int>(&m_planeIterations)->default_value(3),
        "Number of iterations for plane optimization")(
        "fillHoles,f", value<int>(&m_fillHoles)->default_value(0), "Maximum size for hole filling")(
        "removeDanglingArtifacts,rda",
        value<int>(&m_removeDanglingArtifacts)->default_value(0),
        "Remove dangling artifacts, i.e. remove the n smallest not connected surfaces")(
        "planeNormalThreshold,pnt",
        value<float>(&m_planeNormalThreshold)->default_value(0.85),
        "(Plane Normal Threshold) Normal threshold for plane optimization. Default 0.85 equals "
        "about 3 degrees.")("smallRegionThreshold",
                            value<int>(&m_smallRegionThreshold)->default_value(0),
                            "Threshold for small region removal. If 0 nothing will be deleted.")(
        "writeClassificationResult,w", "Write classification results to file 'clusters.clu'")(
        "exportPointNormals,e",
        "Exports original point cloud data together with normals into a single file called "
        "'pointnormals.ply'")("saveGrid,g",
                              "Writes the generated grid to a file called 'fastgrid.grid. The "
                              "result can be rendered with qviewer.")(
        "saveOriginalData,s",
        "Save the original points and the estimated normals together with the reconstruction into "
        "one file ('triangle_mesh.ply')")(
        "scanPoseFile",
        value<string>()->default_value(""),
        "ASCII file containing scan positions that can be used to flip normals")(
        "kd",
        value<int>(&m_kd)->default_value(5),
        "Number of normals used for distance function evaluation")(
        "ki",
        value<int>(&m_ki)->default_value(10),
        "Number of normals used in the normal interpolation process")(
        "kn",
        value<int>(&m_kn)->default_value(10),
        "Size of k-neighborhood used for normal estimation")(
        "minPlaneSize,mp", value<int>(&m_minPlaneSize)->default_value(7), "Minimum value for plane optimzation")(
        "retesselate,t",
        "Retesselate regions that are in a regression plane. Implies --optimizePlanes.")(
        "lineFusionThreshold,lft",
        value<float>(&m_lineFusionThreshold)->default_value(0.01),
        "(Line Fusion Threshold) Threshold for fusing line segments while tesselating.")(
        "generateTextures", "Generate textures during finalization.")(
        "textureAnalysis", "Enable texture analysis features for texture matchung.")(
        "texelSize",
        value<float>(&m_texelSize)->default_value(1),
        "Texel size that determines texture resolution.")(
        "classifier",
        value<string>(&m_classifier)->default_value("PlaneSimpsons"),
        "Classfier object used to color the mesh.")("depth",
                                                    value<int>(&m_depth)->default_value(100),
                                                    "Maximum recursion depth for region growing.")(
        "recalcNormals,r", "Always estimate normals, even if given in .ply file.")(
        "threads",
        value<int>(&m_numThreads)->default_value(lvr2::OpenMPConfig::getNumThreads()),
        "Number of threads")("sft",
                             value<float>(&m_sft)->default_value(0.9),
                             "Sharp feature threshold when using sharp feature decomposition")(
        "sct",
        value<float>(&m_sct)->default_value(0.7),
        "Sharp corner threshold when using sharp feature decomposition")(
        "ecm",
        value<string>(&m_ecm)->default_value("QUADRIC"),
        "Edge collapse method for mesh reduction. Choose from QUADRIC, QUADRIC_TRI, MELAX, "
        "SHORTEST")("ecc",
                    value<int>(&m_numEdgeCollapses)->default_value(0),
                    "Edge collapse count. Number of edges to collapse for mesh reduction.")(
        "tp", value<string>(&m_texturePack)->default_value(""), "Path to texture pack")(
        "co",
        value<string>(&m_statsCoeffs)->default_value(""),
        "Coefficents file for texture matching based on statistics")(
        "nsc",
        value<unsigned int>(&m_numStatsColors)->default_value(16),
        "Number of colors for texture statistics")(
        "nccv",
        value<unsigned int>(&m_numCCVColors)->default_value(64),
        "Number of colors for texture matching based on color information")(
        "ct",
        value<unsigned int>(&m_coherenceThreshold)->default_value(50),
        "Coherence threshold for texture matching based on color information")(
        "colt",
        value<float>(&m_colorThreshold)->default_value(FLT_MAX),
        "Threshold for texture matching based on colors")(
        "stat",
        value<float>(&m_statsThreshold)->default_value(FLT_MAX),
        "Threshold for texture matching based on statistics")(
        "feat",
        value<float>(&m_featuresThreshold)->default_value(FLT_MAX),
        "Threshold for texture matching based on features")(
        "cro", "Use texture matching based on cross correlation.")(
        "patt",
        value<float>(&m_patternThreshold)->default_value(100),
        "Threshold for pattern extraction from textures")(
        "mtv",
        value<int>(&m_minimumTransformationVotes)->default_value(3),
        "Minimum number of votes to consider a texture transformation as correct")(
        "buff",
        value<unsigned int>(&m_bufferSize)->default_value(30000000),
        "Minimum number of votes to consider a texture transformation as correct")(
        "nodeSize, ns",
        value<unsigned int>(&m_octreeNodeSize)->default_value(1000000),
        "Max. Number of Points in a leaf (used to devide pointcloud)")(
        "outputFolder",
        value<string>(&m_outputFolderPath)->default_value(""),
        "Output Folder Path")("useGPU", "Use GPU for normal estimation")(
        "flipPoint", value<vector<float>>()->multitoken(), "Flippoint, used for GPU normal calculation, multitoken option: use it like this: --flipPoint x y z")(
        "lineReaderBuffer",
        value<size_t>(&m_lineReaderBuffer)->default_value(1024),
        "Size of input stream buffer when parsing point cloud files")(
        "interpolateBoxes", "Interpolate Boxes in intersection BoundingBox of two Grids")(
        "useNormals",
        "the ply file contains normals")
        ("bigMesh", value<bool>(&m_bigMesh)->default_value(true),"generate a .ply file of the reconstructed mesh")
            ("debugChunks", value<bool>(&m_debugChunks)->default_value(false), "generate .ply file for every chunk")
            ("scale",
                                         value<float>(&m_scaling)->default_value(1),
                                         "Scaling factor, applied to all input points")(
        "volumenSize",
        value<size_t>(&m_volumenSize)->default_value(0),
        "The volumen of the partitions. Volume = (voxelsize*volumenSize)^3 if not set kd-tree will "
        "be used")("onlyNormals", "If true, only normals will be generated");

    setup();
}
bool Options::printUsage() const
{
    if (m_variables.count("help"))
    {
        cout << endl;
        cout << m_descr << endl;
        return true;
    }
    else if (!m_variables.count("inputFile"))
    {
        cout << "Error: You must specify an input file." << endl;
        cout << endl;
        cout << m_descr << endl;
        return true;
    }
    return false;
}

bool Options::getBigMesh() const { return m_variables["bigMesh"].as<bool>();}

bool Options::getDebugChunks() const { return m_variables["debugChunks"].as<bool>();}

bool Options::useGPU() const { return m_variables.count("useGPU"); }

vector<float> Options::getVoxelSizes() const
{
    vector<float> dest;
    if (m_variables.count("voxelSizes"))
    {
        dest = (m_variables["voxelSizes"].as<vector<float>>());
    }
    else
    {
        dest.push_back(0.1);
    }

    return dest;
}

float Options::getVoxelsize() const { return getVoxelSizes()[0]; }

float Options::getBGVoxelsize() const { return m_variables["bgVoxelSize"].as<float>(); }

float Options::getScaling() const { return m_variables["scale"].as<float>(); }

int Options::getChunkSize() const { return (m_variables["chunkSize"].as<int>()); }

unsigned int Options::getNodeSize() const { return m_variables["nodeSize"].as<unsigned int>(); }

int Options::getPartMethod() const { return (m_variables["partMethod"].as<int>()); }

int Options::getKi() const { return m_variables["ki"].as<int>(); }

int Options::getKd() const { return m_variables["kd"].as<int>(); }

int Options::getKn() const { return m_variables["kn"].as<int>(); }

bool Options::useRansac() const { return (m_variables.count("useRansac")); }

bool Options::extrude() const { return m_extrude; }

int Options::getDanglingArtifacts() const { return (m_variables["removeDanglingArtifacts"].as<int>()); }

int Options::getCleanContourIterations() const { return m_variables["cleanContours"].as<int>(); }

int Options::getFillHoles() const { return (m_variables["fillHoles"].as<int>()); }

bool Options::optimizePlanes() const
{
    return m_variables.count("optimizePlanes") || m_variables.count("retesselate");
}

float Options::getNormalThreshold() const { return m_variables["planeNormalThreshold"].as<float>(); }

int Options::getPlaneIterations() const { return m_variables["planeIterations"].as<int>(); }

int Options::getMinPlaneSize() const { return m_minPlaneSize; }

/*
 * Definition from here on are not used (anymore?)
 */


size_t Options::getLineReaderBuffer() const { return m_variables["lineReaderBuffer"].as<size_t>(); }

unsigned int Options::getBufferSize() const { return m_variables["buff"].as<unsigned int>(); }




float Options::getSharpFeatureThreshold() const { return m_variables["sft"].as<float>(); }

float Options::getSharpCornerThreshold() const { return m_variables["sct"].as<float>(); }

int Options::getNumThreads() const { return m_variables["threads"].as<int>(); }



int Options::getIntersections() const { return m_variables["intersections"].as<int>(); }


vector<string> Options::getInputFileName() const
{
    return (m_variables["inputFile"].as<vector<string>>());
}

string Options::getPCM() const { return (m_variables["pcm"].as<string>()); }

string Options::getClassifier() const { return (m_variables["classifier"].as<string>()); }

string Options::getEdgeCollapseMethod() const { return (m_variables["ecm"].as<string>()); }

string Options::getDecomposition() const { return (m_variables["decomposition"].as<string>()); }

string Options::getScanPoseFile() const { return (m_variables["scanPoseFile"].as<string>()); }

int Options::getNumEdgeCollapses() const { return (m_variables["ecc"].as<int>()); }

bool Options::saveFaceNormals() const { return m_variables.count("saveFaceNormals"); }

bool Options::writeClassificationResult() const
{
    return m_variables.count("writeClassificationResult") || m_variables.count("w");
}

bool Options::doTextureAnalysis() const { return m_variables.count("textureAnalyis"); }

bool Options::filenameSet() const
{
    return (m_variables["inputFile"].as<vector<string>>()).size() > 0;
}

string Options::getOutputFolderPath() const
{
    return (m_variables["outputFolder"].as<vector<string>>())[0];
}

bool Options::recalcNormals() const { return (m_variables.count("recalcNormals")); }

bool Options::savePointNormals() const { return (m_variables.count("exportPointNormals")); }

bool Options::saveNormals() const { return (m_variables.count("saveNormals")); }

bool Options::saveGrid() const { return (m_variables.count("saveGrid")); }

bool Options::saveOriginalData() const { return (m_variables.count("saveOriginalData")); }

bool Options::clusterPlanes() const { return m_variables.count("clusterPlanes"); }

bool Options::colorRegions() const { return m_variables.count("colorRegions"); }

bool Options::retesselate() const { return m_variables.count("retesselate"); }

bool Options::generateTextures() const { return m_variables.count("generateTextures"); }

int Options::getSmallRegionThreshold() const
{
    return m_variables["smallRegionThreshold"].as<int>();
}

int Options::getDepth() const { return m_depth; }

float Options::getTexelSize() const { return m_texelSize; }

float Options::getLineFusionThreshold() const { return m_variables["lineFusionThreshold"].as<float>(); }

string Options::getTexturePack() const { return m_variables["tp"].as<string>(); }

unsigned int Options::getNumStatsColors() const { return m_variables["nsc"].as<unsigned int>(); }

unsigned int Options::getNumCCVColors() const { return m_variables["nccv"].as<unsigned int>(); }

unsigned int Options::getCoherenceThreshold() const { return m_variables["ct"].as<unsigned int>(); }

float Options::getColorThreshold() const { return m_variables["colt"].as<float>(); }

float Options::getStatsThreshold() const { return m_variables["stat"].as<float>(); }

float Options::getFeatureThreshold() const { return m_variables["feat"].as<float>(); }

bool Options::getUseCrossCorr() const { return m_variables.count("cro"); }

float Options::getPatternThreshold() const { return m_variables["patt"].as<float>(); }

int Options::getMinimumTransformationVotes() const { return m_variables["mtv"].as<int>(); }

bool Options::interpolateBoxes() const { return m_variables.count("interpolateBoxes"); }

bool Options::getUseNormals() const { return m_variables.count("useNormals"); }
size_t Options::getVolumenSize() const { return m_variables["volumenSize"].as<size_t>(); }

bool Options::onlyNormals() const { return m_variables.count("onlyNormals"); }
float* Options::getStatsCoeffs() const
{
    float* result = new float[14];
    std::ifstream in(m_variables["tp"].as<string>().c_str());
    if (in.good())
    {
        for (int i = 0; i < 14; i++)
        {
            in >> result[i];
        }
        in.close();
    }
    else
    {
        for (int i = 0; i < 14; i++)
        {
            result[i] = 0.5;
        }
    }
    return result;
}

vector<float> Options::getFlippoint() const
{
    vector<float> dest;
    if (m_variables.count("flipPoint"))
    {
        dest = (m_variables["flipPoint"].as<vector<float>>());
        if (dest.size() != 3)
        {
            dest.clear();
            dest.push_back(10000000);
            dest.push_back(10000000);
            dest.push_back(10000000);
        }
    }
    else
    {
        dest.push_back(10000000);
        dest.push_back(10000000);
        dest.push_back(10000000);
    }
    return dest;
}

Options::~Options()
{
    // TODO Auto-generated destructor stub
}

} // namespace LargeScaleOptions
