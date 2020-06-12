/**
 * Copyright (c) 2019, University Osnabrück
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
 * ChunkingPipeline<BaseVecT>.cpp
 *
 * @date 27.11.2019
 * @author Marcel Wiegand
 */

#include "lvr2/algorithm/ChunkingPipeline.hpp"

#include <yaml-cpp/yaml.h>

#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/config/LSROptionsYamlExtensions.hpp"
#include "lvr2/config/SLAMOptionsYamlExtensions.hpp"
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/registration/RegistrationPipeline.hpp"
#include "lvr2/io/hdf5/ScanProjectIO.hpp"
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"

using BaseHDF5IO = lvr2::Hdf5IO<>;

// Extend IO with features (dependencies are automatically fetched)
using HDF5IO = BaseHDF5IO::AddFeatures<lvr2::hdf5features::ScanProjectIO, lvr2::hdf5features::ScanPositionIO>;

namespace lvr2
{
template <typename BaseVecT>
ChunkingPipeline<BaseVecT>::ChunkingPipeline(
        const boost::filesystem::path& hdf5Path,
        const boost::filesystem::path& configPath,
        std::shared_ptr<ChunkManager> chunkManager
        ) :  m_hdf5Path(hdf5Path), m_configPath(configPath)
{
    if (chunkManager != nullptr)
    {
        m_chunkManager = chunkManager;
    }
    else
    {
        m_chunkManager = std::make_shared<ChunkManager>(m_hdf5Path.string());
    }

    parseYAMLConfig();
}

template <typename BaseVecT>
void ChunkingPipeline<BaseVecT>::parseYAMLConfig()
{
    if (boost::filesystem::exists(m_configPath) && boost::filesystem::is_regular_file(m_configPath))
    {
        YAML::Node config = YAML::LoadFile(m_configPath.string());

        if (config["lvr2_registration"])
        {
            std::cout << timestamp << "Found config entry for lvr2_registration." << std::endl;
            m_regOptions = config["lvr2_registration"].as<SLAMOptions>();
        }

        if (config["lvr2_largescale_reconstruct"])
        {
            std::cout << timestamp << "Found config entry for lvr2_largescale_reconstruct." << std::endl;
            m_lsrOptions = config["lvr2_largescale_reconstruct"].as<LSROptions>();
        }

        if (config["lvr2_practicability_analysis"] && config["lvr2_practicability_analysis"].IsMap())
        {
            std::cout << timestamp << "Found config entry for lvr2_practicability_analysis." << std::endl;
            YAML::Node practicabilityConfig = config["lvr2_practicability_analysis"];
            if (practicabilityConfig["roughnessRadius"])
            {
                m_roughnessRadius = practicabilityConfig["roughnessRadius"].as<double>();
            }
            if (practicabilityConfig["heightDifferencesRadius"])
            {
                m_heightDifferencesRadius = practicabilityConfig["heightDifferencesRadius"].as<double>();
            }
            if (practicabilityConfig["layers"] && practicabilityConfig["layers"].IsSequence())
            {
                m_practicabilityLayers = practicabilityConfig["layers"].as<std::vector<float>>();
            }
        }
    }
    else
    {
        std::cout << timestamp << "Config file does not exist or is not a regular file!" << std::endl;
    }
}

template <typename BaseVecT>
void ChunkingPipeline<BaseVecT>::practicabilityAnalysis(HalfEdgeMesh<BaseVecT>& hem, MeshBufferPtr meshBuffer)
{
    // Calc face normals
    DenseFaceMap <Normal<float>> faceNormals = calcFaceNormals(hem);
    // Calc vertex normals
    DenseVertexMap <Normal<float>> vertexNormals = calcVertexNormals(hem, faceNormals);
    // Calc average vertex angles
    DenseVertexMap<float> averageAngles = calcAverageVertexAngles(hem, vertexNormals);
    // Calc roughness
    DenseVertexMap<float> roughness = calcVertexRoughness(hem, m_roughnessRadius, vertexNormals);
    // Calc vertex height differences
    DenseVertexMap<float> heightDifferences = calcVertexHeightDifferences(hem, m_heightDifferencesRadius);

    // create and fill channels
    FloatChannel faceNormalChannel(faceNormals.numValues(), channel_type < Normal < float >> ::w);
    Index i = 0;
    for (auto handle : FaceIteratorProxy<BaseVecT>(hem)) {
        faceNormalChannel[i++] = faceNormals[handle]; //TODO handle deleted map values.
    }

    FloatChannel vertexNormalsChannel(vertexNormals.numValues(), channel_type<Normal<float>>::w);
    FloatChannel averageAnglesChannel(averageAngles.numValues(), channel_type<float>::w);
    FloatChannel roughnessChannel(roughness.numValues(), channel_type<float>::w);
    FloatChannel heightDifferencesChannel(heightDifferences.numValues(), channel_type<float>::w);

    Index j = 0;
    for (auto handle : VertexIteratorProxy<BaseVecT>(hem))
    {
        vertexNormalsChannel[j] = vertexNormals[handle]; //TODO handle deleted map values.
        averageAnglesChannel[j] = averageAngles[handle]; //TODO handle deleted map values.
        roughnessChannel[j] = roughness[handle]; //TODO handle deleted map values.
        heightDifferencesChannel[j] = heightDifferences[handle]; //TODO handle deleted map values.
        j++;
    }

    // add channels to mesh buffer
    meshBuffer->add("face_normals", faceNormalChannel);
    meshBuffer->add("vertex_normals", vertexNormalsChannel);
    meshBuffer->add("average_angles", averageAnglesChannel);
    meshBuffer->add("roughness", roughnessChannel);
    meshBuffer->add("height_diff", heightDifferencesChannel);
}

template <typename BaseVecT>
bool ChunkingPipeline<BaseVecT>::getScanProject(const boost::filesystem::path& dirPath)
{
    HDF5IO hdf;
    hdf.open(m_hdf5Path.string());

    // load scans from hdf5
    ScanProjectPtr scanProjectPtr = hdf.loadScanProject();

    // load scans from directory
    ScanProject dirScanProject;
    bool importStatus = loadScanProject(dirPath, dirScanProject);

    ScanProjectEditMark tmpScanProject;
    std::vector<bool> init(scanProjectPtr->positions.size(), false);
    tmpScanProject.changed = init;

    if (!importStatus)
    {
        return false;
    }
    else
    {
        std::cout << timestamp << "Found " << dirScanProject.positions.size() - scanProjectPtr->positions.size() << " new scanPosition(s)" << std::endl;
        for (int i = scanProjectPtr->positions.size(); i < dirScanProject.positions.size(); i++)
        {
            scanProjectPtr->positions.push_back(dirScanProject.positions[i]);
            tmpScanProject.changed.push_back(true);
        }
    }

    tmpScanProject.project = scanProjectPtr;
    m_scanProject = std::make_shared<ScanProjectEditMark>(tmpScanProject);
    m_scanProject->changed.resize(scanProjectPtr->positions.size());

    return true;
}

template <typename BaseVecT>
bool ChunkingPipeline<BaseVecT>::start(const boost::filesystem::path& scanDir)
{
    if (m_running)
    {
        std::cout << "Chunking Pipeline is already running!" << std::endl;
        return false;
    }

    m_running = true;

    std::cout << timestamp << "Starting chunking pipeline..." << std::endl;

    std::cout << timestamp << "Starting import tool..." << std::endl;

    if (!getScanProject(scanDir))
    {
        std::cout << "Import failed..." << std::endl;
        std::cout << "Aborting chunking pipeline!" << std::endl;

        m_running = false;
        return false;
    }

    std::cout << timestamp << "Finished import!" << std::endl;

    std::cout << timestamp << "Starting registration..." << std::endl;
    RegistrationPipeline registration(&m_regOptions, m_scanProject);
    registration.doRegistration();
    std::cout << timestamp << "Finished registration!" << std::endl;

    // save raw data
    HDF5IO hdf;
    hdf.open(m_hdf5Path.string());

    for (size_t idx = 0; idx < m_scanProject->changed.size(); idx++)
    {
        if (m_scanProject->changed[idx])
        {
            // only save changed scanPositions
            hdf.save(idx, m_scanProject->project->positions[idx]);
        }
    }

    // remove hyperspectral data from memory
    for (ScanPositionPtr pos : m_scanProject->project->positions)
    {
        pos->hyperspectralCamera.reset(new HyperspectralCamera);
    }

    std::cout << timestamp << "Starting large scale reconstruction..." << std::endl;
    LargeScaleReconstruction<BaseVecT> lsr(m_lsrOptions);
    BoundingBox<BaseVecT> newChunksBB;
    lsr.mpiChunkAndReconstruct(m_scanProject, newChunksBB, m_chunkManager);
    std::cout << timestamp << "Finished large scale reconstruction!" << std::endl;

    for (auto layer : m_lsrOptions.voxelSizes)
    {
        std::string voxelSizeStr = "[Layer " + std::to_string(layer) + "] ";
        std::cout << timestamp << voxelSizeStr << "Starting mesh generation..." << std::endl;
        HalfEdgeMesh<BaseVecT> hem = lsr.getPartialReconstruct(
                newChunksBB,
                m_chunkManager,
                layer);
        std::cout << timestamp << voxelSizeStr  << "Finished mesh generation!" << std::endl;

        std::cout << timestamp << voxelSizeStr  << "Starting mesh buffer creation..." << std::endl;
        lvr2::SimpleFinalizer<BaseVecT> finalize;
        MeshBufferPtr meshBuffer = MeshBufferPtr(finalize.apply(hem));
        std::cout << timestamp << voxelSizeStr  << "Finished mesh buffer creation!" << std::endl;

        auto foundIt = std::find(m_practicabilityLayers.begin(), m_practicabilityLayers.end(), layer);
        if (foundIt != m_practicabilityLayers.end())
        {
            std::cout << timestamp << voxelSizeStr  << "Starting practicability analysis..." << std::endl;
            practicabilityAnalysis(hem, meshBuffer);
            std::cout << timestamp << voxelSizeStr  << "Finished practicability analysis!" << std::endl;
        }
        else
        {
            std::cout << timestamp << voxelSizeStr  << "Skipping practicability analysis..." << std::endl;
        }

        std::cout << timestamp << voxelSizeStr  << "Starting chunking and saving of mesh buffer..." << std::endl;
        // TODO: get maxChunkOverlap size
        // TODO: savePath is not used in buildChunks (remove it?)
        m_chunkManager->buildChunks(meshBuffer, 0.1f, "", "mesh_" + std::to_string(layer));
        std::cout << timestamp << voxelSizeStr  << "Finished chunking and saving of mesh buffer!" << std::endl;
    }

    std::cout << timestamp << "Finished chunking pipeline!" << std::endl;

    m_running = false;

    return true;
}

} /* namespace lvr2 */
