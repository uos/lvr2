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
 * ChunkingPipeline.cpp
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

namespace lvr2
{
ChunkingPipeline::ChunkingPipeline(
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

void ChunkingPipeline::parseYAMLConfig()
{
    if (boost::filesystem::exists(m_configPath) && boost::filesystem::is_regular_file(m_configPath))
    {
        YAML::Node config = YAML::LoadFile(m_configPath.string());

        if (config["lvr2_registration"])
        {
            std::cout << "Found config entry for lvr2_registration." << std::endl;
            m_regOptions = config["lvr2_registration"].as<SLAMOptions>();
        }

        if (config["lvr2_largescale_reconstruct"])
        {
            std::cout << "Found config entry for lvr2_largescale_reconstruct." << std::endl;
            m_lsrOptions = config["lvr2_largescale_reconstruct"].as<LSROptions>();
        }

        if (config["lvr2_practicability_analysis"] && config["lvr2_practicability_analysis"].IsMap())
        {
            std::cout << "Found config entry for lvr2_practicability_analysis." << std::endl;
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
        std::cout << "Config file does not exist or is not a regular file!" << std::endl;
    }
}

void ChunkingPipeline::practicabilityAnalysis(HalfEdgeMesh<lvr2::BaseVector<float>>& hem, MeshBufferPtr meshBuffer)
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
    for (auto handle : FaceIteratorProxy<lvr2::BaseVector<float>>(hem)) {
        faceNormalChannel[i++] = faceNormals[handle]; //TODO handle deleted map values.
    }

    FloatChannel vertexNormalsChannel(vertexNormals.numValues(), channel_type<Normal<float>>::w);
    FloatChannel averageAnglesChannel(averageAngles.numValues(), channel_type<float>::w);
    FloatChannel roughnessChannel(roughness.numValues(), channel_type<float>::w);
    FloatChannel heightDifferencesChannel(heightDifferences.numValues(), channel_type<float>::w);

    Index j = 0;
    for (auto handle : VertexIteratorProxy<lvr2::BaseVector<float>>(hem))
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

bool ChunkingPipeline::start(const boost::filesystem::path& scanDir)
{
    if (m_running)
    {
        std::cout << "Chunking Pipeline is already running!" << std::endl;
        return false;
    }

    m_running = true;

    std::cout << "Starting chunking pipeline..." << std::endl;

    std::cout << "Starting import tool..." << std::endl;
    ScanProject scanProject;
    // tmp disabled until new scanIOUtils is ready!
//    bool importStatus = loadScanProjectFromDirectory(scanDir, scanProject);
//    if (!importStatus)
//    {
//        std::cout << "Import failed..." << std::endl;
//        std::cout << "Aborting chunking pipeline!" << std::endl;
//
//        m_running = false;
//
//        return false;
//    }
//    else
//    {
//        ScanProjectEditMark tmpScanProject;
//        tmpScanProject.project = std::make_shared<ScanProject>(scanProject);
//        m_scanProject = std::make_shared<ScanProjectEditMark>(tmpScanProject);
//    }
    // rm after new scanIOUtils is ready!
    loadAllPreviewsFromHDF5(m_hdf5Path.string(), scanProject);
    ScanProjectEditMark tmpScanProject;
    tmpScanProject.project = std::make_shared<ScanProject>(scanProject);
    m_scanProject = std::make_shared<ScanProjectEditMark>(tmpScanProject);
    // set all scans to true, so they are getting reconstructed
    m_scanProject->changed.resize(scanProject.positions.size());
    // rm after new scanIOUtils is ready!
    std::cout << "Finished import!" << std::endl;

    std::cout << "Starting registration..." << std::endl;
    RegistrationPipeline registration(&m_regOptions, m_scanProject);
    registration.doRegistration();
    std::cout << "Finished registration!" << std::endl;

    std::cout << "Starting large scale reconstruction..." << std::endl;
    LargeScaleReconstruction<lvr2::BaseVector<float>> lsr(m_lsrOptions);
    BoundingBox<BaseVector<float>> newChunksBB;
    lsr.mpiChunkAndReconstruct(m_scanProject, newChunksBB, m_chunkManager);
    std::cout << "Finished large scale reconstruction!" << std::endl;

    for (auto layer : m_lsrOptions.voxelSizes)
    {
        std::string voxelSizeStr = "[Layer " + std::to_string(layer) + "] ";
        std::cout << voxelSizeStr << "Starting mesh generation..." << std::endl;
        HalfEdgeMesh<lvr2::BaseVector<float>> hem = lsr.getPartialReconstruct(
                newChunksBB,
                m_chunkManager,
                layer);
        std::cout << voxelSizeStr  << "Finished mesh generation!" << std::endl;

        std::cout << voxelSizeStr  << "Starting mesh buffer creation..." << std::endl;
        lvr2::SimpleFinalizer<lvr2::BaseVector<float>> finalize;
        MeshBufferPtr meshBuffer = MeshBufferPtr(finalize.apply(hem));
        std::cout << voxelSizeStr  << "Finished mesh buffer creation!" << std::endl;

        auto foundIt = std::find(m_practicabilityLayers.begin(), m_practicabilityLayers.end(), layer);
        if (foundIt != m_practicabilityLayers.end())
        {
            std::cout << voxelSizeStr  << "Starting practicability analysis..." << std::endl;
            practicabilityAnalysis(hem, meshBuffer);
            std::cout << voxelSizeStr  << "Finished practicability analysis!" << std::endl;
        }
        else
        {
            std::cout << voxelSizeStr  << "Skipping practicability analysis..." << std::endl;
        }

        std::cout << voxelSizeStr  << "Starting chunking and saving of mesh buffer..." << std::endl;
        // TODO: get maxChunkOverlap size
        // TODO: savePath is not used in buildChunks (remove it?)
        m_chunkManager->buildChunks(meshBuffer, 0.1f, "", "mesh_" + std::to_string(layer));
        std::cout << voxelSizeStr  << "Finished chunking and saving of mesh buffer!" << std::endl;
    }

    std::cout << "Finished chunking pipeline!" << std::endl;

    m_running = false;

    return true;
}

} /* namespace lvr2 */
