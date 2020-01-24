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
#include "lvr2/io/ScanIOUtils.hpp"
#include "lvr2/registration/RegistrationPipeline.hpp"
#include "lvr2/reconstruction/LargeScaleReconstruction.hpp"

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
    // rm after new scanIOUtils is ready!
    std::cout << "Finished import!" << std::endl;

    std::cout << "Starting registration..." << std::endl;
    SLAMOptions slamOptions;
    // TODO: set options via config file parser
    slamOptions.icpIterations = 500;
    slamOptions.icpMaxDistance = 10;
    slamOptions.doGraphSLAM = true; // TODO: check why graph slam param leads to bad alloc
    slamOptions.slamMaxDistance = 9;
    slamOptions.loopSize = 10;
    slamOptions.closeLoopDistance = 30;
    RegistrationPipeline registration(&slamOptions, m_scanProject);
    
    std::cout << "Final poses before registration:" << std::endl;
    for (int i = 0; i < m_scanProject->project->positions.size(); i++)
    {
        std::cout << "Pose Nummer " << i << std::endl << m_scanProject->project->positions.at(i)->scan->m_registration << std::endl;
    }
    
    registration.doRegistration();
    std::cout << "Finished registration!" << std::endl;


    std::cout << "Final poses after registration:" << std::endl;
    for (int i = 0; i < m_scanProject->project->positions.size(); i++)
    {
        std::cout << "Pose Nummer " << i << std::endl << m_scanProject->project->positions.at(i)->scan->m_registration << std::endl;
    }

    std::cout << "Starting large scale reconstruction..." << std::endl;
    // TODO: use constructor with struct parameter
    LSROptions lsrOptions;
    lsrOptions.filePath = m_hdf5Path.string();
    LargeScaleReconstruction<lvr2::BaseVector<float>> lsr(lsrOptions);
    BoundingBox<BaseVector<float>> newChunksBB;
    std::string layerName = "tsdf_values";
    lsr.mpiChunkAndReconstruct(m_scanProject, newChunksBB, m_chunkManager, layerName);
    std::cout << "Finished large scale reconstruction!" << std::endl;

    HalfEdgeMesh<BaseVector<float>> newMeshWithOverlap = lsr.getPartialReconstruct(newChunksBB, m_chunkManager, layerName);

    std::cout << "Starting practicability analysis..." << std::endl;
    // TODO: call practicability analysis
    std::cout << "Finished practicability analysis!" << std::endl;

    std::cout << "Finished chunking pipeline!" << std::endl;

    m_running = false;

    return true;
}

} /* namespace lvr2 */
