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
    // TODO: call input tool
    // TODO: Fill m_scanProjectPtr with scans
    std::cout << "Finished import!" << std::endl;

    std::cout << "Starting registration..." << std::endl;
    SLAMOptions slamOptions; // TODO: set options
    RegistrationPipeline registration(&slamOptions, m_scanProject);
    // TODO: call registration
    // registration.doRegistration();
    std::cout << "Finished registration!" << std::endl;

    std::cout << "Starting large scale reconstruction..." << std::endl;
    // TODO: use constructor with struct parameter
    LargeScaleReconstruction<lvr2::BaseVector<float>> lsr(m_hdf5Path.string());
    // TODO: call reconstruction
    // lsr.mpiChunkAndReconstruct(m_scanProject, m_chunkManager);
    std::cout << "Finished large scale reconstruction!" << std::endl;

    std::cout << "Starting practicability analysis..." << std::endl;
    // TODO: call practicability analysis
    std::cout << "Finished practicability analysis!" << std::endl;

    std::cout << "Finished chunking pipeline!" << std::endl;

    m_running = false;

    return true;
}

} /* namespace lvr2 */
