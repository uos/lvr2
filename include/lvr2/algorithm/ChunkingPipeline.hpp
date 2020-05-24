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
 * ChunkingPipeline.hpp
 *
 * @date 27.11.2019
 * @author Marcel Wiegand
 */

#ifndef CHUNKING_PIPELINE_HPP
#define CHUNKING_PIPELINE_HPP

#include "lvr2/algorithm/ChunkManager.hpp"

#include <boost/filesystem.hpp>

#include "lvr2/reconstruction/LargeScaleReconstruction.hpp"
#include "lvr2/registration/SLAMOptions.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

template <typename BaseVecT>
class ChunkingPipeline
{
public:
    /**
     * @brief Creates a basic ChunkingPipeline instance
     *
     * @param hdf5Path path to the HDF5 file
     * @param configPath path to the YAML config file
     * @param chunkManager shared pointer to ChunkManager instance if null a new instance is created
     */
    ChunkingPipeline(const boost::filesystem::path& hdf5Path, const boost::filesystem::path& configPath, std::shared_ptr<ChunkManager> chunkManager = nullptr);

    /**
     * @brief Start the chunking pipeline
     *
     * @param scanDir path to scan directory
     *
     * @return true on success and false on failure
     */
    bool start(const boost::filesystem::path& scanDir);

private:
    // path to the HDF5 file
    boost::filesystem::path m_hdf5Path;

    // path to config file
    boost::filesystem::path m_configPath;

    // chunk manger instance
    std::shared_ptr<ChunkManager> m_chunkManager;

    // scan project containing all scans
    ScanProjectEditMarkPtr m_scanProject;

    // registration options
    SLAMOptions m_regOptions;

    // large scale reconstruct options
    LSROptions m_lsrOptions;

    // practicability analysis config values
    double m_roughnessRadius = 0.3;
    double m_heightDifferencesRadius = 0.3;
    std::vector<float> m_practicabilityLayers;

    // status flag
    bool m_running = false;

    /**
     * @brief Parse YAML config (m_configPath)
     */
    void parseYAMLConfig();

    /**
     * @brief Get new scans from scan project
     * 
     * @param dirPath path to scan project directory
     * @return true on success and false on failure
     */
    bool getScanProject(const boost::filesystem::path& dirPath);

    /**
     * @brief Calculates practicability analysis of given mesh and adds it as channels to mesh buffer
     *
     * @param hem HalfEdgeMesh on which practicability analysis is performed
     * @param meshBuffer buffer where practicability analysis channels should be added
     */
    void practicabilityAnalysis(HalfEdgeMesh<BaseVecT>& hem, MeshBufferPtr meshBuffer);
};

} /* namespace lvr2 */

#include "ChunkingPipeline.tcc"
#endif // CHUNKING_PIPELINE_HPP
