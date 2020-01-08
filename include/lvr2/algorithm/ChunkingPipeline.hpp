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

namespace lvr2
{

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
    ChunkingPipeline(std::string hdf5Path, std::string configPath, std::shared_ptr<ChunkManager> chunkManager = nullptr);

private:
    // path to the HDF5 file
    std::string m_hdf5Path;

    // path to config file
    std::string m_configPath;

    // chunk manger instance
    std::shared_ptr<ChunkManager> m_chunkManager;

    // TODO: add old scans, e.g. std::vector<Scan> m_oldScans

    // TODO: add new scans, e.g. std::vector<Scan> m_newScans

public:
    /**
     * @brief Start the chunking pipeline
     *
     * @return true on success and false on failure
     */
    bool start();

    /**
     * TODO
     * @return
     */
    bool stop();
};

} /* namespace lvr2 */

#endif // CHUNKING_PIPELINE_HPP
