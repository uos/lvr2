/**
 * Copyright (c) 2022, University Osnabrück
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
 * Tiles3dIO.hpp
 *
 * @date   01.07.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#ifdef LVR2_USE_3DTILES

#include "lvr2/algorithm/HLODTree.hpp"

#include <Cesium3DTiles/Tile.h>

namespace lvr2
{

template<typename BaseVecT>
class Tiles3dIO
{
public:
    using TreePtr = typename HLODTree<BaseVecT>::Ptr;
    using TreeConstPtr = const typename HLODTree<BaseVecT>::Ptr;

    /**
     * @brief Construct a new Tiles3dIO object
     * 
     * @param directory The directory where the tiles are stored
     */
    Tiles3dIO(const std::string& directory);
    ~Tiles3dIO() = default;

    /**
     * @brief Writes the given tree to the directory
     * 
     * @param tree The tree to write
     * @param compress if true: compress meshes with draco compression
     * @param scale scale factor for the meshes
     */
    void write(TreeConstPtr& tree, bool compress = false, float scale = 1.0f);
    void read(TreePtr& tree)
    {
        throw std::runtime_error("Not implemented yet");
    }

private:
    void writeTiles(Cesium3DTiles::Tile& tile,
                    TreeConstPtr& tree,
                    bool compress,
                    const std::string& outputDir,
                    const std::string& prefix = "");

    std::string m_rootDir;
};

} // namespace lvr2

#include "Tiles3dIO.tcc"

#endif // LVR2_USE_3DTILES
