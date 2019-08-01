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
 * ChunkManager.hpp
 *
 * @date
 * @author
 */

#ifndef CHUNK_MANAGER_HPP
#define CHUNK_MANAGER_HPP

#include "lvr2/io/Model.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/geometry/BaseVector.hpp"

namespace lvr2 {

class ChunkManager
{
    public:
        /**
         * @brief ChunkManager creates chunks from an original model
         *
         * Chunks the original model into chunks of given size.
         * Every created chunk has ste same length in height, width and depth.
         *
         * @param model model to be chunked
         * @param chunksize size of a chunk - unit depends on the given model
         * @param savePath JUST FOR DEBUGGING - REMOVE LATER ON
         */
        ChunkManager(ModelPtr model, float chunksize, std::string savePath);

    private:
        /**
         * @brief initBoundingBox calculates a BoundingBox of the original model
         *
         * This calculates the BoundingBox of the given model and saves it to m_boundingBox.
         */
        void initBoundingBox();

        /**
         * @brief buildChunks builds chunks from an original model
         *
         * Creates chunks from an original model and initializes the initial chunk structure
         *
         * @param chunksize size of a chunk
         * @param savePath UST FOR DEBUGGING - REMOVE LATER ON
         */
        void buildChunks(float chunksize, std::string savePath);

        /**
         * @brief getCenter computes the center of a face in the original model
         *
         * @param index0 index of first vertex in original model
         * @param index1 index of second vertex in original model
         * @param index2 index of third vertex in original model
         * @return center point of the face
         */
        BaseVector<float> getCenter(unsigned int index0, unsigned int index1, unsigned int index2);
        
        // model that is being chunked
        ModelPtr m_originalModel;

        // bounding box of the entire chunked model
        BoundingBox<BaseVector<float>> m_boundingBox;
};

} /* namespace lvr2 */

#endif // CHUNKER_HPP
