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
 * Segmenter.hpp
 *
 * @date   03.02.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/PMPMesh.hpp"

#include <Cesium3DTiles/Tile.h>

namespace lvr2
{

typedef uint32_t SegmentId;
constexpr SegmentId INVALID_SEGMENT = std::numeric_limits<SegmentId>::max();

struct Segment
{
    SegmentId id = INVALID_SEGMENT;
    size_t num_faces = 0;
    size_t num_vertices = 0;
    pmp::BoundingBox bb;
    std::string filename = "";
};

/**
 * @brief creates a new value at the end of a vector and returns its index and a reference to that value
 *
 * @param vec the vector to append to
 * @return std::pair<size_t, T&> the index and reference to the new value
 */
template<typename T>
inline std::pair<size_t, T&> push_and_get_index(std::vector<T>& vec, T&& value = T())
{
    size_t index = vec.size();
    vec.push_back(std::move(value));
    return std::make_pair(index, std::ref(vec.back()));
}

/**
 * @brief partitions all connected regions of a mesh, and bundles small segments into chunks
 *
 * Adds a face property called "f:segment" to the mesh containing the SegmentId of the face.
 *
 * @param input_mesh the mesh to partition
 * @param small_segments a vector to store the segments in
 * @param large_segments a vector to store extracted segments in
 * @param chunk_size the size of a chunk to determine and bundle small segments
 */
void segment_mesh(pmp::SurfaceMesh& input_mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  std::vector<Segment>& small_segments,
                  std::vector<pmp::SurfaceMesh>& large_segments,
                  std::vector<pmp::BoundingBox>& large_segment_bounds);

void split_mesh(pmp::SurfaceMesh& input_mesh,
                const pmp::BoundingBox& bb,
                float chunk_size,
                Cesium3DTiles::Tile& tile);

} // namespace lvr2
