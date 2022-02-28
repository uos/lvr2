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
 * B3dmWriter.hpp
 *
 * @date   24.01.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/PMPMesh.hpp"
#include "Segmenter.hpp"

#include <boost/filesystem.hpp>

namespace lvr2
{

/**
 * @brief converts mesh to b3dm format and writes it to a file
 *
 * @param output_dir the directory that filenames are relative to
 * @param mesh the mesh to convert
 * @param segments the metadata of the segments
 */
void write_b3dm_segments(const boost::filesystem::path& output_dir,
                         const pmp::SurfaceMesh& mesh,
                         const std::vector<Segment>& segments,
                         bool print_progress = true);

/**
 * @brief converts mesh to b3dm format and writes it to a file
 *
 * @param output_dir the directory that filename is relative to
 * @param filename the name of the file to write to
 * @param mesh the mesh to convert
 * @param bb the bounding box of the mesh
 */
inline void write_b3dm(const boost::filesystem::path& output_dir,
                       const std::string& filename,
                       const pmp::SurfaceMesh& mesh,
                       const pmp::BoundingBox& bb,
                       bool print_progress = true)
{
    std::vector<Segment> segments;
    auto& segment = segments.emplace_back();
    segment.id = 0;
    segment.num_faces = mesh.n_faces();
    segment.num_vertices = mesh.n_vertices();
    segment.bb = bb;
    segment.filename = filename;

    write_b3dm_segments(output_dir, mesh, segments, print_progress);
}

} // namespace lvr2
