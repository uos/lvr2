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
 * Segmenter.cpp
 *
 * @date   03.02.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "Segmenter.hpp"

namespace lvr2
{

void segment_mesh(PMPMesh<BaseVector<float>>& input_mesh,
                  std::vector<Segment>& out_segments)
{
    out_segments.clear();

    auto& mesh = input_mesh.getSurfaceMesh();

    auto f_prop = mesh.face_property<uint32_t>("f:segment", INVALID_SEGMENT);
    auto v_prop = mesh.vertex_property<uint32_t>("v:segment", INVALID_SEGMENT);

    std::vector<VertexHandle> queue;

    for (auto vH : mesh.vertices())
    {
        if (v_prop[vH] != INVALID_SEGMENT)
        {
            continue;
        }

        // start new segment
        Segment segment;
        segment.id = out_segments.size();

        v_prop[vH] = segment.id;
        queue.push_back(vH);

        while (!queue.empty())
        {
            VertexHandle vH = queue.back();
            queue.pop_back();
            segment.num_vertices++;
            segment.bb += mesh.position(vH);

            for (auto heH : mesh.halfedges(vH))
            {
                VertexHandle ovH = mesh.to_vertex(heH);
                if (v_prop[ovH] == INVALID_SEGMENT)
                {
                    v_prop[ovH] = segment.id;
                    queue.push_back(ovH);
                }

                FaceHandle fH = mesh.face(heH);
                if (fH.is_valid() && f_prop[fH] == INVALID_SEGMENT)
                {
                    f_prop[fH] = segment.id;
                    segment.num_faces++;
                }
            }
        }

        out_segments.push_back(segment);
    }

    // consistency check
    size_t total_faces = 0;
    size_t total_vertices = 0;
    for (auto& segment : out_segments)
    {
        total_faces += segment.num_faces;
        total_vertices += segment.num_vertices;
    }
    if (total_faces != mesh.n_faces())
    {
        throw std::runtime_error(std::string("Segmenter: inconsistent number of faces: ") +
                                 std::to_string(total_faces) + " != " + std::to_string(mesh.n_faces()));
    }
    if (total_vertices != mesh.n_vertices())
    {
        throw std::runtime_error(std::string("Segmenter: inconsistent number of vertices: ") +
                                 std::to_string(total_vertices) + " != " + std::to_string(mesh.n_vertices()));
    }
}

} // namespace lvr2
