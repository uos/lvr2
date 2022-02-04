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

    std::vector<FaceHandle> queue;

    for (auto fH : mesh.faces())
    {
        if (f_prop[fH] != INVALID_SEGMENT)
        {
            continue;
        }

        // start new segment
        Segment segment;
        segment.id = out_segments.size();

        f_prop[fH] = segment.id;
        segment.num_faces++;
        queue.push_back(fH);

        while (!queue.empty())
        {
            FaceHandle fH = queue.back();
            queue.pop_back();

            for (auto heH : mesh.halfedges(fH))
            {
                segment.bb += mesh.position(mesh.to_vertex(heH));

                FaceHandle ofH = mesh.face(mesh.opposite_halfedge(heH));
                if (ofH.is_valid() && f_prop[ofH] != segment.id)
                {
                    f_prop[ofH] = segment.id;
                    segment.num_faces++;
                    queue.push_back(ofH);
                }
            }
        }

        out_segments.push_back(segment);
    }

    // consistency check
    size_t total_faces = 0;
    for (auto& segment : out_segments)
    {
        total_faces += segment.num_faces;
    }
    if (total_faces != mesh.n_faces())
    {
        throw std::runtime_error(std::string("Segmenter: inconsistent number of faces: ") +
                                 std::to_string(total_faces) + " != " + std::to_string(mesh.n_faces()));
    }
}

} // namespace lvr2
