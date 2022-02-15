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
#include "lvr2/util/Progress.hpp"

namespace lvr2
{

struct consistency_error : public std::runtime_error
{
    consistency_error(size_t expected, size_t found, const char* name)
        : std::runtime_error(std::string("Segmenter: inconsistent number of ") + name + ": "
                             + "expected " + std::to_string(expected)
                             + ", found " + std::to_string(found))
    {}
};

void segment_mesh(pmp::SurfaceMesh& mesh,
                  std::vector<Segment>& out_segments,
                  const pmp::BoundingBox& bb,
                  float chunk_size)
{
    auto f_prop = mesh.add_face_property<SegmentId>("f:segment", INVALID_SEGMENT);
    auto v_prop = mesh.add_vertex_property<SegmentId>("v:segment", INVALID_SEGMENT);

    std::vector<Segment> segments;

    std::vector<FaceHandle> queue;
    ProgressBar progress(mesh.n_faces(), "Segmenting mesh");

    for (FaceHandle fH : mesh.faces())
    {
        if (f_prop[fH] != INVALID_SEGMENT)
        {
            continue;
        }

        Segment segment;
        segment.id = segments.size();

        f_prop[fH] = segment.id;
        queue.push_back(fH);

        while (!queue.empty())
        {
            FaceHandle fH = queue.back();
            queue.pop_back();
            segment.num_faces++;
            ++progress;

            for (auto heH : mesh.halfedges(fH))
            {
                auto vH = mesh.to_vertex(heH);
                if (v_prop[vH] != segment.id)
                {
                    v_prop[vH] = segment.id;
                    segment.bb += mesh.position(vH);
                    segment.num_vertices++;
                }

                FaceHandle ofH = mesh.face(mesh.opposite_halfedge(heH));
                if (!ofH.is_valid())
                {
                    continue;
                }
                SegmentId& oid = f_prop[ofH];
                if (oid != segment.id)
                {
                    if (oid != INVALID_SEGMENT)
                    {
                        throw std::runtime_error("Segmenter: inconsistent neighborhood of faces");
                    }
                    oid = segment.id;
                    queue.push_back(ofH);
                }
            }
        }

        segments.push_back(segment);
    }
    std::cout << "\r" << timestamp << "Found " << segments.size() << " initial segments" << std::endl;

    // consistency check
    size_t total_faces = 0;
    for (auto& segment : segments)
    {
        total_faces += segment.num_faces;
    }
    if (total_faces != mesh.n_faces())
    {
        throw consistency_error(mesh.n_faces(), total_faces, "faces");
    }

    // ==================== merge small segments within a chunk together ====================

    std::unordered_map<uint64_t, SegmentId> chunk_map;
    pmp::Point size = bb.max() - bb.min();
    uint64_t num_chunks_x = std::ceil(size.x() / chunk_size);
    uint64_t num_chunks_y = std::ceil(size.y() / chunk_size);

    std::vector<SegmentId> segment_map(segments.size(), INVALID_SEGMENT);

    out_segments.clear();

    for (auto& segment : segments)
    {
        if (segment.bb.longest_axis_size() >= chunk_size)
        {
            SegmentId new_id = out_segments.size();
            segment_map[segment.id] = new_id;
            segment.id = new_id;
            out_segments.push_back(segment);
            continue;
        }

        pmp::Point pos = segment.bb.center();
        pmp::Point chunk_index = (pos - bb.min()) / chunk_size;
        uint64_t chunk_id = std::floor(chunk_index.x())
                            + std::floor(chunk_index.y()) * num_chunks_x
                            + std::floor(chunk_index.z()) * num_chunks_x * num_chunks_y;
        auto elem = chunk_map.find(chunk_id);
        if (elem == chunk_map.end())
        {
            SegmentId new_id = out_segments.size();
            segment_map[segment.id] = new_id;
            chunk_map[chunk_id] = new_id;
            segment.id = new_id;
            out_segments.push_back(segment);
        }
        else
        {
            SegmentId new_id = elem->second;
            segment_map[segment.id] = new_id;

            Segment& target = out_segments[new_id];
            target.num_faces += segment.num_faces;
            target.num_vertices += segment.num_vertices;
            target.bb += segment.bb;
        }
    }

    std::cout << timestamp << "Reduced to " << out_segments.size() << " segments" << std::endl;

    // consistency check
    total_faces = 0;
    for (auto& segment : out_segments)
    {
        total_faces += segment.num_faces;
    }
    if (total_faces != mesh.n_faces())
    {
        throw consistency_error(mesh.n_faces(), total_faces, "faces");
    }

    for (auto fH : mesh.faces())
    {
        f_prop[fH] = segment_map[f_prop[fH]];
    }

    mesh.remove_vertex_property(v_prop);
}

} // namespace lvr2
