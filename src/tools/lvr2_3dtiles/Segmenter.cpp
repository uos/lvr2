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

/**
 * @brief Calculates a 1D Chunk-index from a 3D position
 * 
 * @param p the 3D position
 * @param chunk_size the size of a chunk
 * @param num_chunks the number of chunks along each axis
 * @return uint64_t the 1D Chunk-index
 */
uint64_t chunk_index(pmp::Point p, float chunk_size, Eigen::Vector3i num_chunks)
{
    return std::floor(p.x() / chunk_size)
           + std::floor(p.y() / chunk_size) * num_chunks.x()
           + std::floor(p.z() / chunk_size) * num_chunks.x() * num_chunks.y();
}

/**
 * @brief Splits large segments into chunks
 * 
 * @param mesh the input mesh
 * @param segments a list of all existing segments
 * @param to_be_split to_be_split[segment.id] == true <=> segment should be split
 * @param out_segments the resulting new segments
 * @param segment_map maps old segment ids to new segment ids
 * @param chunk_size the size of a chunk
 */
void partition_large_segments(pmp::SurfaceMesh& mesh,
                              const std::vector<Segment>& segments,
                              const std::vector<bool>& to_be_split,
                              std::vector<Segment>& out_segments,
                              std::vector<SegmentId>& segment_map,
                              float chunk_size);

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
        segment.start_face = fH;

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
    // each face should be assigned to exactly one segment
    // vertices might be shared between segments or isolated, so we don't count them
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
    pmp::Point size = (bb.max() - bb.min()) / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    std::vector<SegmentId> segment_map(segments.size(), INVALID_SEGMENT);
    std::vector<bool> to_be_split(segments.size(), false);
    size_t to_be_split_count = 0;

    out_segments.clear();

    for (auto& segment : segments)
    {
        // Segments that are larger than a chunk are split in the next step
        if (segment.bb.longest_axis_size() >= chunk_size)
        {
            to_be_split[segment.id] = true;
            to_be_split_count++;
            continue;
        }

        // all other segments are merged based on the chunk that their center lies in
        uint64_t chunk_id = chunk_index(segment.bb.center() - bb.min(), chunk_size, num_chunks);
        auto elem = chunk_map.find(chunk_id);
        if (elem == chunk_map.end())
        {
            // start a new chunk with this segment
            SegmentId new_id = out_segments.size();
            segment_map[segment.id] = new_id;
            chunk_map[chunk_id] = new_id;
            segment.id = new_id;
            out_segments.push_back(segment);
        }
        else
        {
            // merge this segment with the existing chunk
            SegmentId new_id = elem->second;
            segment_map[segment.id] = new_id;

            Segment& target = out_segments[new_id];
            target.num_faces += segment.num_faces;
            target.num_vertices += segment.num_vertices;
            target.bb += segment.bb;
        }
    }

    std::cout << timestamp << "Merged " << (segments.size() - to_be_split_count) << " small segments into "
              << out_segments.size() << " chunks" << std::endl;

    if (to_be_split_count > 0)
    {
        size_t old_size = out_segments.size();

        partition_large_segments(mesh, segments, to_be_split, out_segments, segment_map, chunk_size);

        std::cout << timestamp << "Partitioned " << to_be_split_count << " large segments into "
                  << (out_segments.size() - old_size) << " chunks" << std::endl;
    }


    // consistency check
    total_faces = 0;
    for (const auto& segment : out_segments)
    {
        total_faces += segment.num_faces;
    }
    if (total_faces != mesh.n_faces())
    {
        throw consistency_error(mesh.n_faces(), total_faces, "faces");
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (mesh.is_valid(fH) && !mesh.is_deleted(fH))
        {
            f_prop[fH] = segment_map[f_prop[fH]];
        }
    }

    mesh.remove_vertex_property(v_prop);
}

void partition_large_segments(pmp::SurfaceMesh& mesh,
                              const std::vector<Segment>& segments,
                              const std::vector<bool>& to_be_split,
                              std::vector<Segment>& out_segments,
                              std::vector<SegmentId>& segment_map,
                              float chunk_size)
{
    auto f_prop = mesh.get_face_property<SegmentId>("f:segment");
    auto v_prop = mesh.get_vertex_property<SegmentId>("v:segment");

    std::vector<pmp::Point> offsets(segments.size());
    std::vector<Eigen::Vector3i> num_chunks(segments.size());
    size_t faces_in_split_segments = 0;

    for (auto& segment : segments)
    {
        if (!to_be_split[segment.id])
        {
            continue;
        }

        pmp::Point size_of_segment = segment.bb.max() - segment.bb.min();
        pmp::Point size = size_of_segment / chunk_size;
        Eigen::Vector3i num(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

        // align the chunking to the center of the segment:
        // modify the offset so that there is equal overlap on all sides
        pmp::Point size_of_chunks = num.cast<float>() * chunk_size;
        pmp::Point offset = segment.bb.min() - (size_of_chunks - size_of_segment) / 2.0f;

        offsets[segment.id] = offset;
        num_chunks[segment.id] = num;
        faces_in_split_segments += segment.num_faces;
    }

    auto f_chunk_id = mesh.add_face_property<uint64_t>("f:chunk_id");
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (!mesh.is_valid(fH) || mesh.is_deleted(fH) || !to_be_split[f_prop[fH]])
        {
            continue;
        }
        pmp::Point pos(0, 0, 0);
        size_t count = 0;
        for (auto vH : mesh.vertices(fH))
        {
            pos += mesh.position(vH);
            count++;
        }
        assert(count == 3);
        pos /= count;

        SegmentId segment_id = f_prop[fH];
        f_chunk_id[fH] = chunk_index(pos - offsets[segment_id], chunk_size, num_chunks[segment_id]);
    }

    ProgressBar progress(faces_in_split_segments, "Splitting large segments");

    std::vector<std::unordered_map<uint64_t, SegmentId>> chunk_maps(segments.size());

    for (auto fH : mesh.faces())
    {
        if (!to_be_split[f_prop[fH]])
        {
            continue;
        }

        SegmentId in_id, out_id;
        auto& chunk_map = chunk_maps[f_prop[fH]];
        uint64_t chunk_id = f_chunk_id[fH];
        auto elem = chunk_map.find(chunk_id);
        if (elem == chunk_map.end())
        {
            // all faces in this segment will first get the fake id in_id to avoid overlap with existing segments
            // the real id will be assigned in the segment mapping step
            in_id = segment_map.size();
            out_id = out_segments.size();
            segment_map.push_back(out_id);
            auto& new_segment = out_segments.emplace_back();
            new_segment.id = out_id;
            new_segment.start_face = fH;

            chunk_map[chunk_id] = in_id;
        }
        else
        {
            in_id = elem->second;
            out_id = segment_map[in_id];
        }

        f_prop[fH] = in_id;

        auto& segment = out_segments[out_id];
        segment.num_faces++;

        for (auto vH : mesh.vertices(fH))
        {
            if (v_prop[vH] != in_id)
            {
                v_prop[vH] = in_id;
                segment.num_vertices++;
                segment.bb += mesh.position(vH);
            }
        }

        ++progress;
    }
    std::cout << "\r";

    mesh.remove_face_property(f_chunk_id);
}

} // namespace lvr2
