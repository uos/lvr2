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

void distribute_segments(pmp::SurfaceMesh& mesh,
                         std::vector<Segment>& segments,
                         std::vector<SegmentId>& segment_map,
                         std::vector<bool> to_be_split,
                         size_t to_be_split_count,
                         std::vector<pmp::SurfaceMesh>& out_meshes,
                         std::vector<pmp::BoundingBox>& out_bounds,
                         std::vector<Segment>& small_segments);

void segment_mesh(pmp::SurfaceMesh& mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  std::vector<Segment>& small_segments,
                  std::vector<pmp::SurfaceMesh>& large_segments,
                  std::vector<pmp::BoundingBox>& large_segment_bounds)
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
    mesh.remove_vertex_property(v_prop);
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

    small_segments.clear();

    for (auto& segment : segments)
    {
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
            auto [ new_id, new_segment ] = push_and_get_index(small_segments, std::move(segment));
            new_segment.id = new_id;
            segment_map[segment.id] = new_id;
            chunk_map[chunk_id] = new_id;
        }
        else
        {
            // merge this segment with the existing chunk
            SegmentId new_id = elem->second;
            segment_map[segment.id] = new_id;

            Segment& target = small_segments[new_id];
            target.num_faces += segment.num_faces;
            target.num_vertices += segment.num_vertices;
            target.bb += segment.bb;
        }
    }

    std::cout << timestamp << "Merged " << (segments.size() - to_be_split_count) << " small segments into "
              << small_segments.size() << " chunks" << std::endl;

    if (to_be_split_count == 0)
    {
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            FaceHandle fH(i);
            if (!mesh.is_deleted(fH))
            {
                f_prop[fH] = segment_map[f_prop[fH]];
            }
        }
    }
    else
    {
        distribute_segments(mesh, segments, segment_map, to_be_split, to_be_split_count, large_segments, large_segment_bounds, small_segments);
    }

    // consistency check
    total_faces = 0;
    for (const auto& segment : small_segments)
    {
        total_faces += segment.num_faces;
    }
    if (total_faces != mesh.n_faces())
    {
        std::cerr << consistency_error(mesh.n_faces(), total_faces, "faces").what() << std::endl;
    }
}

bool add_face(pmp::SurfaceMesh& target,
              FaceHandle fH,
              std::unordered_map<VertexHandle, VertexHandle>& vertex_map,
              const pmp::SurfaceMesh& src)
{
    static std::vector<VertexHandle> face_vertices;
    face_vertices.clear();
    for (VertexHandle vH : src.vertices(fH))
    {
        auto it = vertex_map.find(vH);
        if (it == vertex_map.end())
        {
            VertexHandle new_vH = target.add_vertex(src.position(vH));
            target.copy_vprops(src, vH, new_vH);
            vertex_map[vH] = new_vH;
            face_vertices.push_back(new_vH);
        }
        else
        {
            face_vertices.push_back(it->second);
        }
    }
    try
    {
        FaceHandle new_fH = target.add_face(face_vertices);
        target.copy_fprops(src, fH, new_fH);
    }
    catch (pmp::TopologyException& e)
    {
        return false;
    }
    return true;
}

void distribute_segments(pmp::SurfaceMesh& mesh,
                         std::vector<Segment>& segments,
                         std::vector<SegmentId>& segment_map,
                         std::vector<bool> to_be_split,
                         size_t to_be_split_count,
                         std::vector<pmp::SurfaceMesh>& out_meshes,
                         std::vector<pmp::BoundingBox>& out_bounds,
                         std::vector<Segment>& small_segments)
{
    out_meshes.clear();
    out_bounds.clear();
    std::vector<std::unordered_map<VertexHandle, VertexHandle>> vertex_maps(to_be_split_count);

    SegmentId biggest_id = INVALID_SEGMENT;
    pmp::FaceProperty<bool> f_biggest_delete;

    auto f_prop = mesh.get_face_property<SegmentId>("f:segment");
    auto f_delete = mesh.add_face_property<bool>("f:flag_delete", false);


    for (auto& segment : segments)
    {
        if (!to_be_split[segment.id])
        {
            continue;
        }

        SegmentId new_id = out_meshes.size();
        segment_map[segment.id] = new_id;
        vertex_maps[new_id].reserve(segment.num_vertices);
        out_bounds.push_back(segment.bb);

        if (segment.num_faces < mesh.n_faces() / 2)
        {
            auto& out_mesh = out_meshes.emplace_back();
            out_mesh.reserve(segment.num_vertices, 0, segment.num_faces);
            out_mesh.copy_properties(mesh);
        }
        else
        {
            // segment is too large, just copy the original the mesh
            auto& out_mesh = out_meshes.emplace_back(mesh);
            biggest_id = new_id;
            f_biggest_delete = out_mesh.get_face_property<bool>("f:flag_delete");
        }
    }

    ProgressBar progress(mesh.n_faces(), "Extracting large segments");
    size_t fail_count = 0;
    for (FaceHandle fH : mesh.faces())
    {
        ++progress;
        SegmentId& id = f_prop[fH];
        SegmentId new_id = segment_map[id];
        if (!to_be_split[id])
        {
            id = new_id;
            continue;
        }
        f_delete[fH] = true;
        if (new_id != biggest_id)
        {
            if (!add_face(out_meshes[new_id], fH, vertex_maps[new_id], mesh))
            {
                fail_count++;
            }
            if (f_biggest_delete)
            {
                f_biggest_delete[fH] = true;
            }
        }
    }
    std::cout << "\r";
    if (fail_count > 0)
    {
        std::cout << fail_count << " broken faces removed" << std::endl;
    }
    mesh.delete_many_faces(f_delete);
    mesh.remove_face_property(f_delete);
    mesh.garbage_collection();

    if (f_biggest_delete)
    {
        out_meshes[biggest_id].delete_many_faces(f_biggest_delete);
        // out_meshes[biggest_id].garbage_collection();
    }

    for (auto& out_mesh : out_meshes)
    {
        out_mesh.remove_face_property<SegmentId>("f:segment");
        out_mesh.remove_face_property<bool>("f:flag_delete");
    }
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
        if (mesh.is_deleted(fH) || !to_be_split[f_prop[fH]])
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
