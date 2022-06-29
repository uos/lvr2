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
#include "lvr2/algorithm/pmp/SurfaceSimplification.h"

namespace lvr2
{

void chunk_index(const pmp::Point& vec, float chunk_size, Vector3i& index)
{
    index.x() = std::floor(vec.x() / chunk_size);
    index.y() = std::floor(vec.y() / chunk_size);
    index.z() = std::floor(vec.z() / chunk_size);
}
Vector3i chunk_index(const pmp::Point& vec, float chunk_size)
{
    Vector3i ret;
    chunk_index(vec, chunk_size, ret);
    return ret;
}

template<typename T>
using ChunkMap = std::unordered_map<Vector3i, T>;

struct consistency_error : public std::runtime_error
{
    consistency_error(size_t expected, size_t found, const char* name)
        : std::runtime_error(std::string("Segmenter: inconsistent number of ") + name + ": "
                             + "expected " + std::to_string(expected)
                             + ", found " + std::to_string(found))
    {}
};

typedef pmp::IndexType SegmentId;
constexpr SegmentId INVALID_SEGMENT = pmp::PMP_MAX_INDEX;

struct SegmentMetaData
{
    SegmentId id = INVALID_SEGMENT;
    size_t num_faces = 0;
    size_t num_vertices = 0;
    pmp::BoundingBox bb;
};

void segment_mesh(pmp::SurfaceMesh& mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  ChunkMap<MeshSegment>& chunks,
                  std::vector<MeshSegment>& large_segments,
                  std::shared_ptr<HighFive::File> mesh_file)
{
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

    auto f_prop = mesh.add_face_property<SegmentId>("f:segment", INVALID_SEGMENT);
    auto v_prop = mesh.add_vertex_property<SegmentId>("v:segment", INVALID_SEGMENT);

    std::vector<SegmentMetaData> segments;

    std::vector<pmp::Vertex> queue;
    ProgressBar progress(mesh.n_vertices(), "Segmenting mesh");

    for (auto vH : mesh.vertices())
    {
        if (v_prop[vH] != INVALID_SEGMENT)
        {
            continue;
        }

        auto [ id, segment ] = push_and_get_index(segments);
        segment.id = id;

        v_prop[vH] = segment.id;
        queue.push_back(vH);

        while (!queue.empty())
        {
            pmp::Vertex vH = queue.back();
            queue.pop_back();
            segment.num_vertices++;
            segment.bb += mesh.position(vH);
            ++progress;

            for (auto heH : mesh.halfedges(vH))
            {
                auto ovH = mesh.to_vertex(heH);
                auto& ovH_id = v_prop[ovH];
                if (ovH_id != segment.id)
                {
                    if (ovH_id != INVALID_SEGMENT)
                    {
                        throw std::runtime_error("Segmenter: found vertex with multiple segments");
                    }
                    ovH_id = segment.id;
                    queue.push_back(ovH);
                }

                pmp::Face fH = mesh.face(heH);
                if (fH.is_valid() && f_prop[fH] != segment.id)
                {
                    f_prop[fH] = segment.id;
                    segment.num_faces++;
                }
            }
        }
    }
    std::cout << "\r" << timestamp << "Found " << segments.size() << " initial segments" << std::endl;

    // consistency check
    size_t total_faces = 0, total_vertices = 0;
    for (auto& segment : segments)
    {
        total_faces += segment.num_faces;
        total_vertices += segment.num_vertices;
    }
    if (total_faces != mesh.n_faces())
    {
        throw consistency_error(mesh.n_faces(), total_faces, "SegmentMetaData faces");
    }
    if (total_vertices != mesh.n_vertices())
    {
        throw consistency_error(mesh.n_vertices(), total_vertices, "SegmentMetaData vertices");
    }

    // ==================== merge small segments within a chunk together ====================

    ChunkMap<SegmentId> chunk_map;
    std::vector<SegmentId> segment_map(segments.size(), INVALID_SEGMENT);
    std::vector<SegmentMetaData> meta_data;
    std::vector<bool> is_large;

    for (auto& segment : segments)
    {
        if (segment.num_faces == 0)
        {
            continue;
        }

        if (segment.bb.longest_axis_size() >= chunk_size)
        {
            SegmentId id = segment.id;
            auto [ new_id, meta ] = push_and_get_index(meta_data, std::move(segment));
            meta.id = new_id;
            segment_map[id] = new_id;
            is_large.push_back(true);
            continue;
        }

        // all other segments are merged based on the chunk that their center lies in
        auto chunk_id = chunk_index(segment.bb.center(), chunk_size);
        auto elem = chunk_map.find(chunk_id);
        if (elem == chunk_map.end())
        {
            // start a new chunk with this segment
            SegmentId id = segment.id;
            auto [ new_id, meta ] = push_and_get_index(meta_data, std::move(segment));
            meta.id = new_id;
            segment_map[id] = new_id;
            chunk_map[chunk_id] = new_id;
            is_large.push_back(false);
        }
        else
        {
            // merge this segment with the existing chunk
            SegmentId new_id = elem->second;
            segment_map[segment.id] = new_id;

            auto& target = meta_data[new_id];
            target.num_faces += segment.num_faces;
            target.num_vertices += segment.num_vertices;
            target.bb += segment.bb;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        f_prop[pmp::Face(i)] = segment_map[f_prop[pmp::Face(i)]];
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        v_prop[pmp::Vertex(i)] = segment_map[v_prop[pmp::Vertex(i)]];
    }

    std::vector<pmp::SurfaceMesh> meshes(meta_data.size());
    for (size_t i = 0; i < meta_data.size(); i++)
    {
        meshes[i].reserve(meta_data[i].num_vertices, 0, meta_data[i].num_faces);
    }
    mesh.split_mesh(meshes, f_prop, v_prop);
    mesh.remove_face_property(f_prop);
    mesh.remove_vertex_property(v_prop);

    for (size_t i = 0; i < meshes.size(); i++)
    {
        if (is_large[i])
        {
            MeshSegment& out = large_segments.emplace_back();
            PMPMesh<BaseVector<float>> pmp_mesh;
            pmp_mesh.getSurfaceMesh() = std::move(meshes[i]);
            out.mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
            out.bb = meta_data[i].bb;
        }
    }
    for (auto [ chunk_id, index ] : chunk_map)
    {
        auto& segment = chunks[chunk_id];

        PMPMesh<BaseVector<float>> pmp_mesh;
        pmp_mesh.getSurfaceMesh() = std::move(meshes[index]);
        segment.mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
        segment.bb = meta_data[index].bb;
    }

    std::cout << timestamp << "Merged " << (segments.size() - large_segments.size()) << " small segments into "
              << chunks.size() << " chunks" << std::endl;
}

SegmentTree::Ptr split_mesh(MeshSegment& in_segment,
                            float chunk_size,
                            std::shared_ptr<HighFive::File> mesh_file,
                            int combine_depth)
{
    if (in_segment.bb.longest_axis_size() <= chunk_size)
    {
        return SegmentTree::Ptr(new SegmentTreeLeaf(in_segment));
    }

    auto pmp_mesh = in_segment.mesh->get();
    auto& mesh = pmp_mesh->getSurfaceMesh();
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

    pmp::Point size_of_segment = in_segment.bb.max() - in_segment.bb.min();
    pmp::Point size = size_of_segment / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    // align the chunking to the center of the segment:
    // modify the offset so that there is equal overlap on all sides
    pmp::Point size_of_chunks = num_chunks.cast<float>() * chunk_size;
    pmp::Point chunk_offset = in_segment.bb.min() - (size_of_chunks - size_of_segment) / 2.0f;

    ChunkMap<pmp::IndexType> chunk_map;
    std::vector<pmp::Face> feature_faces;

    auto v_chunk_id = mesh.add_vertex_property<pmp::IndexType>("v:chunk_id", pmp::PMP_MAX_INDEX);
    auto f_chunk_id = mesh.add_face_property<pmp::IndexType>("f:chunk_id", pmp::PMP_MAX_INDEX);

    #pragma omp parallel
    {
        ChunkMap<pmp::IndexType> local_chunk_map; // copy of chunk_map to avoid having to lock on
                                                  // every lookup. updated when necessary

        #pragma omp for schedule(static)
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            pmp::Face fH(i);
            auto pos = mesh.position(mesh.to_vertex(mesh.halfedge(fH)));
            auto chunk_id = chunk_index(pos - chunk_offset, chunk_size);
            auto it = local_chunk_map.find(chunk_id);
            if (it != local_chunk_map.end())
            {
                f_chunk_id[fH] = it->second;
            }
            else
            {
                #pragma omp critical
                {
                    it = chunk_map.find(chunk_id);
                    if (it == chunk_map.end())
                    {
                        pmp::IndexType id = chunk_map.size();
                        it = chunk_map.emplace(chunk_id, id).first;
                    }
                    f_chunk_id[fH] = it->second;
                    local_chunk_map = chunk_map;
                }
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < mesh.vertices_size(); i++)
        {
            pmp::Vertex vH(i);
            auto chunk_id = chunk_index(mesh.position(vH) - chunk_offset, chunk_size);
            auto it = chunk_map.find(chunk_id);
            v_chunk_id[vH] = it != chunk_map.end() ? it->second : f_chunk_id[*mesh.faces(vH)];
        }

        #pragma omp for schedule(dynamic,64)
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            pmp::Face fH(i);
            auto id = f_chunk_id[fH];

            for (auto vH : mesh.vertices(fH))
            {
                if (v_chunk_id[vH] != id)
                {
                    #pragma omp critical
                    feature_faces.push_back(fH);
                    break;
                }
            }
        }
    }

    if (!feature_faces.empty())
    {
        auto v_feature = mesh.add_vertex_property<bool>("v:feature");
        mesh.add_edge_property("e:feature", false);

        std::unordered_map<pmp::IndexType, std::unordered_map<pmp::Vertex, pmp::Vertex>> vertex_map;
        std::vector<pmp::Vertex> vertices;
        std::vector<pmp::Point> positions;
        vertices.reserve(3);
        positions.reserve(3);

        auto fprop_map = mesh.gen_fprop_map(mesh);
        auto vprop_map = mesh.gen_vprop_map(mesh);
        for (auto fH : feature_faces)
        {
            auto target_id = f_chunk_id[fH];
            auto& v_map = vertex_map[target_id];

            vertices.clear();
            positions.clear();
            for (auto vH : mesh.vertices(fH))
            {
                vertices.push_back(vH);
                positions.push_back(mesh.position(vH));
                if (v_chunk_id[vH] != target_id)
                {
                    v_feature[vH] = true;
                }
            }
            mesh.delete_face(fH);

            for (size_t i = 0; i < 3; i++)
            {
                auto& vH = vertices[i];
                auto it = v_map.find(vH);
                if (it != v_map.end())
                {
                    vH = it->second;
                }
                else if (mesh.is_deleted(vH) || v_chunk_id[vH] != target_id)
                {
                    auto new_vH = mesh.add_vertex(positions[i]);
                    mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
                    v_chunk_id[new_vH] = target_id;
                    v_map[vH] = new_vH;
                    vH = new_vH;
                }
            }
            auto new_fH = mesh.add_face(vertices);
            mesh.copy_fprops(mesh, fH, new_fH, fprop_map);
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(chunk_map.size());

    mesh.split_mesh(meshes, f_chunk_id, v_chunk_id);

    mesh.remove_face_property(f_chunk_id);
    mesh.remove_vertex_property(v_chunk_id);

    ChunkMap<MeshSegment> chunks;
    for (auto [ chunk_id, index ] : chunk_map)
    {
        auto& out_segment = chunks[chunk_id];
        out_segment.bb = meshes[index].bounds();

        PMPMesh<BaseVector<float>> pmp_mesh;
        pmp_mesh.getSurfaceMesh() = std::move(meshes[index]);
        out_segment.mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
    }

    return SegmentTree::octree_partition(chunks, combine_depth);
}

} // namespace lvr2
