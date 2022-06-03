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
                  std::vector<std::pair<pmp::Point, MeshSegment>>& chunks,
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

    std::unordered_map<pmp::IndexType, SegmentId> chunk_map;
    pmp::Point total_size = bb.max() - bb.min();
    pmp::Point size = total_size / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    // align the chunking to the center of the segment:
    // modify the offset so that there is equal overlap on all sides
    pmp::Point size_of_chunks = num_chunks.cast<float>() * chunk_size;
    pmp::Point chunk_offset = bb.min() - (size_of_chunks - total_size) / 2.0f;

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
        auto chunk_id = chunk_index(segment.bb.center() - chunk_offset, chunk_size, num_chunks);
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
        auto& [ chunk_pos, segment ] = chunks.emplace_back();
        chunk_pos = chunk_position(chunk_id, 1, num_chunks);

        meshes[index].add_object_property<pmp::IndexType>("o:chunk_id")[0] = index;
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

    auto v_chunk_id = mesh.add_vertex_property<pmp::IndexType>("v:chunk_id", pmp::PMP_MAX_INDEX);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        v_chunk_id[pmp::Vertex(i)] = chunk_index(mesh.position(pmp::Vertex(i)) - chunk_offset, chunk_size, num_chunks);
    }

    auto f_chunk_id = mesh.add_face_property<pmp::IndexType>("f:chunk_id", pmp::PMP_MAX_INDEX);
    std::unordered_map<pmp::IndexType, size_t> face_counts;
    #pragma omp parallel
    {
        std::unordered_map<pmp::IndexType, size_t> local_face_counts;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            pmp::Face fH(i);
            auto chunk_id = v_chunk_id[mesh.to_vertex(mesh.halfedge(fH))];
            f_chunk_id[fH] = chunk_id;
            local_face_counts[chunk_id]++;
        }
        #pragma omp critical
        {
            for (auto [ chunk_id, count ] : local_face_counts)
            {
                face_counts[chunk_id] += count;
            }
        }
    }

    std::unordered_map<pmp::IndexType, pmp::IndexType> chunk_map;
    std::vector<pmp::SurfaceMesh> meshes;
    chunk_map.reserve(face_counts.size());
    meshes.reserve(face_counts.size());
    for (auto [ chunk_id, face_count ] : face_counts)
    {
        auto [ index, mesh ] = push_and_get_index(meshes);
        chunk_map[chunk_id] = index;
        mesh.reserve(0, face_count * 3 / 2, face_count);
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        f_chunk_id[pmp::Face(i)] = chunk_map[f_chunk_id[pmp::Face(i)]];
    }

    mesh.split_mesh(meshes, f_chunk_id);

    std::vector<std::pair<pmp::Point, MeshSegment>> chunks;
    for (auto [ chunk_id, index ] : chunk_map)
    {
        auto& [ chunk_pos, out_segment ] = chunks.emplace_back();
        chunk_pos = chunk_position(chunk_id, 1, num_chunks);

        out_segment.bb = meshes[index].bounds();
        meshes[index].remove_vertex_property<pmp::IndexType>(v_chunk_id.name());
        meshes[index].add_object_property<pmp::IndexType>("o:chunk_id")[0] = index;
        PMPMesh<BaseVector<float>> pmp_mesh;
        pmp_mesh.getSurfaceMesh() = std::move(meshes[index]);
        out_segment.mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
        out_segment.texture_file = in_segment.texture_file;
    }

    auto tree = SegmentTree::octree_partition(chunks, num_chunks, combine_depth);

    std::vector<pmp::IndexType> combine_map(meshes.size(), pmp::PMP_MAX_INDEX);
    std::vector<MeshSegment*> combined_meshes;

    std::vector<SegmentTree*> queue;
    queue.push_back(tree.get());
    while (!queue.empty())
    {
        auto current = queue.back();
        queue.pop_back();
        if (current->is_leaf())
        {
            continue;
        }
        SegmentTreeNode* node = dynamic_cast<SegmentTreeNode*>(current);
        if (node->m_depth > 1)
        {
            for (auto& child : node->children())
            {
                queue.push_back(child.get());
            }
            continue;
        }

        pmp::IndexType new_id = combined_meshes.size();
        combined_meshes.push_back(&node->segment());

        for (auto& child : node->children())
        {
            auto pmp_mesh = child->segment().mesh->get();
            auto& mesh = pmp_mesh->getSurfaceMesh();
            auto o_prop = mesh.get_object_property<pmp::IndexType>("o:chunk_id");
            pmp::IndexType old_id = o_prop[0];
            combine_map[old_id] = new_id;

            mesh.remove_object_property(o_prop);
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex vH(i);
        auto& id = v_chunk_id[vH];
        auto it = chunk_map.find(id);
        // id not found in chunk_map means there was no complete face in the chunk, just the vertex
        id = it != chunk_map.end() ? it->second : f_chunk_id[*mesh.faces(vH)];
        id = combine_map[id];
    }

    std::vector<pmp::Face> feature_faces;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        auto& id = f_chunk_id[fH];
        id = combine_map[id];

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

    if (!feature_faces.empty())
    {
        auto v_feature = mesh.add_vertex_property<bool>("v:feature");
        mesh.add_edge_property("e:feature", false);

        std::vector<pmp::Vertex> vertices;
        vertices.reserve(3);
        auto fprop_map = mesh.gen_fprop_map(mesh);
        auto vprop_map = mesh.gen_vprop_map(mesh);
        for (auto fH : feature_faces)
        {
            vertices.clear();
            auto target_id = f_chunk_id[fH];
            for (pmp::Vertex vH : mesh.vertices(fH))
            {
                v_feature[vH] = true;
                if (v_chunk_id[vH] != target_id)
                {
                    auto new_vH = mesh.add_vertex(mesh.position(vH));
                    mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
                    vH = new_vH;
                    v_chunk_id[vH] = target_id;
                }
                vertices.push_back(vH);
            }
            mesh.delete_face(fH);
            for (auto& vH : vertices)
            {
                if (mesh.is_deleted(vH))
                {
                    auto new_vH = mesh.add_vertex(mesh.position(vH));
                    mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
                    vH = new_vH;
                }
            }
            auto new_fH = mesh.add_face(vertices);
            mesh.copy_fprops(mesh, fH, new_fH, fprop_map);
        }
    }


    meshes.clear();
    meshes.resize(combined_meshes.size());
    mesh.split_mesh(meshes, f_chunk_id, v_chunk_id);

    mesh.remove_face_property(f_chunk_id);
    mesh.remove_vertex_property(v_chunk_id);

    #pragma omp parallel for
    for (size_t i = 0; i < combined_meshes.size(); i++)
    {
        PMPMesh<BaseVector<float>> pmp_mesh;
        pmp_mesh.getSurfaceMesh() = std::move(meshes[i]);
        combined_meshes[i]->mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
    }

    return tree;
}

} // namespace lvr2
