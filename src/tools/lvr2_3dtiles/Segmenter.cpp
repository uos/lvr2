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

struct SegmentMetaData
{
    SegmentId id = INVALID_SEGMENT;
    size_t num_faces = 0;
    size_t num_vertices = 0;
    pmp::BoundingBox bb;
    std::string filename = "";
};

/**
 * @brief Calculates a 1D Chunk-index from a 3D position
 *
 * @param p the 3D position
 * @param chunk_size the size of a chunk
 * @param num_chunks the number of chunks along each axis
 * @return pmp::IndexType the 1D Chunk-index
 */
pmp::IndexType chunk_index(pmp::Point p, float chunk_size, Eigen::Vector3i num_chunks)
{
    return std::floor(p.x() / chunk_size)
           + std::floor(p.y() / chunk_size) * num_chunks.x()
           + std::floor(p.z() / chunk_size) * num_chunks.x() * num_chunks.y();
}

void segment_mesh(pmp::SurfaceMesh& mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  std::vector<MeshSegment>& chunks,
                  std::vector<MeshSegment>& large_segments)
{
    auto f_prop = mesh.add_face_property<SegmentId>("f:segment", INVALID_SEGMENT);
    auto v_prop = mesh.add_vertex_property<SegmentId>("v:segment", INVALID_SEGMENT);
    auto h_prop = mesh.add_halfedge_property<SegmentId>("h:segment", INVALID_SEGMENT);

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

                h_prop[heH] = segment.id;

                FaceHandle fH = mesh.face(heH);
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
        pmp::Face f(i);
        if (!mesh.is_deleted(f))
        {
            f_prop[f] = segment_map[f_prop[f]];
        }
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex v(i);
        if (!mesh.is_deleted(v))
        {
            v_prop[v] = segment_map[v_prop[v]];
        }
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.edges_size(); i++)
    {
        pmp::Edge e(i);
        if (!mesh.is_deleted(e))
        {
            pmp::Halfedge h0 = mesh.halfedge(e, 0);
            pmp::Halfedge h1 = mesh.halfedge(e, 1);
            h_prop[h0] = segment_map[h_prop[h0]];
            h_prop[h1] = segment_map[h_prop[h1]];
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(meta_data.size());
    for (size_t i = 0; i < meta_data.size(); i++)
    {
        meshes[i].reserve(meta_data[i].num_vertices, 0, meta_data[i].num_faces);
    }
    mesh.split_mesh(meshes, f_prop, v_prop, h_prop);
    mesh.remove_face_property(f_prop);
    mesh.remove_vertex_property(v_prop);
    mesh.remove_halfedge_property(h_prop);

    for (size_t i = 0; i < meta_data.size(); i++)
    {
        auto& meta = meta_data[i];
        MeshSegment& out = is_large[i] ? large_segments.emplace_back() : chunks.emplace_back();
        out.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[i])));
        out.bb = meta.bb;

        // consistency check
        if (out.mesh->n_faces() != meta.num_faces)
        {
            std::cerr << consistency_error(meta.num_faces, out.mesh->n_faces(), "MeshSegment faces").what() << std::endl;
        }
        if (out.mesh->n_vertices() != meta.num_vertices)
        {
            std::cerr << consistency_error(meta.num_vertices, out.mesh->n_vertices(), "MeshSegment vertices").what() << std::endl;
        }
    }

    std::cout << timestamp << "Merged " << (segments.size() - large_segments.size()) << " small segments into "
              << chunks.size() << " chunks" << std::endl;
}

SegmentTree::Ptr split_mesh_bottom_up(MeshSegment& segment, float chunk_size)
{
    auto& mesh = *segment.mesh;

    pmp::Point size_of_segment = segment.bb.max() - segment.bb.min();
    pmp::Point size = size_of_segment / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    // align the chunking to the center of the segment:
    // modify the offset so that there is equal overlap on all sides
    pmp::Point size_of_chunks = num_chunks.cast<float>() * chunk_size;
    pmp::Point chunk_offset = segment.bb.min() - (size_of_chunks - size_of_segment) / 2.0f;

    auto f_chunk_id = mesh.face_property<pmp::IndexType>("f:chunk_id", pmp::PMP_MAX_INDEX);
    std::unordered_set<pmp::IndexType> chunk_ids;
    #pragma omp parallel
    {
        std::unordered_set<pmp::IndexType> local_chunk_ids;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            FaceHandle fH(i);
            if (mesh.is_deleted(fH))
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

            auto chunk_id = chunk_index(pos - chunk_offset, chunk_size, num_chunks);
            f_chunk_id[fH] = chunk_id;
            local_chunk_ids.insert(chunk_id);
        }
        #pragma omp critical
        {
            chunk_ids.insert(local_chunk_ids.begin(), local_chunk_ids.end());
        }
    }

    std::unordered_map<pmp::IndexType, pmp::IndexType> chunk_map;
    for (auto chunk_id : chunk_ids)
    {
        auto index = chunk_map.size();
        chunk_map[chunk_id] = index;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (!mesh.is_deleted(fH))
        {
            f_chunk_id[fH] = chunk_map[f_chunk_id[fH]];
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(chunk_map.size());
    mesh.split_mesh(meshes, f_chunk_id);

    std::vector<MeshSegment> segments;

    for (size_t i = 0; i < meshes.size(); i++)
    {
        auto& mesh = meshes[i];
        mesh.object_property<pmp::IndexType>("o:chunk_id")[0] = i;

        auto& out_mesh = segments.emplace_back();
        out_mesh.mesh.reset(new pmp::SurfaceMesh(std::move(mesh)));
        out_mesh.bb = out_mesh.mesh->bounds();
    }

    auto tree = SegmentTree::octree_partition(segments, 2);

    std::vector<pmp::IndexType> combine_map(meshes.size(), pmp::PMP_MAX_INDEX);
    std::vector<std::shared_ptr<pmp::SurfaceMesh>> combined_meshes;

    std::vector<SegmentTree*> queue;
    queue.push_back(tree.get());
    while (!queue.empty())
    {
        auto current = queue.back();
        queue.pop_back();
        SegmentTreeNode* node = dynamic_cast<SegmentTreeNode*>(current);
        if (!node)
        {
            continue;
        }
        if (node->m_depth > 1)
        {
            for (auto& child : node->children())
            {
                queue.push_back(child.get());
            }
            continue;
        }
        node->segment().mesh.reset(new pmp::SurfaceMesh());

        pmp::IndexType new_id = combined_meshes.size();
        combined_meshes.push_back(node->segment().mesh);

        for (auto& child : node->children())
        {
            auto& mesh = *child->segment().mesh;
            auto o_prop = mesh.get_object_property<pmp::IndexType>("o:chunk_id");
            pmp::IndexType old_id = o_prop[0];
            combine_map[old_id] = new_id;

            mesh.remove_object_property(o_prop);
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (!mesh.is_deleted(fH))
        {
            f_chunk_id[fH] = combine_map[f_chunk_id[fH]];
        }
    }

    // mark all vertices on chunk borders as features
    auto v_feature = mesh.add_vertex_property("v:feature", false);
    #pragma omp parallel for schedule(static,256)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        VertexHandle vH(i);
        if (mesh.is_deleted(vH))
        {
            continue;
        }
        pmp::IndexType id = pmp::PMP_MAX_INDEX;
        for (auto fH : mesh.faces(vH))
        {
            if (id == pmp::PMP_MAX_INDEX)
            {
                id = f_chunk_id[fH];
            }
            else if (id != f_chunk_id[fH])
            {
                v_feature[vH] = true;
                break;
            }
        }
    }

    meshes.clear();
    meshes.resize(combined_meshes.size());
    mesh.split_mesh(meshes, f_chunk_id);

    #pragma omp parallel for
    for (size_t i = 0; i < combined_meshes.size(); i++)
    {
        *combined_meshes[i] = std::move(meshes[i]);
        combined_meshes[i]->add_edge_property("e:feature", false);
    }

    mesh.remove_face_property(f_chunk_id);

    return tree;
}

SegmentTree::Ptr split_mesh_top_down(MeshSegment& segment, float chunk_size, bool print)
{
    int total_depth = std::ceil(std::log2(segment.bb.longest_axis_size() / chunk_size));
    pmp::Point size = segment.bb.max() - segment.bb.min();
    if (print)
    {
        std::cout << timestamp << "Segment " << size.x() << " x " << size.y() << " x " << size.z()
                  << " has " << total_depth << " levels" << std::endl;
    }
    if (total_depth < 1)
    {
        return SegmentTree::Ptr(new SegmentTreeLeaf(segment));
    }

    auto& base_mesh = *segment.mesh;

    if (!base_mesh.has_vertex_property("v:quadric"))
    {
        pmp::SurfaceSimplification::calculate_quadrics(base_mesh);
    }

    auto f_chunk_id = base_mesh.face_property<pmp::IndexType>("f:chunk_id");

    pmp::Point center = segment.bb.center();
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < base_mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (base_mesh.is_deleted(fH))
        {
            continue;
        }
        auto pos = base_mesh.center(fH);

        pmp::IndexType index = 0;
        for (size_t axis = 0; axis < 3; axis++)
        {
            if (pos[axis] > center[axis])
            {
                index |= (1 << axis);
            }
        }
        f_chunk_id[fH] = index;
    }

    std::vector<std::vector<pmp::SurfaceMesh>> layers(total_depth + 1);
    std::vector<std::vector<pmp::Point>> centers(total_depth);
    centers[0].push_back(segment.bb.center());

    if (print)
    {
        std::cout << timestamp << "Simplifying: " << base_mesh.n_faces() << " -> " << std::flush;
    }
    pmp::SurfaceSimplification simplifier(base_mesh);

    for (size_t i = 0; i < total_depth; i++)
    {
        auto& next_layer = layers[i];
        next_layer.resize(8);
        base_mesh.split_mesh(next_layer, f_chunk_id);

        simplifier.simplify(base_mesh.n_vertices() * 0.2);
        if (print)
        {
            std::cout << base_mesh.n_faces() << (i < total_depth - 1 ? " -> " : "") << std::flush;
        }
    }
    if (print)
    {
        std::cout << std::endl;
        std::cout << timestamp << "Constructing tree" << std::endl;
    }

    // final layer: only most simplified mesh
    layers[total_depth].push_back(std::move(base_mesh));

    for (size_t depth = 1; depth < total_depth; depth++)
    {
        size_t n = layers[0].size();

        auto& depth_centers = centers[depth];
        depth_centers.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            auto bb = layers[0][i].bounds();
            depth_centers[i] = bb.center();
        }

        size_t next_n = n * 8;
        for (size_t layer = 0; layer < total_depth - depth; layer++)
        {
            auto& meshes = layers[layer];
            std::vector<pmp::SurfaceMesh> next_meshes;
            for (size_t i = 0; i < n; i++)
            {
                auto& center = depth_centers[i];
                auto& mesh = meshes[i];
                auto f_chunk_id = mesh.face_property<pmp::IndexType>("f:chunk_id");
                for (auto fH : mesh.faces())
                {
                    auto pos = mesh.center(fH);
                    pmp::IndexType index = 0;
                    for (size_t axis = 0; axis < 3; axis++)
                    {
                        if (pos[axis] > center[axis])
                        {
                            index |= (1 << axis);
                        }
                    }
                    f_chunk_id[fH] = index;
                }
                std::vector<pmp::SurfaceMesh> sub_meshes(8);
                mesh.split_mesh(sub_meshes, f_chunk_id);
                for (auto& sub_mesh : sub_meshes)
                {
                    next_meshes.push_back(std::move(sub_mesh));
                }
            }
            if (next_meshes.size() != next_n)
            {
                throw consistency_error(next_n, next_meshes.size(), "next Meshes");
            }
            std::swap(meshes, next_meshes);
        }
    }

    std::vector<SegmentTree::Ptr> row;
    for (auto& mesh : layers[0])
    {
        if (mesh.n_faces() == 0)
        {
            row.push_back(nullptr);
            continue;
        }
        MeshSegment segment;
        segment.mesh.reset(new pmp::SurfaceMesh(std::move(mesh)));
        segment.bb = segment.mesh->bounds();
        row.push_back(SegmentTree::Ptr(new SegmentTreeLeaf(segment)));
    }

    for (size_t layer = 1; layer < layers.size(); layer++)
    {
        std::vector<SegmentTree::Ptr> next_row;
        for (size_t block = 0; block < row.size(); block += 8)
        {
            auto node = new SegmentTreeNode();
            for (size_t i = 0; i < 8; i++)
            {
                if (row[block + i] == nullptr)
                {
                    continue;
                }
                node->add_child(std::move(row[block + i]));
            }
            if (node->num_children() == 0)
            {
                delete node;
                next_row.push_back(nullptr);
            }
            else
            {
                node->segment().mesh.reset(new pmp::SurfaceMesh(std::move(layers[layer][block / 8])));
                next_row.push_back(SegmentTree::Ptr(node));
            }
        }
        std::swap(row, next_row);
    }
    if (row.size() != 1)
    {
        throw consistency_error(1, row.size(), "rows of segment trees");
    }
    return std::move(row[0]);
}

} // namespace lvr2
