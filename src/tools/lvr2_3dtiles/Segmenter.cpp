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
                  std::vector<MeshSegment>& large_segments)
{
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

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
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.halfedges_size(); i++)
    {
        h_prop[pmp::Halfedge(i)] = segment_map[h_prop[pmp::Halfedge(i)]];
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

    for (size_t i = 0; i < meshes.size(); i++)
    {
        if (is_large[i])
        {
            MeshSegment& out = large_segments.emplace_back();
            out.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[i])));
            out.bb = meta_data[i].bb;
        }
    }
    for (auto [ chunk_id, index ] : chunk_map)
    {
        auto& [ chunk_pos, segment ] = chunks.emplace_back();
        chunk_pos = chunk_position(chunk_id, 1, num_chunks);

        segment.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[index])));
        segment.mesh->add_object_property<pmp::IndexType>("o:chunk_id")[0] = index;
        segment.bb = meta_data[index].bb;
    }

    std::cout << timestamp << "Merged " << (segments.size() - large_segments.size()) << " small segments into "
              << chunks.size() << " chunks" << std::endl;
}

SegmentTree::Ptr split_mesh_bottom_up(MeshSegment& in_segment, float chunk_size)
{
    if (in_segment.bb.longest_axis_size() <= chunk_size)
    {
        return SegmentTree::Ptr(new SegmentTreeLeaf(in_segment));
    }

    auto& mesh = *in_segment.mesh;
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

        out_segment.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[index])));
        out_segment.mesh->add_object_property<pmp::IndexType>("o:chunk_id")[0] = index;
        out_segment.bb = out_segment.mesh->bounds();
        out_segment.texture_file = in_segment.texture_file;
    }

    auto tree = SegmentTree::octree_partition(chunks, num_chunks, 2);

    std::vector<pmp::IndexType> combine_map(meshes.size(), pmp::PMP_MAX_INDEX);
    std::vector<std::shared_ptr<pmp::SurfaceMesh>> combined_meshes;

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
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex vH(i);
        auto& id = v_chunk_id[vH];
        auto it = chunk_map.find(id);
        // id not found in chunk_map means there was no complete face in the chunk, just the vertex
        id = it != chunk_map.end() ? it->second : f_chunk_id[*mesh.faces(vH)];
        id = combine_map[id];
    }

    auto v_feature = mesh.add_vertex_property<bool>("v:feature");
    auto v_original = mesh.add_vertex_property<pmp::Vertex>("v:original");
    auto f_boundary = mesh.add_face_property<bool>("f:boundary");
    size_t feature_count = 0;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        auto id = pmp::PMP_MAX_INDEX;

        bool boundary = false;
        for (auto vH : mesh.vertices(fH))
        {
            auto v_id = v_chunk_id[vH];
            if (id == pmp::PMP_MAX_INDEX)
            {
                id = v_id;
            }
            else if (v_id != pmp::PMP_MAX_INDEX && v_id != id)
            {
                boundary = true;
                break;
            }
        }
        f_chunk_id[fH] = id;
        if (boundary)
        {
            #pragma omp critical
            {
                f_boundary[fH] = true;
                for (auto vH : mesh.vertices(fH))
                {
                    v_feature[vH] = true;
                    v_original[vH] = vH;
                }
                ++feature_count;
            }
        }
    }
    if (feature_count == 0)
    {
        mesh.remove_vertex_property(v_feature);
        mesh.remove_vertex_property(v_original);
        mesh.remove_face_property(f_boundary);
    }
    else
    {
        auto v_merge_count = mesh.add_vertex_property<uint32_t>("v:merge_count");
        std::unordered_set<pmp::IndexType> ids;
        #pragma omp parallel for schedule(dynamic,64) private(ids)
        for (size_t i = 0; i < mesh.vertices_size(); i++)
        {
            pmp::Vertex vH(i);
            if (!v_feature[vH])
            {
                continue;
            }
            ids.clear();
            for (auto fH : mesh.faces(vH))
            {
                ids.insert(f_chunk_id[fH]);
            }
            v_merge_count[vH] = ids.size();
        }
    }


    meshes.clear();
    meshes.resize(combined_meshes.size());
    mesh.split_mesh(meshes, f_chunk_id);

    #pragma omp parallel for
    for (size_t i = 0; i < combined_meshes.size(); i++)
    {
        *combined_meshes[i] = std::move(meshes[i]);
        if (feature_count > 0)
        {
            combined_meshes[i]->add_edge_property("e:feature", false);
        }
    }

    mesh.remove_face_property(f_chunk_id);
    mesh.remove_vertex_property(v_chunk_id);

    return tree;
}

SegmentTree::Ptr split_mesh_medium(MeshSegment& segment, float chunk_size)
{
    if (segment.bb.longest_axis_size() < chunk_size)
    {
        return SegmentTree::Ptr(new SegmentTreeLeaf(segment));
    }

    auto& mesh = *segment.mesh;
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

    struct SplitData
    {
        int split_axis[3];
        float split_points[3][4];

        SplitData()
        {}
        SplitData(const pmp::BoundingBox& bb)
        {
            pmp::Point size = bb.max() - bb.min();
            std::vector<pmp::BoundingBox> bbs = { bb };
            for (int i = 0; i < 3; i++)
            {
                int axis = std::max_element(size.data(), size.data() + 3) - size.data();
                split_axis[i] = axis;
                size[axis] /= 2;
                std::vector<pmp::BoundingBox> new_bbs;
                for (size_t b = 0; b < bbs.size(); b++)
                {
                    split_points[i][b] = bbs[b].min()[axis] + size[axis];
                    auto split = bbs[b].split(axis, split_points[i][b]);
                    new_bbs.insert(new_bbs.end(), split.begin(), split.end());
                }
            }
        }
        uint8_t id(const pmp::Point& p) const
        {
            uint8_t ret = 0;
            size_t index = 0;
            for (int i = 0; i < 3; i++)
            {
                size_t next_index = index * 2;
                if (p[split_axis[i]] > split_points[i][index])
                {
                    ret |= (1 << i);
                    next_index++;
                }
            }
            return ret;
        }
    };

    std::vector<pmp::BoundingBox> bbs = { segment.bb };
    std::vector<SplitData> splits = { SplitData(segment.bb) };
    std::vector<bool> needs_splitting = { true };
    std::vector<SegmentTreeNode*> nodes;
    std::vector<std::pair<pmp::IndexType, SegmentTreeNode*>> leafs;
    auto root = new SegmentTreeNode();
    nodes.push_back(root);

    auto v_chunk_id = mesh.add_vertex_property<pmp::IndexType>("v:chunk_id", 0);

    for (;;)
    {
        bbs.clear();
        bbs.resize(splits.size() * 8);

        #pragma omp parallel
        {
            std::vector<pmp::BoundingBox> local_bbs(bbs.size());
            #pragma omp for schedule(dynamic,64) nowait
            for (size_t i = 0; i < mesh.vertices_size(); i++)
            {
                pmp::Vertex vH(i);
                auto& id = v_chunk_id[vH];
                if (!needs_splitting[id])
                {
                    id *= 8;
                    continue;
                }
                auto& position = mesh.position(vH);
                pmp::IndexType new_id = splits[id].id(position);
                id = id * 8 + new_id;
                local_bbs[id] += position;
            }
            #pragma omp critical
            {
                for (size_t i = 0; i < local_bbs.size(); i++)
                {
                    bbs[i] += local_bbs[i];
                }
            }
        }
        for (auto& leaf : leafs)
        {
            leaf.first *= 8;
        }

        splits.resize(bbs.size());
        needs_splitting.clear();
        needs_splitting.resize(bbs.size(), false);
        std::vector<SegmentTreeNode*> new_nodes(bbs.size(), nullptr);
        bool any_splitting = false;
        for (size_t i = 0; i < bbs.size(); i++)
        {
            auto parent = nodes[i / 8];
            if (parent == nullptr || bbs[i].is_empty())
            {
                continue;
            }

            splits[i] = SplitData(bbs[i]);
            needs_splitting[i] = bbs[i].longest_axis_size() >= chunk_size;

            if (needs_splitting[i])
            {
                new_nodes[i] = new SegmentTreeNode();
                parent->add_child(SegmentTree::Ptr(new_nodes[i]));
                any_splitting = true;
            }
            else
            {
                leafs.push_back(std::make_pair(i, parent));
            }
        }
        nodes = std::move(new_nodes);
        if (!any_splitting)
        {
            break;
        }
    }

    std::vector<pmp::IndexType> leaf_id_map(bbs.size(), pmp::PMP_MAX_INDEX);
    std::vector<pmp::BoundingBox> leaf_bbs(leafs.size());
    for (size_t id = 0; id < leafs.size(); id++)
    {
        leaf_id_map[leafs[id].first] = id;
        leaf_bbs[id] = bbs[leafs[id].first];
    }

    auto f_chunk_id = mesh.add_face_property<pmp::IndexType>("f:chunk_id", pmp::PMP_MAX_INDEX);

    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        auto v_iter = mesh.vertices(fH);
        auto id0 = v_chunk_id[*v_iter];
        auto id1 = v_chunk_id[*++v_iter];
        auto id2 = v_chunk_id[*++v_iter];
        if (id0 == id1 || id1 != id2)
        {
            f_chunk_id[fH] = leaf_id_map[id0];
        }
        else
        {
            f_chunk_id[fH] = leaf_id_map[id1];
        }
    }

    mesh.remove_vertex_property(v_chunk_id);

    mesh.add_edge_property<bool>("e:feature", false);
    auto v_feature = mesh.add_vertex_property<bool>("v:feature", false);
    auto h_original = mesh.add_halfedge_property<pmp::Halfedge>("h:original");
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.edges_size(); i++)
    {
        pmp::Edge eH(i);
        auto heH0 = eH.halfedge(0);
        auto heH1 = eH.halfedge(1);
        auto fH0 = mesh.face(heH0);
        auto fH1 = mesh.face(heH1);
        if (fH0.is_valid() && fH1.is_valid() && f_chunk_id[fH0] != f_chunk_id[fH1])
        {
            h_original[heH0] = heH0;
            h_original[heH1] = heH1;
            #pragma omp critical
            {
                v_feature[mesh.to_vertex(heH0)] = true;
                v_feature[mesh.to_vertex(heH1)] = true;
            }
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(leafs.size());
    mesh.split_mesh(meshes, f_chunk_id);

    for (size_t i = 0; i < leafs.size(); i++)
    {
        MeshSegment segment;
        segment.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[i])));
        segment.bb = leaf_bbs[i];
        leafs[i].second->add_child(SegmentTree::Ptr(new SegmentTreeLeaf(segment)));
    }

    root->update_children(2);

    return SegmentTree::Ptr(root);
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
        pmp::Face fH(i);
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
