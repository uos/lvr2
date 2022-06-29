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
 * SegmentTree.cpp
 *
 * @date   29.03.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "SegmentTree.hpp"
#include "lvr2/algorithm/pmp/SurfaceSimplification.h"
#include "lvr2/config/lvropenmp.hpp"

namespace lvr2
{

void remove_overlapping_features(pmp::SurfaceMesh& mesh, float max_distance, bool print);

// SegmentTree

void SegmentTree::simplify(std::shared_ptr<HighFive::File> mesh_file, float max_merge_dist, bool print)
{
    std::vector<MeshSegment::Inner> meshes;
    size_t num_threads = mesh_file ? 1 : OpenMPConfig::getNumThreads();

    while (!combine_if_possible(mesh_file, max_merge_dist, print))
    {
        collect_simplifyable(meshes);
        // sort meshes descending by number of vertices for better omp scheduling
        std::sort(meshes.begin(), meshes.end(), [](const auto & a, const auto & b)
        {
            return a->n_vertices() > b->n_vertices();
        });

        ProgressBar* progress = nullptr;
        size_t fully_simplified = 0;

        if (print)
        {
            size_t total_count = 0;
            for (auto& m : meshes)
            {
                total_count += m->n_vertices();
            }
            progress = new ProgressBar(total_count, "Simplifying Layer");
        }
        #pragma omp parallel for schedule(dynamic,1) num_threads(num_threads) reduction(+:fully_simplified)
        for (size_t i = 0; i < meshes.size(); i++)
        {
            auto pmp_mesh = meshes[i]->get();
            auto& mesh = pmp_mesh->getSurfaceMesh();
            size_t old_num_vertices = mesh.n_vertices();
            pmp::SurfaceSimplification simplify(mesh, true);
            constexpr float TARGET_RATIO = 0.2;
            simplify.simplify(old_num_vertices * TARGET_RATIO);

            // TODO: remove small faces

            mesh.garbage_collection();
            pmp_mesh->changed();

            if (print)
            {
                *progress += old_num_vertices;
                float ratio = (float)mesh.n_vertices() / old_num_vertices;
                if (ratio > TARGET_RATIO * 1.1)
                {
                    fully_simplified++;
                }
            }
        }

        if (print)
        {
            std::cout << "\r";
            if (fully_simplified > 0)
            {
                std::cout << timestamp << "Simplification: " << fully_simplified << " / " << meshes.size()
                          << " meshes reached simplification limit" << std::endl;
            }
            std::cout << timestamp << "layer complete               " << std::endl;
        }

        meshes.clear();
    }
    // print();
}
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<MeshSegment>& segments, int combine_depth)
{
    std::vector<SegmentTree*> ptrs(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        ptrs[i] = new SegmentTreeLeaf(segments[i]);
    }

    return octree_split_recursive(ptrs.data(), ptrs.data() + ptrs.size(), combine_depth);
}
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<SegmentTree::Ptr>& segments)
{
    std::vector<SegmentTree*> ptrs(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        ptrs[i] = segments[i].release();
    }
    segments.clear();

    int combine_depth = 0; // no combination of unrelated segments
    return octree_split_recursive(ptrs.data(), ptrs.data() + ptrs.size(), combine_depth);
}
SegmentTree::Ptr SegmentTree::octree_split_recursive(SegmentTree** start, SegmentTree** end, int combine_depth)
{
    size_t n = end - start;

    SegmentTreeNode* node = new SegmentTreeNode();

    if (n <= 8)
    {
        for (size_t i = 0; i < n; i++)
        {
            node->add_child(SegmentTree::Ptr(start[i]));
            start[i] = nullptr;
        }
    }
    else
    {
        auto split_fn = [](int axis)
        {
            return [axis](SegmentTree * a, SegmentTree * b)
            {
                return a->segment().bb.center()[axis] < b->segment().bb.center()[axis];
            };
        };

        SegmentTree** starts[9];
        starts[0] = start;
        starts[8] = end; // fake past-the-end start for easier indexing

        for (size_t axis = 0; axis < 3; axis++)
        {
            size_t step = 1 << (3 - axis); // values 8 -> 4 -> 2
            for (size_t i = 0; i < 8; i += step)
            {
                auto& a = starts[i];
                auto& b = starts[i + step];
                auto& mid = starts[i + step / 2];
                mid = a + (b - a) / 2;
                std::nth_element(a, mid, b, split_fn(axis));
            }
        }

        for (size_t i = 0; i < 8; i++)
        {
            node->add_child(octree_split_recursive(starts[i], starts[i + 1], combine_depth));
        }
    }

    node->m_skipped = combine_depth >= 0 && node->m_depth > combine_depth;

    return SegmentTree::Ptr(node);
}
SegmentTree::Ptr SegmentTree::octree_partition(std::unordered_map<Vector3i, MeshSegment>& chunks, int combine_depth)
{
    Vector3i min = Vector3i::Constant(std::numeric_limits<Vector3i::value_type>::max());
    for (auto& [ chunk_pos, _ ] : chunks)
    {
        min = min.cwiseMin(chunk_pos);
    }

    std::unordered_map<Vector3i, SegmentTree::Ptr> nodes;
    nodes.reserve(chunks.size());
    for (auto& [ chunk_pos, segment ] : chunks)
    {
        auto leaf = new SegmentTreeLeaf(segment);
        nodes[chunk_pos - min].reset(leaf);
    }

    std::unordered_map<Vector3i, SegmentTreeNode*> parents;

    while (nodes.size() != 1)
    {
        for (auto& [ chunk_pos, node ] : nodes)
        {
            Vector3i parent_pos = chunk_pos / 2;
            auto parent = parents.find(parent_pos);
            if (parent == parents.end())
            {
                parent = parents.emplace(parent_pos, new SegmentTreeNode()).first;
            }
            parent->second->add_child(std::move(node));
        }
        nodes.clear();
        for (auto [ chunk_pos, node ] : parents)
        {
            if (node->num_children() == 1)
            {
                nodes[chunk_pos] = std::move(node->children()[0]);
                delete node;
                continue;
            }

            size_t child_child_count = 0;
            for (auto& child : node->children())
            {
                child_child_count += child->num_children();
            }
            if (child_child_count <= 8)
            {
                // sparse node -> collapse one layer
                std::vector<SegmentTree::Ptr> new_children;
                for (auto& child : node->children())
                {
                    if (child->is_leaf())
                    {
                        new_children.emplace_back(std::move(child));
                    }
                    else
                    {
                        auto& child_children = dynamic_cast<SegmentTreeNode*>(child.get())->children();
                        new_children.insert(new_children.end(), std::move_iterator(child_children.begin()), std::move_iterator(child_children.end()));
                    }
                }
                node->children().swap(new_children);
            }
            nodes[chunk_pos].reset(node);
        }
        parents.clear();
    }
    auto& ret = nodes.begin()->second;
    ret->update_children(combine_depth);
    return std::move(ret);
}

// SegmentTreeNode

void SegmentTreeNode::add_child(SegmentTree::Ptr child)
{
    m_meta_segment.bb += child->segment().bb;
    m_depth = std::max(m_depth, child->m_depth + 1);
    m_children.push_back(std::move(child));
}
void SegmentTreeNode::update_children(int combine_depth)
{
    m_meta_segment.bb = pmp::BoundingBox();
    m_depth = 0;
    for (auto& child : m_children)
    {
        child->update_children(combine_depth);
        m_meta_segment.bb += child->segment().bb;
        m_depth = std::max(m_depth, child->m_depth + 1);
    }
    m_skipped = combine_depth >= 0 && m_depth > combine_depth;
}
bool SegmentTreeNode::combine_if_possible(const std::shared_ptr<HighFive::File>& mesh_file, float max_merge_dist, bool print)
{
    if (m_simplified)
    {
        return true;
    }
    if (m_meta_segment.mesh != nullptr)
    {
        // done combining, but not simplified
        return false;
    }
    size_t count = 0;
    for (auto& child : m_children)
    {
        if (child->combine_if_possible(mesh_file, max_merge_dist, print))
        {
            count++;
        }
    }
    if (count < m_children.size())
    {
        return false;
    }
    if (m_skipped)
    {
        m_simplified = true;
        return true;
    }

    std::vector<std::shared_ptr<PMPMesh<BaseVector<float>>>> pmp_meshes;
    std::vector<pmp::SurfaceMesh*> meshes;
    for (auto& child : m_children)
    {
        pmp_meshes.push_back(child->segment().mesh->get());
        meshes.push_back(&pmp_meshes.back()->getSurfaceMesh());
    }
    PMPMesh<BaseVector<float>> pmp_mesh;
    auto& mesh = pmp_mesh.getSurfaceMesh();
    mesh.join_mesh(meshes);

    remove_overlapping_features(mesh, max_merge_dist, false);

    if (!mesh.has_vertex_property("v:quadric"))
    {
        pmp::SurfaceSimplification::calculate_quadrics(mesh);
    }
    m_meta_segment.mesh.reset(new LazyMesh(pmp_mesh, mesh_file));
    // only return true after simplification is done
    return false;
}
void SegmentTreeNode::collect_simplifyable(std::vector<MeshSegment::Inner>& meshes)
{
    if (m_simplified)
    {
        return;
    }
    if (m_meta_segment.mesh != nullptr)
    {
        meshes.push_back(m_meta_segment.mesh);
        m_simplified = true;
    }
    else
    {
        for (auto& child : m_children)
        {
            child->collect_simplifyable(meshes);
        }
    }
}
void SegmentTreeNode::fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix)
{
    if (m_finalized)
    {
        return;
    }
    auto prefix = m_meta_segment.filename.empty() ? filename_prefix : m_meta_segment.filename;
    tile.children.resize(m_children.size());
    for (size_t i = 0; i < m_children.size(); i++)
    {
        m_children[i]->fill_tile(tile.children[i], prefix + std::to_string(i));
    }

    if (!m_skipped)
    {
        m_meta_segment.filename = prefix + "_.b3dm";
        Cesium3DTiles::Content content;
        content.uri = m_meta_segment.filename;
        tile.content = content;
        tile.refine = Cesium3DTiles::Tile::Refine::REPLACE;
    }
    tile.geometricError = geometric_error();
    convert_bounding_box(m_meta_segment.bb, tile.boundingVolume);

    m_finalized = true;
}
void SegmentTreeNode::collect_segments(std::vector<MeshSegment>& segments)
{
    if (!m_skipped)
    {
        segments.push_back(m_meta_segment);
    }
    for (auto& child : m_children)
    {
        child->collect_segments(segments);
    }
}
void SegmentTreeNode::print(size_t indent)
{
    std::cout << std::string(indent, ' ') << "Node";
    if (!m_meta_segment.filename.empty())
    {
        std::cout << " " << m_meta_segment.filename;
    }
    if (m_meta_segment.mesh != nullptr)
    {
        std::cout << "(" << m_meta_segment.mesh->n_faces() << ")";
    }
    std::cout << std::endl;
    for (auto& child : m_children)
    {
        child->print(indent + 2);
    }
}

// SegmentTreeLeaf

void SegmentTreeLeaf::print(size_t indent)
{
    std::cout << std::string(indent, ' ') << "Leaf";
    if (!m_segment.filename.empty())
    {
        std::cout << " " << m_segment.filename;
    }
    if (m_segment.mesh != nullptr)
    {
        std::cout << "(" << m_segment.mesh->n_faces() << ")";
    }
    std::cout << std::endl;
}

// other functions

void remove_overlapping_features(pmp::SurfaceMesh& mesh, float max_distance, bool print)
{
    auto v_feature = mesh.get_vertex_property<bool>("v:feature");
    if (!v_feature)
    {
        return;
    }

    std::vector<pmp::Vertex> candidates;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex vH(i);
        if (!mesh.is_deleted(vH) && v_feature[vH])
        {
            #pragma omp critical
            candidates.push_back(vH);
        }
    }

    std::vector<size_t> merged_into(candidates.size());
    for (size_t i = 0; i < candidates.size(); i++)
    {
        merged_into[i] = i;
    }

    // this would ideally be done with a kd-tree, but there is currently no index-preserving kd-tree
    // implementation and making one just for this is not worth the effort

    float max_dist_sq = max_distance * max_distance;
    #pragma omp parallel for
    for (size_t i = 1; i < candidates.size(); i++)
    {
        auto& pos = mesh.position(candidates[i]);
        for (size_t j = 0; j < i; j++)
        {
            float dist_sq = (pos - mesh.position(candidates[j])).squaredNorm();
            if (dist_sq < max_dist_sq)
            {
                merged_into[i] = j;
                break;
            }
        }
    }

    std::unordered_map<pmp::Vertex, pmp::Vertex> merge_map;
    #pragma omp parallel for
    for (size_t i = 0; i < candidates.size(); i++)
    {
        if (merged_into[i] != i)
        {
            size_t target = merged_into[i];
            while (merged_into[target] != target)
            {
                target = merged_into[target];
            }
            #pragma omp critical
            merge_map[candidates[i]] = candidates[target];
        }
    }

    if (merge_map.empty())
    {
        return;
    }

    auto fprop_map = mesh.gen_fprop_map(mesh);
    auto vprop_map = mesh.gen_vprop_map(mesh);

    auto map_vertex = [&merge_map](pmp::Vertex vH) -> pmp::Vertex
    {
        auto it = merge_map.find(vH);
        return it != merge_map.end() ? it->second : vH;
    };

    std::vector<pmp::Face> feature_faces;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        if (mesh.is_deleted(fH))
        {
            continue;
        }
        for (auto vH : mesh.vertices(fH))
        {
            if (map_vertex(vH) != vH)
            {
                #pragma omp critical
                feature_faces.push_back(fH);
                break;
            }
        }
    }

    std::unordered_map<pmp::Vertex, pmp::Point> vertex_positions;
    std::unordered_map<pmp::Face, std::vector<pmp::Vertex>> deleted_faces;
    for (auto fH : feature_faces)
    {
        auto& vertices = deleted_faces[fH];
        vertices.reserve(3);
        for (auto vH : mesh.vertices(fH))
        {
            vH = map_vertex(vH);
            vertices.push_back(vH);
            vertex_positions[vH] = mesh.position(vH);
        }
    }

    for (auto& [ fH, vertices ] : deleted_faces)
    {
        mesh.delete_face(fH);
    }

    for (auto& [ src, target ] : merge_map)
    {
        v_feature[target] = false;
        if (!mesh.is_deleted(src))
        {
            if (mesh.valence(src) == 0)
            {
                mesh.delete_vertex(src);
            }
            else
            {
                std::cout << "Warning: Vertex " << src << " should have been deleted, but it is not." << std::endl;
            }
        }
    }

    merge_map.clear();

    for (auto& [ vH, pos ] : vertex_positions)
    {
        if (mesh.is_deleted(vH))
        {
            auto new_vH = mesh.add_vertex(pos);
            mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
            merge_map[vH] = new_vH;
        }
        else
        {
            merge_map[vH] = vH;
        }
    }

    size_t added = 0;
    for (auto& [ fH, vertices ] : deleted_faces)
    {
        for (auto& vH : vertices)
        {
            vH = merge_map[vH];
        }
        if (vertices[0] == vertices[1] || vertices[0] == vertices[2] || vertices[1] == vertices[2])
        {
            // side effect of merging based on a distance threshold: some faces become degenerate
            continue;
        }
        try
        {
            auto new_fH = mesh.add_face(vertices);
            mesh.copy_fprops(mesh, fH, new_fH, fprop_map);
            added++;
        }
        catch (const pmp::TopologyException&)
        {}
    }

    mesh.remove_degenerate_faces(print);
    mesh.duplicate_non_manifold_vertices(print);

    if (print)
    {
        std::cout << timestamp << "Stitched " << merge_map.size() << " boundary vertices, re-adding " << added << " faces" << std::endl;
    }

    mesh.garbage_collection();
}

} // namespace lvr2
