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

        ProgressBar* progress = nullptr;
        std::vector<std::pair<size_t, float>> results;

        if (print)
        {
            size_t total_count = 0;
            for (auto& m : meshes)
            {
                total_count += m->n_vertices();
            }
            progress = new ProgressBar(total_count, "Simplifying Layer");
        }
        #pragma omp parallel for schedule(dynamic,1) num_threads(num_threads)
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
                ratio = std::floor(ratio * 1000) / 10;
                if (ratio > TARGET_RATIO * 110 && mesh.n_faces() > 10000)
                {
                    #pragma omp critical
                    results.emplace_back(mesh.n_faces(), ratio);
                }
            }
        }

        if (print)
        {
            std::cout << "\r";
            if (!results.empty())
            {
                std::cout << "Simplification: " << results.size() << " / " << meshes.size()
                          << " meshes not fully simplified: [format: num_faces(reduction_ratio%)]" << std::endl;
                for (auto [ n, ratio ] : results)
                {
                    std::cout << n << "(" << ratio << "%)  ";
                }
                std::cout << "                       " << std::endl;
            }
            std::cout << timestamp << "layer complete               " << std::endl;
        }

        meshes.clear();
    }
    // print();
}
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<MeshSegment>& segments, int combine_depth)
{
    std::vector<SegmentTree*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = new SegmentTreeLeaf(segments[i]);
    }

    return octree_split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(),
                                  combine_depth);
}
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<SegmentTree::Ptr>& segments)
{
    std::vector<SegmentTree*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = segments[i].release();
    }
    segments.clear();

    int combine_depth = 0; // no combination of unrelated segments
    auto ret = octree_split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(), combine_depth);
    for (auto& ptr : temp_segments)
    {
        if (ptr != nullptr)
        {
            throw std::runtime_error("SegmentTree::octree_partition: not all segments were released");
        }
    }
    return ret;
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
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<std::pair<pmp::Point, MeshSegment>>& chunks, const Eigen::Vector3i& num_chunks, int combine_depth)
{
    std::vector<std::pair<pmp::Point, SegmentTree::Ptr>> nodes;

    for (auto [ chunk_pos, segment ] : chunks)
    {
        auto leaf = new SegmentTreeLeaf(segment);
        nodes.emplace_back(chunk_pos, SegmentTree::Ptr(leaf));
    }

    std::unordered_map<pmp::IndexType, SegmentTreeNode*> parents;
    Eigen::Vector3i parent_num_chunks = num_chunks / 2;

    while (nodes.size() != 1)
    {
        parent_num_chunks = parent_num_chunks.cwiseMax(1);
        for (auto& [ chunk_pos, node ] : nodes)
        {
            auto parent_id = chunk_index(chunk_pos / 2, 1, parent_num_chunks);
            auto parent = parents.find(parent_id);
            if (parent == parents.end())
            {
                parent = parents.emplace(parent_id, new SegmentTreeNode()).first;
            }
            parent->second->add_child(std::move(node));
        }
        nodes.clear();
        for (auto [ chunk_id, node ] : parents)
        {
            auto pos = chunk_position(chunk_id, 1, parent_num_chunks);
            if (node->num_children() == 1)
            {
                nodes.emplace_back(pos, std::move(node->children()[0]));
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
            nodes.emplace_back(pos, SegmentTree::Ptr(node));
        }
        parents.clear();
        parent_num_chunks /= 2;
    }
    auto& ret = nodes[0].second;
    ret->update_children(combine_depth);
    return std::move(ret);
}

// SegmentTreeNode

void SegmentTreeNode::add_child(SegmentTree::Ptr child)
{
    m_meta_segment.bb += child->segment().bb;
    m_meta_segment.texture_file = child->segment().texture_file;
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
        m_meta_segment.texture_file = child->segment().texture_file;
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

    std::vector<pmp::Point> positions(candidates.size());
    std::vector<size_t> merged_into(candidates.size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < candidates.size(); i++)
    {
        positions[i] = mesh.position(candidates[i]);
        merged_into[i] = i;
    }

    // this would ideally be done with a kd-tree, but there is currently no index-preserving kd-tree
    // implementation and making one just for this is not worth the effort

    float max_dist_sq = max_distance * max_distance;
    #pragma omp parallel for
    for (size_t i = 1; i < candidates.size(); i++)
    {
        for (size_t j = 0; j < i; j++)
        {
            float dist_sq = (positions[i] - positions[j]).squaredNorm();
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

    auto map_vertex = [&merge_map](pmp::Vertex vH) -> pmp::Vertex
    {
        auto it = merge_map.find(vH);
        return it != merge_map.end() ? it->second : vH;
    };

    std::vector<std::tuple<pmp::Face, std::vector<pmp::Vertex>, std::vector<pmp::Point>>> deleted_faces;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        if (mesh.is_deleted(fH))
        {
            continue;
        }
        bool has_merged = false;
        for (auto vH : mesh.vertices(fH))
        {
            has_merged = has_merged || map_vertex(vH) != vH;
        }
        if (!has_merged)
        {
            continue; // the face would not change
        }
        #pragma omp critical
        {
            auto& [ id, vert, pos ] = deleted_faces.emplace_back();
            id = fH;
            vert.reserve(3);
            pos.reserve(3);
            for (auto vH : mesh.vertices(fH))
            {
                vert.push_back(vH);
                pos.push_back(mesh.position(vH));
            }
        }
    }

    for (auto& f : deleted_faces)
    {
        mesh.delete_face(std::get<0>(f));
    }

    for (auto& [ src, target ] : merge_map)
    {
        v_feature[target] = false;
        if (!mesh.is_deleted(src))
        {
            std::cout << "ERROR: Vertex " << src << " not deleted! ";
            std::cout << "valence: " << mesh.valence(src);
            std::cout << std::endl;
        }
    }

    size_t added = 0;
    auto fprop_map = mesh.gen_fprop_map(mesh);
    auto vprop_map = mesh.gen_vprop_map(mesh);
    for (auto& [ fH, vertices, positions ] : deleted_faces)
    {
        for (size_t i = 0; i < 3; i++)
        {
            auto& vH = vertices[i];
            vH = map_vertex(vH);
            if (mesh.is_deleted(vH))
            {
                auto new_vH = mesh.add_vertex(positions[i]);
                mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
                merge_map[vH] = new_vH;
                vH = new_vH;
            }
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
