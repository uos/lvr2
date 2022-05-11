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

namespace lvr2
{

void remove_overlapping_features(pmp::SurfaceMesh& mesh, bool print);

// SegmentTree

void SegmentTree::simplify(bool print)
{
    std::vector<std::shared_ptr<pmp::SurfaceMesh>> meshes;

    while (!combine_if_possible(print))
    {
        collect_simplifyable(meshes);

        // larger meshes tend to take longer, so have OpenMP schedule them first
        std::sort(meshes.begin(), meshes.end(), [](auto & a, auto & b)
        {
            return a->n_vertices() > b->n_vertices();
        });

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
        #pragma omp parallel for schedule(dynamic,1)
        for (size_t i = 0; i < meshes.size(); i++)
        {
            auto& mesh = *meshes[i];
            size_t old_num_vertices = mesh.n_vertices();
            pmp::SurfaceSimplification simplify(mesh, true);
            constexpr float TARGET_RATIO = 0.2;
            simplify.simplify(old_num_vertices * TARGET_RATIO);

            // TODO: remove small faces

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
        meshes.clear();

        if (print)
        {
            std::cout << "\r";
            if (!results.empty())
            {
                for (auto [ n, ratio ] : results)
                {
                    std::cout << n << "(" << ratio << "%)  ";
                }
                std::cout << "                       " << std::endl;
            }
            std::cout << timestamp << "layer complete               " << std::endl;
        }
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

    auto ret = octree_split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(), -1);
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

    node->m_skipped = combine_depth < 0 || node->m_depth > combine_depth;

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
    m_skipped = combine_depth < 0 || m_depth > combine_depth;
}
bool SegmentTreeNode::combine_if_possible(bool print)
{
    if (m_simplified)
    {
        m_meta_segment.mesh->garbage_collection();
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
        if (child->combine_if_possible(print))
        {
            count++;
        }
    }
    if (count < m_children.size())
    {
        return false;
    }

    std::vector<pmp::SurfaceMesh*> meshes;
    for (auto& child : m_children)
    {
        meshes.push_back(child->segment().mesh.get());
    }
    auto mesh = new pmp::SurfaceMesh();
    if (!m_skipped)
    {
        mesh->join_mesh(meshes);

        remove_overlapping_features(*mesh, print);

        if (!mesh->has_vertex_property("v:quadric"))
        {
            pmp::SurfaceSimplification::calculate_quadrics(*mesh);
        }
    }
    m_meta_segment.mesh.reset(mesh);
    // only return true after simplification is done
    return false;
}
void SegmentTreeNode::collect_simplifyable(std::vector<std::shared_ptr<pmp::SurfaceMesh>>& meshes)
{
    if (m_simplified)
    {
        return;
    }
    if (m_meta_segment.mesh != nullptr)
    {
        if (!m_skipped)
        {
            meshes.push_back(m_meta_segment.mesh);
        }

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

    if (m_skipped)
    {
        double sum = 0;
        for (auto& child : tile.children)
        {
            sum += child.geometricError;
        }
        tile.geometricError = (sum + 1) * 10;
    }
    else
    {
        m_meta_segment.filename = prefix + "_.b3dm";
        Cesium3DTiles::Content content;
        content.uri = m_meta_segment.filename;
        tile.content = content;
        tile.geometricError = geometric_error();
        tile.refine = Cesium3DTiles::Tile::Refine::REPLACE;
    }
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

void remove_overlapping_features(pmp::SurfaceMesh& mesh, bool print)
{
    return; // FIXME: fix this function and remove this line

    auto v_original = mesh.get_vertex_property<pmp::Vertex>("v:original");
    if (!v_original)
    {
        return;
    }
    auto f_boundary = mesh.get_face_property<bool>("f:boundary");
    auto v_merge_count = mesh.get_vertex_property<uint32_t>("v:merge_count");
    auto v_feature = mesh.get_vertex_property<bool>("v:feature");

    std::unordered_map<pmp::Vertex, std::vector<pmp::Vertex>> original_to_current;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex vH(i);
        if (!mesh.is_deleted(vH) && v_feature[vH])
        {
            #pragma omp critical
            original_to_current[v_original[vH]].push_back(vH);
        }
    }

    for (auto it = original_to_current.begin(); it != original_to_current.end();)
    {
        if (it->second.size() < 2)
        {
            it = original_to_current.erase(it);
        }
        else
        {
            ++it;
        }
    }

    std::cout << timestamp << "Found " << original_to_current.size() << " vertices to replace." << std::endl; // TODO: temp

    auto map_vertex = [&original_to_current](pmp::Vertex vH) -> pmp::Vertex
    {
        auto it = original_to_current.find(vH);
        return it != original_to_current.end() ? it->second[0] : vH;
    };

    std::unordered_map<pmp::Face, std::vector<pmp::Vertex>> deleted_faces;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face fH(i);
        if (mesh.is_deleted(fH) || !f_boundary[fH])
        {
            continue;
        }
        bool has_merged = false;
        for (auto vH : mesh.vertices(fH))
        {
            if (map_vertex(vH) != vH)
            {
                has_merged = true;
                break;
            }
        }
        if (!has_merged)
        {
            continue; // the face would not change
        }
        #pragma omp critical
        {
            auto v_it = mesh.vertices(fH);
            deleted_faces[fH] =
            {
                map_vertex(*v_it),
                map_vertex(*++v_it),
                map_vertex(*++v_it),
            };
        }
    }

    std::cout << timestamp << "Deleting " << deleted_faces.size() << " faces." << std::endl; // TODO: temp

    for (auto& f : deleted_faces)
    {
        mesh.delete_face(f.first);
    }

    std::cout << timestamp << "Removed " << deleted_faces.size() << " faces." << std::endl; // TODO: temp

    std::vector<pmp::Face> new_faces;
    for (auto& entry : deleted_faces)
    {
        try
        {
            auto new_id = mesh.add_face(entry.second);
            mesh.copy_fprops(mesh, entry.first, new_id);
            new_faces.push_back(new_id);
        }
        catch (const pmp::TopologyException&)
        {}
    }
    if (print)
    {
        std::cout << "Stitched " << original_to_current.size() << " vertices, re-adding " << deleted_faces.size() << " faces";
        if (deleted_faces.size() != new_faces.size())
        {
            std::cout << " (" << (deleted_faces.size() - new_faces.size()) << " failed)";
        }
        std::cout << std::endl;
    }

    mesh.remove_degenerate_faces();
    mesh.duplicate_non_manifold_vertices();

    for (auto& entry : original_to_current)
    {
        size_t decrement = entry.second.size() - 1;
        v_merge_count[entry.second[0]] -= decrement;
    }

    for (auto fH : new_faces)
    {
        bool boundary = false;
        for (auto vH : mesh.vertices(fH))
        {
            if (v_merge_count[vH] > 1)
            {
                boundary = true;
                break;
            }
        }
        f_boundary[fH] = boundary;
    }

    size_t cleared = 0, remaining = 0;
    for (auto& entry : original_to_current)
    {
        auto vH = entry.second[0];
        bool feature = false;
        for (auto fH : mesh.faces(vH))
        {
            if (f_boundary[fH])
            {
                feature = true;
                break;
            }
        }
        if (!feature)
        {
            v_feature[vH] = false;
            ++cleared;
        }
        else
        {
            ++remaining;
        }
    }

    if (print)
    {
        std::cout << "Cleared " << cleared << " vertices." << (remaining == 0 ? " Mesh now feature free." : "") << std::endl;
    }
    if (remaining == 0)
    {
        mesh.remove_face_property(f_boundary);
        mesh.remove_vertex_property(v_original);
        mesh.remove_vertex_property(v_merge_count);
        mesh.remove_vertex_property(v_feature);
        mesh.remove_edge_property<bool>("e:feature");
    }
}

} // namespace lvr2
