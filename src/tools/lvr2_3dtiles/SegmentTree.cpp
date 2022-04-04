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
    while (!combine_if_possible(print))
    {
        ProgressBar* progress = nullptr;
        std::vector<std::pair<size_t, float>> results;
        if (print)
        {
            size_t total_count = count_simplifyable();
            progress = new ProgressBar(total_count, "Simplifying Layer");
        }

        #pragma omp parallel shared(progress, results)
        #pragma omp single
        simplify_if_possible(progress, results);

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
            std::cout << "layer complete               " << std::endl;
        }
    }
    // print();
}
SegmentTree::Ptr SegmentTree::octree_partition(std::vector<MeshSegment>& segments, int combine_depth)
{
    std::vector<MeshSegment*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = &segments[i];
    }

    return octree_split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(),
                                  combine_depth);
}
void one_split(MeshSegment** starts[9], size_t a, size_t b)
{
    pmp::BoundingBox bb;
    for (auto it = starts[a]; it != starts[b]; ++it)
    {
        bb += (*it)->bb;
    }

    size_t mid = (a + b) / 2;
    starts[mid] = starts[a] + (starts[b] - starts[a]) / 2;
    size_t axis = bb.longest_axis();

    std::nth_element(starts[a], starts[mid], starts[b], [axis](auto a, auto b)
    {
        return a->bb.center()[axis] < b->bb.center()[axis];
    });

    if (mid - a > 1)
    {
        one_split(starts, a, mid);
        one_split(starts, mid, b);
    }
}
SegmentTree::Ptr SegmentTree::octree_split_recursive(MeshSegment** start, MeshSegment** end, int combine_depth)
{
    size_t n = end - start;

    SegmentTreeNode* node = new SegmentTreeNode();

    if (n <= 8)
    {
        for (size_t i = 0; i < n; i++)
        {
            node->add_child(SegmentTree::Ptr(new SegmentTreeLeaf(*start[i])));
        }
    }
    else
    {
        auto split_fn = [](int axis)
        {
            return [axis](const MeshSegment * a, const MeshSegment * b)
            {
                return a->bb.center()[axis] < b->bb.center()[axis];
            };
        };

        MeshSegment** starts[9];
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

    if (combine_depth > 0)
    {
        node->m_skipped = node->m_depth > combine_depth;
    }

    return SegmentTree::Ptr(node);
}

// SegmentTreeNode

void SegmentTreeNode::add_child(SegmentTree::Ptr child)
{
    m_meta_segment.bb += child->segment().bb;
    m_depth = std::max(m_depth, child->m_depth + 1);
    m_children.push_back(std::move(child));
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
        meshes.back()->garbage_collection();
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
void SegmentTreeNode::simplify_if_possible(ProgressBar* progress, std::vector<std::pair<size_t, float>>& results)
{
    if (m_simplified)
    {
        return;
    }
    if (m_meta_segment.mesh != nullptr)
    {
        if (!m_skipped)
        {
            auto& mesh = *m_meta_segment.mesh;
            size_t old_num_vertices = mesh.n_vertices();
            pmp::SurfaceSimplification simplify(mesh, true);
            constexpr float TARGET_RATIO = 0.2;
            simplify.simplify(old_num_vertices * TARGET_RATIO);

            if (progress != nullptr)
            {
                ++(*progress);
                float ratio = (float)mesh.n_vertices() / old_num_vertices;
                ratio = std::floor(ratio * 1000) / 10;
                if (ratio > TARGET_RATIO * 110 && mesh.n_faces() > 10000)
                {
                    #pragma omp critical
                    results.emplace_back(mesh.n_faces(), ratio);
                }
            }
        }

        m_simplified = true;
    }
    else
    {
        for (size_t i = 0; i < m_children.size(); i++)
        {
            #pragma omp task shared(progress, results)
            m_children[i]->simplify_if_possible(progress, results);
        }
        #pragma omp taskwait
    }
}
size_t SegmentTreeNode::count_simplifyable()
{
    if (m_simplified)
    {
        return 0;
    }
    size_t ret = 0;
    if (m_meta_segment.mesh != nullptr)
    {
        if (!m_skipped)
        {
            ret = 1;
        }
    }
    else
    {
        for (auto& child : m_children)
        {
            ret += child->count_simplifyable();
        }
    }
    return ret;
}
void SegmentTreeNode::fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix)
{
    if (m_finalized)
    {
        return;
    }
    tile.children.resize(m_children.size());
    for (size_t i = 0; i < m_children.size(); i++)
    {
        m_children[i]->fill_tile(tile.children[i], filename_prefix + std::to_string(i));
    }

    if (m_skipped)
    {
        tile.geometricError = 0;
        for (auto& child : tile.children)
        {
            tile.geometricError += child.geometricError;
        }
    }
    else
    {
        m_meta_segment.filename = filename_prefix + "_.b3dm";
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
    auto h_original = mesh.get_halfedge_property<pmp::Edge>("h:original");
    if (!h_original)
    {
        return;
    }
    std::unordered_map<pmp::Edge, std::pair<pmp::Halfedge, pmp::Halfedge>> matches;
    #pragma omp parallel for schedule(dynamic,64)
    for (size_t i = 0; i < mesh.halfedges_size(); i++)
    {
        pmp::Halfedge heH(i);
        if (mesh.is_deleted(heH) || !h_original[heH].is_valid())
        {
            continue;
        }
        if (!mesh.is_boundary(mesh.opposite_halfedge(heH)))
        {
            throw std::runtime_error("boundary stitching: Marked non-boundary edge as stitch boundary.");
        }
        #pragma omp critical
        {
            auto& entry = matches[h_original[heH]];
            if (heH.idx() < mesh.opposite_halfedge(heH).idx())
            {
                entry.first = heH;
            }
            else
            {
                entry.second = heH;
            }
        }
    }
    size_t stitch_count = 0;
    for (auto match : matches)
    {
        auto heH0 = match.second.first;
        auto heH1 = match.second.second;
        if (heH0.is_valid() && heH1.is_valid())
        {
            if (mesh.is_deleted(heH0) || mesh.is_deleted(heH1) || !h_original[heH0].is_valid() || h_original[heH0] != h_original[heH1])
            {
                throw std::runtime_error("boundary stitching: Deleted or incorrect match.");
            }
            h_original[heH0] = pmp::Edge();
            h_original[heH1] = pmp::Edge();
            mesh.stitch_boundary(mesh.opposite_halfedge(heH0), mesh.opposite_halfedge(heH1));
            stitch_count++;
        }
    }
    if (print)
    {
        std::cout << "Stitched " << stitch_count << " edges.  " << std::flush;
    }

    auto v_feature = mesh.get_vertex_property<bool>("v:feature");
    size_t cleared = 0, remaining = 0;
    #pragma omp parallel reduction(+:remaining)
    {
        std::vector<pmp::Vertex> non_feature_vertices;
        #pragma omp for schedule(dynamic,64) nowait
        for (size_t i = 0; i < mesh.vertices_size(); i++)
        {
            pmp::Vertex vH(i);
            if (mesh.is_deleted(vH) || !v_feature[vH])
            {
                continue;
            }
            bool feature = false;
            for (auto heH : mesh.halfedges(vH))
            {
                if (h_original[heH].is_valid() || h_original[mesh.opposite_halfedge(heH)].is_valid())
                {
                    feature = true;
                    remaining++;
                    break;
                }
            }
            if (!feature)
            {
                non_feature_vertices.push_back(vH);
            }
        }
        #pragma omp critical
        {
            cleared += non_feature_vertices.size();
            for (auto vH : non_feature_vertices)
            {
                v_feature[vH] = false;
            }
        }
    }
    std::cout << "Cleared " << cleared << " vertices." << (remaining == 0 ? " Mesh now feature free." : "") << std::endl;
    if (remaining == 0)
    {
        mesh.remove_halfedge_property(h_original);
        mesh.remove_vertex_property(v_feature);
        mesh.remove_edge_property<bool>("e:feature");
    }

    mesh.duplicate_non_manifold_vertices();
    mesh.remove_degenerate_faces();

}

} // namespace lvr2
