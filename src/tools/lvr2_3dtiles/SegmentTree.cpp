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

// SegmentTree

void SegmentTree::simplify(bool print)
{
    while (!combine_if_possible(print))
    {
        #pragma omp parallel
        #pragma omp single
        simplify_if_possible(print);

        if (print)
        {
            std::cout << "layer complete" << std::endl;
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
        if (!mesh->has_vertex_property("v:quadric"))
        {
            pmp::SurfaceSimplification::calculate_quadrics(*mesh);
        }
    }
    m_meta_segment.mesh.reset(mesh);
    // only return true after simplification is done
    return false;
}
void SegmentTreeNode::simplify_if_possible(bool print)
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

            // TODO: recombine feature vertices

            float ratio = (float)mesh.n_vertices() / old_num_vertices;
            ratio = std::floor(ratio * 1000) / 10;
            if (print)
            {
                std::cout << mesh.n_faces() << '(' << ratio << "%) " << std::flush;
            }
        }

        m_simplified = true;
    }
    else
    {
        for (size_t i = 0; i < m_children.size(); i++)
        {
            #pragma omp task
            m_children[i]->simplify_if_possible(print);
        }
        #pragma omp taskwait
    }
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

} // namespace lvr2
