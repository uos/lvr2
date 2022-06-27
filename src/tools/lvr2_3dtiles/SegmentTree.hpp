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
 * SegmentTree.hpp
 *
 * @date   29.03.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "CesiumPmpInterop.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/LazyMesh.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/types/MatrixTypes.hpp"

namespace lvr2
{

struct MeshSegment
{
    using Mesh = LazyMesh<BaseVector<float>>;
    using Inner = std::shared_ptr<Mesh>;

    Inner mesh = nullptr;
    pmp::BoundingBox bb;
    std::string filename = "";
    /// Texture as png encoded bytes
    std::shared_ptr<Texture> texture;
};

class SegmentTree
{
public:
    using Ptr = std::unique_ptr<SegmentTree>;

    static Ptr octree_partition(std::vector<MeshSegment>& segments, int combine_depth = -1);
    static Ptr octree_partition(std::vector<SegmentTree::Ptr>& segments);
    static Ptr octree_partition(std::unordered_map<Vector3i, MeshSegment>& chunks, int combine_depth = -1);
    void simplify(std::shared_ptr<HighFive::File> mesh_file, float max_merge_dist, bool print = true);
    virtual void print(size_t indent = 0) = 0;
    virtual void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) = 0;
    virtual void update_children(int combine_depth = -1) = 0;
    virtual void collect_segments(std::vector<MeshSegment>& segments) = 0;

    virtual bool combine_if_possible(const std::shared_ptr<HighFive::File>& mesh_file, float max_merge_dist, bool print) = 0;
    virtual void collect_simplifyable(std::vector<MeshSegment::Inner>& meshes) = 0;
    virtual MeshSegment& segment() = 0;
    virtual bool is_leaf() = 0;
    virtual size_t num_children() = 0;

    size_t m_depth = 0;
    bool m_skipped = false;
    bool m_simplified = false;
    bool m_finalized = false;

protected:
    /**
     * @brief @see Cesium3DTiles::Tile::geometricError
     * 
     * Gives a metric for how necessary it is to replace a tile with its children.
     * Returns 0 for leaf nodes because they cannot be replaced and higher values
     * for higher nodes.
     */
    double geometric_error() const
    {
        return m_depth == 0 ? 0.0 : std::pow(10, m_depth - 1);
    }
private:
    static Ptr octree_split_recursive(SegmentTree** start, SegmentTree** end, int combine_depth);
};

class SegmentTreeNode : public SegmentTree
{
public:
    void add_child(SegmentTree::Ptr child);
    size_t num_children() const
    {
        return m_children.size();
    }
    std::vector<SegmentTree::Ptr>& children()
    {
        return m_children;
    }

    void print(size_t indent = 0) override;
    void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) override;
    void update_children(int combine_depth = -1) override;
    void collect_segments(std::vector<MeshSegment>& segments) override;

    bool combine_if_possible(const std::shared_ptr<HighFive::File>& mesh_file, float max_merge_dist, bool print) override;
    void collect_simplifyable(std::vector<MeshSegment::Inner>& meshes) override;
    MeshSegment& segment() override
    {
        return m_meta_segment;
    }
    bool is_leaf() override
    {
        return false;
    }
    size_t num_children() override
    {
        return m_children.size();
    }

private:
    MeshSegment m_meta_segment;
    std::vector<SegmentTree::Ptr> m_children;
};
class SegmentTreeLeaf : public SegmentTree
{
public:
    SegmentTreeLeaf(const MeshSegment& segment)
        : m_segment(segment)
    {
        m_simplified = true;
    }
    void print(size_t indent = 0) override;
    void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) override
    {
        if (m_finalized)
        {
            return;
        }
        auto prefix = m_segment.filename.empty() ? filename_prefix : m_segment.filename;
        m_segment.filename = prefix + ".b3dm";
        Cesium3DTiles::Content content;
        content.uri = m_segment.filename;
        tile.content = content;
        tile.geometricError = geometric_error();
        convert_bounding_box(m_segment.bb, tile.boundingVolume);

        m_finalized = true;
    }
    void update_children(int combine_depth = -1) override
    {}
    void collect_segments(std::vector<MeshSegment>& segments) override
    {
        segments.push_back(m_segment);
    }

    bool combine_if_possible(const std::shared_ptr<HighFive::File>& mesh_file, float max_merge_dist, bool print) override
    {
        return true;
    }
    void collect_simplifyable(std::vector<MeshSegment::Inner>& meshes) override
    {
        m_simplified = true;
    }
    virtual MeshSegment& segment() override
    {
        return m_segment;
    }
    bool is_leaf() override
    {
        return true;
    }
    size_t num_children() override
    {
        return 1;
    }

private:
    MeshSegment m_segment;
};

} // namespace lvr2