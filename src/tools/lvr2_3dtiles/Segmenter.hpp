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
 * Segmenter.hpp
 *
 * @date   03.02.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/PMPMesh.hpp"

#include <Cesium3DTiles/Tile.h>
#include <boost/filesystem.hpp>

namespace lvr2
{

inline void convert_bounding_box(const pmp::BoundingBox& in, Cesium3DTiles::BoundingVolume& out)
{
    auto center = in.center();
    auto half_vector = in.max() - center;
    out.box =
    {
        center.x(), center.y(), center.z(),
        half_vector.x(), 0, 0,
        0, half_vector.y(), 0,
        0, 0, half_vector.z()
    };
}

typedef pmp::IndexType SegmentId;
constexpr SegmentId INVALID_SEGMENT = pmp::PMP_MAX_INDEX;

struct MeshSegment
{
    std::shared_ptr<pmp::SurfaceMesh> mesh = nullptr;
    pmp::BoundingBox bb;
    std::string filename = "";
};

class SegmentTree
{
public:
    using Ptr = std::unique_ptr<SegmentTree>;

    static Ptr octree_partition(std::vector<MeshSegment>& segments, int combine_depth = -1);
    void simplify(bool print = true);
    virtual void print(size_t indent = 0) = 0;
    virtual void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) = 0;
    virtual void collect_segments(std::vector<MeshSegment>& segments) = 0;

    virtual bool combine_if_possible(bool print) = 0;
    virtual void simplify_if_possible(bool print) = 0;
    virtual MeshSegment& segment() = 0;

    size_t m_depth = 0;
    bool m_skipped = false;
    bool m_simplified = false;
    bool m_finalized = false;

protected:
    double geometric_error() const
    {
        return m_depth == 0 ? 0.0 : std::pow(10, m_depth - 1);
    }
private:
    static Ptr octree_split_recursive(MeshSegment** start, MeshSegment** end, int combine_depth);
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
    void collect_segments(std::vector<MeshSegment>& segments) override;
    void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) override;

    bool combine_if_possible(bool print) override;
    void simplify_if_possible(bool print) override;
    virtual MeshSegment& segment() override
    {
        return m_meta_segment;
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
    {}
    void print(size_t indent = 0) override;
    void collect_segments(std::vector<MeshSegment>& segments) override
    {
        segments.push_back(m_segment);
    }
    void fill_tile(Cesium3DTiles::Tile& tile, const std::string& filename_prefix) override
    {
        if (m_finalized)
        {
            return;
        }
        m_segment.filename = filename_prefix + ".b3dm";
        Cesium3DTiles::Content content;
        content.uri = m_segment.filename;
        tile.content = content;
        tile.geometricError = geometric_error();
        convert_bounding_box(m_segment.bb, tile.boundingVolume);

        m_finalized = true;
    }

    bool combine_if_possible(bool print) override
    {
        return true;
    }
    void simplify_if_possible(bool print) override
    {
        m_simplified = true;
    }
    virtual MeshSegment& segment() override
    {
        return m_segment;
    }

private:
    MeshSegment m_segment;
};


/**
 * @brief creates a new value at the end of a vector and returns its index and a reference to that value
 *
 * @param vec the vector to append to
 * @return std::pair<size_t, T&> the index and reference to the new value
 */
template<typename T>
inline std::pair<size_t, T&> push_and_get_index(std::vector<T>& vec, T&& value = T())
{
    size_t index = vec.size();
    vec.push_back(std::move(value));
    return std::make_pair(index, std::ref(vec.back()));
}

/**
 * @brief partitions all connected regions of a mesh, and bundles small segments into chunks
 *
 * Adds a face property called "f:segment" to the mesh containing the SegmentId of the face.
 *
 * @param input_mesh the mesh to partition
 * @param small_segments a vector to store the segments in
 * @param large_segments a vector to store extracted segments in
 * @param chunk_size the size of a chunk to determine and bundle small segments
 */
void segment_mesh(pmp::SurfaceMesh& input_mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  std::vector<MeshSegment>& chunks,
                  std::vector<MeshSegment>& large_segments);

SegmentTree::Ptr split_mesh(MeshSegment& segment, float chunk_size);

} // namespace lvr2
