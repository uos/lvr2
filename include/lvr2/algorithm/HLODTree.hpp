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
 * HLODTree.hpp
 *
 * @date   29.06.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "lvr2/geometry/LazyMesh.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/util/Progress.hpp"

#include <memory>

namespace lvr2
{

/**
 * @brief A class for Hierarchical Level of Detail (HLOD) trees.
 *
 * Level of Detail (LOD) means that a mesh and various simplified versions of it are
 * used depending on the required resolution.
 *
 * A Hierarchy of LODs (HLOD) is a tree structure where each leaf is a detailed mesh
 * and their parent nodes contain the simplified versions of the combination of the
 * leaves.
 *
 * How to use this class:
 * 1. Create the Hierarchy. This can be done by calling one of the static partition
 *    functions or by manually creating nodes and adding children.
 * 2. Call finalize() to generate the LODs.
 *
 * finalize is separate, since multiple trees from different meshes can be combined
 * into one tree.
 *
 * Note that LazyMesh is used everywhere, since HLODs are typically only used for
 * very large meshes, and LazyMesh allows creating the LODs without loading the
 * entire mesh and all its LODs into memory.
 */
template<typename BaseVecT>
class HLODTree
{
public:
    using Ptr = std::unique_ptr<HLODTree>;
    using Mesh = LazyMesh<BaseVecT>;

    /**
     * @brief Partitions the mesh into chunks and arranges them in a hierarchy.
     *        finalize() must be called to create the actual LOD.
     *
     * @param src The mesh to partition.
     * @param chunkSize The size of each chunk.
     * @param combineDepth The combination depth of the hierarchy.
     *
     * The combineDepth parameter is used because it is usually not desirable to
     * generate all LODs, but only up to a certain depth.
     * combineDepth < 0 means generate all LODs.
     * combineDepth == 0 means no LODs are generated, only leafs have meshes.
     * combineDepth == 1 means generate only one level.
     * and so on.
     */
    static Ptr partition(Mesh& src, float chunkSize, int combineDepth = -1);
    /**
     * @brief Arranges the given subtrees into a hierarchy.
     *
     * @param subtrees The subtrees to combine.
     * @param combineDepth The combination depth of the hierarchy. See partition(Mesh) for details.
     */
    static Ptr partition(std::vector<Ptr>&& subtrees, int combineDepth = -1);
    /**
     * @brief Arranges the given chunks into a hierarchy.
     *
     * If chunk boundaries are marked as vertex features, they will be recombined when merging the chunks
     * for higher LOD levels.
     *
     * Features can be marked using:
     * @code {.cpp}
     * auto v_feature = <PMPMesh>.getSurfaceMesh().vertex_property<bool>("v:feature", false));
     * for (VertexHandle vH : boundaryVertices)
     *    v_feature[vH] = true;
     * @endcode
     * Alternatively, if your chunks contain overlap, you may use trimChunkOverlap() to both
     * remove the overlap and mark the vertices as features.
     *
     * @param chunks A map from chunk position to the chunk.
     *               Note that only adjacent chunks according to the integer position are combined.
     * @param combineDepth The combination depth of the hierarchy. See partition(Mesh) for details.
     */
    static Ptr partition(std::unordered_map<Vector3i, Ptr>&& chunks, int combineDepth = -1);

    /**
     * @brief Removes any parts of a mesh that are outside of the given bounding box.
     *        The boundary of the remaining mesh is marked as a feature.
     * 
     * @param mesh The mesh to trim.
     * @param expectedBB The bounding box to trim to.
     */
    static void trimChunkOverlap(PMPMesh<BaseVecT>& mesh, const pmp::BoundingBox& expectedBB);

    /**
     * @brief Create a leaf.
     *
     * @param mesh The mesh of the leaf.
     * @param bb The bounding box of the leaf. Can be calculated using mesh.getSurfaceMesh().bounds().
     */
    static Ptr leaf(Mesh&& mesh, const pmp::BoundingBox& bb);
    /**
     * @brief Create a node with the given children.
     *
     * Keeps the children directly at this level.
     * Use partition() if the children should be arranged into a hierarchy.
     *
     * @param children The children of the node.
     * @param combineDepth The combination depth of the hierarchy. See partition(Mesh) for details.
     */
    static Ptr node(std::vector<Ptr>&& children, int combineDepth = -1);
    /**
     * @brief Create an empty node
     *
     * @param combineDepth The combination depth of the hierarchy. See partition(Mesh) for details.
     */
    static Ptr node(int combineDepth = -1);

    /**
     * @brief Returns if this is a leaf.
     */
    bool isLeaf() const
    {
        return m_children.empty();
    }

    /**
     * @brief Returns the depth of the tree. Only accurate after refresh() has been called.
     */
    size_t depth() const
    {
        return m_depth;
    }

    /**
     * @brief Updates bounding boxes and other internal values of the tree.
     *
     * This method should be called every time the tree has been changed.
     */
    void refresh();

    /**
     * @brief Generates the LOD meshes.
     *
     * This can only happen once, so the entire tree and all children should be created and
     * arranged before calling this method. This will finalize this tree and all its children,
     * marking them as finalized.
     *
     * @param saveMemory If true, tries to save as much memory as possible,
     *                   but makes this process a lot slower.
     */
    void finalize(bool saveMemory = false);

    /**
     * @brief Returns a reference to the mesh on this level.
     *
     * Returns none if this node is not a leaf and either
     * a) finalize() has not been called yet. Or
     * b) it should be skipped according to combineDepth.
     *
     * Note that modifications to this won't be reflected to higher levels if
     * finalize() was called prior to the change.
     */
    boost::optional<Mesh&> mesh()
    {
        return m_mesh ? boost::optional<Mesh&>(*m_mesh) : boost::none;
    }
    /**
     * @brief Returns a reference to the mesh on this level.
     *
     * Returns none if this node is not a leaf and either
     * a) finalize() has not been called yet. Or
     * b) it should be skipped according to combineDepth.
     */
    boost::optional<const Mesh&> mesh() const
    {
        return m_mesh ? boost::optional<const Mesh&>(*m_mesh) : boost::none;
    }

    /**
     * @brief Returns a BoundingBox containing this node and all its children.
     *        Only accurate after refresh() has been called.
     */
    const pmp::BoundingBox& bb() const
    {
        return m_bb;
    }

    /**
     * @brief Returns the children of this node. Empty if this is a leaf.
     *
     * Note: Make sure to call refresh() after making any changes to the children.
     */
    std::vector<Ptr>& children()
    {
        return m_children;
    }
    /**
     * @brief Returns the children of this node. Empty if this is a leaf.
     */
    const std::vector<Ptr>& children() const
    {
        return m_children;
    }

private:
    // make constructors private/deleted. This class should be used through Ptr.
    HLODTree() = default;
    HLODTree(const HLODTree&) = delete;
    HLODTree& operator=(const HLODTree&) = delete;
    HLODTree(HLODTree&&) = delete;
    HLODTree& operator=(HLODTree&&) = delete;

    /**
     * @brief See if this node should be combined with its children
     *
     * This only applies to nodes (m_depth > 0).
     * If true, m_mesh will contain a simplified version of all child meshes after finalize().
     * Otherwise, m_mesh will be empty.
     */
    bool shouldCombine() const
    {
        return m_combineDepth == -1 || m_depth <= m_combineDepth;
    }
    size_t finalizeRecursive(ProgressBar& progress);
    /// Combine the child meshes into this node's mesh.
    void combine();
    /// Simplify the mesh. Returns true if it can be further simplified.
    bool simplify();
    /// Collect all subtrees that can be simplified right now.
    bool collectSimplify(std::vector<HLODTree*>& canBeSimplified);
    /// Counts all subtrees for which shouldCombine() returns true.
    size_t countAllSimplify() const;

    static Ptr partitionRecursive(Ptr* start, Ptr* end, int combineDepth);

    boost::optional<Mesh> m_mesh = boost::none;
    pmp::BoundingBox m_bb;
    std::vector<Ptr> m_children; // empty for leafs
    size_t m_depth = 0; // == max(m_children.m_depth) + 1.  0 for leafs
    int m_combineDepth = -1; // see shouldCombine()

    bool m_simplified = false;
};

} // namespace lvr2

#include "HLODTree.tcc"
