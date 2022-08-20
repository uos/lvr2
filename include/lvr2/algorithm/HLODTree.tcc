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
 * HLODTree.tcc
 *
 * @date   29.06.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#pragma once

#include "HLODTree.hpp"
#include "lvr2/config/lvropenmp.hpp"

namespace lvr2
{

// functions that don't depend on the template parameter and can thus be moved to HLODTree.cpp
namespace HLODTree_internal
{

struct MeshSegment
{
    pmp::SurfaceMesh mesh;
    pmp::BoundingBox bb;
};

void segmentMesh(pmp::SurfaceMesh& mesh, float chunkSize, std::unordered_map<Vector3i, MeshSegment>& outChunks, std::vector<MeshSegment>& outSegments);
void splitMesh(pmp::SurfaceMesh& mesh, const pmp::BoundingBox& bb, float chunkSize, std::unordered_map<Vector3i, pmp::SurfaceMesh>& outChunks);
void trimChunkOverlap(pmp::SurfaceMesh& mesh, const pmp::BoundingBox& expectedBB);
void mergeChunkOverlap(pmp::SurfaceMesh& mesh, const pmp::BoundingBox& bb);

} // namespace HLODTree_internal

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::partition(Mesh& src, float chunkSize, int combineDepth)
{
    if (src.numFaces() == 0)
    {
        return nullptr;
    }

    using namespace HLODTree_internal;

    auto pmpMesh = src.modify();
    auto meshFile = src.getFile();

    std::unordered_map<Vector3i, MeshSegment> chunks;
    std::vector<MeshSegment> largeSegments;
    segmentMesh(pmpMesh->getSurfaceMesh(), chunkSize, chunks, largeSegments);
    pmpMesh.reset();

    Ptr chunkNode;
    if (!chunks.empty())
    {
        std::unordered_map<Vector3i, Ptr> chunkNodes;
        for (auto& [ index, chunk ] : chunks)
        {
            PMPMesh<BaseVecT> pmpMesh;
            pmpMesh.getSurfaceMesh() = std::move(chunk.mesh);
            Mesh mesh(std::move(pmpMesh), meshFile);
            chunkNodes[index] = leaf(std::move(mesh), chunk.bb);
        }
        chunkNode = partition(std::move(chunkNodes), combineDepth);
    }

    std::vector<Ptr> largeNodes;
    for (auto& segment : largeSegments)
    {
        std::unordered_map<Vector3i, pmp::SurfaceMesh> chunks;
        splitMesh(segment.mesh, segment.bb, chunkSize, chunks);

        std::unordered_map<Vector3i, Ptr> chunkNodes;
        for (auto& [ index, chunk ] : chunks)
        {
            auto bb = chunk.bounds();
            PMPMesh<BaseVecT> pmpMesh;
            pmpMesh.getSurfaceMesh() = std::move(chunk);
            Mesh mesh(std::move(pmpMesh), meshFile);
            chunkNodes[index] = leaf(std::move(mesh), bb);
        }
        largeNodes.push_back(partition(std::move(chunkNodes), combineDepth));
    }

    auto ret = node(combineDepth);
    if (chunkNode)
    {
        ret->m_children.push_back(std::move(chunkNode));
    }
    if (!largeNodes.empty())
    {
        ret->m_children.push_back(partition(std::move(largeNodes), combineDepth));
    }
    ret->refresh();
    return ret;
}

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::partition(std::vector<Ptr>&& subtrees, int combineDepth)
{
    if (subtrees.empty())
    {
        return nullptr;
    }
    else if (subtrees.size() == 1)
    {
        return std::move(subtrees[0]);
    }
    auto ret = partitionRecursive(subtrees.data(), subtrees.data() + subtrees.size(), combineDepth);
    ret->refresh();
    return ret;
}

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::partitionRecursive(Ptr* start, Ptr* end, int combineDepth)
{
    size_t n = end - start;

    auto ret = node(combineDepth);
    if (n <= 8)
    {
        std::move(start, end, std::back_inserter(ret->m_children));
    }
    else
    {
        // currying: creates a function that compares elements based on the given axis
        auto split_fn = [](int axis)
        {
            return [axis](const Ptr& a, const Ptr& b)
            {
                return a->bb().center()[axis] < b->bb().center()[axis];
            };
        };

        Ptr* starts[9];
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
            ret->m_children.push_back(partitionRecursive(starts[i], starts[i + 1], combineDepth));
        }
    }
    return ret;
}

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::partition(std::unordered_map<Vector3i, Ptr>&& in_chunks, int combineDepth)
{
    if (in_chunks.empty())
    {
        return nullptr;
    }

    Vector3i min = Vector3i::Constant(std::numeric_limits<Vector3i::value_type>::max());
    for (auto& [ index, _ ] : in_chunks)
    {
        min = min.cwiseMin(index);
    }

    // in_chunks can be anywhere in 3D space, including negative indices. Calculations are easier if they are in the
    // range [0, n] where n is the index of the largest chunk => shift all indices by min
    std::unordered_map<Vector3i, Ptr> chunks;
    chunks.reserve(chunks.size());
    for (auto& [ index, chunk ] : in_chunks)
    {
        chunks[index - min] = std::move(chunk);
    }

    std::unordered_map<Vector3i, Ptr> parents;
    size_t addedDepth = 0;

    while (chunks.size() != 1)
    {
        for (auto& [ index, chunk ] : chunks)
        {
            Vector3i parentIndex = index / 2;
            auto parent = parents.find(parentIndex);
            if (parent == parents.end())
            {
                parent = parents.emplace(parentIndex, node(combineDepth)).first;
            }
            parent->second->m_children.push_back(std::move(chunk));
        }
        chunks.clear();
        for (auto& [ index, parent ] : parents)
        {
            if (parent->m_children.size() == 1)
            {
                chunks[index] = std::move(parent->m_children[0]);
                continue;
            }

            size_t numGrandchildren = 0;
            for (auto& child : parent->m_children)
            {
                numGrandchildren += child->m_children.size();
            }
            if (addedDepth > 0 && numGrandchildren <= 8)
            {
                // sparse node -> collapse one layer
                std::vector<Ptr> newChildren;
                for (auto& child : parent->m_children)
                {
                    if (child->isLeaf())
                    {
                        newChildren.emplace_back(std::move(child));
                    }
                    else
                    {
                        for (auto& childChild : child->m_children)
                        {
                            newChildren.emplace_back(std::move(childChild));
                        }
                    }
                }
                parent->m_children.swap(newChildren);
            }
            chunks[index] = std::move(parent);
        }
        addedDepth++;
        parents.clear();
    }
    auto& ret = chunks.begin()->second;
    ret->refresh();
    return std::move(ret);
}

template<typename BaseVecT>
void HLODTree<BaseVecT>::trimChunkOverlap(PMPMesh<BaseVecT>& mesh, const pmp::BoundingBox& expectedBB)
{
    HLODTree_internal::trimChunkOverlap(mesh.getSurfaceMesh(), expectedBB);
}

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::leaf(Mesh&& mesh, const pmp::BoundingBox& bb)
{
    Ptr ret(new HLODTree<BaseVecT>()); // can't use make_unique because of private constructor
    ret->m_mesh = std::move(mesh);
    ret->m_bb = bb;
    return ret;
}

template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::node(std::vector<Ptr>&& children, int combineDepth)
{
    auto ret = node(combineDepth);
    ret->m_children = std::move(children);
    ret->refresh();
    return ret;
}
template<typename BaseVecT>
typename HLODTree<BaseVecT>::Ptr HLODTree<BaseVecT>::node(int combineDepth)
{
    Ptr ret(new HLODTree<BaseVecT>()); // can't use make_unique because of private constructor
    ret->m_combineDepth = combineDepth;
    return ret;
}

template<typename BaseVecT>
void HLODTree<BaseVecT>::refresh()
{
    if (isLeaf())
    {
        return;
    }
    m_bb = pmp::BoundingBox();
    m_depth = 0;
    for (auto& child : m_children)
    {
        child->refresh();
        m_bb += child->m_bb;
        m_depth = std::max(m_depth, child->m_depth + 1);
    }
}

template<typename BaseVecT>
void HLODTree<BaseVecT>::finalize(float reductionFactor, AllowedMemoryUsage allowedMemUsage)
{
    if (reductionFactor < 0.0f || reductionFactor > 1.0f)
    {
        throw std::invalid_argument("reductionFactor must be in [0, 1]");
    }

    if (isLeaf() || m_simplified)
    {
        return;
    }
    refresh();

    size_t numThreads = OpenMPConfig::getNumThreads();

    if (allowedMemUsage == AllowedMemoryUsage::Minimal || numThreads == 1)
    {
        size_t total = countAllSimplify();
        ProgressBar progress(total, "Generating LOD");
        size_t fullySimplified = finalizeRecursive(reductionFactor, progress);
        std::cout << "\r" << timestamp << "LOD: " << fullySimplified << " / " << total
                  << " meshes reached simplification limit" << std::endl;
        return;
    }

    if (allowedMemUsage == AllowedMemoryUsage::Moderate)
    {
        numThreads = std::max(numThreads / 8, (size_t)1);
    }

    std::vector<HLODTree*> canBeSimplified;
    while (!collectSimplify(canBeSimplified))
    {
        ProgressBar progress(canBeSimplified.size(), "Generating LOD of one Layer");
        size_t fullySimplified = 0;

        #pragma omp parallel for reduction(+:fullySimplified) num_threads(numThreads)
        for (size_t i = 0; i < canBeSimplified.size(); i++)
        {
            if (!canBeSimplified[i]->simplify(reductionFactor))
            {
                fullySimplified++;
            }
            ++progress;
        }

        std::cout << "\r" << timestamp << "LOD: " << fullySimplified << " / " << canBeSimplified.size()
                  << " meshes reached simplification limit" << std::endl;

        canBeSimplified.clear();
    }
    std::cout << timestamp << "Finished generating LOD" << std::endl;
}

template<typename BaseVecT>
size_t HLODTree<BaseVecT>::finalizeRecursive(float reductionFactor, ProgressBar& progress)
{
    if (isLeaf() || m_simplified)
    {
        return 0;
    }
    size_t fullySimplified = 0;
    for (auto& child : m_children)
    {
        fullySimplified += child->finalizeRecursive(reductionFactor, progress);
    }
    if (shouldCombine())
    {
        combine();
        if (!simplify(reductionFactor))
        {
            fullySimplified++;
        }
        ++progress;
    }
    return fullySimplified;
}

template<typename BaseVecT>
void HLODTree<BaseVecT>::combine()
{
    if (isLeaf() || m_mesh)
    {
        return;
    }

    std::vector<std::shared_ptr<const PMPMesh<BaseVecT>>> pmpMeshes;
    std::vector<const pmp::SurfaceMesh*> meshes;
    for (auto& child : m_children)
    {
        pmpMeshes.push_back(child->m_mesh->get());
        meshes.push_back(&pmpMeshes.back()->getSurfaceMesh());
    }
    PMPMesh<BaseVecT> pmpMesh;
    auto& mesh = pmpMesh.getSurfaceMesh();
    mesh.join_mesh(meshes);

    meshes.clear();
    pmpMeshes.clear();

    HLODTree_internal::mergeChunkOverlap(mesh, m_bb);

    if (!mesh.has_vertex_property("v:quadric"))
    {
        pmp::SurfaceSimplification::calculate_quadrics(mesh);
    }

    auto meshFile = m_children[0]->m_mesh->getFile();
    m_mesh = Mesh(std::move(pmpMesh), meshFile);
}

template<typename BaseVecT>
bool HLODTree<BaseVecT>::simplify(float reductionFactor)
{
    m_simplified = true;

    auto pmpMesh = m_mesh->modify();
    size_t target = pmpMesh->numVertices() * reductionFactor;

    pmp::SurfaceSimplification simplify(pmpMesh->getSurfaceMesh(), true);
    return simplify.simplify(target);
}

template<typename BaseVecT>
bool HLODTree<BaseVecT>::collectSimplify(std::vector<HLODTree*>& canBeSimplified)
{
    // this function returns true iff this level is done and the parent can be simplified

    if (isLeaf() || m_simplified)
    {
        return true;
    }
    if (m_mesh) // already combined, but not simplified
    {
        canBeSimplified.push_back(this);
        return false;
    }
    size_t childCount = 0;
    for (auto& child : m_children)
    {
        if (child->collectSimplify(canBeSimplified))
        {
            childCount++;
        }
    }
    if (childCount < m_children.size())
    {
        return false;
    }
    if (shouldCombine())
    {
        combine();
        canBeSimplified.push_back(this);
        return false;
    }
    m_simplified = true;
    return true;
}

template<typename BaseVecT>
size_t HLODTree<BaseVecT>::countAllSimplify() const
{
    if (isLeaf() || m_simplified)
    {
        return 0;
    }
    size_t count = 0;
    for (auto& child : m_children)
    {
        count += child->countAllSimplify();
    }
    if (shouldCombine())
    {
        count++;
    }
    return count;
}


} // namespace lvr2
