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
 * HLODTree.cpp
 *
 * @date   29.06.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "lvr2/algorithm/HLODTree.hpp"

namespace lvr2::HLODTree_internal
{

struct SegmentMetaData
{
    pmp::IndexType id = pmp::PMP_MAX_INDEX;
    size_t numFaces = 0;
    size_t numVertices = 0;
    pmp::BoundingBox bb;
};

Vector3i chunkIndex(const pmp::Point& vec, float chunkSize)
{
    Vector3i ret;
    ret.x() = std::floor(vec.x() / chunkSize);
    ret.y() = std::floor(vec.y() / chunkSize);
    ret.z() = std::floor(vec.z() / chunkSize);
    return ret;
}

void segmentMesh(pmp::SurfaceMesh& mesh, float chunkSize, std::unordered_map<Vector3i, MeshSegment>& outChunks, std::vector<MeshSegment>& outSegments)
{
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

    auto f_prop = mesh.add_face_property<pmp::IndexType>("f:segment", pmp::PMP_MAX_INDEX);
    auto v_prop = mesh.add_vertex_property<pmp::IndexType>("v:segment", pmp::PMP_MAX_INDEX);

    std::vector<SegmentMetaData> segments;

    std::vector<pmp::Vertex> queue;
    ProgressBar progress(mesh.n_vertices(), "Segmenting mesh");

    for (auto vH : mesh.vertices())
    {
        if (v_prop[vH] != pmp::PMP_MAX_INDEX)
        {
            continue;
        }

        auto& segment = segments.emplace_back();
        segment.id = segments.size() - 1;

        v_prop[vH] = segment.id;
        queue.push_back(vH);

        while (!queue.empty())
        {
            pmp::Vertex vH = queue.back();
            queue.pop_back();
            segment.numVertices++;
            segment.bb += mesh.position(vH);
            ++progress;

            for (auto heH : mesh.halfedges(vH))
            {
                auto ovH = mesh.to_vertex(heH);
                auto& ovH_id = v_prop[ovH];
                if (ovH_id != segment.id)
                {
                    if (ovH_id != pmp::PMP_MAX_INDEX)
                    {
                        throw std::runtime_error("Segmenter: found vertex with multiple segments");
                    }
                    ovH_id = segment.id;
                    queue.push_back(ovH);
                }

                pmp::Face fH = mesh.face(heH);
                if (fH.is_valid() && f_prop[fH] != segment.id)
                {
                    f_prop[fH] = segment.id;
                    segment.numFaces++;
                }
            }
        }
    }

    size_t initialSize = segments.size();
    for (auto it = segments.begin(); it != segments.end(); ++it)
    {
        if (it->numFaces == 0)
        {
            segments.erase(it);
            --it;
        }
    }

    std::cout << "\r" << timestamp << "Found " << segments.size() << " initial segments" << std::endl;

    // ==================== merge small segments within a chunk together ====================

    std::unordered_map<Vector3i, pmp::IndexType> chunkMap;
    std::vector<pmp::IndexType> segmentMap(initialSize, pmp::PMP_MAX_INDEX);
    std::vector<SegmentMetaData> metaData;
    std::vector<bool> isLarge;

    for (auto& segment : segments)
    {
        if (segment.bb.longest_axis_size() >= chunkSize)
        {
            pmp::IndexType id = segment.id;
            auto& meta = metaData.emplace_back(std::move(segment));
            meta.id = metaData.size() - 1;
            segmentMap[id] = meta.id;
            isLarge.push_back(true);
            continue;
        }

        // all other segments are merged based on the chunk that their center lies in
        auto chunk_id = chunkIndex(segment.bb.center(), chunkSize);
        auto elem = chunkMap.find(chunk_id);
        if (elem == chunkMap.end())
        {
            // start a new chunk with this segment
            pmp::IndexType id = segment.id;
            auto& meta = metaData.emplace_back(std::move(segment));
            meta.id = metaData.size() - 1;
            segmentMap[id] = meta.id;
            chunkMap[chunk_id] = meta.id;
            isLarge.push_back(false);
        }
        else
        {
            // merge this segment with the existing chunk
            pmp::IndexType new_id = elem->second;
            segmentMap[segment.id] = new_id;

            auto& target = metaData[new_id];
            target.numFaces += segment.numFaces;
            target.numVertices += segment.numVertices;
            target.bb += segment.bb;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        f_prop[pmp::Face(i)] = segmentMap[f_prop[pmp::Face(i)]];
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        v_prop[pmp::Vertex(i)] = segmentMap[v_prop[pmp::Vertex(i)]];
    }

    std::vector<pmp::SurfaceMesh> meshes(metaData.size());
    for (size_t i = 0; i < metaData.size(); i++)
    {
        meshes[i].reserve(metaData[i].numVertices, 0, metaData[i].numFaces);
    }
    mesh.split_mesh(meshes, f_prop, v_prop);
    mesh.remove_face_property(f_prop);
    mesh.remove_vertex_property(v_prop);

    for (size_t i = 0; i < meshes.size(); i++)
    {
        if (isLarge[i])
        {
            auto& out = outSegments.emplace_back();
            out.mesh = std::move(meshes[i]);
            out.bb = metaData[i].bb;
        }
    }
    for (auto [ chunk_id, index ] : chunkMap)
    {
        auto& segment = outChunks[chunk_id];
        segment.mesh = std::move(meshes[index]);
        segment.bb = metaData[index].bb;
    }

    if (!outChunks.empty())
    {
        std::cout << timestamp << "Merged " << (segments.size() - outSegments.size()) << " small segments into "
                  << outChunks.size() << " chunks" << std::endl;
    }
}

void splitMesh(pmp::SurfaceMesh& mesh, const pmp::BoundingBox& bb, float chunkSize, std::unordered_map<Vector3i, pmp::SurfaceMesh>& outChunks)
{
    mesh.garbage_collection(); // ensure that we never need to check mesh.is_deleted(...)

    pmp::Point chunkOffset;
    {
        // align the chunking to the center of the segment:
        // modify the offset so that there is equal overlap on all sides
        pmp::Point sizeOfMesh = bb.max() - bb.min();
        pmp::Point size = sizeOfMesh / chunkSize;
        Eigen::Vector3i numChunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));
        pmp::Point sizeOfChunks = numChunks.cast<float>() * chunkSize;
        chunkOffset = bb.min() - (sizeOfChunks - sizeOfMesh) / 2.0f;
    }

    std::unordered_map<Vector3i, pmp::IndexType> chunkMap;
    std::vector<pmp::Face> featureFaces;

    auto v_chunk_id = mesh.add_vertex_property<pmp::IndexType>("v:chunk_id", pmp::PMP_MAX_INDEX);
    auto f_chunk_id = mesh.add_face_property<pmp::IndexType>("f:chunk_id", pmp::PMP_MAX_INDEX);

    #pragma omp parallel
    {
        // copy of chunkMap to avoid having to lock on every lookup. updated when necessary
        std::unordered_map<Vector3i, pmp::IndexType> local_chunkMap;

        #pragma omp for schedule(static)
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            pmp::Face fH(i);
            auto pos = mesh.position(mesh.to_vertex(mesh.halfedge(fH)));
            auto chunk_id = chunkIndex(pos - chunkOffset, chunkSize);
            auto it = local_chunkMap.find(chunk_id);
            if (it != local_chunkMap.end())
            {
                f_chunk_id[fH] = it->second;
            }
            else
            {
                #pragma omp critical
                {
                    it = chunkMap.find(chunk_id);
                    if (it == chunkMap.end())
                    {
                        pmp::IndexType id = chunkMap.size();
                        it = chunkMap.emplace(chunk_id, id).first;
                    }
                    f_chunk_id[fH] = it->second;
                    local_chunkMap = chunkMap;
                }
            }
        }

        #pragma omp for schedule(static)
        for (size_t i = 0; i < mesh.vertices_size(); i++)
        {
            pmp::Vertex vH(i);
            auto chunk_id = chunkIndex(mesh.position(vH) - chunkOffset, chunkSize);
            auto it = chunkMap.find(chunk_id);
            v_chunk_id[vH] = it != chunkMap.end() ? it->second : f_chunk_id[*mesh.faces(vH)];
        }

        #pragma omp for schedule(dynamic,64)
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            pmp::Face fH(i);
            auto id = f_chunk_id[fH];

            for (auto vH : mesh.vertices(fH))
            {
                if (v_chunk_id[vH] != id)
                {
                    #pragma omp critical
                    featureFaces.push_back(fH);
                    break;
                }
            }
        }
    }

    if (!featureFaces.empty())
    {
        auto v_feature = mesh.add_vertex_property<bool>("v:feature");
        mesh.add_edge_property("e:feature", false);

        std::unordered_map<pmp::IndexType, std::unordered_map<pmp::Vertex, pmp::Vertex>> vertex_map;
        std::vector<pmp::Vertex> vertices;
        std::vector<pmp::Point> positions;
        vertices.reserve(3);
        positions.reserve(3);

        auto fprop_map = mesh.gen_fprop_map(mesh);
        auto vprop_map = mesh.gen_vprop_map(mesh);
        for (auto fH : featureFaces)
        {
            auto target_id = f_chunk_id[fH];
            auto& v_map = vertex_map[target_id];

            vertices.clear();
            positions.clear();
            for (auto vH : mesh.vertices(fH))
            {
                vertices.push_back(vH);
                positions.push_back(mesh.position(vH));
                if (v_chunk_id[vH] != target_id)
                {
                    v_feature[vH] = true;
                }
            }
            mesh.delete_face(fH);

            for (size_t i = 0; i < 3; i++)
            {
                auto& vH = vertices[i];
                auto it = v_map.find(vH);
                if (it != v_map.end())
                {
                    vH = it->second;
                }
                else if (mesh.is_deleted(vH) || v_chunk_id[vH] != target_id)
                {
                    auto new_vH = mesh.add_vertex(positions[i]);
                    mesh.copy_vprops(mesh, vH, new_vH, vprop_map);
                    v_chunk_id[new_vH] = target_id;
                    v_map[vH] = new_vH;
                    vH = new_vH;
                }
            }
            auto new_fH = mesh.add_face(vertices);
            mesh.copy_fprops(mesh, fH, new_fH, fprop_map);
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(chunkMap.size());

    mesh.split_mesh(meshes, f_chunk_id, v_chunk_id);

    mesh.remove_face_property(f_chunk_id);
    mesh.remove_vertex_property(v_chunk_id);

    for (auto [ chunk_id, index ] : chunkMap)
    {
        outChunks[chunk_id] = std::move(meshes[index]);
    }
}

void mergeChunkOverlap(pmp::SurfaceMesh& mesh)
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

    constexpr float MAX_DISTANCE = 1e-6f;
    #pragma omp parallel for
    for (size_t i = 1; i < candidates.size(); i++)
    {
        auto& pos = mesh.position(candidates[i]);
        for (size_t j = 0; j < i; j++)
        {
            float dist_sq = (pos - mesh.position(candidates[j])).squaredNorm();
            if (dist_sq < MAX_DISTANCE)
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

    std::vector<pmp::Face> featureFaces;
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
                featureFaces.push_back(fH);
                break;
            }
        }
    }

    std::unordered_map<pmp::Vertex, pmp::Point> vertex_positions;
    std::unordered_map<pmp::Face, std::vector<pmp::Vertex>> deleted_faces;
    for (auto fH : featureFaces)
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

    mesh.remove_degenerate_faces(false);
    mesh.duplicate_non_manifold_vertices(false);

    mesh.garbage_collection();
}

} // namespace lvr2::HLODTree_internal
