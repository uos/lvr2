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
 * Segmenter.cpp
 *
 * @date   03.02.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "Segmenter.hpp"
#include "B3dmWriter.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/algorithm/pmp/SurfaceNormals.h"

#include <thread>

namespace lvr2
{

struct consistency_error : public std::runtime_error
{
    consistency_error(size_t expected, size_t found, const char* name)
        : std::runtime_error(std::string("Segmenter: inconsistent number of ") + name + ": "
                             + "expected " + std::to_string(expected)
                             + ", found " + std::to_string(found))
    {}
};

struct SegmentMetaData
{
    SegmentId id = INVALID_SEGMENT;
    size_t num_faces = 0;
    size_t num_vertices = 0;
    pmp::BoundingBox bb;
    std::string filename = "";
};

/**
 * @brief Calculates a 1D Chunk-index from a 3D position
 *
 * @param p the 3D position
 * @param chunk_size the size of a chunk
 * @param num_chunks the number of chunks along each axis
 * @return uint32_t the 1D Chunk-index
 */
uint32_t chunk_index(pmp::Point p, float chunk_size, Eigen::Vector3i num_chunks)
{
    return std::floor(p.x() / chunk_size)
           + std::floor(p.y() / chunk_size) * num_chunks.x()
           + std::floor(p.z() / chunk_size) * num_chunks.x() * num_chunks.y();
}

void segment_mesh(pmp::SurfaceMesh& mesh,
                  const pmp::BoundingBox& bb,
                  float chunk_size,
                  std::vector<MeshSegment>& chunks,
                  std::vector<MeshSegment>& large_segments)
{
    auto f_prop = mesh.add_face_property<SegmentId>("f:segment", INVALID_SEGMENT);
    auto v_prop = mesh.add_vertex_property<SegmentId>("v:segment", INVALID_SEGMENT);
    auto h_prop = mesh.add_halfedge_property<SegmentId>("h:segment", INVALID_SEGMENT);

    std::vector<SegmentMetaData> segments;

    std::vector<pmp::Vertex> queue;
    ProgressBar progress(mesh.n_vertices(), "Segmenting mesh");

    for (auto vH : mesh.vertices())
    {
        if (v_prop[vH] != INVALID_SEGMENT)
        {
            continue;
        }

        auto [ id, segment ] = push_and_get_index(segments);
        segment.id = id;

        v_prop[vH] = segment.id;
        queue.push_back(vH);

        while (!queue.empty())
        {
            pmp::Vertex vH = queue.back();
            queue.pop_back();
            segment.num_vertices++;
            segment.bb += mesh.position(vH);
            ++progress;

            for (auto heH : mesh.halfedges(vH))
            {
                auto ovH = mesh.to_vertex(heH);
                auto& ovH_id = v_prop[ovH];
                if (ovH_id != segment.id)
                {
                    if (ovH_id != INVALID_SEGMENT)
                    {
                        throw std::runtime_error("Segmenter: found vertex with multiple segments");
                    }
                    ovH_id = segment.id;
                    queue.push_back(ovH);
                }

                h_prop[heH] = segment.id;

                FaceHandle fH = mesh.face(heH);
                if (fH.is_valid() && f_prop[fH] != segment.id)
                {
                    f_prop[fH] = segment.id;
                    segment.num_faces++;
                }
            }
        }
    }
    std::cout << "\r" << timestamp << "Found " << segments.size() << " initial segments" << std::endl;

    // consistency check
    size_t total_faces = 0, total_vertices = 0;
    for (auto& segment : segments)
    {
        total_faces += segment.num_faces;
        total_vertices += segment.num_vertices;
    }
    if (total_faces != mesh.n_faces())
    {
        throw consistency_error(mesh.n_faces(), total_faces, "SegmentMetaData faces");
    }
    if (total_vertices != mesh.n_vertices())
    {
        throw consistency_error(mesh.n_vertices(), total_vertices, "SegmentMetaData vertices");
    }

    // ==================== merge small segments within a chunk together ====================

    std::unordered_map<uint32_t, SegmentId> chunk_map;
    pmp::Point total_size = bb.max() - bb.min();
    pmp::Point size = total_size / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    // align the chunking to the center of the segment:
    // modify the offset so that there is equal overlap on all sides
    pmp::Point size_of_chunks = num_chunks.cast<float>() * chunk_size;
    pmp::Point chunk_offset = bb.min() - (size_of_chunks - total_size) / 2.0f;

    std::vector<SegmentId> segment_map(segments.size(), INVALID_SEGMENT);
    std::vector<SegmentMetaData> meta_data;
    std::vector<bool> is_large;

    for (auto& segment : segments)
    {
        if (segment.num_faces == 0)
        {
            continue;
        }

        if (segment.bb.longest_axis_size() >= chunk_size)
        {
            SegmentId id = segment.id;
            auto [ new_id, meta ] = push_and_get_index(meta_data, std::move(segment));
            meta.id = new_id;
            segment_map[id] = new_id;
            is_large.push_back(true);
            continue;
        }

        // all other segments are merged based on the chunk that their center lies in
        uint32_t chunk_id = chunk_index(segment.bb.center() - chunk_offset, chunk_size, num_chunks);
        auto elem = chunk_map.find(chunk_id);
        if (elem == chunk_map.end())
        {
            // start a new chunk with this segment
            SegmentId id = segment.id;
            auto [ new_id, meta ] = push_and_get_index(meta_data, std::move(segment));
            meta.id = new_id;
            segment_map[id] = new_id;
            chunk_map[chunk_id] = new_id;
            is_large.push_back(false);
        }
        else
        {
            // merge this segment with the existing chunk
            SegmentId new_id = elem->second;
            segment_map[segment.id] = new_id;

            auto& target = meta_data[new_id];
            target.num_faces += segment.num_faces;
            target.num_vertices += segment.num_vertices;
            target.bb += segment.bb;
        }
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        pmp::Face f(i);
        if (!mesh.is_deleted(f))
        {
            f_prop[f] = segment_map[f_prop[f]];
        }
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.vertices_size(); i++)
    {
        pmp::Vertex v(i);
        if (!mesh.is_deleted(v))
        {
            v_prop[v] = segment_map[v_prop[v]];
        }
    }
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.edges_size(); i++)
    {
        pmp::Edge e(i);
        if (!mesh.is_deleted(e))
        {
            pmp::Halfedge h0 = mesh.halfedge(e, 0);
            pmp::Halfedge h1 = mesh.halfedge(e, 1);
            h_prop[h0] = segment_map[h_prop[h0]];
            h_prop[h1] = segment_map[h_prop[h1]];
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(meta_data.size());
    for (size_t i = 0; i < meta_data.size(); i++)
    {
        meshes[i].reserve(meta_data[i].num_vertices, 0, meta_data[i].num_faces);
    }
    mesh.split_mesh(meshes, f_prop, v_prop, h_prop);
    mesh.remove_face_property(f_prop);
    mesh.remove_vertex_property(v_prop);
    mesh.remove_halfedge_property(h_prop);

    for (size_t i = 0; i < meta_data.size(); i++)
    {
        auto& meta = meta_data[i];
        MeshSegment& out = is_large[i] ? large_segments.emplace_back() : chunks.emplace_back();
        out.mesh.reset(new pmp::SurfaceMesh(std::move(meshes[i])));
        out.bb = meta.bb;

        // consistency check
        if (out.mesh->n_faces() != meta.num_faces)
        {
            std::cerr << consistency_error(meta.num_faces, out.mesh->n_faces(), "MeshSegment faces").what() << std::endl;
        }
        if (out.mesh->n_vertices() != meta.num_vertices)
        {
            std::cerr << consistency_error(meta.num_vertices, out.mesh->n_vertices(), "MeshSegment vertices").what() << std::endl;
        }
    }

    std::cout << timestamp << "Merged " << (segments.size() - large_segments.size()) << " small segments into "
              << chunks.size() << " chunks" << std::endl;
}

bool add_face(pmp::SurfaceMesh& target,
              FaceHandle fH,
              std::unordered_map<pmp::Vertex, pmp::Vertex>& vertex_map,
              const pmp::SurfaceMesh& src)
{
    static thread_local std::vector<pmp::Vertex> face_vertices;
    face_vertices.clear();
    for (pmp::Vertex vH : src.vertices(fH))
    {
        auto it = vertex_map.find(vH);
        if (it == vertex_map.end())
        {
            pmp::Vertex new_vH = target.add_vertex(src.position(vH));
            target.copy_vprops(src, vH, new_vH);
            vertex_map[vH] = new_vH;
            face_vertices.push_back(new_vH);
        }
        else
        {
            face_vertices.push_back(it->second);
        }
    }
    try
    {
        FaceHandle new_fH = target.add_face(face_vertices);
        target.copy_fprops(src, fH, new_fH);
    }
    catch (pmp::TopologyException& e)
    {
        return false;
    }
    return true;
}

void split_mesh(MeshSegment& segment,
                float chunk_size,
                std::vector<MeshSegment>& out_meshes)
{
    auto& mesh = *segment.mesh;

    pmp::Point size_of_segment = segment.bb.max() - segment.bb.min();
    pmp::Point size = size_of_segment / chunk_size;
    Eigen::Vector3i num_chunks(std::ceil(size.x()), std::ceil(size.y()), std::ceil(size.z()));

    // align the chunking to the center of the segment:
    // modify the offset so that there is equal overlap on all sides
    pmp::Point size_of_chunks = num_chunks.cast<float>() * chunk_size;
    pmp::Point chunk_offset = segment.bb.min() - (size_of_chunks - size_of_segment) / 2.0f;

    auto f_chunk_id = mesh.face_property<uint32_t>("f:chunk_id");
    std::unordered_set<uint32_t> chunk_ids;
    #pragma omp parallel
    {
        std::unordered_set<uint32_t> local_chunk_ids;
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < mesh.faces_size(); i++)
        {
            FaceHandle fH(i);
            if (mesh.is_deleted(fH))
            {
                continue;
            }
            pmp::Point pos(0, 0, 0);
            size_t count = 0;
            for (auto vH : mesh.vertices(fH))
            {
                pos += mesh.position(vH);
                count++;
            }
            assert(count == 3);
            pos /= count;

            uint32_t chunk_id = chunk_index(pos - chunk_offset, chunk_size, num_chunks);
            f_chunk_id[fH] = chunk_id;
            local_chunk_ids.insert(chunk_id);
        }
        #pragma omp critical
        {
            chunk_ids.insert(local_chunk_ids.begin(), local_chunk_ids.end());
        }
    }

    std::unordered_map<uint32_t, uint32_t> chunk_map;
    for (uint32_t chunk_id : chunk_ids)
    {
        uint32_t index = chunk_map.size();
        chunk_map[chunk_id] = index;
    }

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < mesh.faces_size(); i++)
    {
        FaceHandle fH(i);
        if (!mesh.is_deleted(fH))
        {
            f_chunk_id[fH] = chunk_map[f_chunk_id[fH]];
        }
    }

    std::vector<pmp::SurfaceMesh> meshes(chunk_map.size());
    mesh.split_mesh(meshes, f_chunk_id);

    mesh.remove_face_property(f_chunk_id);

    for (auto& mesh : meshes)
    {
        auto& out_mesh = out_meshes.emplace_back();
        out_mesh.mesh.reset(new pmp::SurfaceMesh(std::move(mesh)));
        out_mesh.bb = out_mesh.mesh->bounds();
    }
}

// SegmentTree

void SegmentTree::simplify(Cesium3DTiles::Tile& root)
{
    while (!combine_if_possible())
    {
        #pragma omp parallel
        #pragma omp single
        simplify_if_possible(root);

        std::cout << "layer complete" << std::endl;
    }
    // print();
}
SegmentTree::Ptr SegmentTree::octree_partition(
    std::vector<MeshSegment>& segments, Cesium3DTiles::Tile& root,
    const boost::filesystem::path& path, int combine_depth)
{
    std::vector<MeshSegment*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = &segments[i];
    }

    std::string filename = (path / "t").string();

    return octree_split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(),
                                  filename, root, combine_depth);
}
SegmentTree::Ptr SegmentTree::octree_split_recursive(
    MeshSegment** start, MeshSegment** end,
    const std::string& filename, Cesium3DTiles::Tile& tile,
    int combine_depth)
{
    size_t n = end - start;

    SegmentTreeNode* node = new SegmentTreeNode();

    if (n <= 8)
    {
        tile.children.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            auto& segment = start[i];
            auto& child_tile = tile.children[i];

            segment->filename = filename + std::to_string(i) + ".b3dm";

            convert_bounding_box(segment->bb, child_tile.boundingVolume);

            Cesium3DTiles::Content content;
            content.uri = segment->filename;
            child_tile.content = content;

            auto child = new SegmentTreeLeaf(*segment);

            child_tile.geometricError = child->geometric_error();

            node->add_child(SegmentTree::Ptr(child));
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

        tile.children.resize(8);
        for (size_t i = 0; i < 8; i++)
        {
            node->add_child(octree_split_recursive(starts[i], starts[i + 1],
                                                   filename + std::to_string(i), tile.children[i],
                                                   combine_depth));
        }
    }
    node->get_segment().filename = filename + "_.b3dm";
    convert_bounding_box(node->get_segment().bb, tile.boundingVolume);

    if (combine_depth > 0)
    {
        node->skipped = node->depth > combine_depth;
        if (node->depth == combine_depth)
        {
            tile.refine = Cesium3DTiles::Tile::Refine::REPLACE;
        }
    }

    if (!node->skipped)
    {
        Cesium3DTiles::Content content;
        content.uri = node->get_segment().filename;
        tile.content = content;
    }

    return SegmentTree::Ptr(node);
}

// SegmentTreeNode

void SegmentTreeNode::add_child(SegmentTree::Ptr child)
{
    meta_segment.bb += child->get_segment().bb;
    depth = std::max(depth, child->depth + 1);
    children.push_back(std::move(child));
}

bool SegmentTreeNode::combine_if_possible()
{
    if (simplified)
    {
        meta_segment.mesh->garbage_collection();
        return true;
    }
    std::vector<pmp::SurfaceMesh*> meshes;
    for (auto& child : children)
    {
        if (child->combine_if_possible())
        {
            meshes.push_back(child->get_segment().mesh.get());
        }
    }
    if (meshes.size() == children.size())
    {
        auto mesh = new pmp::SurfaceMesh();
        if (!skipped)
        {
            mesh->join_mesh(meshes);
            if (!mesh->has_vertex_property("v:quadric"))
            {
                auto vquadric_ = mesh->add_vertex_property<pmp::Quadric>("v:quadric");
                pmp::SurfaceNormals::compute_face_normals(*mesh);
                auto fnormal_ = mesh->get_face_property<pmp::Normal>("f:normal");
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < mesh->n_vertices(); i++)
                {
                    pmp::Vertex v(i);
                    vquadric_[v].clear();
                    for (auto f : mesh->faces(v))
                    {
                        vquadric_[v] += pmp::Quadric(fnormal_[f], mesh->position(v));
                    }
                }
            }
        }
        meta_segment.mesh.reset(mesh);
    }
    // only return true after simplification is done
    return false;
}
void SegmentTreeNode::simplify_if_possible(Cesium3DTiles::Tile& tile)
{
    if (simplified)
    {
        return;
    }
    if (meta_segment.mesh != nullptr)
    {
        double child_error = 0;
        for (auto& child : tile.children)
        {
            child_error += child.geometricError;
        }
        if (skipped)
        {
            tile.geometricError = child_error;
        }
        else
        {
            size_t old_num_faces = meta_segment.mesh->n_faces();
            pmp::SurfaceSimplification simplify(*meta_segment.mesh, true);
            simplify.simplify(meta_segment.mesh->n_vertices() * 0.2);
            size_t new_num_faces = meta_segment.mesh->n_faces();

            float ratio = (float)new_num_faces / old_num_faces;
            ratio = std::floor(ratio * 10) / 10;
            std::cout << new_num_faces << '(' << ratio << "%) " << std::flush;
            tile.geometricError = geometric_error();
        }

        simplified = true;
    }
    else
    {
        for (size_t i = 0; i < children.size(); i++)
        {
            Cesium3DTiles::Tile& child_tile = tile.children[i];
            #pragma omp task shared(child_tile)
            children[i]->simplify_if_possible(child_tile);
        }
        #pragma omp taskwait
    }
}
void SegmentTreeNode::collect_segments(std::vector<MeshSegment>& segments)
{
    if (!skipped)
    {
        segments.push_back(meta_segment);
    }
    for (auto& child : children)
    {
        child->collect_segments(segments);
    }
}
void SegmentTreeNode::print(size_t depth)
{
    std::cout << std::string(depth, ' ') << "Node";
    if (!meta_segment.filename.empty())
    {
        std::cout << " " << meta_segment.filename;
    }
    if (meta_segment.mesh != nullptr)
    {
        std::cout << "(" << meta_segment.mesh->n_faces() << ")";
    }
    std::cout << std::endl;
    for (auto& child : children)
    {
        child->print(depth + 1);
    }
}

// SegmentTreeLeaf
void SegmentTreeLeaf::print(size_t depth)
{
    std::cout << std::string(depth, ' ') << "Leaf";
    if (!segment.filename.empty())
    {
        std::cout << " " << segment.filename;
    }
    if (segment.mesh != nullptr)
    {
        std::cout << "(" << segment.mesh->n_faces() << ")";
    }
    std::cout << std::endl;
}

} // namespace lvr2
