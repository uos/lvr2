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
 * B3dmWriter.cpp
 *
 * @date   24.01.2022
 * @author Malte Hillmann <mhillmann@uni-osnabrueck.de>
 */

#include "B3dmWriter.hpp"

#include "lvr2/algorithm/pmp/SurfaceNormals.h"

#include <CesiumGltf/Model.h>
#include <CesiumGltfWriter/GltfWriter.h>

namespace lvr2
{

/**
 * @brief creates a new value at the end of a vector and returns its index and a reference to that value
 *
 * @param vec the vector to append to
 * @return std::tuple<size_t, T&> the index and reference to the new value
 */
template<typename T>
inline std::tuple<size_t, T&> push_and_get_index(std::vector<T>& vec)
{
    size_t index = vec.size();
    vec.emplace_back();
    return std::make_tuple(index, std::ref(vec.back()));
}

/**
 * @brief Adds a vertex attribute channel to the model
 *
 * @tparam N the number of components of the attribute (1 - 4)
 * @param buffer the buffer to add the attribute to
 * @param model the model to add the attribute to
 * @param primitive the primitive to add the attribute to
 * @param name the name of the attribute
 * @param num_vertices the number of elements in the attribute
 * @param min the component-wise minimum value of the attribute
 * @param max the component-wise maximum value of the attribute
 * @return size_t the byte offset in the buffer where the attribute starts
 */
template<int N = 3>
size_t add_vertex_attribute(std::vector<std::byte>& buffer,
                            CesiumGltf::Model& model,
                            CesiumGltf::MeshPrimitive& primitive,
                            std::string name,
                            size_t num_vertices,
                            Eigen::Matrix<float, N, 1> min,
                            Eigen::Matrix<float, N, 1> max)
{
    static_assert(N <= 4 && N >= 1);

    size_t byte_offset = buffer.size();
    size_t byte_length = num_vertices * N * sizeof(float);
    buffer.resize(byte_offset + byte_length);

    auto [ buffer_view_id, buffer_view ] = push_and_get_index(model.bufferViews);
    buffer_view.buffer = 0;
    buffer_view.byteOffset = byte_offset;
    buffer_view.byteLength = byte_length;
    buffer_view.byteStride = N * sizeof(float);
    buffer_view.target = CesiumGltf::BufferView::Target::ARRAY_BUFFER;

    auto [ accessor_id, accessor ] = push_and_get_index(model.accessors);
    accessor.bufferView = buffer_view_id;
    accessor.count = num_vertices;
    accessor.componentType = CesiumGltf::Accessor::ComponentType::FLOAT;
    accessor.type = (N == 4 ? CesiumGltf::Accessor::Type::VEC4 :
                     N == 3 ? CesiumGltf::Accessor::Type::VEC3 :
                     N == 2 ? CesiumGltf::Accessor::Type::VEC2 :
                     CesiumGltf::Accessor::Type::SCALAR);

    for (int i = 0; i < N; i++)
    {
        accessor.min.push_back(min[i]);
        accessor.max.push_back(max[i]);
    }

    primitive.attributes[name] = accessor_id;

    return byte_offset;
}

/**
 * @brief writes a uint32_t to an output stream in binary format
 *
 * @param file the output stream
 * @param value the value to write
 */
inline void write_uint32(std::ofstream& file, uint32_t value)
{
    file.write(reinterpret_cast<char*>(&value), sizeof(uint32_t));
}


void write_b3dm(const boost::filesystem::path& filename, PMPMesh<BaseVector<float>>& mesh, pmp::BoundingBox& bb)
{
    std::cout << timestamp << "Writing " << filename << std::endl;

    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Error opening file " << filename << std::endl;
        return;
    }

    CesiumGltf::Model model;
    model.asset.generator = "lvr2";
    model.asset.version = "2.0";

    auto [ out_mesh_id, out_mesh ] = push_and_get_index(model.meshes);

    auto [ primitive_id, primitive ] = push_and_get_index(out_mesh.primitives);
    primitive.mode = CesiumGltf::MeshPrimitive::Mode::TRIANGLES;

    auto [ node_id, node ] = push_and_get_index(model.nodes);
    node.mesh = out_mesh_id;
    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#gltf-transforms
    node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

    auto [ scene_id, scene ] = push_and_get_index(model.scenes);
    scene.nodes.push_back(node_id);

    model.scene = scene_id;

    auto [ buffer_id, raw_buffer ] = push_and_get_index(model.buffers);
    auto& buffer = raw_buffer.cesium.data;

    auto& surface_mesh = mesh.getSurfaceMesh();
    size_t num_vertices = surface_mesh.n_vertices();

    // vertices might have been deleted, so map from VertexHandle to buffer index
    DenseVertexMap<size_t> vertex_id_map(surface_mesh.vertices_size());

    if (!surface_mesh.has_vertex_property("v:normal"))
    {
        pmp::SurfaceNormals::compute_vertex_normals(surface_mesh);
    }
    const auto& normals = surface_mesh.get_vertex_property<pmp::Normal>("v:normal");

    bool has_color = surface_mesh.has_vertex_property("v:color");
    const auto& colors = surface_mesh.get_vertex_property<pmp::Color>("v:color"); // might be invalid

    bool has_tex = surface_mesh.has_vertex_property("v:tex");
    const auto& tex = surface_mesh.get_vertex_property<pmp::TexCoord>("v:tex"); // might be invalid

    {
        size_t vertex_byte_offset = add_vertex_attribute(buffer, model, primitive, "POSITION", num_vertices, bb.min(), bb.max());

        size_t normal_byte_offset = add_vertex_attribute(buffer, model, primitive, "NORMAL", num_vertices, pmp::Point(-1, -1, -1), pmp::Point(1, 1, 1));

        size_t color_byte_offset;
        if (has_color)
        {
            color_byte_offset = add_vertex_attribute(buffer, model, primitive, "COLOR_0", num_vertices, pmp::Color(0, 0, 0), pmp::Color(1, 1, 1));
        }

        size_t tex_byte_offset;
        if (has_tex)
        {
            tex_byte_offset = add_vertex_attribute<2>(buffer, model, primitive, "TEXCOORD_0", num_vertices, pmp::TexCoord(0, 0), pmp::TexCoord(1, 1));
        }

        float* vertex_out = (float*)(buffer.data() + vertex_byte_offset);
        float* normals_out = (float*)(buffer.data() + normal_byte_offset);
        float* colors_out = nullptr;
        float* tex_out = nullptr;
        if (has_color)
        {
            colors_out = (float*)(buffer.data() + color_byte_offset);
        }
        if (has_tex)
        {
            tex_out = (float*)(buffer.data() + tex_byte_offset);
        }

        auto it = surface_mesh.vertices_begin();
        for (size_t i = 0; i < num_vertices; ++i, ++it)
        {
            vertex_id_map[*it] = i;
            std::copy_n(surface_mesh.position(*it).data(), 3, vertex_out);
            vertex_out += 3;

            std::copy_n(normals[*it].data(), 3, normals_out);
            normals_out += 3;

            if (has_color)
            {
                std::copy_n(colors[*it].data(), 3, colors_out);
                colors_out += 3;
            }

            if (has_tex)
            {
                std::copy_n(tex[*it].data(), 2, tex_out);
                tex_out += 2;
            }
        }
    }

    {
        size_t num_triangles = mesh.numFaces();
        size_t byte_offset = buffer.size();
        size_t byte_length = num_triangles * 3 * sizeof(uint32_t);
        buffer.resize(byte_offset + byte_length);
        uint32_t* out = (uint32_t*)(buffer.data() + byte_offset);
        for (const auto fH : surface_mesh.faces())
        {
            auto vertex_iter = mesh.getSurfaceMesh().vertices(fH);
            out[0] = vertex_id_map[*vertex_iter];
            out[1] = vertex_id_map[*(++vertex_iter)];
            out[2] = vertex_id_map[*(++vertex_iter)];
            out += 3;
        }

        auto [ buffer_view_id, buffer_view ] = push_and_get_index(model.bufferViews);
        buffer_view.buffer = buffer_id;
        buffer_view.byteOffset = byte_offset;
        buffer_view.byteLength = byte_length;
        buffer_view.target = CesiumGltf::BufferView::Target::ELEMENT_ARRAY_BUFFER;

        auto [ accessor_id, accessor ] = push_and_get_index(model.accessors);
        accessor.bufferView = buffer_view_id;
        accessor.count = num_triangles * 3;
        accessor.componentType = CesiumGltf::Accessor::ComponentType::UNSIGNED_INT;
        accessor.type = CesiumGltf::Accessor::Type::SCALAR;
        accessor.min = { 0 };
        accessor.max = { (double)num_vertices - 1 };

        primitive.indices = accessor_id;
    }

    CesiumGltfWriter::GltfWriter writer;
    auto gltf = writer.writeGlb(model, buffer);
    if (!gltf.warnings.empty())
    {
        std::cerr << "Warnings writing gltf: " << std::endl;
        for (auto& e : gltf.warnings)
        {
            std::cerr << e << std::endl;
        }
    }
    if (!gltf.errors.empty())
    {
        std::cerr << "Errors writing gltf: " << std::endl;
        for (auto& e : gltf.errors)
        {
            std::cerr << e << std::endl;
        }
        throw std::runtime_error("Failed to write gltf");
    }

    std::string feature_table = "{\"BATCH_LENGTH\":0}";

    std::string magic = "b3dm";
    uint32_t version = 1;
    uint32_t byte_length = 0;
    uint32_t feature_table_json_length = feature_table.length();
    uint32_t feature_table_byte_length = 0;
    uint32_t batch_table_json_length = 0;
    uint32_t batch_table_byte_length = 0;

    size_t header_length = magic.length()
                           + 6 * sizeof(uint32_t)
                           + feature_table_json_length
                           + feature_table_byte_length
                           + batch_table_json_length
                           + batch_table_byte_length;

    while (header_length % 8 != 0)
    {
        // gltf has to start on a multiple of 8 bytes, so pad the feature table to match
        feature_table += ' ';
        feature_table_json_length++;
        header_length++;
    }

    byte_length = header_length + gltf.gltfBytes.size();

    file << magic;
    write_uint32(file, version);
    write_uint32(file, byte_length);
    write_uint32(file, feature_table_json_length);
    write_uint32(file, feature_table_byte_length);
    write_uint32(file, batch_table_json_length);
    write_uint32(file, batch_table_byte_length);

    file << feature_table;

    file.write((char*)gltf.gltfBytes.data(), gltf.gltfBytes.size());
}

} // namespace lvr2
