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
 * @file       B3dmIO.cpp
 * @brief      I/O support for b3dm files.
 * @details    I/O support for b3dm files: Reading and writing meshes, including
 *             color information, textures and normals.
 * @author     Malte Hillmann <mhillmann@uni-osnabrueck.de>
 * @date       13.06.2022
 */

#ifdef LVR2_USE_3DTILES

#include "lvr2/io/modelio/B3dmIO.hpp"

#include <CesiumGltf/Model.h>
#include <CesiumGltfWriter/GltfWriter.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

using namespace CesiumGltf;

namespace lvr2
{

/**
 * @brief creates a new value at the end of a vector and returns its index and a reference to that value
 *
 * @param vec the vector to append to
 * @return std::pair<size_t, T&> the index and reference to the new value
 */
template<typename T>
inline std::pair<size_t, T&> push_id(std::vector<T>& vec)
{
    size_t index = vec.size();
    vec.emplace_back();
    return std::make_pair(index, std::ref(vec.back()));
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

using DynVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

const std::string& accessor_type(size_t elements_per_vertex)
{
    switch (elements_per_vertex)
    {
    case 1: return Accessor::Type::SCALAR;
    case 2: return Accessor::Type::VEC2;
    case 3: return Accessor::Type::VEC3;
    case 4: return Accessor::Type::VEC4;
    default: throw std::runtime_error("unsupported number of elements per vertex");
    }
}

float* add_attribute(std::vector<std::byte>& buffer,
                     CesiumGltf::Model& model,
                     size_t num_vertices,
                     size_t elements_per_vertex,
                     const std::string& name,
                     DynVector min, DynVector max)
{
    size_t byte_length = num_vertices * elements_per_vertex * sizeof(float);

    auto [ buffer_view_id, buffer_view ] = push_id(model.bufferViews);
    buffer_view.buffer = 0;
    buffer_view.byteOffset = buffer.size();
    buffer_view.byteLength = byte_length;
    buffer_view.byteStride = elements_per_vertex * sizeof(float);
    buffer_view.target = BufferView::Target::ARRAY_BUFFER;

    auto [ accessor_id, accessor ] = push_id(model.accessors);
    accessor.bufferView = buffer_view_id;
    accessor.count = num_vertices;
    accessor.componentType = Accessor::ComponentType::FLOAT;
    accessor.type = accessor_type(elements_per_vertex);

    for (int i = 0; i < elements_per_vertex; i++)
    {
        accessor.min.push_back(min[i]);
        accessor.max.push_back(max[i]);
    }

    model.meshes[0].primitives[0].attributes[name] = accessor_id;

    buffer.resize(buffer.size() + byte_length);
    return reinterpret_cast<float*>(buffer.data() + buffer_view.byteOffset);
}

void B3dmIO::save(std::string filename)
{
    auto mesh = m_model->m_mesh;
    if (!mesh)
    {
        throw std::runtime_error("B3dmIO can only save meshes");
    }

    size_t num_vertices = mesh->numVertices();
    size_t num_faces = mesh->numFaces();

    bool has_normals = mesh->hasVertexNormals();
    bool has_colors = mesh->hasVertexColors();
    bool has_tex = mesh->hasChannel<float>("texture_coordinates");
    has_tex = has_tex && !mesh->getTextures().empty();

    has_colors = has_colors && !has_tex; // prefer textures over colors

    if (has_tex && mesh->getTextures().size() > 1)
    {
        std::cout << "Warning: multiple textures are not supported by the b3dm format." << std::endl;
    }

    pmp::BoundingBox bb;
    {
        auto points = mesh->getVertices();
        auto it = points.get(), end = points.get() + num_vertices * 3;
        for (; it != end; it += 3)
        {
            bb += pmp::Point(it[0], it[1], it[2]);
        }
    }

    std::vector<unsigned char> texture;
    if (has_tex)
    {
        auto& tex = mesh->getTextures()[0];
        cv::Mat image(tex.m_height, tex.m_width, CV_8UC3, tex.m_data);
        cv::Mat bgr;
        cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);

        std::vector<int> compression_params;
        compression_params.push_back(cv::IMWRITE_PNG_COMPRESSION);
        compression_params.push_back(9); // maximum compression
        cv::imencode(".png", bgr, texture, compression_params);
    }

    // ==================== Add Model Metadata ====================

    CesiumGltf::Model model;
    model.asset.generator = "lvr2";
    model.asset.version = "2.0";

    auto [ out_mesh_id, out_mesh ] = push_id(model.meshes);

    auto [ material_id, material ] = push_id(model.materials);
    material.doubleSided = true;

    auto [ primitive_id, primitive ] = push_id(out_mesh.primitives);
    primitive.mode = MeshPrimitive::Mode::TRIANGLES;
    primitive.material = material_id;

    auto [ node_id, node ] = push_id(model.nodes);
    node.mesh = out_mesh_id;
    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#y-up-to-z-up
    node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

    auto [ scene_id, scene ] = push_id(model.scenes);
    scene.nodes.push_back(node_id);

    model.scene = scene_id;

    // ==================== Add Attributes ====================

    auto& raw_buffer = model.buffers.emplace_back();
    auto& buffer = raw_buffer.cesium.data;

    float* out = add_attribute(buffer, model, num_vertices, 3, "POSITION",
                               bb.min(), bb.max());
    std::copy_n(mesh->getVertices().get(), num_vertices * 3, out);

    if (has_normals)
    {
        out = add_attribute(buffer, model, num_vertices, 3, "NORMAL",
                            pmp::Point(-1, -1, -1), pmp::Point(1, 1, 1));
        std::copy_n(mesh->getVertexNormals().get(), num_vertices * 3, out);
    }
    if (has_colors)
    {
        out = add_attribute(buffer, model, num_vertices, 3, "COLOR_0",
                            pmp::Point(0, 0, 0), pmp::Point(1, 1, 1));

        size_t w_in, w_out = 3;
        auto colors_in = mesh->getVertexColors(w_in).get();
        for (size_t i = 0; i < num_vertices; i++)
        {
            out[i * w_out + 0] = colors_in[i * w_in + 0] / 255.0f;
            out[i * w_out + 1] = colors_in[i * w_in + 1] / 255.0f;
            out[i * w_out + 2] = colors_in[i * w_in + 2] / 255.0f;
        }
    }
    if (has_tex)
    {
        out = add_attribute(buffer, model, num_vertices, 2, "TEXCOORD_0",
                            pmp::Point(0, 0), pmp::Point(1, 1));
        std::copy_n(mesh->getTextureCoordinates().get(), num_vertices * 2, out);

        size_t texture_offset = buffer.size();
        buffer.resize(buffer.size() + texture.size());
        unsigned char* out = (unsigned char*)(buffer.data() + texture_offset);
        std::copy_n(texture.data(), texture.size(), out);

        auto [ texture_buffer_view_id, texture_buffer_view ] = push_id(model.bufferViews);
        texture_buffer_view.buffer = 0;
        texture_buffer_view.byteOffset = texture_offset;
        texture_buffer_view.byteLength = texture.size();

        auto [ image_id, image ] = push_id(model.images);
        image.bufferView = texture_buffer_view_id;
        image.mimeType = ImageSpec::MimeType::image_png;

        auto [ sampler_id, sampler ] = push_id(model.samplers);
        sampler.magFilter = Sampler::MagFilter::LINEAR;
        sampler.minFilter = Sampler::MinFilter::LINEAR_MIPMAP_LINEAR;
        sampler.wrapS = Sampler::WrapS::CLAMP_TO_EDGE;
        sampler.wrapT = Sampler::WrapT::CLAMP_TO_EDGE;

        auto [ texture_id, texture ] = push_id(model.textures);
        texture.source = image_id;
        texture.sampler = sampler_id;

        TextureInfo info;
        info.index = texture_id;

        MaterialPBRMetallicRoughness pbr;
        pbr.baseColorTexture = info;
        pbr.metallicFactor = 0;
        pbr.roughnessFactor = 1;

        material.pbrMetallicRoughness = pbr;
    }

    // add face metadata
    size_t faces_offset = buffer.size();
    size_t faces_byte_length = num_faces * 3 * sizeof(uint32_t);
    buffer.resize(buffer.size() + faces_byte_length);
    uint32_t* faces_out = (uint32_t*)(buffer.data() + faces_offset);
    std::copy_n(mesh->getFaceIndices().get(), num_faces * 3, faces_out);

    auto [ face_buffer_view_id, face_buffer_view ] = push_id(model.bufferViews);
    face_buffer_view.buffer = 0;
    face_buffer_view.byteOffset = faces_offset;
    face_buffer_view.byteLength = faces_byte_length;
    face_buffer_view.target = BufferView::Target::ELEMENT_ARRAY_BUFFER;

    auto [ face_accessor_id, face_accessor ] = push_id(model.accessors);
    face_accessor.bufferView = face_buffer_view_id;
    face_accessor.count = num_faces * 3;
    face_accessor.componentType = Accessor::ComponentType::UNSIGNED_INT;
    face_accessor.type = Accessor::Type::SCALAR;
    face_accessor.min = { 0 };
    face_accessor.max = { (double)num_vertices - 1 };

    primitive.indices = face_accessor_id;


    raw_buffer.byteLength = buffer.size();

    // ==================== Write to file ====================

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
                            + 6 * sizeof(uint32_t) // number of uint32_t after the magic (version etc.)
                            + feature_table_json_length
                            + feature_table_byte_length
                            + batch_table_json_length
                            + batch_table_byte_length;

    while (header_length % 8 != 0)
    {
        // gltf has to start on a multiple of 8 bytes, so pad the feature table to match
        feature_table.push_back(' ');
        feature_table_json_length++;
        header_length++;
    }

    byte_length = header_length + gltf.gltfBytes.size();

    std::ofstream file(filename, std::ios::binary);

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

#endif // LVR2_USE_3DTILES
