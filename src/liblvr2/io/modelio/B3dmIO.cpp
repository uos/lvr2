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

#include "lvr2/io/modelio/B3dmIO.hpp"

#include "lvr2/util/Logging.hpp"

#include "lvr2/io/modelio/DracoEncoder.hpp"
#include "lvr2/io/modelio/DracoDecoder.hpp"
#include <draco/io/gltf_encoder.h>
#include <draco/io/gltf_decoder.h>
#include <draco/scene/scene.h>

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
inline void read_uint32(std::ifstream& file, uint32_t& value)
{
    file.read(reinterpret_cast<char*>(&value), sizeof(uint32_t));
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

void convert_texture(const Texture& texture, std::vector<uint8_t>& output)
{
    cv::Mat image(texture.m_height, texture.m_width, CV_8UC3, texture.m_data);
    cv::Mat bgr;
    cv::cvtColor(image, bgr, cv::COLOR_RGB2BGR);

    cv::imencode(".webp", bgr, output);
}
const char* MIME_TYPE = "image/webp";

void write_header(std::ofstream& file, size_t body_length);

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

    // ==================== Add Model Metadata ====================

    CesiumGltf::Model model;
    model.asset.generator = "lvr2";
    model.asset.version = "2.0";

    auto [ out_mesh_id, out_mesh ] = push_id(model.meshes);

    auto [ material_id, material ] = push_id(model.materials);
    material.doubleSided = true;

    MaterialPBRMetallicRoughness pbr;
    pbr.metallicFactor = 0;
    pbr.roughnessFactor = 1;
    material.pbrMetallicRoughness = pbr;

    auto [ primitive_id, primitive ] = push_id(out_mesh.primitives);
    primitive.mode = MeshPrimitive::Mode::TRIANGLES;
    primitive.material = material_id;

    auto [ node_id, node ] = push_id(model.nodes);
    node.mesh = out_mesh_id;
    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#y-up-to-z-up
    node.matrix = { 1, 0, 0, 0,
                    0, 0, -1, 0,
                    0, 1, 0, 0,
                    0, 0, 0, 1 };

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
        std::vector<uint8_t> texture_bytes;
        convert_texture(mesh->getTextures()[0], texture_bytes);
        buffer.resize(buffer.size() + texture_bytes.size());
        uint8_t* out = (uint8_t*)(buffer.data() + texture_offset);
        std::copy_n(texture_bytes.data(), texture_bytes.size(), out);

        auto [ texture_buffer_view_id, texture_buffer_view ] = push_id(model.bufferViews);
        texture_buffer_view.buffer = 0;
        texture_buffer_view.byteOffset = texture_offset;
        texture_buffer_view.byteLength = texture_bytes.size();

        auto [ image_id, image ] = push_id(model.images);
        image.bufferView = texture_buffer_view_id;
        image.mimeType = MIME_TYPE;

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
        material.pbrMetallicRoughness->baseColorTexture = info;
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

    std::ofstream file(filename, std::ios::binary);

    write_header(file, gltf.gltfBytes.size());

    file.write((char*)gltf.gltfBytes.data(), gltf.gltfBytes.size());
}

void B3dmIO::saveCompressed(const std::string& filename)
{
    auto mesh = toDracoMesh(m_model);

    mesh->SetCompressionEnabled(true);
    draco::DracoCompressionOptions options;
    options.compression_level = 10;
    options.quantization_bits_position = 0; // causes very visible artifacts
    options.quantization_bits_tex_coord = 0;
    mesh->SetCompressionOptions(options);

    draco::Scene scene;

    auto mesh_id = scene.AddMesh(std::move(mesh));

    auto& materials = scene.GetMaterialLibrary();
    int material_id = 0;
    auto material = materials.MutableMaterial(material_id);
    material->SetDoubleSided(true);
    material->SetMetallicFactor(0);
    material->SetRoughnessFactor(1);

    if (m_model->m_mesh->getTextures().size() == 1)
    {
        auto texture = m_model->m_mesh->getTextures()[0];
        auto draco_texture = std::make_unique<draco::Texture>();
        convert_texture(texture, draco_texture->source_image().MutableEncodedData());

        draco_texture->source_image().set_mime_type(MIME_TYPE);

        material->SetTextureMap(std::move(draco_texture), draco::TextureMap::Type::COLOR, 0);

        material->GetTextureMapByIndex(0)->SetProperties(
            draco::TextureMap::Type::COLOR,
            draco::TextureMap::WrappingMode(draco::TextureMap::AxisWrappingMode::CLAMP_TO_EDGE),
            0,
            draco::TextureMap::FilterType::LINEAR,
            draco::TextureMap::FilterType::LINEAR_MIPMAP_LINEAR
        );
    }

    auto mesh_group_id = scene.AddMeshGroup();
    auto mesh_group = scene.GetMeshGroup(mesh_group_id);
    mesh_group->AddMeshIndex(mesh_id);
    mesh_group->AddMaterialIndex(material_id);

    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#y-up-to-z-up
    Eigen::Matrix4d transform = Eigen::Matrix4d::Identity();
    transform << 1, 0, 0, 0,
                 0, 0, 1, 0,
                 0, -1, 0, 0,
                 0, 0, 0, 1;
    draco::TrsMatrix matrix;
    matrix.SetMatrix(transform);

    auto node_id = scene.AddNode();
    auto node = scene.GetNode(node_id);
    node->SetMeshGroupIndex(mesh_group_id);
    node->SetTrsMatrix(matrix);
    scene.AddRootNodeIndex(node_id);

    draco::GltfEncoder encoder;
    draco::EncoderBuffer buffer;
    auto status = encoder.EncodeToBuffer(scene, &buffer);
    if (!status.ok())
    {
        throw std::runtime_error("draco encoding failed: " + status.error_msg_string());
    }

    std::ofstream file(filename, std::ios::binary);

    write_header(file, buffer.size());

    file.write(buffer.data(), buffer.size());
}

void write_header(std::ofstream& file, size_t body_length)
{
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

    byte_length = header_length + body_length;

    file << magic;
    write_uint32(file, version);
    write_uint32(file, byte_length);
    write_uint32(file, feature_table_json_length);
    write_uint32(file, feature_table_byte_length);
    write_uint32(file, batch_table_json_length);
    write_uint32(file, batch_table_byte_length);

    file << feature_table;
}

ModelPtr B3dmIO::read(std::string filename)
{
    std::ifstream file(filename, std::ios::binary);

    char magic[5];
    file.read(magic, 4);
    magic[4] = '\0';
    if (strcmp(magic, "b3dm") != 0)
    {
        std::cout << timestamp << "B3dmIO: Not a b3dm file" << std::endl;
        return nullptr;
    }
    uint32_t version, byte_length, feature_table_json_length, feature_table_byte_length, batch_table_json_length, batch_table_byte_length;
    read_uint32(file, version);
    read_uint32(file, byte_length);
    read_uint32(file, feature_table_json_length);
    read_uint32(file, feature_table_byte_length);
    read_uint32(file, batch_table_json_length);
    read_uint32(file, batch_table_byte_length);

    std::string feature_table(feature_table_json_length, ' ');
    file.read(feature_table.data(), feature_table_json_length);

    size_t header_length = 4 + 6 * sizeof(uint32_t)
                        + feature_table_json_length
                        + feature_table_byte_length
                        + batch_table_json_length
                        + batch_table_byte_length;

    size_t body_length = byte_length - header_length;

    draco::DecoderBuffer buffer;
    auto buffer_data = std::make_unique<char[]>(body_length);
    file.read(buffer_data.get(), body_length);
    buffer.Init(buffer_data.get(), body_length);

    draco::GltfDecoder decoder;
    auto res = decoder.DecodeFromBufferToScene(&buffer);
    if (!res.ok())
    {
        std::cout << "B3dmIO: draco decoding failed: " << res.status().error_msg_string() << std::endl;
        return nullptr;
    }
    auto scene = std::move(res).value();

    auto node_id = scene->GetRootNodeIndex(0);
    auto node = scene->GetNode(node_id);

    auto mesh_group_id = node->GetMeshGroupIndex();
    auto mesh_group = scene->GetMeshGroup(mesh_group_id);
    if (mesh_group->NumMeshIndices() > 1)
    {
        std::cout << "B3dmIO Warning: multiple meshes are not supported. Ignoring all but the first." << std::endl;
    }

    auto mesh_id = mesh_group->GetMeshIndex(0);
    auto& mesh = scene->GetMesh(mesh_id);

    auto model = fromDracoMesh(mesh);
    if (!model)
    {
        std::cout << "B3dmIO Warning: failed to convert mesh to model" << std::endl;
        return nullptr;
    }

    auto& materials = scene->GetMaterialLibrary();
    auto material_id = mesh_group->GetMaterialIndex(0);
    auto material = materials.GetMaterial(material_id);
    auto texture_map = material->GetTextureMapByType(draco::TextureMap::Type::COLOR);

    if (texture_map)
    {
        auto data = texture_map->texture()->source_image().encoded_data();
        auto img = cv::imdecode(data, cv::IMREAD_COLOR);
        if (!img.data)
        {
            std::cout << "B3dmIO Warning: failed to decode texture. Ignoring." << std::endl;
        }
        else
        {
            cv::Mat rgb;
            cv::cvtColor(img, rgb, cv::COLOR_BGR2RGB);

            Texture tex(0, rgb.cols, rgb.rows, 3, 1, 1.0, rgb.data);
            model->m_mesh->getTextures().emplace_back(std::move(tex));
        }
    }

    return model;
}

} // namespace lvr2
