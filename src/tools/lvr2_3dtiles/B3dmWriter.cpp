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

#include <CesiumGltf/Model.h>
#include <CesiumGltfWriter/GltfWriter.h>

namespace lvr2
{

using DynVector = Eigen::Matrix<float, Eigen::Dynamic, 1>;

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

struct PropertyWriter
{
    std::string name;
    size_t byte_offset;
    size_t byte_length;
    size_t num_vertices;
    size_t elements_per_vertex;
    DynVector min, max;
    float* data_out = nullptr;
    const float* data_in;

    PropertyWriter(std::string name, size_t elements_per_vertex, const float* data_in,
                   DynVector min, DynVector max)
        : name(name), data_in(data_in), elements_per_vertex(elements_per_vertex), min(min), max(max)
    {
        if (elements_per_vertex < 1 || elements_per_vertex > 4)
        {
            throw std::invalid_argument("PropertyWriter: elements_per_vertex must be between 1 and 4");
        }
    }

    size_t update_offset(size_t offset, size_t num_vertices)
    {
        byte_offset = offset;
        this->num_vertices = num_vertices;
        byte_length = num_vertices * elements_per_vertex * sizeof(float);
        return offset + byte_length;
    }

    void add_metadata(std::vector<std::byte>& buffer,
                      CesiumGltf::Model& model,
                      CesiumGltf::MeshPrimitive& primitive)
    {
        data_out = reinterpret_cast<float*>(buffer.data() + byte_offset);

        auto [ buffer_view_id, buffer_view ] = push_and_get_index(model.bufferViews);
        buffer_view.buffer = 0;
        buffer_view.byteOffset = byte_offset;
        buffer_view.byteLength = byte_length;
        buffer_view.byteStride = elements_per_vertex * sizeof(float);
        buffer_view.target = CesiumGltf::BufferView::Target::ARRAY_BUFFER;

        auto [ accessor_id, accessor ] = push_and_get_index(model.accessors);
        accessor.bufferView = buffer_view_id;
        accessor.count = num_vertices;
        accessor.componentType = CesiumGltf::Accessor::ComponentType::FLOAT;
        accessor.type = (elements_per_vertex == 4 ? CesiumGltf::Accessor::Type::VEC4 :
                         elements_per_vertex == 3 ? CesiumGltf::Accessor::Type::VEC3 :
                         elements_per_vertex == 2 ? CesiumGltf::Accessor::Type::VEC2 :
                         CesiumGltf::Accessor::Type::SCALAR);

        for (int i = 0; i < elements_per_vertex; i++)
        {
            accessor.min.push_back(min[i]);
            accessor.max.push_back(max[i]);
        }

        primitive.attributes[name] = accessor_id;
    }

    void add_value(size_t id_in, size_t id_out)
    {
        std::copy_n(data_in + id_in * elements_per_vertex,
                    elements_per_vertex,
                    data_out + id_out * elements_per_vertex);
    }
    void copy_all()
    {
        std::copy_n(data_in, num_vertices * elements_per_vertex, data_out);
    }
};


void write_b3dm(const fs::path& output_dir, const MeshSegment& segment, bool print_progress)
{
    std::string output_path = (output_dir / segment.filename).string();
    // create the file to ensure all the paths exist and are possible
    if (!std::ofstream(output_path, std::ios::binary))
    {
        throw std::runtime_error("Could not open " + output_path);
    }

    auto pmp_mesh = segment.mesh->get();
    auto& mesh = pmp_mesh->getSurfaceMesh();

    if (mesh.n_faces() < mesh.faces_size() * 2 / 3)
    {
        mesh.garbage_collection();
    }

    const auto positions = mesh.get_vertex_property<pmp::Point>("v:point");
    const auto normals = mesh.get_vertex_property<pmp::Normal>("v:normal");
    const auto colors = mesh.get_vertex_property<pmp::Color>("v:color");
    const auto tex = mesh.get_vertex_property<pmp::TexCoord>("v:tex");

    if (!tex && segment.texture_file)
    {
        throw std::runtime_error("Texture file specified, but no texture coordinates found");
    }
    bool use_tex = tex && segment.texture_file;

    CesiumGltf::Model model;
    std::vector<std::byte> buffer;
    std::vector<PropertyWriter> property_writers;

    model.asset.generator = "lvr2";
    model.asset.version = "2.0";

    auto [ out_mesh_id, out_mesh ] = push_and_get_index(model.meshes);

    auto [ material_id, material ] = push_and_get_index(model.materials);
    material.doubleSided = true;

    auto [ primitive_id, primitive ] = push_and_get_index(out_mesh.primitives);
    primitive.mode = CesiumGltf::MeshPrimitive::Mode::TRIANGLES;
    primitive.material = material_id;

    if (use_tex)
    {
        auto [ image_id, image ] = push_and_get_index(model.images);
        image.uri = *segment.texture_file;

        auto [ sampler_id, sampler ] = push_and_get_index(model.samplers);
        using CesiumGltf::Sampler;
        sampler.magFilter = Sampler::MagFilter::LINEAR;
        sampler.minFilter = Sampler::MinFilter::LINEAR_MIPMAP_LINEAR;
        sampler.wrapS = Sampler::WrapS::CLAMP_TO_EDGE;
        sampler.wrapT = Sampler::WrapT::CLAMP_TO_EDGE;

        auto [ texture_id, texture ] = push_and_get_index(model.textures);
        texture.source = image_id;
        texture.sampler = sampler_id;

        CesiumGltf::TextureInfo info;
        info.index = texture_id;

        CesiumGltf::MaterialPBRMetallicRoughness pbr;
        pbr.baseColorTexture = info;
        pbr.metallicFactor = 0;
        pbr.roughnessFactor = 0.5;

        material.pbrMetallicRoughness = pbr;
    }

    auto [ node_id, node ] = push_and_get_index(model.nodes);
    node.mesh = out_mesh_id;
    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#y-up-to-z-up
    node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

    auto [ scene_id, scene ] = push_and_get_index(model.scenes);
    scene.nodes.push_back(node_id);

    model.scene = scene_id;

    property_writers.emplace_back("POSITION", 3, (float*)positions.data(), segment.bb.min(), segment.bb.max());

    if (normals)
    {
        property_writers.emplace_back("NORMAL", 3, (float*)normals.data(), pmp::Point(-1, -1, -1), pmp::Point(1, 1, 1));
    }
    if (colors && !use_tex)
    {
        property_writers.emplace_back("COLOR_0", 3, (float*)colors.data(), pmp::Color(0, 0, 0), pmp::Color(1, 1, 1));
    }
    if (use_tex)
    {
        property_writers.emplace_back("TEXCOORD_0", 2, (float*)tex.data(), pmp::TexCoord(0, 0), pmp::TexCoord(1, 1));
    }

    size_t byte_offset = 0;
    for (auto& writer : property_writers)
    {
        byte_offset = writer.update_offset(byte_offset, mesh.n_vertices());
    }

    // add face metadata
    size_t face_byte_offset = byte_offset;
    size_t face_byte_length = mesh.n_faces() * 3 * sizeof(uint32_t);
    byte_offset += face_byte_length;

    auto [ face_buffer_view_id, face_buffer_view ] = push_and_get_index(model.bufferViews);
    face_buffer_view.buffer = 0;
    face_buffer_view.byteOffset = face_byte_offset;
    face_buffer_view.byteLength = face_byte_length;
    face_buffer_view.target = CesiumGltf::BufferView::Target::ELEMENT_ARRAY_BUFFER;

    auto [ face_accessor_id, face_accessor ] = push_and_get_index(model.accessors);
    face_accessor.bufferView = face_buffer_view_id;
    face_accessor.count = mesh.n_faces() * 3;
    face_accessor.componentType = CesiumGltf::Accessor::ComponentType::UNSIGNED_INT;
    face_accessor.type = CesiumGltf::Accessor::Type::SCALAR;
    face_accessor.min = { 0 };
    face_accessor.max = { (double)mesh.n_vertices() - 1 };

    primitive.indices = face_accessor_id;


    auto [ buffer_id, raw_buffer ] = push_and_get_index(model.buffers);
    size_t total_byte_length = byte_offset;
    raw_buffer.byteLength = total_byte_length;

    buffer.resize(total_byte_length);
    for (auto& writer : property_writers)
    {
        writer.add_metadata(buffer, model, primitive);
    }
    uint32_t* face_out = (uint32_t*)(buffer.data() + face_byte_offset);

    ProgressBar* progress = nullptr;
    if (print_progress)
    {
        progress = new ProgressBar(mesh.n_faces(), "Writing Data");
    }

    for (auto& writer : property_writers)
    {
        writer.copy_all();
    }
    for (auto fH : mesh.faces())
    {
        for (auto vH : mesh.vertices(fH))
        {
            *face_out++ = vH.idx();
        }
        if (print_progress)
        {
            ++(*progress);
        }
    }
    std::cout << "\r";

    // consistency check
    auto face_difference = face_out - (uint32_t*)(buffer.data() + buffer.size());
    if (face_difference % 3 != 0)
    {
        std::cerr << "Mesh had a non-triangle" << std::endl;
    }
    face_difference /= 3;
    if (face_difference > 0)
    {
        std::cerr << "Mesh added " << face_difference << " too many faces" << std::endl;
    }
    else if (face_difference < 0)
    {
        std::cerr << "Mesh added " << -face_difference << " too few faces" << std::endl;
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

    std::ofstream file(output_path, std::ios::binary);

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
