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
                   size_t byte_offset, size_t num_vertices,
                   DynVector min, DynVector max)
        : name(name), byte_offset(byte_offset), data_in(data_in), num_vertices(num_vertices),
          elements_per_vertex(elements_per_vertex), min(min), max(max)
    {
        if (elements_per_vertex < 1 || elements_per_vertex > 4)
        {
            throw std::invalid_argument("PropertyWriter: elements_per_vertex must be between 1 and 4");
        }
        byte_length = num_vertices * elements_per_vertex * sizeof(float);
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
};


void write_b3dm(const boost::filesystem::path& output_dir,
                const pmp::SurfaceMesh& mesh,
                const std::vector<Segment>& segments)
{
    std::vector<std::string> b3dm_files;
    std::vector<std::string> bin_files;
    std::vector<std::string> local_bin_filenames;
    for (const auto& segment : segments)
    {
        boost::filesystem::path b3dm_file_name = output_dir / segment.filename;
        b3dm_files.push_back(b3dm_file_name.string());
        boost::filesystem::path bin_file_name = b3dm_file_name;
        bin_file_name.replace_extension(".bin");
        bin_files.push_back(bin_file_name.string());
        local_bin_filenames.push_back(bin_file_name.filename().string());

        // create the files to ensure all the paths exist and are possible
        std::ofstream b3dm_file(b3dm_files.back(), std::ios::binary);
        std::ofstream bin_file(bin_files.back(), std::ios::binary);
        if (!b3dm_file)
        {
            throw std::runtime_error("Could not open " + b3dm_files.back());
        }
        if (!bin_file)
        {
            throw std::runtime_error("Could not open " + bin_files.back());
        }
    }

    size_t num_segments = segments.size();

    const auto& positions = mesh.get_vertex_property<pmp::Point>("v:point");

    bool has_normal = mesh.has_vertex_property("v:normal");
    const auto& normals = mesh.get_vertex_property<pmp::Normal>("v:normal");

    bool has_color = mesh.has_vertex_property("v:color");
    const auto& colors = mesh.get_vertex_property<pmp::Color>("v:color");

    bool has_tex = mesh.has_vertex_property("v:tex");
    const auto& tex = mesh.get_vertex_property<pmp::TexCoord>("v:tex");

    auto f_segment = mesh.get_face_property<SegmentId>("f:segment");

    std::vector<CesiumGltf::Model> models(num_segments);
    std::vector<std::vector<std::byte>> buffers(num_segments);
    std::vector<std::unordered_map<VertexHandle, VertexHandle>> vertex_maps(num_segments);
    std::vector<std::vector<PropertyWriter>> property_writers(num_segments);
    std::vector<uint32_t*> face_outs(num_segments);

    for (size_t i = 0; i < num_segments; i++)
    {
        auto& model = models[i];
        auto& writers = property_writers[i];
        auto& vertex_map = vertex_maps[i];
        auto& segment = segments[i];

        model.asset.generator = "lvr2";
        model.asset.version = "2.0";

        auto [ out_mesh_id, out_mesh ] = push_and_get_index(model.meshes);

        auto [ primitive_id, primitive ] = push_and_get_index(out_mesh.primitives);
        primitive.mode = CesiumGltf::MeshPrimitive::Mode::TRIANGLES;

        auto [ node_id, node ] = push_and_get_index(model.nodes);
        node.mesh = out_mesh_id;
        // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
        // So we need to pre-undo that transformation to maintain consistency.
        // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#y-up-to-z-up
        node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

        auto [ scene_id, scene ] = push_and_get_index(model.scenes);
        scene.nodes.push_back(node_id);

        model.scene = scene_id;

        auto [ buffer_id, raw_buffer ] = push_and_get_index(model.buffers);
        raw_buffer.uri = local_bin_filenames[i];

        size_t byte_offset = 0;

        writers.emplace_back("POSITION", 3, (float*)positions.data(),
                             byte_offset, segment.num_vertices,
                             segment.bb.min(), segment.bb.max());
        byte_offset += writers.back().byte_length;

        if (has_normal)
        {
            writers.emplace_back("NORMAL", 3, (float*)normals.data(),
                                 byte_offset, segment.num_vertices,
                                 pmp::Point(-1, -1, -1), pmp::Point(1, 1, 1));
            byte_offset += writers.back().byte_length;
        }

        if (has_color)
        {
            writers.emplace_back("COLOR_0", 3, (float*)colors.data(),
                                 byte_offset, segment.num_vertices,
                                 pmp::Color(0, 0, 0), pmp::Color(1, 1, 1));
            byte_offset += writers.back().byte_length;
        }

        if (has_tex)
        {
            writers.emplace_back("TEXCOORD_0", 2, (float*)tex.data(),
                                 byte_offset, segment.num_vertices,
                                 pmp::TexCoord(0, 0), pmp::TexCoord(1, 1));
            byte_offset += writers.back().byte_length;
        }

        // add face metadata
        size_t face_byte_offset = byte_offset;
        size_t face_byte_length = segment.num_faces * 3 * sizeof(uint32_t);
        byte_offset += face_byte_length;

        auto [ face_buffer_view_id, face_buffer_view ] = push_and_get_index(model.bufferViews);
        face_buffer_view.buffer = 0;
        face_buffer_view.byteOffset = face_byte_offset;
        face_buffer_view.byteLength = face_byte_length;
        face_buffer_view.target = CesiumGltf::BufferView::Target::ELEMENT_ARRAY_BUFFER;

        auto [ face_accessor_id, face_accessor ] = push_and_get_index(model.accessors);
        face_accessor.bufferView = face_buffer_view_id;
        face_accessor.count = segment.num_faces * 3;
        face_accessor.componentType = CesiumGltf::Accessor::ComponentType::UNSIGNED_INT;
        face_accessor.type = CesiumGltf::Accessor::Type::SCALAR;
        face_accessor.min = { 0 };
        face_accessor.max = { (double)segment.num_vertices - 1 };

        primitive.indices = face_accessor_id;


        size_t total_byte_length = byte_offset;
        raw_buffer.byteLength = total_byte_length;

        auto& buffer = buffers[i];
        buffer.resize(total_byte_length);

        for (auto& writer : writers)
        {
            writer.add_metadata(buffer, model, primitive);
        }
        face_outs[i] = (uint32_t*)(buffer.data() + face_byte_offset);

        vertex_map.reserve(segment.num_vertices);
    }

    ProgressBar progress(mesh.n_faces(), "Writing Data");
    std::vector<size_t> added_faces(num_segments, 0);
    for (auto fH : mesh.faces())
    {
        SegmentId segment_id = f_segment[fH];

        auto& face_out = face_outs[segment_id];
        if (face_out == nullptr)
        {
            std::cerr << "Added too many faces for segment " << segment_id << std::endl;
            return;
        }
        auto& vertex_map = vertex_maps[segment_id];
        for (auto vH : mesh.vertices(fH))
        {
            auto it = vertex_map.find(vH);
            if (it == vertex_map.end())
            {
                size_t new_vH = vertex_map.size();
                vertex_map[vH] = VertexHandle(new_vH);
                *face_out++ = new_vH;
                for (auto& writer : property_writers[segment_id])
                {
                    writer.add_value(vH.idx(), new_vH);
                }
            }
            else
            {
                *face_out++ = it->second.idx();
            }
        }

        if (++added_faces[segment_id] == segments[segment_id].num_faces)
        {
            std::ofstream bin_file(bin_files[segment_id], std::ios::binary);
            auto& buffer = buffers[segment_id];
            bin_file.write((char*)buffer.data(), buffer.size());
            buffer = {};
            face_outs[segment_id] = nullptr;
        }
        ++progress;
    }
    std::cout << "\r";

    // consistency check
    for (size_t i = 0; i < num_segments; i++)
    {
        if (added_faces[i] != segments[i].num_faces)
        {
            std::cerr << "Segment " << i << " has " << added_faces[i] << " faces, but "
                      << segments[i].num_faces << " were expected." << std::endl;
        }
        // there might be fewer vertices because of segment mergin, but more would cause a buffer overflow
        if (vertex_maps[i].size() > segments[i].num_vertices)
        {
            std::cerr << "Segment " << i << " has " << vertex_maps[i].size() << " vertices, but only "
                      << segments[i].num_vertices << " were expected." << std::endl;
        }
    }


    for (size_t i = 0; i < num_segments; i++)
    {
        auto& file = b3dm_files[i];
        auto& model = models[i];

        CesiumGltfWriter::GltfWriter writer;
        auto gltf = writer.writeGlb(model, gsl::span<std::byte>());
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

        std::ofstream b3dm_file(b3dm_files[i], std::ios::binary);

        b3dm_file << magic;
        write_uint32(b3dm_file, version);
        write_uint32(b3dm_file, byte_length);
        write_uint32(b3dm_file, feature_table_json_length);
        write_uint32(b3dm_file, feature_table_byte_length);
        write_uint32(b3dm_file, batch_table_json_length);
        write_uint32(b3dm_file, batch_table_byte_length);

        b3dm_file << feature_table;

        b3dm_file.write((char*)gltf.gltfBytes.data(), gltf.gltfBytes.size());

        model = {};
    }
}

} // namespace lvr2
