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


#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/PMPMesh.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"

#include <boost/filesystem.hpp>

#include <Cesium3DTiles/Tileset.h>
#include <Cesium3DTiles/Tile.h>
#include <Cesium3DTilesWriter/TilesetWriter.h>
#include <CesiumGltf/Model.h>
#include <CesiumGltfWriter/GltfWriter.h>

using namespace lvr2;
using namespace Cesium3DTiles;

using boost::filesystem::path;

using Vec = BaseVector<float>;

/**
 * @brief converts mesh to b3dm format and writes it to a file
 *
 * @param filename the name of the file to write to
 * @param mesh the mesh to convert
 */
void write_b3dm(const path& filename, const PMPMesh<Vec>& mesh);

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

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> [<output_directory>]" << std::endl;
        return 1;
    }

    std::cout << timestamp << "Reading model " << argv[1] << std::endl;
    ModelPtr model = ModelFactory::readModel(argv[1]);
    if (!model)
    {
        std::cerr << "Error reading model" << std::endl;
        return 1;
    }
    if (!model->m_mesh)
    {
        std::cerr << "Model has no mesh" << std::endl;
        return 1;
    }
    PMPMesh<Vec> mesh(model->m_mesh);

    path outpath(argc >= 3 ? argv[2] : "chunk.3dtiles");
    if (!boost::filesystem::exists(outpath))
    {
        boost::filesystem::create_directories(outpath);
    }
    path tileset_file = outpath / "tileset.json";
    path mesh_file = "mesh.b3dm";

    Cesium3DTiles::Tileset tileset;
    tileset.geometricError = 500.0;
    tileset.asset.version = "1.0";
    auto& root = tileset.root;
    root.refine = Cesium3DTiles::Tile::Refine::REPLACE;
    root.geometricError = 500.0;
    root.transform =
    {
        // 4x4 matrix to place the object somewhere on the globe
        96.86356343768793, 24.848542777253734, 0, 0,
        -15.986465724980844, 62.317780594908875, 76.5566922962899, 0,
        19.02322243409411, -74.15554020821229, 64.3356267137516, 0,
        1215107.7612304366, -4736682.902037748, 4081926.095098698, 1
    };

    {
        auto bb = mesh.getSurfaceMesh().bounds();
        auto center = bb.center();
        auto half_vector = bb.max() - center;
        root.boundingVolume.box =
        {
            center.x(), center.y(), center.z(),
            half_vector.x(), 0, 0,
            0, half_vector.y(), 0,
            0, 0, half_vector.z()
        };
    }

    write_b3dm(outpath / mesh_file, mesh);

    Cesium3DTiles::Content content;
    content.uri = mesh_file.string();
    root.content = content;

    Cesium3DTilesWriter::TilesetWriter writer;
    auto result = writer.writeTileset(tileset);

    if (!result.warnings.empty())
    {
        std::cerr << "Warnings writing tileset: " << std::endl;
        for (auto& e : result.warnings)
        {
            std::cerr << e << std::endl;
        }
    }
    if (!result.errors.empty())
    {
        std::cerr << "Errors writing tileset: " << std::endl;
        for (auto& e : result.errors)
        {
            std::cerr << e << std::endl;
        }
        return 1;
    }

    std::cout << timestamp << "Writing " << tileset_file << std::endl;

    std::ofstream out(tileset_file.string(), std::ios::binary);
    out.write((char*)result.tilesetBytes.data(), result.tilesetBytes.size());

    std::cout << timestamp << "Finished" << std::endl;

    return 0;
}

void write_b3dm(const path& filename, const PMPMesh<Vec>& mesh)
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

    auto [ buffer_id, raw_buffer ] = push_and_get_index(model.buffers);
    auto& buffer = raw_buffer.cesium.data;

    auto& surface_mesh = mesh.getSurfaceMesh();
    size_t num_vertices = surface_mesh.n_vertices();

    // vertices might have been deleted, so map from VertexHandle to buffer index
    DenseVertexMap<size_t> vertex_id_map(surface_mesh.vertices_size());

    auto [ vertex_accessor_id, vertex_accessor ] = push_and_get_index(model.accessors);
    {
        size_t byte_offset = buffer.size();
        size_t byte_length = num_vertices * 3 * sizeof(float);
        buffer.resize(byte_offset + byte_length);
        float* out = (float*)(buffer.data() + byte_offset);

        Eigen::Vector3f min = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
        Eigen::Vector3f max = Eigen::Vector3f::Constant(-std::numeric_limits<float>::infinity());
        auto it = surface_mesh.vertices_begin();
        for (size_t i = 0; i < num_vertices; ++i, ++it, out += 3)
        {
            vertex_id_map[*it] = i;
            const auto& vertex = surface_mesh.position(*it);
            out[0] = vertex.x();
            out[1] = vertex.y();
            out[2] = vertex.z();
            min = min.cwiseMin(vertex);
            max = max.cwiseMax(vertex);
        }

        auto [ buffer_view_id, buffer_view ] = push_and_get_index(model.bufferViews);

        buffer_view.buffer = buffer_id;
        buffer_view.byteOffset = byte_offset;
        buffer_view.byteLength = byte_length;
        buffer_view.byteStride = 3 * sizeof(float);
        buffer_view.target = CesiumGltf::BufferView::Target::ARRAY_BUFFER;

        vertex_accessor.bufferView = buffer_view_id;
        vertex_accessor.count = num_vertices;
        vertex_accessor.componentType = CesiumGltf::Accessor::ComponentType::FLOAT;
        vertex_accessor.type = CesiumGltf::Accessor::Type::VEC3;
        vertex_accessor.min = { min.x(), min.y(), min.z() };
        vertex_accessor.max = { max.x(), max.y(), max.z() };
    }

    auto [ index_accessor_id, index_accessor ] = push_and_get_index(model.accessors);
    {
        size_t num_triangles = mesh.numFaces();
        size_t byte_offset = buffer.size();
        size_t byte_length = num_triangles * 3 * sizeof(uint32_t);
        buffer.resize(byte_offset + byte_length);
        uint32_t* out = (uint32_t*)(buffer.data() + byte_offset);
        for (auto it = mesh.facesBegin(), end = mesh.facesEnd(); it != end; ++it, out += 3)
        {
            auto vertex_iter = mesh.getSurfaceMesh().vertices(*it);
            out[0] = vertex_id_map[*vertex_iter];
            out[1] = vertex_id_map[*(++vertex_iter)];
            out[2] = vertex_id_map[*(++vertex_iter)];
        }

        auto [ buffer_view_id, buffer_view ] = push_and_get_index(model.bufferViews);
        buffer_view.buffer = buffer_id;
        buffer_view.byteOffset = byte_offset;
        buffer_view.byteLength = byte_length;
        buffer_view.target = CesiumGltf::BufferView::Target::ELEMENT_ARRAY_BUFFER;

        index_accessor.bufferView = buffer_view_id;
        index_accessor.count = num_triangles * 3;
        index_accessor.componentType = CesiumGltf::Accessor::ComponentType::UNSIGNED_INT;
        index_accessor.type = CesiumGltf::Accessor::Type::SCALAR;
        index_accessor.min = { 0 };
        index_accessor.max = { (double)num_vertices - 1 };
    }

    auto [ out_mesh_id, out_mesh ] = push_and_get_index(model.meshes);

    auto [ primitive_id, primitive ] = push_and_get_index(out_mesh.primitives);
    primitive.mode = CesiumGltf::MeshPrimitive::Mode::TRIANGLES;
    primitive.indices = index_accessor_id;
    primitive.attributes["POSITION"] = vertex_accessor_id;

    auto [ node_id, node ] = push_and_get_index(model.nodes);
    node.mesh = out_mesh_id;
    // gltf uses y-up, but 3d tiles uses z-up and automatically transforms gltf data.
    // So we need to pre-undo that transformation to maintain consistency.
    // See the "Implementation note" section in https://github.com/CesiumGS/3d-tiles/tree/main/specification#gltf-transforms
    node.matrix = {1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1};

    auto [ scene_id, scene ] = push_and_get_index(model.scenes);
    scene.nodes.push_back(node_id);

    model.scene = scene_id;

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
