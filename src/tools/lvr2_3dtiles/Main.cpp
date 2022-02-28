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
#include "lvr2/algorithm/pmp/SurfaceNormals.h"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/Progress.hpp"
#include "B3dmWriter.hpp"
#include "Segmenter.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include <Cesium3DTiles/Tileset.h>
#include <Cesium3DTiles/Tile.h>
#include <Cesium3DTilesWriter/TilesetWriter.h>

using namespace lvr2;
using namespace Cesium3DTiles;

using boost::filesystem::path;

constexpr double FACE_TO_ERROR_FACTOR = 1.0 / 2000.0;

void convert_bounding_box(const pmp::BoundingBox& in, Cesium3DTiles::BoundingVolume& out)
{
    auto center = in.center();
    auto half_vector = in.max() - center;
    out.box =
    {
        center.x(), center.y(), center.z(),
        half_vector.x(), 0, 0,
        0, half_vector.y(), 0,
        0, 0, half_vector.z()
    };
}

void partition_chunks(std::vector<Segment>& segments, Tile& root, const path& output_dir);

int main(int argc, char** argv)
{
    path input_file;
    path output_dir;
    bool calc_normals = false;
    float chunk_size = -1;
    path mesh_out_file;

    try
    {
        using namespace boost::program_options;

        bool help = false;

        options_description options("General Options");
        options.add_options()
        ("calcNormals,N", bool_switch(&calc_normals),
         "Calculate normals if there are none in the input")

        ("segment,s", value<float>(&chunk_size),
         "Segment the mesh into connected regions with the given chunk size")

        ("write,w", value<path>(&mesh_out_file),
         "Save the mesh to the given file")

        ("help,h", bool_switch(&help),
         "Print this message here")
        ;

        options_description hidden_options("hidden_options");
        hidden_options.add_options()
        ("input_file", value<path>(&input_file))
        ("output_dir", value<path>(&output_dir))
        ;

        positional_options_description pos;
        pos.add("input_file", 1);
        pos.add("output_dir", 1);

        options_description all_options("options");
        all_options.add(options).add(hidden_options);

        variables_map variables;
        store(command_line_parser(argc, argv).options(all_options).positional(pos).run(), variables);
        notify(variables);

        if (help)
        {
            std::cout << "The Scan Registration Tool" << std::endl;
            std::cout << "Usage: " << std::endl;
            std::cout << "\tlvr2_3dtiles [OPTIONS] <input_file> [<output_dir>]" << std::endl;
            std::cout << std::endl;
            options.print(std::cout);
            std::cout << std::endl;
            std::cout << "<input_file> is the file where the input mesh is stored" << std::endl;
            std::cout << "<output_dir> is the directory to create the output in. Should be empty" << std::endl;
            return EXIT_SUCCESS;
        }

        if (variables.count("input_file") != 1)
        {
            throw error("Missing <input_file> Parameter");
        }

        if (variables.count("output_dir") == 0)
        {
            output_dir = "chunk.3dtiles";
        }
    }
    catch (const boost::program_options::error& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "Use '--help' to see the list of possible options" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << timestamp << "Reading mesh " << input_file << std::endl;
    PMPMesh<BaseVector<float>> mesh;
    try
    {
        mesh.getSurfaceMesh().read(input_file.string());
    }
    catch (std::exception& e)
    {
        std::cerr << "SurfaceMeshIO failed: " << e.what() << std::endl;
        std::cout << "Trying ModelFactory next" << std::endl;

        ModelPtr model = ModelFactory::readModel(input_file.string());
        if (!model)
        {
            std::cerr << "Error reading model" << std::endl;
            return EXIT_FAILURE;
        }
        if (!model->m_mesh)
        {
            std::cerr << "Model has no mesh" << std::endl;
            return EXIT_FAILURE;
        }
        std::cout << timestamp << "Converting to PMPMesh" << std::endl;
        mesh = PMPMesh<BaseVector<float>>(model->m_mesh);
    }

    auto& surface_mesh = mesh.getSurfaceMesh();

    if (calc_normals)
    {
        std::cout << timestamp << "Calculating normals" << std::endl;
        pmp::SurfaceNormals::compute_vertex_normals(surface_mesh);
    }

    if (!mesh_out_file.empty())
    {
        std::cout << timestamp << "Writing mesh to " << mesh_out_file << std::endl;
        pmp::IOFlags flags;
        flags.use_binary = true;
        surface_mesh.write(mesh_out_file.string(), flags);
    }

    std::cout << timestamp << "Creating 3D Tiles" << std::endl;

    if (boost::filesystem::exists(output_dir))
    {
        boost::filesystem::remove_all(output_dir);
    }
    boost::filesystem::create_directories(output_dir);
    path tileset_file = output_dir / "tileset.json";

    Cesium3DTiles::Tileset tileset;
    tileset.asset.version = "1.0";
    auto& root = tileset.root;
    root.refine = Cesium3DTiles::Tile::Refine::REPLACE;
    root.transform =
    {
        // 4x4 matrix to place the object somewhere on the globe
        96.86356343768793, 24.848542777253734, 0, 0,
        -15.986465724980844, 62.317780594908875, 76.5566922962899, 0,
        19.02322243409411, -74.15554020821229, 64.3356267137516, 0,
        1215107.7612304366, -4736682.902037748, 4081926.095098698, 1
    };

    pmp::BoundingBox bb = surface_mesh.bounds();
    convert_bounding_box(bb, root.boundingVolume);

    std::vector<Segment> segments;
    if (chunk_size > 0)
    {
        std::vector<pmp::SurfaceMesh> large_segments;
        std::vector<pmp::BoundingBox> large_segment_bounds;

        std::cout << timestamp << "Segmenting mesh" << std::endl;
        segment_mesh(surface_mesh, bb, chunk_size, segments, large_segments, large_segment_bounds);

        path chunk_path = output_dir / "chunks";
        boost::filesystem::create_directories(chunk_path);

        std::cout << timestamp << "Writing chunks                " << std::endl;
        partition_chunks(segments, root, "chunks");
        write_b3dm_segments(output_dir, surface_mesh, segments);

        std::vector<std::string> filenames;
        size_t biggest = 0;

        for (size_t i = 0; i < large_segments.size(); i++)
        {
            auto& mesh = large_segments[i];
            std::string filename = "segments/s" + std::to_string(i) + "/mesh.b3dm";
            std::cout << timestamp << "Writing segment " << i << " to " << filename << std::endl;
            filenames.push_back(filename);

            boost::filesystem::create_directories((output_dir / filename).parent_path());

            Tile& tile = root.children.emplace_back();
            tile.geometricError = mesh.n_faces() * FACE_TO_ERROR_FACTOR;
            convert_bounding_box(large_segment_bounds[i], tile.boundingVolume);

            Cesium3DTiles::Content content;
            content.uri = filename;
            tile.content = content;

            if (mesh.n_faces() > large_segments[biggest].n_faces())
            {
                biggest = i;
            }
        }
        #pragma omp parallel for
        for (size_t i = 0; i < large_segments.size(); i++)
        {
            write_b3dm(output_dir, filenames[i], large_segments[i], large_segment_bounds[i], i == biggest);
        }
    }
    else
    {
        std::string mesh_file = "mesh.b3dm";
        std::cout << timestamp << "Writing " << mesh_file << std::endl;

        Cesium3DTiles::Content content;
        content.uri = mesh_file;
        root.content = content;
        root.geometricError = mesh.numFaces() * FACE_TO_ERROR_FACTOR;

        write_b3dm(output_dir, mesh_file, surface_mesh, bb);
    }

    tileset.geometricError = root.geometricError;

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
        return EXIT_FAILURE;
    }

    std::cout << timestamp << "Writing " << tileset_file << std::endl;

    std::ofstream out(tileset_file.string(), std::ios::binary);
    out.write((char*)result.tilesetBytes.data(), result.tilesetBytes.size());

    std::cout << timestamp << "Finished" << std::endl;

    return 0;
}

pmp::BoundingBox split_recursive(Segment** start,
                                 Segment** end,
                                 std::string& filename,
                                 Tile& tile,
                                 const std::string& output_dir)
{
    size_t n = end - start;
    pmp::BoundingBox bb;

    if (n <= 8)
    {
        tile.children.resize(n);
        for (size_t i = 0; i < n; i++)
        {
            auto& segment = start[i];
            auto& child_tile = tile.children[i];

            filename.push_back('0' + i);
            segment->filename = output_dir + filename + ".b3dm";
            filename.pop_back();

            convert_bounding_box(segment->bb, child_tile.boundingVolume);
            bb += segment->bb;

            child_tile.geometricError = segment->num_faces * FACE_TO_ERROR_FACTOR;
            tile.geometricError += child_tile.geometricError;

            Cesium3DTiles::Content content;
            content.uri = segment->filename;
            child_tile.content = content;
        }
    }
    else
    {
        auto split_fn = [](int axis)
        {
            return [axis](const Segment * a, const Segment * b)
            {
                return a->bb.center()[axis] < b->bb.center()[axis];
            };
        };

        Segment** starts[9];
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
            filename.push_back('0' + i);
            bb += split_recursive(starts[i], starts[i + 1], filename, tile.children[i], output_dir);
            filename.pop_back();

            tile.geometricError += tile.children[i].geometricError;
        }
    }
    convert_bounding_box(bb, tile.boundingVolume);
    return bb;
}

void partition_chunks(std::vector<Segment>& segments, Tile& root, const path& output_dir)
{
    root.refine = Cesium3DTiles::Tile::Refine::ADD;

    std::vector<Segment*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = &segments[i];
    }

    std::string output = output_dir.string();
    if (output.back() != '/')
    {
        output.push_back('/');
    }

    std::string filename = "t";
    split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(),
                    filename, root, output);
}
