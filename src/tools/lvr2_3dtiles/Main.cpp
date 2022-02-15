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

void write_segments(std::vector<Segment>& segments, Tile& root, const path& output_dir);

int main(int argc, char** argv)
{
    path input_file;
    path output_dir;
    bool calc_normals = false;
    bool segment = false;
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

        ("segment,s", bool_switch(&segment),
         "Segment the mesh into connected regions")

        ("chunkSize", value<float>(&chunk_size)->default_value(chunk_size),
         "Size of the chunks, only needed if segment is set")

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

        if (segment && chunk_size < 0)
        {
            throw error("Chunk size needs to be set if segment is set");
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

    if (!boost::filesystem::exists(output_dir))
    {
        boost::filesystem::create_directories(output_dir);
    }
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
    if (segment)
    {
        std::cout << timestamp << "Segmenting mesh" << std::endl;
        segment_mesh(surface_mesh, segments, bb, chunk_size);

        path segment_path = output_dir / "segments";
        if (boost::filesystem::exists(segment_path))
        {
            boost::filesystem::remove_all(segment_path);
        }
        boost::filesystem::create_directories(segment_path);

        write_segments(segments, root, "segments");
    }
    else
    {
        path mesh_file = "mesh.b3dm";
        std::cout << timestamp << "Writing " << mesh_file << std::endl;

        Cesium3DTiles::Content content;
        content.uri = mesh_file.string();
        root.content = content;
        root.geometricError = mesh.numFaces() * FACE_TO_ERROR_FACTOR;

        // add a fake segment containing the whole mesh
        surface_mesh.add_vertex_property<SegmentId>("v:segment", 0);
        surface_mesh.add_face_property<SegmentId>("f:segment", 0);

        segments.emplace_back();
        auto& segment = segments.back();
        segment.id = 0;
        segment.num_faces = surface_mesh.n_faces();
        segment.num_vertices = surface_mesh.n_vertices();
        segment.bb = bb;
        segment.filename = "mesh.b3dm";
    }
    write_b3dm(output_dir, surface_mesh, segments);

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

void split_recursive(Segment** start,
                     Segment** end,
                     const std::string& filename,
                     Tile& tile,
                     const path& output_dir)
{
    size_t n = end - start;
    if (n == 0)
    {
        return;
    }

    if (n == 1)
    {
        auto& segment = **start;
        segment.filename = (output_dir / (filename + ".b3dm")).string();

        convert_bounding_box(segment.bb, tile.boundingVolume);

        tile.geometricError = segment.num_faces * FACE_TO_ERROR_FACTOR;

        Cesium3DTiles::Content content;
        content.uri = segment.filename;
        tile.content = content;
    }
    else
    {
        pmp::BoundingBox bb;
        for (auto it = start; it != end; ++it)
        {
            bb += (**it).bb;
        }
        convert_bounding_box(bb, tile.boundingVolume);

        pmp::Point size = bb.max() - bb.min();
        size_t longest_axis = std::max_element(size.data(), size.data() + 3) - size.data();

        Segment** mid = start + n / 2;
        std::nth_element(start, mid, end, [longest_axis](const Segment* a, const Segment* b)
        {
            return a->bb.center()[longest_axis] < b->bb.center()[longest_axis];
        });

        constexpr char NAME[3][2] = { { 'x', 'X' }, { 'y', 'Y' }, { 'z', 'Z' } };

        tile.children.resize(2);

        split_recursive(start, mid, filename + NAME[longest_axis][0], tile.children[0], output_dir);
        split_recursive(mid, end, filename + NAME[longest_axis][1], tile.children[1], output_dir);

        tile.geometricError = tile.children[0].geometricError + tile.children[1].geometricError;
    }
}

void write_segments(std::vector<Segment>& segments, Tile& root, const path& output_dir)
{
    root.refine = Cesium3DTiles::Tile::Refine::ADD;

    std::vector<Segment*> temp_segments(segments.size());
    for (size_t i = 0; i < segments.size(); ++i)
    {
        temp_segments[i] = &segments[i];
    }

    split_recursive(temp_segments.data(), temp_segments.data() + temp_segments.size(),
                    "t", root, output_dir);
}
