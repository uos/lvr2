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

int main(int argc, char** argv)
{
    fs::path input_file;
    fs::path output_dir;
    bool calc_normals;
    float chunk_size = -1;
    fs::path mesh_out_file;
    fs::path segment_out_path;
    bool fix_mesh;
    bool load_segments;

    try
    {
        using namespace boost::program_options;

        bool help = false;

        options_description options("General Options");
        options.add_options()
        ("calcNormals,N", bool_switch(&calc_normals),
         "Calculate normals if there are none in the input")

        ("fix,f", bool_switch(&fix_mesh),
         "Assume the mesh might have errors, attempt to fix it")

        ("segment,s", value<float>(&chunk_size),
         "Segment the mesh into connected regions with the given chunk size")

        ("wm", value<fs::path>(&mesh_out_file),
         "Save the mesh after fix and calcNormals to the given file")

        ("ws", value<fs::path>(&segment_out_path),
         "Save the segments to the given directory")

        ("ls", bool_switch(&load_segments),
         "Interpret <input_file> as a directory and load segments written by the --ws option")

        ("help,h", bool_switch(&help),
         "Print this message here")
        ;

        options_description hidden_options("hidden_options");
        hidden_options.add_options()
        ("input_file", value<fs::path>(&input_file))
        ("output_dir", value<fs::path>(&output_dir))
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

        if (fs::is_directory(input_file) && !load_segments)
        {
            if (fs::is_directory(input_file / "chunks") || fs::is_directory(input_file / "segments"))
            {
                load_segments = true;
            }
            else
            {
                throw error("<input_file> is a directory but doesn't match --ws output");
            }
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
    std::vector<MeshSegment> chunks;
    std::vector<MeshSegment> segments;
    pmp::BoundingBox bb;

    if (load_segments)
    {
        for (auto file : boost::make_iterator_range(fs::directory_iterator(input_file / "chunks"), {}))
        {
            auto& out = chunks.emplace_back();
            out.mesh.reset(new pmp::SurfaceMesh());
            out.mesh->read(file.path().string());
            out.bb = out.mesh->bounds();
            bb += out.bb;
        }
        for (auto file : boost::make_iterator_range(fs::directory_iterator(input_file / "segments"), {}))
        {
            auto& out = segments.emplace_back();
            out.mesh.reset(new pmp::SurfaceMesh());
            out.mesh->read(file.path().string());
            out.bb = out.mesh->bounds();
            bb += out.bb;
        }
    }
    else
    {
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
                throw std::runtime_error("Error reading model");
            }
            if (!model->m_mesh)
            {
                throw std::runtime_error("Model has no mesh");
            }
            std::cout << timestamp << "Converting to PMPMesh" << std::endl;
            mesh = PMPMesh<BaseVector<float>>(model->m_mesh);
        }

        auto& surface_mesh = mesh.getSurfaceMesh();

        if (fix_mesh)
        {
            std::cout << timestamp << "Fixing mesh" << std::endl;
            surface_mesh.duplicate_non_manifold_vertices();
            surface_mesh.remove_degenerate_faces();
        }

        if (calc_normals && !surface_mesh.has_vertex_property("v:normal"))
        {
            std::cout << timestamp << "Calculating normals" << std::endl;
            pmp::SurfaceNormals::compute_vertex_normals(surface_mesh);
        }

        if (!mesh_out_file.empty())
        {
            std::cout << timestamp << "Writing mesh to " << mesh_out_file << std::endl;
            surface_mesh.garbage_collection();
            pmp::IOFlags flags;
            flags.use_binary = true;
            surface_mesh.write(mesh_out_file.string(), flags);
        }

        bb = surface_mesh.bounds();

        if (chunk_size > 0)
        {
            std::cout << timestamp << "Segmenting mesh" << std::endl;
            segment_mesh(surface_mesh, bb, chunk_size, chunks, segments);
        }
        else
        {
            auto& out = segments.emplace_back();
            out.mesh = std::make_shared<pmp::SurfaceMesh>(std::move(surface_mesh));
            out.bb = bb;
            out.filename = "mesh.b3dm";
        }
    }

    if (!segment_out_path.empty())
    {
        std::cout << timestamp << "Writing segments to " << segment_out_path << std::endl;
        fs::path chunk_dir = segment_out_path / "chunks";
        fs::path segment_dir = segment_out_path / "segments";
        fs::remove_all(chunk_dir);
        fs::remove_all(segment_dir);
        fs::create_directories(chunk_dir);
        fs::create_directories(segment_dir);
        for (auto& chunk : chunks)
        {
            chunk.mesh->garbage_collection();
        }
        for (auto& segment : segments)
        {
            segment.mesh->garbage_collection();
        }
        #pragma omp parallel for
        for (size_t i = 0; i < chunks.size() + segments.size(); ++i)
        {
            if (i < chunks.size())
            {
                chunks[i].mesh->write((chunk_dir / (std::to_string(i) + ".pmp")).string());
            }
            else
            {
                size_t idx = i - chunks.size();
                segments[idx].mesh->write((segment_dir / (std::to_string(idx) + ".pmp")).string());
            }
        }
    }

    std::cout << timestamp << "Creating 3D Tiles" << std::endl;

    fs::remove_all(output_dir);
    fs::create_directories(output_dir);
    fs::path tileset_file = output_dir / "tileset.json";

    Cesium3DTiles::Tileset tileset;
    tileset.asset.version = "1.0";
    auto& root = tileset.root;
    root.refine = Cesium3DTiles::Tile::Refine::ADD;
    root.transform =
    {
        // 4x4 matrix to place the object somewhere on the globe
        96.86356343768793, 24.848542777253734, 0, 0,
        -15.986465724980844, 62.317780594908875, 76.5566922962899, 0,
        19.02322243409411, -74.15554020821229, 64.3356267137516, 0,
        1215107.7612304366, -4736682.902037748, 4081926.095098698, 1
    };

    convert_bounding_box(bb, root.boundingVolume);
    SegmentTreeNode root_segment;
    root_segment.skipped = true;

    if (!chunks.empty())
    {
        std::string path = "chunks";
        fs::create_directories(output_dir / path);

        std::cout << timestamp << "Partitioning chunks                " << std::endl;
        auto& tile = root.children.emplace_back();
        auto chunk_root = SegmentTree::octree_partition(chunks, tile, path, 2);
        chunk_root->skipped = true;
        root_segment.add_child(std::move(chunk_root));
    }

    if (chunk_size > 0)
    {
        std::cout << timestamp << "Splitting " << segments.size() << " large segments" << std::endl;
        size_t total_faces = 0;
        for (auto& segment : segments)
        {
            total_faces += segment.mesh->n_faces();
        }
        ProgressBar progress(total_faces, "Splitting segments");

        for (size_t i = 0; i < segments.size(); i++)
        {
            auto& segment = segments[i];
            std::string path = "segments/s" + std::to_string(i) + "/";

            fs::create_directories(output_dir / path);

            Tile& tile = root.children.emplace_back();
            convert_bounding_box(segment.bb, tile.boundingVolume);

            std::vector<MeshSegment> children;
            split_mesh(segment, chunk_size, children);
            root_segment.add_child(SegmentTree::octree_partition(children, tile, path));

            progress += segment.mesh->n_faces();
        }
        std::cout << "\r";
    }
    else
    {
        auto& segment = segments[0];
        Cesium3DTiles::Content content;
        content.uri = segment.filename;
        auto& tile = root.children.emplace_back();
        tile.content = content;
        root_segment.add_child(std::make_unique<SegmentTreeLeaf>(segment));
    }

    std::cout << timestamp << "Constructed tree with depth " << root_segment.depth << ". Creating meta-segments" << std::endl;
    root_segment.simplify(root);

    std::vector<MeshSegment> all_segments;
    root_segment.collect_segments(all_segments);

    std::cout << timestamp << "Writing " << all_segments.size() << " segments" << std::endl;
    #pragma omp parallel for
    for (size_t i = 0; i < all_segments.size(); i++)
    {
        write_b3dm(output_dir, all_segments[i].filename, *all_segments[i].mesh, all_segments[i].bb, false);
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
