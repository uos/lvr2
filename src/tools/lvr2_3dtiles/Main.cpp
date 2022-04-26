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
#include "lvr2/geometry/pmp/SurfaceMeshIO.h"
#include "lvr2/algorithm/pmp/SurfaceNormals.h"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/meshio/HDF5IO.hpp"
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

#include "viewer.html"

int main(int argc, char** argv)
{
    fs::path input_file;
    fs::path output_dir;
    bool calc_normals;
    float chunk_size = -1;
    std::vector<fs::path> mesh_out_files;
    fs::path segment_out_path;
    bool fix_mesh;
    bool load_segments;
    bool assign_colors;

    try
    {
        using namespace boost::program_options;

        bool help = false;

        options_description options("General Options");
        options.add_options()
        ("calcNormals,N", bool_switch(&calc_normals),
         "Calculate normals")

        ("fix,f", bool_switch(&fix_mesh),
         "Assume the mesh might have errors, attempt to fix it")

        ("segment,s", value<float>(&chunk_size),
         "Segment the mesh into connected regions with the given chunk size")

        ("wm", value<std::vector<fs::path>>(&mesh_out_files),
         "Save the mesh after fix and calcNormals to the given file")

        ("ws", value<fs::path>(&segment_out_path),
         "Save the segments to the given directory")

        ("ls", bool_switch(&load_segments),
         "Interpret <input_file> as a directory and load segments written by the --ws option")

        ("assignColors,C", bool_switch(&assign_colors),
         "Assign colors to vertices")

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

        if (!fs::exists(input_file))
        {
            throw error("<input_file> does not exist");
        }
        else if (fs::is_regular_file(input_file))
        {
            if (load_segments)
            {
                throw error("Option --ls is set but <input_file> is not a directory");
            }
        }
        else if (fs::is_directory(input_file))
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
        else
        {
            throw error("<input_file> is neither a file nor a directory");
        }

        if (load_segments && chunk_size <= 0)
        {
            throw error("Loading segments requires a chunk size");
        }
    }
    catch (const boost::program_options::error& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "Use '--help' to see the list of possible options" << std::endl;
        return EXIT_FAILURE;
    }


    PMPMesh<BaseVector<float>> mesh;
    std::vector<std::pair<pmp::Point, MeshSegment>> chunks;
    std::vector<MeshSegment> segments;
    pmp::BoundingBox bb;
    Eigen::Vector3i num_chunks = Eigen::Vector3i::Zero();

    if (load_segments)
    {
        std::cout << timestamp << "Reading segments from " << input_file << std::endl;

        for (auto file : boost::make_iterator_range(fs::directory_iterator(input_file / "chunks"), {}))
        {
            auto& [ chunk_pos, segment ] = chunks.emplace_back();

            int x, y, z;
            std::sscanf(file.path().filename().c_str(), "%d_%d_%d.pmp", &x, &y, &z);
            chunk_pos = pmp::Point(x, y, z);
            num_chunks = num_chunks.cwiseMax(Eigen::Vector3i(x, y, z));

            segment.mesh.reset(new pmp::SurfaceMesh());
            segment.mesh->read(file.path().string());
            segment.bb = segment.mesh->bounds();
            bb += segment.bb;
        }
        num_chunks += Eigen::Vector3i::Ones(); // count is max element + 1

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
        auto extension = input_file.extension().string().substr(1);

        std::cout << timestamp << "Reading mesh " << input_file;
        if (pmp::SurfaceMeshIO::supported_extensions().count(extension))
        {
            std::cout << " using pmp::SurfaceMeshIO" << std::endl;
            pmp::SurfaceMeshIO io(input_file.string(), pmp::IOFlags());
            io.read(mesh.getSurfaceMesh());
        }
        else if (extension == "h5")
        {
            std::cout << " using meshio::HDF5IO" << std::endl;
            auto kernel = std::make_shared<HDF5Kernel>(input_file.string());
            auto schema = std::make_shared<MeshSchemaHDF5>();
            meshio::HDF5IO io(kernel, schema);
            MeshBufferPtr buffer;
            try
            {
                buffer = io.loadMesh("Mesh0"); // TODO: replace with mesh finder
            }
            catch(const std::exception& e)
            {
                buffer = io.loadMesh("default");
            }
            std::cout << timestamp << "Converting to PMPMesh" << std::endl;
            mesh = PMPMesh<BaseVector<float>>(buffer);
        }
        else
        {
            std::cout << " using ModelFactory" << std::endl;
            ModelPtr model = ModelFactory::readModel(input_file.string());
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

        // if (calc_normals)
        // {
        //     std::cout << timestamp << "Calculating normals" << std::endl;
        //     pmp::SurfaceNormals::compute_vertex_normals(surface_mesh, true);
        //     auto v_normal = surface_mesh.vertex_property<pmp::Normal>("v:normal");
        //     #pragma omp parallel for schedule(static)
        //     for (size_t i = 0; i < surface_mesh.vertices_size(); i++)
        //     {
        //         pmp::Vertex v(i);
        //         if (!surface_mesh.is_deleted(v))
        //         {
        //             auto& n = v_normal[v];
        //             if (n.dot(pmp::Point::UnitZ()) < 0)
        //             {
        //                 n = -n;
        //             }
        //         }
        //     }
        // }

        for (auto file : mesh_out_files)
        {
            std::cout << timestamp << "Writing mesh to " << file << std::endl;
            surface_mesh.garbage_collection();
            pmp::IOFlags flags;
            flags.use_binary = true;
            surface_mesh.write(file.string(), flags);
        }

        bb = surface_mesh.bounds();

        if (chunk_size > 0)
        {
            std::cout << timestamp << "Segmenting mesh" << std::endl;
            segment_mesh(surface_mesh, bb, chunk_size * 2, chunks, segments);
            // chunk_size * 2 to merge the small segments into larger chunks, otherwise we get a lot of small chunks
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
            chunk.second.mesh->garbage_collection();
        }
        for (auto& segment : segments)
        {
            segment.mesh->garbage_collection();
        }
        #pragma omp parallel for
        for (size_t i = 0; i < chunks.size(); ++i)
        {
            auto& [ chunk_pos, chunk ] = chunks[i];
            std::stringstream ss;
            ss << chunk_dir.string() << "/" << (int)chunk_pos.x() << "_" << (int)chunk_pos.y() << "_" << (int)chunk_pos.z() << ".pmp";
            chunk.mesh->write(ss.str());
        }
        #pragma omp parallel for
        for (size_t i = 0; i < segments.size(); ++i)
        {
            segments[i].mesh->write((segment_dir / (std::to_string(i) + ".pmp")).string());
        }
    }

    if (assign_colors)
    {
        std::cout << timestamp << "Assigning colors" << std::endl;
        float step_size = chunk_size > 0 ? chunk_size : 10;
        float variation = 0.1;
        for (size_t i = 0; i < segments.size() + chunks.size(); i++)
        {
            auto& segment = i < segments.size() ? segments[i] : chunks[i - segments.size()].second;
            auto& mesh = *segment.mesh;
            auto v_color = mesh.vertex_property<pmp::Color>("v:color");
            float r = std::abs(std::sin(i * 2));
            float g = std::abs(std::cos(i));
            float b = std::abs(std::sin(i * 30));
            float min_z = segment.bb.min().z();
            float max_z = segment.bb.max().z();
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < mesh.vertices_size(); i++)
            {
                pmp::Vertex v(i);
                if (!mesh.is_deleted(v))
                {
                    auto pos = mesh.position(v);
                    float dr = std::sin(pos.x() / step_size) * variation;
                    float dg = std::sin(pos.y() / step_size) * variation;
                    float db = ((pos.z() - min_z) / (max_z - min_z) - 0.5f) * variation;
                    v_color[v] = pmp::Color(
                        std::clamp(r + dr, 0.0f, 1.0f),
                        std::clamp(g + dg, 0.0f, 1.0f),
                        std::clamp(b + db, 0.0f, 1.0f)
                    );
                }
            }
        }
    }

    std::cout << timestamp << "Creating 3D Tiles" << std::endl;

    fs::remove_all(output_dir);
    fs::create_directories(output_dir);

    Cesium3DTiles::Tileset tileset;
    tileset.asset.version = "1.0";
    tileset.geometricError = 1e6; // tileset should always be rendered -> set error very high
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
    root_segment.m_skipped = true;

    if (!chunks.empty())
    {
        std::string path = "chunks/";
        fs::create_directories(output_dir / path);

        std::cout << timestamp << "Partitioning chunks                " << std::endl;
        auto& tile = root.children.emplace_back();
        auto chunk_root = SegmentTree::octree_partition(chunks, num_chunks, 2);
        chunk_root->m_skipped = true;
        chunk_root->fill_tile(tile, path + "c");
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

            auto tree = split_mesh(segment, chunk_size);
            tree->fill_tile(tile, path + "s");
            root_segment.add_child(std::move(tree));

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

    std::cout << timestamp << "Constructed tree with depth " << root_segment.m_depth << ". Creating meta-segments" << std::endl;
    root_segment.simplify();

    root_segment.fill_tile(root, "");

    std::vector<MeshSegment> all_segments;
    root_segment.collect_segments(all_segments);

    if (calc_normals)
    {
        std::cout << timestamp << "Calculating normals" << std::endl;
        ProgressBar progress_normals(all_segments.size(), "Calculating normals");
        for (size_t i = 0; i < all_segments.size(); i++)
        {
            auto& mesh = *all_segments[i].mesh;
            auto v_normal = mesh.vertex_property<pmp::Normal>("v:normal");
            // auto v_feature = mesh.get_vertex_property<bool>("v:feature");
            // auto v_color = mesh.get_vertex_property<pmp::Color>("v:color");
            #pragma omp parallel for schedule(dynamic,64)
            for (size_t i = 0; i < mesh.vertices_size(); i++)
            {
                pmp::Vertex v(i);
                if (mesh.is_deleted(v))
                {
                    continue;
                }
                // if (v_feature && v_feature[v])
                // {
                //     v_color[v] = pmp::Color(1, 0, 0);
                // }
                // else
                // {
                    v_normal[v] = pmp::SurfaceNormals::compute_vertex_normal(mesh, v);
                // }
            }
            ++progress_normals;
        }
    }
    std::cout << "\r";

    std::cout << timestamp << "Writing " << all_segments.size() << " segments        " << std::endl;
    ProgressBar progress_write(all_segments.size(), "Writing segments");
    #pragma omp parallel for
    for (size_t i = 0; i < all_segments.size(); i++)
    {
        write_b3dm(output_dir, all_segments[i].filename, *all_segments[i].mesh, all_segments[i].bb, "", false);
        ++progress_write;
    }
    std::cout << "\r";

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

    fs::path tileset_file = output_dir / "tileset.json";
    std::cout << timestamp << "Writing " << tileset_file << std::endl;

    std::ofstream tileset_out(tileset_file.string(), std::ios::binary);
    tileset_out.write((char*)result.tilesetBytes.data(), result.tilesetBytes.size());
    tileset_out.close();

    fs::path viewer_file = output_dir / "index.html";
    std::cout << timestamp << "Writing " << viewer_file << std::endl;

    std::ofstream viewer_out(viewer_file.string());
    viewer_out << VIEWER_HTML;
    viewer_out.close();

    std::cout << timestamp << "Finished" << std::endl;

    return 0;
}
