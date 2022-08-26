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

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/LazyMesh.hpp"
#include "lvr2/geometry/pmp/SurfaceMeshIO.h"
#include "lvr2/algorithm/pmp/SurfaceNormals.h"
#include "lvr2/algorithm/HLODTree.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Tiles3dIO.hpp"
#include "lvr2/io/meshio/HDF5IO.hpp"
#include "lvr2/util/Timestamp.hpp"
#include "lvr2/util/Progress.hpp"
#include "lvr2/config/lvropenmp.hpp"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

using namespace lvr2;
namespace fs = boost::filesystem;

using Vec = BaseVector<float>;
using Mesh = PMPMesh<Vec>;
using Tree = HLODTree<Vec>;
using IO = Tiles3dIO<Vec>;

void read_chunks(std::unordered_map<Vector3i, Tree::Ptr>& chunks,
                 const std::vector<std::string>& chunk_files,
                 float chunk_size, float voxel_size,
                 std::shared_ptr<HighFive::File> mesh_file,
                 std::function<void(pmp::SurfaceMesh&, const std::string&)> read_mesh);

const pmp::Point flip_point(100000, 100000, 100000);

void print_chunk_size_error()
{
    std::cout << "Error: If your input does not contain Chunks, you need to specify a chunk size." << std::endl;
    std::cout << "       Even if you don't want any splitting, you still need to explicitly set it to -1." << std::endl;
}

int main(int argc, char** argv)
{
    fs::path input_file;
    std::string input_file_extension;
    fs::path output_dir = "mesh.3dtiles";
    float chunk_size = 0;
    bool has_chunk_size = false;
    int combine_depth = 3;
    float reduction_factor = 0.2f;
    float normal_deviation = -1;
    float scale = 1.0f;
    std::vector<fs::path> mesh_out_files;
    AllowedMemoryUsage allowedMemUsage = AllowedMemoryUsage::Moderate;
    bool fix_mesh;
    bool compress;

    try
    {
        using namespace boost::program_options;

        bool help = false;

        options_description options("General Options");
        options.add_options()
        ("inputFile", value<fs::path>(&input_file),
         "The input. See below for accepted formats.")

        ("outputDir", value<fs::path>(&output_dir)->default_value(output_dir),
         "A Directory for the output.")

        ("memUsage", value<AllowedMemoryUsage>(&allowedMemUsage)->default_value(allowedMemUsage),
         "How strictly should the algorithm try to save memory.\n"
         "Available Options: 'minimal', 'moderate', 'unbounded' or a number in [0, 2].\n"
         "Less Memory used always means more time required to generate tiles.")

        ("chunkSize,c", value<float>(&chunk_size),
         "When loading a single Mesh: Split the Mesh into parts with this size.\n"
         "(Chunked inputs use their own chunk size)\n"
         "NOTE: This option is required on single Mesh inputs, so that users don't forget to set it.\n"
         "      If you don't want any splitting, explicitly set it to -1.")

        ("combineDepth,d", value<int>(&combine_depth)->default_value(combine_depth),
         "How many layers to combine and simplify. -1 means all.")

        ("reductionFactor,r", value<float>(&reduction_factor)->default_value(reduction_factor),
         "Factor between 0 and 1 indicating how far the meshes should be simplified.\\"
         "0 means as much as possible, 1 means no simplification.")

        ("normalDeviation,n", value<float>(&normal_deviation),
         "Set a maximum angle in degrees that normals are allowed to change by.")

        ("fix,f", bool_switch(&fix_mesh),
         "Fixes some common errors in meshes that might break this algorithm.\n"
         "Only works when loading a single mesh.")

        ("writeMesh,w", value<std::vector<fs::path>>(&mesh_out_files)->multitoken(),
         "Save the mesh after --fix has been applied.")

        ("scale,s", value<float>(&scale)->default_value(scale),
         "Scale the mesh.\n"
         "This defaults to 100 because small meshes are hard to navigate in the default html viewer.\n"
         "Only change this if you use a different viewer and/or you have an accurate position on the globe.")

        ("compress,z", bool_switch(&compress),
         "Compress the output meshes using Draco Compression.\n"
         "This will significantly reduce filesize and improve loading times when remotely viewing the tiles "
         "over a slow connection, but greatly increase loading times for local viewing.")

        ("help,h", bool_switch(&help),
         "Print this message here.")
        ;

        positional_options_description pos;
        pos.add("inputFile", 1);
        pos.add("outputDir", 1);

        variables_map variables;
        store(command_line_parser(argc, argv).options(options).positional(pos).run(), variables);

        if (help)
        {
            std::cout << "The Mesh to 3D Tiles conversion tool" << std::endl;
            std::cout << "Usage: " << std::endl;
            std::cout << "\tlvr2_3dtiles [OPTIONS] <inputFile> [<outputDir>]" << std::endl;
            std::cout << std::endl;
            options.print(std::cout);
            std::cout << std::endl;
            std::cout << "<inputFile> is the file where the input mesh is stored" << std::endl;
            std::cout << "    Possible inputs:" << std::endl;
            std::cout << "        - most Mesh formats (see meshio for a full list)" << std::endl;
            std::cout << "        - a HDF5 file with a single mesh" << std::endl;
            std::cout << "        - a HDF5 file with chunks in /chunks/x_y_z" << std::endl;
            std::cout << "        - a directory containing chunks named x_y_z.*" << std::endl;
            std::cout << "          This option requires a chunk_metadata.yaml in the folder containing" << std::endl;
            std::cout << "          at least chunk_size and voxel_size." << std::endl;
            std::cout << std::endl;
            std::cout << "<outputDir> is the directory to create the output in." << std::endl;
            std::cout << "    THE CONTENT OF THIS DIRECTORY WILL BE DELETED!" << std::endl;
        }

        notify(variables);

        if (help)
        {
            return EXIT_SUCCESS;
        }

        if (variables.count("inputFile") == 0)
        {
            throw error("Missing <inputFile> Parameter");
        }
        if (!fs::exists(input_file))
        {
            throw error("Input file does not exist");
        }
        if (!fs::is_directory(input_file))
        {
            input_file_extension = input_file.extension().string();
            if (input_file_extension.empty())
            {
                throw error("Input file has no extension");
            }
            input_file_extension = input_file_extension.substr(1);
        }

        if (reduction_factor < 0.0f || reduction_factor > 1.0f)
        {
            throw error("reductionFactor must be between 0 and 1");
        }

        has_chunk_size = variables.count("chunkSize") > 0;
    }
    catch (const boost::program_options::error& ex)
    {
        std::cerr << ex.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << "Use '--help' to see the list of possible options" << std::endl;
        return EXIT_FAILURE;
    }

    fs::remove_all(output_dir);
    fs::create_directories(output_dir);

    Mesh mesh;
    std::unordered_map<Vector3i, Tree::Ptr> chunks;
    std::vector<Texture> textures;

    std::cout << timestamp << "Reading mesh " << input_file;

    std::shared_ptr<HighFive::File> mesh_file = nullptr;
    if (allowedMemUsage < AllowedMemoryUsage::Unbounded)
    {
        mesh_file = hdf5util::open("temp_meshes.h5", HighFive::File::Truncate);
    }

    // ==================== Read Mesh ====================

    if (fs::is_directory(input_file))
    {
        std::cout << " from chunks in a folder" << std::endl;
        std::string path = input_file.string();
        if (path.back() != '/')
        {
            path += '/';
        }
        YAML::Node metadata = YAML::LoadFile(path + "chunk_metadata.yaml");
        chunk_size = metadata["chunk_size"].as<float>();
        float voxel_size = metadata["voxel_size"].as<float>();

        std::vector<std::string> chunk_files;
        for (auto& entry : fs::directory_iterator(input_file))
        {
            chunk_files.push_back(entry.path().string());
        }

        read_chunks(chunks, chunk_files, chunk_size, voxel_size, mesh_file,
                    [&path](auto & mesh, auto & filename)
        {
            mesh.read(path + filename);
        });
    }
    else if (input_file_extension == "h5")
    {
        const auto kernel = std::make_shared<HDF5Kernel>(input_file.string());
        const auto root = kernel->m_hdf5File->getGroup("/");
        if (root.hasAttribute("chunk_size"))
        {
            std::cout << " from chunks in Hdf5" << std::endl;

            chunk_size = hdf5util::getAttribute<float>(root, "chunk_size").get();
            float voxel_size = hdf5util::getAttribute<float>(root, "voxel_size").get();

            auto chunk_group = std::make_shared<HighFive::Group>(root.getGroup("/chunks"));
            std::vector<std::string> chunk_files = chunk_group->listObjectNames();

            read_chunks(chunks, chunk_files, chunk_size, voxel_size, mesh_file,
                        [&chunk_group](auto & mesh, auto & group_name)
            {
                mesh.read(chunk_group->getGroup(group_name));
            });
        }
        else
        {
            std::cout << " using meshio::HDF5IO" << std::endl;

            if (!has_chunk_size)
            {
                print_chunk_size_error();
                return EXIT_FAILURE;
            }

            std::vector<std::string> mesh_names;
            kernel->subGroupNames("/meshes/", mesh_names);
            auto mesh_name = mesh_names.front();

            auto schema = std::make_shared<MeshSchemaHDF5>();
            meshio::HDF5IO io(kernel, schema);
            MeshBufferPtr buffer;

            buffer = io.loadMesh(mesh_name);

            std::cout << timestamp << "Converting to PMPMesh" << std::endl;
            mesh = Mesh(buffer);

            if (!buffer->getMaterials().empty())
            {
                textures = buffer->getTextures();

                auto materials = buffer->getMaterials();
                std::vector<pmp::IndexType> mat_to_texture(materials.size(), pmp::PMP_MAX_INDEX);
                for (size_t i = 0; i < materials.size(); i++)
                {
                    auto& material = materials[i];
                    if (!material.m_texture)
                    {
                        continue;
                    }
                    auto& texture = mat_to_texture[i];
                    texture = material.m_texture->idx();
                    for (auto& [ name, tex ] : material.m_layers)
                    {
                        if (name.find("rgb") != std::string::npos || name.find("RGB") != std::string::npos)
                        {
                            texture = tex.idx();
                            break;
                        }
                    }
                }

                auto& surface_mesh = mesh.getSurfaceMesh();
                auto f_material = surface_mesh.face_property<pmp::IndexType>("f:material");
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < surface_mesh.faces_size(); i++)
                {
                    pmp::Face fH(i);
                    if (!surface_mesh.is_deleted(fH))
                    {
                        f_material[fH] = mat_to_texture[f_material[fH]];
                    }
                }
            }

        }
    }
    else if (pmp::SurfaceMeshIO::supports_extension(input_file_extension))
    {
        std::cout << " using pmp::SurfaceMeshIO" << std::endl;

        if (!has_chunk_size)
        {
            print_chunk_size_error();
            return EXIT_FAILURE;
        }

        mesh.getSurfaceMesh().read(input_file.string());
    }
    else
    {
        std::cout << " using ModelFactory" << std::endl;

        if (!has_chunk_size)
        {
            print_chunk_size_error();
            return EXIT_FAILURE;
        }

        ModelPtr model = ModelFactory::readModel(input_file.string());
        std::cout << timestamp << "Converting to PMPMesh" << std::endl;
        mesh = Mesh(model->m_mesh);
    }


    if (mesh.numFaces() > 0)
    {
        auto& surface_mesh = mesh.getSurfaceMesh();
        if (fix_mesh)
        {
            std::cout << timestamp << "Fixing mesh" << std::endl;
            surface_mesh.duplicate_non_manifold_vertices();
            surface_mesh.remove_degenerate_faces();
        }
        surface_mesh.garbage_collection();

        std::cout << timestamp << "Calculating normals" << std::endl;
        pmp::SurfaceNormals::compute_vertex_normals(surface_mesh, flip_point);

        for (auto file : mesh_out_files)
        {
            std::cout << timestamp << "Writing mesh to " << file << std::endl;
            surface_mesh.write(file.string());
        }
    }

    // ==================== Split Mesh and create Tree ====================

    Tree::Ptr tree;

    if (!chunks.empty())
    {
        tree = Tree::partition(std::move(chunks), combine_depth);
    }
    else
    {
        std::vector<Mesh> meshes;

        if (textures.empty())
        {
            meshes.push_back(std::move(mesh));
        }
        else
        {
            auto& surface_mesh = mesh.getSurfaceMesh();
            auto f_dist = surface_mesh.get_face_property<pmp::IndexType>("f:material");
            auto v_dist = surface_mesh.add_vertex_property<pmp::IndexType>("v:material", pmp::PMP_MAX_INDEX);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < surface_mesh.faces_size(); i++)
            {
                pmp::Face fH(i);
                if (!surface_mesh.is_deleted(fH))
                {
                    auto id = f_dist[fH];
                    for (auto vH : surface_mesh.vertices(fH))
                    {
                        if (v_dist[vH] != pmp::PMP_MAX_INDEX && v_dist[vH] != id)
                        {
                            std::cout << "ERROR: vertex " << vH << " has multiple materials" << std::endl;
                        }
                        v_dist[vH] = id;
                    }
                }
            }
            std::vector<pmp::SurfaceMesh> split_meshes(textures.size());
            surface_mesh.split_mesh(split_meshes, f_dist, v_dist);

            mesh = {};

            for (size_t i = 0; i < split_meshes.size(); ++i)
            {
                auto& mesh = meshes.emplace_back();
                mesh.getSurfaceMesh() = std::move(split_meshes[i]);
                mesh.setTexture(textures[i]);
            }
        }

        std::vector<Tree::Ptr> segments;
        for (auto& mesh : meshes)
        {
            LazyMesh lazy_mesh(std::move(mesh), mesh_file, true);
            if (chunk_size > 0)
            {
                segments.push_back(Tree::partition(lazy_mesh, chunk_size, combine_depth));
            }
            else
            {
                auto bb = lazy_mesh.get()->getSurfaceMesh().bounds();
                if (allowedMemUsage < AllowedMemoryUsage::Unbounded)
                {
                    lazy_mesh.allowUnload();
                }
                // Add the mesh as a child to a node to have one LOD
                auto leaf = Tree::leaf(std::move(lazy_mesh), bb);
                segments.emplace_back(Tree::node())->children().push_back(std::move(leaf));
            }
        }
        tree = Tree::partition(std::move(segments), 0);
    }

    tree->refresh();
    std::cout << timestamp << "Constructed tree with depth " << tree->depth() << ". Creating LOD" << std::endl;
    tree->finalize(allowedMemUsage, reduction_factor, normal_deviation);

    // ==================== Write to file ====================

    std::cout << timestamp << "Creating 3D Tiles" << std::endl;

    IO io(output_dir.string());
    io.write(tree, compress, scale);

    tree.reset();

    if (mesh_file)
    {
        auto name = mesh_file->getName();
        mesh_file.reset();
        fs::remove(name);
    }

    std::cout << timestamp << "Finished" << std::endl;

    return 0;
}


void read_chunks(std::unordered_map<Vector3i, Tree::Ptr>& chunks,
                 const std::vector<std::string>& chunk_files,
                 float chunk_size, float voxel_size,
                 std::shared_ptr<HighFive::File> mesh_file,
                 std::function<void(pmp::SurfaceMesh&, const std::string&)> read_mesh)
{
    size_t empty = 0;

    ProgressBar progress(chunk_files.size(), "Reading chunks");
    for (size_t i = 0; i < chunk_files.size(); i++)
    {
        auto name = chunk_files[i];
        int x, y, z;
        int read = std::sscanf(name.c_str(), "%d_%d_%d", &x, &y, &z);
        if (read != 3)
        {
            std::cout << "Skipping " << name << std::endl;
            ++progress;
            continue;
        }
        Vector3i chunk_pos(x, y, z);

        Mesh pmp_mesh;
        auto& mesh = pmp_mesh.getSurfaceMesh();
        read_mesh(mesh, name);

        // calculate normals before trimming for better normals along borders
        pmp::SurfaceNormals::compute_vertex_normals(mesh, flip_point);

        pmp::Point min = chunk_pos.cast<float>() * chunk_size;
        pmp::Point max = min + pmp::Point::Constant(chunk_size);
        pmp::Point epsilon = pmp::Point::Constant(0.0001);
        pmp::BoundingBox expected_bb(min - epsilon, max + epsilon);
        Tree::trimChunkOverlap(pmp_mesh, expected_bb);

        if (mesh.n_faces() == 0)
        {
            empty++;
            ++progress;
            continue;
        }

        auto bb = mesh.bounds();
        chunks.emplace(chunk_pos, Tree::leaf(LazyMesh(std::move(pmp_mesh), mesh_file), bb));

        ++progress;
    }
    std::cout << "\r";

    if (empty > 0)
    {
        std::cout << timestamp << empty << " chunks contained only overlap with other chunks" << std::endl;
    }

    std::cout << timestamp << "Found " << chunks.size() << " Chunks" << std::endl;
}
