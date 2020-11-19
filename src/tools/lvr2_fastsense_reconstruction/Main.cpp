/**
 * Main.cpp
 *
 * Created on: 11.11.2020
 * Author: Steffen Hinderink, Marc Eisoldt, Juri Vana, (Patrick Hoffmann)
 * 
 * Function used to reconstruct a mesh from the GlobalMap of the FastSenseSLAM to inspect the quality of the SLAM approach
 */

#include <iostream>
#include <highfive/H5File.hpp>

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/BoundingBox.hpp"
#include "lvr2/reconstruction/HashGrid.hpp"
#include "lvr2/reconstruction/FastBox.hpp"
#include "lvr2/reconstruction/FastReconstruction.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"

#include "lvr2/algorithm/FinalizeAlgorithms.hpp"
#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"

#include "lvr2/io/PLYIO.hpp"

/**
 * log(CHUNK_SIZE).
 * The side length is a power of 2 so that divisions by the side length can be accomplished by shifting.
 */
constexpr int CHUNK_SHIFT = 6;

/// Side length of the cube-shaped chunks (2^CHUNK_SHIFT).
constexpr int CHUNK_SIZE = 1 << CHUNK_SHIFT;

/// Scale for the boudingbox vectors
constexpr int SCALE = 100;

/// HDF5 class structure for saving meshes
using HDF5MeshToolIO = lvr2::Hdf5IO<lvr2::hdf5features::ArrayIO,
                                    lvr2::hdf5features::ChannelIO,
                                    lvr2::hdf5features::VariantChannelIO,
                                    lvr2::hdf5features::MeshIO>;

/**
 * @brief Tells the user how to call this program
 * 
 * @param prog_name Name of this program
 */
void print_usage(const std::string& prog_name)
{
    std::cout << prog_name << " <path-to_hdf5-map-file> <save-directory> <mesh-name>" << std::endl;
}

int main(int argc, char** argv)
{   
    if (argc != 4)
    {
        print_usage(argv[0]);
        return 0;
    }

    // Initialize file variables
    std::string src_name(argv[1]);
    std::string dst_dir_name(argv[2]);
    std::string mesh_name_h5(std::string(argv[3]) + std::string(".h5"));
    std::string mesh_name_ply(std::string(argv[3]) + std::string(".ply"));

    std::cout << "Open map file: " << src_name << std::endl;
    
    // Read
    HighFive::File f(src_name, HighFive::File::ReadOnly); // TODO: Path and name as command line input
    HighFive::Group g = f.getGroup("/map");

    lvr2::BaseVector<int> min(0, 0, 0); 
    lvr2::BaseVector<int> max(0, 0, 0);

    std::cout << "Determine map boundingbox..." << std::endl;

    // Determine the boundingbox of the complete map
    for (auto tag : g.listObjectNames())
    {
        std::vector<int> chunk_pos;
        std::string delimiter = "_";
        size_t pos = 0;
        std::string token;

        while ((pos = tag.find(delimiter)) != std::string::npos)
        {
            token = tag.substr(0, pos);
            chunk_pos.push_back(std::stoi(token));
            tag.erase(0, pos + delimiter.length());
        }

        chunk_pos.push_back(std::stoi(tag));

        int index = 0;

        for (const auto& coord : chunk_pos)
        {   
            if (coord < min[index])
            {
                min[index] = coord;
            }

            if (coord > max[index])
            {
                max[index] = coord;
            }

            ++index;
        }

    }

    min *= CHUNK_SIZE;
    max = max * CHUNK_SIZE + lvr2::BaseVector<int>(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);

    std::cout << "Create and fill reconstruction grid..." << std::endl;

    // Create the Grid for the reconstruction algorithm
    lvr2::BoundingBox<lvr2::BaseVector<int>> bb(min * SCALE, max * SCALE);
    auto grid = std::make_shared<lvr2::HashGrid<lvr2::BaseVector<int>, lvr2::FastBox<lvr2::BaseVector<int>>>>(CHUNK_SIZE, bb);

    // Fill the grid with the valid TSDF values of the map
    for (auto tag : g.listObjectNames())
    {
        // Get the chunk data
        HighFive::DataSet d = g.getDataSet(tag);
        std::vector<int> chunk_data;
        d.read(chunk_data);
        // Get the chunk position
        std::vector<int> chunk_pos;
        std::string delimiter = "_";
        size_t pos = 0;
        std::string token;
        while ((pos = tag.find(delimiter)) != std::string::npos)
        {
            token = tag.substr(0, pos);
            chunk_pos.push_back(std::stoi(token));
            tag.erase(0, pos + delimiter.length());
        }
        chunk_pos.push_back(std::stoi(tag));
        
        for (int i = 0; i < CHUNK_SIZE; i++)
        {
            for (int j = 0; j < CHUNK_SIZE; j++)
            {
                for (int k = 0; k < CHUNK_SIZE; k++)
                {
                    int tsdf_value = chunk_data[(CHUNK_SIZE * CHUNK_SIZE * i + CHUNK_SIZE * j + k) * 2];
                    int weight = chunk_data[(CHUNK_SIZE * CHUNK_SIZE * i + CHUNK_SIZE * j + k) * 2 + 1];
                    int x = CHUNK_SIZE * chunk_pos[0] + i;
                    int y = CHUNK_SIZE * chunk_pos[1] + j;
                    int z = CHUNK_SIZE * chunk_pos[2] + k;
                    
                    // Only touched cells are considered
                    if (weight > 0)
                    {
                        // Insert TSDF value
                        grid->addLatticePoint(x, y, z, tsdf_value);
                
                    }
                }
            }
        }
    }

    std::cout << "Reconstruct mesh with a marching cubes algorithm..." << std::endl;

    // Reconstruct the mesh with a marching cubes algorithm
    lvr2::FastReconstruction<lvr2::BaseVector<int>, lvr2::FastBox<lvr2::BaseVector<int>>> reconstruction(grid);
    lvr2::HalfEdgeMesh<lvr2::BaseVector<int>> mesh;
    reconstruction.getMesh(mesh);

    std::cout << "Finished reconstruction!" << std::endl;

    // Convert halfedgemesh to an IO format
    lvr2::SimpleFinalizer<lvr2::BaseVector<int>> finalizer;
    auto buffer = finalizer.apply(mesh);
    
    std::cout << "Write mesh into HDF5 file..." << std::endl;

    // Write mesh into HDF5 file
    HDF5MeshToolIO hdf5;
    hdf5.open(dst_dir_name + "/" + mesh_name_h5);
    hdf5.save("tsdf_mesh", buffer);

    std::cout << "Write mesh into PLY file..." << std::endl;

    // Write mesh into PLY file
    auto model_ptr = std::make_shared<lvr2::Model>(buffer);
    lvr2::PLYIO ply_io;
    ply_io.save(model_ptr, dst_dir_name + "/" + mesh_name_ply);

    std::cout << "mesh saved!" << std::endl;

    return 0;
}
