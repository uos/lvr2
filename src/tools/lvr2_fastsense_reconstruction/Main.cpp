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

/**
 * log(CHUNK_SIZE).
 * The side length is a power of 2 so that divisions by the side length can be accomplished by shifting.
 */
constexpr int CHUNK_SHIFT = 6;

/// Side length of the cube-shaped chunks (2^CHUNK_SHIFT).
constexpr int CHUNK_SIZE = 1 << CHUNK_SHIFT;

int main(int argc, char** argv)
{
    /*
    1. Einlesen -- fertig :D
    2. In geerbte Klasse von HashGrid einfügen
    3. Mit HashGrid FastReconstruction aufrufen
    4. Mesh mit vorhandenen IO rausschreiben
    */
    
    // Read
    HighFive::File f("/home/fastsense/Develop/map.h5", HighFive::File::ReadOnly); // TODO: Path and name as command line input
    HighFive::Group g = f.getGroup("/map");
    
    lvr2::BaseVector<int> min(0, 0, 0); 
    lvr2::BaseVector<int> max(0, 0, 0);

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

    std::cout << min << std::endl;
    std::cout << max << std::endl;

    lvr2::BoundingBox<lvr2::BaseVector<int>> bb(min, max);
    auto grid = std::make_shared<lvr2::HashGrid<lvr2::BaseVector<int>, lvr2::FastBox<lvr2::BaseVector<int>>>>(CHUNK_SIZE, bb);

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
        
        // Fill the grid with valid tsdf values
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
                    if (weight > 0)
                    {
                        grid->addLatticePoint(x, y, z, tsdf_value);
                    }
                }
            }
        }
    }

    lvr2::FastReconstruction<lvr2::BaseVector<int>, lvr2::FastBox<lvr2::BaseVector<int>>> reconstruction(grid);
    lvr2::HalfEdgeMesh<lvr2::BaseVector<int>> mesh;
    reconstruction.getMesh(mesh);

    std::cout << "The end ^_^" << std::endl;

    return 0;
}
