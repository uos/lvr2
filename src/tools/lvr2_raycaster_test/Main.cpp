#include <iostream>
#include <memory>
#include <tuple>
#include <stdlib.h>

#include <boost/optional.hpp>

// lvr2 includes
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/algorithm/raycasting/CLRaycaster.hpp>

using boost::optional;
using std::unique_ptr;
using std::make_unique;

using namespace lvr2;

using Vec = BaseVector<float>;
using PsSurface = lvr2::PointsetSurface<Vec>;


MeshBufferPtr genMesh(){
    MeshBufferPtr dst_mesh = MeshBufferPtr(new MeshBuffer);

    float *vertices = new float[12];
    float *normals = new float[12];
    unsigned int *face_indices = new unsigned int[12];

    // gen vertices
    vertices[0] = 200.0; vertices[1] = -500.0; vertices[2] = -500.0;
    vertices[3] = 200.0; vertices[4] = -500.0; vertices[5] = 500.0;
    vertices[6] = 200.0; vertices[7] = 500.0; vertices[8] = 500.0;
    vertices[9] = 200.0; vertices[10] = 500.0; vertices[11] = -500.0;

    // gen normals
    normals[0] = 1.0; normals[1] = 0.0; normals[2] = 0.0;
    normals[3] = 1.0; normals[4] = 0.0; normals[5] = 0.0;
    normals[6] = 1.0; normals[7] = 0.0; normals[8] = 0.0;
    normals[9] = 1.0; normals[10] = 0.0; normals[11] = 0.0;

    // gen faces
    face_indices[0] = 0; face_indices[1] = 1; face_indices[2] = 2;
    face_indices[3] = 0; face_indices[4] = 2; face_indices[5] = 3;

    floatArr vertex_arr(vertices);
    floatArr normal_arr(normals);
    indexArray face_index_arr(face_indices);
    

    // append to mesh
    dst_mesh->setVertices(vertex_arr, 4);
    dst_mesh->setVertexNormals(normal_arr);
    dst_mesh->setFaceIndices(face_index_arr, 2);

    return dst_mesh;
}

int main(int argc, char** argv){

    std::cout << "Raycast Test started" << std::endl;

    MeshBufferPtr buffer = genMesh();

    CLRaycaster<Vec> rc(buffer);

    Point<Vec> origin = {20.0,40.0,50.0};
    Vector<Vec> ray = {1.0,0.0,0.0};


    std::vector<Point<Vec> > origins;
    std::vector<Vector<Vec> > rays;

    for(int i=0; i<100; i++)
    {
        origins.push_back(origin);
        rays.push_back(ray);
    }

    
    std::cout << "TEST 1: one origin, one ray." << std::endl;

    Point<Vec> intersection;
    bool success = rc.castRay(origin, ray, intersection);

    if(success)
    {
        std::cout << "success!" << std::endl;
        std::cout << intersection << std::endl;
    } else {
        std::cout << "NOT succesful!" << std::endl;
    }

    std::cout << "TEST 2: one origin, mulitple rays." << std::endl;

    std::vector<Point<Vec> > intersections1, intersections2;
    std::vector<uint8_t> hits1, hits2;

    rc.castRays(origin, rays, intersections1, hits1);

    success = true;
    for(int i=0;i<hits1.size(); i++)
    {
        if(!hits1[i])
        {
            success = false;
        }
    }

    if(success)
    {
        std::cout << "success!" << std::endl;
        std::cout << intersections1[0] << std::endl;
    } else {
        std::cout << "NOT succesful!" << std::endl;
    }


    std::cout << "TEST 3: multiple origins, mulitple rays." << std::endl;

    rc.castRays(origins, rays, intersections2, hits2);

    success = true;
    for(int i=0;i<hits2.size(); i++)
    {
        if(!hits2[i])
        {
            success = false;
        }
    }
    if(success)
    {
        std::cout << "success!" << std::endl;
        std::cout << intersections2[0] << std::endl;
    } else {
        std::cout << "NOT succesful!" << std::endl;
    }



    return 0;
}