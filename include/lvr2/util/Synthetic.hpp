
#ifndef LVR2_UTIL_SYNTHETIC_HPP
#define LVR2_UTIL_SYNTHETIC_HPP

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/io/MeshBuffer.hpp"

namespace lvr2 {

namespace synthetic {

MeshBufferPtr genSphere(
    int num_long=50,
    int num_lat=50)
{
    MeshBufferPtr dst_mesh = MeshBufferPtr(new MeshBuffer);

    // x = cx + r * sin(alpha) * cos(beta)
    // y = cy + r * sin(alpha) * sin(beta)
    // z = cz + r * cos(alpha)

    // alpha [-pi, pi)
    // beta [-pi, pi)

    std::vector<float> vertices;
    std::vector<unsigned int> face_indices;


    float long_inc = M_PI / static_cast<float>(num_long + 1);
    float lat_inc = (2 * M_PI) / static_cast<float>(num_lat);

    // add first and last point


    // add first vertex manually
    vertices.push_back(sin(0.0) * cos(0.0));
    vertices.push_back(sin(0.0) * sin(0.0));
    vertices.push_back(cos(0.0));

    // add first faces manually
    for(int i=0; i<num_lat; i++)
    {
        int id_bl = 0;
        int id_tl = i;
        int id_tr = i + 1;

        if(i == num_lat - 1)
        {
            id_tr -= num_lat;
        }

        face_indices.push_back(id_bl);
        face_indices.push_back(id_tl + 1);
        face_indices.push_back(id_tr + 1);
    }

    for(int i=0; i<num_long; i++)
    {
        float alpha = long_inc * (i+1);

        for(int j=0; j<num_lat; j++)
        {
            float beta = lat_inc * (j+1);

            vertices.push_back(sin(alpha) * cos(beta));
            vertices.push_back(sin(alpha) * sin(beta));
            vertices.push_back(cos(alpha));

            if(i > 0)
            {

                int id_bl = num_lat * (i-1) + j;
                int id_br = num_lat * (i-1) + j + 1;
                int id_tl = num_lat * (i)   + j;
                int id_tr = num_lat * (i)   + j + 1;

                if(j == num_lat - 1)
                {
                    id_br -= num_lat;
                    id_tr -= num_lat;
                }

                // clockwise
                
                // first face
                face_indices.push_back(id_br + 1);
                face_indices.push_back(id_bl + 1);
                face_indices.push_back(id_tl + 1);

                // second face
                face_indices.push_back(id_tl + 1);
                face_indices.push_back(id_tr + 1);
                face_indices.push_back(id_br + 1);
            }
        }
    }

    // add last vertex
    vertices.push_back(sin(M_PI) * cos(2.0 * M_PI));
    vertices.push_back(sin(M_PI) * sin(2.0 * M_PI));
    vertices.push_back(cos(M_PI));

    int num_vertices = vertices.size() / 3;
    for(int i=num_vertices-1-num_lat; i<num_vertices-1; i++)
    {
        int id_bl = i;
        int id_br = i+1;
        int id_tl = num_vertices-1;

        if(id_br == id_tl)
        {
            id_br -= num_lat;
        }

        face_indices.push_back(id_br);
        face_indices.push_back(id_bl);
        face_indices.push_back(id_tl);
    }


    // COPY
    floatArr vertex_arr(new float[vertices.size()]);
    indexArray index_arr(new unsigned int[face_indices.size()]);

    for(int i=0; i<vertices.size(); i++)
    {
        vertex_arr[i] = vertices[i];
    }

    for(int i=0; i<face_indices.size(); i++)
    {
        index_arr[i] = face_indices[i];
    }


    dst_mesh->setVertices(vertex_arr, vertices.size() / 3);
    dst_mesh->setFaceIndices(index_arr, face_indices.size() / 3);

    return dst_mesh;
}

} // namespace synthetic

} // namespace lvr2

#endif // LVR2_UTIL_SYNTHETIC_HPP