/**
 * Copyright (c) 2018, University Osnabrück
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

/**
 * @file       HDF5Tool.cpp
 * @brief      Reads spectral PNGs and point clouds and writes them into a
 *             HDF5 file.
 * @details    
 * @author     Thomas Wiemann
 */

#include <iostream>
#include <vector>
#include <algorithm>

#include <string.h>

#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/io/Timestamp.hpp>
#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/algorithm/NormalAlgorithms.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>

#include "Options.hpp"

#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <memory>

using namespace lvr2;

int main( int argc, char ** argv )
{
    hdf5meshtool::Options options(argc, argv);
    HDF5IO hdf5(options.getOutputFile(), true);

    ModelPtr model = ModelFactory::readModel(options.getInputFile());
    MeshBufferPtr meshBuffer = model->m_mesh;



    if(meshBuffer)
    {
        floatArr vertices = meshBuffer->getVertices();
        indexArray indices = meshBuffer->getFaceIndices();

        HalfEdgeMesh<BaseVector<float>>* hem = nullptr;
        if(vertices && indices)
        {
            hem = new HalfEdgeMesh<BaseVector<float>>(meshBuffer);

            std::cout << timestamp << "Calculating face normals..." << std::endl;
            DenseFaceMap<Normal<BaseVector<float>>> faceNormals;
            faceNormals = calcFaceNormals(*hem);


            std::cout << timestamp << "Calculating vertex normals..." << std::endl;
            auto normals = calcVertexNormals(*hem, faceNormals);

            //            for(auto i : normals)
            //            {
            //                cout << normals[i] << endl;
            //            }

            std::cout << "Creating HDF5 file..." << std::endl;
            hdf5.addArray("meshes/triangle_mesh", "vertices", meshBuffer->numVertices() * 3, vertices);
            hdf5.addArray("meshes/triangle_mesh", "indices", meshBuffer->numFaces() * 3, indices);

            std::cout << "Converting face normals..." << std::endl;
            floatArr faceNormalArray(new float [meshBuffer->numFaces() * 3]);
            int c = 0;
            for(auto i : faceNormals)
            {
                Normal<BaseVector<float>> n = faceNormals[i];
                faceNormalArray[3 * c    ] = n[0];
                faceNormalArray[3 * c + 1] = n[1];
                faceNormalArray[3 * c + 2] = n[2];
                c++;
            }

            std::cout << "Converting vertex normals" << std::endl;
            floatArr vertexNormalArray(new float[meshBuffer->numVertices() * 3]);
            c = 0;
            for(auto i : normals)
            {
                Normal<BaseVector<float>> n = normals[i];
                vertexNormalArray[3 * c    ] = n[0];
                vertexNormalArray[3 * c + 1] = n[1];
                vertexNormalArray[3 * c + 2] = n[2];
                c++;
            }

            hdf5.addArray("meshes/triangle_mesh/face_attributes", "normals", meshBuffer->numFaces() * 3, faceNormalArray);
            hdf5.addArray("meshes/triangle_mesh/vertex_attributes", "normals", meshBuffer->numVertices() * 3, vertexNormalArray);

        }
    }
    else
    {
        std::cout << timestamp << "Error reading mesh data from "
                  << options.getOutputFile() << std::endl;
    }

    return 0;
}
