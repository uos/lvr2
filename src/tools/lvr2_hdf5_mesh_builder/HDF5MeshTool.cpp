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

        //floatArr vertexNormals = meshBuffer->getVertexNormals();
        //floatArr faceNormals = meshBuffer->getFaceNormals();

        HalfEdgeMesh<BaseVector<float>>* hem = nullptr;
        if(vertices && indices)
        {
            hem = new HalfEdgeMesh<BaseVector<float>>(meshBuffer);

            std::cout << timestamp << "Calculating face normals..." << std::endl;
            DenseFaceMap<Normal<BaseVector<float>>> faceNormals;
            faceNormals = calcFaceNormals(*hem);


            std::cout << timestamp << "Calculating vertex normals..." << std::endl;
            auto normals = calcVertexNormals(*hem, faceNormals);

            for(auto i : normals)
            {
                cout << normals[i] << endl;
            }

        }
    }
    else
    {
        std::cout << timestamp << "Error reading mesh data from "
                  << options.getOutputFile() << std::endl;
    }

    return 0;
}
