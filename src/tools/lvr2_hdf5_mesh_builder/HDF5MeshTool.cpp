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
#include <lvr2/algorithm/GeometryAlgorithms.hpp>
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
  std::cout << timestamp << "Load HDF5 file structure..." << std::endl;
  HDF5IO hdf5(options.getOutputFile(), options.getMeshName(), true);

  ModelPtr model = ModelFactory::readModel(options.getInputFile());
  if(MeshBufferPtr meshBuffer = model->m_mesh){
    std::cout << timestamp << "Building mesh from buffers..." << std::endl;

    HalfEdgeMesh<BaseVector<float>> hem(meshBuffer);

    std::cout << timestamp << "Adding mesh to file..." << std::endl;
    // mesh
    bool addedMesh = hdf5.addMesh(hem);
    if(addedMesh) std::cout << timestamp << "successfully added mesh" << std::endl;
    else std::cout << timestamp << "could not add the mesh!" << std::endl;

    // face normals
    std::cout << timestamp << "Computing face normals..." << std::endl;
    auto faceNormals = calcFaceNormals(hem);
    bool addedFaceNormals = hdf5.addDenseAttributeMap<DenseFaceMap<Normal<float>>>(
        faceNormals, "face_normals");
    if(addedFaceNormals) std::cout << timestamp << "successfully added face normals" << std::endl;
    else std::cout << timestamp << "could not add face normals!" << std::endl;

    // vertex normals
    std::cout << timestamp << "Computing vertex normals..." << std::endl;
    auto vertexNormals = calcVertexNormals(hem, faceNormals);
    bool addedVertexNormals = hdf5.addDenseAttributeMap<DenseVertexMap<Normal<float>>>(
        vertexNormals, "vertex_normals");
    if(addedVertexNormals) std::cout << timestamp << "successfully added vertex normals" << std::endl;
    else std::cout << timestamp << "could not add vertex normals!" << std::endl;

    // vertex average angles
    std::cout << timestamp << "Computing average vertex angles..." << std::endl;
    auto averageAngles = calcAverageVertexAngles(hem, vertexNormals);
    bool addedAverageAngles = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(averageAngles, "average_angles");
    if(addedAverageAngles) std::cout << timestamp << "successfully added vertex average angles" << std::endl;
    else std::cout << timestamp << "could not add vertex average angles!" << std::endl;

    std::cout << timestamp << "Computing roughness..." << std::endl;
    auto roughness = calcVertexRoughness(hem, 0.3, vertexNormals);
    bool addedRoughness = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(roughness, "roughness");
    if(addedRoughness) std::cout << timestamp << "successfully added roughness." << std::endl;
    else std::cout << timestamp << "could not add roughness!" << std::endl;

    std::cout << timestamp << "Computing height differences..." << std::endl;
    auto heightDifferences = calcVertexHeightDifferences(hem, 0.3);
    bool addedHeightDiff = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(heightDifferences, "height_diff");
    if(addedHeightDiff) std::cout << timestamp << "successfully added height differences." << std::endl;
    else std::cout << timestamp << "could not add height differences!" << std::endl;
  }
  else
  {
    std::cout << timestamp << "Error reading mesh data from "
              << options.getOutputFile() << std::endl;
  }

  return 0;
}
