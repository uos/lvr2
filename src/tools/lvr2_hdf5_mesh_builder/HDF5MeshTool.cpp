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

#include <iostream>
#include <vector>
#include <algorithm>

#include <string.h>

#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/algorithm/NormalAlgorithms.hpp"
#include "lvr2/algorithm/GeometryAlgorithms.hpp"
#include "lvr2/geometry/HalfEdgeMesh.hpp"
#include "lvr2/algorithm/ReductionAlgorithms.hpp"

#include "Options.hpp"

#include <string>
#include <sstream>
#include <algorithm>
#include <cstring>
#include <memory>

#include "lvr2/io/GHDF5IO.hpp"
#include "lvr2/io/hdf5/MeshIO.hpp"

using namespace lvr2;

int main( int argc, char ** argv )
{
  hdf5meshtool::Options options(argc, argv);
  std::cout << timestamp << "Load HDF5 file structure..." << std::endl;
  using HDF5MeshToolIO = lvr2::Hdf5IO<
          lvr2::hdf5features::ArrayIO,
          lvr2::hdf5features::ChannelIO,
          lvr2::hdf5features::VariantChannelIO,
          lvr2::hdf5features::MeshIO>;

  // Get extension
  boost::filesystem::path selectedFile(options.getInputFile());
  std::string extension = selectedFile.extension().string();
  MeshBufferPtr meshBuffer;
  HDF5MeshToolIO hdf5In;
  bool readFromHdf5 = false;

  // check extension
  if (extension == ".h5") // use new Hdf5IO
  {
    hdf5In.open(options.getInputFile());
    if (hdf5In.m_hdf5_file->isValid()) // TODO: update hdf5io to return bool on open()
    {
        readFromHdf5 = true;
    }
    meshBuffer = hdf5In.loadMesh(options.getMeshName());
  }
  else // use model reader
  {
    ModelPtr model = ModelFactory::readModel(options.getInputFile());
    meshBuffer = model->m_mesh;
  }
  if (meshBuffer != nullptr)
  {
    std::cout << timestamp << "Building mesh from buffers..." << std::endl;
    HalfEdgeMesh<BaseVector<float>> hem(meshBuffer);
    HDF5MeshToolIO hdf5;
    bool writeToHdf5Input = false;
    if (readFromHdf5 && options.getInputFile() == options.getOutputFile())
    {
      hdf5 = hdf5In;
      writeToHdf5Input = true;
    }
    else
    {
      hdf5.open(options.getOutputFile());
    }
    hdf5.setMeshName(options.getMeshName());

    // face normals
    DenseFaceMap<Normal<float>> faceNormals;
    boost::optional<DenseFaceMap<Normal<float>>> faceNormalsOpt;
    if (readFromHdf5)
    {
      faceNormalsOpt = hdf5In.getDenseAttributeMap<DenseFaceMap<Normal<float>>>("face_normals");
    }
    if (faceNormalsOpt)
    {
      std::cout << timestamp << "Using existing face normals..." << std::endl;
      faceNormals = *faceNormalsOpt;
    }
    else
    {
      std::cout << timestamp << "Computing face normals..." << std::endl;
      faceNormals = calcFaceNormals(hem);
    }
    if(options.getEdgeCollapseNum() > 0)
    {
      double percent = options.getEdgeCollapseNum() > 100 ? 1 : options.getEdgeCollapseNum() / 100.0;
      size_t numCollapse = static_cast<size_t>(percent * hem.numEdges());
      std::cout << timestamp << "Reduce mesh by collapsing " << percent * 100
        << "% of the edges (" << numCollapse << " out of " << hem.numEdges() << ")" << std::endl;
      simpleMeshReduction(hem, numCollapse, faceNormals);
    }

    // add mesh to file
    if(options.getEdgeCollapseNum() > 0 || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding mesh to file..." << std::endl;
      // add mesh to file
      bool addedMesh = hdf5.addMesh(hem);
      if (addedMesh)
      {
        std::cout << timestamp << "successfully added mesh" << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add the mesh!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Mesh already included." << std::endl;
    }

    // add face normals to file
    if(!faceNormalsOpt || options.getEdgeCollapseNum() > 0 || !writeToHdf5Input)
    {
      bool addedFaceNormals = hdf5.addDenseAttributeMap<DenseFaceMap<Normal<float>>>(
              hem, faceNormals, "face_normals");
      if(addedFaceNormals)
      {
        std::cout << timestamp << "successfully added face normals" << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add face normals!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Face normals already included." << std::endl;
    }

    // vertex normals
    DenseVertexMap<Normal<float>> vertexNormals;
    boost::optional<DenseVertexMap<Normal<float>>> vertexNormalsOpt;
    if (readFromHdf5)
    {
      vertexNormalsOpt = hdf5In.getDenseAttributeMap<DenseVertexMap<Normal<float>>>("vertex_normals");
    }
    if (vertexNormalsOpt)
    {
      std::cout << timestamp << "Using existing vertex normals..." << std::endl;
      vertexNormals = *vertexNormalsOpt;
    }
    else
    {
      std::cout << timestamp << "Computing vertex normals..." << std::endl;
      vertexNormals = calcVertexNormals(hem, faceNormals);
    }
    if (!vertexNormalsOpt || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding vertex normals..." << std::endl;
      bool addedVertexNormals = hdf5.addDenseAttributeMap<DenseVertexMap<Normal<float>>>(
              hem, vertexNormals, "vertex_normals");
      if (addedVertexNormals)
      {
        std::cout << timestamp << "successfully added vertex normals" << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add vertex normals!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Vertex normals already included." << std::endl;
    }

    // vertex average angles
    DenseVertexMap<float> averageAngles;
    boost::optional<DenseVertexMap<float>> averageAnglesOpt;
    if (readFromHdf5)
    {
      averageAnglesOpt = hdf5In.getDenseAttributeMap<DenseVertexMap<float>>("average_angles");
    }
    if (averageAnglesOpt)
    {
      std::cout << timestamp << "Using existing vertex average angles..." << std::endl;
      averageAngles = *averageAnglesOpt;
    }
    else
    {
      std::cout << timestamp << "Computing vertex average angles..." << std::endl;
      averageAngles = calcAverageVertexAngles(hem, vertexNormals);
    }
    if (!averageAnglesOpt || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding vertex average angles..." << std::endl;
      bool addedAverageAngles = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(
              hem, averageAngles, "average_angles");
      if (addedAverageAngles)
      {
        std::cout << timestamp << "successfully added vertex average angles" << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add vertex average angles!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Vertex average angles already included." << std::endl;
    }

    // roughness
    DenseVertexMap<float> roughness;
    boost::optional<DenseVertexMap<float>> roughnessOpt;
    if (readFromHdf5)
    {
      roughnessOpt = hdf5In.getDenseAttributeMap<DenseVertexMap<float>>("roughness");
    }
    if (roughnessOpt)
    {
      std::cout << timestamp << "Using existing roughness..." << std::endl;
      roughness = *roughnessOpt;
    }
    else
    {
      std::cout << timestamp << "Computing roughness..." << std::endl;
      roughness = calcVertexRoughness(hem, 0.3, vertexNormals);
    }
    if (!roughnessOpt || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding roughness..." << std::endl;
      bool addedRoughness = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(
              hem, roughness, "roughness");
      if (addedRoughness)
      {
        std::cout << timestamp << "successfully added roughness." << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add roughness!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Roughness already included." << std::endl;
    }

    // height differences
    DenseVertexMap<float> heightDifferences;
    boost::optional<DenseVertexMap<float>> heightDifferencesOpt;
    if (readFromHdf5)
    {
      heightDifferencesOpt = hdf5In.getDenseAttributeMap<DenseVertexMap<float>>("height_diff");
    }
    if (heightDifferencesOpt)
    {
      std::cout << timestamp << "Using existing height differences..." << std::endl;
      heightDifferences = *heightDifferencesOpt;
    }
    else
    {
      std::cout << timestamp << "Computing height differences..." << std::endl;
      heightDifferences = calcVertexHeightDifferences(hem, 0.3);
    }
    if (!heightDifferencesOpt || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding roughness..." << std::endl;
      bool addedHeightDiff = hdf5.addDenseAttributeMap<DenseVertexMap<float>>(
              hem, heightDifferences, "height_diff");
      if (addedHeightDiff)
      {
        std::cout << timestamp << "successfully added height differences." << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add height differences!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Height differences already included." << std::endl;
    }
  }
  else
  {
    std::cout << timestamp << "Error reading mesh data from "
              << options.getOutputFile() << std::endl;
  }

  return 0;
}
