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

#include "lvr2/io/hdf5/HDF5FeatureBase.hpp"
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
    HalfEdgeMesh<BaseVector<float>> hem;
    size_t numFaces = meshBuffer->numFaces();
    size_t numVertices = meshBuffer->numVertices();
    std::cout << timestamp << "Building mesh from buffers with " << numFaces
      << " faces and " << numVertices << " vertices..." << std::endl;

    floatArr vertices = meshBuffer->getVertices();
    indexArray indices = meshBuffer->getFaceIndices();

    for(size_t i = 0; i < numVertices; i++)
    {
      size_t pos = 3 * i;
      hem.addVertex(BaseVector<float>(
          vertices[pos],
          vertices[pos + 1],
          vertices[pos + 2]));
    }

    size_t invalid_face_cnt = 0;
    for(size_t i = 0; i < numFaces; i++) {
      size_t pos = 3 * i;
      VertexHandle v1(indices[pos]);
      VertexHandle v2(indices[pos + 1]);
      VertexHandle v3(indices[pos + 2]);
      try{
        hem.addFace(v1, v2, v3);
      }
      catch(lvr2::PanicException)
      {
        invalid_face_cnt++;
      }
    }

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
    else if (meshBuffer != nullptr && meshBuffer->hasVertexNormals())
    {
      std::cout << timestamp << "Using existing vertex normals from mesh buffer..." << std::endl;
      const FloatChannelOptional channel_opt = meshBuffer->getChannel<float>("vertex_normals");
      if (channel_opt && channel_opt.get().width() == 3 and channel_opt.get().numElements() == hem.numVertices())
      {
        auto &channel = channel_opt.get();
        vertexNormals.reserve(channel.numElements());
        for (size_t i = 0; i < channel.numElements(); i++)
        {
          vertexNormals.insert(VertexHandle(i), channel[i]);
        }
      }
      else
      {
        std::cerr << timestamp << "Error while reading vertex normals..." << std::endl;
      }
    }

    if(vertexNormals.numValues() == 0)
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

    // vertex colors
    using color = std::array<uint8_t, 3>;
    DenseVertexMap<color> colors;
    boost::optional<DenseVertexMap<color>> colorsOpt;
    ChannelOptional<uint8_t> channel_opt;
    if (readFromHdf5)
    {
      colorsOpt = hdf5In.getDenseAttributeMap<DenseVertexMap<color>>("vertex_colors");
    }
    if (colorsOpt)
    {
      std::cout << timestamp << "Using existing vertex colors..." << std::endl;
      colors = *colorsOpt;
    }
    else if (meshBuffer != nullptr && (channel_opt = meshBuffer->getChannel<uint8_t>("vertex_colors"))
      && channel_opt && channel_opt.get().width() == 3 && channel_opt.get().numElements() == hem.numVertices()) {
      std::cout << timestamp << "Using existing colors from mesh buffer..." << std::endl;

      auto &channel = channel_opt.get();
      colors.reserve(channel.numElements());
      for (size_t i = 0; i < channel.numElements(); i++)
      {
        colors.insert(VertexHandle(i), channel[i]);
      }
    }
    if (!colorsOpt || !writeToHdf5Input)
    {
      std::cout << timestamp << "Adding vertex colors..." << std::endl;
      bool addedVertexColors = hdf5.addDenseAttributeMap<DenseVertexMap<color>>(
          hem, colors, "vertex_colors");
      if (addedVertexColors)
      {
        std::cout << timestamp << "successfully added vertex colors" << std::endl;
      }
      else
      {
        std::cout << timestamp << "could not add vertex colors!" << std::endl;
      }
    }
    else
    {
      std::cout << timestamp << "Vertex colors already included." << std::endl;
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
      std::cout << timestamp << "Computing roughness with a local radius of "
                << options.getLocalRadius() << "m ..." << std::endl;
      roughness = calcVertexRoughness(hem, options.getLocalRadius(), vertexNormals);
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
      std::cout << timestamp << "Computing height diff with a local radius of "
                << options.getLocalRadius() << "m ..." << std::endl;
      heightDifferences = calcVertexHeightDifferences(hem, options.getLocalRadius());
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
