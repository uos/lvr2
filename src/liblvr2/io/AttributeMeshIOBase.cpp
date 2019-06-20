/**
 * Copyright (c) 2019, University Osnabrück
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

#include "lvr2/io/AttributeMeshIOBase.hpp"
namespace lvr2{

const std::string attribute_type<EdgeHandle>::attr_group            = "edge_attributes";
const std::string attribute_type<OptionalEdgeHandle>::attr_group    = "edge_attributes";
const std::string attribute_type<VertexHandle>::attr_group          = "vertex_attributes";
const std::string attribute_type<OptionalVertexHandle>::attr_group  = "vertex_attributes";
const std::string attribute_type<FaceHandle>::attr_group            = "face_attributes";
const std::string attribute_type<OptionalFaceHandle>::attr_group    = "face_attributes";
const std::string attribute_type<ClusterHandle>::attr_group         = "cluster_attributes";
const std::string attribute_type<OptionalClusterHandle>::attr_group = "cluster_attributes";

bool AttributeMeshIOBase::addMesh(const HalfEdgeMesh<BaseVec>& hem)
{
  FloatChannel vertices(hem.numVertices(), 3);
  IndexChannel indices(hem.numFaces(), 3);

  DenseVertexMap<size_t > new_indices;
  new_indices.reserve(hem.numVertices());

  size_t k = 0;
  for(size_t i=0; i<hem.nextVertexIndex(); i++)
  {
    VertexHandle vH(i);
    if(!hem.containsVertex(vH)) continue;
    new_indices.insert(vH, k);
    vertices[k++] = hem.getVertexPosition(vH);
  }

  Index i = 0;
  for(auto fH : hem.faces())
  {
    auto vHs = hem.getVerticesOfFace(fH);
    indices[i][0] = new_indices[vHs[0]];
    indices[i][1] = new_indices[vHs[1]];
    indices[i][2] = new_indices[vHs[2]];
    i++;
  }
  return addVertices(vertices) && addIndices(indices);
}

boost::optional<HalfEdgeMesh<BaseVec>> AttributeMeshIOBase::getMesh()
{
  auto vertices_opt = getVertices();
  auto indices_opt = getIndices();

  if(vertices_opt && indices_opt)
  {
    auto& vertices = vertices_opt.get();
    auto& indices = indices_opt.get();

    HalfEdgeMesh<BaseVec> hem;
    for (size_t i = 0; i < vertices.numElements(); i++)
    {
      hem.addVertex(vertices[i]);
    }

    for (size_t i = 0; i < indices.numElements(); i++)
    {
      const std::array<VertexHandle, 3>& face = indices[i];
      hem.addFace(face[0], face[1], face[2]);
    }
    return hem;
  }
  return boost::none;
}

}

