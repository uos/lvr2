/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/*
 * CGALPolyhedronMesh.cpp
 *
 *  @date 15.03.2016
 *  @author Henning Strüber (hstruebe@uos.de)
 */

#include "CGALPolyhedronMesh.hpp"

namespace lvr2 {

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::CGALPolyhedronMesh() {}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::CGALPolyhedronMesh(PointsetSurfacePtr<VertexT> pm) {
    m_pointCloudManager = pm;
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::addVertex(VertexT v) {
  Point_3 p(v[0], v[1], v[2]);
  addVertexModifier m = addVertexModifier(p);
  P.delegate(m);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::addNormal(NormalT n) {
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::addTriangle(uint a,
                                                                      uint b,
                                                                      uint c) {
  addTriangleModifier m = addTriangleModifier(a, b, c);
  P.delegate(m);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::flipEdge(uint v1,
                                                                   uint v2) {}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::preProcessing() {
  m_numFaces = P.size_of_facets();
  for (Facet_iterator i = P.facets_begin(); i != P.facets_end(); ++i) {
    if (!i->is_triangle()) {
      BOOST_LOG_TRIVIAL(warning) << "Found non-triangle.";
      m_numFaces--;
    }
  }
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::translate(
    Point_3 &p, Vector_3 vec) {
  Aff_transformation_3 aff(CGAL::TRANSLATION, vec);
  p = p.transform(aff);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::removeVertex(
    Vertex_handle &v) {
  removeVertexModifier m = removeVertexModifier(v);
  P.delegate(m);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel,
                        Items>::simpleTriangulation() {
  BOOST_LOG_TRIVIAL(info) << "Triangulate faces.";
  CGAL::Polygon_mesh_processing::triangulate_faces(P);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
void CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::finalize() {

  preProcessing();
  BOOST_LOG_TRIVIAL(debug) << "Number of Points/Vertices: "
                           << P.size_of_vertices();
  BOOST_LOG_TRIVIAL(debug) << "Number of Faces: " << P.size_of_facets();

  size_t numVertices = P.size_of_vertices();

  // Buffer
  floatArr vertexBuffer(new float[3 * numVertices]);
  uintArr indexBuffer(new unsigned int[3 * m_numFaces]);

  std::map<Point_3, uint> m_vertexMap;
  int i = 0;
  for (auto v = P.vertices_begin(); v != P.vertices_end(); ++v) {
    // BOOST_LOG_TRIVIAL(trace) << "Add Vertex: " << v->point();
    vertexBuffer[3 * i] = v->point().x();
    vertexBuffer[3 * i + 1] = v->point().y();
    vertexBuffer[3 * i + 2] = v->point().z();
    m_vertexMap[v->point()] = i++;
  }
  BOOST_LOG_TRIVIAL(debug) << "Added " << i << " Vertices.";

  size_t k = 0;
  for (Facet_iterator i = P.facets_begin(); i != P.facets_end(); ++i) {
    while (i != P.facets_end() && !(i->is_triangle())) {
      BOOST_LOG_TRIVIAL(warning) << "Warning, no triangle.";
      ++i;
    }
    if (i == P.facets_end())
      break;
    Halfedge_around_facet_circulator j = i->facet_begin();
    int l = 0;
    do {
      indexBuffer[3 * k + l++] = m_vertexMap[j->vertex()->point()];
    } while (++j != i->facet_begin());
    k++;
  }

  // init mesh buffer if necessary
  if (!this->m_meshBuffer) {
    this->m_meshBuffer = MeshBufferPtr(new MeshBuffer);
  }
  // set buffer
  this->m_meshBuffer->setVertexArray(vertexBuffer, numVertices);
  this->m_meshBuffer->setFaceArray(indexBuffer, m_numFaces);
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
size_t CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::meshSize() {
  return P.size_of_vertices();
}

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
CGALPolyhedronMesh<VertexT, NormalT, Kernel, Items>::~CGALPolyhedronMesh() {}
} // namespace lvr2
