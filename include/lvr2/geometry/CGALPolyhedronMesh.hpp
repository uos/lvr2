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
 * CGALPolyhedronMesh.hpp
 *
 *  @date 15.03.2016
 *  @author Henning Strüber (hstruebe@uos.de)
 */

#ifndef CGALPOLYHEDRONMESH_H_
#define CGALPOLYHEDRONMESH_H_

#include "BaseMesh.hpp"

#include <iostream>

#include <lvr2/reconstruction/PointsetSurface.hpp>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/triangulate_faces.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Polyhedron_incremental_builder_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/Vector_3.h>
#include <CGAL/aff_transformation_tags.h>

#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

#include <CGAL/boost/graph/properties.h>
#include <boost/graph/graph_traits.hpp>

namespace lvr2 {

template <typename VertexT, typename NormalT,
          typename Kernel = CGAL::Exact_predicates_inexact_constructions_kernel,
          typename Items = CGAL::Polyhedron_items_with_id_3>
class CGALPolyhedronMesh : public BaseMesh<VertexT> {
public:
  CGALPolyhedronMesh();

  CGALPolyhedronMesh(typename PointsetSurface<VertexT>::Ptr pm);

  virtual void addVertex(VertexT v);

  virtual void addNormal(NormalT n);

  virtual void addTriangle(uint a, uint b, uint c);

  virtual void flipEdge(uint v1, uint v2);

  virtual void finalize();

  virtual size_t meshSize();

  virtual ~CGALPolyhedronMesh();

  /* CGAL related typedefs */
  typedef CGAL::Polyhedron_3<Kernel, Items> Polyhedron_3;

  typedef typename Kernel::FT FT;

  // hds
  typedef typename Polyhedron_3::HalfedgeDS HalfedgeDS;

  // Handles to items
  typedef typename Polyhedron_3::Halfedge_handle Halfedge_handle;
  typedef typename Polyhedron_3::Vertex_handle Vertex_handle;
  typedef typename Polyhedron_3::Facet_handle Facet_handle;
  typedef typename Polyhedron_3::Point_3 Point_3;

  // Model of HalfedgeDS[Item] concept
  typedef typename Polyhedron_3::Halfedge HalfEdge;
  typedef typename Polyhedron_3::Vertex Vertex;
  typedef typename Polyhedron_3::Facet Facet;

  // Iterator and circulator
  typedef typename Polyhedron_3::Halfedge_iterator Halfedge_iterator;
  typedef typename Polyhedron_3::Vertex_iterator Vertex_iterator;
  typedef typename Polyhedron_3::Facet_iterator Facet_iterator;

  typedef typename Polyhedron_3::Halfedge_around_facet_circulator
      Halfedge_around_facet_circulator;
  typedef typename Polyhedron_3::Halfedge_around_vertex_circulator
      Halfedge_around_vertex_circulator;

  typedef typename Kernel::Vector_3 Vector_3;
  typedef typename Kernel::Aff_transformation_3 Aff_transformation_3;

  // Simplify getting iterators
  Facet_iterator facets_end() { return P.facets_end(); };
  Facet_iterator facets_begin() { return P.facets_begin(); };

  Halfedge_iterator halfedges_begin() { return P.halfedges_begin(); };
  Halfedge_iterator halfedges_end() { return P.halfedges_end(); };

  Vertex_iterator vertices_begin() { return P.vertices_begin(); };
  Vertex_iterator vertices_end() { return P.vertices_end(); };

  static Facet_handle null_face() {
    return boost::graph_traits<Polyhedron_3>::null_face();
  };

  size_t size() { return P.size_of_vertices(); };
  size_t size_of_facets() { return P.size_of_facets(); };
  size_t size_of_halfedges() { return P.size_of_halfedges(); };
  size_t size_of_vertices() { return P.size_of_vertices(); };

  void translate(Point_3 &p, Vector_3 vec);

  void removeVertex(Vertex_handle &v);

  Polyhedron_3 &getPolyhedron() { return P; };

  const Polyhedron_3 &getPolyhedron() const { return P; };

  void simpleTriangulation();

protected:
  void preProcessing();

  struct addTriangleModifier : public CGAL::Modifier_base<HalfedgeDS> {
    uint a, b, c;
    addTriangleModifier(uint a, uint b, uint c) {
      this->a = a;
      this->b = b;
      this->c = c;
    };
    void operator()(HalfedgeDS &hds) {
      CGAL::Polyhedron_incremental_builder_3<HalfedgeDS> B(hds, true);

      B.begin_surface(0, 1, 0, CGAL::Polyhedron_incremental_builder_3<
                                   HalfedgeDS>::ABSOLUTE_INDEXING);
      auto f = B.begin_facet();
      B.add_vertex_to_facet(a);
      B.add_vertex_to_facet(b);
      B.add_vertex_to_facet(c);
      B.end_facet();
      B.end_surface();
    }
  };

  struct addVertexModifier : public CGAL::Modifier_base<HalfedgeDS> {
    // typename HalfedgeDS::Vertex::Point point;
    Point_3 point;
    addVertexModifier(Point_3 &p) { point = p; };
    void operator()(HalfedgeDS &hds) {
      CGAL::Polyhedron_incremental_builder_3<HalfedgeDS> B(hds, true);
      B.begin_surface(1, 0, 0, CGAL::Polyhedron_incremental_builder_3<
                                   HalfedgeDS>::ABSOLUTE_INDEXING);
      B.add_vertex(point);
      B.end_surface();
    }
  };

  struct removeVertexModifier : public CGAL::Modifier_base<HalfedgeDS> {
    Vertex_handle v;
    removeVertexModifier(Vertex_handle &v) : v(v){};
    void operator()(HalfedgeDS &hds) { hds.vertices_erase(v); }
  };
  /* Internal variables*/
  size_t m_numFaces;
  typename PointsetSurface<VertexT>::Ptr m_pointCloudManager;

private:
  Polyhedron_3 P;

}; // class CGALPolyhedronMesh

template <typename VertexT, typename NormalT, typename Kernel, typename Items>
std::ostream &operator<<(std::ostream &os,
                         typename CGALPolyhedronMesh<VertexT, NormalT, Kernel,
                                                     Items>::Vertex_handle &v) {
  os << "Vertex";
  return os;
}

} // namespace lvr2
#include "CGALPolyhedronMesh.cpp"

#endif /* CGALPOLYHEDRONMESH_H_ */
