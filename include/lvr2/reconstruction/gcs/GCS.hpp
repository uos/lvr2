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
 * GCS.hpp
 *
 *  @date 14.04.16
 *  @author Henning Strueber (hstruebe@uos.de)
 */

#ifndef GCS_H_
#define GCS_H_

#define BOOST_LOG_DYN_LINK 1

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdlib.h>

// lvr related includes
#include <lvr2/io/DataStruct.hpp>

#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/CGALPolyhedronMesh.hpp>
#include <lvr2/geometry/ColorVertex.hpp>
#include <lvr2/geometry/Normal.hpp>

#include <CGAL/Point_3.h>
#include <CGAL/Random.h>
#include <CGAL/convex_hull_3.h>

#include <CGAL/Exact_predicates_exact_constructions_kernel.h>
#include <CGAL/HalfedgeDS_vertex_base.h>
#include <CGAL/Polyhedron_items_3.h>
#include <CGAL/Surface_mesh_deformation.h>
#include <CGAL/boost/graph/Euler_operations.h>
#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>

// The follwing is for Triangulated Surface Mesh Simplification
// Simplification function
#include <CGAL/Surface_mesh_simplification/edge_collapse.h>

// SMS
#include <CGAL/Polygon_mesh_processing/fair.h>
#include <CGAL/Polygon_mesh_processing/refine.h>
#include <CGAL/Surface_mesh_simplification/Edge_collapse_visitor_base.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Constrained_placement.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Count_stop_predicate.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_length_cost.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Edge_profile.h>
#include <CGAL/Surface_mesh_simplification/Policies/Edge_collapse/Midpoint_placement.h>
#include <CGAL/Unique_hash_map.h>
#include <CGAL/property_map.h>

// search
#include <CGAL/Orthogonal_k_neighbor_search.h>
#include <CGAL/Search_traits_3.h>
#include <CGAL/Search_traits_adapter.h>
#include <boost/iterator/zip_iterator.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/progress.hpp>

// filter chain
#include <lvr2/reconstruction/gcs/FilterChain.hpp>

namespace SMS = CGAL::Surface_mesh_simplification;

namespace lvr2 {
template <typename VertexT = ColorVertex<float, unsigned char>,
          typename NormalT = Normal<float>>
class GCS {
public:
  // forward declare GCSItems
  struct GCSItems;

  // typedef Kernel and Mesh
  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  // typedef CGAL::Simple_cartesian<double> Kernel;
  typedef CGALPolyhedronMesh<VertexT, NormalT, Kernel, GCSItems> PolyhedronMesh;

  // Refines HalfedgeDS_vertex_base and provides signal counter
  template <class Refs, class Point, class ID>
  struct GCSVertex
      : public CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID> {
  public:
    GCSVertex(const Point &p)
        : CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID>(p){};
    GCSVertex() : CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID>(){};

    double getSignalCounter() { return signal_counter; };

    double signal_counter = 0.0;
    size_t vId = -1;
    double latestDist = 0.0;
  };

  // Refines HalfedgeDS_vertex_base and provides signal counter
  template <class Refs, class ID, class Traits>
  struct GCSHalfedge
      : public CGAL::HalfedgeDS_halfedge_max_base_with_id<Refs, ID> {
  public:
    GCSHalfedge() : CGAL::HalfedgeDS_halfedge_max_base_with_id<Refs, ID>() {
      m_marked = true;
    };

    bool isMarked() { return m_marked; };

    void setDelete() { m_marked = false; };
    void setSave() { m_marked = true; };

  private:
    // indicates that edge is prohibited from collapsing
    bool m_marked = true;
  };

  // Refines Polyhedron_items_3 with own vertex
  struct GCSItems : public CGAL::Polyhedron_items_with_id_3 {
    template <class Refs, class Traits> struct Vertex_wrapper {
      typedef typename CGAL::Polyhedron_traits_3<Kernel>::Point_3 Point;
      typedef GCSVertex<Refs, Point, std::size_t> Vertex;
    };
    template <class Refs, class Traits> struct Halfedge_wrapper {
      typedef GCSHalfedge<Refs, size_t, Traits> Halfedge;
    };
  };

  GCS(PointBufferPtr pBuffer, std::string config);

  virtual ~GCS();

  virtual void getMesh(PolyhedronMesh &mesh);

protected:
  // type of Polyhedron_3 member of mesh
  typedef typename PolyhedronMesh::Polyhedron_3 Polyhedron_3;

  // graph_traits
  typedef typename boost::graph_traits<Polyhedron_3>::edge_descriptor
      b_edge_descriptor;
  typedef typename boost::graph_traits<Polyhedron_3>::vertex_descriptor
      b_vertex_descriptor;
  typedef typename boost::graph_traits<Polyhedron_3>::halfedge_descriptor
      b_halfedge_descriptor;
  typedef typename boost::graph_traits<Polyhedron_3>::vertices_size_type
      b_vertices_size_type;

  // Handles
  typedef typename PolyhedronMesh::Vertex_handle Vertex_handle;
  typedef typename PolyhedronMesh::Halfedge_handle Halfedge_handle;
  typedef typename PolyhedronMesh::Facet_handle Facet_handle;
  typedef typename PolyhedronMesh::Point_3 Point_3;
  typedef typename PolyhedronMesh::Vector_3 Vector_3;

  // K_neighbor_search
  typedef typename boost::property_map<Polyhedron_3, CGAL::vertex_point_t>::type
      Vertex_point_pmap;
  typedef typename CGAL::Search_traits_3<Kernel> Traits_base;
  typedef typename CGAL::Search_traits_adapter<Vertex_handle, Vertex_point_pmap,
                                               Traits_base>
      Traits;
  typedef typename CGAL::Orthogonal_k_neighbor_search<Traits> K_neighbor_search;
  typedef typename K_neighbor_search::Tree Tree;
  typedef typename Tree::Splitter Splitter;
  typedef typename K_neighbor_search::Distance Distance;

  // filterchain
  typedef FilterChain<PolyhedronMesh> Filter;

  // boost time measure
  typedef boost::posix_time::ptime Time;
  typedef boost::posix_time::time_duration TimeDuration;

  //
  // BGL property map which indicates whether an edge is marked as non-removable
  //
  struct Collapse_constrained_edge_map {

    const Polyhedron_3 *meshPtr;

    typedef b_edge_descriptor key_type;
    typedef bool value_type;
    typedef value_type reference;
    typedef boost::readable_property_map_tag category;

    Collapse_constrained_edge_map(Polyhedron_3 &mesh) : meshPtr(&mesh) {}

    friend bool get(Collapse_constrained_edge_map m, const key_type &edge) {
      Halfedge_handle he = CGAL::halfedge(edge, *m.meshPtr);
      return (!he->isMarked() || !he->opposite()->isMarked());
    }
  };

  typedef
      typename SMS::Constrained_placement<SMS::Midpoint_placement<Polyhedron_3>,
                                          Collapse_constrained_edge_map>
          Placement;

  struct GCSEdgeVisitior
      : public SMS::Edge_collapse_visitor_base<Polyhedron_3> {
    typedef SMS::Edge_profile<Polyhedron_3> Profile;
    typedef boost::graph_traits<Polyhedron_3> GraphTraits;
    typedef typename GraphTraits::vertex_descriptor vertex_descriptor;
    typedef typename Polyhedron_3::Halfedge_around_vertex_circulator
        Halfedge_around_vertex_circulator;
    void OnCollapsed(Profile const &p, vertex_descriptor const &v) {
      Halfedge_around_vertex_circulator havc = v->vertex_begin();
      do {
        havc->setSave();
        havc->opposite()->setSave();
      } while (++havc != v->vertex_begin());
    }
  };

  inline coord<float> point3ToCoord(Point_3 &p) {
    coord<float> point;
    point.x = p.x();
    point.y = p.y();
    point.z = p.z();

    return point;
  };

  friend std::ostream &operator<<(std::ostream &o, const Vertex_handle &v) {
    o << "Vertex: " << v->id();
    return o;
  }

  friend std::ostream &operator<<(std::ostream &o, const Halfedge_handle &h) {
    o << h->opposite()->vertex() << "("
      << h->opposite()->vertex()->vertex_degree() << ")"
      << "-->" << h->vertex() << "(" << h->vertex()->vertex_degree() << ") "
      << h->isMarked();
    return o;
  }

  struct compare_heBySourceDegreeDesc {
    bool operator()(const Halfedge_handle &a, const Halfedge_handle &b) {
      return (a->vertex()->vertex_degree() >= b->vertex()->vertex_degree());
    }
  };

  struct compare_heBySourceDegreeAsc {
    bool operator()(const Halfedge_handle &a, const Halfedge_handle &b) {
      return (a->vertex()->vertex_degree() <= b->vertex()->vertex_degree());
    }
  };

  // Data
  coord3fArr m_pointCoord; // aka boost::shared_array< coord<float> > from
                           // DataStruct.hpp
  std::vector<Point_3> m_point3Vec; // contains raw point data as Point_3 DT
  size_t m_pointNumber;             // number of points
  size_t m_vertexIndex = 0;
  CGAL::Random m_random;

  typename Filter::VertexContainer m_vertexContainer;
  typename Filter::FilterContainer m_filterContainer;
  FilterChain<PolyhedronMesh> *m_filterChain;

  // search
  Tree *tree;
  Distance *tr_dist;
  Vertex_point_pmap vppmap;

  // Counter
  uint m_iterationCounter;
  uint m_basicStepCounter;
  uint m_vertexSplitCounter;

  // Params
  int m_smoothing;
  float m_learningRate;
  float m_neighborLearningRate;
  float m_decreasingFactor;
  float m_collapsThreshold;
  float m_boxFactor;
  uint m_runtime;
  uint m_allowedMiss;
  uint m_basicSteps;
  uint m_numSplits;
  bool m_withCollaps;
  bool m_showPoints;
  bool m_withFilter;
  double m_deleteLongHE;

  TimeDuration d_runtime;
  double d_signalCounterMean;
  double d_avgHELength;
  uint d_longEdgesDeleted;
  double d_vertexDistance;

  ofstream datafile;

  // internal mesh
  PolyhedronMesh *m_mesh;

  /*
   * @brief Performs basic step as described in hannuth14
   */
  void performBasicStep();

  /*
   * @brief Performs vertex split alongside v with highest sc
   */
  void performVertexSplit();

  /*
   * @brief Performs edge collapsing
   */
  void performEdgeCollaps();

  /*
   * @brief Returns a random point coord from m_pointCoord
   */
  coord<float> randomPointCoord();

  /*
   *
  */
  void updateSignalCounter(Vertex_handle &winner);

  /*
   * @brief Generates initial mesh
   */
  void initSimpleMesh();

  /*
   * @brief Generates initial tetrahedron
   */
  void initTetrahedron();

  /*
   * @brief Move Vertex v in direction p
   *
   * @param v Point reference from vertex which should be moved
   * @param p Point from pointcloud that denotes direction of the translation
  */
  void moveVertex(Vertex_handle &v, coord<float> p);

  /*
   * @brief Move Neighbor v in direction p
   *
   * @param v Neighbor
   * @param p Direction
   */
  void moveNeighbor(Vertex_handle &v, coord<float> p);

  /*
   * @brief Perform laplacian smoothing
   *
   * @param source Source vertex
   */
  void laplacianSmoothing(Vertex_handle &source);

  /*
   * @brief Perform neighbor smoothing, i.e. move all neighbors
   * according to neighbor learning rate towards target
   */
  void neighborSmoothing(Vertex_handle &source, coord<float> targetPoint);

  /*
   * @brief Get the nearest vertex to point
   *
   * @param point Reference point
   *
   * @return Nearest vertex to point
   */
  auto nearestVertex(Point_3 p) -> Vertex_handle;

  /*
   * @brief Return vertex with highest signal counter
   *
   * @return Winning vertex
   */
  auto vertexWithHighestSC() -> Vertex_handle;

  /**
   * @brief print info
   */
  void printInfo();

  /**
   * @brief Calculate Signal counter mean
   */
  void calcSCMean();

}; // class GCS
}; // namespace lvr2

#include "GCS.cpp"

#endif /* GCS_H_ */
