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
 * GrowingCellStructures.hpp
 *
 *  @author Henning Strueber (hstruebe@uos.de)
 *
 * Implementation for algorithm described in H. Anuth (2014)
 */

#ifndef GrowingSurfaceStructure_HPP_
#define GrowingSurfaceStructure_HPP_

#include <iostream>
#include <limits>
#include <typeinfo>

#include <CGAL/Bbox_3.h>
#include <CGAL/Polyhedron_items_with_id_3.h>
#include <CGAL/centroid.h>

#include <CGAL/Polygon_mesh_processing/compute_normal.h>
#include <CGAL/Polygon_mesh_processing/stitch_borders.h>

#include <CGAL/boost/graph/graph_traits_Polyhedron_3.h>

// Face distance calculation
#include <CGAL/AABB_face_graph_triangle_primitive.h>
#include <CGAL/AABB_traits.h>
#include <CGAL/AABB_tree.h>

// required for face_area()
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Polygon_mesh_processing/measure.h>
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

// filter chain
#include <lvr2/reconstruction/gcs/FilterChain.hpp>

// boost rolling mean
#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/rolling_mean.hpp>
#include <boost/accumulators/statistics/rolling_window.hpp>
#include <boost/accumulators/statistics/stats.hpp>

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/progress.hpp>

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

namespace SMS = CGAL::Surface_mesh_simplification;
namespace PMP = CGAL::Polygon_mesh_processing;

namespace lvr2 {

class GrowingSurfaceStructure {
public:
  struct GrowingItems;

  // typedef of Kernel and Mesh with our items
  typedef CGAL::Exact_predicates_inexact_constructions_kernel Kernel;
  typedef CGALPolyhedronMesh<ColorVertex<float, unsigned char>, Normal<float>,
                             Kernel, GrowingItems>
      PolyhedronMesh;

  // Refine HalfedgeDS_vertex_max_base_with_id
  template <class Refs, class Point, class ID>
  struct GrowingVertex
      : public CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID> {
    GrowingVertex(const Point &p)
        : CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID>(p){};
    GrowingVertex()
        : CGAL::HalfedgeDS_vertex_max_base_with_id<Refs, Point, ID>(){};

    uint numBorderEdges() {
      PolyhedronMesh::Halfedge_around_vertex_circulator havc =
          this->halfedge()->vertex_begin();
      uint num = 0;

      do {
        havc->is_border() ? ++num : false;
      } while (++havc != this->halfedge()->vertex_begin());
      return num;
    }

    double avgSurrEdgeLength() {
      PolyhedronMesh::Halfedge_around_vertex_circulator havc =
          this->halfedge()->vertex_begin();
      double length = 0.0;
      uint numSurrEdges = 0;

      do {
        length += havc->length();
        ++numSurrEdges;
      } while (++havc != this->halfedge()->vertex_begin());
      return length /= numSurrEdges;
    }
  };

  // Refine HalfedgeDS_face_max_base_with_id
  template <class Refs, class Plane, class ID, class Traits>
  struct GrowingFace
      : public CGAL::HalfedgeDS_face_max_base_with_id<Refs, Plane, ID> {
    typedef typename Traits::Point_3 Point_3;
    typedef typename Traits::FT FT;
    typedef typename Traits::Triangle_3 Triangle_3;

    GrowingFace(const Plane &pln)
        : CGAL::HalfedgeDS_face_max_base_with_id<Refs, Plane, ID>(pln){};
    GrowingFace(const Plane &pln, ID i)
        : CGAL::HalfedgeDS_face_max_base_with_id<Refs, Plane, ID>(pln, i){};
    GrowingFace() : CGAL::HalfedgeDS_face_max_base_with_id<Refs, Plane, ID>(){};

    Point_3 point() {
      if (this->halfedge()->facet() == PolyhedronMesh::null_face()) {
        BOOST_LOG_TRIVIAL(error)
            << "GrowingFace: Tried to get point from null_face";
        return Point_3(0.0, 0.0, 0.0);
      }
      Halfedge_around_facet_circulator hafc = this->halfedge()->facet_begin();
      std::size_t degree = this->halfedge()->facet_degree();
      FT x = 0.0;
      FT y = 0.0;
      FT z = 0.0;
      do {
        Point_3 p = hafc->vertex()->point();
        x += p.x();
        y += p.y();
        z += p.z();
      } while (++hafc != this->halfedge()->facet_begin());
      return Point_3(x / degree, y / degree, z / degree);
    }

    FT face_area() {
      if (this->halfedge()->facet() == PolyhedronMesh::null_face()) {
        BOOST_LOG_TRIVIAL(error) << "GrowingFace: Cannot calculate face area "
                                    "of null face.";
        return 0.0;
      }
      if (!this->halfedge()->is_triangle()) {
        BOOST_LOG_TRIVIAL(error) << "GrowingFace: Cannot calculate face area "
                                    "of non-triangle.";
        return 0.0; // TODO
      }

      Point_3 a = this->halfedge()->vertex()->point();
      Point_3 b = this->halfedge()->next()->vertex()->point();
      Point_3 c = this->halfedge()->next()->next()->vertex()->point();
      Triangle_3 t(a, b, c);
      FT sqa = t.squared_area();
      return std::sqrt(sqa);
    }

    FT getError() { return m_error; }
    void setError(FT e) { m_error = e; }

    FT getAge() { return m_age; }
    void setAge(FT a) { m_age = a; }

  private:
    double m_error = 0.0;
    double m_age = 0.0;
  };

  // Refine HalfedgeDS_halfedge_max_base_with_id
  template <class Refs, class ID, class Traits>
  struct GrowingHalfedge
      : public CGAL::HalfedgeDS_halfedge_max_base_with_id<Refs, ID> {
    typedef typename Traits::Point_3 Point_3;
    typedef typename Traits::FT FT;
    GrowingHalfedge() : CGAL::HalfedgeDS_halfedge_max_base_with_id<Refs, ID>() {
      m_marked = true;
    };

    Point_3 point() {
      Point_3 p1 = this->vertex()->point();
      Point_3 p2 = this->opposite()->vertex()->point();
      return Point_3((p1.x() + p2.x()) / 2.0, (p1.y() + p2.y()) / 2.0,
                     (p1.z() + p2.z()) / 2.0);
    }
    FT length() {
      Point_3 p1 = this->vertex()->point();
      Point_3 p2 = this->opposite()->vertex()->point();
      return std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                       std::pow((p1.y() - p2.y()), 2) +
                       std::pow(p1.z() - p2.z(), 2));
    }
    bool isMarked() { return m_marked; };

    void setDelete() { m_marked = false; };
    void setSave() { m_marked = true; };

  private:
    bool m_marked = true;
  };

  // Refines Polyhedron_items_with_id_3 with own items
  struct GrowingItems : public CGAL::Polyhedron_items_with_id_3 {
    template <class Refs, class Traits> struct Vertex_wrapper {
      typedef typename CGAL::Polyhedron_traits_3<Kernel>::Point_3 Point;
      typedef GrowingVertex<Refs, Point, size_t> Vertex;
    };
    template <class Refs, class Traits> struct Face_wrapper {
      typedef GrowingFace<Refs, Kernel::Plane_3, size_t, Traits> Face;
    };
    template <class Refs, class Traits> struct Halfedge_wrapper {
      typedef GrowingHalfedge<Refs, size_t, Traits> Halfedge;
    };
  };

  GrowingSurfaceStructure();
  GrowingSurfaceStructure(PointBufferPtr pBuffer, std::string config);

  virtual void getMesh(PolyhedronMesh &mesh);

  virtual ~GrowingSurfaceStructure();

  typedef PolyhedronMesh::Polyhedron_3 Polyhedron_3;

  typedef Kernel::Plane_3 Plane_3;

  typedef PolyhedronMesh::Vertex_handle Vertex_handle;
  typedef PolyhedronMesh::Facet_handle Facet_handle;
  typedef PolyhedronMesh::Halfedge_handle Halfedge_handle;

  typedef PolyhedronMesh::Halfedge_around_vertex_circulator
      Halfedge_around_vertex_circulator;
  typedef PolyhedronMesh::Halfedge_around_facet_circulator
      Halfedge_around_facet_circulator;

  typedef PolyhedronMesh::Point_3 Point_3;
  typedef PolyhedronMesh::Vector_3 Vector_3;

  typedef CGAL::Segment_3<Kernel> Segment_3;

  typedef Point_3::FT FT;

  typedef FilterChain<PolyhedronMesh> Chain;

  // graph_traits
  typedef typename boost::graph_traits<Polyhedron_3>::edge_descriptor
      b_edge_descriptor;
  typedef typename boost::graph_traits<Polyhedron_3>::vertex_descriptor
      b_vertex_descriptor;
  typedef typename boost::graph_traits<Polyhedron_3>::halfedge_descriptor
      b_halfedge_descriptor;

  // search for nearest vertex to p related typdefs
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

  // boost time measure
  typedef boost::posix_time::ptime Time;
  typedef boost::posix_time::time_duration TimeDuration;

protected:
  friend std::ostream &operator<<(std::ostream &o, const Vertex_handle &v) {
    o << "Vertex: " << v->id() << " Point: " << v->point();
    return o;
  }
  friend std::ostream &operator<<(std::ostream &o, const Halfedge_handle &h) {
    o << h->opposite()->vertex() << "->" << h->vertex();
    return o;
  }
  friend std::ostream &operator<<(std::ostream &o, const Facet_handle &f) {
    Halfedge_around_facet_circulator hafc = f->facet_begin();
    o << "";
    do {
      o << hafc->vertex();
    } while (++hafc != f->facet_begin());

    return o;
  }
  template <class T> bool contains(const std::vector<T> &vec, const T &value) {
    return std::find(vec.begin(), vec.end(), value) != vec.end();
  }

  struct compare_heBySourceDegreeDesc {
    bool operator()(const Halfedge_handle &a, const Halfedge_handle &b) {
      return (a->vertex()->vertex_degree() >= b->vertex()->vertex_degree());
    }
  };

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
      // returns true if edge is kept in mesh
      return (!he->isMarked() || !he->opposite()->isMarked());
    }
  };

  typedef
      typename SMS::Constrained_placement<SMS::Midpoint_placement<Polyhedron_3>,
                                          Collapse_constrained_edge_map>
          Placement;

  PolyhedronMesh *m_mesh;

  FloatChannel m_pointCoord;          // aka. boost::shared_array< coord<float> >
  std::vector<Point_3> m_point3Vec; // contains raw point data as Point_3 DT
  size_t m_pointNumber;             // number of points
  double m_k;                       // k = 1 / 1 - 0.5^(1/p_lambda)
  double m_avgTriangleSize;
  double m_beta;        // increaseRage for age of triangle =
  double m_avgDistance; // Áverage sample to surface distance
  double m_sDistance;   // Sum of distance to surface
  double m_avgTriangleErr;
  double m_avgSampleToSurface;
  // filterChain
  // container for vertices on which the filterchain should apply
  typename Chain::VertexContainer m_vertexContainer;
  // cointer for filter which should be applied
  typename Chain::FilterContainer m_filterContainer;
  Chain *m_filterChain;

  // counter
  uint p_runtime;
  uint p_numBasicSteps;
  uint p_numSplits;
  uint it_counter;

  // data collection
  uint d_facetsErased;
  uint d_edgeCollapsed;
  uint d_edgeSplitted;
  uint d_holesFilled;
  uint d_triangleHoleFilled;
  uint d_timesNonTriangleFound;
  double d_winningDistance;
  TimeDuration d_runtime;

  ofstream datafile;

  // parameter
  uint p_lambda;                 // half-life of a distance error
  double p_learningRate;         // learningRate for movement of vertices,
  double p_neighborLearningRate; //
  double p_limitSkip;   // Threshold for the relation of d_x to avgDistance that
                        // indicates if movement is skipped
  double p_limitSingle; // Threshold for the relation of d_x to avgDistance that
                        // indicates that only a Slightly improvement is needed
  double p_maxDeltaN;   // Threshold before edge collapse
  double p_maxAge;      // Maximum age for triangles
  double p_allowedMiss; // Allowedd misses before deletion threshold is reached
  bool p_printInfo;     // Print Infos at Exit
  double p_scale;       // Scale of inital Mesh
  uint p_holeSize;      // Hole Filling Threshold
  uint p_rollingMean;
  bool p_skipSTSCalculation; // Skip exact sample-to-surface distance
                             // calculation at info
  bool p_skipRemoveFacets;   // Skip removing facets
  bool p_skipRemove;         // Skip removing entirely
  double p_deleteLongHE;     // Delete long HE after termination
  bool p_useConvex; // Generate convex hull instead of simple initial mesh

  CGAL::Random m_random; // CGAL rng

  boost::accumulators::accumulator_set<
      double,
      boost::accumulators::stats<boost::accumulators::tag::rolling_mean>> *acc;

  // search related member
  Tree *tree;               // search tree
  Distance *tr_dist;        // aka. K_neighbor_search::Distance
  Vertex_point_pmap vppmap; // point map of points from vertices

  /**
   * Performs basic step
   */
  void performBasicStep();
  /**
   * Performs vertex split
   */
  void performVertexSplit();
  /**
   * perform remove step
   */
  void performRemove();

  /**
   * refresh search tree
   */
  void refreshSearchTree();

  /**
   * initialize tetrahedron as inital mesh
   */
  void initializeTetrahedron();

  /**
   * Select random sample p_x of Pointset
   * @return Random point p
   */
  Point_3 getRandomPoint();

  /**
   * coalescing step
   * @param v vertices on which coalescing should apply
   */
  void coalescing(std::vector<Vertex_handle> vertices);
  /**
   * continue basic step with vertex
   * @param v winning vertex
   */
  std::vector<Vertex_handle> furtherBasicStep(Vertex_handle &v, Point_3 p);

  /**
   * continue basic step with facet
   * @param v facet
   */
  std::vector<Vertex_handle> furtherBasicStep(Facet_handle &v, Point_3 p);

  /**
   * continue basic step with halfedge
   * @param v halfedge
   */
  std::vector<Vertex_handle> furtherBasicStep(Halfedge_handle &v, Point_3 p);

  /**
   * Checks if a face adjecent to v is nearer to sample p than wp
   * @param v reference vertex
   * @param wp so far winning point
   * @param p sample point
   *
   * @return std::pair<bool, Facet_handle> with first == true if face was
   * found
   * and face returned as second argument
   */
  std::pair<bool, Facet_handle> checkFaceNearer(Vertex_handle &v, Point_3 wp,
                                                Point_3 p);

  /**
   * Checks if a halfedge adjecent to v is nearer to sample p than wp
   * @param v reference vertex
   * @param wp so far winning point
   * @param p sample point
   *
   * @return std::pair<bool, Halfedge_handle> with first == true if he was
   * found and he returned as second argument
   */
  std::pair<bool, Halfedge_handle> checkEdgeNearer(Vertex_handle &v, Point_3 wp,
                                                   Point_3 p);
  /**
   * Update facet averages m_avgTriangleSize and m_avgTriangleErr
   */
  void updateFacetAverages();

  /**
   * Calculates eucl. distance between to points
   * @param  p1 Point 1
   * @param  p2 Point 2
   * @return    Distance between p1 and p2
   */
  FT distance(Point_3 p1, Point_3 p2);

  /**
   * Smooth neighbors of v
   * @param v winning vertex
   * @param p random point
   * @param exclude Exclude those vertices
   */
  void neighborSmoothing(Vertex_handle &v, Point_3 p,
                         std::vector<Vertex_handle> exclude);

  /**
   * Move neighbor vertex based on neighbor lr in direction p
   * @param v vertex
   * @param p point
   */
  void moveNeighbor(Vertex_handle &v, Point_3 p);

  /**
   * Increase error for facet
   * @param f facet
   */
  void increaseErrorAndSetDistance(Facet_handle &f, Point_3 p);
  /**
   * Set inital age of facet
   * @param f facet
   */
  void setInitialAge(Facet_handle &f);

  /**
   * increase age of facets
   * @param exclude facets whose age shall not be increased
   */
  void increaseAge(std::vector<Facet_handle> exclude);

  /**
   * move vertices based on learning rate
   * @param vertices vertices to move
   * @param p Point direction
   * @param sDistance distance between structure and point
   */
  void moveVertices(std::vector<Vertex_handle> vertices, Point_3 p,
                    double sDistance);

  /**
   * print info
   */
  void printInfo();
};
} // namespace lvr2
#include "GrowingSurfaceStructure.cpp"
#endif
