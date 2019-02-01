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
 * Filter.hpp
 *
 *  @author Henning Strueber (hstruebe@uos.de)
 *
 *  Idea and Documentation from H. Anuth (2014)
 */

#ifndef FILTERCHAIN_HPP_
#define FILTERCHAIN_HPP_

#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>

#include <iostream>

// PMP
#include <CGAL/Polygon_mesh_processing/triangulate_hole.h>

namespace PMP = CGAL::Polygon_mesh_processing;

namespace lvr2 {
/**
 * FilterChain
 */
template <typename MeshClass> class FilterChain {
public:
  typedef typename MeshClass::Polyhedron_3 Polyhedron_3;
  typedef typename MeshClass::Vertex_handle Vertex_handle;
  typedef typename MeshClass::Facet_handle Facet_handle;
  typedef typename MeshClass::Halfedge_handle Halfedge_handle;

  typedef typename Polyhedron_3::Traits::Kernel Kernel;

  typedef typename MeshClass::Halfedge_around_vertex_circulator
      Halfedge_around_vertex_circulator;
  typedef typename MeshClass::Halfedge_around_facet_circulator
      Halfedge_around_facet_circulator;

  typedef struct FilterBase {
    virtual bool operator()(MeshClass &m, Vertex_handle &v) = 0;
  } FilterBase;

  typedef std::vector<FilterBase *> FilterContainer;
  typedef std::vector<Vertex_handle> VertexContainer;

  FilterContainer m_filterContainer;

  VertexContainer *m_vertexContainer;
  MeshClass *m_mesh;

  FilterChain(FilterContainer fc) : m_filterContainer(fc){};
  ~FilterChain() {
    for (FilterBase *fb : m_filterContainer) {
      delete fb;
    }
  }
  /**
   * @brief Invokes filter on vertex
   * @param  fb Reference of model of @ref FilterBase
   * @param  m  Reference of @ref MeshClass
   * @param  v  Reference of @ref Vertex_handle
   * @return    true iff filter is applied
   */
  inline static bool invokeFilter(FilterBase &fb, MeshClass &m,
                                  Vertex_handle &v) {
    return fb(m, v);
  }

  /**
   * Runs all filters of @ref m_filterContainer on all vertices in @ref
   * m_vertexContainer
   * @return true iff all vertices have passed all filters
   */
  inline bool execute() {
    for (Vertex_handle v : *m_vertexContainer) {
      bool applied = false;
      for (FilterBase *fb : m_filterContainer) {
        applied = FilterChain::invokeFilter(*fb, *m_mesh, v);
        if (applied) {
          break;
        } else {
        }
      }
    }
    m_vertexContainer->clear();
  }

  /**
   * Sets the @ref m_vertexContainer
   * @param vc The actually valid container container
   */
  inline void setVertexContainer(VertexContainer &vc) {
    m_vertexContainer = &vc;
  };

  /**
   * Sets the @ref m_mesh;
   * @param m The actually valid mesh
   */
  inline void setMesh(MeshClass &m) { m_mesh = &m; };

  /************************** Artifact Filters *******************************/

  /**
   * If a vertex is not connected to any triangle it can be removed
   */
  typedef struct EdgelessVertex : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      if (v->vertex_degree() == 0) {
        BOOST_LOG_TRIVIAL(info) << "EdgelessVertex: " << v->point();
        CGAL::remove_vertex(v, m.getPolyhedron()); // CGAL and the boost gl
        return true;
      }
      return false;
    };
  } EdgelessVertex;

  /**
   * An edge without connection to any triangle is an artifact in a triangle
   * based surface representation and can be removed
   */
  typedef struct EdgeWithoutTriangles : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      Halfedge_around_vertex_circulator havc = v->vertex_begin();
      do {
        if (havc->is_border() && havc->opposite()->is_border()) {
          BOOST_LOG_TRIVIAL(info) << "Borderedge found.";

          //
          // T1---he4---V2---he1---T2
          //            Λ|
          //    havc.opp|| havc
          //            ||
          //            |V
          // T3---he2---V1---he3---T4
          //
          // Since havc and its opposite are border edges and therefore not
          // connected to any facet no reference updates has to be done.
          // References of Halfedges and Vertices to havc which is to be
          // deleted must be updated. So next() of he2 is havc.opp at the
          // moment but after its deletion there would be a segfault. Therefore
          // we set next() of he2 to he3 and next() of he1 to he4.
          // Also V1 and V2 are reference a HE which directs towards them.
          // Those references has to be changed to h2 and he1.

          Vertex_handle v1 = havc->vertex();
          Vertex_handle v2 = havc->opposite()->vertex();

          Halfedge_handle he1 = havc->prev();
          Halfedge_handle he2 = havc->opposite()->prev();

          Halfedge_handle he3 = havc->next();
          Halfedge_handle he4 = havc->opposite()->next();

          v1->set_halfedge(he2);
          v2->set_halfedge(he1);

          CGAL::set_next(he2, he3, m.getPolyhedron());
          CGAL::set_next(he1, he4, m.getPolyhedron());

          CGAL::remove_edge(CGAL::edge(havc, m.getPolyhedron()),
                            m.getPolyhedron());
          return true;
        }
      } while (++havc != v->vertex_begin());
      return false;
    };
  } EdgeWithoutTriangles;

  /**
   * A vertex connected to more than two boundaries isn't valid in most
   * reconstruction algorithms and can therefore be removed along with all
   * triangles and edges of the vertex, except those belonging to the
   * "biggest"
   * wedge.
   */
  typedef struct MultipleBoundaryRemove : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      BOOST_LOG_TRIVIAL(warning)
          << "Filter MultipleBoundaryRemove is not yet implemented.";
    }
  } MultipleBoundaryRemove;

  /************************** Constructing Filters **************************/

  /**
   * As in \ref MultipleBoundaryRemove a multiple boundary vertex v can also
   * be repaired constructively by adding new triangles until v is connected to
   * only two boundaries.
   */
  typedef struct MultipleBoundaryAdd : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      BOOST_LOG_TRIVIAL(warning)
          << "Filter MultipleBoundaryAdd is not yet implemented.";
    }
  } MultipleBoundaryAdd;

  /**
   * Holes between three to five vertices are rarely desired and therefore
   * closed by this filter.
   */
  typedef struct HoleFilling : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      if (CGAL::is_border(v, m.getPolyhedron())) {
        BOOST_LOG_TRIVIAL(debug) << "FilterChain: HoleFilling";

        Halfedge_around_vertex_circulator havc = v->vertex_begin();
        int holeSize = 0;
        do {
          if (havc->is_border()) {
            holeSize = 0;
            // walk the line
            Halfedge_handle line = havc->next();
            do {
              ++holeSize;
              line = line->next();
            } while (line != havc &&
                     (CGAL::is_border(line->vertex(), m.getPolyhedron())));
            if (holeSize < 6) {

              std::vector<Facet_handle> patch_facets;
              std::vector<Vertex_handle> patch_vertices;
              PMP::triangulate_and_refine_hole(
                  m.getPolyhedron(), havc, std::back_inserter(patch_facets),
                  std::back_inserter(patch_vertices),
                  PMP::parameters::vertex_point_map(
                      get(CGAL::vertex_point, m.getPolyhedron()))
                      .geom_traits(Kernel()));
              BOOST_LOG_TRIVIAL(info)
                  << "FilterChain successfully closed hole.";
            }
          } // if havc is border
        } while (++havc != v->vertex_begin());
      } // iff v is border vertex

      return false;
    }
  } HoleFilling;

  /************************** Remove Filters ********************************/

  /**
   * A vertex with valence two can lead to bad results and is generally
   * undesired and can therefore be deleted.
   */
  typedef struct ValenceTwoRemove : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      if (v->vertex_degree() == 2) {
        BOOST_LOG_TRIVIAL(info) << "Remove valence two vertex. "
                                << v->vertex_degree();
        // check if border edge choose other then
        Halfedge_around_vertex_circulator havc = v->vertex_begin();
        int deleted = 0;
        do {
          if (!havc->is_border()) {
            ++deleted;
            m.getPolyhedron().erase_facet(havc);
          }
        } while (++havc != v->vertex_begin());
        BOOST_LOG_TRIVIAL(info) << deleted << " Facets removed.";
        if (deleted != 0) {
          return true;
        }
      }
      return false;
    };
  } ValenceTwoRemove;

  /**
   * A vertex of high valence can indicate misplaced surfaces and it can
   * therefore be cut out.
   */
  typedef struct HighValenceRemove : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      if (v->vertex_degree() > 10) {
        BOOST_LOG_TRIVIAL(warning) << "Try to remove high valence vertex.";
        auto he = v->vertex_begin();
        do {
          // TODO is this really necessary?
          if (he->is_border_edge()) {
            BOOST_LOG_TRIVIAL(warning) << "Skip remove high valence vertex, "
                                          "because adjecent to a hole.";
            return false;
          }
        } while (++he != v->vertex_begin());

        Halfedge_handle h =
            m.getPolyhedron().erase_center_vertex(v->halfedge()->opposite());
        return true;
      }
      return false;
    }
  } HighValenceRemove;

  /**
   * Removes the "bridge" of an edge which is connected to another surface.
   */
  typedef struct BridgeRepair : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      std::cout << "BridgeRepair" << std::endl;
    }
  } BridgeRepair;

  /**
   * Crumbs are clusters of varying size that are not connected to another
   * surface.
   */
  typedef struct CrumbRemove : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      std::cout << "CrumbRemove" << std::endl;
    }
  } CrumbRemove;

  /***************************Editing Filters ******************************/

  /**
   * Searches for possible edge swap operations to optimize valence.
   */
  typedef struct OptimizeValence : FilterBase {
    bool operator()(MeshClass &m, Vertex_handle &v) {
      BOOST_LOG_TRIVIAL(info) << "OptimizeValence filter not yet implemented.";
    }
  } OptimizeValence;
};
}; // namespace lvr2
#endif
