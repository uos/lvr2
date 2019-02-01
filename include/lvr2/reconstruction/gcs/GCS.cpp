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
 * GCS.tcc
 *
 *  @date 14.04.16
 *  @author Henning Strueber (hstruebe@uos.de)
 */
namespace logging = boost::log;
namespace lvr2 {
template <typename VertexT, typename NormalT>
GCS<VertexT, NormalT>::GCS(PointBufferPtr pBuffer, std::string config) {

  // Read pointBuffer to get points and number of points
  m_pointCoord = pBuffer.get()->getIndexedPointArray(m_pointNumber);

  // copy point coords to vector with Point_3 which is common in CGAL
  for (size_t i = 0; i < m_pointNumber; i++) {
    Point_3 p(m_pointCoord[i].x, m_pointCoord[i].y, m_pointCoord[i].z);
    m_point3Vec.push_back(p);
  }

  typename Filter::EdgelessVertex *a = new typename Filter::EdgelessVertex();
  typename Filter::EdgeWithoutTriangles *b =
      new typename Filter::EdgeWithoutTriangles();
  typename Filter::HoleFilling *c = new typename Filter::HoleFilling();
  typename Filter::ValenceTwoRemove *d =
      new typename Filter::ValenceTwoRemove();
  typename Filter::HighValenceRemove *e =
      new typename Filter::HighValenceRemove();

  // m_filterContainer.push_back(a);
  // m_filterContainer.push_back(b);
  // m_filterContainer.push_back(c);
  // m_filterContainer.push_back(d);
  // m_filterContainer.push_back(e);

  m_filterChain = new FilterChain<PolyhedronMesh>(m_filterContainer);
  m_filterChain->setVertexContainer(m_vertexContainer);
  boost::posix_time::ptime p(boost::posix_time::microsec_clock::local_time());
  datafile.open("gcs_" + to_simple_string(p) + ".csv");
  if (!datafile.is_open()) {
    BOOST_LOG_TRIVIAL(error) << "Could not open data file.";
  }
  std::string line;
  ifstream configfile(config);
  if (configfile.is_open()) {
    while (getline(configfile, line)) {
      std::istringstream is_line(line);
      std::string key;
      if (std::getline(is_line, key, '=')) {
        std::string value;
        if (std::getline(is_line, value))
          if (key == "runtime") {
            m_runtime = std::stoi(value);
            BOOST_LOG_TRIVIAL(info) << "Runtime: " << m_runtime;
            datafile << "# m_runtime : " + std::to_string(m_runtime) << "\n";
          }
        if (key == "boxFactor") {
          m_boxFactor = std::stof(value);
          BOOST_LOG_TRIVIAL(info) << "Box Factor: " << m_boxFactor;
          datafile << "# m_boxFactor : " + std::to_string(m_boxFactor) << "\n";
        }
        if (key == "numSplits") {
          m_numSplits = std::stoi(value);
          BOOST_LOG_TRIVIAL(info) << "Number of splits: " << m_numSplits;
          datafile << "# m_numSplits : " + std::to_string(m_numSplits) << "\n";
        }
        if (key == "withCollaps") {
          m_withCollaps = std::stoi(value);
          BOOST_LOG_TRIVIAL(info) << "Collapse edge: " << m_withCollaps;
          datafile << "# m_withCollaps : " + std::to_string(m_withCollaps)
                   << "\n";
        }
        if (key == "learningRate") {
          m_learningRate = std::stof(value);
          BOOST_LOG_TRIVIAL(info) << "Learning rate: " << m_learningRate;
          datafile << "# m_learningRate : " + std::to_string(m_learningRate)
                   << "\n";
        }
        if (key == "decreaseFactor") {
          m_decreasingFactor = std::stof(value);
          BOOST_LOG_TRIVIAL(info) << "Decreasing factor: "
                                  << m_decreasingFactor;
          datafile << "# m_decreasingFactor : " +
                          std::to_string(m_decreasingFactor)
                   << "\n";
        }
        if (key == "neighborLr") {
          m_neighborLearningRate = std::stof(value);
          BOOST_LOG_TRIVIAL(info) << "Neighbor learning rate: "
                                  << m_neighborLearningRate;
          datafile << "# m_neighborLearningRate : " +
                          std::to_string(m_neighborLearningRate)
                   << "\n";
        }
        if (key == "basicSteps") {
          m_basicSteps = std::stoi(value);
          BOOST_LOG_TRIVIAL(info) << "Number of basic steps per runtime: "
                                  << m_basicSteps;
          datafile << "# m_basicSteps : " + std::to_string(m_basicSteps)
                   << "\n";
        }
        if (key == "allowedMiss") {
          m_allowedMiss = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "Allowed miss: " << m_allowedMiss;
          datafile << "# m_allowedMiss : " + std::to_string(m_allowedMiss)
                   << "\n";
        }
        if (key == "collapsThreshold") {
          m_collapsThreshold = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "Collapse Threshold: "
                                     << m_collapsThreshold;
          datafile << "# m_collapsThreshold : " +
                          std::to_string(m_collapsThreshold)
                   << "\n";
        }
        if (key == "delteLongEdgesFactor") {
          m_deleteLongHE = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "m_deleteLongHE: " << m_deleteLongHE;
          datafile << "# m_deleteLongHE : " + std::to_string(m_deleteLongHE)
                   << "\n";
        }
        if (key == "filterChain") {
          m_withFilter = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "FilterChain: " << m_withFilter;
          datafile << "# m_withFilter : " + std::to_string(m_withFilter)
                   << "\n";
        }
      }
    }
    configfile.close();
  } else {
    BOOST_LOG_TRIVIAL(fatal) << "Could not open config file.";
    exit(-1);
  }
  datafile << "\n";
}

// initSimpleMesh
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::initSimpleMesh() {
  // Create BoundingBox and expand it around Pointcloud
  BoundingBox<VertexT> m_boundingBox = BoundingBox<VertexT>();
  for (size_t i = 0; i < m_pointNumber; i++) {
    m_boundingBox.expand(m_pointCoord[i][0], m_pointCoord[i][1],
                         m_pointCoord[i][2]);
  }
  // Check if BoundingBox is valid and exit otherwise
  if (!m_boundingBox.isValid()) {
    BOOST_LOG_TRIVIAL(fatal) << "BoundingBox is invalid.";
    exit(-1);
  }

  /*
   * Get min and max vertex from BoundingBox and calculate the x/y/z
   * differences.
   * Half them and weight by 1-boxFactor parameter to get the positions for the
   * new shrinked bounding box (inital mesh).
   * minx         maxx
   * O-->-------<---O
   */
  VertexT min(m_boundingBox.getMin());
  VertexT max(m_boundingBox.getMax());

  float xdiff, ydiff, zdiff;
  xdiff = (max.x - min.x) / 2;
  ydiff = (max.y - min.y) / 2;
  zdiff = (max.z - min.z) / 2;

  xdiff *= (1 - m_boxFactor);
  ydiff *= (1 - m_boxFactor);
  zdiff *= (1 - m_boxFactor);
  BOOST_LOG_TRIVIAL(debug) << "Weighted halved differences: " << xdiff << "/"
                           << ydiff << "/" << zdiff;

  // New Vector positions
  float minx, miny, minz, maxx, maxy, maxz;
  minx = min.x + xdiff;
  miny = min.y + ydiff;
  minz = min.z + zdiff;
  maxx = max.x - xdiff;
  maxy = max.y - ydiff;
  maxz = max.z - zdiff;
  BOOST_LOG_TRIVIAL(debug) << "Min. " << minx << "/" << miny << "/" << minz;
  BOOST_LOG_TRIVIAL(debug) << "Max. " << maxx << "/" << maxy << "/" << maxz;

  // add them to mesh and create triangles
  m_mesh->addVertex(VertexT(minx, miny, minz)); // 0 (min)
  m_mesh->addVertex(VertexT(maxx, miny, minz)); // 1
  m_mesh->addVertex(VertexT(maxx, miny, maxz)); // 2
  m_mesh->addVertex(VertexT(minx, miny, maxz)); // 3
  m_mesh->addVertex(VertexT(minx, maxy, maxz)); // 4
  m_mesh->addVertex(VertexT(minx, maxy, minz)); // 5
  m_mesh->addVertex(VertexT(maxx, maxy, minz)); // 6
  m_mesh->addVertex(VertexT(maxx, maxy, maxz)); // 7 (max)

  // Bottom
  m_mesh->addTriangle(0, 2, 3);
  m_mesh->addTriangle(0, 1, 2);

  // top
  m_mesh->addTriangle(4, 6, 5);
  m_mesh->addTriangle(4, 7, 6);

  // front
  m_mesh->addTriangle(3, 7, 4);
  m_mesh->addTriangle(2, 7, 3);

  // left
  m_mesh->addTriangle(0, 4, 5);
  m_mesh->addTriangle(0, 3, 4);

  // right
  m_mesh->addTriangle(2, 6, 7);
  m_mesh->addTriangle(2, 1, 6);

  // back
  m_mesh->addTriangle(1, 5, 6);
  m_mesh->addTriangle(0, 5, 1);
}

template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::initTetrahedron() {
  // Create BoundingBox and expand it around Pointcloud
  BoundingBox<VertexT> m_boundingBox = BoundingBox<VertexT>();
  for (size_t i = 0; i < m_pointNumber; i++) {
    m_boundingBox.expand(m_pointCoord[i][0], m_pointCoord[i][1],
                         m_pointCoord[i][2]);
  }
  // Check if BoundingBox is valid and exit otherwise
  if (!m_boundingBox.isValid()) {
    BOOST_LOG_TRIVIAL(fatal) << "BoundingBox is invalid.";
    exit(-1);
  }

  VertexT center = m_boundingBox.getCentroid();
  /*
   * Get min and max vertex from BoundingBox and calculate the x/y/z
   * differences.
   * Half them and weight by 1-boxFactor parameter to get the positions for the
   * new shrinked bounding box (inital mesh).
   * minx         maxx
   * O-->-------<---O
   */
  VertexT min(m_boundingBox.getMin());
  VertexT max(m_boundingBox.getMax());

  float xdiff, ydiff, zdiff;
  xdiff = (max.x - min.x) / 2;
  ydiff = (max.y - min.y) / 2;
  zdiff = (max.z - min.z) / 2;

  xdiff *= (1 - m_boxFactor);
  ydiff *= (1 - m_boxFactor);
  zdiff *= (1 - m_boxFactor);
  BOOST_LOG_TRIVIAL(debug) << "Weighted halved differences: " << xdiff << "/"
                           << ydiff << "/" << zdiff;

  // New Vector positions
  float minx, miny, minz, maxx, maxy, maxz;
  minx = min.x + xdiff;
  miny = min.y + ydiff;
  minz = min.z + zdiff;
  maxx = max.x - xdiff;
  maxy = max.y - ydiff;
  maxz = max.z - zdiff;
  BOOST_LOG_TRIVIAL(debug) << "Min. " << minx << "/" << miny << "/" << minz;
  BOOST_LOG_TRIVIAL(debug) << "Max. " << maxx << "/" << maxy << "/" << maxz;

  Point_3 top(center.x, maxy, center.z);
  Point_3 left(minx, miny, minz);
  Point_3 right(maxx, miny, minz);
  Point_3 back(center.x, miny, maxz);

  m_mesh->getPolyhedron().make_tetrahedron(top, left, right, back);
}

// getMesh
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::getMesh(
    GCS<VertexT, NormalT>::PolyhedronMesh &mesh) {

  // Logging severity from config
  logging::core::get()->set_filter(logging::trivial::severity >=
                                   logging::trivial::info);

  m_mesh = &mesh;
  m_filterChain->setMesh(*m_mesh);
  d_signalCounterMean = 0.0;
  // initSimpleMesh();
  initTetrahedron();

  // Enumerate initial vertices
  for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
    v->vId = m_vertexIndex++;
    // v->signal_counter = 10;
  }
  vppmap = CGAL::get(CGAL::vertex_point, m_mesh->getPolyhedron());
  tree = new Tree(CGAL::vertices(m_mesh->getPolyhedron()).begin(),
                  CGAL::vertices(m_mesh->getPolyhedron()).end(), Splitter(),
                  Traits(vppmap));
  tr_dist = new Distance(vppmap);

  boost::progress_display show_progress(
      (long)(m_runtime * m_numSplits * m_basicSteps));
  Time t_start(boost::posix_time::microsec_clock::local_time());
  datafile << "m_runtime,#vertices,#facets,#scmean,avgvertexDist,mstime\n";

  for (size_t k = 0; k < m_runtime; k++) {
    for (size_t i = 0; i < m_numSplits; i++) {
      delete tree;
      delete tr_dist;
      vppmap = CGAL::get(CGAL::vertex_point, m_mesh->getPolyhedron());
      tree = new Tree(CGAL::vertices(m_mesh->getPolyhedron()).begin(),
                      CGAL::vertices(m_mesh->getPolyhedron()).end(), Splitter(),
                      Traits(vppmap));
      tr_dist = new Distance(vppmap);
      for (size_t j = 0; j < m_basicSteps; j++) {
        performBasicStep();
        ++show_progress;
        if (m_withFilter) {
          m_filterChain->execute();
        }
      }
      performVertexSplit();
      if (m_withFilter) {
        m_filterChain->execute();
      }
    }

    if (m_withCollaps) {
      performEdgeCollaps();
      if (m_withFilter) {
        m_filterChain->execute();
      }
    }
    calcSCMean();
    datafile << k << "," << m_mesh->size() << "," << m_mesh->size_of_facets()
             << "," << std::fixed << std::setprecision(8) << d_signalCounterMean
             << "," << std::fixed << std::setprecision(8) << d_vertexDistance
             << ","
             << (boost::posix_time::microsec_clock::local_time() - t_start)
                    .total_milliseconds()
             << "\n";
    // m_filterChain->execute();
  }
  Time t_end(boost::posix_time::microsec_clock::local_time());
  d_runtime = t_end - t_start;

  d_avgHELength = 0.0;

  for (auto h = m_mesh->halfedges_begin(); h != m_mesh->halfedges_end(); ++h) {
    Point_3 p1 = h->vertex()->point();
    Point_3 p2 = h->opposite()->vertex()->point();
    double length = std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                              std::pow((p1.y() - p2.y()), 2) +
                              std::pow(p1.z() - p2.z(), 2));
    d_avgHELength += length;
  }

  d_avgHELength /= m_mesh->size_of_halfedges();
  d_longEdgesDeleted = 0;
  if (m_deleteLongHE > 0.0) {
    for (auto h = m_mesh->halfedges_begin(); h != m_mesh->halfedges_end();
         ++h) {
      Point_3 p1 = h->vertex()->point();
      Point_3 p2 = h->opposite()->vertex()->point();
      double length = std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                                std::pow((p1.y() - p2.y()), 2) +
                                std::pow(p1.z() - p2.z(), 2));
      if (length > (d_avgHELength * m_deleteLongHE) && !h->is_border()) {

        d_longEdgesDeleted++;
        m_mesh->getPolyhedron().erase_facet(h);
      }
    }
  }
  printInfo();
}

// performBasicStep
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::performBasicStep() {
  // Select random sample p x of P
  coord<float> p1 = randomPointCoord();
  Point_3 p(p1.x, p1.y, p1.z);
  K_neighbor_search search(*tree, p, 1, 0, true, *tr_dist);

  Vertex_handle winner = search.begin()->first;

  if (winner == NULL) {
    BOOST_LOG_TRIVIAL(error) << "Skip one vertex.";
    return;
  }

  // Move v x as much toward p x as determined by the learning rate: ∆v x =
  // lr(p
  // x − v x )
  moveVertex(winner, p1);

  neighborSmoothing(winner, p1);

  updateSignalCounter(winner);

  double dist = CGAL::squared_distance(winner->point(), p);
  dist = std::sqrt(dist);
  winner->latestDist = dist;

  if (m_withFilter) {
    // Container on which filterchain is applied
    m_vertexContainer.push_back(winner);
    // m_filterChain->execute();
  }
}

// performVertexSplit
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::performVertexSplit() {
  // select v with highest signal couter
  Vertex_handle source = vertexWithHighestSC();
  Vertex_handle target = source;

  // Looking for longest Edge from source to any of his neighbors
  typename PolyhedronMesh::Halfedge_around_vertex_circulator hasc =
      source->vertex_begin();

  do {
    if (CGAL::has_larger_distance_to_point(source->point(),
                                           hasc->opposite()->vertex()->point(),
                                           target->point())) {
      target = hasc->opposite()->vertex();
    }
  } while (++hasc != source->vertex_begin());

  if (target == source) {
    BOOST_LOG_TRIVIAL(fatal) << "Error.";
    exit(-1);
  }

  Halfedge_handle halfedgeToSplit;
  bool success;
  boost::tie(halfedgeToSplit, success) =
      CGAL::halfedge(source, target, m_mesh->getPolyhedron());

  if (!success) {
    BOOST_LOG_TRIVIAL(fatal) << "Can't find splitting HE.";
    exit(-1);
  }
  if (halfedgeToSplit->is_border_edge()) {
    BOOST_LOG_TRIVIAL(error) << "Dont split border edge!";
    return;
  }
  // Checks only if degree < 3
  if (CGAL::source(halfedgeToSplit, m_mesh->getPolyhedron())->vertex_degree() <
          3 ||
      CGAL::target(halfedgeToSplit, m_mesh->getPolyhedron())->vertex_degree() <
          3) {
    std::cout << "Nope ... should apply filterChain." << std::endl;
    source->signal_counter /= 2;
    return;
  }

  /* TRANSLATION */

  // splits the halfedge halfedgeToSplit into two halfedges inserting a new
  // vertex that is a copy of halfedgeToSplit->opposite()->vertex() (source!)
  Halfedge_handle hnew = m_mesh->getPolyhedron().split_edge(halfedgeToSplit);
  Vertex_handle vnew = hnew->vertex();
  vnew->vId = m_vertexIndex++;

  // Half the distance to target to move new vertex
  // h->vertex()->point() == source
  float xDiff = (vnew->point().x() - target->point().x()) / 2;
  float yDiff = (vnew->point().y() - target->point().y()) / 2;
  float zDiff = (vnew->point().z() - target->point().z()) / 2;

  Vector_3 translationVector = Vector_3(-xDiff, -yDiff, -zDiff);
  m_mesh->translate(vnew->point(), translationVector);

  /* SET SIGNAL COUNTER AND ADD TO FILTERCHAIN */
  source->signal_counter /= 2;
  vnew->signal_counter = source->signal_counter;
  if (m_withFilter) {
    m_vertexContainer.push_back(vnew);
    m_vertexContainer.push_back(source);
  }

  /* TRIANGULATION */
  if (!hnew->is_border()) {
    CGAL::Polygon_mesh_processing::triangulate_face(hnew->facet(),
                                                    m_mesh->getPolyhedron());
  } else {
    return;
  }
  if (!hnew->opposite()->is_border()) {
    CGAL::Polygon_mesh_processing::triangulate_face(hnew->opposite()->facet(),
                                                    m_mesh->getPolyhedron());
  } else {
    return;
  }

  /* CORRECTNESS OF VALENCE */
  Vertex_handle t1 = hnew->next()->vertex();
  Vertex_handle t2 = hnew->opposite()->next()->vertex();

  std::set<Halfedge_handle, compare_heBySourceDegreeDesc> possibleHE;
  hasc = vnew->vertex_begin();
  do {
    possibleHE.insert(hasc->opposite());

  } while (++hasc != vnew->vertex_begin());

  for (Halfedge_handle h : possibleHE) {
    if (h->vertex()->vertex_degree() > 6 && vnew->vertex_degree() < 6) {
      Halfedge_handle he1 = h->next()->opposite();
      Halfedge_handle he2 = h->opposite()->prev();
      if ((CGAL::source(he1, m_mesh->getPolyhedron())->vertex_degree() >=
           CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree()) &&
          (CGAL::source(he1, m_mesh->getPolyhedron())->vertex_degree() > 4) &&
          !he1->is_border()) {
        m_mesh->getPolyhedron().flip_edge(he1);
      } else if ((CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree() >=
                  CGAL::source(he1, m_mesh->getPolyhedron())
                      ->vertex_degree()) &&
                 (CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree() >
                  4) &&
                 !he2->is_border()) {
        m_mesh->getPolyhedron().flip_edge(he2);
      } else {
      }
    }
  }
}

// performEdgeCollaps
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::performEdgeCollaps() {
  // Searching for v with lowest sc
  float lowestSC = m_mesh->vertices_begin()->signal_counter;
  Vertex_handle source = m_mesh->vertices_begin();
  Vertex_handle target = source;

  for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
    if (v->signal_counter <= lowestSC) {
      lowestSC = v->signal_counter;
      source = v;
    }
  };
  if (lowestSC <= m_collapsThreshold) {
    // Looking for collapse he
    typename PolyhedronMesh::Halfedge_around_vertex_circulator hasc =
        source->vertex_begin();
    target = hasc->opposite()->vertex();
    size_t minDegree = target->vertex_degree();
    do {
      if (hasc->opposite()->vertex()->vertex_degree() < minDegree) {
        target = hasc->opposite()->vertex();
        size_t minDegree = target->vertex_degree();
      } else if (hasc->opposite()->vertex()->vertex_degree() == minDegree) {
        if (hasc->opposite()->vertex()->signal_counter >
            target->signal_counter) {
          target = hasc->opposite()->vertex();
          size_t minDegree = target->vertex_degree();
        }
      }
    } while (++hasc != source->vertex_begin());

    Halfedge_handle halfedgeToCollapse;
    bool success;
    boost::tie(halfedgeToCollapse, success) =
        CGAL::halfedge(source, target, m_mesh->getPolyhedron());

    if (!success) {
      BOOST_LOG_TRIVIAL(fatal) << "Can't find collapse HE.";
      return;
    }
    b_edge_descriptor e =
        CGAL::edge(halfedgeToCollapse, m_mesh->getPolyhedron());
    halfedgeToCollapse->setDelete();
    CGAL::set_halfedgeds_items_id(m_mesh->getPolyhedron());
    std::size_t delta_f = m_mesh->size_of_facets();
    // Experimental
    SMS::Count_stop_predicate<Polyhedron_3> stop(
        CGAL::num_edges(m_mesh->getPolyhedron()) - 1);
    GCSEdgeVisitior visitor;
    Collapse_constrained_edge_map collapse_map(m_mesh->getPolyhedron());

    int r = SMS::edge_collapse(
        m_mesh->getPolyhedron(), stop,
        CGAL::parameters::edge_is_constrained_map(collapse_map)
            .visitor(visitor)
            .get_placement(Placement(collapse_map)));

    delta_f -= m_mesh->size_of_facets();
    halfedgeToCollapse->setSave();

    if (delta_f != 2) {
      BOOST_LOG_TRIVIAL(error) << "Collapsing removed other than 2 facets.";
    }

    // edge_descriptor e = CGAL::edge(halfedgeToCollapse,
    // m_mesh->getPolyhedron());
    // Vertex_handle vnew = CGAL::Euler::collapse_edge(e,
    // m_mesh->getPolyhedron());

    // std::cout << vnew << std::endl;
    if (m_withFilter) {
      // Container on which filterchain is applied
      //   m_vertexContainer.push_back(vnew);
      //   m_filterChain->execute();
    }
  }
}

// randomPointCoord
template <typename VertexT, typename NormalT>
coord<float> GCS<VertexT, NormalT>::randomPointCoord() {
  int index = m_random.get_int(0, m_pointNumber);
  return m_pointCoord[index];
}

// updateSignalCounter
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::updateSignalCounter(Vertex_handle &winner) {
  // decrease signal counter
  if (m_decreasingFactor == 1.0) {
    uint n = m_allowedMiss * m_mesh->size();
    double dynamicDecrease = 1 - pow(m_collapsThreshold, (1.0 / n));
    for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
      if (!(winner == v)) {
        double delta = -dynamicDecrease * v->signal_counter;
        v->signal_counter += delta;
      }
    }
  } else {
    for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
      if (!(winner == v)) {
        v->signal_counter -= (m_decreasingFactor * v->signal_counter);
      }
    }
  }
}

// nearestVertex
template <typename VertexT, typename NormalT>
auto GCS<VertexT, NormalT>::nearestVertex(Point_3 p) -> Vertex_handle {
  Vertex_handle minV = m_mesh->vertices_begin();
  for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
    // returns true if the distance between v and p is smaller than the
    // distance between minV and p.
    if (CGAL::has_smaller_distance_to_point(p, v->point(), minV->point())) {
      minV = v;
    }
  }
  return minV;
}

// moveVertex
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::moveVertex(Vertex_handle &v, coord<float> p) {
  // Calcualte distances
  float x = m_learningRate * (v->point().x() - p.x);
  float y = m_learningRate * (v->point().y() - p.y);
  float z = m_learningRate * (v->point().z() - p.z);

  m_mesh->translate(v->point(), Vector_3(-x, -y, -z));

  v->signal_counter += 1.0;
}

// moveNeighbor
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::moveNeighbor(Vertex_handle &v, coord<float> p) {
  // Calcualte distances
  float x = m_neighborLearningRate * (v->point().x() - p.x);
  float y = m_neighborLearningRate * (v->point().y() - p.y);
  float z = m_neighborLearningRate * (v->point().z() - p.z);
  m_mesh->translate(v->point(), Vector_3(-x, -y, -z));
}

// vertexWithHighestSC
template <typename VertexT, typename NormalT>
auto GCS<VertexT, NormalT>::vertexWithHighestSC() -> Vertex_handle {
  Vertex_handle w = m_mesh->vertices_begin();
  double maxSC = w->signal_counter;
  for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
    if (v->signal_counter > maxSC) {
      w = v;
      maxSC = w->signal_counter;
    }
  }
  return w;
}

// neighborSmoothing
template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::neighborSmoothing(Vertex_handle &source,
                                              coord<float> targetPoint) {
  typename PolyhedronMesh::Halfedge_around_vertex_circulator hasc =
      source->vertex_begin();
  do {
    Vertex_handle n = hasc->opposite()->vertex();
    moveNeighbor(n, targetPoint);
  } while (++hasc != source->vertex_begin());
}

template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::calcSCMean() {
  d_signalCounterMean = 0.0;
  d_vertexDistance = 0.0;
  for (auto v = m_mesh->vertices_begin(); v != m_mesh->vertices_end(); ++v) {
    d_signalCounterMean += v->getSignalCounter();
    d_vertexDistance += v->latestDist;
  }
  d_signalCounterMean /= m_mesh->size();
  d_vertexDistance /= m_mesh->size();
}

template <typename VertexT, typename NormalT>
void GCS<VertexT, NormalT>::printInfo() {
  std::cout << "\n+++++++ Info +++++++\n" << std::endl;
  std::cout << "Runtime: " << d_runtime << "\n";
  std::cout << "# Facets: " << m_mesh->size_of_facets() << "\n";
  std::cout << "# Vertices: " << m_mesh->size() << "\n";
  std::cout << "# HE: " << m_mesh->size_of_halfedges() << "\n";
  std::cout << "Average HE Length: " << d_avgHELength << "\n";
  std::cout << "# HE deleted: " << d_longEdgesDeleted << "\n";

  calcSCMean();
  std::cout << "Mean SC: " << d_signalCounterMean << "\n";
  std::cout << "Mean distance v->p: " << d_vertexDistance << "\n";
}
template <typename VertexT, typename NormalT> GCS<VertexT, NormalT>::~GCS() {
  BOOST_LOG_TRIVIAL(info) << "Growing Cell Structures: Destruction";
  delete tree;
  delete tr_dist;
  delete m_filterChain;
  datafile.close();
}
} // namespace lvr2
