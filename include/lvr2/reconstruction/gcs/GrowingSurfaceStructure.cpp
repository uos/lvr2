
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

namespace lvr2 {

GrowingSurfaceStructure::GrowingSurfaceStructure() {
}

GrowingSurfaceStructure::GrowingSurfaceStructure(PointBufferPtr pBuffer,
                                                 std::string config) {

  // Get pointarray from pBuffer, save it to m_pointCoord and set m_pointNumber
  m_pointNumber = pBuffer->numPoints();
  //GET POINT ARRAY (FloatChannel)
  m_pointCoord = *(pBuffer->getFloatChannel("points"));

  //m_pointCoord = pBuffer.get()->getIndexedPointArray(m_pointNumber);

  // copy point coords to vector with Point_3 which the common format in CGAL
  for (size_t i = 0; i < m_pointNumber; i++) {
    m_point3Vec.push_back(
        Point_3(m_pointCoord[i][0], m_pointCoord[i][1], m_pointCoord[i][2]));
  }
  boost::posix_time::ptime p(boost::posix_time::microsec_clock::local_time());
  datafile.open("gss_" + to_simple_string(p) + ".csv");

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
            p_runtime = std::stoi(value);
            BOOST_LOG_TRIVIAL(info) << "runtime: " << p_runtime;
            datafile << "# p_runtime : " + std::to_string(p_runtime) << "\n";
          }
        if (key == "printInfo") {
          p_printInfo = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "printInfo: " << p_printInfo;
          datafile << "# p_printInfo : " + std::to_string(p_printInfo) << "\n";
        }
        if (key == "lr") {
          p_learningRate = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "lr: " << p_learningRate;
          datafile << "# p_learningRate : " + std::to_string(p_learningRate)
                   << "\n";
        }
        if (key == "lrn") {
          p_neighborLearningRate = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "lrn: " << p_neighborLearningRate;
          datafile << "# p_neighborLearningRate : " +
                          std::to_string(p_neighborLearningRate)
                   << "\n";
        }
        if (key == "limitSkip") {
          p_limitSkip = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "limitSkip: " << p_limitSkip;
          datafile << "# p_limitSkip : " + std::to_string(p_limitSkip) << "\n";
        }
        if (key == "limitSingle") {
          p_limitSingle = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "p_limitSingle: " << p_limitSingle;
          datafile << "# p_limitSingle : " + std::to_string(p_limitSingle)
                   << "\n";
        }
        if (key == "maxDeltaN") {
          p_maxDeltaN = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "p_maxDeltaN: " << p_maxDeltaN;
          datafile << "# p_maxDeltaN : " + std::to_string(p_maxDeltaN) << "\n";
        }
        if (key == "maxAge") {
          p_maxAge = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_maxAge: " << p_maxAge;
          datafile << "# p_maxAge : " + std::to_string(p_maxAge) << "\n";
        }
        if (key == "allowedMissed") {
          p_allowedMiss = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_allowedMiss: " << p_allowedMiss;
          datafile << "# p_allowedMiss : " + std::to_string(p_allowedMiss)
                   << "\n";
        }
        if (key == "numSplits") {
          p_numSplits = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_numSplits: " << p_numSplits;
          datafile << "# p_numSplits : " + std::to_string(p_numSplits) << "\n";
        }
        if (key == "numBasicSteps") {
          p_numBasicSteps = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_numBasicSteps: " << p_numBasicSteps;
          datafile << "# p_numBasicSteps : " + std::to_string(p_numBasicSteps)
                   << "\n";
        }
        if (key == "lambda") {
          p_lambda = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_lambda: " << p_lambda;
          datafile << "# p_lambda : " + std::to_string(p_lambda) << "\n";
        }
        if (key == "scale") {
          p_scale = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "p_scale: " << p_scale;
          datafile << "# p_scale : " + std::to_string(p_scale) << "\n";
        }
        if (key == "holeSize") {
          p_holeSize = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_holeSize: " << p_holeSize;
          datafile << "# p_holeSize : " + std::to_string(p_holeSize) << "\n";
        }
        if (key == "rollingMean") {
          p_rollingMean = std::stoi(value);
          if (p_rollingMean == 0) {
            p_rollingMean = m_pointNumber;
          }
          BOOST_LOG_TRIVIAL(warning) << "p_rollingMean: " << p_rollingMean;
          datafile << "# p_rollingMean : " + std::to_string(p_rollingMean)
                   << "\n";
        }
        if (key == "deleteLongHe") {
          p_deleteLongHE = std::stof(value);
          BOOST_LOG_TRIVIAL(warning) << "p_deleteLongHE: " << p_deleteLongHE;
          datafile << "# p_deleteLongHE : " + std::to_string(p_deleteLongHE)
                   << "\n";
        }
        if (key == "skipSTS") {
          p_skipSTSCalculation = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_skipSTSCalculation: "
                                     << p_skipSTSCalculation;
          datafile << "# p_skipSTSCalculation : " +
                          std::to_string(p_skipSTSCalculation)
                   << "\n";
        }
        if (key == "skipRemoveFacets") {
          p_skipRemoveFacets = std::stoi(value);

          BOOST_LOG_TRIVIAL(warning) << "p_skipRemoveFacets: "
                                     << p_skipRemoveFacets;
          datafile << "# p_skipRemoveFacets : " +
                          std::to_string(p_skipRemoveFacets)
                   << "\n";
        }
        if (key == "skipRemove") {
          p_skipRemove = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "skipRemove: " << p_skipRemove;
          datafile << "# p_skipRemove : " + std::to_string(p_skipRemove)
                   << "\n";
        }
        if (key == "convex") {
          p_useConvex = std::stoi(value);
          BOOST_LOG_TRIVIAL(warning) << "p_useConvex: " << p_useConvex;
          datafile << "# p_useConvex : " + std::to_string(p_useConvex) << "\n";
        }
      }
    }
  }
  datafile << "\n";
  configfile.close();
  it_counter = 0;

  // Calculation
  m_k = (1 / (1 - (std::pow(0.5, (1.0 / p_lambda)))));
  m_sDistance = 0.0;
  m_avgSampleToSurface = 0.0;

  // data collection
  d_holesFilled = 0;
  d_facetsErased = 0;
  d_edgeCollapsed = 0;
  d_edgeSplitted = 0;
  d_edgeSplitted = 0;
  d_winningDistance = 0.0;

  // Create filter
  typename Chain::EdgelessVertex *a = new typename Chain::EdgelessVertex();
  typename Chain::EdgeWithoutTriangles *b =
      new typename Chain::EdgeWithoutTriangles();
  typename Chain::ValenceTwoRemove *c = new typename Chain::ValenceTwoRemove();
  typename Chain::HoleFilling *f = new typename Chain::HoleFilling();

  // push filter to container tempoary disabled
  // m_filterContainer.push_back(a);
  // m_filterContainer.push_back(b);
  // m_filterContainer.push_back(c);
  // m_filterContainer.push_back(f);

  // create filter chain with filter container and set vertex container
  m_filterChain = new FilterChain<PolyhedronMesh>(m_filterContainer);
  m_filterChain->setVertexContainer(m_vertexContainer);

  acc = new boost::accumulators::accumulator_set<
      double,
      boost::accumulators::stats<boost::accumulators::tag::rolling_mean>>(
      boost::accumulators::tag::rolling_window::window_size = p_rollingMean);
};

void GrowingSurfaceStructure::getMesh(PolyhedronMesh &mesh) {

  boost::log::core::get()->set_filter(boost::log::trivial::severity >=
                                      boost::log::trivial::warning);
  // Save reference for internal use
  m_mesh = &mesh;

  // set mesh for filter
  m_filterChain->setMesh(*m_mesh);

  // initialize tetrahedron
  if (p_useConvex) {
    CGAL::convex_hull_3(m_point3Vec.begin(), m_point3Vec.end(),
                        m_mesh->getPolyhedron());
    updateFacetAverages();

    for (auto f = m_mesh->facets_begin(); f != m_mesh->facets_end(); ++f) {
      setInitialAge(f);
    }
  } else {
    initializeTetrahedron();
  }
  // first setup of search tree
  vppmap = CGAL::get(CGAL::vertex_point, m_mesh->getPolyhedron());
  tree = new Tree(CGAL::vertices(m_mesh->getPolyhedron()).begin(),
                  CGAL::vertices(m_mesh->getPolyhedron()).end(), Splitter(),
                  Traits(vppmap));
  tr_dist = new Distance(vppmap);

  for (auto iter = m_point3Vec.begin(); iter != m_point3Vec.end(); ++iter) {
    double oldDist = std::numeric_limits<double>::max();
    for (auto f = m_mesh->facets_begin(); f != m_mesh->facets_end(); ++f) {
      Vector_3 dist = f->point() - (*iter);
      double euclidianDistance = std::sqrt(dist.squared_length());
      if (euclidianDistance < oldDist) {
        oldDist = euclidianDistance;
      }
    }
    m_avgSampleToSurface += oldDist;
  }
  m_avgSampleToSurface /= m_pointNumber;
  updateFacetAverages();

  BOOST_LOG_TRIVIAL(info) << "Initial sample-to-surface distance: "
                          << m_avgSampleToSurface;
  (*acc)(m_avgSampleToSurface);
  if (p_runtime == 0) {
    return;
  }

  std::cout << boost::accumulators::rolling_mean(*acc) << "\n";
  boost::progress_display show_progress((long)p_runtime);
  Time t_start(boost::posix_time::microsec_clock::local_time());
  datafile << "it_counter,#vertices,#facets,m_avgTriangleSize,m_avgTriangleErr,"
              "m_avgSampleToSurface,mstime\n";

  for (int i = 0; i < p_runtime; ++i) {
    ++show_progress;
    for (int j = 0; j < p_numSplits; ++j) {
      for (int k = 0; k < p_numBasicSteps; ++k) {
        refreshSearchTree();   // suchbaum aktualisieren
        updateFacetAverages(); // avg. size der dreiecke
        performBasicStep();
        // m_filterChain->execute(); // filter chain
      }
      performVertexSplit();
      // m_filterChain->execute(); // filter chain
    }
    if (!p_skipRemove) {
      performRemove();
    }
    updateFacetAverages();
    datafile << it_counter << "," << m_mesh->size() << ","
             << m_mesh->size_of_facets() << "," << std::fixed
             << std::setprecision(8) << m_avgTriangleSize << "," << std::fixed
             << std::setprecision(8) << m_avgTriangleErr << "," << std::fixed
             << std::setprecision(8) << m_avgSampleToSurface << ","
             << std::fixed << std::setprecision(8)
             << (boost::posix_time::microsec_clock::local_time() - t_start)
                    .total_milliseconds()
             << "\n";
  }

  Time t_end(boost::posix_time::microsec_clock::local_time());
  d_runtime = t_end - t_start;

  double d_avgHELength = 0.0;

  for (auto h = m_mesh->halfedges_begin(); h != m_mesh->halfedges_end(); ++h) {
    Point_3 p1 = h->vertex()->point();
    Point_3 p2 = h->opposite()->vertex()->point();
    double length = std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                              std::pow((p1.y() - p2.y()), 2) +
                              std::pow(p1.z() - p2.z(), 2));
    d_avgHELength += length;
  }

  d_avgHELength /= m_mesh->size_of_halfedges();

  if (p_deleteLongHE > 0.0) {
    for (auto h = m_mesh->halfedges_begin(); h != m_mesh->halfedges_end();
         ++h) {
      Point_3 p1 = h->vertex()->point();
      Point_3 p2 = h->opposite()->vertex()->point();
      double length = std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                                std::pow((p1.y() - p2.y()), 2) +
                                std::pow(p1.z() - p2.z(), 2));
      if (length > (d_avgHELength * p_deleteLongHE) && !h->is_border()) {
        m_mesh->getPolyhedron().erase_facet(h);
      }
    }
  }

  if (p_printInfo) {
    printInfo();
  }
}

void GrowingSurfaceStructure::performBasicStep() {
  it_counter++;
  // Get random point and perform k-neighbor search with k=3
  Point_3 p = getRandomPoint();

  K_neighbor_search search(*tree, p, 5, 0, true, *tr_dist);
  Point_3 winningPoint = search.begin()->first->point();
  // We have k=3 vertices near to p. winningPoint is the point from the first
  // vertex
  bool withVertex = true;
  bool withFacet = false;
  bool withHE = false;

  Vertex_handle potVertex = search.begin()->first;
  Facet_handle potFacet;
  Halfedge_handle potHE;
  // Now Check all structures around NN if they are nearer to sample p
  for (auto it = search.begin(); it != search.end(); it++) {
    Vertex_handle vref = (it->first);
    if (CGAL::has_larger_distance_to_point(p, winningPoint, vref->point())) {
      // indicates that new vertex vref are closer to p than winningPoint
      winningPoint = vref->point();
      withVertex = true;
      withFacet = false;
      withHE = false;
      potVertex = vref;
    }

    std::pair<bool, Facet_handle> f = checkFaceNearer(vref, winningPoint, p);
    if (f.first) {
      // if true the retuned facet is closer to p than winningPoint
      winningPoint = f.second->point();
      withVertex = false;
      withFacet = true;
      withHE = false;
      potFacet = f.second;
    }

    std::pair<bool, Halfedge_handle> he =
        checkEdgeNearer(vref, winningPoint, p);
    if (he.first) {
      // if true the retuned he is closer to p than winningPoint
      winningPoint = he.second->point();
      withVertex = false;
      withFacet = false;
      withHE = true;
      potHE = he.second;
    }
  }

  // vertices contains all vertices that should be moved
  // furtherBasicStep contains follwing steps:
  // * add all relevant vertices to m_vertexContainer for filterChain
  // * increaseError for facet(s)
  // * setInitialAge for facet(s)
  // * increaseAge for all facets then the ones mentioned above
  std::vector<Vertex_handle> vertices;
  if (withVertex) {
    vertices = furtherBasicStep(potVertex, p);
  } else if (withFacet) {
    vertices = furtherBasicStep(potFacet, p);
  } else if (withHE) {
    vertices = furtherBasicStep(potHE, p);
  }

  if (vertices.size() == 0) {
    BOOST_LOG_TRIVIAL(error) << "No vertices to move or for coalescing.";
    updateFacetAverages();
    return;
  }

  // move all vertices towards p
  moveVertices(vertices, p, distance(winningPoint, p));

  // add distance to rolling mean
  (*acc)(distance(winningPoint, p));
  m_avgSampleToSurface = boost::accumulators::rolling_mean(*acc);

  d_winningDistance += distance(winningPoint, p);
  d_winningDistance /= m_mesh->size();

  // After vertices movement recalculate facet averages (size)
  // updateFacetAverages();

  coalescing(vertices);

  // TODO Coalscing
  // TODO FilterChain
};

void GrowingSurfaceStructure::performVertexSplit() {
  // Search for facet with hightest error
  Facet_handle winningFace = m_mesh->facets_begin();
  double highestError = winningFace->getError();
  for (auto f_it = m_mesh->facets_begin(); f_it != m_mesh->facets_end();
       ++f_it) {
    if (f_it->getError() >= highestError) {
      highestError = f_it->getError();
      winningFace = f_it;
    }
  }
  // From this face get longest edge
  Halfedge_handle halfedgeToSplit = winningFace->facet_begin();
  double longestHE = halfedgeToSplit->length();
  bool foundSplitEdge = false;

  Halfedge_around_facet_circulator hafc = winningFace->facet_begin();
  do {
    if (hafc->length() >= longestHE && !hafc->is_border_edge()) {
      longestHE = hafc->length();
      halfedgeToSplit = hafc;
      foundSplitEdge = true;
    }
  } while (++hafc != winningFace->facet_begin());

  // check for border edge
  if (halfedgeToSplit->is_border_edge()) {
    BOOST_LOG_TRIVIAL(info) << "Split edge: Chosen edge is border edge, "
                               "therefore has no incident facets.";
    return;
  }
  if (!foundSplitEdge) {
    BOOST_LOG_TRIVIAL(error) << "Split edge: Found no splitting edge on "
                                "facet with lowest approx. error.";
    return;
  }
  // Checks only if degree < 3
  if (CGAL::source(halfedgeToSplit, m_mesh->getPolyhedron())->vertex_degree() <
          3 ||
      CGAL::target(halfedgeToSplit, m_mesh->getPolyhedron())->vertex_degree() <
          3) {
    BOOST_LOG_TRIVIAL(error)
        << "Split edge: Either source, target or both vertices of splitedge "
           "have degrees below 3, should apply filter chain.";
    return;
  }
  m_vertexContainer.push_back(halfedgeToSplit->vertex());
  m_vertexContainer.push_back(halfedgeToSplit->opposite()->vertex());
  /* TRANSLATION */

  // splits the halfedge halfedgeToSplit into two halfedges inserting a new
  // vertex that is a copy of halfedgeToSplit->opposite()->vertex()
  // (source!)
  // BOOST_LOG_TRIVIAL(info) << "Split edge: " << halfedgeToSplit;
  Halfedge_handle hnew = m_mesh->getPolyhedron().split_edge(halfedgeToSplit);
  ++d_edgeSplitted;
  // BOOST_LOG_TRIVIAL(debug) << "Procede with edge split ... ";

  Vertex_handle vnew = hnew->vertex();
  Vertex_handle target = halfedgeToSplit->vertex();

  // Half the distance to target to move new vertex
  // h->vertex()->point() == source
  float xDiff = (vnew->point().x() - target->point().x()) / 2;
  float yDiff = (vnew->point().y() - target->point().y()) / 2;
  float zDiff = (vnew->point().z() - target->point().z()) / 2;

  Vector_3 translationVector = Vector_3(-xDiff, -yDiff, -zDiff);
  m_mesh->translate(vnew->point(), translationVector);

  m_vertexContainer.push_back(vnew);

  /* TRIANGULATION */
  bool foundBorder = false;
  if (!hnew->is_border()) {

    double oldErr = hnew->facet()->getError();
    Halfedge_handle ht =
        m_mesh->getPolyhedron().split_facet(hnew, hnew->next()->next());

    Facet_handle f1 = ht->facet();
    Facet_handle f2 = ht->opposite()->facet();
    f1->setError(oldErr / 2.0);
    f2->setError(oldErr / 2.0);
    setInitialAge(f1);
    setInitialAge(f2);

  } else {
    BOOST_LOG_TRIVIAL(info)
        << "Split edge: Split edge has at least one incident hole.";
    foundBorder = true;
  }
  if (!hnew->opposite()->is_border()) {
    double oldErr = hnew->opposite()->facet()->getError();

    Halfedge_handle ht = m_mesh->getPolyhedron().split_facet(
        hnew->opposite()->prev(), hnew->opposite()->next());

    Facet_handle f1 = ht->facet();
    Facet_handle f2 = ht->opposite()->facet();
    f1->setError(oldErr / 2.0);
    f2->setError(oldErr / 2.0);

    setInitialAge(f1);
    setInitialAge(f2);
  } else {
    BOOST_LOG_TRIVIAL(info)
        << "Split edge: Split edge has at least one incident hole.";
    foundBorder = true;
  }

  /* CORRECTNESS OF VALENCE */
  Vertex_handle t1 = hnew->next()->vertex();
  Vertex_handle t2 = hnew->opposite()->next()->vertex();
  std::set<Halfedge_handle, compare_heBySourceDegreeDesc> possibleHE;
  Halfedge_around_vertex_circulator hasc = vnew->vertex_begin();
  do {
    if (!hasc->is_border_edge()) {
      possibleHE.insert(hasc->opposite());
    }
  } while (++hasc != vnew->vertex_begin());

  for (Halfedge_handle h : possibleHE) {
    if (h->vertex()->vertex_degree() > 6 && vnew->vertex_degree() < 6) {
      Halfedge_handle he1 = h->next()->opposite();
      Halfedge_handle he2 = h->opposite()->prev();
      BOOST_LOG_TRIVIAL(info)
          << "Split edge: Check if valence must be corrected.";
      if ((CGAL::source(he1, m_mesh->getPolyhedron())->vertex_degree() >=
           CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree()) &&
          (CGAL::source(he1, m_mesh->getPolyhedron())->vertex_degree() > 4) &&
          !he1->is_border_edge() && he1->is_triangle() &&
          he1->opposite()->is_triangle()) {

        m_mesh->getPolyhedron().flip_edge(he1);
        Facet_handle f1 = he1->facet();
        Facet_handle f2 = he1->opposite()->facet();
        setInitialAge(f1);
        setInitialAge(f2);

      } else if ((CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree() >=
                  CGAL::source(he1, m_mesh->getPolyhedron())
                      ->vertex_degree()) &&
                 (CGAL::source(he2, m_mesh->getPolyhedron())->vertex_degree() >
                  4) &&
                 !he2->is_border_edge() && he2->is_triangle() &&
                 he2->opposite()->is_triangle()) {

        m_mesh->getPolyhedron().flip_edge(he2);
        Facet_handle f1 = he2->facet();
        Facet_handle f2 = he2->opposite()->facet();
        setInitialAge(f1);
        setInitialAge(f2);

      } else {
        BOOST_LOG_TRIVIAL(info)
            << "Split edge: No correctness of valence needed.";
      }
    }
  }
  // BOOST_LOG_TRIVIAL(info) << "Split edge: Mesh still triangulated? "
  //                         << m_mesh->getPolyhedron().is_pure_triangle();
};

void GrowingSurfaceStructure::performRemove() {
  // Search for triangle with lowest approx. error
  Facet_handle lowestErrTriangle =
      m_mesh->facets_begin(); // Facet with lowest Error
  double lowestErr = lowestErrTriangle->getError();

  for (auto it = m_mesh->facets_begin(); it != m_mesh->facets_end(); ++it) {
    if (it->getError() <= lowestErr) {
      lowestErrTriangle = it;
      lowestErr = lowestErrTriangle->getError();
    }
  }

  // Get HE whose vertices exposes the highest dot product of their normals
  Halfedge_around_facet_circulator hafc = lowestErrTriangle->facet_begin();
  Halfedge_handle halfedgeToCollapse = lowestErrTriangle->facet_begin();

  bool foundCollapsingEdge = false;
  double highestCrossproduct = 0.0;

  Halfedge_handle h1 = lowestErrTriangle->halfedge();
  Halfedge_handle h2 = lowestErrTriangle->halfedge()->next();
  Halfedge_handle h3 = lowestErrTriangle->halfedge()->next()->next();

  if ((!h1->is_border_edge() && !h2->is_border_edge() &&
       !h3->is_border_edge()) ||
      (h1->is_border_edge() ^ h2->is_border_edge() ^ h3->is_border_edge())) {
    do {
      Vector_3 n_x = CGAL::Polygon_mesh_processing::compute_vertex_normal(
          hafc->vertex(), m_mesh->getPolyhedron());
      Vector_3 n_y = CGAL::Polygon_mesh_processing::compute_vertex_normal(
          hafc->opposite()->vertex(), m_mesh->getPolyhedron());

      FT crossProduct = (n_x * n_y);
      if (crossProduct >= highestCrossproduct && crossProduct < p_maxDeltaN) {
        highestCrossproduct = crossProduct;
        halfedgeToCollapse = hafc;
        foundCollapsingEdge = true;
      }
    } while (++hafc != lowestErrTriangle->facet_begin());

    if (foundCollapsingEdge) {
      BOOST_LOG_TRIVIAL(info) << "Collapse edge: " << halfedgeToCollapse;

      halfedgeToCollapse->setDelete();
      CGAL::set_halfedgeds_items_id(m_mesh->getPolyhedron());
      std::size_t delta_f = m_mesh->size_of_facets();
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
      d_edgeCollapsed++;
      if (delta_f != 2) {
        BOOST_LOG_TRIVIAL(warning) << "Collapsing removed other than 2 facets: "
                                   << delta_f;
      }
      // Correct VALENCE
    } else {
      BOOST_LOG_TRIVIAL(info) << "Found no edge to collapse ...";
    }
  } else {
    // at least two holes incident
    BOOST_LOG_TRIVIAL(info)
        << "Triangle with lowest error is  adjacent to at least two holes";
    // m_mesh->getPolyhedron().erase_facet(h1);
  }

  // Search for oldest triangle.
  Facet_handle oldestTriangle = m_mesh->facets_begin();
  double highestAge = oldestTriangle->getAge();
  for (auto it = m_mesh->facets_begin(); it != m_mesh->facets_end(); ++it) {
    if (it->getAge() >= highestAge) {
      oldestTriangle = it;
      highestAge = oldestTriangle->getAge();
    }
  }

  // Delete oldest triangle
  if (oldestTriangle->getAge() > p_maxAge && !p_skipRemoveFacets) {
    BOOST_LOG_TRIVIAL(info) << "Delete facet: " << oldestTriangle;
    m_mesh->getPolyhedron().erase_facet(oldestTriangle->halfedge());
    d_facetsErased++;
  }
};

void GrowingSurfaceStructure::coalescing(std::vector<Vertex_handle> vertices) {

  for (auto v : vertices) {
    if (CGAL::is_border(v, m_mesh->getPolyhedron())) {

      uint v_numBorderEdges = v->numBorderEdges();
      double v_avgEdgeLength = v->avgSurrEdgeLength();

      Halfedge_around_vertex_circulator havc = v->vertex_begin();

      std::vector<Vertex_handle> t1;
      std::vector<Vertex_handle> t2;

      std::vector<Vertex_handle> vertices_line;
      std::vector<Halfedge_handle> edges;
      bool applyCoalescing = false;
      Vector_3 n_v = CGAL::Polygon_mesh_processing::compute_vertex_normal(
          v, m_mesh->getPolyhedron());

      do {
        if (havc->is_border()) {
          // clear patch vector
          vertices_line.clear();
          edges.clear();
          // walk the line
          Halfedge_handle line = havc->next();
          do {
            edges.push_back(line);
            vertices_line.push_back(line->vertex());
            line = line->next();
          } while (line != havc && line->is_border());
          //    (CGAL::is_border(line->vertex(), m_mesh->getPolyhedron())));

          if (edges.size() < p_holeSize) {
            bool fillHole = true;
            // Hole filling
            for (Vertex_handle v_opp : vertices_line) {
              if (distance(v_opp->point(), v->point()) >
                  (v_avgEdgeLength * 1.5)) {
                fillHole = false;
              }
            }

            if (fillHole) {
              std::vector<Facet_handle> patch_facets;
              std::vector<Vertex_handle> patch_vertices;
              PMP::triangulate_and_refine_hole(
                  m_mesh->getPolyhedron(), havc,
                  std::back_inserter(patch_facets),
                  std::back_inserter(patch_vertices),
                  CGAL::Polygon_mesh_processing::parameters::vertex_point_map(
                      get(CGAL::vertex_point, m_mesh->getPolyhedron()))
                      .geom_traits(Kernel()));
              for (Facet_handle f : patch_facets) {
                setInitialAge(f);
                f->setError(m_avgTriangleErr);
              }
              d_holesFilled++;
            }      // hole filling
          } else { // hole is to big for normal filling
            // for (Halfedge_handle e_v_opp : edges) {
            //   if (!e_v_opp->is_border())
            //     BOOST_LOG_TRIVIAL(error) << "No border edge.";
            //
            //   Vertex_handle v_opp = e_v_opp->vertex();
            //
            //   Vector_3 n_v_opp =
            //       CGAL::Polygon_mesh_processing::compute_vertex_normal(
            //           v_opp, m_mesh->getPolyhedron());
            //
            //   FT crossProduct = (n_v * n_v_opp);
            //
            //   double dist = distance(v_opp->point(), v->point());
            //
            //   if (v != v_opp && v_opp != havc->opposite()->vertex() &&
            //       v_opp != havc->next()->vertex() &&
            //       dist < (v_avgEdgeLength * 1.5) && (crossProduct > 0.0)) {
            //     t1.clear();
            //     t2.clear();
            //
            //     t1.push_back(v);
            //     t1.push_back(v_opp);
            //     t1.push_back(havc->opposite()->vertex());
            //
            //     t2.push_back(v);
            //     t2.push_back(e_v_opp->opposite()->vertex());
            //     t2.push_back(v_opp);
            //
            //     applyCoalescing = true;
            //   } // iff correct Opponent
            // }   // for e in edges
          } // else hole filling, size > 6
        }   // if havc is border
      } while (++havc != v->vertex_begin());

      if (applyCoalescing) {
        // CGAL::add_face seems not to work properly with facets without
        // border eg. and add_face_to_border cant be used
        // u--------s----v---------w
        // |        .   /          |
        // |        .  /           |
        // |        . /            |
        // u'--havc-t----v'--------w'

        auto f1 = CGAL::Euler::add_face(t1, m_mesh->getPolyhedron());
        auto f2 = CGAL::Euler::add_face(t2, m_mesh->getPolyhedron());
        applyCoalescing = false;

        if (f1 == m_mesh->null_face() && f2 == m_mesh->null_face()) {
          BOOST_LOG_TRIVIAL(error) << "Added two null facets";
        } else if (f1 == m_mesh->null_face() || f2 == m_mesh->null_face()) {
          if (f1 != m_mesh->null_face()) {
            setInitialAge(f1);
            f1->setError(m_avgTriangleErr);
          }
          if (f2 != m_mesh->null_face()) {
            setInitialAge(f2);
            f2->setError(m_avgTriangleErr);
          }
        } else {
          BOOST_LOG_TRIVIAL(info) << "Successfully applied coalescing.";
          setInitialAge(f1);
          setInitialAge(f2);
          f1->setError(m_avgTriangleErr);
          f2->setError(m_avgTriangleErr);
        }
      }
    } // iff v is border vertex
  }   // for v in vertices
}

std::vector<GrowingSurfaceStructure::Vertex_handle>
GrowingSurfaceStructure::furtherBasicStep(Vertex_handle &v, Point_3 p) {
  // push back to FilterContainer to apply filter on v
  m_vertexContainer.push_back(v);

  std::vector<Facet_handle> exclude;
  std::vector<Vertex_handle> vertices(1, v);
  Halfedge_around_vertex_circulator havc = v->vertex_begin();
  do {
    if (!havc->is_border()) {
      Facet_handle f = havc->facet();
      increaseErrorAndSetDistance(f, p);
      setInitialAge(f);
      exclude.push_back(f);
    }
  } while (++havc != v->vertex_begin());
  increaseAge(exclude);
  return vertices;
}

std::vector<GrowingSurfaceStructure::Vertex_handle>
GrowingSurfaceStructure::furtherBasicStep(Facet_handle &f, Point_3 p) {
  std::vector<Vertex_handle> vertices;
  if (!f->is_triangle()) {
    BOOST_LOG_TRIVIAL(error) << "Chosen facet is no triangle ... ";
    std::cout << f->facet_degree() << "\n";
    return vertices;
  }
  increaseErrorAndSetDistance(f, p);
  setInitialAge(f);
  increaseAge(std::vector<Facet_handle>(1, f));
  Halfedge_around_facet_circulator hafc = f->facet_begin();
  do {
    vertices.push_back(hafc->vertex());
    m_vertexContainer.push_back(hafc->vertex());
  } while (++hafc != f->facet_begin());
  // push vertices to moveContainer (vertices) and FilterContainer
  // (VertexContainer)
  return vertices;
}

std::vector<GrowingSurfaceStructure::Vertex_handle>
GrowingSurfaceStructure::furtherBasicStep(Halfedge_handle &he, Point_3 p) {
  std::vector<Facet_handle> exclude;
  std::vector<Vertex_handle> vertices;
  if (!he->is_border()) {
    Facet_handle f1 = he->facet();
    increaseErrorAndSetDistance(f1, p);
    setInitialAge(f1);
    exclude.push_back(f1);
  }

  if (!he->opposite()->is_border()) {
    Facet_handle f2 = he->opposite()->facet();
    increaseErrorAndSetDistance(f2, p);
    setInitialAge(f2);
    exclude.push_back(f2);
  }

  // push vertices to moveContainer (vertices) and FilterContainer
  // (VertexContainer)
  vertices.push_back(he->vertex());
  m_vertexContainer.push_back(he->vertex());
  vertices.push_back(he->opposite()->vertex());
  m_vertexContainer.push_back(he->opposite()->vertex());

  increaseAge(exclude);
  return vertices;
}

void GrowingSurfaceStructure::increaseErrorAndSetDistance(Facet_handle &f,
                                                          Point_3 p) {
  double err = (distance(f->point(), p) + f->getError()) / m_k;
  f->setError(err);
}

void GrowingSurfaceStructure::setInitialAge(Facet_handle &f) {
  double newAge = (f->face_area() / m_avgTriangleSize);
  f->setAge(newAge);
}

void GrowingSurfaceStructure::updateFacetAverages() {
  m_avgTriangleSize = 0.0;
  m_avgTriangleErr = 0.0;
  for (auto f = m_mesh->facets_begin(); f != m_mesh->facets_end(); ++f) {
    m_avgTriangleSize += f->face_area();
    m_avgTriangleErr += f->getError();
  }
  m_avgTriangleSize /= m_mesh->size_of_facets();
  m_avgTriangleErr /= m_mesh->size_of_facets();
}

void GrowingSurfaceStructure::increaseAge(std::vector<Facet_handle> exclude) {
  // Calcualte beta
  uint n = p_allowedMiss * m_mesh->size_of_facets();
  m_beta = pow(p_maxAge, (1.0 / n)) - 1.0;
  for (auto f = m_mesh->facets_begin(); f != m_mesh->facets_end(); ++f) {
    if (std::find(exclude.begin(), exclude.end(), f) == exclude.end()) {
      // cannot find in exlude, procede with increasing age of all triangles
      // with following formula:
      //  Δf_age = β * f_age
      f->setAge(f->getAge() + (f->getAge() * m_beta));
    }
  }
}

void GrowingSurfaceStructure::moveVertices(std::vector<Vertex_handle> vertices,
                                           Point_3 p, double sDistance) {
  Vertex_handle singleNearestVertex = vertices[0];
  for (auto v : vertices) {
    if (distance(v->point(), p) <= distance(singleNearestVertex->point(), p)) {
      singleNearestVertex = v;
    }
  }

  if (sDistance / m_avgSampleToSurface > p_limitSkip) {
    if (sDistance / m_avgSampleToSurface > p_limitSingle) {
      Vector_3 moveVec = (singleNearestVertex->point() - p);
      moveVec = moveVec * p_learningRate;
      m_mesh->translate(singleNearestVertex->point(), -moveVec);
      //   neighborSmoothing(singleNearestVertex, p,
      //                     std::vector<Vertex_handle>(1,
      //                     singleNearestVertex));
    } else {
      for (auto v : vertices) {
        Vector_3 moveVec = (v->point() - p);
        moveVec = moveVec * p_learningRate;
        m_mesh->translate(v->point(), -moveVec);
        neighborSmoothing(v, p, vertices);
      }
    }
  }
}

void GrowingSurfaceStructure::neighborSmoothing(
    Vertex_handle &v, Point_3 p, std::vector<Vertex_handle> exclude) {
  Halfedge_around_vertex_circulator hasc = v->vertex_begin();
  do {
    Vertex_handle n = hasc->opposite()->vertex();
    if (std::find(exclude.begin(), exclude.end(), n) == exclude.end()) {
      moveNeighbor(n, p);
    }
  } while (++hasc != v->vertex_begin());
}

void GrowingSurfaceStructure::moveNeighbor(Vertex_handle &v, Point_3 p) {
  float x = p_neighborLearningRate * (v->point().x() - p.x());
  float y = p_neighborLearningRate * (v->point().y() - p.y());
  float z = p_neighborLearningRate * (v->point().z() - p.z());
  m_mesh->translate(v->point(), Vector_3(-x, -y, -z));
}

void GrowingSurfaceStructure::refreshSearchTree() {
  delete tree;
  delete tr_dist;
  vppmap = CGAL::get(CGAL::vertex_point, m_mesh->getPolyhedron());
  tree = new Tree(CGAL::vertices(m_mesh->getPolyhedron()).begin(),
                  CGAL::vertices(m_mesh->getPolyhedron()).end(), Splitter(),
                  Traits(vppmap));
  tr_dist = new Distance(vppmap);
}

GrowingSurfaceStructure::FT GrowingSurfaceStructure::distance(Point_3 p1,
                                                              Point_3 p2) {
  return std::sqrt(std::pow((p1.x() - p2.x()), 2) +
                   std::pow((p1.y() - p2.y()), 2) +
                   std::pow(p1.z() - p2.z(), 2));
}

void GrowingSurfaceStructure::initializeTetrahedron() {
  Point_3 center = CGAL::centroid(m_point3Vec.begin(), m_point3Vec.end(),
                                  CGAL::Dimension_tag<0>());
  CGAL::Bbox_3 boundingBox =
      CGAL::bbox_3(m_point3Vec.begin(), m_point3Vec.end());
  /*
   * Get min and max vertex from BoundingBox and calculate the x/y/z
   * differences.
   * Half them and weight by 1-boxFactor parameter to get the positions for
   * the
   * new shrinked bounding box (inital mesh).
   * minx         maxx
   * O-->-------<---O
   */
  Point_3 min(boundingBox.min(0), boundingBox.min(1), boundingBox.min(2));
  Point_3 max(boundingBox.max(0), boundingBox.max(1), boundingBox.max(2));

  float xdiff, ydiff, zdiff;
  xdiff = (max.x() - min.x()) / 2;
  ydiff = (max.y() - min.y()) / 2;
  zdiff = (max.z() - min.z()) / 2;

  xdiff *= (1.0 - p_scale);
  ydiff *= (1.0 - p_scale);
  zdiff *= (1.0 - p_scale);

  BOOST_LOG_TRIVIAL(debug) << "Weighted halved differences: " << xdiff << "/"
                           << ydiff << "/" << zdiff;

  // New Vector positions
  float minx, miny, minz, maxx, maxy, maxz;
  minx = min.x() + xdiff;
  miny = min.y() + ydiff;
  minz = min.z() + zdiff;
  maxx = max.x() - xdiff;
  maxy = max.y() - ydiff;
  maxz = max.z() - zdiff;
  BOOST_LOG_TRIVIAL(debug) << "Min. " << minx << "/" << miny << "/" << minz;
  BOOST_LOG_TRIVIAL(debug) << "Max. " << maxx << "/" << maxy << "/" << maxz;

  Point_3 top(center.x(), maxy, center.z());
  Point_3 left(minx, miny, minz);
  Point_3 right(maxx, miny, minz);
  Point_3 back(center.x(), miny, maxz);
  m_mesh->getPolyhedron().make_tetrahedron(top, left, right, back);
  // TODO Slightly differ points
};

GrowingSurfaceStructure::Point_3 GrowingSurfaceStructure::getRandomPoint() {
  return m_point3Vec[m_random.get_int(0, m_pointNumber)];
};

std::pair<bool, GrowingSurfaceStructure::Facet_handle>
GrowingSurfaceStructure::checkFaceNearer(Vertex_handle &v, Point_3 w_p,
                                         Point_3 p) {
  Halfedge_around_vertex_circulator havc = v->vertex_begin();
  Facet_handle winningFacet = m_mesh->facets_begin();
  bool isNearer = false;

  do {
    if (!havc->is_border()) {

      Facet_handle f = havc->facet();
      Point_3 f_p = f->point();
      if (CGAL::has_larger_distance_to_point(p, w_p, f_p)) {
        // indicates that w_p has a larger distance to o than f_p and
        // therefore
        // f_p is nearer to  sample point p
        isNearer = true;
        winningFacet = f;
      }
    }
  } while (++havc != v->vertex_begin());
  return std::pair<bool, Facet_handle>(isNearer, winningFacet);
}

std::pair<bool, GrowingSurfaceStructure::Halfedge_handle>
GrowingSurfaceStructure::checkEdgeNearer(Vertex_handle &v, Point_3 w_p,
                                         Point_3 p) {
  // Circulate HE around winner vertex
  bool isNearer = false;
  Halfedge_handle winningHE = m_mesh->halfedges_begin();

  Halfedge_around_vertex_circulator havc = v->vertex_begin();
  do {
    if (CGAL::has_larger_distance_to_point(p, w_p, havc->point())) {
      // indicates that w_p has a larger distance to o than f_p and
      // therefore
      // f_p is nearer to  sample point p
      isNearer = true;
      winningHE = havc;
    }
  } while (++havc != v->vertex_begin());
  return std::pair<bool, Halfedge_handle>(isNearer, winningHE);
}

void GrowingSurfaceStructure::printInfo() {
  std::cout << "\n";
  std::cout << "++++++++ Info ++++++++" << std::endl;
  std::cout << "Runtime (hh::mm::ss::ms): " << d_runtime << std::endl;
  std::cout << "Number of final facets: " << m_mesh->size_of_facets() << "\n";
  std::cout << "Number of final vertices: " << m_mesh->size() << "\n";
  std::cout << "Number of final halfedges: " << m_mesh->size_of_halfedges()
            << "\n";

  std::cout << "Holes filled with PMP: " << d_holesFilled << "\n";
  std::cout << "Edges collapsed with SMS: " << d_edgeCollapsed << "\n";
  std::cout << "Edges splitted: " << d_edgeSplitted << "\n";
  std::cout << "Facets erased: " << d_facetsErased << "\n";
  std::cout << "Rolling mean average sample to surface distance: "
            << boost::accumulators::rolling_mean(*acc) << "\n";
  if (!p_skipSTSCalculation) {
    std::cout << "Calcualte exact sample-to-surface distance ... please wait."
              << "\n";
    boost::progress_display show_progress(m_pointNumber);
    m_avgSampleToSurface = 0.0;
    for (auto iter = m_point3Vec.begin(); iter != m_point3Vec.end(); ++iter) {
      ++show_progress;
      double oldDist = std::numeric_limits<double>::max();
      for (auto f = m_mesh->facets_begin(); f != m_mesh->facets_end(); ++f) {
        Vector_3 dist = f->point() - (*iter);
        double euclidianDistance = std::sqrt(dist.squared_length());
        if (euclidianDistance < oldDist) {
          oldDist = euclidianDistance;
        }
      }
      m_avgSampleToSurface += oldDist;
    }
    m_avgSampleToSurface /= m_pointNumber;
    std::cout << "Actual sample-to-surface distance: " << m_avgSampleToSurface
              << "\n \n";
  }
}
GrowingSurfaceStructure::~GrowingSurfaceStructure() {
  delete tree;
  delete tr_dist;
  delete m_filterChain;
  delete acc;
  datafile.close();
}
} //namespace lvr2
