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
// #define CGAL_EIGEN3_ENABLED

//#include <lvr/geometry/CGALPolyhedronMesh.hpp>
#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/ColorVertex.hpp>
#include <lvr2/geometry/Normal.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/reconstruction/gcs/GCS.hpp>
#include <lvr2/reconstruction/gcs/GrowingSurfaceStructure.hpp>

#include <CGAL/Polyhedron_3.h>
#include <CGAL/Simple_cartesian.h>

using namespace lvr2;

typedef CGALPolyhedronMesh<ColorVertex<float, unsigned char>, Normal<float>,
                           CGAL::Exact_predicates_inexact_constructions_kernel,
                           lvr2::GrowingSurfaceStructure::GrowingItems>
    GrowingMesh;
typedef HalfEdgeMesh<ColorVertex<float, unsigned char>> Mesh;

int main(int argc, char const *argv[]) {

  if (argc < 3) {
    BOOST_LOG_TRIVIAL(fatal) << "Missing command-line argument.";
    exit(-1);
  }

  ModelPtr model;

  try {
    model = ModelFactory::readModel(argv[1]);
  } catch (...) {
    BOOST_LOG_TRIVIAL(error) << "Error reading model from input file.";
  }

  if (!model) {
    BOOST_LOG_TRIVIAL(fatal) << "Unable to parse model.";
    exit(-1);
  }

  PointBufferPtr p_loader;
  p_loader = model->m_pointCloud;

  ModelPtr pn(new Model);
  pn->m_pointCloud = p_loader;

  // cGCS *recon = new cGCS(p_loader, argv[2]);
  GrowingSurfaceStructure gss(p_loader, argv[2]);

  GrowingMesh mesh;
  gss.getMesh(mesh);

  mesh.finalize();

  ModelPtr m(new Model(mesh.meshBuffer()));

  ModelFactory::saveModel(m, "gss.ply");
}
