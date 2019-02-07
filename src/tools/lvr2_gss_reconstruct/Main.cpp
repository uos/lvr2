/*
 * Main.cpp
 *
 *  Created on: 24.02.2016
 *      Author: Henning Strueber (hstruebe@uos.de)
 */

#include <CGAL/Polyhedron_3.h>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/CGALPolyhedronMesh.hpp>
//#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/gcs/GCS.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>

using namespace lvr2;
typedef GCS<ColorVertex<float, unsigned char>, Normal<float>> cGCS;
typedef CGAL::Simple_cartesian<double> SimpleCartesian;

// template <typename VertexT, typename NormalT, typename Kernel>
// class Henning : public CGAL::Polyhedron_3<Kernel>,
//                 public BaseMesh<VertexT, NormalT> {
// public:
//   Henning() : CGAL::Polyhedron_3<Kernel>(){};
//
//   virtual void addVertex(VertexT v){};
//
//   virtual void addNormal(NormalT n){};
//
//   virtual void addTriangle(uint a, uint b, uint c){};
//
//   virtual void flipEdge(uint v1, uint v2){};
//
//   virtual void finalize(){};
//
//   virtual size_t meshSize(){};
// };

int main(int argc, char **argv) {
  BOOST_LOG_TRIVIAL(info) << "Using GrowingCellStructures Reconstruction.";

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
  cGCS *recon = new cGCS(p_loader, argv[2]);
  // Henning<ColorVertex<float, unsigned char>, Normal<float>> *h = new
  // Henning();

  cGCS::PolyhedronMesh mesh;
  // Henning<ColorVertex<float, unsigned char>, Normal<float>, SimpleCartesian>
  // h;
  // h.make_tetrahedron();
  // FastReconstruction<ColorVertex<float, unsigned char>, Normal<float>>
  // fr;
  // CGALPolyhedronMesh<ColorVertex<float, unsigned char>, Normal<float>>
  // mesh;

  recon->getMesh(mesh);

  //mesh.finalize();

  SimpleFinalizer<ColorVertex<float, unsigned int>> fin;

  MeshBufferPtr res = fin.apply(mesh);


  ModelPtr m(new Model());

  ModelFactory::saveModel(m, "gcs_tri_mesh.ply");

  delete recon;
}
