/*
 * Main.cpp
 *
 *  Created on: 24.02.2016
 *      Author: Henning Strueber (hstruebe@uos.de)
 */


/// Old includes, to be evaluated
/*#include <CGAL/Polyhedron_3.h>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/CGALPolyhedronMesh.hpp>
//#include <lvr2/reconstruction/FastReconstruction.hpp>
#include <lvr2/reconstruction/gcs/GCS.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>*/

/// New includes, to be evaluated, which we actually need

#include "OptionsGSS.hpp"



//using namespace lvr2;
/*typedef GCS<ColorVertex<float, unsigned char>, Normal<float>> cGCS;
typedef CGAL::Simple_cartesian<double> SimpleCartesian;*/

using namespace gs_reconstruction;

int main(int argc, char **argv) {

    Options options(argc, argv);

    // if one of the needed parameters is missing,
    if(options.printUsage()){
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    return 0;




















/*  BOOST_LOG_TRIVIAL(info) << "Using GrowingCellStructures Reconstruction.";

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

  recon->getMesh(mesh);

  //mesh.finalize();

  SimpleFinalizer<ColorVertex<float, unsigned int>> fin;

  MeshBufferPtr res = fin.apply(mesh);


  ModelPtr m(new Model());

  ModelFactory::saveModel(m, "gcs_tri_mesh.ply");

  delete recon;*/

}
