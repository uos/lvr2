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

#include "OptionsGS.hpp"



//using namespace lvr2;
/*typedef GCS<ColorVertex<float, unsigned char>, Normal<float>> cGCS;
typedef CGAL::Simple_cartesian<double> SimpleCartesian;*/

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>

using Vec = BaseVector<float>;

/*template <typename BaseVecT>
PointsetSurface<BaseVecT> loadPointCloud(const gs_reconstruction::Options &options){

}*/


int main(int argc, char **argv) {

    gs_reconstruction::Options options(argc, argv);

    // if one of the needed parameters is missing,
    if(options.printUsage()){
        return EXIT_SUCCESS;
    }

    std::cout << options << std::endl;

    //try to parse the model
    ModelPtr model = ModelFactory::readModel(options.getInputFileName());

    // did model parse succeed
    if (!model)
    {
        cout << timestamp << "IO Error: Unable to parse " << options.getInputFileName() << endl;
        return EXIT_FAILURE;
    }

    PointBufferPtr buffer = model->m_pointCloud;

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
