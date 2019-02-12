/*
 * MainGS.cpp
 *
 *  Created on: somewhen.02.2019
 *      Author: Patrick Hoffmann (pahoffmann@uos.de)
 */


/// Old includes, to be evaluated
/*#include <CGAL/Polyhedron_3.h>
#include <lvr2/geometry/BoundingBox.hpp>
#include <lvr2/geometry/CGALPolyhedronMesh.hpp>
#include <lvr2/reconstruction/gcs/GCS.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/algorithm/FinalizeAlgorithms.hpp>*/

/// New includes, to be evaluated, which we actually need

#include "OptionsGS.hpp"



using namespace lvr2;
/*typedef GCS<ColorVertex<float, unsigned char>, Normal<float>> cGCS;
typedef CGAL::Simple_cartesian<double> SimpleCartesian;*/

#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/gs2/GrowingCellStructure.hpp>



/**
 * Extended Vertex, which now includes the signal counter, TODO: make it work...how to inherit this wagshit?
 * @tparam CoordT
 */
/*template <typename CoordT>
struct GCSVector : BaseVector<CoordT>{
public:

    GCSVector() : BaseVector(x,y,z){
        this->signal_counter = 0;
    }
    GCSVector(const CoordT &x, const CoordT &y, const CoordT &z)
            : BaseVector(x,y,z)
    {
        this->signal_counter = 0;
    }

    void incrementSC() {
        this->signal_counter++;
    }

    int getSC() const{
        return this->signal_counter;
    }

    void setSC(int new_sc) {
        this->signal_counter = new_sc;
    }

    int signal_counter;
};*/

//TODO: dont use BaseVector
using Vec = BaseVector<float>;

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const gs_reconstruction::Options &options, PointBufferPtr buffer){
    // Create a point cloud manager
    string pcm_name = options.getPcm();
    PointsetSurfacePtr<Vec> surface;

    // Create point set surface object
    if(pcm_name == "PCL")
    {
        cout << timestamp << "Using PCL as point cloud manager is not implemented yet!" << endl;
        panic_unimplemented("PCL as point cloud manager");
    }
    else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
    {
        surface = make_shared<AdaptiveKSearchSurface<BaseVecT>>(
                buffer,
                pcm_name,
                options.getKn(),
                options.getKi(),
                options.getKd(),
                1,
                ""
        );
    }
    else
    {
        cout << timestamp << "Unable to create PointCloudManager." << endl;
        cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
        return nullptr;
    }

    // Set search options for normal estimation and distance evaluation
    surface->setKd(options.getKd());
    surface->setKi(options.getKi());
    surface->setKn(options.getKn());

    //calc normals if there are none, TODO: Seg-Fault beheben, woher kommt er?
    if(!buffer->hasNormals()){
        surface->calculateSurfaceNormals();
    }
    return surface;
}


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

    // Create a point cloud manager
    string pcm_name = options.getPcm();
    auto surface = loadPointCloud<Vec>(options, buffer);
    if (!surface)
    {
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
    }

    //TODO: check, whether the centroid (needed for mesh positioning) is usable, else do it by myself..
    //TODO: generate possibility to call GCS and GSS with one operation (create insantance, call "getMesh" or similar)
    //      -> getMesh returns Pointer to a Mesh


    HalfEdgeMesh<Vec> mesh;
    GrowingCellStructure<Vec, Normal<float>> gcs(surface);

    //set gcs variables
    gcs.setRuntime(options.getRuntime());
    gcs.setBasicSteps(options.getBasicSteps());
    gcs.setBoxFactor(options.getBoxFactor());
    gcs.setAllowMiss(options.getAllowMiss());
    gcs.setCollapseThreshold(options.getCollapseThreshold());
    gcs.setDecreaseFactor(options.getDecreaseFactor());
    gcs.setDeleteLongEdgesFactor(options.getDeleteLongEdgesFactor());
    gcs.setFilterChain(options.isFilterChain());
    gcs.setLearningRate(options.getLearningRate());
    gcs.setNeighborLearningRate(options.getNeighborLearningRate());
    gcs.setNumSplits(options.getNumSplits());
    gcs.setWithCollapse(options.getWithCollapse());

    std::cout << "Test: " << gcs.getBasicSteps() << std::endl;
    gcs.getInitialMesh(mesh);

    SimpleFinalizer<Vec> fin;
    MeshBufferPtr res = fin.apply(mesh);

    ModelPtr m( new Model( res ) );

    cout << timestamp << "Saving mesh." << endl;
    ModelFactory::saveModel( m, "triangle_init_mesh.ply");

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
