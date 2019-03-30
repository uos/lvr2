/*
 * MainGS.cpp
 *
 *  Created on: somewhen.02.2019
 *      Author: Patrick Hoffmann (pahoffmann@uos.de)
 */


/// New includes, to be evaluated, which we actually need

#include "OptionsGS.hpp"


#include <lvr2/geometry/HalfEdgeMesh.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/reconstruction/PointsetSurface.hpp>
#include <lvr2/io/PointBuffer.hpp>
#include <lvr2/io/MeshBuffer.hpp>
#include <lvr2/io/ModelFactory.hpp>
#include <lvr2/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr2/reconstruction/gs2/GrowingCellStructure.hpp>
#include <lvr2/algorithm/CleanupAlgorithms.hpp>

using namespace lvr2;

/**
 * Extended Vertex, which now includes the signal counter
 * @tparam CoordT
 */

template<typename CoordT>
struct GCSVector : public BaseVector<CoordT>{
    GCSVector() : BaseVector<CoordT>(), signal_counter(0) {}
    GCSVector(const CoordT &x, const CoordT &y, const CoordT &z)
            : BaseVector<CoordT>(x,y,z), signal_counter(0)
    {}

    GCSVector(const GCSVector& o) : BaseVector<CoordT>(o.x,o.y,o.z), signal_counter(o.signal_counter)
    {
    }
    void incSC(){this->signal_counter++;}
    int getSC(){return this->signal_counter;}
    void setSC(int new_sc){this->signal_counter = new_sc;}

    float signal_counter;

    GCSVector<CoordT> cross(const GCSVector &other) const
    {
        BaseVector<CoordT> cp = BaseVector<CoordT>::cross(other);
        return GCSVector<CoordT>(cp.x, cp.y, cp.z);
    }

    GCSVector operator*(const CoordT &scale) const
    {
        return GCSVector(*this) *= scale;
    }

    GCSVector operator/(const CoordT &scale) const
    {
        return GCSVector(*this) /= scale;
    }

    GCSVector& operator*=(const CoordT &scale)
    {
        BaseVector<CoordT>::operator *= (scale);
        //return static_cast<GCSVector&>(BaseVector::operator *= (scale));

        return *this;
    }

    GCSVector& operator/=(const CoordT &scale)
    {
        BaseVector<CoordT>::operator /= (scale);

        return *this;
    }

    GCSVector<CoordT> operator+(const GCSVector &other) const
    {
        return GCSVector(*this) += other;
    }

    GCSVector<CoordT> operator-(const GCSVector &other) const
    {
        return GCSVector(*this) -= other;
    }

    GCSVector<CoordT>& operator+=(const GCSVector<CoordT> &other)
    {
        BaseVector<CoordT>::operator += (other);

        return *this;
    }

    GCSVector<CoordT>& operator-=(const GCSVector<CoordT> &other)
    {
        BaseVector<CoordT>::operator -= (other);

        return *this;
    }

    bool operator==(const GCSVector<CoordT> &other) const
    {
        return BaseVector<CoordT>::operator == (other);
    }

    bool operator!=(const GCSVector<CoordT> &other) const
    {
        return BaseVector<CoordT>::operator != (other);
    }

    CoordT operator*(const GCSVector<CoordT> &other) const
    {
        return BaseVector<CoordT>::dot(other);
    }
};

using Vec = GCSVector<float>;

template <typename BaseVecT>
PointsetSurfacePtr<BaseVecT> loadPointCloud(const gs_reconstruction::Options &options, PointBufferPtr buffer){
    // Create a point cloud manager
    string pcm_name = options.getPcm();
    PointsetSurfacePtr<BaseVecT> surface;

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

    //calc normals if there are none
    if(!buffer->hasNormals()){
        //surface->calculateSurfaceNormals();
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
    if (!surface) {
        cout << "Failed to create pointcloud. Exiting." << endl;
        return EXIT_FAILURE;
    }

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
    gcs.setInterior(options.isInterior());

    gcs.getMesh(mesh);
    naiveFillSmallHoles(mesh, 10, true);
    SimpleFinalizer<Vec> fin;
    MeshBufferPtr res = fin.apply(mesh);

    ModelPtr m( new Model( res ) );

    cout << timestamp << "Saving mesh." << endl;
    ModelFactory::saveModel( m, "triangle_init_mesh.ply");

    return 0;

}
