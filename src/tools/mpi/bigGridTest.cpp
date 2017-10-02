#include <iostream>
#include <string>
#include <lvr/io/Model.hpp>
#include "BigGrid.hpp"
#include "lvr/io/DataStruct.hpp"
#include "lvr/io/PointBuffer.hpp"
#include "lvr/io/Model.hpp"
#include "lvr/io/PLYIO.hpp"
#include "lvr/geometry/BoundingBox.hpp"
#include "BigGridKdTree.hpp"
#include <fstream>
#include <sstream>
#include <lvr/io/Timestamp.hpp>
#include <lvr/reconstruction/PointsetSurface.hpp>
#include <lvr/geometry/ColorVertex.hpp>
#include <lvr/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr/geometry/ColorVertex.hpp>
#include <lvr/geometry/Normal.hpp>
#include <lvr/reconstruction/HashGrid.hpp>
#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/reconstruction/PointsetGrid.hpp>
#include <lvr/geometry/QuadricVertexCosts.hpp>
#include <lvr/reconstruction/FastBox.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include "lvr/reconstruction/QueryPoint.hpp"
#include <fstream>
#include <sstream>
#include "LargeScaleOptions.hpp"


using std::cout;
using std::endl;
using namespace lvr;

#if defined CUDA_FOUND
    #define GPU_FOUND

    #include <lvr/reconstruction/cuda/CudaSurface.hpp>

    typedef CudaSurface GpuSurface;
#elif defined OPENCL_FOUND
    #define GPU_FOUND

    #include <lvr/reconstruction/opencl/ClSurface.hpp>
    typedef ClSurface GpuSurface;
#endif


typedef lvr::PointsetSurface<lvr::ColorVertex<float, unsigned char> > psSurface;
typedef lvr::AdaptiveKSearchSurface<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> > akSurface;

int main(int argc, char** argv)
{

    LargeScaleOptions::Options options(argc, argv);

    string filePath = options.getInputFileName();
    float voxelsize = options.getVoxelsize();
    float scale = options.getScaling();
    std::vector<float> flipPoint = options.getFlippoint();
    cout << lvr::timestamp << "Starting grid" << endl;
    BigGrid bg(filePath, voxelsize, scale);
    cout << lvr::timestamp << "grid finished " << endl;
    lvr::BoundingBox<lvr::Vertexf> bb = bg.getBB();
    cout << bb << endl;


    //lvr::floatArr points = bg.getPointCloud(numPoints);

    cout << lvr::timestamp << "making tree" << endl;
    BigGridKdTree gridKd(bg.getBB(),options.getNodeSize(),&bg, voxelsize);
    gridKd.insert(bg.pointSize(),bg.getBB().getCentroid());
    cout << lvr::timestamp << "finished tree" << endl;

    std::cout << lvr::timestamp << "got: " << gridKd.getLeafs().size() << " leafs, saving leafs" << std::endl;



    BoundingBox<ColorVertex<float,unsigned char> > cbb(bb.getMin().x, bb.getMin().y, bb.getMin().z,
                                                       bb.getMax().x, bb.getMax().y, bb.getMax().z);

    vector<string> grid_files;

    for(int i = 0 ; i < gridKd.getLeafs().size() ; i++)
    {
        size_t numPoints;

        //todo: okay?
        lvr::floatArr points = bg.points(gridKd.getLeafs()[i]->getBB().getMin().x, gridKd.getLeafs()[i]->getBB().getMin().y, gridKd.getLeafs()[i]->getBB().getMin().z ,
                                         gridKd.getLeafs()[i]->getBB().getMax().x, gridKd.getLeafs()[i]->getBB().getMax().y, gridKd.getLeafs()[i]->getBB().getMax().z,numPoints);

        //std::cout << "i: " << std::endl << bb << std::endl << "got : " << numPoints << std::endl;
        if(numPoints<=50) continue;

        BoundingBox<ColorVertex<float,unsigned char> > gridbb(gridKd.getLeafs()[i]->getBB().getMin().x, gridKd.getLeafs()[i]->getBB().getMin().y, gridKd.getLeafs()[i]->getBB().getMin().z,
                                                              gridKd.getLeafs()[i]->getBB().getMax().x, gridKd.getLeafs()[i]->getBB().getMax().y, gridKd.getLeafs()[i]->getBB().getMax().z);
        cout << "grid: " << i << "/" << gridKd.getLeafs().size()-1 << endl;
        cout << "grid has " << numPoints << " points" << endl;
        cout << "kn=" << options.getKn() << endl;
        cout << "ki=" << options.getKi() << endl;
        cout << "kd=" << options.getKd() << endl;
        cout << gridbb << endl;
        lvr::PointBufferPtr p_loader(new lvr::PointBuffer);
        p_loader->setPointArray(points, numPoints);

        if(bg.hasNormals())
        {

            size_t numNormals;
            lvr::floatArr normals = bg.normals(gridKd.getLeafs()[i]->getBB().getMin().x, gridKd.getLeafs()[i]->getBB().getMin().y, gridKd.getLeafs()[i]->getBB().getMin().z ,
                                              gridKd.getLeafs()[i]->getBB().getMax().x, gridKd.getLeafs()[i]->getBB().getMax().y, gridKd.getLeafs()[i]->getBB().getMax().z,numNormals);


            p_loader->setPointNormalArray(normals, numNormals);
        } else {
            #ifdef GPU_FOUND
            if( options.useGPU() )
            {

                floatArr normals = floatArr(new float[ numPoints * 3 ]);
                cout << timestamp << "Constructing kd-tree..." << endl;
                GpuSurface gpu_surface(points, numPoints);
                cout << timestamp << "Finished kd-tree construction." << endl;
                gpu_surface.setKn( options.getKn() );
                gpu_surface.setKi( options.getKi() );
                gpu_surface.setFlippoint(flipPoint[0], flipPoint[1], flipPoint[2]);
                cout << timestamp << "Start Normal Calculation..." << endl;
                gpu_surface.calculateNormals();
                gpu_surface.getNormals(normals);
                cout << timestamp << "Finished Normal Calculation. " << endl;
                p_loader->setPointNormalArray(normals, numPoints);
                gpu_surface.freeGPU();
            }
            #else
                cout << "ERROR: OpenCl not found" << endl;
                exit(-1);
            #endif
        }

        psSurface::Ptr surface = psSurface::Ptr(new akSurface(
            p_loader,
            "FLANN",
            options.getKn(),
            options.getKi(),
            options.getKd(),
            options.useRansac()

//                p_loader, pcm_name,
//                options.getKn(),
//                options.getKi(),
//                options.getKd(),
//                options.useRansac(),
//                options.getScanPoseFile(),
//                center

        ));



        if(! bg.hasNormals() && !options.useGPU()) {
            surface->calculateSurfaceNormals();
        }

        lvr::GridBase* grid;
        lvr::FastReconstructionBase<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> >* reconstruction;

        grid = new PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(voxelsize, surface,gridbb , true, options.extrude());
        //FastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
        PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
        ps_grid->setBB(gridbb);
        ps_grid->calcIndices();
        ps_grid->calcDistanceValues();

        std::stringstream ss2;
        ss2 << "testgrid-" << i << ".ser";
        ps_grid->saveCells(ss2.str());
        grid_files.push_back(ss2.str());
        delete ps_grid;



    }

    cout << lvr::timestamp << "finished" << endl;


    auto vmax = cbb.getMax();
    auto vmin = cbb.getMin();
    vmin.x-=voxelsize*2;
    vmin.y-=voxelsize*2;
    vmin.z-=voxelsize*2;
    vmax.x+=voxelsize*2;
    vmax.y+=voxelsize*2;
    vmax.z+=voxelsize*2;
    cbb.expand(vmin);
    cbb.expand(vmax);
    HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >
            hg(grid_files, cbb, voxelsize);



    //hg.saveGrid("largeScale.grid");





    lvr::FastReconstructionBase<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> >* reconstruction;
    reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(&hg);
    HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh;
    if(options.getDepth())
    {
        mesh.setDepth(options.getDepth());
    }
    reconstruction->getMesh(mesh);
    if(options.getDanglingArtifacts())
    {
        mesh.removeDanglingArtifacts(options.getDanglingArtifacts());
    }
    mesh.cleanContours(options.getCleanContourIterations());
    mesh.setClassifier(options.getClassifier());
    mesh.getClassifier().setMinRegionSize(options.getSmallRegionThreshold());
    if(options.optimizePlanes())
    {
        mesh.optimizePlanes(options.getPlaneIterations(),
                            options.getNormalThreshold(),
                            options.getMinPlaneSize(),
                            options.getSmallRegionThreshold(),
                            true);

        mesh.fillHoles(options.getFillHoles());
        mesh.optimizePlaneIntersections();
        mesh.restorePlanes(options.getMinPlaneSize());

        if(options.getNumEdgeCollapses())
        {
            QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> > c = QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> >(true);
            mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
        }
    }
    else if(options.clusterPlanes())
    {
        mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
        mesh.fillHoles(options.getFillHoles());
    }


    if ( options.retesselate() )
    {
        mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
    }
    else
    {
        mesh.finalize();
    }
    ModelPtr m( new Model( mesh.meshBuffer() ) );
    ModelFactory::saveModel( m, "largeScale.ply");
    delete reconstruction;
//
//    lvr::PointBufferPtr pb2(new lvr::PointBuffer);
//    pb2->setPointArray(points2, numPoints2);
//    lvr::ModelPtr m2( new lvr::Model(pb2));
//    lvr::PLYIO io2;
//    io.save(m2,"testPoints2.ply");




    return 0;
}
