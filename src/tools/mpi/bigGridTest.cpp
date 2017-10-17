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
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/algorithm/string/replace.hpp>
#include "LineReader.hpp"
#include <random>
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
//make sure to include the random number generators and such

//    std::random_device seeder;
//    std::mt19937 engine(seeder());
//    std::uniform_real_distribution<float> dist(10, 10.05);
//    std::random_device seeder2;
//    std::mt19937 engine2(seeder());
//    std::uniform_real_distribution<float> dist2(0, 10);
//    ofstream testofs("plane.pts");
//    for(float x = 0; x < 10; x+=0.05)
//    {
//        for(float y = 0 ; y < 10 ; y+=0.05)
//        {
//            testofs << dist2(engine2) << " " << dist2(engine2) << " " << dist(engine) << endl;
//        }
//    }


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

    vector<BoundingBox<Vertexf > > partitionBoxes;
    //lvr::floatArr points = bg.getPointCloud(numPoints);

    cout << lvr::timestamp << "making tree" << endl;
    float volumenSize = (float)(options.getVolumenSize()); // 10 x 10 x 10 voxel
    if(volumenSize > 0)
    {
//        BoundingBox<Vertexf> volumenBB = bg.getBB();
//        float xdiff = fabs(fmod(volumenBB.getXSize(), volumenSize*voxelsize ));
//        float ydiff = fabs(fmod(volumenBB.getYSize(), volumenSize*voxelsize ));
//        float zdiff = fabs(fmod(volumenBB.getZSize(), volumenSize*voxelsize ));
//        Vertexf new_max = volumenBB.getMax();
//        new_max[0]+=xdiff;
//        new_max[1]+=ydiff;
//        new_max[2]+=zdiff;
//        volumenBB.expand(new_max);

        float current_minx = bg.getBB().getMin()[0];


        while(current_minx < bg.getBB().getMax()[0])
        {
            float current_miny = bg.getBB().getMin()[1];
            while(current_miny < bg.getBB().getMax()[1])
            {
                float current_minz = bg.getBB().getMin()[2];
                while(current_minz < bg.getBB().getMax()[2])
                {
                    BoundingBox<Vertexf > partBB(
                            current_minx,
                            current_miny,
                            current_minz,
                            current_minx+volumenSize*voxelsize,
                            current_miny+volumenSize*voxelsize,
                            current_minz+volumenSize*voxelsize

                    );
                    cout << "current\t\t " << current_minx << "\t" << current_miny << "\t" << current_minz << endl;
                    cout << "max\t\t " <<  bg.getBB().getMax()[0] << "\t" <<  bg.getBB().getMax()[1] << "\t" <<  bg.getBB().getMax()[2] << endl;
                    partitionBoxes.push_back(partBB);
                    current_minz+=volumenSize*voxelsize;
                }
                current_miny+=volumenSize*voxelsize;
            }
            current_minx+=volumenSize*voxelsize;
        }

    }
    else
    {
        BigGridKdTree gridKd(bg.getBB(),options.getNodeSize(),&bg, voxelsize);
        gridKd.insert(bg.pointSize(),bg.getBB().getCentroid());
        for(size_t i = 0 ; i <  gridKd.getLeafs().size(); i++)
        {
            BoundingBox<Vertexf > partBB = gridKd.getLeafs()[i]->getBB();
            partitionBoxes.push_back(partBB);
        }
    }

    cout << lvr::timestamp << "finished tree" << endl;

    std::cout << lvr::timestamp << "got: " << partitionBoxes.size() << " leafs, saving leafs" << std::endl;



    BoundingBox<ColorVertex<float,unsigned char> > cbb(bb.getMin().x, bb.getMin().y, bb.getMin().z,
                                                       bb.getMax().x, bb.getMax().y, bb.getMax().z);

    vector<string> grid_files;

    for(size_t i = 0 ; i < partitionBoxes.size() ; i++)
    {
        size_t numPoints;

        //todo: okay?
        lvr::floatArr points = bg.points(partitionBoxes[i].getMin().x - voxelsize*3, partitionBoxes[i].getMin().y - voxelsize*3, partitionBoxes[i].getMin().z - voxelsize*3 ,
                                         partitionBoxes[i].getMax().x + voxelsize*3, partitionBoxes[i].getMax().y + voxelsize*3, partitionBoxes[i].getMax().z + voxelsize*3,numPoints);

        //std::cout << "i: " << std::endl << bb << std::endl << "got : " << numPoints << std::endl;
        if(numPoints<=50) continue;

        BoundingBox<ColorVertex<float,unsigned char> > gridbb(partitionBoxes[i].getMin().x, partitionBoxes[i].getMin().y, partitionBoxes[i].getMin().z,
                                                              partitionBoxes[i].getMax().x, partitionBoxes[i].getMax().y, partitionBoxes[i].getMax().z);
        cout << "grid: " << i << "/" << partitionBoxes.size()-1 << endl;
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
            lvr::floatArr normals = bg.normals(partitionBoxes[i].getMin().x, partitionBoxes[i].getMin().y, partitionBoxes[i].getMin().z ,
                                              partitionBoxes[i].getMax().x, partitionBoxes[i].getMax().y, partitionBoxes[i].getMax().z,numNormals);


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
//        FastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
        PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
        ps_grid->setBB(gridbb);
        ps_grid->calcIndices();
        ps_grid->calcDistanceValues();

        reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
        HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh;

        vector<unsigned int> duplicates;
        reconstruction->getMesh(mesh, gridbb, duplicates, voxelsize/10);
        mesh.finalize();

        std::stringstream ss_mesh;
        ss_mesh << "part-" << i << "-mesh.ply";
        ModelPtr m( new Model( mesh.meshBuffer() ) );
        ModelFactory::saveModel( m, ss_mesh.str());
        delete reconstruction;

        std::stringstream ss_grid;
        ss_grid << "part-" << i << "-grid.ser";
        ps_grid->saveCells(ss_grid.str());
        grid_files.push_back(ss_grid.str());

        std::stringstream ss_duplicates;
        ss_duplicates << "part-" << i << "-duplicates.ser";
        std::ofstream ofs(ss_duplicates.str(), std::ofstream::out | std::ofstream::trunc);
        boost::archive::text_oarchive oa(ofs);
        oa & duplicates;

        delete ps_grid;




    }

    vector<size_t> offsets;
    offsets.push_back(0);
    vector<unsigned int> all_duplicates;
    vector<float> duplicateVertices;
    for(int i = 0 ; i <grid_files.size() ; i++)
    {
        string duplicate_path = grid_files[i];
        string ply_path = grid_files[i];
        boost::algorithm::replace_last(duplicate_path, "-grid.ser", "-duplicates.ser");
        boost::algorithm::replace_last(ply_path, "-grid.ser", "-mesh.ply");
        std::ifstream ifs(duplicate_path);
        boost::archive::text_iarchive ia(ifs);
        std::vector<unsigned int> duplicates;
        ia & duplicates;
        LineReader lr(ply_path);
        size_t numPoints = lr.getNumPoints();
        offsets.push_back(numPoints+offsets[i]);
        if(numPoints==0) continue;
        lvr::PLYIO io;
        ModelPtr modelPtr = io.read(ply_path);
        //ModelPtr modelPtr = ModelFactory::readModel(ply_path);
        size_t numVertices;
        floatArr modelVertices = modelPtr->m_mesh->getVertexArray(numVertices);

//        for (size_t j = 0; j < numVertices; j++)
//        {
//            duplicates.push_back(static_cast<unsigned int>(j));
//        }

        for(size_t j  = 0 ; j<duplicates.size() ; j++)
        {
            duplicateVertices.push_back(modelVertices[duplicates[j]*3]);
            duplicateVertices.push_back(modelVertices[duplicates[j]*3+1]);
            duplicateVertices.push_back(modelVertices[duplicates[j]*3+2]);
        }

        std::transform (duplicates.begin(), duplicates.end(), duplicates.begin(), [&](unsigned int x){return x+offsets[i];});
        all_duplicates.insert(all_duplicates.end(),duplicates.begin(), duplicates.end());

    }
    std::unordered_map<unsigned int, unsigned int> oldToNew;
    float dist_epsilon_squared = (voxelsize/100)*(voxelsize/100);
    for(size_t i = 0 ; i < duplicateVertices.size() ; i+=3)
    {
        float ax = duplicateVertices[i];
        float ay = duplicateVertices[i+1];
        float az = duplicateVertices[i+2];
        vector<unsigned int> duplicates_of_i;
        for(size_t j = 0 ; j < duplicateVertices.size() ; j+=3)
        {
            if(i==j) continue;
            float bx = duplicateVertices[j];
            float by = duplicateVertices[j+1];
            float bz = duplicateVertices[j+2];
            float dist_squared = (ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz);
            if(dist_squared < dist_epsilon_squared)
            {
                auto find_it = oldToNew.find(all_duplicates[i/3]);
                if( find_it == oldToNew.end())
                {

                    oldToNew[all_duplicates[j/3]] = all_duplicates[i/3];
                    cout << "dup found! mapping " << all_duplicates[j/3] << " -> " << all_duplicates[i/3] << endl;

                } //else{

//                    if(all_duplicates[j/3] != all_duplicates[i/3])
//                    {
//                        oldToNew[all_duplicates[j/3]] = oldToNew[all_duplicates[i/3]];
//                        cout << "SHIT: " << all_duplicates[j/3] << " -> " << oldToNew[all_duplicates[i/3]] << endl;
//                    }
//                    else
//                    {
//                        cout << "SHIT SHIT SHIT" << endl;
//                    }
//
//              }
            }
        }

    }
    for(auto testit = oldToNew.begin(); testit != oldToNew.end(); testit++)
    {
        if(oldToNew.find(testit->second) != oldToNew.end()) cout << "SHIT FUCK SHIT" << endl;
    }
    ofstream ofs_vertices("largeVertices.bin", std::ofstream::out | std::ofstream::trunc);
    ofstream ofs_faces("largeFaces.bin", std::ofstream::out | std::ofstream::trunc);
    size_t increment=0;
    vector<size_t> increments;
    size_t newNumVertices = 0;
    size_t newNumFaces = 0;
    for(size_t i = 0 ; i <grid_files.size() ; i++)
    {

        string ply_path = grid_files[i];
        boost::algorithm::replace_last(ply_path, "-grid.ser", "-mesh.ply");
        LineReader lr(ply_path);
        size_t numPoints = lr.getNumPoints();
        if(numPoints==0) continue;
        lvr::PLYIO io;
        ModelPtr modelPtr = io.read(ply_path);
        size_t numVertices;
        size_t numFaces;
        size_t offset = offsets[i];
        floatArr modelVertices = modelPtr->m_mesh->getVertexArray(numVertices);
        uintArr modelFaces = modelPtr->m_mesh->getFaceArray(numFaces);
        newNumFaces+=numFaces;
        for(size_t j = 0; j<numVertices ; j++)
        {
            float x = modelVertices[j*3];
            float y = modelVertices[j*3+1];
            float z = modelVertices[j*3+2];
            if(oldToNew.find(j+offset)==oldToNew.end())
            {
                ofs_vertices << x << " " << y << " " << z << endl;
                newNumVertices++;
            }
            else
            {
                increment++;
            }
            increments.push_back(increment);

        }
        for(int j = 0 ; j<numFaces; j++)
        {
            unsigned int f[3];
            f[0] = modelFaces[j*3] + offset;
            f[1] = modelFaces[j*3+1] + offset;
            f[2] = modelFaces[j*3+2] + offset;
            ofs_faces << "3 ";
            for(int k = 0 ; k < 3; k++)
            {
                size_t face_idx = 0;
                if(oldToNew.find(f[k]) == oldToNew.end())
                {
                    face_idx = f[k] - increments[f[k]];
//                    if (face_idx > 18939) // debug number of faces of scan.pts
//                    {
//                        cout << "no old to new " << face_idx << endl;
//                    }
                }
                else
                {
                    face_idx = oldToNew[f[k]]- increments[oldToNew[f[k]]];
//                    if (face_idx > 18939) // debug number of faces of scan.pts
//                    {
//                        cout << "WTF!!!!! " << f[k] << " (key) mapped to " << face_idx << endl;
//                    }
                }
                ofs_faces << face_idx;
                if(k!=2) ofs_faces << " ";

            }
            // todo: sort
            ofs_faces << endl;

        }




    }
    cout << "Largest decrement: " << increment << endl;

    ofstream ofs_ply("bigMesh.ply", std::ofstream::out | std::ofstream::trunc);
    ifstream ifs_faces("largeFaces.bin");
    ifstream ifs_vertices("largeVertices.bin");
    string line;
    ofs_ply << "ply\n"
            "format ascii 1.0\n"
            "element vertex " << newNumVertices << "\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "element face " << newNumFaces << "\n"
            "property list uchar int vertex_indices\n"
            "end_header" << endl;
    while(std::getline(ifs_vertices,line))
    {
        ofs_ply << line << endl;
    }
    while(std::getline(ifs_faces,line))
    {
        ofs_ply << line << endl;
    }




    cout << lvr::timestamp << "finished" << endl;

//
//    auto vmax = cbb.getMax();
//    auto vmin = cbb.getMin();
//    vmin.x-=voxelsize*2;
//    vmin.y-=voxelsize*2;
//    vmin.z-=voxelsize*2;
//    vmax.x+=voxelsize*2;
//    vmax.y+=voxelsize*2;
//    vmax.z+=voxelsize*2;
//    cbb.expand(vmin);
//    cbb.expand(vmax);
//    HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >
//            hg(grid_files, cbb, voxelsize);
//
//
//
//    //hg.saveGrid("largeScale.grid");
//
//
//
//
//
//    lvr::FastReconstructionBase<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> >* reconstruction;
//    reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(&hg);
//    std::vector<float> vBuffer;
//    std::vector<unsigned int> fBuffer;
//    size_t vi,fi;
//    reconstruction->getMesh(vBuffer, fBuffer,fi,vi);
////    HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh;
////    if(options.getDepth())
////    {
////        mesh.setDepth(options.getDepth());
////    }
////    reconstruction->getMesh(mesh);
////    if(options.getDanglingArtifacts())
////    {
////        mesh.removeDanglingArtifacts(options.getDanglingArtifacts());
////    }
////    mesh.cleanContours(options.getCleanContourIterations());
////    mesh.setClassifier(options.getClassifier());
////    mesh.getClassifier().setMinRegionSize(options.getSmallRegionThreshold());
////    if(options.optimizePlanes())
////    {
////        mesh.optimizePlanes(options.getPlaneIterations(),
////                            options.getNormalThreshold(),
////                            options.getMinPlaneSize(),
////                            options.getSmallRegionThreshold(),
////                            true);
////
////        mesh.fillHoles(options.getFillHoles());
////        mesh.optimizePlaneIntersections();
////        mesh.restorePlanes(options.getMinPlaneSize());
////
////        if(options.getNumEdgeCollapses())
////        {
////            QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> > c = QuadricVertexCosts<ColorVertex<float, unsigned char> , Normal<float> >(true);
////            mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
////        }
////    }
////    else if(options.clusterPlanes())
////    {
////        mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
////        mesh.fillHoles(options.getFillHoles());
////    }
////
////
////    if ( options.retesselate() )
////    {
////        mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
////    }
////    else
////    {
////        mesh.finalize();
////    }
//    lvr::MeshBufferPtr mb(new lvr::MeshBuffer);
//    mb->setFaceArray(fBuffer);
//    mb->setVertexArray(vBuffer);
//    ModelPtr m( new Model( mb ) );
//    ModelFactory::saveModel( m, "largeScale.ply");
//    delete reconstruction;
//
//    lvr::PointBufferPtr pb2(new lvr::PointBuffer);
//    pb2->setPointArray(points2, numPoints2);
//    lvr::ModelPtr m2( new lvr::Model(pb2));
//    lvr::PLYIO io2;
//    io.save(m2,"testPoints2.ply");




    return 0;
}
