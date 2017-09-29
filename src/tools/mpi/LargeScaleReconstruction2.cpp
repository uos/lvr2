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
#include <lvr/reconstruction/FastBox.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <unordered_map>
#include "lvr/reconstruction/QueryPoint.hpp"
#include <fstream>
#include <sstream>
#include <mpi.h>
#include "LargeScaleOptions.hpp"
using std::cout;
using std::endl;
using namespace lvr;

typedef lvr::PointsetSurface<lvr::ColorVertex<float, unsigned char> > psSurface;
typedef lvr::AdaptiveKSearchSurface<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> > akSurface;

enum MPIMSG {NUMPOINTS, BB, POINTS,NORMALS, READY, PATH, POINTSTATUS};
enum POINTTYPE {XYZ, XYZN};
int main(int argc, char** argv)
{


    LargeScaleOptions::Options options(argc, argv);

    string filePath = options.getInputFileName();
    float voxelsize = options.getVoxelsize();
    float scale = options.getScaling();


    MPI_Init(NULL, NULL);

    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0)
    {
        cout << lvr::timestamp << "Starting grid" << endl;
        BigGrid bg(filePath, voxelsize, scale);
        cout << lvr::timestamp << "grid finished " << endl;
        lvr::BoundingBox<lvr::Vertexf> bb = bg.getBB();
        cout << bb << endl;
        cout << lvr::timestamp << "making tree" << endl;
        BigGridKdTree gridKd(bg.getBB(),options.getNodeSize(),&bg, voxelsize);
        gridKd.insert(bg.pointSize(),bg.getBB().getCentroid());
        cout << lvr::timestamp << "finished tree" << endl;

        std::cout << lvr::timestamp << "got: " << gridKd.getLeafs().size() << " leafs, saving leafs" << std::endl;



        BoundingBox<ColorVertex<float,unsigned char> > cbb(bb.getMin().x, bb.getMin().y, bb.getMin().z,
                                                           bb.getMax().x, bb.getMax().y, bb.getMax().z);

        vector<string> grid_files;
        std::cout <<  lvr::timestamp <<"Calculating normals and Distances..." << std::endl;
        size_t files_left = gridKd.getLeafs().size();
        size_t next_kd_node = 0;
        size_t working_nodes = 0;
        std::unordered_map<int, std::string> workingNodes;
        for(int i = 1 ; i<world_size && i<gridKd.getLeafs().size(); i++)
        {
            // SEND NumPoints
            size_t numPoints;
            lvr::floatArr points = bg.points(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z ,
                                             gridKd.getLeafs()[next_kd_node]->getBB().getMax().x, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z,numPoints);
            lvr::floatArr normals;
            if(bg.hasNormals())
            {
                normals = bg.normals(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z ,
                gridKd.getLeafs()[next_kd_node]->getBB().getMax().x, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z,numPoints);

            }
                                            
                                             // SEND NumPoints
            MPI_Send(
                    &numPoints,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    i,
                    NUMPOINTS,
                    MPI_COMM_WORLD);
            int normalStatus = XYZ;
            if(bg.hasNormals())
            {
                normalStatus = XYZN;
            }
            MPI_Send(
                    &normalStatus,
                    1,
                    MPI_INT,
                    i,
                    POINTSTATUS,
                    MPI_COMM_WORLD);        
            // Send PointArray
            MPI_Send(
                    points.get(),
                    numPoints*3,
                    MPI_FLOAT,
                    i,
                    POINTS,
                    MPI_COMM_WORLD);
            if(bg.hasNormals())
            {
                MPI_Send(
                        normals.get(),
                        numPoints*3,
                        MPI_FLOAT,
                        i,
                        NORMALS,
                        MPI_COMM_WORLD);
            }
            float bb[6];
            bb[0] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().x;
            bb[1] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().y;
            bb[2] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().z;
            bb[3] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().x;
            bb[4] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().y;
            bb[5] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().z;
            // Send BoundingBox
            MPI_Send(
                    bb,
                    6,
                    MPI_FLOAT,
                    i,
                    BB,
                    MPI_COMM_WORLD);

            std::stringstream ss2;
            ss2 << "testgrid-" << next_kd_node << ".ser";
            const char* opath = ss2.str().c_str();
            workingNodes[i] = ss2.str();
            //grid_files.push_back(ss2.str());
            MPI_Send(
                    opath,
                    ss2.str().size(),
                    MPI_CHAR,
                    i,
                    PATH,
                    MPI_COMM_WORLD);

            next_kd_node++;
            working_nodes++;
        }
        while(working_nodes != 0)
        {
            // check if not all nodes are working
            int data;
            MPI_Status status;
            MPI_Recv(
                    &data,
                    1,
                    MPI_INT,
                    MPI_ANY_SOURCE,
                    READY,
                    MPI_COMM_WORLD,
                    &status);
            if(data == 1)
            {
                grid_files.push_back(workingNodes[status.MPI_SOURCE]);
            }
            workingNodes.erase(status.MPI_SOURCE);
            
            working_nodes--;
            if(working_nodes < world_size-1 && next_kd_node < gridKd.getLeafs().size())
            {
                size_t numPoints;
                lvr::floatArr points = bg.points(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z ,
                                                 gridKd.getLeafs()[next_kd_node]->getBB().getMax().x, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z,numPoints);
                
                lvr::floatArr normals;
                if(bg.hasNormals())
                {
                normals = bg.normals(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z ,
                gridKd.getLeafs()[next_kd_node]->getBB().getMax().x, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z,numPoints);

                }
                                                 // SEND NumPoints
                MPI_Send(
                        &numPoints,
                        1,
                        MPI_UNSIGNED_LONG_LONG,
                        status.MPI_SOURCE,
                        NUMPOINTS,
                        MPI_COMM_WORLD);
                int normalStatus2 = XYZ;
                if(bg.hasNormals())
                {
                        normalStatus2 = XYZN;
                }
                MPI_Send(
                        &normalStatus2,
                        1,
                        MPI_INT,
                        status.MPI_SOURCE,
                        POINTSTATUS,
                        MPI_COMM_WORLD); 
                // Send PointArray
                MPI_Send(
                        points.get(),
                        numPoints*3,
                        MPI_FLOAT,
                        status.MPI_SOURCE,
                        POINTS,
                        MPI_COMM_WORLD);
                if(bg.hasNormals())
                {
                        MPI_Send(
                                normals.get(),
                                numPoints*3,
                                MPI_FLOAT,
                                status.MPI_SOURCE,
                                NORMALS,
                                MPI_COMM_WORLD);
                }
                float bb[6];
                bb[0] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().x;
                bb[1] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().y;
                bb[2] = gridKd.getLeafs()[next_kd_node]->getBB().getMin().z;
                bb[3] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().x;
                bb[4] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().y;
                bb[5] = gridKd.getLeafs()[next_kd_node]->getBB().getMax().z;
                // Send BoundingBox
                MPI_Send(
                        bb,
                        6,
                        MPI_FLOAT,
                        status.MPI_SOURCE,
                        BB,
                        MPI_COMM_WORLD);


                std::stringstream ss2;
                ss2 << "testgrid-" << next_kd_node << ".ser";
                const char* opath = ss2.str().c_str();
                workingNodes[status.MPI_SOURCE] = ss2.str();
                MPI_Send(
                        opath,
                        ss2.str().size(),
                        MPI_CHAR,
                        status.MPI_SOURCE,
                        PATH,
                        MPI_COMM_WORLD);
                next_kd_node++;
                working_nodes++;
            }
        }
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
        reconstruction->getMesh(mesh);

        mesh.finalize();
        ModelPtr m( new Model( mesh.meshBuffer() ) );
        ModelFactory::saveModel( m, "largeScale.ply");
        MPI_Abort(MPI_COMM_WORLD,1);
        MPI_Finalize();
    }
    else
    {
        while(1)
        {
            size_t data;
            MPI_Recv(
                    &data,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_ANY_SOURCE,
                    NUMPOINTS,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            
            int normalStatus;
            MPI_Recv(
                &normalStatus,
                1,
                MPI_INT,
                MPI_ANY_SOURCE,
                POINTSTATUS,
                MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);


            cout <<world_rank << " - will get " << data <<  "points" << endl;

            lvr::floatArr points(new float[data*3]);
            MPI_Recv(
                    points.get(),
                    data*3,
                    MPI_FLOAT,
                    MPI_ANY_SOURCE,
                    POINTS,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            lvr::floatArr normals;  
            if(normalStatus == XYZN)
            {
                normals = lvr::floatArr(new float[data*3]);
                MPI_Recv(
                        normals.get(),
                        data*3,
                        MPI_FLOAT,
                        MPI_ANY_SOURCE,
                        NORMALS,
                        MPI_COMM_WORLD,
                        MPI_STATUS_IGNORE);
            }
            float bb[6];
            MPI_Recv(
                    &bb,
                    6,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_ANY_SOURCE,
                    BB,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            char outPath[1024];
            MPI_Recv(
                    &outPath,
                    1024,
                    MPI_CHAR,
                    MPI_ANY_SOURCE,
                    PATH,
                    MPI_COMM_WORLD,
                    MPI_STATUS_IGNORE);
            string output(outPath);
            int fin = 0;
            if(data > options.getKn() && data > options.getKi() && data > options.getKd())
            {
                BoundingBox<ColorVertex<float,unsigned char> > gridbb(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
                cout << gridbb << endl;
                lvr::PointBufferPtr p_loader(new lvr::PointBuffer);
                p_loader->setPointArray(points, data);
                if(normalStatus == XYZN)
                {
                        p_loader->setPointNormalArray(normals, data);
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

                if(normalStatus == XYZ)
                {
                        surface->calculateSurfaceNormals();
                }
                
                lvr::GridBase* grid;
                lvr::FastReconstructionBase<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> >* reconstruction;

                grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(voxelsize, surface,gridbb , true, true);
                BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->setBB(gridbb);
                ps_grid->calcIndices();
                ps_grid->calcDistanceValues();

                ps_grid->saveCells(output);
                delete ps_grid;
                fin = 1;
            }
            MPI_Send(
                    &fin,
                    1,
                    MPI_INT,
                    0,
                    READY,
                    MPI_COMM_WORLD);
        }

    }




  return 0;
}