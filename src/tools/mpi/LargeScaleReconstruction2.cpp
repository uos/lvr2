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
#include "LineReader.hpp"
#include <boost/algorithm/string.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <mpi.h>
#include "LargeScaleOptions.hpp"
using std::cout;
using std::endl;
using namespace lvr;
struct duplicateVertex{
    float x;
    float y;
    float z;
    unsigned int id;

};
typedef lvr::PointsetSurface<lvr::ColorVertex<float, unsigned char> > psSurface;
typedef lvr::AdaptiveKSearchSurface<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> > akSurface;

enum MPIMSG {NUMPOINTS, BB, POINTS,NORMALS, READY, PATH, POINTSTATUS, STOP};
enum POINTTYPE {MPIXYZ, MPIXYZN};
int main(int argc, char** argv)
{


    LargeScaleOptions::Options options(argc, argv);

    vector<string> filePaths = options.getInputFileName();
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
        BigGrid bg(filePaths, voxelsize, scale);
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
            lvr::floatArr points = bg.points(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z - voxelsize*5,
                                             gridKd.getLeafs()[next_kd_node]->getBB().getMax().x + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z + voxelsize*5,numPoints);
            lvr::floatArr normals;
            if(bg.hasNormals())
            {
                normals = bg.normals(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z - voxelsize*5 ,
                                     gridKd.getLeafs()[next_kd_node]->getBB().getMax().x + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z + voxelsize*5,numPoints);

            }

                                             // SEND NumPoints
            MPI_Send(
                    &numPoints,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    i,
                    NUMPOINTS,
                    MPI_COMM_WORLD);
            int normalStatus = MPIXYZ;
            if(bg.hasNormals())
            {
                normalStatus = MPIXYZN;
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
            ss2 << "part-" << next_kd_node << ".ser";
            const char* opath = ss2.str().c_str();
            workingNodes[i] = ss2.str();
            //grid_files.push_back(ss2.str());
            MPI_Send(
                    opath,
                    ss2.str().size()+2,
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
                lvr::floatArr points = bg.points(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z - voxelsize*5,
                                                 gridKd.getLeafs()[next_kd_node]->getBB().getMax().x + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z + voxelsize*5,numPoints);
                lvr::floatArr normals;
                if(bg.hasNormals())
                {
                    normals = bg.normals(gridKd.getLeafs()[next_kd_node]->getBB().getMin().x - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().y - voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMin().z - voxelsize*5 ,
                                         gridKd.getLeafs()[next_kd_node]->getBB().getMax().x + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().y + voxelsize*5, gridKd.getLeafs()[next_kd_node]->getBB().getMax().z + voxelsize*5,numPoints);

                }
                                                 // SEND NumPoints
                MPI_Send(
                        &numPoints,
                        1,
                        MPI_UNSIGNED_LONG_LONG,
                        status.MPI_SOURCE,
                        NUMPOINTS,
                        MPI_COMM_WORLD);
                int normalStatus2 = MPIXYZ;
                if(bg.hasNormals())
                {
                        normalStatus2 = MPIXYZN;
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
                ss2 << "part-" << next_kd_node << ".ser";
                const char* opath = ss2.str().c_str();
                workingNodes[status.MPI_SOURCE] = ss2.str();
                MPI_Send(
                        opath,
                        ss2.str().size()+2,
                        MPI_CHAR,
                        status.MPI_SOURCE,
                        PATH,
                        MPI_COMM_WORLD);
                next_kd_node++;
                working_nodes++;
            }
        }
        // Tell all nodes to shutdown
        int stop_msg = 1;
        for(int i = 1 ; i<world_size && i<gridKd.getLeafs().size(); i++)
        {
            MPI_Send(
                    &stop_msg,
                    sizeof(int),
                    MPI_INT,
                    i,
                    STOP,
                    MPI_COMM_WORLD);
        }

        vector<size_t> offsets;
        offsets.push_back(0);
        vector<unsigned int> all_duplicates;
//    vector<float> duplicateVertices;
        vector<duplicateVertex> duplicateVertices;
        for(int i = 0 ; i <grid_files.size() ; i++)
        {
            string duplicate_path = grid_files[i];
            string ply_path = grid_files[i];
            boost::algorithm::replace_last(duplicate_path, ".ser", "-duplicates.ser");
            boost::algorithm::replace_last(ply_path, ".ser", "-mesh.ply");
            std::ifstream ifs(duplicate_path);
            cout << "opening: " << duplicate_path << endl;
            boost::archive::text_iarchive ia(ifs);
            std::vector<unsigned int> duplicates;
            ia & duplicates;
            cout << "MESH " << i << " has " << duplicates.size() << " possible duplicates" << endl;
            LineReader lr(ply_path);

            size_t numPoints = lr.getNumPoints();
            offsets.push_back(numPoints+offsets[i]);
            if(numPoints==0) continue;
            lvr::PLYIO io;
            ModelPtr modelPtr = io.read(ply_path);
            //ModelPtr modelPtr = ModelFactory::readModel(ply_path);
            size_t numVertices;
            floatArr modelVertices = modelPtr->m_mesh->getVertexArray(numVertices);


            for(size_t j  = 0 ; j<duplicates.size() ; j++)
            {
                duplicateVertex v;
                v.x = modelVertices[duplicates[j]*3];
                v.y = modelVertices[duplicates[j]*3+1];
                v.z = modelVertices[duplicates[j]*3+2];
                v.id = duplicates[j] + offsets[i];
                duplicateVertices.push_back(v);
            }


        }
        std::sort(duplicateVertices.begin(), duplicateVertices.end(), [](const duplicateVertex &left, const duplicateVertex &right) {
            return left.id < right.id;
        });
        std::unordered_map<unsigned int, unsigned int> oldToNew;

        ofstream ofsd;
        ofsd.open("duplicate_colors.pts", ios_base::app);


        float comp_dist = std::max(voxelsize/100, 0.0001f);
        double dist_epsilon_squared = comp_dist*comp_dist;
        cout << lvr::timestamp << "removing duplicate vertices" << endl;


        omp_lock_t writelock;
        omp_init_lock(&writelock);
#pragma omp parallel for
        for(size_t i = 0 ; i < duplicateVertices.size() ; i++)
        {
            float ax = duplicateVertices[i].x;
            float ay = duplicateVertices[i].y;
            float az = duplicateVertices[i].z;
//        vector<unsigned int> duplicates_of_i;
            int found = 0;
            auto find_it = oldToNew.find(duplicateVertices[i].id);
            if( find_it == oldToNew.end())
            {
#pragma omp parallel for schedule(dynamic,1)
                for(size_t j = 0 ; j < duplicateVertices.size()  ; j++)
                {
                    if(i==j || found >5) continue;
                    float bx = duplicateVertices[j].x;
                    float by = duplicateVertices[j].y;
                    float bz = duplicateVertices[j].z;
                    double dist_squared = (ax-bx)*(ax-bx) + (ay-by)*(ay-by) + (az-bz)*(az-bz);
                    if(dist_squared < dist_epsilon_squared)
                    {
//

                        if(duplicateVertices[j].id < duplicateVertices[i].id)
                        {
//                        cout << "FUCK THIS SHIT" << endl;
                            continue;
                        }
                        omp_set_lock(&writelock);
                        oldToNew[duplicateVertices[j].id] = duplicateVertices[i].id;
                        found++;
                        omp_unset_lock(&writelock);
//                    cout << "dup found! mapping " <<duplicateVertices[j].id << " -> " << duplicateVertices[i].id << " dist: " << sqrt(dist_squared)<< endl;

//                    ofsd << duplicateVertices[j].x << " " << duplicateVertices[j].y << " " << duplicateVertices[j].z << endl;


//                omp_unset_lock(&writelock);
                    }
                }
            }


        }
        cout << "FOUND: " << oldToNew.size() << " duplicates" << endl;
//    for(auto testit = oldToNew.begin(); testit != oldToNew.end(); testit++)
//    {
//        if(oldToNew.find(testit->second) != oldToNew.end()) cout << "SHIT FUCK SHIT" << endl;
//    }
        ofstream ofs_vertices("largeVertices.bin", std::ofstream::out | std::ofstream::trunc );
        ofstream ofs_faces("largeFaces.bin", std::ofstream::out | std::ofstream::trunc);

        size_t increment=0;
        std::map<size_t, size_t> decrements;
        decrements[0] = 0;

        size_t newNumVertices = 0;
        size_t newNumFaces = 0;
        cout << lvr::timestamp << "merging mesh..." << endl;
        for(size_t i = 0 ; i <grid_files.size() ; i++)
        {

            string ply_path = grid_files[i];
            boost::algorithm::replace_last(ply_path, ".ser", "-mesh.ply");
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
                float p[3];
                p[0] = modelVertices[j*3];
                p[1] = modelVertices[j*3+1];
                p[2] = modelVertices[j*3+2];
                if(oldToNew.find(j+offset)==oldToNew.end())
                {
//                ofs_vertices.write((char*)p,sizeof(float)*3);
                    ofs_vertices << std::setprecision(16) << p[0] << " " << p[1] << " " << p[2] << endl;
                    newNumVertices++;
                }
                else
                {
                    increment++;
                    decrements[j+offset] = increment;
                }
            }
            size_t new_face_num = 0;
            for(int j = 0 ; j<numFaces; j++)
            {
                unsigned int f[3];
                f[0] = modelFaces[j*3] + offset;
                f[1] = modelFaces[j*3+1] + offset;
                f[2] = modelFaces[j*3+2] + offset;

                ofs_faces << "3 ";
                unsigned int newface[3];
                unsigned char a = 3;
//            ofs_faces.write((char*)&a, sizeof(unsigned char));
                for(int k = 0 ; k < 3; k++)
                {
                    size_t face_idx = 0;
                    if(oldToNew.find(f[k]) == oldToNew.end())
                    {
                        auto decr_itr = decrements.upper_bound(f[k]);
                        decr_itr--;
                        face_idx = f[k] - decr_itr->second;
                    }
                    else
                    {
                        auto decr_itr = decrements.upper_bound(oldToNew[f[k]]);
                        decr_itr--;
                        face_idx = oldToNew[f[k]]- decr_itr->second;
//
                    }
                    ofs_faces << face_idx;
                    if(k!=2) ofs_faces << " ";

                }
                ofs_faces << endl;
//            ofs_faces.write( (char*) newface,sizeof(unsigned int)*3);
                // todo: sort


            }




        }
        ofs_faces.close();
        ofs_vertices.close();
        cout << lvr::timestamp << "saving ply" << endl;
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
//    ofs_ply.close();
//    ofstream ofs_ply_binary("bigMesh.ply",std::ofstream::out | std::ofstream::app | std::ofstream::binary );
//    char a1 = 10;
//    char a2 = 32;
//    char a3 = 180;
//    char a4 = 174;
//    ofs_ply_binary.write(&a1,1);
//    ofs_ply_binary.write(&a2,1);
//    ofs_ply_binary.write(&a3,1);
//    ofs_ply_binary.write(&a4,1);
//    istreambuf_iterator<char> begin_source(ifs_vertices);
//    istreambuf_iterator<char> end_source;
//    ostreambuf_iterator<char> begin_dest(ofs_ply_binary);
//    copy(begin_source, end_source, begin_dest);
//    istreambuf_iterator<char> begin_source2(ifs_faces);
//    istreambuf_iterator<char> end_source2;
//    ostreambuf_iterator<char> begin_dest2(ofs_ply_binary);
//    copy(begin_source2, end_source2, begin_dest2);
        while(std::getline(ifs_vertices,line))
        {
            ofs_ply << line << endl;
        }
        size_t c = 0;
        while(std::getline(ifs_faces,line))
        {

            ofs_ply << line << endl;
            stringstream ss(line);
            unsigned int v[3];
            ss >> v[0];
            ss >> v[1];
            ss >> v[2];
            for(int i = 0 ; i< 3; i++)
            {
                if(v[i]>=newNumVertices)
                {
                    cout << "WTF: FACE " << c << " has index " << v[i] << endl;
                }

            }
            c++;
        }
        ofs_ply << endl;


        cout << lvr::timestamp << "finished" << endl;




        ofs_ply.close();
        ifs_faces.close();
        ifs_vertices.close();

        MPI_Abort(MPI_COMM_WORLD,1);
        MPI_Finalize();
    }
    else
    {
        MPI_Request req_stop;
        int stop_data;
        MPI_Irecv(
                &stop_data,
                1,
                MPI_INT,
                MPI_ANY_SOURCE,
                STOP,
                MPI_COMM_WORLD,
                &req_stop);

        omp_set_num_threads(1);
        int stop_work = 0;
        while(stop_work == 0)
        {
            int got_num_points = 0;
            int got_use_normals = 0;
            int got_points = 0;
            int got_normals = 0;
            int got_bb = 0;
            int got_path = 0;

            //Number of Points
            MPI_Request req_num_points;
            size_t data;
            MPI_Irecv(
                    &data,
                    1,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_ANY_SOURCE,
                    NUMPOINTS,
                    MPI_COMM_WORLD,
                    &req_num_points);
            while(!got_num_points )
            {

                MPI_Test(&req_num_points, &got_num_points, MPI_STATUS_IGNORE);
                MPI_Test(&req_stop, &stop_work, MPI_STATUS_IGNORE);
                if(stop_work == 1) break;
                sleep(1);
            }
            // Use normals?
            MPI_Request req_use_normals;
            int normalStatus;
            MPI_Irecv(
                    &normalStatus,
                    1,
                    MPI_INT,
                    MPI_ANY_SOURCE,
                    POINTSTATUS,
                    MPI_COMM_WORLD,
                    &req_use_normals);

            // Points
            MPI_Request req_got_points;
            lvr::floatArr points(new float[data*3]);
            MPI_Irecv(
                    points.get(),
                    data*3,
                    MPI_FLOAT,
                    MPI_ANY_SOURCE,
                    POINTS,
                    MPI_COMM_WORLD,
                    &req_got_points);
            lvr::floatArr normals;

            //Normals
            MPI_Request req_normals;
            if(normalStatus == MPIXYZN)
            {
                normals = lvr::floatArr(new float[data*3]);
                MPI_Irecv(
                        normals.get(),
                        data*3,
                        MPI_FLOAT,
                        MPI_ANY_SOURCE,
                        NORMALS,
                        MPI_COMM_WORLD,
                        &req_normals);
            }

            //Bounding Box
            MPI_Request req_bb;
            float bb[6];
            MPI_Irecv(
                    &bb,
                    6,
                    MPI_UNSIGNED_LONG_LONG,
                    MPI_ANY_SOURCE,
                    BB,
                    MPI_COMM_WORLD,
                    &req_bb);

            // Path
            MPI_Request req_path;
            char outPath[1024];
            MPI_Irecv(
                    &outPath,
                    1024,
                    MPI_CHAR,
                    MPI_ANY_SOURCE,
                    PATH,
                    MPI_COMM_WORLD,
                    &req_path);


            while(! ( got_use_normals && got_points && got_bb && got_path && got_normals ))
            {
                cout << 1 << endl;
                MPI_Test(&req_num_points, &got_num_points, MPI_STATUS_IGNORE);
                cout << 2 << endl;
                MPI_Test(&req_use_normals, &got_use_normals, MPI_STATUS_IGNORE);
                cout << 3 << endl;
                MPI_Test(&req_got_points, &got_points, MPI_STATUS_IGNORE);
                cout << 4 << endl;
                MPI_Test(&req_bb, &got_bb, MPI_STATUS_IGNORE);
                cout << 5 << endl;
                MPI_Test(&req_path, &got_path, MPI_STATUS_IGNORE);
                cout << 6 << endl;
                MPI_Test(&req_stop, &stop_work, MPI_STATUS_IGNORE);
                cout << 7 << endl;
                if(normalStatus == MPIXYZN)
                {
                    MPI_Test(&req_normals, &got_normals, MPI_STATUS_IGNORE);
                }
                else
                {
                    got_normals = 1;
                }
                if(stop_work == 1) break;
                sleep(1);


            }

            if(stop_work==1) break;
            string output(outPath);
            int fin = 0;
            if(data > options.getKn() && data > options.getKi() && data > options.getKd())
            {
                BoundingBox<ColorVertex<float,unsigned char> > gridbb(bb[0], bb[1], bb[2], bb[3], bb[4], bb[5]);
                cout << gridbb << endl;
                lvr::PointBufferPtr p_loader(new lvr::PointBuffer);
                p_loader->setPointArray(points, data);
                if(normalStatus == MPIXYZN)
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
                // Todo: GPU
                if(normalStatus == MPIXYZ)
                {
                        surface->calculateSurfaceNormals();
                }

                lvr::GridBase* grid;
                lvr::FastReconstructionBase<lvr::ColorVertex<float, unsigned char>, lvr::Normal<float> >* reconstruction;

                grid = new PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(voxelsize, surface,gridbb , true, true);
//                FastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->setBB(gridbb);
                ps_grid->calcIndices();
                ps_grid->calcDistanceValues();



                reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
                HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh;
                vector<unsigned int> duplicates;
                reconstruction->getMesh(mesh, ps_grid->qp_bb, duplicates, voxelsize*5);
                mesh.finalize();
                boost::algorithm::erase_all(output, " ");
                boost::algorithm::replace_first(output,".ser","-mesh.ply");
                cout << "######################### " << output << endl;
                ModelPtr m( new Model( mesh.meshBuffer() ) );
                ModelFactory::saveModel( m, output);
                boost::algorithm::replace_first(output,"-mesh.ply","-duplicates.ser");
                std::ofstream ofs_dup(output, std::ofstream::out | std::ofstream::trunc);
                boost::archive::text_oarchive oa(ofs_dup);

//                ModelPtr pn( new Model);
//                pn->m_pointCloud = surface->pointBuffer();
//                boost::algorithm::replace_first(output,"-duplicates.ser","-normals.ply");
//                ModelFactory::saveModel(pn, output);

                oa & duplicates;



                delete reconstruction;


//                ps_grid->saveCells(output);
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
