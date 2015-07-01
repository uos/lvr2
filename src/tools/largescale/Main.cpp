#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/nonblocking.hpp>
#include <boost/serialization/vector.hpp>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <unistd.h>

// Las vegas Toolkit
#include "io/PointBuffer.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"
#include "mpi/MPITree.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/Normal.hpp"
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "io/Progress.hpp"

#include <boost/program_options.hpp>
#include "largescale/LSTree.hpp"
#include <random>
#include <boost/iostreams/device/mapped_file.hpp>
#include <iostream>
#include <thread>


using namespace lvr;
namespace mpi = boost::mpi;
size_t countLines(string filename);
std::string mmf_path = "asd.raw";
enum ClientServerMsgs{READY, BUSY,STOP, START, BBOX, READY};

int main (int argc , char *argv[]) {
    // get all options
    mpi::environment env;
    mpi::communicator world;
    // Main Process (rank==0):
    if(world.rank()==0)
    {
        boost::program_options::options_description desc("Allowed options");
        desc.add_options() ("file", boost::program_options::value<string>()->default_value("noinput"), "Inputfile");

        boost::program_options::variables_map vm;
        boost::program_options::store(boost::program_options::parse_command_line(argc, argv, desc), vm);
        boost::program_options::notify(vm);

        std::string filePath = vm["file"].as<string>();



        LSTree tree(mmf_path, 100, 24000000);


        vector<bool> clients_ready(world.size()-1,false);
        int next_element=0;
        bool finished = false;
        std::vector<mpi::request> mpi_requests;

        //Send Bounding Box:
        float fbbox[6];
        fbbox[0]= tree.getM_bbox().getMin().x;
        fbbox[1]= tree.getM_bbox().getMin().y;
        fbbox[2]= tree.getM_bbox().getMin().z;
        fbbox[3]= tree.getM_bbox().getMax().x;
        fbbox[4]= tree.getM_bbox().getMax().y;
        fbbox[5]= tree.getM_bbox().getMax().z;

        for(int i = 0; i<clients_ready.size() ; i++)
        {

            world.send(i+1, BBOX, fbbox);

        }
        while(!finished)
        {
            //Check if clients need new data
            for(int i = 0; i<clients_ready.size() ; i++)
            {
                if(clients_ready[i])
                {
                    mpi_requests.push_back(world.isend(i+1, START, tree.getM_points()[next_element++]));
                    clients_ready[i] = false;
                }
            }


        }


        mpi::wait_all(mpi_requests.begin(), mpi_requests.end());

        std::cout << timestamp << "finished" << endl;
    }
    // CLIENTS ########################################################################
    else
    {
        std::vector<std::thread> threads;
        boost::iostreams::mapped_file_source mmf;


        mmf.open(mmf_path);
        float * mmf_data = (float *)mmf.data();
        std::vector<unsigned long long int> * index_vec = new std::vector<unsigned long long int>();
        float bbox[6];
        world.recv(0,BBOX,bbox);
        while(true)
        {
            index_vec->clear();
            mpi::request reqs[2];
            reqs[0] = world.irecv(0,START,*(index_vec));
            reqs[1] = world.irecv(0,STOP);
            mpi::wait_any(reqs, reqs+1);
            if(index_vec->size()<=1) break;
            cout << "Nr " << world.rank() << "got vector with:" << endl;
            float * vertex_array = new float[3 * index_vec->size()];


            int j = 0;
            for(int i = 0 ; i<index_vec->size() ;i++)
            {
                vertex_array[j  ]=mmf_data[(*(index_vec))[i]  ];
                vertex_array[j+1]=mmf_data[(*(index_vec))[i]+1];
                vertex_array[j+2]=mmf_data[(*(index_vec))[i]+2];
                j+=3;
            }
            boost::shared_array<float> vertices (vertex_array);
            PointBufferPtr pointcloud(new PointBuffer());
            pointcloud->setPointArray(vertices, index_vec->size());
            PointsetSurface<ColorVertex<float, unsigned char> >* surface;
            surface = new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >(pointcloud, "FLANN", 100, 100, 100, false);

            // set global Bounding-Box
            surface->expandBoundingBox(bbox[0], bbox[1], bbox[2],
                                       bbox[3], bbox[4], bbox[5]);



            // calculate the normals
            std::cout << timestamp << "Client " << world.rank() << " calculates surface normals with " << index_vec->size() << " points." <<  std::endl;
            surface->calculateSurfaceNormals();
            pointcloud = surface->pointBuffer();
            size_t n;
            floatArr normals = pointcloud->getPointNormalArray(n);
            world.send(0, READY);
            threads.push_back(std::thread(save_normals, normals, index_vec->size(),index_vec));
        }

    }


}

void save_normals(floatArr normals, size_t n, std::vector<unsigned long long int> * index_vec )
{
    boost::iostreams::mapped_file_params mfp;
    mfp.new_file_size = n*3* sizeof(float);
    mfp.path = "normals.raw";

    boost::iostreams::mapped_file_sink mmf_normals(mfp);
    float * normal_data = (float *)mmf_normals.data();

    for(int i = 0 ; i<n ;i++)
    {
        normal_data[(*(index_vec))[i]]  = normals[i*3];
        normal_data[(*(index_vec))[i]+1] = normals[i*3 +1];
        normal_data[(*(index_vec))[i]+2] = normals[i*3 +2];
    }
    mmf_normals.close();
    delete index_vec;
}

size_t countLines(string filename)
{
    // Open file for reading
    ifstream in(filename.c_str());

    // Count lines in file
    size_t c = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        c++;
    }
    in.close();
    return c;
}

