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
enum ClientServerMsgs{READY, BUSY,STOP, START, BBOX, SIZE};
void save_normals(floatArr normals, size_t n, std::vector<unsigned long long int> index_vec );

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



        LSTree tree(mmf_path, 100, 140000);

        cout << timestamp << "Tree built" << " running MPI on 1 server and " << world.size()-1 << " clients" << endl;
        vector<bool> clients_ready(world.size()-1,true);
        int next_element=0;
        bool finished = false;
        std::vector<mpi::request> mpi_requests;
        std::vector<mpi::request> mpi_requests_ready(world.size()-1);

        //Send Bounding Box:
        float fbbox[6];
        fbbox[0]= tree.getM_bbox().getMin().x;
        fbbox[1]= tree.getM_bbox().getMin().y;
        fbbox[2]= tree.getM_bbox().getMin().z;
        fbbox[3]= tree.getM_bbox().getMax().x;
        fbbox[4]= tree.getM_bbox().getMax().y;
        fbbox[5]= tree.getM_bbox().getMax().z;
        cout << timestamp << "Sending Timestamp" << endl;
        for(int i = 0; i<clients_ready.size() ; i++)
        {

            world.send(i+1, BBOX, fbbox);

        }
        cout << timestamp << "starting loop " << clients_ready.size() << endl;
        while(next_element<tree.getM_nodes().size())
        {
            //Check if clients need new data
            for(int i = 0; i<clients_ready.size() ; i++)
            {

                if(clients_ready[i])
                {
                    cout << "yes"<< endl;

                    cout << "sending " << tree.getM_nodes()[next_element]->getPoints().size()<< " to " << i+1 << endl;
                    world.isend(i+1, START, tree.getM_nodes()[next_element]->getPoints());
                    cout << 2 << endl;
                    mpi_requests_ready[i] = world.irecv(i+1, READY);
                    cout << 3 << endl;
                    clients_ready[i] = false;
                    cout << 4 << endl;
                    next_element++;
                }

            }

            for(int i = 0 ; i<mpi_requests_ready.size() ; i++) {
                /* std::cout << *it; ... */
                if (mpi_requests_ready[i].test())
                {
                    cout << "'bla" << endl;
                    cout << timestamp << i+1 << " is Ready again" << endl;
                    clients_ready[i] = true;

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

        int size;
        float bbox[6];
        world.recv(0,BBOX,bbox);
        cout << timestamp << "Client "<< world.rank()<< " got BoundingBox" << endl;
        mpi::request reqs[3];
        std::vector<unsigned long long int>  index_vec;
        reqs[0] = world.irecv(0,START,index_vec);
        reqs[1] = world.irecv(0,STOP);

        while(true)
        {


            cout << timestamp << "Client "<< world.rank()<< " waiting for any" << endl;
            //mpi::wait_any(reqs, reqs+1);
            if(reqs[1].test()) {
                cout << timestamp << "Client "<< world.rank()<< " leaving loop" << endl;
                break;
            }
            if(reqs[0].test())
            {
                std::vector<unsigned long long int>  index_vec2(index_vec);
                cout << timestamp << "Client "<< world.rank()<< " waiting for vec" << endl;

                cout << "Nr " << world.rank() << "got vector with: " << index_vec2.size() << endl;
                vector<float> vertex_array;

                cout << timestamp << "Client "<< world.rank()<< " loading vertices to ram" << endl;
                int j = 0;
                for(int i = 0 ; i<index_vec2.size() ;i++)
                {
                    //cout << mmf_data[index_vec[i]] << "|" << mmf_data[index_vec[i]+1] << "|" << mmf_data[index_vec[i]+2] << endl;


                    float x = mmf_data[index_vec2[i]];
                    float y = mmf_data[index_vec2[i]+1];
                    float z = mmf_data[index_vec2[i]+2];

                    vertex_array.push_back(x);
                    vertex_array.push_back(y);
                    vertex_array.push_back(z);
                    j+=3;
                    if(world.rank() == 1)cout << index_vec2[i] << "," << endl;
                }
                cout << timestamp << "Client "<< world.rank()<< " done loading vertices to ram" << endl;
                boost::shared_array<float> vertices (vertex_array.data());
                PointBufferPtr pointcloud(new PointBuffer());
                pointcloud->setPointArray(vertices, index_vec2.size());
                PointsetSurface<ColorVertex<float, unsigned char> >* surface;
                surface = new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >(pointcloud, "FLANN", 100, 100, 100, false);

                // set global Bounding-Box
                surface->expandBoundingBox(bbox[0], bbox[1], bbox[2],
                                           bbox[3], bbox[4], bbox[5]);



                // calculate the normals
                std::cout << timestamp << "Client " << world.rank() << " calculates surface normals with " <<  index_vec2.size() << " points." <<  std::endl;
                surface->calculateSurfaceNormals();
                pointcloud = surface->pointBuffer();
                size_t n;
                cout << "asdasdasdasd" << endl;
                floatArr normals = pointcloud->getPointNormalArray(n);
                cout << "saving normals to file" << endl;
                //std::vector<unsigned long long int> *  tmp = new std::vector<unsigned long long int>( index_vec2) ;
                save_normals( normals, index_vec2.size(),index_vec2);

                reqs[0] = world.irecv(0,START,index_vec);
                reqs[2] = world.irecv(0,SIZE,size);
                world.send(0, READY);
            }

        }

    }


}

void save_normals(floatArr normals, size_t n, std::vector<unsigned long long int>  index_vec )
{
    boost::iostreams::mapped_file_params mfp;
    mfp.new_file_size = n*3* sizeof(float);
    mfp.path = "normals.raw";

    boost::iostreams::mapped_file_sink mmf_normals(mfp);
    float * normal_data = (float *)mmf_normals.data();

    for(int i = 0 ; i<n ;i++)
    {
        cout << index_vec[i] << endl;
        normal_data[index_vec[i]]  = 1;
        normal_data[index_vec[i]+1] = 2;
        normal_data[index_vec[i]+2] = 3;
    }
    cout << "test" <<endl;
    mmf_normals.close();
    //delete index_vec;
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

