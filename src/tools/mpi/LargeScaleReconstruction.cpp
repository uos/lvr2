//
// Created by eiseck on 17.12.15.
//
#include <fstream>
#include "NodeData.hpp"
#include <algorithm>
#include "LargeScaleOctree.hpp"
#include <string>
#include <sstream>
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
#include <boost/serialization/string.hpp>
#include <lvr/geometry/Vertex.hpp>
#include <iostream>
#include <string>
#include <boost/serialization/string.hpp>
#include <lvr/geometry/ColorVertex.hpp>
#include <lvr/geometry/Normal.hpp>
#include <lvr/reconstruction/PointsetSurface.hpp>
#include <lvr/reconstruction/AdaptiveKSearchSurface.hpp>
#include <lvr/config/lvropenmp.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <lvr/reconstruction/HashGrid.hpp>
#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/reconstruction/PointsetGrid.hpp>
#include <boost/optional/optional_io.hpp>

namespace mpi = boost::mpi;
using namespace std;
using  namespace lvr;
typedef ColorVertex<float, unsigned char> cVertex;
typedef Normal<float> cNormal;
typedef PointsetSurface<ColorVertex<float, unsigned char> > psSurface;
typedef AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> > akSurface;
enum MPIMSGSTATUS {DATA, FINISHED};

int main(int argc, char* argv[])
{
    OpenMPConfig::setNumThreads(1);
    mpi::environment env;
    mpi::communicator world;

/*    ifstream ifs(argv[1]);
    ofstream ofs("schloss4.xyz");
    string str;
    int i =0;
    while( getline( ifs, str ) )
    {
        if(i%3==0)
        {
            stringstream ss;
            ss.str(str);
            Vertexf v;
            ss >> v.x >> v.y >> v.z;
            ofs << v.x << v.y << v.z;
        }

    }*/

    //---------------------------------------------
    // MASTER NODE
    //---------------------------------------------
    if (world.rank() == 0)
    {
        cout << lvr::timestamp << "start" << endl;
        clock_t begin = clock();
        ifstream inputData(argv[1]);
        string s;
        BoundingBox<Vertexf> box;
        int j = 1;
        while( getline( inputData, s ) )
        {
            stringstream ss;
            ss.str(s);
            Vertexf v;
            ss >> v.x >> v.y >> v.z;
            box.expand(v);

        }

        float size = box.getLongestSide();
        Vertexf center = box.getMax()+box.getMin();
        center/=2;
        cout << box << endl << "--------" << endl;

        LargeScaleOctree octree(center, size);

        inputData.close();
        inputData.clear();
        ifstream inputData2(argv[1]);

        while( getline( inputData2, s ) )
        {
            stringstream ss;
            ss.str(s);
            Vertexf v;
            ss >> v.x >> v.y >> v.z;
            octree.insert(v);

        }

        cout << "...Octree finished" << endl;
        clock_t end = clock();
        vector<LargeScaleOctree*> nodes = octree.getNodes() ;
        vector<LargeScaleOctree*> leafs;

        //auto it = std::copy_if (nodes.begin(), nodes.end(), leafs.begin(), [](LargeScaleOctree* oc){return oc->isLeaf();} );
        for(int i = 0 ; i< nodes.size() ; i++)
        {
            if(nodes[i]->isLeaf() && nodes[i]->getSize()>=100 ) leafs.push_back(nodes[i]);
        }
        //leafs.resize(std::distance(nodes.begin(),it));  // shrink container to new size
        cout << "...Octree finished" << endl;
        cout << "...got leafs1" << endl;
        stack<char> waitingfor;
        for(int i = 1 ; i< world.size() && !leafs.empty() ; i++)
        {
            cout << "...sending to " << i << " do:  " << leafs.back()->getFilePath() << endl;
            world.send(i, DATA, leafs.back()->getFilePath());
            waitingfor.push(1);
            cout << "...poping to " << i << endl;
            leafs.pop_back();
        }
        cout << "...got leafs" << endl;
        //WHILE DATA to send

        while(! leafs.empty() || waitingfor.size()>0)
        {
            cout << "leafs left: " << leafs.size() << endl;
            string msg;
            mpi::status requests = world.recv(boost::mpi::any_source, FINISHED, msg);
            waitingfor.pop();
            cout << "######## rank 0 got message" << endl;

            //cout << "######## testing " << requests.test()<< endl;
            int rid = requests.source();
            cout << "from" << requests.source() << endl;
            cout << "QQQQQQQQQQQQQQQQQQQQQQQQ " << waitingfor.size() << "|" << leafs.size() << endl;
            if(! leafs.empty())
            {
                world.send(rid, DATA, leafs.back()->getFilePath());
                leafs.pop_back();
                waitingfor.push(1);
            }
            else if(waitingfor.size()==0)
            {
                for(int i = 1 ; i< world.size() && !leafs.empty() ; i++)
                {
                    cout << "...sending to " << i << " finished"  << endl;
                    world.send(i, DATA, "ready");

                }
                break;
            }

        }

        cout << "FINESHED in " << lvr::timestamp  << endl;
        world.abort(0);
    }
        //---------------------------------------------
        // SLAVE NODE
        //---------------------------------------------
    else
    {
        bool ready = false;
        while(! ready)
        {
            std::string filePath;
            world.recv(0, DATA, filePath);
            if(filePath=="ready") break;
            std::cout << "NODE: " << world.rank() << " will use file: " << filePath << endl;
            ModelPtr model = ModelFactory::readModel( filePath );
            PointBufferPtr p_loader;
            if ( !model )
            {
                cout << timestamp << "IO Error: Unable to parse " << filePath << endl;
                exit(-1);
            }
            p_loader = model->m_pointCloud;

            psSurface::Ptr surface;
            akSurface* aks = new akSurface(
                    p_loader, "STANN",
                    20,
                    60,
                    60,
                    true,
                    ""
            );

            surface = psSurface::Ptr(aks);
            aks->useRansac(true);
            surface->setKd(20);
            surface->setKi(60);
            surface->setKn(60);
            surface->calculateSurfaceNormals();
            HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );
            float resolution;
            bool useVoxelsize;
            resolution = 0.05;
            useVoxelsize = true;
            string decomposition = "PMC";
            GridBase* grid;
            FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
            grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize);
            grid->setExtrusion(true);
            BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
            PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
            ps_grid->calcDistanceValues();

            reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);

            reconstruction->getMesh(mesh);
            //mesh.cleanContours(2);
            mesh.setClassifier("PlaneSimpsons");
            mesh.getClassifier().setMinRegionSize(0.1);

            mesh.optimizePlanes(3,
                                0.85,
                                7,
                                0,
                                true);

            mesh.fillHoles(1);
            mesh.optimizePlaneIntersections();
            mesh.restorePlanes(1);

            mesh.finalize();
            ModelPtr m( new Model( mesh.meshBuffer() ) );
            cout << timestamp << "Saving mesh." << endl;
            string output = filePath;
            output.pop_back();
            output.pop_back();
            output.pop_back();
            output.append("ply");
            ModelFactory::saveModel( m, output);
            world.send(0, FINISHED, std::string("world"));
        }

    }

    return 0;
}
