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

    //---------------------------------------------
    // MASTER NODE
    //---------------------------------------------
    if (world.rank() == 0)
    {
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
        vector<LargeScaleOctree*> leafs(nodes.size());

        auto it = std::copy_if (nodes.begin(), nodes.end(), leafs.begin(), [](LargeScaleOctree* oc){return oc->isLeaf();} );
        leafs.resize(std::distance(nodes.begin(),it));  // shrink container to new size


        for(int i = 1 ; i< world.size() && !leafs.empty() ; i++)
        {
            world.send(i, DATA, leafs.back()->getFilePath());
            leafs.pop_back();
        }
        //WHILE DATA to send
        while(! leafs.empty())
        {
            cout << "leafs left: " << leafs.size() << endl;
            string msg;
            mpi::request requests = world.irecv(boost::mpi::any_source, FINISHED, msg);
            mpi::wait_all(&requests, &requests);
            int rid = requests.test()->source();
            if(! leafs.empty())
            {
                world.send(rid, DATA, leafs.back()->getFilePath());
                leafs.pop_back();
            }

        }
        double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
        cout << "FINESHED in " << elapsed_secs << " Seconds." << endl;

    }
        //---------------------------------------------
        // SLAVE NODE
        //---------------------------------------------
    else
    {
        std::string filePath;
        world.recv(0, DATA, filePath);
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
                10,
                10,
                5,
                false,
                ""
        );

        surface = psSurface::Ptr(aks);
        surface->setKd(10);
        surface->setKi(10);
        surface->setKn(5);
        surface->calculateSurfaceNormals();
        HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );
        float resolution;
        bool useVoxelsize;
        resolution = 10;
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
        mesh.cleanContours(0);
        mesh.setClassifier("PlaneSimpsons");
        mesh.getClassifier().setMinRegionSize(10);
        mesh.finalize();
        ModelPtr m( new Model( mesh.meshBuffer() ) );
        cout << timestamp << "Saving mesh." << endl;
        string output = filePath;
        output.pop_back();
        output.pop_back();
        output.pop_back();
        output.append(".ply");
        ModelFactory::saveModel( m, output);
        world.send(0, 1, std::string("world"));
    }

    return 0;
}
