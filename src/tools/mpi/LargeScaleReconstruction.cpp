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
#include <lvr/geometry/QuadricVertexCosts.hpp>
#include "Options.hpp"

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
    mpi::environment env;
    mpi::communicator world;
    Largescale::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if ( options.printUsage() )
    {
        return 0;
    }
    OpenMPConfig::setNumThreads(options.getNumThreads());
    //std::cout << options << std::endl;

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
        std::cout << options << std::endl;
        cout << lvr::timestamp << "start" << endl;
        clock_t begin = clock();
        ifstream inputData(options.getInputFileName());
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
        ifstream inputData2(options.getInputFileName());

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
            string pcm_name = options.getPCM();
            p_loader = model->m_pointCloud;

            psSurface::Ptr surface;
            akSurface* aks = new akSurface(
                    p_loader, "STANN",
                    options.getKn(),
                    options.getKi(),
                    options.getKd(),
                    options.useRansac(),
                    options.getScanPoseFile()
            );

            surface = psSurface::Ptr(aks);
            if(options.useRansac())
            {
                aks->useRansac(true);
            }
            surface->setKd(options.getKd());
            surface->setKi(options.getKi());
            surface->setKn(options.getKn());
            surface->calculateSurfaceNormals();

            HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );
            // Set recursion depth for region growing
            if(options.getDepth())
            {
                mesh.setDepth(options.getDepth());
            }
            if(options.getSharpFeatureThreshold())
            {
                SharpBox<Vertex<float> , Normal<float> >::m_theta_sharp = options.getSharpFeatureThreshold();
            }
            if(options.getSharpCornerThreshold())
            {
                SharpBox<Vertex<float> , Normal<float> >::m_phi_corner = options.getSharpCornerThreshold();
            }

            float resolution;
            bool useVoxelsize;
            if(options.getIntersections() > 0)
            {
                resolution = options.getIntersections();
                useVoxelsize = false;
            }
            else
            {
                resolution = options.getVoxelsize();
                useVoxelsize = true;
            }
            string decomposition = options.getDecomposition();
            GridBase* grid;
            FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
            if(decomposition == "MC")
            {
                grid = new PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize);
                grid->setExtrusion(options.extrude());
                PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->calcDistanceValues();

                reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);

            }
            else if(decomposition == "PMC")
            {
                grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize);
                grid->setExtrusion(options.extrude());
                BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->calcDistanceValues();

                reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);

            }
            else if(decomposition == "SF")
            {
                SharpBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                grid = new PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, surface->getBoundingBox(), useVoxelsize);
                grid->setExtrusion(options.extrude());
                PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->calcDistanceValues();
                reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
            }

            reconstruction->getMesh(mesh);
            //mesh.cleanContours(2);
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
