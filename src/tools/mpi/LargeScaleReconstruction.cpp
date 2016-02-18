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
#include <boost/algorithm/string.hpp>
#ifdef LVR_USE_PCL
#include <lvr/reconstruction/PCLKSurface.hpp>
#endif
namespace mpi = boost::mpi;
using namespace std;
using  namespace lvr;
typedef ColorVertex<float, unsigned char> cVertex;
typedef Normal<float> cNormal;
typedef PointsetSurface<ColorVertex<float, unsigned char> > psSurface;
typedef AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> > akSurface;
#ifdef LVR_USE_PCL
typedef PCLKSurface<ColorVertex<float, unsigned char> , Normal<float> > pclSurface;
#endif
enum MPIMSGSTATUS {DATA, FINISHED};

bool nodePtrCompare(LargeScaleOctree* rhs, LargeScaleOctree* lhs)
{
    return (*lhs) < (*rhs);
}

void getNeighborsOnSide(Vertexf dir, vector<std::pair<Vertexf, LargeScaleOctree*> >& neighbors, LargeScaleOctree* currentNode)
{
    if(currentNode->isLeaf())
    {
        neighbors.push_back(std::pair<Vertexf, LargeScaleOctree*>(dir*-1,currentNode));
        return;
    }
    int ids[4];
    if(dir.x == 1)
    {
        ids[0] = 4;
        ids[1] = 5;
        ids[2] = 6;
        ids[3] = 7;
    }
    else if(dir.x == -1)
    {
        ids[0] = 0;
        ids[1] = 1;
        ids[2] = 2;
        ids[3] = 3;
    }
    else if(dir.y == 1)
    {
        ids[0] = 2;
        ids[1] = 3;
        ids[2] = 6;
        ids[3] = 7;
    }
    else if(dir.y == -1)
    {
        ids[0] = 0;
        ids[1] = 1;
        ids[2] = 4;
        ids[3] = 5;
    }
    else if(dir.z == 1)
    {
        ids[0] = 1;
        ids[1] = 3;
        ids[2] = 5;
        ids[3] = 7;
    }
    else if(dir.z == -1)
    {
        ids[0] = 0;
        ids[1] = 2;
        ids[2] = 4;
        ids[3] = 6;
    }
    for(int i = 0 ; i<4 ;i++)
    {
        getNeighborsOnSide(dir, neighbors, (currentNode->getChildren()[ids[i]]));
    }
}

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
        cout << lvr::timestamp << box << endl << "--------" << endl;

        LargeScaleOctree octree(center, size, options.getOctreeNodeSize() );

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

        cout << lvr::timestamp << "...Octree finished" << endl;
        clock_t end = clock();
        vector<LargeScaleOctree*> nodes = octree.getNodes() ;
        vector<LargeScaleOctree*> leafs;
        vector<LargeScaleOctree*> originleafs;

        //auto it = std::copy_if (nodes.begin(), nodes.end(), leafs.begin(), [](LargeScaleOctree* oc){return oc->isLeaf();} );
        size_t minSize = std::max(std::max(options.getKn(), options.getKd()), options.getKi());
        cout << "min size: " << options.getKn() << endl;
	    for(int i = 0 ; i< nodes.size() ; i++)
        {
            if(nodes[i]->isLeaf() && nodes[i]->getSize()>minSize )
            {
                leafs.push_back(nodes[i]);

            }
        }
        std::string firstpath = leafs[0]->getFilePath();
        std::sort(leafs.begin(), leafs.end(), nodePtrCompare);
        for(int i = 0 ; i< leafs.size(); i++)
        {
            cout << lvr::timestamp << leafs[i]->getFilePath() << " size: " << leafs[i]->getSize() << endl;
        }

        //leafs.resize(std::distance(nodes.begin(),it));  // shrink container to new size
        cout << lvr::timestamp << "...got leafs, amount = " <<  leafs.size()<< endl;
        originleafs = leafs;
        //Creating neighbor map
        std::map<string,vector<std::pair<Vertexf, LargeScaleOctree*> > > nmap;
        for(LargeScaleOctree* OTNode : leafs)
        {
            Vertexf center = OTNode->getCenter();
            float radius = OTNode->getLength()/2;
            const Vertexf directions[] = {{0,0,1}, {0,1,0}, {1,0,0}, {0,0,-1}, {0,-1,0}, {-1,0,0}};
            vector<std::pair<Vertexf, LargeScaleOctree*> > currentNeigbors;
            LargeScaleOctree* currentNode = &octree;
            Vertexf max = octree.getCenter() + Vertexf(1,1,1)*(octree.getLength()/2);
            Vertexf min = octree.getCenter() + Vertexf(-1,-1,-1)*(octree.getLength()/2);
            //get neighbor for each direction (left, right, up, down, front, back)
            for(int i = 0 ; i<6 ; i++)
            {
                LargeScaleOctree* currentNode = &octree;
                Vertexf dirPoint = center + (directions[i]*(radius+(radius/2)));
                int depth = 0;

                while (!currentNode->isLeaf())
                {
                    if( dirPoint.x > max.x || dirPoint.y > max.y || dirPoint.z > max.z ||
                        dirPoint.x < min.x || dirPoint.y < min.y || dirPoint.z < min.z )
                    {
                        break;
                    }
                    //Check if neighbor Node is the same size
                    if(fabs(currentNode->getLength() - OTNode->getLength())<=1 || currentNode->isLeaf()) break;
                    int nextChildNode = currentNode->getOctant(dirPoint);
                    currentNode = currentNode->getChildren()[nextChildNode];
                    //currentNode = &(currentNode->getChildren()[currentNode->getOctant(dirPoint)]);
                    depth++;
                    if(currentNode == 0) {
                        cout <<"no!" << endl;
                        break;
                    }

                }
                if(depth>0 && currentNode->isLeaf() && currentNode!=0)
                {
                    currentNeigbors.push_back(std::pair<Vertexf, LargeScaleOctree*>(directions[i],currentNode));
                }
                else if(!(dirPoint.x > max.x || dirPoint.y > max.y || dirPoint.z > max.z ||
                        dirPoint.x < min.x || dirPoint.y < min.y || dirPoint.z < min.z ))
                {
                    getNeighborsOnSide(directions[i]*-1, currentNeigbors,  currentNode);
                }


            }
            if(! currentNeigbors.empty())
            {
                nmap[OTNode->getFilePath()] = currentNeigbors;
            }

        }

        /*cout << "MAP size: " << nmap.size() << endl;
        for(auto it = nmap.begin() ; it != nmap.end() ; it++)
        {
            cout << it->first << " : " ;
            for(auto n : it->second)
            {
                cout << n->getFilePath() << " ";
            }
            cout << endl;
        }*/


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

        //test
        /*firstpath.pop_back();
        firstpath.pop_back();
        firstpath.pop_back();
        firstpath.append("grid");
        cout << "fpath: " << firstpath << endl;
        HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > hg(firstpath);
        cout << hg.getQueryPoints().size() << " " << hg.getNumberOfCells() << endl;
        hg.serialize("test.grid");
         */

        int latticeDirID[6][4] =
        {
            {2,6,1,5},
            {3,7,2,6},
            {7,4,5,6},
            {0,4,3,7},
            {4,0,1,5},
            {0,1,2,3}
        };

        for(auto it = nmap.begin() ; it != nmap.end() ; it++)
        {

            string mainPath = it->first;
            boost::replace_all(mainPath, "xyz", "grid");
            HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > mainGrid(mainPath);
            BoundingBox<ColorVertex<float, unsigned char> > & mbb = mainGrid.getBoundingBox();
            Vertexf maxMainIndices(mainGrid.getMaxIndexX(), mainGrid.getMaxIndexY(), mainGrid.getMaxIndexZ());
            for(auto neighbor : it->second)
            {

                string neighborPath = neighbor.second->getFilePath();
                cout << "interpolating points of " << it->first << " with: " << neighborPath<< endl;
                boost::replace_all(neighborPath, "xyz", "grid");


                if(boost::filesystem::exists(neighborPath))
                {
                    HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > neighborGrid(neighborPath);

                    BoundingBox<ColorVertex<float, unsigned char> > & nbb = neighborGrid.getBoundingBox();
                    ColorVertex<float, unsigned char> posDiff = nbb.getMin() - mbb.getMin();
                    Vertexf & dir = neighbor.first;
                    Vertex<int> diri(dir.x, dir.y, dir.z);
                    diri = diri * -1;
                    size_t x,y,z;
                    size_t  maxx, maxy, maxz;

                    size_t mainAxisId=0;
                    if     ((diri*-1).x == 1)  mainAxisId = mainGrid.getMaxIndexX();
                    else if((diri*-1).y == 1)  mainAxisId = mainGrid.getMaxIndexY();
                    else if((diri*-1).z == 1)  mainAxisId = mainGrid.getMaxIndexZ();


                    x = y = z = 0;
                    maxx = neighborGrid.getMaxIndexX();
                    maxy = neighborGrid.getMaxIndexY();
                    maxz = neighborGrid.getMaxIndexZ();

                    int lpSideId=0;
                    int lpSideId2 = 0;

                    if		(diri.x == 1)
                    {
                        x = maxx;
                        lpSideId = 0;
                        lpSideId2 = 5;
                    }
                    else if (diri.y == 1)
                    {
                        y = maxy;
                        lpSideId = 1;
                        lpSideId2 = 4;
                    }
                    else if (diri.z == 1)
                    {
                        z = maxz;
                        lpSideId = 2;
                        lpSideId2 = 3;
                    }
                    else if (diri.x == -1)
                    {
                        x = 0;
                        maxx = 0;
                        lpSideId = 3;
                        lpSideId2 = 2;
                    }
                    else if (diri.y == -1)
                    {
                        y = 0;
                        maxy = 0;
                        lpSideId = 4;
                        lpSideId2 = 1;
                    }
                    else if (diri.z == -1)
                    {
                        z = 0;
                        maxz = 0;
                        lpSideId = 5;
                        lpSideId2 = 0;
                    }
                    for(int i = x ; i<=maxx ; i++)
                    {
                        for(int j = y ; j<=maxy; j++)
                        {

                            for(int k = z ; k<=maxz; k++)
                            {
                                bool abbruch = false;
                                float distMW=0;
                                for(int l = 0 ; l<4 && !abbruch; l++)
                                {
                                    size_t qp_ID = neighborGrid.findQueryPoint(latticeDirID[lpSideId][l],x,y,z);


                                    if (qp_ID == FastBox<ColorVertex<float, unsigned char>, Normal<float> >::INVALID_INDEX)
                                    {
                                        abbruch = true;
                                        break;
                                    }
                                    float  distN = neighborGrid.getQueryPoints()[qp_ID].m_distance;
                                    distMW +=distN;
                                    Vertex<int> nv(diri);
                                    if(nv.x == 0) nv.x = 1;
                                    else if(nv.x == 1 || nv.x == -1) nv.x = 0;
                                    if(nv.y == 0) nv.y = 1;
                                    else if(nv.y == 1 || nv.y == -1) nv.y = 0;
                                    if(nv.z == 0) nv.z = 1;
                                    else if(nv.z == 1 || nv.z == -1) nv.z = 0;

                                    Vertex<int> mainCellCoord(x,y,z);
                                    mainCellCoord.x *= nv.x;
                                    mainCellCoord.y *= nv.y;
                                    mainCellCoord.z *= nv.z;

                                    if(mainCellCoord.x == 0) mainCellCoord.x = mainAxisId;
                                    else if(mainCellCoord.y == 0) mainCellCoord.y = mainAxisId;
                                    else if(mainCellCoord.z == 0) mainCellCoord.z = mainAxisId;

                                    size_t qpMG_ID = mainGrid.findQueryPoint(latticeDirID[lpSideId2][l],mainCellCoord.x,mainCellCoord.y,mainCellCoord.z);


                                    if (qpMG_ID == FastBox<ColorVertex<float, unsigned char>, Normal<float> >::INVALID_INDEX)
                                    {
                                        break;
                                        abbruch = true;
                                    }
                                    float  distMGN = mainGrid.getQueryPoints()[qpMG_ID].m_distance;
                                    distMW +=distMGN;
                                    if(!abbruch && l ==3)
                                    {
                                        distMW = distMW/8;
                                        for(int m = 0; m<4 ;m++)
                                        {
                                            size_t qp_ID = neighborGrid.findQueryPoint(latticeDirID[lpSideId][m],x,y,z);
                                            neighborGrid.getQueryPoints()[qp_ID].m_distance = distMW;
                                            size_t qpMG_ID = mainGrid.findQueryPoint(latticeDirID[lpSideId2][m],mainCellCoord.x,mainCellCoord.y,mainCellCoord.z);
                                            mainGrid.getQueryPoints()[qpMG_ID].m_distance = distMW;



                                        }
                                    }



                                }

                            }

                        }
                    }
                    cout << timestamp <<" saving grid " << neighborPath << endl;
                    neighborGrid.serialize(neighborPath);
                }

            }
            cout << timestamp <<" saving grid " << mainPath << endl;
            mainGrid.serialize(mainPath);
        }

        std::vector<string> grids(nmap.size());

        for(int i = 0 ; i<originleafs.size() ;i++)
        {
            string mainPath = originleafs[i]->getFilePath();
            boost::replace_all(mainPath, "xyz", "grid");
            grids.push_back(mainPath);

        }


        while(! waitingfor.empty()) waitingfor.pop();
        for(int i = 1 ; i< world.size() && !grids.empty() ; i++)
        {
            cout << "...sending to " << i << " do:  " << grids.back() << endl;
            world.send(i, DATA, grids.back());
            waitingfor.push(1);
            cout << "...poping to " << i << endl;
            grids.pop_back();
        }
        cout << "...got grids" << endl;
        //WHILE DATA to send

        while(! grids.empty() || waitingfor.size()>0)
        {
            cout << "grids left: " << grids.size() << endl;
            string msg;
            mpi::status requests = world.recv(boost::mpi::any_source, FINISHED, msg);
            waitingfor.pop();
            cout << "######## rank 0 got message" << endl;

            //cout << "######## testing " << requests.test()<< endl;
            int rid = requests.source();
            cout << "from" << requests.source() << endl;
            cout << "QQQQQQQQQQQQQQQQQQQQQQQQ " << waitingfor.size() << "|" << grids.size() << endl;
            if(! grids.empty())
            {
                world.send(rid, DATA, grids.back());
                grids.pop_back();
                waitingfor.push(1);
            }
            else if(waitingfor.size()==0)
            {
                for(int i = 1 ; i< world.size() && !grids.empty() ; i++)
                {
                    cout << "...sending to " << i << " finished"  << endl;
                    world.send(i, DATA, "ready");

                }
                break;
            }

        }
        cout << "FINESHED in " << lvr::timestamp << endl;
        world.abort(0);
    }
        //---------------------------------------------
        // SLAVE NODE
        //--------------------------------------------
    else
    {
        bool ready = false;
        while(! ready)
        {
            std::string filePath;
            world.recv(0, DATA, filePath);
            if(filePath=="ready") break;
            std::cout << "NODE: " << world.rank() << " will use file: " << filePath << endl;


            //If filePath does not contain "grid" it will generate a grid
            // else ist will generate a mesh from a given grid file
            if(filePath.find("xyz") != std::string::npos)
            {
                ModelPtr model = ModelFactory::readModel( filePath );
                PointBufferPtr p_loader;
                if ( !model )
                {
                    cout << timestamp << "IO Error: Unable to parse " << filePath << endl;
                    exit(-1);
                }
                p_loader = model->m_pointCloud;

                string pcm_name = options.getPCM();
                psSurface::Ptr surface;

                // Create point set surface object
                if(pcm_name == "PCL")
                {
#ifdef LVR_USE_PCL
                    surface = psSurface::Ptr( new pclSurface(p_loader));
#else
                    cout << timestamp << "Can't create a PCL point set surface without PCL installed." << endl;
			exit(-1);
#endif
                }
                else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
                {
                    akSurface* aks = new akSurface(
                            p_loader, pcm_name,
                            options.getKn(),
                            options.getKi(),
                            options.getKd(),
                            options.useRansac(),
                            options.getScanPoseFile()
                    );

                    surface = psSurface::Ptr(aks);
                    // Set RANSAC flag
                    if(options.useRansac())
                    {
                        aks->useRansac(true);
                    }
                }
                else
                {
                    cout << timestamp << "Unable to create PointCloudManager." << endl;
                    cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
                    cout << timestamp << "Available PCMs are: " << endl;
                    cout << timestamp << "STANN, STANN_RANSAC";
#ifdef LVR_USE_PCL
                    cout << ", PCL";
#endif
#ifdef LVR_USE_NABO
                    cout << ", Nabo";
#endif
                    cout << endl;
                    return 0;
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
                    string out = filePath;
                    out.pop_back();
                    out.pop_back();
                    out.pop_back();
                    out.append("grid");
                    ps_grid->serialize(out);

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
            }
            // Create Mesh from Grid
            else if(filePath.find("grid") != std::string::npos)
            {
                cout << "going to rreconstruct " << filePath << endl;
                string cloudPath = filePath;
                boost::algorithm::replace_last(cloudPath, "grid", "xyz");
                ModelPtr model = ModelFactory::readModel(cloudPath );
                PointBufferPtr p_loader;
                if ( !model )
                {
                    cout << timestamp << "IO Error: Unable to parse " << filePath << endl;
                    exit(-1);
                }
                p_loader = model->m_pointCloud;
                cout << "loaded " << cloudPath << " with : "<< model->m_pointCloud->getNumPoints() << endl;

                string pcm_name = options.getPCM();
                psSurface::Ptr surface;

                // Create point set surface object
                if(pcm_name == "PCL")
                {
#ifdef LVR_USE_PCL
                    surface = psSurface::Ptr( new pclSurface(p_loader));
#else
                    cout << timestamp << "Can't create a PCL point set surface without PCL installed." << endl;
			exit(-1);
#endif
                }
                else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
                {
                    akSurface* aks = new akSurface(
                            p_loader, pcm_name,
                            options.getKn(),
                            options.getKi(),
                            options.getKd(),
                            options.useRansac(),
                            options.getScanPoseFile()
                    );

                    surface = psSurface::Ptr(aks);
                    // Set RANSAC flag
                    if(options.useRansac())
                    {
                        aks->useRansac(true);
                    }
                }
                else
                {
                    cout << timestamp << "Unable to create PointCloudManager." << endl;
                    cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
                    cout << timestamp << "Available PCMs are: " << endl;
                    cout << timestamp << "STANN, STANN_RANSAC";
#ifdef LVR_USE_PCL
                    cout << ", PCL";
#endif
#ifdef LVR_USE_NABO
                    cout << ", Nabo";
#endif
                    cout << endl;
                    return 0;
                }

                surface->setKd(options.getKd());
                surface->setKi(options.getKi());
                surface->setKn(options.getKn());
                HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );
                // Set recursion depth for region growing
                if(options.getDepth())
                {
                    mesh.setDepth(options.getDepth());
                }

                HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > mainGrid(filePath);
                string out2 = filePath;
                boost::algorithm::replace_first(out2, ".grid", "-2.grid");
                mainGrid.saveGrid(out2);
                cout << "finished reading the grid " << filePath << endl;
                FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
                reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(&mainGrid);
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


                if ( options.retesselate() )
                {
                    mesh.finalizeAndRetesselate(options.generateTextures(), options.getLineFusionThreshold());
                }
                else
                {
                    mesh.finalize();
                }
                ModelPtr m( new Model( mesh.meshBuffer() ) );

                string output = filePath;
                boost::algorithm::replace_first(output, "grid", "ply");
                ModelFactory::saveModel( m, output);
            }

            world.send(0, FINISHED, std::string("world"));
            cout << timestamp << "Node: " << world.rank() << "finished "  << endl;
        }

    }

    return 0;
}
