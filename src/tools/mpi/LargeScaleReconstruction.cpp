//
// Created by eiseck on 17.12.15.
//
#include <cstdio>
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
#include <lvr/io/Model.hpp>
#include <lvr/io/MeshBuffer.hpp>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <lvr/reconstruction/HashGrid.hpp>
#include <lvr/reconstruction/FastReconstruction.hpp>
#include <lvr/reconstruction/PointsetGrid.hpp>
#include <boost/optional/optional_io.hpp>
#include <utility>
#include "DuplicateRemover.hpp"

#include <lvr/geometry/QuadricVertexCosts.hpp>
#include "Options.hpp"
#include <unordered_map>
#include <boost/algorithm/string.hpp>
#include "InterceptionBoundingBox.hpp"
#include "SparseMatrix.hpp"
#include "LineReader.hpp"
#ifdef LVR_USE_PCL
#include <lvr/reconstruction/PCLKSurface.hpp>
#include <lvr/io/PLYIO.hpp>
#include <boost/timer/timer.hpp>
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
Vertexf center;

/**
 *  Function to compare two LargeScaleOctree pointers (needed to sort octree nodes)
 */
bool nodePtrCompare(LargeScaleOctree* rhs, LargeScaleOctree* lhs)
{
    return  (*rhs) < (*lhs);
}

int main(int argc, char* argv[])
{


    // Init MPI Enviroment
    mpi::environment env;
    mpi::communicator world;
    Largescale::Options options(argc, argv);

    // Exit if options had to generate a usage message
    // (this means required parameters are missing)
    if ( options.printUsage() )
    {
        return 0;
    }

    // Some Methods in LVR-Toolkit use OpenMP, The MPI Application should always use one Thred
    OpenMPConfig::setNumThreads(1);

    bool use_ply = false;
    if(boost::algorithm::contains(options.getInputFileName(), ".ply"))
    {
        use_ply = true;
    }

    //---------------------------------------------
    // MASTER NODE
    //---------------------------------------------
    if (world.rank() == 0)
    {

        boost::timer::cpu_timer itimer;
        boost::timer::cpu_timer otimer;
        itimer.stop ();
        otimer.stop ();
        std::cout << options << std::endl;
        cout << lvr::timestamp << "start" << endl;
        cout << env.processor_name() << " is MASTER" << endl;
        boost::filesystem::path p(options.getInputFileName().c_str());

        vector<LargeScaleOctree*> nodes;
        vector<LargeScaleOctree*> leafs;
        vector<LargeScaleOctree*> originleafs;
        vector<string> cloudPahts;
        unordered_map<string, LargeScaleOctree*> nameToLeaf;
        string octreeFolder;
        size_t minSize = std::max(std::max(options.getKn(), options.getKd()), options.getKi());
        string folder_prefix = "";

        // if Path is a directory, check if octree files have already been generated
        if(! boost::filesystem::is_directory(p))
        {
            if(use_ply)
            {
                cout << "USING a PLY file" << endl;
                LineReader lr(options.getInputFileName());

                    size_t rsize;
                    BoundingBox<Vertexf> box;
                    cout << lr.getFileType() << endl;

                        while (true)
                        {
                            if(lr.getFileType() == XYZNRGB)
                            {
                                boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    box.expand(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                }
                            }
                            else if(lr.getFileType() == XYZN)
                            {
                                boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    box.expand(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                }
                            }
                            else if(lr.getFileType() == XYZ)
                            {
                                boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    box.expand(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                }
                            }
                else if(lr.getFileType() == XYZRGB)
                            {
                                boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    box.expand(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                }
                            }


                        }
                        lr.rewind();

                        center = box.getCentroid();

                        // Make sure Bounding Box is not to narrow
                        box.expand(center.x + options.getVoxelsize(), center.y + options.getVoxelsize(), center.z + options.getVoxelsize()  );
                        box.expand(center.x - options.getVoxelsize(), center.y - options.getVoxelsize(), center.z - options.getVoxelsize()  );

                        float size = box.getLongestSide();
                        double newLength = options.getVoxelsize();

                        // The Length of the Bounding Box (all sides have the same length) must be
                        // a multiple of two, to ensure that voxels fit exactly in the volume
                        while(newLength<=size) newLength*=2;

                        cout << "Bounding Box Longest Side: " << newLength << endl;

                        Vertexf newXMax(box.getCentroid().x+(newLength/2),box.getCentroid().y, box.getCentroid().z );
                        Vertexf newXMin(box.getCentroid().x-(newLength/2),box.getCentroid().y, box.getCentroid().z  );

                        // Set new Bounding Box size
                        box.expand(newXMax);
                        box.expand(newXMin);



                        cout << lvr::timestamp << box << endl << "--------" << endl;

                        // Create Octree Root
                        LargeScaleOctree octree(center, newLength, options.getOctreeNodeSize() );
                        octreeFolder = octree.getFolder();

                        while (true)
                        {
                            if(lr.getFileType() == XYZNRGB)
                            {
                                boost::shared_ptr<xyznc> a = boost::static_pointer_cast<xyznc> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    Vertexf v(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                    Vertexf n(a.get()[i].normal.x,a.get()[i].normal.y,a.get()[i].normal.z);
                                    octree.insert(v,n);
                                }
                            }
                            else if(lr.getFileType() == XYZN)
                            {
                                boost::shared_ptr<xyzn> a = boost::static_pointer_cast<xyzn> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    Vertexf v(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                    Vertexf n(a.get()[i].normal.x,a.get()[i].normal.y,a.get()[i].normal.z);
                                    octree.insert(v,n);
                                }
                            }
                            else if(lr.getFileType() == XYZ)
                            {
                                boost::shared_ptr<xyz> a = boost::static_pointer_cast<xyz> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    Vertexf v(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                    octree.insert(v);
                                }
                            }
                else if(lr.getFileType() == XYZRGB)
                            {
                                boost::shared_ptr<xyzc> a = boost::static_pointer_cast<xyzc> (lr.getNextPoints(rsize));
                                if (rsize <= 0 )
                                {
                                    break;
                                }
                                for(int i = 0 ; i< rsize ; i++)
                                {
                                    Vertexf v(a.get()[i].point.x,a.get()[i].point.y,a.get()[i].point.z);
                                    octree.insert(v);
                                }
                            }


                        }
                        octree.writeData();
                        octree.PrintTimer ();

                        nodes = octree.getNodes() ;

                        cout << "nodes size: " << nodes.size() << endl;
                        cout << "min size: " << options.getKn() << endl;


                        // Check if nodes have more then minSize ( = max(ki,kd,kn) ) points
                        for(int i = 0 ; i< nodes.size() ; i++)
                        {
                            if(nodes[i]->isLeaf() && nodes[i]->getSize()>minSize )
                            {
                                leafs.push_back(nodes[i]);

                            }
                        }
                        // Sort Leafs so the Nodes with morge Points are processed first
                        //std::sort(leafs.begin(), leafs.end(), nodePtrCompare);

                        // Tell each leaf to find it's neighbours
                        // And save filepath to leaf pointer in map
                        for(int i = 0 ; i<leafs.size() ; i++)
                        {
                            nameToLeaf[leafs[i]->getFilePath()] = leafs[i];
                            leafs[i]->generateNeighbourhood();

                        }

                        cout << lvr::timestamp << "...got leafs, amount = " <<  leafs.size()<< endl;


                        originleafs = leafs;

                        //Write BoundingBoxes to files
                        for(int i = 0 ; i<originleafs.size() ; i++)
                        {

                            // Write Center of Scan to file (used for normal approximation)
                            string center_path = originleafs[i]->getFilePath();
                            boost::algorithm::replace_last(center_path, "xyz", "-center.pt");
                            ofstream cofs(center_path);
                            otimer.resume ();
                            cofs << center.x << " " << center.y << " " << center.z << endl;
                            otimer.stop ();
                            cofs.close();

                            // Write Bounding Box of Octree Node to file
                            string path = originleafs[i]->getFilePath();
                            boost::algorithm::replace_last(path, "xyz", "bb");
                            float r = originleafs[i]->getWidth()/2;
                            Vertexf rr(r,r,r);

                            // calc min and max point of bounding box from center and length
                            Vertexf min = originleafs[i]->getCenter()-rr;
                            Vertexf max = originleafs[i]->getCenter()+rr;
                            ofstream ofs(path);
                            Vertexf min_new = originleafs[i]->getPointBB().getMin();
                            Vertexf max_new = originleafs[i]->getPointBB().getMax();
                            otimer.resume ();
                            ofs << min[0] << " " << min[1] << " " << min[2]  << " " << max[0]<< " " << max[1] << " " << max[2]<< endl;
                            otimer.stop ();
                            ofs.close();
                        }

                        for(int i = 0 ; i< leafs.size(); i++) cloudPahts.push_back(leafs[i]->getFilePath());



            }
            else
            {
                ifstream inputData(options.getInputFileName());

                //Use FILE for faster reading of large files
                FILE * bbFile = fopen(options.getInputFileName().c_str(),"r");
                BoundingBox<Vertexf> box;
                float ix,iy,iz;

                itimer.resume ();
                // get all points and calc. Bounding Box (axis aligned)
                while( fscanf(bbFile,"%f %f %f", &ix, &iy, &iz) != EOF )
                {
                    itimer.stop ();
                    box.expand(ix,iy,iz);
                    itimer.resume ();
                }
                itimer.stop ();

                center = box.getCentroid();

                // Make sure Bounding Box is not to narrow
                box.expand(center.x + options.getVoxelsize(), center.y + options.getVoxelsize(), center.z + options.getVoxelsize()  );
                box.expand(center.x - options.getVoxelsize(), center.y - options.getVoxelsize(), center.z - options.getVoxelsize()  );

                float size = box.getLongestSide();
                double newLength = options.getVoxelsize();

                // The Length of the Bounding Box (all sides have the same length) must be
                // a multiple of two, to ensure that voxels fit exactly in the volume
                while(newLength<=size) newLength*=2;

                cout << "Bounding Box Longest Side: " << newLength << endl;

                Vertexf newXMax(box.getCentroid().x+(newLength/2),box.getCentroid().y, box.getCentroid().z );
                Vertexf newXMin(box.getCentroid().x-(newLength/2),box.getCentroid().y, box.getCentroid().z  );

                // Set new Bounding Box size
                box.expand(newXMax);
                box.expand(newXMin);



                cout << lvr::timestamp << box << endl << "--------" << endl;

                // Create Octree Root
                LargeScaleOctree octree(center, newLength, options.getOctreeNodeSize() );
                octreeFolder = octree.getFolder();

                rewind(bbFile);
                Vertexf ov;
                // Insert points to Octree, (points can't be loaded to memory first, because the files might be to large)
                // Todo: Program Parameter for smaler files, to load them all into memory first
                itimer.resume ();
                while( fscanf(bbFile,"%f %f %f", &(ov.x), &(ov.y), &(ov.z)) != EOF )
                {
                    itimer.stop ();
                    box.expand(ix,iy,iz);
                    octree.insert(ov);
                    itimer.resume ();
                }
                itimer.stop ();
                fclose(bbFile);

                // makes sure all data has been writen to disk
                octree.writeData();

                cout << lvr::timestamp << "...Octree finished" << endl;

                octree.PrintTimer ();

                nodes = octree.getNodes() ;

                cout << "nodes size: " << nodes.size() << endl;
                cout << "min size: " << options.getKn() << endl;


                // Check if nodes have more then minSize ( = max(ki,kd,kn) ) points
                for(int i = 0 ; i< nodes.size() ; i++)
                {
                    if(nodes[i]->isLeaf() && nodes[i]->getSize()>minSize )
                    {
                        leafs.push_back(nodes[i]);

                    }
                }
                // Sort Leafs so the Nodes with morge Points are processed first
                //std::sort(leafs.begin(), leafs.end(), nodePtrCompare);

                // Tell each leaf to find it's neighbours
                // And save filepath to leaf pointer in map
                for(int i = 0 ; i<leafs.size() ; i++)
                {
                    nameToLeaf[leafs[i]->getFilePath()] = leafs[i];
                    leafs[i]->generateNeighbourhood();

                }

                cout << lvr::timestamp << "...got leafs, amount = " <<  leafs.size()<< endl;


                originleafs = leafs;

                //Write BoundingBoxes to files
                for(int i = 0 ; i<originleafs.size() ; i++)
                {

                    // Write Center of Scan to file (used for normal approximation)
                    string center_path = originleafs[i]->getFilePath();
                    boost::algorithm::replace_last(center_path, "xyz", "-center.pt");
                    ofstream cofs(center_path);
                    otimer.resume ();
                    cofs << center.x << " " << center.y << " " << center.z << endl;
                    otimer.stop ();
                    cofs.close();

                    // Write Bounding Box of Octree Node to file
                    string path = originleafs[i]->getFilePath();
                    boost::algorithm::replace_last(path, "xyz", "bb");
                    float r = originleafs[i]->getWidth()/2;
                    Vertexf rr(r,r,r);

                    // calc min and max point of bounding box from center and length
                    Vertexf min = originleafs[i]->getCenter()-rr;
                    Vertexf max = originleafs[i]->getCenter()+rr;
                    ofstream ofs(path);
                    Vertexf min_new = originleafs[i]->getPointBB().getMin();
                    Vertexf max_new = originleafs[i]->getPointBB().getMax();
                    otimer.resume ();
                    ofs << min[0] << " " << min[1] << " " << min[2]  << " " << max[0]<< " " << max[1] << " " << max[2]<< endl;
                    otimer.stop ();
                    ofs.close();
                }

                for(int i = 0 ; i< leafs.size(); i++) cloudPahts.push_back(leafs[i]->getFilePath());
            }

        }
        // if Path is a folder
        else
        {
            folder_prefix = p.generic_string();
            folder_prefix.append("/");
            cout << "Got Folder as filePath, looking for .xyz files" << endl;
            vector<boost::filesystem::path> ret;
            octreeFolder = p.generic_string();
            if(!boost::filesystem::exists(p) || !boost::filesystem::is_directory(p))
            {
                cout << "no such files" << endl;
                throw (std::ios_base::failure("no such file!") );
            }
            boost::filesystem::recursive_directory_iterator it(p);
            boost::filesystem::recursive_directory_iterator endit;


            //get all octree node files
            while(it != endit)
            {
                if(boost::filesystem::is_regular_file(*it) && it->path().extension() == ".xyz") ret.push_back(it->path().filename());
                ++it;

            }

            // Check if octree file has enough points
            for(int i = 0 ; i< ret.size(); i++)
            {
                string path = folder_prefix;
                path.append(ret[i].generic_string());
                cout << "got xyz: " << ret[i].generic_string() << endl;

                //check if file hast  minSize lines
                int lines = 0;
                std::string line;
                std::ifstream fi(path);
                itimer.resume ();
                while (std::getline(fi, line) && lines <= minSize ) ++lines;
                itimer.stop ();
                if(lines>=minSize) cloudPahts.push_back(path);


            }




        }













        // Path to each file that should be processed, (processed files will be removed when finished)
        vector<string> cloudPathsCopy;

        // check if file exists
        for(int i = 0 ; i< cloudPahts.size() ; i++)
        {
            string gpath = cloudPahts[i];
            boost::algorithm::replace_first(gpath, "xyz", "grid");
            boost::filesystem::path bgpath(gpath);
            if(! boost::filesystem::exists(bgpath)) cloudPathsCopy.push_back(cloudPahts[i]);
        }

        // Number Point clouds (Octree Nodes) beeing processed bei MPI-Nodes
        int waitingfor=0;

        // Send a path (of a point cloud) to each MPI-Node
        for(int i = 1 ; i< world.size() && !cloudPathsCopy.empty() ; i++)
        {
            cout << "...sending to " << i << " do:  " << cloudPathsCopy.back() << endl;
            world.send(i, DATA, cloudPathsCopy.back());
            waitingfor++;
            cout << "...poping to " << i << endl;
            cloudPathsCopy.pop_back();
        }
        cout << "...got leafs" << endl;

        //WHILE still DATA to send
        while(! cloudPathsCopy.empty() || waitingfor>0)
        {
            cout << "leafs left: " << cloudPathsCopy.size() << endl;
            string msg;

            // Wait for MPI-Message that lets us know that the a MPI-Node finished it's calculation
            mpi::status requests = world.recv(boost::mpi::any_source, FINISHED, msg);
            waitingfor--;
            cout << "######## rank 0 got message" << endl;
            // Get ID of MPI-Node that finished
            int rid = requests.source();

            cout << "from" << requests.source() << endl;
            cout << "waiting for: " << waitingfor << "| files left: " << cloudPathsCopy.size() << endl;

            // Send next file to MPI-Node
            if(! cloudPathsCopy.empty())
            {
                world.send(rid, DATA, cloudPathsCopy.back());
                cloudPathsCopy.pop_back();
                waitingfor++;
            }
            // No more files left, only waiting until MPI-Nodes finished
            else if(waitingfor==0)
            {
                for(int i = 1 ; i< world.size() && !cloudPathsCopy.empty() ; i++)
                {
                    cout << "...sending to " << i << " finished"  << endl;
                    world.send(i, DATA, "ready");

                }
                break;
            }

        }

        // All HashGrids have been generated
        // Interpolate Neighbour Grids:

        std::vector<string> grids;

        // Get Path to all grid Files on Disk:
        for(int i = 0 ; i<cloudPahts.size() ;i++)
        {
            string mainPath = cloudPahts[i];
            boost::replace_all(mainPath, "xyz", "grid");
            string plyPath = mainPath;
            boost::replace_all(plyPath, "grid", "ply");
            boost::filesystem::path gridPath(plyPath);
            if(! boost::filesystem::exists(gridPath)) grids.push_back(mainPath);


        }

        // if interpolation flag set:
        if(options.interpolateBoxes())
        {

            // Map Grid file paths to index in grid array
            std::unordered_map<string, size_t> nameToPos;
            for(size_t i = 0 ; i<grids.size() ; i++)
            {
                nameToPos[grids[i]] = i;
            }

            //Sparse Matrix: used to only interpolate two grids once, if Mat(i,j) != 0 grid i and grid j have already been compared
            SparseMatrix compMat(grids.size(),grids.size());

            bool gotGoodNeighbour = false;
            string goodNeighbourPath;
            for(int i = 0 ; i< grids.size(); i++)
            {



                cout << timestamp << "Interpolating Grids: " << (int)((i/grids.size())*100) << "%" << endl;

                string nodepath = grids[i];
                boost::replace_all(nodepath, "grid", "xyz");

                // Get Neighbour of HashGrid from Octree Structure:
                LargeScaleOctree* currentNode = nameToLeaf[nodepath];
                itimer.resume ();
                HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > mainGrid(grids[i]);
                itimer.stop ();
                bool interpMainGrid = false;
                // Interpolate HashGrid with every neighbour
                for(int j = 0 ; j < currentNode->getSavedNeighbours().size(); j++)
                {

                    // Check if Neighbour exists and has not been already compared with
                    boost::filesystem::path bgpath(grids[i]);
                    if(! boost::filesystem::exists(bgpath)) continue;
                    string check_grid_path =  leafs[i]->getSavedNeighbours()[j]->getFilePath();
                    boost::replace_all(check_grid_path, "xyz", "grid");
                    boost::filesystem::path bgpath2(check_grid_path);
                    if(! boost::filesystem::exists(bgpath2)) continue;
                    if( (compMat[nameToPos[grids[i]]][nameToPos[check_grid_path]] != 0) ||
                            (compMat[nameToPos[check_grid_path]][nameToPos[grids[i]]] != 0)) continue;


                    // Load grids that should be interpolated from disk:
                    itimer.resume ();
                    HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > checkgrid(check_grid_path);
                    itimer.stop ();

                    cout << timestamp << "interpolating " << grids[i] << " with:  " << check_grid_path << endl;
                    // Interpolation
                    mainGrid.interpolateEdge(checkgrid);
                    cout << timestamp << "... finished" << endl;

                    // Save interpolated grids to disk

                    otimer.resume ();
                    checkgrid.serialize(check_grid_path);
                    otimer.stop ();
                    // Add Matrix entry
                    compMat.insert(nameToPos[grids[i]],nameToPos[check_grid_path],1);
                    interpMainGrid =  true;

                }
                otimer.resume ();
                if(interpMainGrid) mainGrid.serialize(grids[i]);
                otimer.stop ();
            }
        }


        // Let nodes create mesh from HashGrids:
        waitingfor = 0;
        for(int i = 1 ; i< world.size() && !grids.empty() ; i++)
        {
            cout << "...sending to " << i << " do:  " << grids.back() << endl;
            world.send(i, DATA, grids.back());
            waitingfor++;
            cout << "...poping to " << i << endl;
            grids.pop_back();
        }
        cout << "...got grids" << endl;
        //WHILE DATA to send

        while(! grids.empty() || waitingfor>0)
        {
            cout << "grids left: " << grids.size() << endl;
            string msg;
            mpi::status requests = world.recv(boost::mpi::any_source, FINISHED, msg);
            waitingfor--;
            cout << "######## rank 0 got message" << endl;

            //cout << "######## testing " << requests.test()<< endl;
            int rid = requests.source();
            cout << "from" << requests.source() << endl;
            cout << "waiting for: " << waitingfor << "| grids left: " << grids.size() << endl;
            if(! grids.empty())
            {
                world.send(rid, DATA, grids.back());
                grids.pop_back();
                waitingfor++;
            }
            else if(waitingfor==0)
            {
                for(int i = 1 ; i< world.size() && !grids.empty() ; i++)
                {
                    cout << "...sending to " << i << " finished"  << endl;
                    world.send(i, DATA, "ready");

                }
                break;
            }

        }
        cout << timestamp << "all nodes finished, generating final mesh" << endl;

        // All meshed habe been created, now they need to be merged to a single mesh


        PLYIO io;
        MeshBufferPtr completeMesh(new MeshBuffer);
        // Vertex and Face Buffer for new Mesh
        std::vector<float> vertexArray;
        std::vector<unsigned int> faceArray;

        // Map Grid
        unordered_map<string,size_t> nodePathtoID;
        for(int i = 0 ; i<leafs.size() ; i++)
        {
            string gridpath = leafs[i]->getFilePath();
            boost::replace_all(gridpath, "xyz", "grid");
            boost::filesystem::path bgpath(gridpath);
            if(! boost::filesystem::exists(bgpath)) continue;
            nodePathtoID[gridpath] = i;
        }
        vector<BoundingBox<Vertexf> > intersections;

        float added_vsize=options.getVoxelsize()*3;



        SparseMatrix compMat(leafs.size() ,leafs.size());
        cout << timestamp << " calculating duplicate vertice indices" << endl;
        for(int i = 0 ; i<leafs.size() ; i++)
        {
            string gridpath = leafs[i]->getFilePath();
            boost::replace_all(gridpath, "xyz", "grid");
            boost::filesystem::path bgpath(gridpath);
            if(! boost::filesystem::exists(bgpath)) continue;

            float leaf_rad = leafs[i]->getWidth()/2;
            BoundingBox<Vertexf> bb(
                leafs[i]->getCenter()[0] - leaf_rad - added_vsize,
                leafs[i]->getCenter()[1] - leaf_rad - added_vsize,
                leafs[i]->getCenter()[2] - leaf_rad - added_vsize,
                leafs[i]->getCenter()[0] + leaf_rad + added_vsize,
                leafs[i]->getCenter()[1] + leaf_rad + added_vsize,
                leafs[i]->getCenter()[2] + leaf_rad + added_vsize
            );


            cout << leafs[i]->getFilePath() << "has: " << leafs[i]->getSavedNeighbours().size() << " neighbors" << endl;
            for(int j = 0 ; j < leafs[i]->getSavedNeighbours().size(); j++)
            {
                string rhsgridpath = leafs[i]->getSavedNeighbours()[j]->getFilePath();
                boost::replace_all(rhsgridpath, "xyz", "grid");
                boost::filesystem::path rhsbgpath(rhsgridpath);
                if(! boost::filesystem::exists(rhsbgpath)) continue;
                if(compMat[i][nodePathtoID[rhsgridpath]] != 0 || compMat[nodePathtoID[rhsgridpath]][i] != 0) continue;
                float chec_leaf_rad = leafs[i]->getSavedNeighbours()[j]->getWidth()/2;
                BoundingBox<Vertexf> checkbb(
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[0] - chec_leaf_rad - added_vsize,
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[1] - chec_leaf_rad - added_vsize,
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[2] - chec_leaf_rad - added_vsize,
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[0] + chec_leaf_rad + added_vsize,
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[1] + chec_leaf_rad + added_vsize,
                    leafs[i]->getSavedNeighbours()[j]->getCenter()[2] + chec_leaf_rad + added_vsize
                );

//                cout << "diffboxes: " << bb << endl << checkbb << endl;
                BoundingBox<Vertexf> diffbb = checkbb.getIntersectionBB(bb);

                //expand the box so that the size in each direction is at least +/- voxelsize, otherwise if length in one direction=0, calculation of intersection may fail
                diffbb.expand(diffbb.getCentroid()[0] + options.getVoxelsize(), diffbb.getCentroid()[1] + options.getVoxelsize(), diffbb.getCentroid()[2] + options.getVoxelsize() );
                diffbb.expand(diffbb.getCentroid()[0] - options.getVoxelsize(), diffbb.getCentroid()[1] - options.getVoxelsize(), diffbb.getCentroid()[2] - options.getVoxelsize() );
//                cout << "diffbb: " << endl << diffbb << endl;

                intersections.push_back(diffbb);



            }


        }
        cout << timestamp << " finished calculating duplicate vertice indices" << endl;



        size_t face_amount = 0;
        size_t vertex_amount = 0;
        std::unordered_map<unsigned int, unsigned int> duplicate_indices;
        std::vector<size_t> intersectionPoints;
        std::vector<size_t> intersectionFaces;
        cout << timestamp <<" reading all meshed to memory" << endl;
        size_t v_all_amount = 0, f_all_amount = 0;
        for(int i = 0 ; i < cloudPahts.size() ; i++) {
            string tmpp = cloudPahts[i];
            boost::algorithm::replace_last(tmpp, "xyz", "ply");
            ifstream plyifs(tmpp);
            size_t vAmount = 0;
            size_t fAmount = 0;
            bool readv = false;
            bool readf = false;
            int foundAmount = 0;
            for (int x = 0; x < 100 && foundAmount < 2; x++) {
                string tmpread;
                itimer.resume ();
                plyifs >> tmpread;
                itimer.stop ();
                if (readf) {
                    fAmount = stoi(tmpread);
                    readf = false;
                    foundAmount++;
                }
                if (readv) {
                    vAmount = stoi(tmpread);
                    readv = false;
                    foundAmount++;
                }
                //cout << "parsing: " << tmpread << endl;
                if (tmpread == "vertex") readv = true;
                else if (tmpread == "face") readf = true;

            }
            v_all_amount += vAmount;
            f_all_amount += fAmount;
        }
        vertexArray.reserve(v_all_amount * 3);
        faceArray.reserve(f_all_amount * 3);
         for(int i = 0 ; i < cloudPahts.size() ; i++)
        {
            string tmpp = cloudPahts[i];
            boost::algorithm::replace_last(tmpp, "xyz", "ply");
            ifstream plyifs(tmpp);
            size_t vAmount=0;
            size_t fAmount=0;
            bool readv=false;
            bool readf=false;
            int foundAmount = 0;
            for(int x=0; x < 100 && foundAmount<2 ; x++)
            {
                string tmpread;
                itimer.resume ();
                plyifs >> tmpread;
                itimer.stop ();
                if(readf)
                {
                    fAmount = stoi(tmpread);
                    readf = false;
                    foundAmount++;
                }
                if(readv)
                {
                    vAmount = stoi(tmpread);
                    readv = false;
                    foundAmount++;
                }
                //cout << "parsing: " << tmpread << endl;
                if(tmpread=="vertex") readv = true;
                else if(tmpread=="face") readf = true;
            }
            //cout << tmpp << ": f: " << fAmount << " v: " << vAmount << endl;
            if(fAmount > 0 && vAmount > 0)
            {
                itimer.resume ();
                ModelPtr mptr = io.read(tmpp);
                itimer.stop ();
                MeshBufferPtr mBuffer = mptr->m_mesh;
//                floatArr vArray = mBuffer->getVertexArray(vAmount);
                coord3fArr viArray = mBuffer->getIndexedVertexArray(vAmount);
                uintArr  fArray = mBuffer->getFaceArray(fAmount);
//                std::vector<float> currentVertexArray(vArray.get(), vArray.get() + (vAmount*3));
                size_t oldSize = vertexArray.size()/3;
                //vertexArray.reserve(vertexArray.size() + vAmount * 3 );

                for(size_t i = 0 ; i<vAmount ; i++)
                {
                    for(auto it = intersections.begin(); it!=intersections.end() ; ++it)
                    {
                        if(it->contains(viArray[i][0], viArray[i][1], viArray[i][2]))
                        {
                            intersectionPoints.push_back(oldSize+i);
                            break;
                        }
                    }
                    vertexArray.push_back(viArray[i][0]);
                    vertexArray.push_back(viArray[i][1]);
                    vertexArray.push_back(viArray[i][2]);
                }
                //faceArray.reserve(faceArray.size() + fAmount * 3 );
                for(size_t j = 0 ; j<fAmount*3; j+=3)
                {
                    for(auto it = intersections.begin(); it!=intersections.end() ; ++it)
                    {
                        if(it->contains(vertexArray[(oldSize + *(fArray.get() + j))],
                                        vertexArray[(oldSize + *(fArray.get() + j))+1],
                                        vertexArray[(oldSize + *(fArray.get() + j))+2])
                                )
                        {
                            intersectionFaces.push_back(faceArray.size()/3);
                            break;
                        }
                        if(it->contains(vertexArray[(oldSize + *(fArray.get() + j +1))],
                                        vertexArray[(oldSize + *(fArray.get() + j +1))+1],
                                        vertexArray[(oldSize + *(fArray.get() + j +1))+2])
                                )
                        {
                            intersectionFaces.push_back(faceArray.size()/3);
                            break;
                        }
                        if(it->contains(vertexArray[(oldSize + *(fArray.get() + j +2))],
                                        vertexArray[(oldSize + *(fArray.get() + j +2))+1],
                                        vertexArray[(oldSize + *(fArray.get() + j +2))+2])
                                )
                        {
                            intersectionFaces.push_back(faceArray.size()/3);
                            break;
                        }
                    }
                    faceArray.push_back(oldSize + *(fArray.get() + j));
                    faceArray.push_back(oldSize + *(fArray.get() + j +1));
                    faceArray.push_back(oldSize + *(fArray.get() + j +2));
                }

            }

        }

        cout << timestamp <<"finished reading all meshed to memory" << endl;
        sortPoint<float>::tollerance = options.getVoxelsize()/1000;
        cout << timestamp << "starting to remove duplicates " << endl;
        set<sortPoint<float> > vertexSet;
        unordered_map<size_t, size_t> oldToNewVertices;
        for(auto it = intersectionPoints.begin() ; it!=intersectionPoints.end() ; ++it)
        {
            sortPoint<float> tmp(vertexArray.data() + (*it*3), *it);
            auto ret = vertexSet.insert(tmp);
            if(! ret.second)
            {
                // vertex already in set: found a duplicate
                oldToNewVertices[tmp.id()] = ret.first->id();
            }
        }

        vector<float> newVertexArray;
        newVertexArray.reserve(vertexArray.size());
        unordered_map<size_t, size_t> oldVertexArrayToNew;
        size_t gid = 0;
        for(size_t i = 0 ; i < vertexArray.size(); i+=3)
        {
            if(oldToNewVertices.find(i/3)!=oldToNewVertices.end())
            {
                //don't copy duplicate vertex
                gid++;
                continue;
            }
            oldVertexArrayToNew[i/3] = i/3 - gid;
            newVertexArray.push_back(vertexArray[i]);
            newVertexArray.push_back(vertexArray[i+1]);
            newVertexArray.push_back(vertexArray[i+2]);
        }
        newVertexArray.shrink_to_fit();
        vertexArray.clear();
        vector<float>().swap(vertexArray);

        unsigned int swapped = 0;
        for(auto it = faceArray.begin(), end = faceArray.end(); it != end; ++it)
        {
            if(oldToNewVertices.find(*it)!=oldToNewVertices.end())
            {
                *(it) = oldVertexArrayToNew[oldToNewVertices[*(it)]];
            }
            else if(oldVertexArrayToNew.find(*it)!=oldVertexArrayToNew.end())
            {
                *(it) = oldVertexArrayToNew[*(it)];

            }
                // if face point not found, delete that face (should not happen)
            else
            {
                cout << "wtf" << endl;
                unsigned int dist = std::distance(faceArray.begin(), it) / 3;
                dist*=3;
                std::iter_swap(faceArray.begin() + dist, faceArray.end()-3-swapped);
                std::iter_swap(faceArray.begin() + dist +1, faceArray.end()-2-swapped);
                std::iter_swap(faceArray.begin() + dist +2, faceArray.end()-1-swapped);
                swapped+=3;
                it = faceArray.begin() + dist -1;
            }

        }
        faceArray.resize(std::distance(faceArray.begin(),faceArray.end() - swapped ));

        cout << lvr::timestamp << "copying faces" << endl;
        vector<sortPoint<unsigned int> > newFaces;
        newFaces.reserve(faceArray.size()/3);
        for(int i = 0 ; i<faceArray.size() ; i+=3)
        {
            newFaces.push_back(sortPoint<unsigned int>(&faceArray[i]));
        }
        cout << lvr::timestamp << "sorting faces" << endl;

        std::sort(newFaces.begin(), newFaces.end());
        cout << lvr::timestamp << "get duplicate faces" << endl;
        auto fend = std::unique(newFaces.begin(), newFaces.end());
        size_t old_fsize = newFaces.size();
        cout << lvr::timestamp << "remove duplicate faces" << endl;

        newFaces.resize( std::distance(newFaces.begin(),fend) );
        size_t new_fsize = newFaces.size();
        std::vector<unsigned int> newFaceArray;
        newFaceArray.reserve(newFaces.size()*3);
        for(auto it = newFaces.begin(), end = newFaces.end(); it != end; ++it)
        {
            newFaceArray.push_back(it->x());
            newFaceArray.push_back(it->y());
            newFaceArray.push_back(it->z());
        }
        faceArray.clear();
        vector<unsigned int>().swap(faceArray);
        cout << lvr::timestamp << "finished, removed " <<oldToNewVertices.size() << " vertices and " << old_fsize - new_fsize << " faces" << endl;

        completeMesh->setFaceArray(newFaceArray);
        completeMesh->setVertexArray(newVertexArray);

//        DuplicateRemover dr;
//
//        completeMesh= dr.removeDuplicates(completeMesh);

        //mainMesh.finalize();
        ModelPtr tmpm( new Model(completeMesh) );

        string finalPath = octreeFolder;
        finalPath.append("/final.ply");
        otimer.resume ();
        ModelFactory::saveModel( tmpm, finalPath);
        otimer.stop ();
        cout << "##############################" << endl << "##############################" << endl << "FINESHED in " << lvr::timestamp << endl;
        double iotime=0;
        for(int i = 1 ; i< world.size()  ; i++)
          {
            cout << "...sending to " << i << " finish:  " << i << endl;
            std::string re("ready");
            world.send(i, DATA, re);
            double msg;
            world.recv(boost::mpi::any_source, FINISHED, msg);
            iotime += msg;
          }
        std::cout << "IO-Timer of Node :" << world.rank ()  << std::endl
                  << "READ: " << itimer.format () << std::endl
                  << "WRITE: " << otimer.format () << std::endl;
        cout << "IO TIME of NODES: " << iotime << "s" << endl;

        //world.abort(0);


    }
        //---------------------------------------------
        // SLAVE NODE
        //--------------------------------------------
    else
    {
        boost::timer::cpu_timer itimer;
        boost::timer::cpu_timer otimer;
        itimer.stop ();
        otimer.stop ();
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
                FILE * fp = fopen(filePath.c_str(), "rb" );
                size_t sz = 0;
                if(fp != NULL)
                {
                    itimer.resume();
                    fseek(fp, 0L, SEEK_END);
                    itimer.stop();
                    sz = ftell(fp);

                    sz/=sizeof(float);
                }

                floatArr fArray(new float[sz]);


                rewind(fp);
                itimer.resume();
                size_t readamount = fread ( fArray.get(), sizeof(float), sz, fp );
                itimer.stop();
                fclose(fp);
                PointBufferPtr p_loader(new PointBuffer);
                p_loader->setPointArray(fArray, sz/3);
                if(options.getUseNormals())
                {
                    string normalPath = filePath;
                    boost::algorithm::replace_last(normalPath,".xyz",".normals");
                    FILE * fp2 = fopen(normalPath.c_str(), "rb" );
                    size_t sz2 = 0;
                    if(fp2 != NULL)
                    {
                        itimer.resume();
                        fseek(fp2, 0L, SEEK_END);
                        itimer.stop();
                        sz2 = ftell(fp2);

                        sz2/=sizeof(float);
                    }

                    floatArr nArray(new float[sz2]);


                    rewind(fp2);
                    itimer.resume();
                    size_t readamount2 = fread ( nArray.get(), sizeof(float), sz2, fp2 );
                    p_loader->setPointNormalArray(nArray,sz2/3);
                    fclose(fp2);
                }




                string pcm_name = options.getPCM();
                psSurface::Ptr surface;

                // Create point set surface object
                string centerpath = filePath;
                boost::algorithm::replace_last(centerpath, "xyz", "-center.pt");
                ifstream centerifs(centerpath);
                Vertexf center;
                itimer.resume();
                centerifs >> center.x >> center.y >> center.z;
                itimer.stop();
                centerifs.close();
                if(pcm_name == "PCL")
                {
#ifdef LVR_USE_PCL
                    surface = psSurface::Ptr( new pclSurface(p_loader, options.getKn(), options.getKd()));
#else
                    cout << timestamp << "Can't create a PCL point set surface without PCL installed." << endl;
            exit(-1);
#endif
                }
                else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
                {
                    surface = psSurface::Ptr(new akSurface(
                      p_loader, pcm_name,
                      options.getKn(),
                      options.getKi(),
                      options.getKd(),
                      options.useRansac(),
                      options.getScanPoseFile(),
                      center
                    ));
                    // Set RANSAC flag
/*                    if(options.useRansac())
                    {
                      ((akSurface)(*surface.get())).useRansac(true);
                    }*/
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


                string bbpath = filePath;
                boost::algorithm::replace_last(bbpath, "xyz", "bb");
                otimer.resume();
                ifstream bbifs(bbpath);
                float minx, miny, minz, maxx, maxy, maxz;
                bbifs >> minx >> miny >> minz >> maxx >> maxy >> maxz;
                otimer.stop();
                BoundingBox<ColorVertex<float, unsigned char> > tmpbb(minx, miny, minz, maxx, maxy, maxz);
                cout << "grid bb: " << tmpbb << endl;


                surface->setKd(options.getKd());
                surface->setKi(options.getKi());
                surface->setKn(options.getKn());
                if(! options.getUseNormals())
                {
                    surface->calculateSurfaceNormals();
                }

                if(options.savePointNormals())
                {
                    string normalpath = filePath;
                    boost::algorithm::replace_last(normalpath, ".xyz", "-normals.ply");
                    ModelPtr pn( new Model);
                    pn->m_pointCloud = surface->pointBuffer();
                    cout << timestamp << " saving normals to" <<normalpath << endl;
                    otimer.resume();
                    ModelFactory::saveModel(pn, normalpath);
                    otimer.stop();
                }
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


                    grid = new PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, tmpbb, useVoxelsize, options.extrude());
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
                    boost::algorithm::replace_first(out, ".grid", "-3.grid");
                    otimer.resume();
                    ps_grid->saveGrid(out);
                    otimer.stop();
                }
                else if(decomposition == "PMC")
                {
                    grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, tmpbb, useVoxelsize, options.extrude());
                    grid->setExtrusion(options.extrude());
                    BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                    PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                    ps_grid->calcDistanceValues();

                    reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);

                    string out = filePath;
                    out.pop_back();
                    out.pop_back();
                    out.pop_back();
                    out.append("grid");
                    otimer.resume();
                    ps_grid->serialize(out);
                    out = filePath;
                    out.pop_back();
                    out.pop_back();
                    out.pop_back();
                    out.append("show.grid");
                    ps_grid->saveGrid(out);
                    otimer.stop();

                }
                else if(decomposition == "SF")
                {
                    SharpBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                    grid = new PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >(resolution, surface, tmpbb, useVoxelsize, options.extrude());
                    grid->setExtrusion(options.extrude());
                    PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                    ps_grid->calcDistanceValues();
                    reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, SharpBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
                    string out = filePath;
                    out.pop_back();
                    out.pop_back();
                    out.pop_back();
                    out.append("grid");
                    otimer.resume();
                    ps_grid->serialize(out);
                    otimer.stop();
                }
               delete grid;
               delete reconstruction;
            }
            // Create Mesh from Grid
            else if(filePath.find("grid") != std::string::npos)
            {
                cout << "going to rreconstruct " << filePath << endl;
                string cloudPath = filePath;
                boost::algorithm::replace_last(cloudPath, "grid", "xyz");
                FILE * fp = fopen(cloudPath.c_str(), "rb" );
                size_t sz = 0;
                if(fp != NULL)
                {
                    itimer.resume();
                    fseek(fp, 0L, SEEK_END);
                    sz = ftell(fp);
                    itimer.stop();
                    sz/=sizeof(float);
                }

                floatArr fArray(new float[sz]);
                rewind(fp);
                itimer.resume();
                size_t readamount = fread ( fArray.get(), sizeof(float), sz, fp );
                itimer.stop();
                fclose(fp);
                PointBufferPtr p_loader(new PointBuffer);
                p_loader->setPointArray(fArray, sz/3);
                cout << "loaded " << cloudPath << " with : "<< sz/3  << endl;

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
                    surface = psSurface::Ptr(new akSurface(
                      p_loader, pcm_name,
                      options.getKn(),
                      options.getKi(),
                      options.getKd(),
                      options.useRansac(),
                      options.getScanPoseFile(),
                      center
                    ));
                    // Set RANSAC flag
/*                    if(options.useRansac())
                    {
                        aks->useRansac(true);
                    }*/
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

                itimer.resume();
                HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > mainGrid(filePath);
                itimer.stop();
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
                cout << "MESH: " << filePath << " bebore: " << mesh.getFaces().size() << endl;
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
                cout << "MESH: " << filePath << " after: " << mesh.getFaces().size() << endl;

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
                otimer.resume();
                ModelFactory::saveModel( m, output);
                otimer.stop();
                delete reconstruction;
            }

            world.send(0, FINISHED, std::string("world"));
            cout << timestamp << "Node: " << world.rank() << "finished "  << endl;
        }
        double t = 0;
        auto tsi = itimer.elapsed ();
        auto tso = otimer.elapsed ();
        t += (double)tsi.wall/1000000000.0;
        t += (double)tso.wall/1000000000.0;
        world.send(0, FINISHED, t);
//        std::cout << "IO-Timer of Node :" << world.rank ()  << std::endl
//                  << "READ: " << itimer.format () << std::endl
//                  << "WRITE: " << otimer.format () << std::endl;

    }

    return 0;
}