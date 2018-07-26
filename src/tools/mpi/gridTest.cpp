//
// Created by isaak on 28.06.16.
//
#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <sstream>
#include <algorithm>
#include "LargeScaleOctree.hpp"
#include <lvr/geometry/BoundingBox.hpp>
#include <lvr/geometry/Vertex.hpp>
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
#include <boost/algorithm/string.hpp>
#include <lvr/reconstruction/PCLKSurface.hpp>
#include "BigGrid.hpp"

using namespace std;
using namespace lvr;

typedef lvr::PointsetSurface<ColorVertex<float, unsigned char> > psSurface;
typedef lvr::AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> > akSurface;
typedef lvr::PCLKSurface<ColorVertex<float, unsigned char> , Normal<float> > pclSurface;

int main(int argc, char* argv[])
{
    //argv 1 os size, argv 2 filepath, argv 3 voxelsize
    int osize = atoi(argv[1]);
    string filepath(argv[2]);
    int voxelsize = atoi(argv[3]);
    cout << osize << "|" << filepath << endl;
    int ki = 50;
    int kd = 10;
    int kn = 50;
    string decomposition("PMC");
    size_t minSize = std::max(std::max(kn, kd), ki);

    ifstream inputData(filepath);

    string s;
    vector<Vertexf> points;
    while( getline( inputData, s ) )
    {
        stringstream ss;
        ss.str(s);
        Vertexf v;
        ss >> v.x >> v.y >> v.z;
        cout << v << endl;
        points.push_back(v);
    }

    inputData.close();

    BoundingBox<Vertexf> box;
    for(std::vector<Vertexf>::iterator it = points.begin(); it!=points.end() ; ++it) box.expand((*it));

    Vertexf center = box.getCentroid();
    float size = box.getLongestSide();
    double newLength = voxelsize;
    while(newLength<=size) newLength*=2;
    Vertexf newXMax(box.getCentroid().x+(newLength/2),box.getCentroid().y, box.getCentroid().z );
    Vertexf newXMin(box.getCentroid().x-(newLength/2),box.getCentroid().y, box.getCentroid().z  );
    box.expand(newXMax);
    box.expand(newXMin);

    LargeScaleOctree octree(center, newLength, osize );
    for(std::vector<Vertexf>::iterator it = points.begin(); it!=points.end() ; ++it) octree.insert((*it));

    vector<LargeScaleOctree*> leafs;
    vector<LargeScaleOctree*> nodes = octree.getNodes();
    for(int i = 0 ; i< nodes.size() ; i++)
    {
        if(nodes[i]->isLeaf() && nodes[i]->getSize()>minSize && nodes[i]->getFilePath() != "" )
        {
            leafs.push_back(nodes[i]);
        }
    }
    for(int i = 0 ; i<leafs.size() ; i++)
    {
        leafs[i]->generateNeighbourhood();
    }

    for(int i = 0 ; i<leafs.size() ; i++)
    {
        string path = leafs[i]->getFilePath();
        boost::algorithm::replace_last(path, "xyz", "bb");
        float r = leafs[i]->getWidth()/2;
        Vertexf rr(r,r,r);
        Vertexf min = leafs[i]->getCenter()-rr;
        Vertexf max = leafs[i]->getCenter()+rr;
        ofstream ofs(path);
        ofs << min.x << " " << min.y << " " << min.z  << " " << max.x << " " << max.y << " " << max.z << endl;
        ofs.close();
    }


    for(std::vector<LargeScaleOctree*>::iterator it = leafs.begin(); it!=leafs.end() ; ++it)
    {
        ModelPtr model = ModelFactory::readModel((*it)->getFilePath());
        PointBufferPtr p_loader;
        if ( !model )
        {
            exit(-1);
        }
        p_loader = model->m_pointCloud;
        string pcm_name = "FLANN";
        psSurface::Ptr surface;

        // Create point set surface object
        if(pcm_name == "PCL")
        {
            surface = psSurface::Ptr( new pclSurface(p_loader));
        }
        else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
        {
            akSurface* aks = new akSurface(
                    p_loader, pcm_name,
                    kn,
                    ki,
                    kd,
                    false,
                    ""
            );

            surface = psSurface::Ptr(aks);

            string bbpath = (*it)->getFilePath();
            boost::algorithm::replace_last(bbpath, "xyz", "bb");
            ifstream bbifs((*it)->getFilePath());
            float minx, miny, minz, maxx, maxy, maxz;
            bbifs >> minx >> miny >> minz >> maxx >> maxy >> maxz;
            BoundingBox<ColorVertex<float, unsigned char> > tmpbb(minx, miny, minz, maxx, maxy, maxz);
            bbifs.close();
            surface->setKd(kd);
            surface->setKi(ki);
            surface->setKn(kn);
            surface->calculateSurfaceNormals();

            HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );
            GridBase* grid;

            if(decomposition == "MC")
            {
                grid = new PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(voxelsize, surface, tmpbb, true, true);
                grid->setExtrusion(true);
                PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->calcDistanceValues();
                string out = (*it)->getFilePath();
                out.pop_back();
                out.pop_back();
                out.pop_back();
                out.append("grid");
                ps_grid->serialize(out);
                string out2 = out;
                boost::algorithm::replace_first(out2, ".grid", "-2.grid");
                grid->saveGrid(out2);
            }
            else if(decomposition == "PMC")
            {
                grid = new PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >(voxelsize, surface, tmpbb, true, true);
                grid->setExtrusion(true);
                BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >::m_surface = surface;
                PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
                ps_grid->calcDistanceValues();
                string out = (*it)->getFilePath();
                out.pop_back();
                out.pop_back();
                out.pop_back();
                out.append("grid");
                ps_grid->serialize(out);
                string out2 = out;
                boost::algorithm::replace_first(out2, ".grid", "-2.grid");
                grid->saveGrid(out2);
            }

            delete grid;

        }

    }

    for(std::vector<LargeScaleOctree*>::iterator it = leafs.begin(); it!=leafs.end() ; ++it)
    {
        string filePath = (*it)->getFilePath();
        boost::algorithm::replace_first(filePath, ".xyz", ".grid");

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

        string pcm_name = "FLANN";
        psSurface::Ptr surface;


        if(pcm_name == "PCL")
        {
            surface = psSurface::Ptr( new pclSurface(p_loader));
        }
        else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
        {
            akSurface* aks = new akSurface(
                    p_loader, pcm_name,
                    kn,
                    ki,
                    kd,
                    false,
                    ""
            );

            surface = psSurface::Ptr(aks);

        }

        surface->setKd(kd);
        surface->setKi(ki);
        surface->setKn(kn);
        HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mesh( surface );




        FastReconstructionBase<ColorVertex<float, unsigned char>, Normal<float> >* reconstruction;
        GridBase* grid = new HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >(filePath);
        //HashGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > mainGrid(filePath);
        string out2 = filePath;

        cout << "finished reading the grid " << filePath << endl;
        if(decomposition == "MC")
        {
            PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, FastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
            reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, FastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
        }
        else if(decomposition == "PMC")
        {

            PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > >* ps_grid = static_cast<PointsetGrid<ColorVertex<float, unsigned char>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> > > *>(grid);
            reconstruction = new FastReconstruction<ColorVertex<float, unsigned char> , Normal<float>, BilinearFastBox<ColorVertex<float, unsigned char>, Normal<float> >  >(ps_grid);
        }


        cout << "going to reconstruct" << endl;
        reconstruction->getMesh(mesh);
        mesh.setClassifier("PlaneSimpsons");
        mesh.getClassifier().setMinRegionSize(10);
        mesh.finalize();
        ModelPtr m( new Model( mesh.meshBuffer() ) );

        string output = filePath;
        boost::algorithm::replace_first(output, "grid", "ply");
        ModelFactory::saveModel( m, output);
        delete reconstruction;
        delete grid;
    }



    PLYIO io;

    //HalfEdgeMesh<ColorVertex<float, unsigned char> , Normal<float> > mainMesh;
    MeshBufferPtr completeMesh(new MeshBuffer);
    std::vector<float> vertexArray;
    std::vector<unsigned int> faceArray;

    size_t face_amount = 0;
    size_t vertex_amount = 0;
    for(int i = 0 ; i < nodes.size() ; i++)
    {
        string tmpp = nodes[i]->getFilePath();
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
            plyifs >> tmpread;

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
            ModelPtr mptr = io.read(tmpp);
            MeshBufferPtr mBuffer = mptr->m_mesh;
            floatArr vArray = mBuffer->getVertexArray(vAmount);
            uintArr  fArray = mBuffer->getFaceArray(fAmount);
//                std::vector<float> currentVertexArray(vArray.get(), vArray.get() + (vAmount*3));
            size_t oldSize = vertexArray.size()/3;
            vertexArray.reserve(vertexArray.size() + vAmount * 3 );
            vertexArray.insert(vertexArray.end(), vArray.get(), vArray.get() + (vAmount*3));
            faceArray.reserve(faceArray.size() + fAmount * 3 );
            for(size_t j = 0 ; j<fAmount*3; j+=3)
            {
                faceArray.push_back(oldSize + *(fArray.get() + j));
                faceArray.push_back(oldSize + *(fArray.get() + j +1));
                faceArray.push_back(oldSize + *(fArray.get() + j +2));
            }

        }

    }
    completeMesh->setFaceArray(faceArray);
    completeMesh->setVertexArray(vertexArray);
    //mainMesh.finalize();
    ModelPtr tmpm( new Model(completeMesh) );

    string finalPath = "";
    finalPath.append("final.ply");
    ModelFactory::saveModel( tmpm, finalPath);





}
