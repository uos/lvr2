#include <iostream>
#include <cstdlib>
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
#include <unordered_set>
#include <lvr/geometry/HalfEdgeMesh.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/io/Model.hpp>
#include <lvr/io/MeshBuffer.hpp>
#include <lvr/io/PLYIO.hpp>
#include <omp.h>
#include "sortPoint.hpp"
#include "DuplicateRemover.hpp"
#include <random>
using namespace std;
using namespace lvr;








template<typename compareType>
bool comparePoints(compareType ax, compareType ay, compareType az, compareType bx, compareType by, compareType bz)
{
    if(ax != bx) return ax < bx;
    if(ay != by) return ay < by;
    return az < bz;
}

int main(int argc, char** argv)
{
    ofstream ofs("testsurface2.xyz");

    for(int i = 0 ; i< 100 ;i++)
    {
        for(int y = -50 ; y<50 ; y++)
        {
            ofs << y << " " << y*y*0.01 << " " <<i << endl;
        }
    }
/*
    for(int i = 0 ; i<100;i++)
    {
        for(int j = 0 ; j<100 ;j++)
        {
            for(int k = 0 ; k<100;k++)
            {

            }
        }
    }
    */
    if(argc != 2)
    {
        cout << "usage: ./lvr_removeDuplicates <path mesh>" << endl;
        return -1;
    }

    cout << lvr::timestamp << "god model" << endl;
    string path(argv[1]);

    PLYIO io;

    ModelPtr mptr = io.read(path);

    DuplicateRemover dp;
    MeshBufferPtr mBuffer = dp.removeDuplicates(mptr->m_mesh);

    cout << lvr::timestamp << "saving model" << endl;

    ModelPtr tmpm( new Model(mBuffer) );

    ModelFactory::saveModel( tmpm, "no_duplicates.ply");



}
