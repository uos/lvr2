//
// Created by eiseck on 08.12.15.
//
#include <fstream>
#include "NodeData.hpp"
#include <algorithm>
#include "LargeScaleOctree.hpp"
#include <string>
#include <sstream>

using namespace lvr;

void printTest(Vertexf v)
{
    cout << v;
}

int main(int argc, char** argv)
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


    clock_t end = clock();
    double elapsed_secs = double(end - begin) / CLOCKS_PER_SEC;
    cout << "FINESHED in " << elapsed_secs << " Seconds." << endl;


}


