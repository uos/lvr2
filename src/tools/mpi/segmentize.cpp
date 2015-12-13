//
// Created by eiseck on 08.12.15.
//
#include <fstream>
#include "NodeData.h"
#include <algorithm>
#include "LargeScaldeOctreeNode.h"
#include <string>
#include <sstream>

using namespace lvr;

void printTest(Vertexf v)
{
    cout << v;
}

int main(int argc, char** argv)
{
/*
    string filepath(argv[1]);
    ifstream inputData(filepath);
    string s;
    int j = 1;
    NodeData nd;
    while( getline( inputData, s ) )
    {
        stringstream ss;
        ss.str(s);
        Vertexf v;
        ss >> v.x >> v.y >> v.z;
        nd.add(v);

    }
    for(Vertexf vv : nd)
    {
        cout << vv;
    }

    */



    string filepath(argv[1]);
    ifstream inputData(filepath);
    string s;
    BoundingBox<Vertexf> box;
    int j = 1;
    while( getline( inputData, s ) )
    {
        //cout << "j: " << j++ << endl;
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

    LargeScaldeOctreeNode octree(center, size);

    inputData.close();
    inputData.clear();
    ifstream inputData2(filepath);
    //inputData2.open(filepath);
    //inputData.seekg(0, ios::beg);
    int i = 1;
    while( getline( inputData2, s ) )
    {
        //cout << "i: " << i++ << endl;
        stringstream ss;
        ss.str(s);
        Vertexf v;
        ss >> v.x >> v.y >> v.z;
        //cout << "Inserting " << v;
        octree.insert(v);

    }





}


