//
// Created by eiseck on 11.12.15.
//

#ifndef LAS_VEGAS_NODEDATA_H
#define LAS_VEGAS_NODEDATA_H
#include <string>
#include <iostream>
#include <lvr/geometry/Vertex.hpp>
#include <fstream>
#include <ctime>
#include <vector>
using namespace std;
namespace lvr
{
class NodeData
{
    class Iterator;
public:
    NodeData(string inputPoints, string nodePoints, size_t bufferSize );
    NodeData(size_t bufferSize);
    NodeData(NodeData& origin);
    NodeData& operator=(NodeData& origin);
    void create(string inputPoints, string nodePoints);
    void open(string path);
    lvr::Vertexf operator[](unsigned int);
    const string &getDataPath() const { return m_dataPath; }
    Iterator begin();
    Iterator end();
    void remove();
    void remove(unsigned int i);
    void add(lvr::Vertex<float> input);
    void addBuffered(lvr::Vertex<float> input);
    void writeBuffer();
    lvr::Vertex<float>& get(int);
    lvr::Vertex<float> next();
    size_t size();

private:
    void copy(NodeData& origin);
    void fillBuffer(size_t start_id);
    string m_dataPath;
    bool m_gotSize;
    size_t m_size;
    int m_id;
    streampos las_pos;
    static int c_last_id;
    static time_t c_tstamp;
    vector<Vertexf> m_readBuffer;
    size_t m_bufferSize;
    size_t m_bufferIndex;
    vector<Vertexf> m_writeBuffer;




};



class NodeData::Iterator
{
public:
    Iterator(NodeData& nodeData, size_t pos) : m_NodeData(nodeData), m_pos(pos){}
    Iterator(NodeData& nodeData) : m_NodeData(nodeData), m_pos(0){}
    Iterator(const Iterator& copy) : m_pos(copy.m_pos), m_NodeData(copy.m_NodeData){}

    Iterator operator++(int) {Iterator tmp(*this); operator++(); return tmp;}

    bool operator==(const Iterator& rhs) {return m_pos == rhs.m_pos;}
    bool operator!=(const Iterator& rhs) {return m_pos != rhs.m_pos;}
    //Todo: more if needed
    void operator+(int i) {m_pos+=i;}
    void operator-(int i) {m_pos-=i;}
    void operator++() {++m_pos;}
    void operator--() {--m_pos;}

    lvr::Vertex<float> operator*()  { return m_NodeData.get(m_pos); }
    lvr::Vertex<float> operator->() { return m_NodeData.get(m_pos); }
private:
    NodeData& m_NodeData;
    size_t m_pos;
};

}



#endif //LAS_VEGAS_NODEDATA_H
