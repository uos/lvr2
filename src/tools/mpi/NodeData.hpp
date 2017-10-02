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
#include <boost/timer/timer.hpp>
using namespace std;
namespace lvr
{
/**
 * Class that contains the actual points of a node
 */
class NodeData
{
  static boost::timer::cpu_timer itimer;
  static boost::timer::cpu_timer otimer;
  static bool timer_init;

    // Iterator
    class Iterator;
public:
    /**
     * Constructor
     * @param bufferSize amount of points to store in a buffer. when buffersize is reached all points are stored to hdd
     */
    NodeData(size_t bufferSize);
    /**
     * copy constructor
     * @param origin
     * @return
     */
    NodeData(NodeData& origin);

    /**
     * assignment operator
     * @param origin
     * @return
     */
    NodeData& operator=(NodeData& origin);

    /**
     * sets data path
     * @param path
     */
    void open(string path);

    /**
     * index operator
     * @return vertex at position
     */
    lvr::Vertexf operator[](unsigned int);

    /**
     * returns path where data is stored on hdd
     * @return
     */
    const string &getDataPath() const { return m_dataPath; }

    /**
     * returns path to folder on hdd where data is stored
     * @return
     */
    const string getFolder() const { return to_string(c_tstamp).insert(0, "node-"); }

    /**
     * Iterator referring to first element
     * @return
     */
    Iterator begin();

    /**
     * Returns an iterator referring to the past-the-end element
     * @return
     */
    Iterator end();

    /**
     * removes the data file from hdd
     */
    void remove();

    /**
     * removes element at index i (not implemented)
     * @param i
     */
    void remove(unsigned int i);

    /**
     * add a vertex to the data
     * @param input
     */
    void add(lvr::Vertex<float> input);

    void addNormal(lvr::Vertex<float> input);

    /**
     * adds a vertex to buffer
     * @param input
     */
    void addBuffered(lvr::Vertex<float> input);

    void addBufferedNormal(lvr::Vertex<float> input);

    /**
     * writes buffer to file
     */
    void writeBuffer();

    /**
     * gets amount of points stored in write buffer
     * @return
     */
    size_t getMaxWriteBufferSize();
    /**
     * gets element at index
     * @return
     */
    lvr::Vertex<float> get(int);

    lvr::Vertex<float> getNormal(int);
    /**
     * gets next element
     * @return
     */
    lvr::Vertex<float> next();

    /**
     * returns amount of elements stored in data
     * @return
     */
    size_t size();

    /**
     * returns amount of poiunts stored in buffer
     * @return
     */
    size_t getBufferSize();

    static   void printTimer()
    {
      std::cout << "IO-Timer of Octree:" << std::endl
                << "READ: " << itimer.format () << std::endl
                << "WRITE: " << otimer.format () << std::endl;

    }

private:
    // copys the data from origin to this
    void copy(NodeData& origin);
    // fills the buffer, copys from start_id to buffersize elements into buffer
    void fillBuffer(size_t start_id);
    void fillBufferNormal(size_t start_id);
    // path to data
    string m_dataPath;

    string m_dataPathNormal;
    // if m_gotSize is set, the m_size will be used, else the size will be calculated
    bool m_gotSize;
    // amount of points stored in data
    size_t m_size;
    // current id of the dataobject, used to generate file name
    int m_id;
    // input buffer
    vector<float> m_readBuffer;
    // output buffer
    vector<float> m_writeBuffer;

    vector<float> m_readBufferNormal;
    // output buffer
    vector<float> m_writeBufferNormal;

    // maximum buffer size
    size_t m_bufferSize;
    // current inputbuffer position
    size_t m_readBufferIndex;

    size_t m_readBufferIndexNormal;
    // last id
    static int c_last_id;
    // timestamp, of first creation
    static time_t c_tstamp;



};


/**
 * Iterator for NodeData
 */
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
