//
// Created by eiseck on 11.12.15.
//

#include <vector>
#include <sstream>
#include <boost/filesystem/path.hpp>
#include <boost/filesystem/operations.hpp>
#include "NodeData.hpp"



using namespace std;
namespace lvr
{


int NodeData::c_last_id = 0;
time_t NodeData::c_tstamp =  std::time(0);

NodeData::NodeData(string inputPoints, string nodePoints, size_t bufferSize) : NodeData(bufferSize)
{
    create(inputPoints, nodePoints);
}

NodeData::NodeData(size_t bufferSize) : m_bufferSize(bufferSize)
{

    m_gotSize = false;
    m_id = ++c_last_id;
    m_dataPath = "node-";
    m_dataPath.append(to_string(c_tstamp));
    m_dataPath.append("/");
    boost::filesystem::path dir(m_dataPath);

    if(!(boost::filesystem::exists(dir)))
    {
       boost::filesystem::create_directory(dir);
    }
    m_dataPath.append(to_string(m_id));
    m_dataPath.append(".xyz");
    m_bufferIndex = 0;
    m_writeBuffer.clear();
}

void NodeData::fillBuffer(size_t start_id)
{
    m_readBuffer.clear();
    ifstream ifs(m_dataPath);
    m_bufferIndex=start_id;
    int i = 0;
    int j = 0;
    float x,y,z;
    string s;
    while( getline( ifs, s ) )
    {
        if(i>=size() || j>=m_bufferSize) break;
        else if(i>=start_id)
        {
            stringstream ss;
            ss.str(s);
            ss >> x >> y >> z;
            m_readBuffer.push_back(Vertexf(x,y,z));
            j++;
        }
        i++;

    }


}

void NodeData::create(string inputPoints, string nodePoints)
{
    //Todo: copy only x, y, z and ignore other values like intensity etc...
    ifstream ifs(inputPoints.c_str(), std::ios::binary);
    ofstream ofs(nodePoints.c_str(),  std::ios::binary);
    ofs << ifs.rdbuf();
    m_dataPath = nodePoints;
    ifs.close();
    ofs.close();


}

void NodeData::open(string path)
{
    m_dataPath = path;

}


void NodeData::remove()
{
    std::remove(m_dataPath.c_str());
    m_dataPath = "";
    m_readBuffer.clear();
}

void NodeData::remove(unsigned int i)
{
//Todo: remove vertex at index
}

void NodeData::add(Vertex<float> input)
{
    ofstream ofs(m_dataPath, fstream::app);
    ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
    if(m_gotSize) m_size++;
    ofs.close();

}

void NodeData::addBuffered(lvr::Vertex<float> input)
{

    if(m_gotSize) m_size++;
    m_writeBuffer.push_back(input);
}
void NodeData::writeBuffer()
{

    ofstream ofs(m_dataPath, fstream::app);
    for(Vertexf input : m_writeBuffer)
    {
        ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
    }
    ofs.close();
    m_writeBuffer.clear();
}

size_t NodeData::getWriteBufferSize()
{
    return m_writeBuffer.size();
}

Vertex<float>& NodeData::get(int i)
{

    if(i>=m_bufferIndex && i - m_bufferIndex < m_readBuffer.size())
    {
        return m_readBuffer[i - m_bufferIndex];
    }
    else
    {
        fillBuffer(i);
        return m_readBuffer[i - m_bufferIndex];
    }


}

Vertex<float> NodeData::next()
{
    return Vertex<float>();
}

NodeData::Iterator NodeData::begin()
{
    return NodeData::Iterator(*this);
}

NodeData::Iterator NodeData::end()
{
    return NodeData::Iterator(*this, this->size());
}

NodeData::NodeData(NodeData &origin) : NodeData(origin.m_bufferSize)
{
    copy(origin);
}
NodeData &lvr::NodeData::operator=(lvr::NodeData &origin)
{
    if(this != &origin)
    {
        copy(origin);
    }
    return *this;
}


lvr::Vertexf NodeData::operator[](unsigned int i)
{
    return get(i);
}

void NodeData::copy(NodeData &origin)
{
    this->m_dataPath = origin.m_dataPath;
    this->m_gotSize  = origin.m_gotSize;
    this->m_size     = origin.m_size;

}

size_t NodeData::size()
{
    if(m_gotSize) return m_size;
    else
    {

        ifstream ifs( m_dataPath );
        size_t size = 0;
        string s;
        while( getline( ifs, s ) ) size++;

        m_size = size;
        m_gotSize = true;
        ifs.close();
        size+=m_writeBuffer.size();
        return size;
    }
}

}


