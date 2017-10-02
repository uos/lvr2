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

  boost::timer::cpu_timer NodeData::itimer;
  boost::timer::cpu_timer NodeData::otimer;
  bool NodeData::timer_init=false;

int NodeData::c_last_id = 0;
time_t NodeData::c_tstamp =  std::time(0);

    NodeData::NodeData(size_t bufferSize) : m_bufferSize(bufferSize)
{

    if(! timer_init)
    {
        itimer.stop ();
        otimer.stop ();
        timer_init = true;
    }
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
    m_dataPathNormal = m_dataPath;
    m_dataPath.append(".xyz");
    m_dataPathNormal.append(".normals");
    m_readBufferIndex = 0;
    m_readBufferIndexNormal = 0;
    m_writeBuffer.clear();
    m_writeBufferNormal.clear();
    vector<float>().swap(m_writeBuffer);
    vector<float>().swap(m_writeBufferNormal);
}

void NodeData::fillBuffer(size_t start_id)
{
    m_readBuffer.clear();
    m_readBuffer.resize(m_bufferSize);
    FILE * pfile = fopen(m_dataPath.c_str(), "rb");
    m_readBufferIndex=start_id;
    itimer.resume ();
    fseek(pfile, sizeof(float)*start_id*3, SEEK_SET);
    itimer.stop ();
    size_t readamount = fread ( m_readBuffer.data(), sizeof(float), m_bufferSize, pfile );
    fclose(pfile);
    m_readBuffer.resize(readamount);


}

void NodeData::fillBufferNormal(size_t start_id)
{
    m_readBufferNormal.clear();
    m_readBufferNormal.resize(m_bufferSize);
    FILE * pfilen = fopen(m_dataPathNormal.c_str(), "rb");
    m_readBufferIndexNormal=start_id;
    itimer.resume ();
    fseek(pfilen, sizeof(float)*start_id*3, SEEK_SET);
    itimer.stop ();
    size_t readamount = fread ( m_readBufferNormal.data(), sizeof(float), m_bufferSize, pfilen );
    fclose(pfilen);
    m_readBufferNormal.resize(readamount);
}



void NodeData::open(string path)
{
    m_dataPath = path;

}


void NodeData::remove()
{
    boost::filesystem::remove(m_dataPath);
    boost::filesystem::remove(m_dataPathNormal);
    m_dataPath = "";
    m_dataPathNormal = "";
    m_readBuffer.clear();
    m_readBufferNormal.clear();
    vector<float>().swap(m_readBuffer);
    vector<float>().swap(m_readBufferNormal);
}

void NodeData::remove(unsigned int i)
{
//Todo: remove vertex at index
}

void NodeData::add(Vertex<float> input)
{
    FILE * oFile = fopen(m_dataPath.c_str(), "ab");
    float v[3];
    v[0] = input[0];
    v[1] = input[1];
    v[2] = input[2];
    otimer.resume ();
    fwrite (m_writeBuffer.data() , sizeof(float), 3, oFile);
    otimer.stop ();
    fclose (oFile);

//    ofstream ofs(m_dataPath, fstream::app);
//    ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
//    if(m_gotSize) m_size++;
//    ofs.close();

}

void NodeData::addNormal(Vertex<float> input)
{
    FILE * oFile = fopen(m_dataPathNormal.c_str(), "ab");
    float v[3];
    v[0] = input[0];
    v[1] = input[1];
    v[2] = input[2];
    otimer.resume ();
    fwrite (m_writeBufferNormal.data() , sizeof(float), 3, oFile);
    otimer.stop ();
    fclose (oFile);

//    ofstream ofs(m_dataPath, fstream::app);
//    ofs << input.x << " " << input.y << " " <<  input.z << " " <<  std::endl;
//    if(m_gotSize) m_size++;
//    ofs.close();

}

void NodeData::addBuffered(lvr::Vertex<float> input)
{

    if(m_gotSize) m_size++;
    if(m_writeBuffer.size() > m_bufferSize) writeBuffer();
    m_writeBuffer.push_back(input.x);
    m_writeBuffer.push_back(input.y);
    m_writeBuffer.push_back(input.z);
}

void NodeData::addBufferedNormal(lvr::Vertex<float> input)
{
    if(m_writeBufferNormal.size() > m_bufferSize) writeBuffer();
    m_writeBufferNormal.push_back(input.x);
    m_writeBufferNormal.push_back(input.y);
    m_writeBufferNormal.push_back(input.z);
}
void NodeData::writeBuffer()
{

    if(m_writeBuffer.size()==0) return;
    FILE * oFile = fopen(m_dataPath.c_str(), "a+b");
    if(oFile!=NULL)
    {
        otimer.resume ();
        fwrite (m_writeBuffer.data() , sizeof(float), m_writeBuffer.size(), oFile);
        otimer.stop ();
        fclose (oFile);
    }
    else
    {
        cout << "ERROR: " << errno << ": " <<  strerror(errno) << endl;
        throw std::runtime_error("asd");
    }

    m_writeBuffer.clear();
    vector<float>().swap(m_writeBuffer);
//Normals
    if(m_writeBufferNormal.size()==0) return;
    oFile = fopen(m_dataPathNormal.c_str(), "a+b");
    if(oFile!=NULL)
    {
        otimer.resume ();
        fwrite (m_writeBufferNormal.data() , sizeof(float), m_writeBufferNormal.size(), oFile);
        otimer.stop ();
        fclose (oFile);
    }
    else
    {
        cout << "ERROR: " << errno << ": " <<  strerror(errno) << endl;
        throw std::runtime_error("asd");
    }

    m_writeBufferNormal.clear();
    vector<float>().swap(m_writeBufferNormal);
}

size_t NodeData::getMaxWriteBufferSize()
{
    return m_bufferSize;
}

size_t NodeData::getBufferSize()
{
    return m_writeBuffer.size();
}

Vertex<float> NodeData::get(int i)
{

//    cout << "s " << m_readBuffer.size() << endl;
    if(i>=m_readBufferIndex && i - m_readBufferIndex < m_readBuffer.size()/3)
    {
//        cout << "got data from buffer" << endl;
        Vertexf ret(m_readBuffer[(i - m_readBufferIndex)*3],m_readBuffer[((i - m_readBufferIndex)*3) + 1],m_readBuffer[((i - m_readBufferIndex)*3) +2]);
        return ret;
    }
    else
    {
//        cout << "read buffer again" << endl;
        fillBuffer(i);

        Vertexf ret(m_readBuffer[(i - m_readBufferIndex)*3],m_readBuffer[((i - m_readBufferIndex)*3) + 1],m_readBuffer[((i - m_readBufferIndex)*3) +2]);
        return ret;
    }


}

Vertex<float> NodeData::getNormal(int i)
{

//    cout << "s " << m_readBuffer.size() << endl;
    if(i>=m_readBufferIndexNormal && i - m_readBufferIndexNormal < m_readBufferNormal.size()/3)
    {
//        cout << "got data from buffer" << endl;
        Vertexf ret(m_readBufferNormal[(i - m_readBufferIndexNormal)*3],m_readBufferNormal[((i - m_readBufferIndexNormal)*3) + 1],m_readBufferNormal[((i - m_readBufferIndexNormal)*3) +2]);
        return ret;
    }
    else
    {
//        cout << "read buffer again" << endl;
        fillBufferNormal(i);

        Vertexf ret(m_readBufferNormal[(i - m_readBufferIndexNormal)*3],m_readBufferNormal[((i - m_readBufferIndexNormal)*3) + 1],m_readBufferNormal[((i - m_readBufferIndexNormal)*3) +2]);
        return ret;
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
    //writeBuffer();
    if(m_gotSize) return m_size;
    else
    {

        FILE * fp = fopen(m_dataPath.c_str(), "rb" );
        size_t sz = 0;
        if(fp != NULL)
        {
            itimer.resume ();
            fseek(fp, 0L, SEEK_END);
            sz = ftell(fp);
            itimer.stop ();
            fclose(fp);
            sz/=sizeof(float);
            sz/=3;
        }

//        ifstream ifs( m_dataPath );
//        size_t size = 0;
//        string s;
//        while( getline( ifs, s ) ) size++;

        m_size = sz;
        m_gotSize = true;
//        ifs.close();
        sz+=(m_writeBuffer.size()/3);
        return sz;
    }

}

}
