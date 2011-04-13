/*
 * PointCloudManager.tcc
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#include <cassert>
#include <string>
using std::string;

#include "../io/PLYIO.hpp"
#include <boost/filesystem.hpp>

namespace lssr
{

template<typename T>
BoundingBox<T>& PointCloudManager<T>::getBoundingBox()
{
    return m_boundingBox;
}

template<typename T>
T* PointCloudManager<T>::getPoint(size_t index)
{
    assert(index < m_numPoints);
    return m_points[index];
}

template<typename T>
size_t PointCloudManager<T>::getNumPoints()
{
    return m_numPoints;
}

template<typename T>
const T* PointCloudManager<T>::operator[](const size_t& index) const
{
    return m_points[index];
}


template<typename T>
void PointCloudManager<T>::readFromFile(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension = selectedFile.extension();

    if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        this->m_points = AsciiIO<T>::read(filename, this->m_numPoints);
    }
    else if(extension == ".ply")
    {
        // Read given input file
        PLYIO plyio;
        plyio.read(filename);
        this->m_points = plyio.getIndexedVertexArray(this->m_numPoints);
        this->m_normals = plyio.getIndexedNormalArray(this->m_numPoints);
    }
}

}

