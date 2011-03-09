/*
 * PointCloudManager.tcc
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#include <cassert>
#include <string>
using std::string;

#include "PLYIO.hpp"
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
        AsciiIO<T>(filename, m_points, m_numPoints);
    }
    else if(extension == ".ply")
    {
        // Read given input file
        PLYIO plyio;
        plyio.read(filename);
        m_points = plyio.getIndexedVertexArray(m_numPoints);
        m_normals = plyio.getIndexedNormalArray(m_numPoints);
    }
}

}

