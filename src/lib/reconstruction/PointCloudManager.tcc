/*
 * PointCloudManager.tcc
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#include <cassert>
#include <string>
using std::string;

#include "../io/Timestamp.hpp"
#include "../io/IOFactory.hpp"

#include <boost/filesystem.hpp>

namespace lssr
{

template<typename VertexT, typename NormalT>
BoundingBox<VertexT>& PointCloudManager<VertexT, NormalT>::getBoundingBox()
{
    return m_boundingBox;
}

template<typename VertexT, typename NormalT>
VertexT PointCloudManager<VertexT, NormalT>::getPoint(size_t index)
{
    assert(index < m_numPoints);
    return VertexT(m_points[index][0], m_points[index][1], m_points[index][2]);
}

template<typename VertexT, typename NormalT>
size_t PointCloudManager<VertexT, NormalT>::getNumPoints()
{
    return m_numPoints;
}

template<typename VertexT, typename NormalT>
const VertexT PointCloudManager<VertexT, NormalT>::operator[](const size_t& index) const
{
    return VertexT(m_points[index][0], m_points[index][1], m_points[index][2]);
}


template<typename VertexT, typename NormalT>
void PointCloudManager<VertexT, NormalT>::readFromFile(string filename)
{
    // Try to parse file
    IOFactory io(filename);

    // Get PoinLoader
    PointLoader* loader = io.getPointLoader();

    // Save points and normals (if present)
    if(loader)
    {
        m_points = loader->getPointArray();
        m_normals = loader->getPointNormalArray();
        m_numPoints = loader->getNumPoints();
    }
    else
    {
        cout << timestamp << "PointCloudManager::readFromFile: Unable to read point cloud data from "
             << filename << endl;
    }

}

}

