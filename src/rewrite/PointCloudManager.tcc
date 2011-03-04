/*
 * PointCloudManager.tcc
 *
 *  Created on: 02.03.2011
 *      Author: Thomas Wiemann
 */

#include <cassert>

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

}

