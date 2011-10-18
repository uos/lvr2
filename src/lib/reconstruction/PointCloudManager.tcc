/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


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
    return VertexT(m_points[index][0], m_points[index][1], m_points[index][2], m_colors[index][0], m_colors[index][1], m_colors[index][2]);
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
        m_points  = loader->getIndexedPointArray( m_numPoints );
        size_t n(0);
        m_normals = loader->getIndexedPointNormalArray( n );
        if ( n != m_numPoints ) {
            m_normals = NULL;
        }
        m_colors  = loader->getIndexedPointColorArray( n );
        if ( n != m_numPoints ) {
            m_colors = NULL;
        }
    }
    else
    {
        cout << timestamp << "PointCloudManager::readFromFile: Unable to read point cloud data from "
             << filename << endl;
    }

}

}

