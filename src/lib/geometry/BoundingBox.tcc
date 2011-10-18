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
 * BoundingBox.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename VertexT>
BoundingBox<VertexT>::BoundingBox()
{
	float max_val = numeric_limits<float>::max();
	float min_val = numeric_limits<float>::min();

	m_min = VertexT(max_val, max_val, max_val);
	m_max = VertexT(min_val, min_val, min_val);
}

template<typename VertexT>
BoundingBox<VertexT>::BoundingBox(VertexT v1, VertexT v2)
{
	m_min = v1;
	m_max = v2;
}

template<typename VertexT>
BoundingBox<VertexT>::BoundingBox(float x_min, float y_min, float z_min,
		                          float x_max, float y_max, float z_max)
{
	m_min = VertexT(x_min, y_min, z_min);
	m_max = VertexT(x_max, y_max, z_max);
}

template<typename VertexT>
bool BoundingBox<VertexT>::isValid()
{
    float max_val = numeric_limits<float>::max();
    float min_val = numeric_limits<float>::min();

	VertexT v_min(min_val, min_val, min_val);
	VertexT v_max(max_val, max_val, max_val);
	return (m_min != v_max && m_max != v_min);
}

template<typename VertexT>
float BoundingBox<VertexT>::getRadius()
{
	// Shift bounding box to (0,0,0)
	VertexT m_min0 = m_min - m_centroid;
	VertexT m_max0 = m_max - m_centroid;

	// Return radius
	if(m_min0.length() > m_max0.length())
		return m_min0.length();
	else
		return m_max0.length();
}

template<typename VertexT>
inline void BoundingBox<VertexT>::expand(VertexT v)
{
    m_min[0] = std::min(v[0], m_min[0]);
    m_min[1] = std::min(v[1], m_min[1]);
    m_min[2] = std::min(v[2], m_min[2]);

    m_max[0] = std::max(v[0], m_max[0]);
    m_max[1] = std::max(v[1], m_max[1]);
    m_max[2] = std::max(v[2], m_max[2]);

    m_xSize = fabs(m_max[0] - m_min[0]);
    m_ySize = fabs(m_max[1] - m_min[1]);
    m_zSize = fabs(m_max[2] - m_min[2]);

    m_centroid = VertexT(m_max[0] - m_min[0],
                           m_max[1] - m_min[1],
                           m_max[2] - m_min[2]);

}

template<typename VertexT>
inline void BoundingBox<VertexT>::expand(float x, float y, float z)
{
    m_min[0] = std::min(x, m_min[0]);
    m_min[1] = std::min(y, m_min[1]);
    m_min[2] = std::min(z, m_min[2]);

    m_max[0] = std::max(x, m_max[0]);
    m_max[1] = std::max(y, m_max[1]);
    m_max[2] = std::max(z, m_max[2]);

    m_xSize = fabs(m_max[0] - m_min[0]);
    m_ySize = fabs(m_max[1] - m_min[1]);
    m_zSize = fabs(m_max[2] - m_min[2]);

    m_centroid = VertexT(m_min[0] + 0.5 * m_xSize,
                           m_min[1] + 0.5 * m_ySize,
                           m_min[2] + 0.5 * m_zSize);

}

template<typename VertexT>
inline void BoundingBox<VertexT>::expand(BoundingBox<VertexT>& bb)
{
    //expand(bb.m_centroid);
    expand(bb.m_min);
    expand(bb.m_max);
}

template<typename VertexT>
float BoundingBox<VertexT>::getLongestSide()
{
    return std::max(m_xSize, std::max(m_ySize, m_zSize));
}

template<typename VertexT>
float BoundingBox<VertexT>::getXSize()
{
    return m_xSize;
}

template<typename VertexT>
float BoundingBox<VertexT>::getYSize()
{
    return m_ySize;
}


template<typename VertexT>
float BoundingBox<VertexT>::getZSize()
{
    return m_zSize;
}

template<typename VertexT>
VertexT BoundingBox<VertexT>::getMin() const
{
    return m_min;
}


template<typename VertexT>
VertexT BoundingBox<VertexT>::getMax() const
{
    return m_max;
}



} // namespace lssr
