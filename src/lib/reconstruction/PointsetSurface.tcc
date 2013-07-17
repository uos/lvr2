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

/**
 * PointsetSurface.tcc
 *
 *  @date 25.01.2012
 *  @author Thomas Wiemann
 */
namespace lssr
{

template<typename VertexT>
PointsetSurface<VertexT>::PointsetSurface(PointBufferPtr pointcloud)
    : m_pointBuffer(pointcloud)
{
    // Calculate bounding box
    size_t numPoints;
    coord3fArr points = this->m_pointBuffer->getIndexedPointArray(numPoints);

    for(size_t i = 0; i < numPoints; i++)
    {
        this->m_boundingBox.expand(points[i][0], points[i][1], points[i][2]);
    }
}

template<typename VertexT>
void PointsetSurface<VertexT>::expandBoundingBox( 
	  float xmin, float ymin, float zmin,
	  float xmax, float ymax, float zmax)
{
    this->m_boundingBox.expand(xmin, ymin, zmin);
    this->m_boundingBox.expand(xmax, ymax, zmax);
}

template<typename VertexT>
VertexT PointsetSurface<VertexT>::getInterpolatedNormal(VertexT position)
{
	vector< ulong > indices;
	VertexT result(0,0,0);
	size_t n;
	this->searchTree()->kSearch(position, this->m_kn, indices);
	for (int i = 0; i < this->m_kn; i++)
	{
		result[0] += this->pointBuffer()->getIndexedPointNormalArray(n)[indices[i]][0];
		result[1] += this->pointBuffer()->getIndexedPointNormalArray(n)[indices[i]][1];
		result[2] += this->pointBuffer()->getIndexedPointNormalArray(n)[indices[i]][2];
	}
	result /= this->m_kn;
	return result;
}

} // namespace lssr
