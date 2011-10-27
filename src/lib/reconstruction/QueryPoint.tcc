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
 * QueryPoint.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename VertexT>
QueryPoint<VertexT>::QueryPoint() {
	m_position = VertexT(0.0, 0.0, 0.0);
	m_distance = 0.0;
}

template<typename VertexT>
QueryPoint<VertexT>::QueryPoint(VertexT v){
	m_position = v;
	m_distance = 0.0;
}

template<typename VertexT>
QueryPoint<VertexT>::QueryPoint(VertexT v, float d){
	m_position = v;
	m_distance = d;
}

template<typename VertexT>
QueryPoint<VertexT>::QueryPoint(const QueryPoint<VertexT> &o){
	m_position = o.m_position;
	m_distance = o.m_distance;
}


} // namespace lssr
