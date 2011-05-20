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
