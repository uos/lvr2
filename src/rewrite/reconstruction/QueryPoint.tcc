/*
 * QueryPoint.cpp
 *
 *  Created on: 22.10.2008
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename T>
QueryPoint<T>::QueryPoint() {
	m_position = Vertex<T>(0.0, 0.0, 0.0);
	m_distance = 0.0;
}

template<typename T>
QueryPoint<T>::QueryPoint(Vertex<T> v){
	m_position = v;
	m_distance = 0.0;
}

template<typename T>
QueryPoint<T>::QueryPoint(Vertex<T> v, T d){
	m_position = v;
	m_distance = d;
}

template<typename T>
QueryPoint<T>::QueryPoint(const QueryPoint<T> &o){
	m_position = o.m_position;
	m_distance = o.m_distance;
}


} // namespace lssr
