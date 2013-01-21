/*
 * KdNode.cpp
 *
 *  Created on: 17.01.2013
 *      Author: dofeldsc
 */
//#include "KdNode.h"

namespace lssr
{

template<typename VertexT>
KdNode<VertexT>::KdNode(coord3fArr points, VertexT min, VertexT max)
{
	m_minvertex = min;
	m_maxvertex = max;
	node_points = points;
	m_numpoints = 0;

}

template<typename VertexT>
double KdNode<VertexT>::getnumpoints(){
	return m_numpoints;
}

template<typename VertexT>
void KdNode<VertexT>::setnumpoints(double num){
	m_numpoints = num;
}

template<typename VertexT>
KdNode<VertexT>::~KdNode() {
	// TODO Auto-generated destructor stub
}

}

