/*
 * KdNode.h
 *
 *  Created on: 17.01.2013
 *      Author: dofeldsc
 */

#ifndef KDNODE_H_
#define KDNODE_H_

#include "geometry/Vertex.hpp"
#include "io/Model.hpp"
#include "io/DataStruct.hpp"
#include "geometry/Vertex.hpp"


namespace lssr{
template<typename VertexT>
class KdNode {
public:
	KdNode(coord3fArr points, VertexT min, VertexT max);
	virtual ~KdNode();

	double getnumpoints();

	void setnumpoints(double num);

	coord3fArr node_points;

	// Anzahl Punkte im Node
	double m_numpoints;

	/* The Bounce of the BoundingBox */
	VertexT m_maxvertex;
	VertexT m_minvertex;

};
}
#include "KdNode.tcc"
#endif /* KDNODE_H_ */
