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
	/**
	 * @brief Der Konstruktor setzt die initialen Werte.
	 *
	 * @param points IndexBuffer der Punktwolke bzw- der Teilpunktwolke
	 * @param min Dieser Vektor repräsentiert den "untersten" Punkt der Szene ( minimaler x,y und z Wert)
	 * @param max Dieser Vektor repräsentiert den "obersten" Punkt der Szene (maximaler x, y, und z Wert)
	 */
	KdNode(coord3fArr points, VertexT min, VertexT max);
	virtual ~KdNode();

	/**
	 * @brief Gibt die Anzahl der enthaltenen Punkte des Nodes zurueck.
	 */
	double getnumpoints();

	/**
	 * @brief Setzt die Anzahl der enthaltenen Punkte des Nodes.
	 *
	 * @param num Zu setzender Wert.
	 */
	void setnumpoints(double num);

	/* Index Buffer fuer die Punkte */
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
