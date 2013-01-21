/*
 * KdTree.cpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

//#include "KdTree.h"

#include <cstdlib>

#define MAX_POINTS 1000
namespace lssr{
template<typename VertexT>
KdTree<VertexT>::KdTree()
{
	// TODO Auto-generated constructor stub

	// Create a point loader object - Aus der Main (reconstruct)
	ModelFactory io_factory;
	// Übergabe noch variabel machen
	m_model = io_factory.readModel( "flur3.pts" );


	m_loader = m_model->m_pointCloud;

    // Calculate bounding box
	m_points = m_loader->getIndexedPointArray(m_numpoint);
	cout << m_numpoint << endl;
	// Anpassen der Bounding Box, damit max und min x,y und z Werte ausgelesen werden können
	for(size_t i = 0; i < m_numpoint; i++)
	{
	    this->m_boundingBox.expand(m_points[i][0], m_points[i][1], m_points[i][2]);

	}

	// lese max und min von x,y und z aus
	VertexT max = m_boundingBox.getMax();
	VertexT min = m_boundingBox.getMin();

	KdNode<VertexT> * child = new KdNode<VertexT>(m_points, min , max);

	this->Rekursion(child);

}
template<typename VertexT>
void KdTree<VertexT>::Rekursion(KdNode<VertexT> * first){
	splitPointcloud(first);

	std::cout << "Rekursion ist durchlaufen" << endl;

	// store list
	int count = 1;
	char number [1];

	for (typename std::list<KdNode<VertexT>*>::iterator it=nodelist.begin() ; it != nodelist.end() ; ++it)
	{
	   // std::string filename = "scan" + std::string(count) + ".3d";
	    string filename = "test";
		count++;

		ModelFactory io_factory;
	    ModelPtr model( new Model( PointBufferPtr( new PointBuffer)));

		io_factory.saveModel(model, filename);
	}

}


//void splitPointcloud(coord3fArr points , VertexT min , VertexT max)
template<typename VertexT>
void KdTree<VertexT>::splitPointcloud(KdNode<VertexT> * child)
{
	if ( child->node_points.getnumpoints() < MAX_POINTS)
	{
		//hier fertig -> reaktion
		nodelist.push_back(child);
	}
	// weiter aufteilen
	else
	{
		VertexT min = child->m_minvertex;
		VertexT max = child->m_maxvertex;
		float xmin, xmax, ymin, ymax, zmin, zmax;
		xmin = min[0];
		xmax = max[0];

		ymin = min[1];
		ymax = max[1];

		zmin = min[1];
		zmax = max[2];

		float dx, dy ,dz;
		dx = (m_xmax - m_xmin);
		dy = (m_ymax - m_ymin);
		dz = (m_zmax - m_zmin);

		/*
		 * bestimmen der Achse an der gesplittet wird
		 * eigentlich unguenstig, da nicht gesagt ist das sich so am besten die punkte aufteilen lassen
		 * x = 0
		 * y = 1
		 * z = 2
		 */
		if ( dx > dy)
		{
			if ( dx > dz)
			{
				splitaxis = 0;
				split = ( m_xmin + m_xmax ) / 2;
			}
			else
			{
				splitaxis = 2;
				split = ( m_zmin + m_zmax ) / 2;
			}
		}
		else
		{
			if ( dy > dz)
			{
				splitaxis = 1;
				split = ( m_ymin + m_ymax ) / 2;
			}
			else
			{
				splitaxis = 2;
				split = ( m_zmin + m_zmax ) / 2;
			}
		}

		VertexT child2_min = max;
		VertexT child1_max = min;
		child2_min[splitaxis] = split;
		child1_max[splitaxis] = split;

		double countleft = 0;
		double countright = 0;

		/* Bestimmung des Medians / Mittelwerts der Achse an der geslittet werden soll*/
		for (size_t j = 0 ; j < m_numpoint; j++)
		{

			// vllt anderer Datentyp?!
			coord3fArr left, right;

			if (m_points[j][splitaxis] <= split)
			{
				//richtig kopiert????
				left[countleft] = m_points[j];
				countleft++;
			}
			else
			{
				right[countright] = m_points[j];
				countright++;
			}

		}


		// nach Aufteilung Nodes initialisieren und Rekursiver aufruf
		KdNode<VertexT> * child1 = new KdNode<VertexT>(left, min , child1_max);
		KdNode<VertexT> * child2 = new KdNode<VertexT>(right, child2_min, max);
		child1->setnumpoints(countleft);
		child2->setnumpoints(countright);

		//Rekursion
		splitPointcloud(child1);
		splitPointcloud(child2);

	}// Ende else fall
}

template<typename VertexT>
KdTree<VertexT>::~KdTree() {
	// TODO Auto-generated destructor stub
}
}
