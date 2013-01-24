/*
 * KdTree.cpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

//#include "KdTree.h"

#include <cstdlib>

#define MAX_POINTS 10000
namespace lssr{
template<typename VertexT>
KdTree<VertexT>::KdTree()
{
	std::cout << "Konstruktor von KdTree wurde aufgerufen" << endl;
	// TODO Auto-generated constructor stub

	// Create a point loader object - Aus der Main (reconstruct)
	ModelFactory io_factory;
	// Übergabe noch variabel machen
	m_model = io_factory.readModel( "flur3.pts" );
	std::cout << "Datei wurde eingelesen" << endl;

	if (m_model != NULL)
	{
		m_loader = m_model->m_pointCloud;
	}
	else std::cout << " Model ist nicht vorhanden" << endl;
	std::cout << "model wurde ausgelesen" << endl;
    // Calculate bounding box
	m_points = m_loader->getIndexedPointArray(m_numpoint);
	cout << "Anzahl der Punkte:" << m_numpoint << endl;
	// Anpassen der Bounding Box, damit max und min x,y und z Werte ausgelesen werden können
	for(size_t i = 0; i < m_numpoint; i++)
	{
	    this->m_boundingBox.expand(m_points[i][0], m_points[i][1], m_points[i][2]);

	}

	// lese max und min von x,y und z aus
	VertexT max = m_boundingBox.getMax();
	VertexT min = m_boundingBox.getMin();
	std::cout << "Vertecies wurden erstellt" <<endl;
	std::cout << "AusgangsVertex: "<< min[0] << "  " << min[1] << "  " << min[2] << endl;
	std::cout << "AusgangsVertex: "<< max[0] << "  " << max[1] << "  " << max[2] << endl;

	sleep(1);
	KdNode<VertexT> * child = new KdNode<VertexT>(m_points, min , max);
	child->setnumpoints(m_numpoint);

	this->Rekursion(child);

}
template<typename VertexT>
void KdTree<VertexT>::Rekursion(KdNode<VertexT> * first){
	std::cout << "Rekursion wurde gestartet" << endl;
	splitPointcloud(first);

	std::cout << "Rekursion ist durchlaufen" << endl;

	// store list
	int count = 1;
	char number [1];

	for (typename std::list<KdNode<VertexT>*>::iterator it=nodelist.begin() ; it != nodelist.end() ; ++it)
	{
	   // std::string filename = "scan" + std::string(count) + ".3d";
	    char name[256];
	    sprintf(name, "scan%03d.3d", count);
	    string filename(name);


		ModelFactory io_factory;
		/* Die Factory benötigt ein Model zum speichern.
		 * Das Model wiederum benötigt ein PointBuffer und ein Meshbuffer.
		 * In dem Pointbuffer wird die Indexliste gespeichert.
		 */
		PointBufferPtr pointcloud(new PointBuffer());

		size_t num = (*it)->getnumpoints();
		pointcloud->setIndexedPointArray((*it)->node_points, num );

		std::cout << "Ich bin Nummer: " << count << "und habe soviele Einträge: " << (*it)->getnumpoints() << endl;
		std::cout << "Mein erster Eintrag sieht so aus: " << (*it)->node_points[0][0] << " " <<(*it)->node_points[0][1] << " "<< (*it)->node_points[0][2] << endl;
		std::cout << "Mein letzter Eintrag sieht so aus: " << (*it)->node_points[(*it)->getnumpoints() -1][0] << " " <<(*it)->node_points[(*it)->getnumpoints() - 1][1] << " "<< (*it)->node_points[(*it)->getnumpoints() - 1][2] << endl;

		MeshBufferPtr mesh;
	    ModelPtr model( new Model( pointcloud, mesh ) );

		io_factory.saveModel(model, filename);

	    count++;
	}

}


//void splitPointcloud(coord3fArr points , VertexT min , VertexT max)
template<typename VertexT>
void KdTree<VertexT>::splitPointcloud(KdNode<VertexT> * child)
{
	std::cout << " " << endl;
	//sleep(1);
	std::cout << "Anzahl Punkte bei aufruf split: " << child->getnumpoints() << endl;
	//std::cout << "Split-Methode wurde aufgerufen" << endl;
	if (child->getnumpoints() == 0)
	{
		// nothing to do
	}
	else if ( child->getnumpoints() < MAX_POINTS)
	{

		//hier fertig -> reaktion
		nodelist.push_back(child);
		std::cout << "Wurde richtig erkannt und in liste gespeichert. Liste hat jetzt: " << nodelist.size() << endl;
	}
	// weiter aufteilen
	else
	{
		VertexT min = child->m_minvertex;
		VertexT max = child->m_maxvertex;

		std::cout << "min: "<<  min[0] << "  " << min[1] << "  " << min[2] << endl;
		std::cout << "max " << max[0] << "  " << max[1] << "  " << max[2] << endl;

		float xmin, xmax, ymin, ymax, zmin, zmax;
		xmin = min[0];
		xmax = max[0];

		ymin = min[1];
		ymax = max[1];

		zmin = min[2];
		zmax = max[2];

		float dx, dy ,dz;
		dx = (xmax - xmin);
		dy = (ymax - ymin);
		dz = (zmax - zmin);

		/*
		 * bestimmen der Achse an der gesplittet wird
		 * eigentlich unguenstig, da nicht gesagt ist das sich so am besten die punkte aufteilen lassen
		 * x = 0
		 * y = 1
		 * z = 2
		 */

		int splitaxis;
		float split;

		if ( dx > dy)
		{
			if ( dx > dz)
			{
				splitaxis = 0;
				split = ( xmin + xmax ) / 2.0f;
			}
			else
			{
				splitaxis = 2;
				split = ( zmin + zmax ) / 2.0f;
			}
		}
		else
		{
			if ( dy > dz)
			{
				splitaxis = 1;
				split = ( ymin + ymax ) / 2.0f;
			}
			else
			{
				splitaxis = 2;
				split = ( zmin + zmax ) / 2.0f;
			}
		}
		std::cout << "Achse an der geteilt wird:" << splitaxis << endl;
		std::cout << "Stelle an der geteilt wird: " << split << endl;

		/* Es gibt öfter den fall das Achsen "gleich" sind und deshalb nicht mehr richtig aufgeteilt wird */
		if (xmin == xmax && ymin == ymax && zmin == zmax) return;

		VertexT child2_min = min;
		VertexT child1_max = max;
		child2_min[splitaxis] = split;
		child1_max[splitaxis] = split;

		std::cout << "child2 min: " << child2_min[0] << "  " << child2_min[1] << "  " << child2_min[2] << endl;
		std::cout << "child1 max " << child1_max[0] << "  " << child1_max[1] << "  " << child1_max[2] << endl;

		double countleft = 0;
		double countright = 0;

		// vllt anderer Datentyp?!
		coord3fArr left(new coord<float>[m_numpoint]); //= new coord3fArr[m_numpoint]; //(coord3fArr ) malloc(sizeof(coord3fArr) * m_numpoint);
		coord3fArr right(new coord<float>[m_numpoint]);// = (coord3fArr ) malloc(sizeof(coord3fArr) * m_numpoint);

		/* Bestimmung des Medians / Mittelwerts der Achse an der geslittet werden soll*/
		for (size_t j = 0 ; j < child->getnumpoints(); j++)
		{



			if (child->node_points[j][splitaxis] <= split)
			{
				//richtig kopiert????
				left[countleft] = child->node_points[j];
				countleft++;
			}
			else if (child->node_points[j][splitaxis] > split)
			{
				right[countright] = child->node_points[j];
				countright++;
			}
			else
			{
				std::cout << "!!!!!!!!!!!!!!Sollte niemals passieren. Punkt nicht im betrachteten Interval!!!!!!!!!" << endl;
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
