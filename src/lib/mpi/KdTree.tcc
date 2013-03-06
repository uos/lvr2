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
	// Create a point loader object - Aus der Main (reconstruct)
	ModelFactory io_factory;

	// Übergabe noch variabel machen
	m_model = io_factory.readModel( "flur3.pts" );


	if (m_model != NULL)
	{
		m_loader = m_model->m_pointCloud;
	}
	else std::cout << " Model not existent" << endl;


	// Calculate bounding box
	m_points = m_loader->getIndexedPointArray(m_numpoint);


	// resize Boundingbox
	for(size_t i = 0; i < m_numpoint; i++)
	{
	    this->m_boundingBox.expand(m_points[i][0], m_points[i][1], m_points[i][2]);

	}

	// store the border of the boundingbox
	VertexT max = m_boundingBox.getMax();
	VertexT min = m_boundingBox.getMin();

	KdNode<VertexT> * child = new KdNode<VertexT>(m_points, min , max);
	child->setnumpoints(m_numpoint);

	this->Rekursion(child);

}

template<typename VertexT>
KdTree<VertexT>::KdTree(PointBufferPtr loader, long int max_p)
{

	max_points = max_p;

	if (loader != NULL)
	{
		m_loader = loader;
	}
	else std::cout << " Model == Null" << endl;

    // Calculate bounding box
	m_points = m_loader->getIndexedPointArray(m_numpoint);

	// Resize the bounding box to be read out max and min values
	for(size_t i = 0; i < m_numpoint; i++)
	{
	    this->m_boundingBox.expand(m_points[i][0], m_points[i][1], m_points[i][2]);

	}

	// read Min(lower left corner) and Max(upper right Corner)
	VertexT max = m_boundingBox.getMax();
	VertexT min = m_boundingBox.getMin();


	KdNode<VertexT> * child = new KdNode<VertexT>(m_points, min , max);
	child->setnumpoints(m_numpoint);

	this->Rekursion(child);

}


template<typename VertexT>

void KdTree<VertexT>::Rekursion(KdNode<VertexT> * first){


	boost::shared_array<size_t> tmp(new size_t [static_cast<unsigned int>(first->getnumpoints())]);
	//fülle Indize Buffer mit Initialen Werten
	for (size_t j = 0 ; j < first->getnumpoints(); j++)
	{
		// das zweite j wurd vorher auch gecastet
		tmp[static_cast<unsigned int>(j)] = j;
	}
	first->setIndizes(tmp);
	// start the recursion
	splitPointcloud(first);

	// store list of nodes in files
	int count = 1;
	char number [1];

	for (typename std::list<KdNode<VertexT>*>::iterator it=nodelist.begin() ; it != nodelist.end() ; ++it)
	{
		// composed of the file name
	    char name[256];
	    sprintf(name, "scan%03d.3d", count);
	    string filename(name);


		ModelFactory io_factory;
		// The factory requires a model to save.
		// The model in turn requires a Pointbuffer and a Meshbuffer (can be emtpy).
		// The Pointbuffer contains the Indexlist.

		PointBufferPtr pointcloud(new PointBuffer());


		size_t num = (*it)->getnumpoints();
		pointcloud->setIndexedPointArray((*it)->node_points, num );


		MeshBufferPtr mesh;
	    ModelPtr model( new Model( pointcloud, mesh ) );

		io_factory.saveModel(model, filename);

	    count++;
	}

}



template<typename VertexT>
void KdTree<VertexT>::splitPointcloud(KdNode<VertexT> * child)
{
	//std::cout << "start vom split" << std::endl;
	// recursion rule
	if (child->getnumpoints() == 0)
	{
		// nothing to do
	}
	else if ( child->getnumpoints() < MAX_POINTS)
	{
		// Node has less than MAX_POINTS in it, so store it in the list
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

		zmin = min[2];
		zmax = max[2];

		float dx, dy ,dz;
		dx = (xmax - xmin);
		dy = (ymax - ymin);
		dz = (zmax - zmin);


		// determine the split axis and the location
		// x = 0
		// y = 1
		// z = 2
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

		// termination criterion for the case that something went wrong
		if (xmin == xmax && ymin == ymax && zmin == zmax) return;

		VertexT child2_min = min;
		VertexT child1_max = max;
		child2_min[splitaxis] = split;
		child1_max[splitaxis] = split;


		int countleft = 0;
		int countright = 0;

		coord3fArr left(new coord<float>[m_numpoint]);
		coord3fArr right(new coord<float>[m_numpoint]);
		//double * left_indizes = (double*) calloc (m_numpoint, sizeof(double));
		//double * right_indizes = (double*) calloc (m_numpoint, sizeof(double));
		boost::shared_array<size_t> left_indizes(new size_t [static_cast<unsigned int>(m_numpoint)]);
		boost::shared_array<size_t> right_indizes(new size_t [static_cast<unsigned int>(m_numpoint)]);
		for (int y = 0 ; y < static_cast<unsigned int>(m_numpoint) ; y++)
		{
			left_indizes[y] = 0;
			right_indizes[y] = 0;
		}

		boost::shared_array<size_t> child_indizes = child->getIndizes();


		// divide the pointcloud (still on th axle average)
		for (size_t j = 0 ; j < child->getnumpoints(); j++)
		{
			//std::cout << "for-SChleife" << std::endl;
			if (child->node_points[j][splitaxis] <= split)
			{
				//std::cout << "if" << std::endl;
				left[countleft] = child->node_points[j];

				//std::cout << "bis hier tut alles" << std::endl;
				//abspeichern einer liste der Indizes aus der Anfangsmenge
				if (child_indizes[static_cast<unsigned int>(j)] == 0)
				{
					//std::cout << " hier ist der fehler, wahrscheinlich 2" << std::endl;
					left_indizes[countleft] = j;
				}
				else
				{
					//std::cout << " hier ist der fehler, wahrscheinlich" << std::endl;
					left_indizes[countleft] =  child->indizes[static_cast<unsigned int>(j)];
				}
				countleft++;
				//std::cout << "nach dem fehler" << std::endl;
			}
			else if (child->node_points[j][splitaxis] > split)
			{
				//std::cout << "else" << std::endl;
				right[countright] = child->node_points[j];

				//abspeichern einer liste der Indizes aus der Anfangsmenge
				if (child_indizes[static_cast<unsigned int>(j)] == 0)
				{
					//std::cout << " hier ist der fehler, wahrscheinlich 2" << std::endl;
					right_indizes[countright] = j;
				}
				else
				{
					//std::cout << " hier ist der fehler, wahrscheinlich" << std::endl;
					right_indizes[countright] =  child->indizes[static_cast<unsigned int>(j)];
				}
				//std::cout << "nach dem fehler" << std::endl;
				countright++;
			}
			else
			{
				std::cout << "!!!!!!!!!!!!!!Should never happen! Point is not considered in the interval!!!!!!!!!" << endl;
			}

		}// ende For


		// after splitting, initialize  nodes and recursive call
		KdNode<VertexT> * child1 = new KdNode<VertexT>(left, min , child1_max);
		KdNode<VertexT> * child2 = new KdNode<VertexT>(right, child2_min, max);
		child1->setnumpoints(countleft);
		child1->setIndizes(left_indizes);
		child2->setnumpoints(countright);
		child2->setIndizes(right_indizes);

		//recursion
		splitPointcloud(child1);
		splitPointcloud(child2);

	}// End else
}

template<typename VertexT>
KdTree<VertexT>::~KdTree() {
	// TODO Auto-generated destructor stub
}

template<typename VertexT>
std::list<KdNode<VertexT>*> KdTree<VertexT>::GetList(){
	return nodelist;
}
}
