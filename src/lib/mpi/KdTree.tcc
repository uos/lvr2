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
KdTree<VertexT>::KdTree(PointBufferPtr pointcloud)
{

	if (pointcloud != NULL)
	{
		m_loader = pointcloud;
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
void KdTree<VertexT>::Rekursion(KdNode<VertexT> * first){

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


		double countleft = 0;
		double countright = 0;


		coord3fArr left(new coord<float>[m_numpoint]);
		coord3fArr right(new coord<float>[m_numpoint]);

		// divide the pointcloud (still on th axle average)
		for (size_t j = 0 ; j < child->getnumpoints(); j++)
		{

			if (child->node_points[j][splitaxis] <= split)
			{
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
				std::cout << "!!!!!!!!!!!!!!Should never happen! Point is not considered in the interval!!!!!!!!!" << endl;
			}

		}


		// after splitting, initialize  nodes and recursive call
		KdNode<VertexT> * child1 = new KdNode<VertexT>(left, min , child1_max);
		KdNode<VertexT> * child2 = new KdNode<VertexT>(right, child2_min, max);
		child1->setnumpoints(countleft);
		child2->setnumpoints(countright);

		//recursion
		splitPointcloud(child1);
		splitPointcloud(child2);

	}// End else
}

template<typename VertexT>
KdTree<VertexT>::~KdTree() {
	// TODO Auto-generated destructor stub
}
}
