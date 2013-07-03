/*
 * KdTree.cpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

//#include "KdTree.h"

#include <cstdlib>


namespace lssr{
template<typename VertexT>
bool KdTree<VertexT>::compare_nocase (KdNode<VertexT> * first, KdNode<VertexT> * second)
{
    if ( first->getnumpoints() > second->getnumpoints() ) return true;
    else return false;
}

template<typename VertexT>
KdTree<VertexT>::KdTree(PointBufferPtr loader, long int max_p , long int min_p, bool median)
{

	max_points = max_p;
	min_points = min_p;
	m_median   = median;

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
	
	// start the main-programm
	this->Rekursion(child);

	// sort the list
	nodelist.sort(compare_nocase);

}


template<typename VertexT>
void KdTree<VertexT>::Rekursion(KdNode<VertexT> * first){


	boost::shared_array<size_t> tmp(new size_t [static_cast<unsigned int>(first->getnumpoints())]);
	
	//fill indice Buffer initial values
	for (size_t j = 0 ; j < first->getnumpoints(); j++)
	{
		// das zweite j wurd vorher auch gecastet
		tmp[static_cast<unsigned int>(j)] = j;
	}
	first->setIndizes(tmp);
	
	// start the recursion
	splitPointcloud(first, 0);

}



template<typename VertexT>
void KdTree<VertexT>::splitPointcloud(KdNode<VertexT> * child , int again)
{
  
	float median = 0.0f;
	
	// recursion rule
	if (child->getnumpoints() == 0 || child->again == 6)
	{

		child->node_points.reset();
		child->indizes.reset();
		delete child;
		// nothing to do
	}
	else if ( child->getnumpoints() < max_points)
	{

		// Node has less than MAX_POINTS in it, so store it in the list
		nodelist.push_back(child);
	}
	// some splits are still necessary
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

		//get length of axis
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
		if (!m_median)
		{
		// determine the split axis and the location at center or some special case
			if ( dx > dy )
			{
				if ( dx > dz)
				{
					splitaxis = 0;
					if      (again == 0) split = ( xmin + xmax ) / 2.0f;
					else if (again == 1) split = ( ( xmin + xmax ) / 2.0f ) + ( ( (child->again * 2) * ( xmax - xmin)) / 11 );
					else if (again == 2) split = ( ( xmin + xmax ) / 2.0f ) - ( ( (child->again * 2) * ( xmax - xmin)) / 11 );
				}
				else
				{
					splitaxis = 2;
					if      (again == 0) split = ( zmin + zmax ) / 2.0f;
					else if (again == 1) split = ( ( zmin + zmax ) / 2.0f ) + ( ( (child->again * 2) * ( zmax - zmin)) / 11 );
					else if (again == 2) split = ( ( zmin + zmax ) / 2.0f ) - ( ( (child->again * 2) * ( zmax - zmin)) / 11 );
				}
			}
			else
			{
				if ( dy > dz)
				{
					splitaxis = 1;
					if      (again == 0) split = ( ymin + ymax ) / 2.0f;
					else if (again == 1) split = ( ( ymin + ymax ) / 2.0f ) + ( ( (child->again * 2 ) * ( ymax - ymin)) / 11 );
					else if (again == 2) split = ( ( ymin + ymax ) / 2.0f ) - ( ( (child->again * 2 ) * ( ymax - ymin)) / 11 );
				}
				else
				{
					splitaxis = 2;
					if      (again == 0) split = ( zmin + zmax ) / 2.0f;
					else if (again == 1) split = ( ( zmin + zmax ) / 2.0f ) + ( ( (child->again * 2) * ( zmax - zmin)) / 11 );
					else if (again == 2) split = ( ( zmin + zmax ) / 2.0f ) - ( ( (child->again * 2) * ( zmax - zmin)) / 11 );
				}
			}
		}
		else
		{
		// determine the split axis and the location with median
			vector<float> med;
		
			if ( dx > dy )
			{
				if ( dx > dz) splitaxis = 0;
				else          splitaxis = 2;
			}
			else
			{
				if ( dy > dz) splitaxis = 1;
				else	      splitaxis = 2;
			}
	  	
			for(size_t j = 0 ; j < child->getnumpoints(); j++)
	  		{
	    			med.push_back(child->node_points[j][splitaxis]);
	  		}
	  		std::sort(med.begin(), med.end());
	
		  	int tmp;
		 	  
		 	 tmp = child->getnumpoints() / 2;
		  	
			  median = med[tmp];
			  split = median;
		 
			if(split == min[splitaxis]) split += (max[splitaxis] - min[splitaxis]) / 4.0f;
			if(split == max[splitaxis]) split -= (max[splitaxis] - min[splitaxis]) / 4.0f;
	
		}

		// count the loops with the same pointcloud
		if ( again != 0) child->again++;
  
		// termination criterion for the case that something went wrong
		if (xmin == xmax && ymin == ymax && zmin == zmax) return;

		// set new range
		VertexT child2_min = min;
		VertexT child1_max = max;
		child2_min[splitaxis] = split;
		child1_max[splitaxis] = split;
	
		// count the points in both regions
		int not_count_left  = 0;
		int not_count_right = 0;
		for (size_t j = 0 ; j < child->getnumpoints(); j++)
		 {
			if (child->node_points[j][splitaxis] <= split)
			{
			    not_count_left++;
			}
			else
			{
			    not_count_right++;
			} 
	    	}		
		
		
	   	 if ( not_count_left > ( min_points ) && not_count_right > ( min_points ) )
	   	 {
		    int countleft = 0;
		    int countright = 0;

		    // Array for the new Regions
		    coord3fArr left(new coord<float>[static_cast<unsigned int>(not_count_left)]);
		    coord3fArr right(new coord<float>[static_cast<unsigned int>(not_count_right)]);



		    boost::shared_array<size_t> left_indizes(new size_t [static_cast<unsigned int>(not_count_left)]);
		    boost::shared_array<size_t> right_indizes(new size_t [static_cast<unsigned int>(not_count_right)]);
		    
		    // count points in both parts
		    for (int y = 0 ; y < static_cast<unsigned int>(not_count_left) ; y++)
		    {
			    left_indizes[y] = 0;
		    }
		    for (int y = 0 ; y < static_cast<unsigned int>(not_count_right) ; y++)
		    {
			  right_indizes[y] = 0;
		    }
    
		    boost::shared_array<size_t> child_indizes = child->getIndizes();

		    // divide the pointcloud (still on th axle average)
		    for (size_t j = 0 ; j < child->getnumpoints(); j++)
		    {		  
			    if (child->node_points[j][splitaxis] <= split)
			    {
				    left[countleft] = child->node_points[j];
				
				    //save the global Indizes
				    if (child_indizes[static_cast<unsigned int>(j)] == 0)
				    {					  
					    left_indizes[countleft] = j;
				    }
				    else
				    {
					
					    left_indizes[countleft] =  child->indizes[static_cast<unsigned int>(j)];
				    }
				    countleft++;
				
			    }
			    else if (child->node_points[j][splitaxis] > split)
			    {
				    right[countright] = child->node_points[j];
    
				    //save the global Indizes
				    if (child_indizes[static_cast<unsigned int>(j)] == 0)
				    {
					    right_indizes[countright] = j;
				    }
				    else
				    {
					    right_indizes[countright] =  child->indizes[static_cast<unsigned int>(j)];
				    }
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

		    // free Memory
		    child_indizes.reset();
		    child->node_points.reset();
		    delete child;
		    //recursion

		    splitPointcloud(child1, 0);
		    splitPointcloud(child2, 0);
		}
		else
		{
		  

		    if ( not_count_left == 0 )
		    {	
			KdNode<VertexT> * child2 = new KdNode<VertexT>(child->node_points, child2_min, max);
			child2->setnumpoints(not_count_right);
			child2->setIndizes(child->indizes);
			splitPointcloud(child2, 0);
		    }
		    else if (not_count_right == 0 )
		    {
			KdNode<VertexT> * child1 = new KdNode<VertexT>(child->node_points, min , child1_max);
			child1->setnumpoints(not_count_left);
			child1->setIndizes(child->indizes);
			splitPointcloud(child1, 0);
		    }
		    else
		    {
			if ( not_count_left < not_count_right) splitPointcloud(child, 1);
			else splitPointcloud(child, 2);
		    }
		}

	}// End else
}

template<typename VertexT>
KdTree<VertexT>::~KdTree() {
	// TODO Auto-generated destructor stub
}

template<typename VertexT>
BoundingBox<VertexT> KdTree<VertexT>::GetBoundingBox() {
    return m_boundingBox;
} 

template<typename VertexT>
std::list<KdNode<VertexT>*> KdTree<VertexT>::GetList(){
	return nodelist;
}

}
