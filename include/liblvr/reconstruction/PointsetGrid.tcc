/*
 * PointsetGrid.tcc
 *
 *  Created on: Nov 27, 2014
 *      Author: twiemann
 */

#include "PointsetGrid.hpp"

namespace lvr
{

template<typename VertexT, typename BoxT>
PointsetGrid<VertexT, BoxT>::PointsetGrid(float cellSize, typename PointsetSurface<VertexT>::Ptr& surface, BoundingBox<VertexT> bb, bool isVoxelsize)
	: HashGrid<VertexT, BoxT>(cellSize, bb, isVoxelsize), m_surface(surface)
{
	PointBufferPtr buffer = surface->pointBuffer();

	VertexT v_min = this->m_boundingBox.getMin();
	VertexT v_max = this->m_boundingBox.getMax();

	// Get indexed point buffer pointer
	size_t num_points;
	coord3fArr points = this->m_surface->pointBuffer()->getIndexedPointArray(num_points);

	size_t index_x, index_y, index_z;

	cout << timestamp << "Creating Grid..." << endl;

	// Iterator over all points, calc lattice indices and add lattice points to the grid
	for(size_t i = 0; i < num_points; i++)
	{
		index_x = calcIndex((points[i][0] - v_min[0]) / this->m_voxelsize);
		index_y = calcIndex((points[i][1] - v_min[1]) / this->m_voxelsize);
		index_z = calcIndex((points[i][2] - v_min[2]) / this->m_voxelsize);
		this->addLatticePoint(index_x, index_y, index_z);
	}

}

template<typename VertexT, typename BoxT>
void PointsetGrid<VertexT, BoxT>::calcDistanceValues()
{
	// Status message output
	string comment = timestamp.getElapsedTime() + "Calculating distance values ";
	ProgressBar progress(this->m_queryPoints.size(), comment);

	Timestamp ts;

	// Calculate a distance value for each query point
	#pragma omp parallel for
	for( int i = 0; i < (int)this->m_queryPoints.size(); i++){
		float projectedDistance;
		float euklideanDistance;

		//cout << euklideanDistance << " " << projectedDistance << endl;

		this->m_surface->distance(this->m_queryPoints[i].m_position, projectedDistance, euklideanDistance);
		if (euklideanDistance > 1.7320 * this->m_voxelsize)
		{
			this->m_queryPoints[i].m_invalid = true;
		}
		this->m_queryPoints[i].m_distance = projectedDistance;
		++progress;
	}
	cout << endl;
	cout << timestamp << "Elapsed time: " << ts << endl;
}

template<typename VertexT, typename BoxT>
PointsetGrid<VertexT, BoxT>::~PointsetGrid()
{
	// TODO Auto-generated destructor stub
}

} /* namespace lvr */
