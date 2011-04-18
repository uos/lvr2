/*
 * LocalApproximation.hpp
 *
 *  Created on: 15.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef LOCALAPPROXIMATION_HPP_
#define LOCALAPPROXIMATION_HPP_

#include "../geometry/BaseMesh.hpp"
#include "../reconstruction/PointCloudManager.hpp"

namespace lssr
{

/**
 * @brief	An interface class for local approximation operations
 * 			(e.g. in a Marching Cubes box)
 */
template<typename CoordType, typename IndexType>
class LocalApproximation
{
public:

	/**
	 * @brief	Adds the local reconstruction to the given mesh
	 *
	 * @param	mesh		 The used mesh.
	 * @param	manager		 A point cloud manager object
	 * @param	globalIndex	 The index of the latest vertex in the mesh
	 */
    virtual void getSurface(BaseMesh<Vertex<CoordType>, IndexType> &mesh,
            PointCloudManager<CoordType> &manager,
            IndexType globalIndex);

};


} // namspace lssr

#endif /* LOCALAPPROXIMATION_HPP_ */
