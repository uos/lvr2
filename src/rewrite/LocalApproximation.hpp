/*
 * LocalApproximation.hpp
 *
 *  Created on: 15.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef LOCALAPPROXIMATION_HPP_
#define LOCALAPPROXIMATION_HPP_

#include "BaseMesh.hpp"
#include "PointCloudManager.hpp"

namespace lssr
{

template<typename VertexType, typename IndexType, typename DimType, typename PointType>
class LocalApproximation
{
public:
    virtual void getSurface(BaseMesh<VertexType, IndexType> &mesh,
            PointCloudManager<PointType> &manager,
            VertexType center,
            DimType voxelsize) = 0;
};


} // namspace lssr

#endif /* LOCALAPPROXIMATION_HPP_ */
