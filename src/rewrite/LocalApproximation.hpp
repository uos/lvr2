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

template<typename VertexType, typename DimType>
class LocalApproximation
{
public:
    virtual void getSurface(BaseMesh &mesh, PointCloudManager manager&, VertexType center, DimType voxelsize) = 0;
};


} // namspace lssr

#endif /* LOCALAPPROXIMATION_HPP_ */
