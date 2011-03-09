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

template<typename CoordType, typename IndexType>
class LocalApproximation
{
public:
    virtual void getSurface(BaseMesh<Vertex<CoordType>, IndexType> &mesh,
            PointCloudManager<CoordType> &manager,
            IndexType globalIndex);

};


} // namspace lssr

#endif /* LOCALAPPROXIMATION_HPP_ */
