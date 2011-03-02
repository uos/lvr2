/*
 * MCReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef MCRECONSTRUCTION_H_
#define MCRECONSTRUCTION_H_

#include "Vertex.hpp"
#include "BoundingBox.hpp"
#include "Options.hpp"
#include "Reconstructor.hpp"
#include "LocalApproximation.hpp"

namespace lssr
{

/**
 * @brief A surface reconstruction objects that implements the standard
 *        marching cubes algorithm.
 */
template<typename CoordType, typename IndexType>
class MCReconstruction : public Reconstructor<CoordType, IndexType>
{
public:
    MCReconstruction(PointCloudManager<CoordType> &manager, CoordType voxelsize) : Reconstructor<CoordType, IndexType>(manager)  {};
    virtual ~MCReconstruction() {};
    virtual void getMesh(BaseMesh<Vertex<CoordType>, IndexType> &mesh) {};

private:

    CoordType              m_voxelsize;

};


} // namespace lssr

#endif /* MCRECONSTRUCTION_H_ */
