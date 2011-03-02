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

    /**
     * @brief Constructor.
     *
     * @param manager       A point cloud manager instance
     * @param resolution    The number of intersections (on the longest side
     *                      of the volume tabe by the data points) used by
     *                      the reconstruction.
     */
    MCReconstruction(PointCloudManager<CoordType> &manager,  int resolution)
        : Reconstructor<CoordType, IndexType>(manager)  {};

    /**
     * @brief Destructor.
     */
    virtual ~MCReconstruction() {};

    /**
     * @brief Returns the surface reconstruction of the given point set.
     *
     * @param mesh
     */
    virtual void getMesh(BaseMesh<Vertex<CoordType>, IndexType> &mesh);

private:



    /// The used voxelsize fpr reconstruction
    CoordType              m_voxelsize;

};


} // namespace lssr


#include "MCReconstruction.tcc"

#endif /* MCRECONSTRUCTION_H_ */
