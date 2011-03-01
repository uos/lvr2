/*
 * Reconstructor.h
 *
 *  Created on: 16.02.2011
 *      Author: Thomas Wiemann
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include "BaseMesh.hpp"
#include "PointCloudManager.hpp"

namespace lssr
{

/**
 * @brief Interface class for surface reconstruction algorithms
 *        that generate triangle meshes.
 */
template<typename CoordType, typename IndexType>
class Reconstructor
{
public:

    /**
     * @brief Constructs a Reconstructor object using the given point
     *        cloud handler
     */
    Reconstructor(PointCloudManager<CoordType> &manager) : m_manager(manager) {}

    /**
     * @brief Generated a triangle mesh representation of the current
     *        point set.
     *
     * @param mesh      A surface representation of the current point
     *                  set.
     */
    virtual void getMesh(BaseMesh<Vertex<CoordType>, IndexType>& mesh) = 0;

protected:

    /// The point cloud manager that handles the loaded point cloud data.
    PointCloudManager<CoordType>&      m_manager;
};

} //namespace lssr

#endif /* RECONSTRUCTOR_H_ */
