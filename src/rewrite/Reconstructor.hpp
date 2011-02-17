/*
 * Reconstructor.h
 *
 *  Created on: 16.02.2011
 *      Author: twiemann
 */

#ifndef RECONSTRUCTOR_H_
#define RECONSTRUCTOR_H_

#include "BaseMesh.hpp"
#include "PointCloudManager.hpp"

/**
 * @brief Interface class for surface reconstruction algorithms
 *        that generate triangle meshes.
 */
class Reconstructor
{
public:

    /**
     * @brief Constructs a Reconstructor object using the given point
     *        cloud handler
     */
    Reconstructor(PointCloudManger &manager) : m_manager(manager) {}

    /**
     * @brief Generated a triangle mesh representation of the current
     *        point set.
     *
     * @param mesh      A surface representation of the current point
     *                  set.
     */
    virtual void getMesh(BaseMesh& mesh) = 0;

protected:

    /// The point cloud manager that handles the loaded point cloud data.
    PointCloudManager&      m_manager;
};

#endif /* RECONSTRUCTOR_H_ */
