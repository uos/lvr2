/*
 * MCReconstruction.h
 *
 *  Created on: 16.02.2011
 *      Author: twiemann
 */

#ifndef MCRECONSTRUCTION_H_
#define MCRECONSTRUCTION_H_

#include "Reconstructor.hpp"

/**
 * @brief A surface reconstruction objects that implements the standard
 *        marching cubes algorithm.
 */
class MCReconstruction : public Reconstructor
{
public:
    MCReconstruction(PointCloudManger &manager) : Reconstructor(manager) {};
    virtual ~MCReconstruction();

protected:
    LocalApproximation          m_localApproximation;
};

#endif /* MCRECONSTRUCTION_H_ */
