/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */

/**
 * LVRModel.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMODEL_HPP_
#define LVRMODEL_HPP_

#include "io/Model.hpp"
#include "LVRPointBufferBridge.hpp"
#include "LVRMeshBufferBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

#include <boost/shared_ptr.hpp>

namespace lvr
{

struct Pose
{
    float x, y, z, r, t, p;
};

/**
 * @brief   Main class for conversion of LVR model instances to vtk actors. This class
 *          parses the internal model structures to vtk representations that can be
 *          added to a vtkRenderer instance.
 */
class LVRModelBridge
{
public:

    /**
     * @brief       Constructor. Parses the model information and generates vtk actor
     *              instances for the given data.
     */
    LVRModelBridge(ModelPtr model);

    LVRModelBridge(const LVRModelBridge& b);

    /**
     * @brief       Destructor.
     */
    virtual ~LVRModelBridge();

    /**
     * @brief       Adds the generated actors to the given renderer
     */
    void         addActors(vtkSmartPointer<vtkRenderer> renderer);

    /**
     * @brief       Removes the generated actors from the given renderer
     */
    void        removeActors(vtkSmartPointer<vtkRenderer> renderer);

    void        setPose(Pose& pose);
    Pose        getPose();

    // Declare model item classes as friends to have fast access to data chunks
    friend class LVRModelItem;

//private:

    PointBufferBridgePtr            m_pointBridge;
    MeshBufferBridgePtr             m_meshBridge;
    Pose                            m_pose;

};

typedef boost::shared_ptr<LVRModelBridge> ModelBridgePtr;

} /* namespace lvr */

#endif /* LVRMODEL_HPP_ */
