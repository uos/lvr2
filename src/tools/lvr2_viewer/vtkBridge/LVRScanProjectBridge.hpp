/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * LVRModel.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRSCANPROJECTBRIDGE_HPP_
#define LVRSCANPROJECTBRIDGE_HPP_


#include "lvr2/types/MatrixTypes.hpp"
#include "LVRScanPositionBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

#include <boost/shared_ptr.hpp>


namespace lvr2
{

enum ProjectScale
{
    mm = 1, cm = 10, m = 1000
};

/**
 * @brief   Main class for conversion of LVR ScanProjects instances to vtk actors. This class
 *          parses the internal ScanProject structures to vtk representations that can be
 *          added to a vtkRenderer instance.
 */
class LVRScanProjectBridge
{
public:

    /**
     * @brief       Constructor. Parses the model information and generates vtk actor
     *              instances for the given data.
     */
    LVRScanProjectBridge(ScanProjectPtr project, ProjectScale scale = m);

    LVRScanProjectBridge(const LVRScanProjectBridge& b);
    LVRScanProjectBridge(ModelBridgePtr project);
    /**
     * @brief       Destructor.
     */
    virtual ~LVRScanProjectBridge();

    /**
     * @brief       Adds the generated actors to the given renderer
     */
    void        addActors(vtkSmartPointer<vtkRenderer> renderer);

    /**
     * @brief       Removes the generated actors from the given renderer
     */
    void        removeActors(vtkSmartPointer<vtkRenderer> renderer);

    // Declare model item classes as friends to have fast access to data chunks
    friend class LVRScanProjectItem;

    ScanProjectPtr getScanProject();


    /**
     *  @brief      returns the scanpositions stored in the bridge
     */ 
    std::vector<ScanPositionBridgePtr> getScanPositions();

    /**
     *  @brief      set the ScanPositions to the given vector
     */
    void setScanPositions(std::vector<ScanPositionBridgePtr> scanPositions);

    ProjectScale getScale();

private:

    std::vector<ScanPositionBridgePtr> m_scanPositions;
    ScanProjectPtr m_scanproject;
    ProjectScale m_scale;

};

typedef boost::shared_ptr<LVRScanProjectBridge> ScanProjectBridgePtr;

} /* namespace lvr2 */

#endif /* LVRMODEL_HPP_ */
