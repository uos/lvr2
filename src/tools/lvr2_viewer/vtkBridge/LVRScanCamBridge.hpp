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
 * LVRScanCamBridge.hpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRSCANCAMBRIDGE_HPP_
#define LVRSCANCAMBRIDGE_HPP_

#include "lvr2/io/descriptions/ScanCameraIO.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"

#include "LVRPointBufferBridge.hpp"
#include "LVRMeshBufferBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

#include <boost/shared_ptr.hpp>

namespace lvr2
{

/**
 * @brief   Main class for conversion of LVR model instances to vtk actors. This class
 *          parses the internal model structures to vtk representations that can be
 *          added to a vtkRenderer instance.
 */
class LVRScanCamBridge
{
public:

    /**
     * @brief       Constructor. Parses the model information and generates vtk actor
     *              instances for the given data.
     */
    LVRScanCamBridge(ScanCameraPtr camera);

    LVRScanCamBridge(const LVRScanCamBridge& b);

    /**
     * @brief       Destructor.
     */
    virtual ~LVRScanCamBridge();

    /**
     * @brief       Adds the generated actors to the given renderer
     */
    void         addActors(vtkSmartPointer<vtkRenderer> renderer);

    /**
     * @brief       Removes the generated actors from the given renderer
     */
    void        removeActors(vtkSmartPointer<vtkRenderer> renderer);



    void		setVisibility(bool visible);


    // Declare model item classes as friends to have fast access to data chunks
    friend class LVRScanCamItem;

private:

};

typedef boost::shared_ptr<LVRScanCamBridge> ScanCamBridgePtr;

} /* namespace lvr2 */

#endif /* LVRSCANCAMBRIDGE_HPP_ */
