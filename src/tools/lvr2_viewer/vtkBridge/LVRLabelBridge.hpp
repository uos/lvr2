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
#ifndef LVRLABELBRIDGE_HPP_
#define LVRLABELBRIDGE_HPP_


#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/types/ScanTypes.hpp"
#include "LVRPointBufferBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkRenderer.h>

#include <boost/shared_ptr.hpp>


namespace lvr2
{


/**
 * @brief   Main class for conversion of LVR ScanProjects instances to vtk actors. This class
 *          parses the internal ScanProject structures to vtk representations that can be
 *          added to a vtkRenderer instance.
 */
class LVRLabelBridge
{
public:

    /**
     * @brief       Constructor. Parses the model information and generates vtk actor
     *              instances for the given data.
     */
    LVRLabelBridge(LabelRootPtr project);

    LVRLabelBridge(const LVRLabelBridge& b);

    /**
     * @brief       Destructor.
     */
    virtual ~LVRLabelBridge();

    /**
     * @brief       Adds the generated actors to the given renderer
     */
    void        addActors(vtkSmartPointer<vtkRenderer> renderer);

    /**
     * @brief       Removes the generated actors from the given renderer
     */
    void        removeActors(vtkSmartPointer<vtkRenderer> renderer);

    bool        validPointBridge();
    void		setVisibility(bool visible);
    PointBufferBridgePtr getPointBridge()
    {
        return m_pointBridge;
    }
 
    // Declare model item classes as friends to have fast access to data chunks
    friend class LVRLabelItem;

    LabelRootPtr getLabelRoot();


private:

    PointBufferBridgePtr            m_pointBridge;
    LabelRootPtr m_labelRoot;

};

typedef boost::shared_ptr<LVRLabelBridge> LabelBridgePtr;

} /* namespace lvr2 */

#endif /* LVRMODEL_HPP_ */
