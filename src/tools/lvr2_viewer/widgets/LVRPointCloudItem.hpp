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
 * LVRPointCloudItem.hpp
 *
 *  @date Feb 7, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRPOINTCLOUDITEM_HPP_
#define LVRPOINTCLOUDITEM_HPP_

#include "../vtkBridge/LVRPointBufferBridge.hpp"

#include <QTreeWidgetItem>
#include <QColor>

namespace lvr2
{

class LVRPointCloudItem : public QTreeWidgetItem
{
public:

    LVRPointCloudItem(PointBufferBridgePtr ptr, QTreeWidgetItem* parent = 0);
    virtual ~LVRPointCloudItem();
    QColor	getColor();
    void    setColor(QColor &c);
    void    setSelectionColor(QColor &c);
    void    resetColor();
    int		getPointSize();
    void    setPointSize(int &pointSize);
    float	getOpacity();
    void    setOpacity(float &opacity);
    bool	getVisibility();
    void    setVisibility(bool &visiblity);
    size_t  getNumPoints();
    void    update();
    PointBufferPtr getPointBuffer();
    PointBufferBridgePtr getPointBufferBridge();
    vtkSmartPointer<vtkActor> getActor();

protected:
    QTreeWidgetItem*        m_parent;
    QTreeWidgetItem*        m_numItem;
    QTreeWidgetItem*        m_normalItem;
    QTreeWidgetItem*        m_colorItem;
    QTreeWidgetItem*        m_specItem;

    PointBufferBridgePtr    m_pointBridge;
    QColor                  m_color;
    int						m_pointSize;
    float					m_opacity;
    bool					m_visible;
};

} /* namespace lvr2 */

#endif /* LVRPOINTCLOUDITEM_HPP_ */
