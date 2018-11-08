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
 * LVRMeshItem.h
 *
 *  @date Feb 11, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRMESHITEM_H_
#define LVRMESHITEM_H_

#include <QTreeWidgetItem>
#include <QColor>

#include "../vtkBridge/LVRMeshBufferBridge.hpp"

namespace lvr2
{

class LVRMeshItem : public QTreeWidgetItem
{
public:
    LVRMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent = 0);
    virtual ~LVRMeshItem();
    QColor	getColor();
    void    setColor(QColor &c);
    void    setSelectionColor(QColor &c);
    void    resetColor();
    float	getOpacity();
    void    setOpacity(float &opacity);
    bool	getVisibility();
    void    setVisibility(bool &visiblity);
    int     getShading();
    void    setShading(int &shader);
    vtkSmartPointer<vtkActor>    getWireframeActor();
    MeshBufferPtr   getMeshBuffer();
    vtkSmartPointer<vtkActor> getActor();

protected:
    virtual void addSubItems();
    MeshBufferBridgePtr     m_meshBridge;

private:
    QColor                  m_color;

    float					m_opacity;
    bool					m_visible;
    int                     m_shader;

protected:
    QTreeWidgetItem* 		m_parent;
};

} /* namespace lvr2 */

#endif /* LVRMESHITEM_H_ */
