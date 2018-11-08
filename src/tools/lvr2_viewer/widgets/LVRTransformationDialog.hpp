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
 * TransformationDialog.h
 *
 *  @date 01.08.2011
 *  @author Thomas Wiemann
 */

#ifndef TRANSFORMATIONDIALOG_H_
#define TRANSFORMATIONDIALOG_H_

#include <vtkRenderWindow.h>

#include "ui_LVRTransformationDialogUI.h"
#include "LVRModelItem.hpp"

using Ui::TransformationDialogUI;

namespace lvr2
{

class LVRTransformationDialog : public QObject
{
    Q_OBJECT

public:
    LVRTransformationDialog(LVRModelItem* parent, vtkRenderWindow* renderer);
    virtual ~LVRTransformationDialog();

public Q_SLOTS:

    void rotationXSlided(int value);
    void rotationYSlided(int value);
    void rotationZSlided(int value);

    void rotationXEntered(double value);
    void rotationYEntered(double value);
    void rotationZEntered(double value);

    void translationXEntered(double value);
    void translationYEntered(double value);
    void translationZEntered(double value);

    void stepChanged(double value);

    void reset();
    void restoreAndClose();
    void save();


private:

    void connectSignalsAndSlots();
    void transformLocal();
    void transformGlobal();

    Pose                        m_pose;
    Pose                        m_pose_original;
    TransformationDialogUI*     m_dialogUI;
    LVRModelItem*               m_parent;
    vtkRenderWindow*            m_renderWindow;

};

} // namespace lvr2

#endif /* TRANSFORMATIONDIALOG_H_ */
