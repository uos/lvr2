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

#ifndef ANIMATIONDIALOG_H_
#define ANIMATIONDIALOG_H_

#include <vtkRenderer.h>
#include <vtkSmartPointer.h>
#include <vtkCamera.h>
#include <vtkCameraRepresentation.h>
#include <vtkCameraInterpolator.h>
#include <vtkCommand.h>
#include <vtkRendererCollection.h>
#include <vtkRenderWindow.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkWindowToImageFilter.h>

#include <QtWidgets>
#include "ui_LVRAnimationDialogUI.h"
#include "LVRRecordedFrameItem.hpp"

using Ui::AnimationDialog;

namespace lvr2
{

class LVRAnimationDialog : public QObject
{
    Q_OBJECT

public:
    LVRAnimationDialog(vtkSmartPointer<vtkRenderWindowInteractor> renderWindowInteractor, vtkSmartPointer<vtkCameraRepresentation> pathCamera, QTreeWidget* treeWidget);
    virtual ~LVRAnimationDialog();

public Q_SLOTS:
    void addFrame();
    void removeFrame();
    void clearFrames();
    void changeInterpolation(const QString& text);
    void savePath();
    void loadPath();
    void saveVideo();
    void play();
    void goToCamPosition(QListWidgetItem *item);

private:
    void connectSignalsAndSlots();


    AnimationDialog*                            m_dialog;
    QListWidget*                                m_timeline;
    vtkSmartPointer<vtkRenderWindowInteractor>  m_renderWindowInteractor;
    vtkSmartPointer<vtkCameraRepresentation>    m_pathCamera;
    vtkSmartPointer<vtkCamera>                  m_mainCamera;
    QTreeWidget*                                m_treeWidget;
    unsigned int                                m_frameCounter;
};

} // namespace lvr2

#endif /* ANIMATIONDIALOG_H_ */
