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

#ifndef RECONSTRUCTIONESTIMATENORMALSDIALOG_H_
#define RECONSTRUCTIONESTIMATENORMALSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "ui_LVRReconstructionEstimateNormalsDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"
#include "LVRScanDataItem.hpp"

using Ui::EstimateNormalsDialog;

namespace lvr2
{

class LVREstimateNormalsDialog : public QObject
{
    Q_OBJECT

public:
    // old
    LVREstimateNormalsDialog(LVRPointCloudItem* pc_item, QTreeWidgetItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);

    // new
    LVREstimateNormalsDialog(
        QList<LVRPointCloudItem*> pc_items,
        QList<QTreeWidgetItem*> parents,
        QTreeWidget* treeWidget,
        vtkRenderWindow* renderer
    );

    virtual ~LVREstimateNormalsDialog();

private Q_SLOTS:
    void estimateNormals();
    void toggleAutoNormalEstimation(int state);
    void toggleAutoNormalInterpolation(int state);
    void toggleSpecialOptions(QString current_text);
    

private:
    void connectSignalsAndSlots();

    EstimateNormalsDialog*                  m_dialog;

    // new
    QList<LVRPointCloudItem*>               m_pcs;
    // QList<LVRPointCloudItem*>               m_pcs_with_normals;
    QList<QTreeWidgetItem*>                 m_parents;

    // old
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_pointCloudWithNormals;
    QTreeWidgetItem*                        m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr2

#endif /* RECONSTRUCTIONESTIMATENORMALSDIALOG_H_ */
