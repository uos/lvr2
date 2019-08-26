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

#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "../vtkBridge/LVRModelBridge.hpp"
#include "../util/qttf.hpp"

#include "ui_LVRReconstructionMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

#include <QProgressDialog>
#include <QTreeWidget>
#include <QTreeWidgetItem>


using Ui::ReconstructViaMarchingCubesDialog;

namespace lvr2
{

class LVRReconstructViaMarchingCubesDialog : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaMarchingCubesDialog(
        string decomposition,
        LVRPointCloudItem* pc,
        QTreeWidgetItem* parent,
        QTreeWidget* treeWidget,
        vtkRenderWindow* renderer
    );

    // LVRReconstructViaMarchingCubesDialog(
    //     string decomposition,
    //     QList<LVRPointCloudItem*> pcs,
    //     QList<QTreeWidgetItem*> parent,
    //     QTreeWidget* treeWidget,
    //     vtkRenderWindow* renderer
    // );

    virtual ~LVRReconstructViaMarchingCubesDialog();

    static void updateProgressbar(int p);
    static void updateProgressbarTitle(string t);


    void setProgressValue(int v);
    void setProgressTitle(string);

public Q_SLOTS:
    void generateMesh();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);

Q_SIGNALS:
    void progressValueChanged(int);
    void progressTitleChanged(const QString&);


private:
    void connectSignalsAndSlots();

    string                                          m_decomposition;
    ReconstructViaMarchingCubesDialog*              m_dialog;
    LVRPointCloudItem*                              m_pc;
    QTreeWidgetItem*                                m_parent;
    QTreeWidget*                                    m_treeWidget;
    LVRModelItem*                                   m_generatedModel;
    vtkRenderWindow*                                m_renderWindow;
    QProgressDialog*                                m_progressDialog;
    static LVRReconstructViaMarchingCubesDialog*    m_master;


};

} // namespace lvr2

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
