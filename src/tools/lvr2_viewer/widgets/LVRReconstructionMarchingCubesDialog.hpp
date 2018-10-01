#ifndef RECONSTRUCTIONMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONMARCHINGCUBESDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "../vtkBridge/LVRModelBridge.hpp"

#include "ui_LVRReconstructionMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

#include <QProgressDialog>

using Ui::ReconstructViaMarchingCubesDialog;

namespace lvr2
{

class LVRReconstructViaMarchingCubesDialog : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaMarchingCubesDialog(string decomposition, LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
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
    LVRModelItem*                                   m_parent;
    QTreeWidget*                                    m_treeWidget;
    LVRModelItem*                                   m_generatedModel;
    vtkRenderWindow*                                m_renderWindow;
    QProgressDialog*                                m_progressDialog;
    static LVRReconstructViaMarchingCubesDialog*    m_master;


};

} // namespace lvr2

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
