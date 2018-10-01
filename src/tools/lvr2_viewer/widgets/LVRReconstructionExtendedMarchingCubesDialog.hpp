#ifndef RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_
#define RECONSTRUCTIONEXTENDEDMARCHINGCUBESDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "../vtkBridge/LVRModelBridge.hpp"

#include "ui_LVRReconstructionExtendedMarchingCubesDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::ReconstructViaExtendedMarchingCubesDialog;

namespace lvr2
{

class LVRReconstructViaExtendedMarchingCubesDialog  : public QObject
{
    Q_OBJECT

public:
    LVRReconstructViaExtendedMarchingCubesDialog(string decomposition, LVRPointCloudItem* pc, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRReconstructViaExtendedMarchingCubesDialog();

public Q_SLOTS:
    void generateMesh();
    void toggleRANSACcheckBox(const QString &text);
    void switchGridSizeDetermination(int index);

private:
    void connectSignalsAndSlots();

    string                                          m_decomposition;
    ReconstructViaExtendedMarchingCubesDialog*      m_dialog;
    LVRPointCloudItem*                              m_pc;
    LVRModelItem*                                   m_parent;
    QTreeWidget*                                    m_treeWidget;
    LVRModelItem*                                   m_generatedModel;
    vtkRenderWindow*                                m_renderWindow;

};

} // namespace lvr2

#endif /* RECONSTRUCTIONMARCHINGCUBESDIALOG_H_ */
