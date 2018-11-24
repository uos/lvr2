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
    LVREstimateNormalsDialog(LVRPointCloudItem* pc_item, QTreeWidgetItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVREstimateNormalsDialog();

private Q_SLOTS:
    void estimateNormals();
    void toggleNormalInterpolation(int state);

private:
    void connectSignalsAndSlots();

    EstimateNormalsDialog*                  m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_pointCloudWithNormals;
    QTreeWidgetItem*                        m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr2

#endif /* RECONSTRUCTIONESTIMATENORMALSDIALOG_H_ */
