#ifndef RECONSTRUCTIONESTIMATENORMALSDIALOG_H_
#define RECONSTRUCTIONESTIMATENORMALSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include <lvr/io/AsciiIO.hpp>
#include <lvr/io/Timestamp.hpp>
#include <lvr/io/Progress.hpp>
#include <lvr/io/DataStruct.hpp>
#include <lvr/io/ModelFactory.hpp>
#include <lvr/geometry/Matrix4.hpp>
#include <lvr/geometry/Normal.hpp>
#include <lvr/reconstruction/SearchTree.hpp>

#include "LVRReconstructionEstimateNormalsDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::EstimateNormalsDialog;

namespace lvr
{

class LVREstimateNormalsDialog : public QObject
{
    Q_OBJECT

public:
    LVREstimateNormalsDialog(LVRPointCloudItem* pc_item, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVREstimateNormalsDialog();

private Q_SLOTS:
    void estimateNormals();
    void toggleNormalInterpolation(int state);

private:
    void connectSignalsAndSlots();

    EstimateNormalsDialog*                  m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_pointCloudWithNormals;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr

#endif /* RECONSTRUCTIONESTIMATENORMALSDIALOG_H_ */
