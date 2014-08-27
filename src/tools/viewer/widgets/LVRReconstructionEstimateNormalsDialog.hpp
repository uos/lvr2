#ifndef RECONSTRUCTIONESTIMATENORMALSDIALOG_H_
#define RECONSTRUCTIONESTIMATENORMALSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "io/AsciiIO.hpp"
#include "io/Timestamp.hpp"
#include "io/Progress.hpp"
#include "io/DataStruct.hpp"
#include "io/ModelFactory.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/Normal.hpp"
#include "reconstruction/SearchTree.hpp"
#include "reconstruction/SearchTreeStann.hpp"

// SearchTreePCL
#ifdef _USE_PCL_
    #include "reconstruction/SearchTreeFlann.hpp"
#endif

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

public Q_SLOTS:
    void estimateNormals();

private:
    void connectSignalsAndSlots();

    EstimateNormalsDialog*                  m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_optimizedPointCloud;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr

#endif /* RECONSTRUCTIONESTIMATENORMALSDIALOG_H_ */
