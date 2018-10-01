#ifndef OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_
#define OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "../vtkBridge/LVRModelBridge.hpp"

#include "ui_LVROptimizationPlanarOptimizationDialogUI.h"
#include "LVRMeshItem.hpp"
#include "LVRModelItem.hpp"

using Ui::PlanarOptimizationDialog;

namespace lvr2
{

class LVRPlanarOptimizationDialog : public QObject
{
    Q_OBJECT

public:
    LVRPlanarOptimizationDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRPlanarOptimizationDialog();

public Q_SLOTS:
    void optimizeMesh();
    void toggleSmallRegionRemoval(int state);
    void toggleRetesselation(int state);

private:
    void connectSignalsAndSlots();

    PlanarOptimizationDialog*               m_dialog;
    LVRMeshItem*                            m_mesh;
    LVRModelItem*                           m_optimizedModel;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr2

#endif /* OPTIMIZATIONPLANAROPTIMIZATIONDIALOG_H_ */
