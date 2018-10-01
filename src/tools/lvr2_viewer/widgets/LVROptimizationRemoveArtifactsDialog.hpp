#ifndef OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_
#define OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "../vtkBridge/LVRModelBridge.hpp"

#include "ui_LVROptimizationRemoveArtifactsDialogUI.h"
#include "LVRMeshItem.hpp"
#include "LVRModelItem.hpp"

using Ui::RemoveArtifactsDialog;

namespace lvr2
{

class LVRRemoveArtifactsDialog : public QObject
{
    Q_OBJECT

public:
    LVRRemoveArtifactsDialog(LVRMeshItem* mesh, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRRemoveArtifactsDialog();

public Q_SLOTS:
    void removeArtifacts();

private:
    void connectSignalsAndSlots();

    RemoveArtifactsDialog*                  m_dialog;
    LVRMeshItem*                            m_mesh;
    LVRModelItem*                           m_optimizedModel;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;

};

} // namespace lvr2

#endif /* OPTIMIZATIONREMOVEARTIFACTSDIALOG_H_ */
