#ifndef FILTERINGMLSPROJECTIONDIALOG_H_
#define FILTERINGMLSPROJECTIONDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "io/ModelFactory.hpp"
#include "reconstruction/PCLFiltering.hpp"

#include "LVRFilteringMLSProjectionDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::MLSProjectionDialog;

namespace lvr
{

class LVRMLSProjectionDialog : public QObject
{
    Q_OBJECT

public:
    LVRMLSProjectionDialog(LVRPointCloudItem* pc_item, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRMLSProjectionDialog();

public Q_SLOTS:
    void applyMLSProjection();

private:
    void connectSignalsAndSlots();

    MLSProjectionDialog*                    m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_optimizedPointCloud;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr

#endif /* FILTERINGMLSPROJECTIONDIALOG_H_ */
