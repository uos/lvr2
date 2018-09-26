#ifndef FILTERINGMLSPROJECTIONDIALOG_H_
#define FILTERINGMLSPROJECTIONDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include <lvr2/io/ModelFactory.hpp>
// @TODO should this be ported? In .cpp the use of this include is commented out...
//#include <lvr2/reconstruction/PCLFiltering.hpp>

#include "ui_LVRFilteringMLSProjectionDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::MLSProjectionDialog;

namespace lvr2
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

} // namespace lvr2

#endif /* FILTERINGMLSPROJECTIONDIALOG_H_ */
