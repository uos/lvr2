#ifndef FILTERINGREMOVEOUTLIERSDIALOG_H_
#define FILTERINGREMOVEOUTLIERSDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "io/ModelFactory.hpp"
#include "reconstruction/PCLFiltering.hpp"

#include "LVRFilteringRemoveOutliersDialogUI.h"
#include "LVRPointCloudItem.hpp"
#include "LVRModelItem.hpp"

using Ui::RemoveOutliersDialog;

namespace lvr
{

class LVRRemoveOutliersDialog : public QObject
{
    Q_OBJECT

public:
    LVRRemoveOutliersDialog(LVRPointCloudItem* pc_item, LVRModelItem* parent, QTreeWidget* treeWidget, vtkRenderWindow* renderer);
    virtual ~LVRRemoveOutliersDialog();

public Q_SLOTS:
    void removeOutliers();

private:
    void connectSignalsAndSlots();

    RemoveOutliersDialog*                   m_dialog;
    LVRPointCloudItem*                      m_pc;
    LVRModelItem*                           m_optimizedPointCloud;
    LVRModelItem*                           m_parent;
    QTreeWidget*                            m_treeWidget;
    vtkRenderWindow*                        m_renderWindow;
};

} // namespace lvr

#endif /* FILTERINGREMOVEOUTLIERSDIALOG_H_ */
