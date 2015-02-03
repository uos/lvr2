#ifndef RENAMEDIALOG_H_
#define RENAMEDIALOG_H_

#include <vtkRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkRendererCollection.h>
#include <vtkSmartPointer.h>

#include "LVRRenameDialogUI.h"
#include "LVRModelItem.hpp"

using Ui::RenameDialog;

namespace lvr
{

class LVRRenameDialog : public QObject
{
    Q_OBJECT

public:
    LVRRenameDialog(LVRModelItem* item, QTreeWidget* treeWidget);
    virtual ~LVRRenameDialog();

public Q_SLOTS:
    void renameItem();

private:
    void connectSignalsAndSlots();

    RenameDialog*                           m_dialog;
    LVRModelItem*                           m_item;
    QTreeWidget*                            m_treeWidget;
};

} // namespace lvr

#endif /* RENAMEDIALOG_H_ */
