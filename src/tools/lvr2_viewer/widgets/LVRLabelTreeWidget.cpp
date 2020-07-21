#include "LVRLabelTreeWidget.hpp"
#include <iostream>
#include "LVRItemTypes.hpp"
#include "LVRLabelClassTreeItem.hpp"

LVRLabelTreeWidget::LVRLabelTreeWidget(QWidget *parent)
{
    m_root = lvr2::LabelRootPtr(new lvr2::LabelRoot()); 
}

void LVRLabelTreeWidget::addTopLevelItem(QTreeWidgetItem *item)
{
    if(item->type() == lvr2::LVRLabelClassItemType)
    {
        lvr2::LVRLabelClassTreeItem *classItem = static_cast<lvr2::LVRLabelClassTreeItem *>(item);

        m_root->labelClasses.push_back(classItem->getLabelClassPtr());
        QTreeWidget::addTopLevelItem(item);
    }
}

lvr2::LabelRootPtr LVRLabelTreeWidget::getLabelRoot()
{
    return m_root;
}
