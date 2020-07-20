#include "LVRLabelTreeWidget.hpp"
#include <iostream>
#include "LVRItemTypes.hpp"

LVRLabelTreeWidget::LVRLabelTreeWidget(QWidget *parent)
{
    m_root = lvr2::LabelRootPtr(new lvr2::LabelRoot()); 
}

void LVRLabelTreeWidget::addTopLevelItem(QTreeWidgetItem *item)
{
    if(item->type() == lvr2::LVRLabelClassItemType)
    {
        QTreeWidget::addTopLevelItem(item);
    }
}

lvr2::LabelRootPtr LVRLabelTreeWidget::getLabelRoot()
{
    return m_root;
}
