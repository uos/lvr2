#include "LVRModelTreeWidget.hpp"
#include <iostream>
#include "LVRItemTypes.hpp"

LVRModelTreeWidget::LVRModelTreeWidget(QWidget *parent)
{
}

void LVRModelTreeWidget::addTopLevelItem(QTreeWidgetItem *item)
{
    if(item->type() == lvr2::LVRModelItemType)
    {
        lvr2::LVRModelItem *modelItem = static_cast<lvr2::LVRModelItem *>(item);
        addModelItem(modelItem);
    }else if(item->type() == lvr2::LVRLabeledScanProjectEditMarkItemType)
    {
        lvr2::LVRLabeledScanProjectEditMarkItem *modelItem = static_cast<lvr2::LVRLabeledScanProjectEditMarkItem *>(item);
        addLabelScanProjectEditMarkItem(modelItem);
    }
    else
    {
        QTreeWidget::addTopLevelItem(item);
    }
}

void LVRModelTreeWidget::addModelItem(lvr2::LVRModelItem *item)
{
    QTreeWidget::addTopLevelItem(item);
}

void LVRModelTreeWidget::addLabelScanProjectEditMarkItem(lvr2::LVRLabeledScanProjectEditMarkItem *item)
{
}
