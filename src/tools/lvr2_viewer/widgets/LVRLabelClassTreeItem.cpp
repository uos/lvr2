#include "LVRLabelClassTreeItem.hpp"
#include <vtkPolyDataMapper.h>
#include "LVRItemTypes.hpp"
#include "LVRLabelInstanceTreeItem.hpp"

namespace lvr2
{

LVRLabelClassTreeItem::LVRLabelClassTreeItem(std::string className, int labeledPointCount, bool visible, bool editable, QColor color) :
    QTreeWidgetItem(LVRLabelClassItemType)
{
    // Setup item properties
    setText(LABEL_NAME_COLUMN, QString::fromStdString(className));
    setText(LABELED_POINT_COLUMN, QString::number(labeledPointCount));
    if (visible)
    {
        setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    } else
    {
        setCheckState(LABEL_VISIBLE_COLUMN, Qt::Unchecked);
    }
    if (editable)
    {
        setCheckState(LABEL_EDITABLE_COLUMN, Qt::Checked);
    } else
    {
        setCheckState(LABEL_EDITABLE_COLUMN, Qt::Unchecked);
    }

    setData(LABEL_ID_COLUMN, LABEL_COLOR_GROUP, color);
    m_labelClassPtr = LabelClassPtr(new LabelClass);
    m_labelClassPtr->className = className;


}

LVRLabelClassTreeItem::LVRLabelClassTreeItem(LabelClassPtr classPtr) :
    QTreeWidgetItem(LVRLabelClassItemType)
{
    // Setup item properties
    setText(LABEL_NAME_COLUMN, QString::fromStdString(classPtr->className));
    setText(LABELED_POINT_COLUMN, QString::number(0));
    setCheckState(LABEL_VISIBLE_COLUMN, Qt::Checked);
    setCheckState(LABEL_EDITABLE_COLUMN, Qt::Checked);
    m_labelClassPtr = classPtr;

}

void LVRLabelClassTreeItem::addChild(QTreeWidgetItem *child)
{
    if(child->type() == LVRLabelInstanceItemType)
    {
        LVRLabelInstanceTreeItem *instanceItem = static_cast<LVRLabelInstanceTreeItem *>(child);
        if(this->childCount() == 0)
        {
            setData(LABEL_ID_COLUMN, LABEL_COLOR_GROUP, instanceItem->getColor());
        }
        m_labelClassPtr->instances.push_back(instanceItem->getInstancePtr());
        QTreeWidgetItem::addChild(child);
    }
}
void LVRLabelClassTreeItem::addChildnoChanges(QTreeWidgetItem *child)
{
    if(child->type() == LVRLabelInstanceItemType)
    {
        LVRLabelInstanceTreeItem *instanceItem = static_cast<LVRLabelInstanceTreeItem *>(child);
        if(this->childCount() == 0)
        {
            setData(LABEL_ID_COLUMN, LABEL_COLOR_GROUP, instanceItem->getColor());
        }
        QTreeWidgetItem::addChild(child);
    }
}
LVRLabelClassTreeItem::LVRLabelClassTreeItem(const LVRLabelClassTreeItem& item)
{
    m_labelClassPtr   = item.m_labelClassPtr;
}

void LVRLabelClassTreeItem::setColor(QColor color)
{
    setData(LABEL_ID_COLUMN, LABEL_COLOR_GROUP, color);
}
QColor LVRLabelClassTreeItem::getDefaultColor()
{
    return data(LABEL_ID_COLUMN, 1).value<QColor>();
}
bool LVRLabelClassTreeItem::isVisible()
{
    return checkState(LABEL_VISIBLE_COLUMN) == Qt::Checked;
}
bool LVRLabelClassTreeItem::isEditable()
{
    return checkState(LABEL_EDITABLE_COLUMN) == Qt::Checked;
}
int LVRLabelClassTreeItem::getNumberOfLabeledPoints()
{
    return text(LABELED_POINT_COLUMN).toInt();
}

std::string LVRLabelClassTreeItem::getName()
{
    return text(LABEL_NAME_COLUMN).toStdString();
}

LabelClassPtr LVRLabelClassTreeItem::getLabelClassPtr()
{
    return m_labelClassPtr;
}
LVRLabelClassTreeItem::~LVRLabelClassTreeItem()
{
}

}
