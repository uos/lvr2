#include "LVRLabelInstanceTreeItem.hpp"
#include <vtkPolyDataMapper.h>
#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRLabelInstanceTreeItem::LVRLabelInstanceTreeItem(std::string className,int id, int labeledPointCount, bool visible, bool editable, QColor color) :
    QTreeWidgetItem(LVRLabelInstanceItemType),
    m_id(id)
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
    }
        
    setData(LABEL_ID_COLUMN, LABEL_COLOR_GROUP, color);
    setData(LABEL_ID_COLUMN, LABEL_ID_GROUP, id);
    m_labelInstancePtr = LabelInstancePtr(new LabelInstance);
    m_labelInstancePtr->instanceName = className;
    int r, g, b;
    color.getRgb(&r, &g, &b);
    m_labelInstancePtr->color[0] = r;
    m_labelInstancePtr->color[1] = g;
    m_labelInstancePtr->color[2] = b;



}

LVRLabelInstanceTreeItem::LVRLabelInstanceTreeItem(const LVRLabelInstanceTreeItem& item)
{
    m_labelInstancePtr   = item.m_labelInstancePtr;
    m_id = item.m_id;
}

QColor LVRLabelInstanceTreeItem::getColor()
{
    return data(LABEL_ID_COLUMN, LABEL_COLOR_GROUP).value<QColor>();
}

std::string LVRLabelInstanceTreeItem::getName()
{
    return text(LABEL_NAME_COLUMN).toStdString(); 
}

bool LVRLabelInstanceTreeItem::isVisible()
{
    return checkState(LABEL_VISIBLE_COLUMN) == Qt::Checked;
}
bool LVRLabelInstanceTreeItem::isEditable()
{
    return checkState(LABEL_EDITABLE_COLUMN) == Qt::Checked;
}
int LVRLabelInstanceTreeItem::getNumberOfLabeledPoints()
{
    return text(LABELED_POINT_COLUMN).toInt();
}
int LVRLabelInstanceTreeItem::getId()
{
    return data(LABEL_ID_COLUMN, LABEL_ID_GROUP).toInt();
}

LabelInstancePtr LVRLabelInstanceTreeItem::getInstancePtr()
{
    return m_labelInstancePtr;
}
LVRLabelInstanceTreeItem::~LVRLabelInstanceTreeItem()
{
}

}
