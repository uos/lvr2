#include "LVRBoundingBoxItem.hpp"

#include "LVRItemTypes.hpp"

namespace lvr2
{

LVRBoundingBoxItem::LVRBoundingBoxItem(
        BoundingBoxBridgePtr bb,
        QString name,
        QTreeWidgetItem *parent) : QTreeWidgetItem(parent, LVRBoundingBoxItemType),
                                   m_bb(bb),
                                   m_name(name)
{
    setText(0, m_name);
    setCheckState(0, Qt::Unchecked);

    auto tmp = m_bb->getBoundingBox();

    QTreeWidgetItem *min_x_item = new QTreeWidgetItem(this);
    min_x_item->setText(0, "Min X:");
    min_x_item->setText(1, std::to_string(tmp.getMin().x).c_str());

    QTreeWidgetItem *min_y_item = new QTreeWidgetItem(this);
    min_y_item->setText(0, "Min Y:");
    min_y_item->setText(1, std::to_string(tmp.getMin().y).c_str());

    QTreeWidgetItem *min_z_item = new QTreeWidgetItem(this);
    min_z_item->setText(0, "Min Z:");
    min_z_item->setText(1, std::to_string(tmp.getMin().z).c_str());

    QTreeWidgetItem *max_x_item = new QTreeWidgetItem(this);
    max_x_item->setText(0, "Max X:");
    max_x_item->setText(1, std::to_string(tmp.getMax().x).c_str());

    QTreeWidgetItem *max_y_item = new QTreeWidgetItem(this);
    max_y_item->setText(0, "Max Y:");
    max_y_item->setText(1, std::to_string(tmp.getMax().y).c_str());

    QTreeWidgetItem *max_z_item = new QTreeWidgetItem(this);
    max_z_item->setText(0, "Max Z:");
    max_z_item->setText(1, std::to_string(tmp.getMax().z).c_str());
}

void LVRBoundingBoxItem::setVisibility(bool visible)
{
    bool parent_enabled = true;
    if (parent() && !parent()->checkState(0))
    {
        parent_enabled = false;
    }

    if (m_bb)
    {
        m_bb->setVisibility(parent_enabled && checkState(0) && visible);
    }
}

} // namesapce lvr2
