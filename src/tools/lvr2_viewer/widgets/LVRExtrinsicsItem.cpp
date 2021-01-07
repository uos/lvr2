#include "LVRExtrinsicsItem.hpp"

namespace lvr2
{

LVRExtrinsicsItem::LVRExtrinsicsItem(Extrinsicsd extrinsics) :
    QTreeWidgetItem(LVRExtrinsicsItemType)
{
    setText(0, "Matrix");
    setText(1, "Click to expand");
    m_extrinsics = extrinsics;
}

Extrinsicsd LVRExtrinsicsItem::extrinsics()
{
    return m_extrinsics;
}

} //namespace lvr2