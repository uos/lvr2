#include "LVRImageModelItem.hpp"

namespace lvr2
{

LVRImageModelItem::LVRImageModelItem(ScanImage& img) : 
    QTreeWidgetItem(LVRImageModelItemType)
{
    setText(0, "Meta");
    m_extrinsics = new LVRExtrinsicsItem(img.extrinsics);
    m_timestamp = new QTreeWidgetItem(this);

    addChild(m_extrinsics);
    addChild(m_timestamp);

    m_timestamp->setText(0, "timestamp");
    m_timestamp->setText(1, QString::number(img.timestamp));
}

} //namespace lvr2