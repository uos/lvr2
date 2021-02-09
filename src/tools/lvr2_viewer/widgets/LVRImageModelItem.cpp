#include "LVRImageModelItem.hpp"

namespace lvr2
{

LVRImageModelItem::LVRImageModelItem(ScanImage& img) : 
    QTreeWidgetItem(LVRImageModelItemType)
{
    //set text off item to meta
    setText(0, "Meta");

    //create Extrinsics and timestamp item
    m_extrinsics = std::make_shared<LVRExtrinsicsItem>(img.extrinsics);
    m_timestamp = std::make_shared<QTreeWidgetItem>(this);

    addChild(m_extrinsics.get());
    addChild(m_timestamp.get());

    //set text for children
    m_timestamp->setText(0, "timestamp");
    m_timestamp->setText(1, QString::number(img.timestamp));
}

} //namespace lvr2