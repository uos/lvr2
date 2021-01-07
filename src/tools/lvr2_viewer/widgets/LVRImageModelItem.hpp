#ifndef LVRIMAGEMODELITEM_HPP
#define LVRIMAGEMODELITEM_HPP

#include <QTreeWidgetItem>
#include "lvr2/types/ScanTypes.hpp"
#include <QtWidgets/qtreewidget.h>
#include "LVRExtrinsicsItem.hpp"
#include "LVRItemTypes.hpp"


namespace lvr2
{

class LVRImageModelItem : public QTreeWidgetItem
{
public:
    LVRImageModelItem(ScanImage& img);
    virtual ~LVRImageModelItem() = default;

protected:
    LVRExtrinsicsItem* m_extrinsics;
    QTreeWidgetItem* m_timestamp;

};

} /* namespace lvr2 */

#endif