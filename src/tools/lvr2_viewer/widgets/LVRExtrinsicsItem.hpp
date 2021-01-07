#ifndef LVREXTRINSICSITEM_HPP
#define LVREXTRINSICSITEM_HPP

#include <QTreeWidgetItem>
#include "lvr2/types/ScanTypes.hpp"
#include <QtWidgets/qtreewidget.h>
#include "LVRItemTypes.hpp"


namespace lvr2
{

class LVRExtrinsicsItem : public QTreeWidgetItem
{
public:
    LVRExtrinsicsItem(Extrinsicsd extrinsics);
    virtual ~LVRExtrinsicsItem() = default;
    Extrinsicsd extrinsics();

protected:
    Extrinsicsd m_extrinsics;

};

} /* namespace lvr2 */

#endif