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
    /**
     *  @brief          Constructor. Constructs an ImageModelItem from the ScanImage.
     */
    LVRImageModelItem(CameraImage& img);

    /**
     *  @brief          Destructor.
     */
    virtual ~LVRImageModelItem() = default;

protected:
    std::shared_ptr<LVRExtrinsicsItem> m_extrinsics;
    std::shared_ptr<QTreeWidgetItem> m_timestamp;

};

} /* namespace lvr2 */

#endif