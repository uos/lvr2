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

    /**
     *  @brief          Constructor. Constructs an Extrinsicsitem from the given extrinsics
     */
    LVRExtrinsicsItem(Extrinsicsd extrinsics);

    /**
     *  @brief          Destructor.
     */
    virtual ~LVRExtrinsicsItem() = default;
    
    /**
     *  @brief          returns the stored extrinsics
     */
    Extrinsicsd extrinsics();

protected:
    Extrinsicsd m_extrinsics;

};

} /* namespace lvr2 */

#endif