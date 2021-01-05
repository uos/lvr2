#ifndef LVRCAMERAMODELITEM_HPP
#define LVRCAMERAMODELITEM_HPP

#include <QTreeWidgetItem>
#include "lvr2/types/ScanTypes.hpp"
#include <QtWidgets/qtreewidget.h>

namespace lvr2
{

class LVRCameraModelItem : public QTreeWidgetItem
{
public:
    LVRCameraModelItem(ScanCamera& cam);
    virtual ~LVRCameraModelItem() = default;
    void setModel(PinholeModeld& model);

protected:
    PinholeModeld m_model;
    QTreeWidgetItem* m_fxItem;
    QTreeWidgetItem* m_cxItem;
    QTreeWidgetItem* m_fyItem;
    QTreeWidgetItem* m_cyItem;
    QTreeWidgetItem* m_distortionItem;

};

} /* namespace lvr2 */

#endif