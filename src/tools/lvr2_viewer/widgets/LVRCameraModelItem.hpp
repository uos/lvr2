#ifndef LVRCAMERAMODELITEM_HPP
#define LVRCAMERAMODELITEM_HPP

#include <QTreeWidgetItem>
#include <QtWidgets/qtreewidget.h>

#include "lvr2/types/ScanTypes.hpp"
#include "lvr2/types/CameraModels.hpp"

#include "LVRItemTypes.hpp"

namespace lvr2
{

class LVRCameraModelItem : public QTreeWidgetItem
{
public:
    /**
     *  @brief          Constructor. Constructs an item with the cameramodel for the given ScanCamera
     */
    LVRCameraModelItem(Camera& cam);
    
    /**
     *  @brief          Destructor.
     */
    virtual ~LVRCameraModelItem() = default;

    /**
     *  @brief          Set the internal model to the given model
     */
    void setModel(PinholeModel& model);

protected:
    PinholeModel m_model;
    std::shared_ptr<QTreeWidgetItem> m_fxItem;
    std::shared_ptr<QTreeWidgetItem> m_cxItem;
    std::shared_ptr<QTreeWidgetItem> m_fyItem;
    std::shared_ptr<QTreeWidgetItem> m_cyItem;
    std::shared_ptr<QTreeWidgetItem> m_distortionItem;
    std::shared_ptr<QTreeWidgetItem> m_distortionCoef[4];

};

} /* namespace lvr2 */

#endif