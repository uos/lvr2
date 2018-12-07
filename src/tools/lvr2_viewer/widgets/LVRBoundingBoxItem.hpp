#ifndef LVR2_TOOLS_VIEWER_LVRBOUNDINGBOXITEM_HPP
#define LVR2_TOOLS_VIEWER_LVRBOUNDINGBOXITEM_HPP

#include <QTreeWidgetItem>
#include <QString>

#include "../vtkBridge/LVRBoundingBoxBridge.hpp"

namespace lvr2
{

class LVRBoundingBoxItem : public QTreeWidgetItem
{
    public:
        LVRBoundingBoxItem(
                BoundingBoxBridgePtr bb,
                QString name = "",
                QTreeWidgetItem *parent = NULL);

        void setVisibility(bool visible);

        BoundingBoxBridgePtr getBoundingBoxBridge() { return m_bb; }

    private:
        BoundingBoxBridgePtr m_bb;
        QString              m_name;
};

} // namespace lvr2

#endif
