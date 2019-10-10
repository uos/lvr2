#ifndef LVR2_TOOLS_VIEWER_WIDGETS_LVRCVIMAGEITEM_HPP
#define LVR2_TOOLS_VIEWER_WIDGETS_LVRCVIMAGEITEM_HPP

#include <QString>
#include <QTreeWidgetItem>
#include <QAbstractItemModel>
#include <QObject>
#include <QImage>
#include <QLabel>
#include <QGraphicsScene>
#include <QGraphicsView>
#include "LVRGraphicsView.hpp"

#include <vtkMatrix4x4.h>

#include "lvr2/io/ScanDataManager.hpp"


namespace lvr2
{

class LVRCvImageItem : public QTreeWidgetItem
{
    // Q_OBJECT
public:

    LVRCvImageItem(std::shared_ptr<ScanDataManager> sdm,
                    vtkSmartPointer<vtkRenderer> renderer,
                    QString name = "",
                    QTreeWidgetItem *parent = NULL);

    ~LVRCvImageItem();

    QString getName() { return m_name; }

    void setVisibility(bool visible);

    void openWindow();

    void closeWindow();

    void graphicsViewClosed();

// public Q_SLOTS:
//     void graphicsViewClosed();

private:

    QImage* convertCvImageToQt(cv::Mat& cv_img);

    void reload(vtkSmartPointer<vtkRenderer> renderer);

    QString                                 m_name;
    std::shared_ptr<ScanDataManager>        m_sdm;
    size_t                                  m_idx;
    vtkSmartPointer<vtkRenderer>            m_renderer;
    QLabel*                                 m_label;
    QGraphicsScene*                         m_graphics_scene;
    LVRGraphicsView*                        m_graphics_view;

};

} // namespace lvr2

#endif
