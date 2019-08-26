
#include "LVRItemTypes.hpp"
#include "LVRScanDataItem.hpp"
#include "LVRCamDataItem.hpp"

namespace lvr2
{

LVRCvImageItem::LVRCvImageItem(std::shared_ptr<ScanDataManager> sdm, vtkSmartPointer<vtkRenderer> renderer, QString name, QTreeWidgetItem *parent) : QTreeWidgetItem(parent, LVRCvImageItemType)
{
    m_name   = name;
    m_sdm    = sdm;
    m_renderer = renderer;
    m_label = nullptr;
    m_graphics_scene = nullptr,
    m_graphics_view = nullptr;

    reload(renderer);
    setText(0, m_name);
}

void LVRCvImageItem::reload(vtkSmartPointer<vtkRenderer> renderer)
{
    
}

void LVRCvImageItem::setVisibility(bool visible)
{

    // if(checkState(0) && visible)
    // {
    //     m_renderer->AddActor(m_frustrum_actor);
    // } else {
    //     m_renderer->RemoveActor(m_frustrum_actor);
    // }
}

void LVRCvImageItem::openWindow()
{
    int scan_id = -1;
    int cam_id = -1;

    QTreeWidgetItem* parent_it = parent();

    while(parent_it != NULL)
    {
        
        if(parent_it->type() == LVRCamDataItemType)
        {
            LVRCamDataItem* item = static_cast<LVRCamDataItem*>(parent_it);
            cam_id = item->getCamId();
        } 

        if(parent_it->type() == LVRScanDataItemType)
        {
            LVRScanDataItem* item = static_cast<LVRScanDataItem*>(parent_it);
            scan_id = item->getScanId();
        }
        
        parent_it = parent_it->parent();
    }

    if(scan_id >= 0 && cam_id >= 0)
    {
        if(!m_graphics_view)
        {
            m_graphics_view = new LVRGraphicsView();
            m_graphics_view->init();

            cv::Mat image = m_sdm->loadImageData(scan_id, cam_id);
            cv::Mat cv_rgb;
            cv::cvtColor(image, cv_rgb, cv::COLOR_BGR2RGB);

            QImage* q_image = convertCvImageToQt(cv_rgb);

            QPixmap pixmap = QPixmap::fromImage(*q_image);

            QGraphicsScene* gs = new QGraphicsScene();
            gs->addPixmap(pixmap);
            gs->setSceneRect(pixmap.rect());

            m_graphics_view->setScene(gs);
            m_graphics_view->set_modifiers(Qt::NoModifier);
            m_graphics_view->fitInView(pixmap.rect(), Qt::KeepAspectRatio);

            // connect SLOT

            QObject::connect(
                m_graphics_view,
                &LVRGraphicsView::closed,
                [=]() { 
                    QObject::disconnect(m_graphics_view, &LVRGraphicsView::closed, 0, 0);
                    m_graphics_view->scene()->clear();
                    m_graphics_view = nullptr;
                }
            );
            
            m_graphics_view->show();

            delete q_image;
        } 
    } else {
        std::cout << "Couldnt find scan_id or cam_id. cant load Image from HDF5 file." << std::endl;
    }
}

void LVRCvImageItem::closeWindow()
{
    delete m_graphics_view;
}

QImage* LVRCvImageItem::convertCvImageToQt(cv::Mat& cv_rgb)
{
    //  QImage image(input->width, input->height, QImage::Format_RGB32);
    QImage* ret = nullptr;
    if(cv_rgb.type() == CV_8U)
    {
        ret = new QImage(cv_rgb.data, cv_rgb.rows, cv_rgb.cols, cv_rgb.step, QImage::Format_Indexed8);
    } else if(cv_rgb.type() == CV_8UC3) {

        ret = new QImage(cv_rgb.data, cv_rgb.cols, cv_rgb.rows, cv_rgb.step, QImage::Format_RGB888);
    }

    return ret;
}

void LVRCvImageItem::graphicsViewClosed()
{

}

LVRCvImageItem::~LVRCvImageItem()
{
    
}

} // namespace lvr2
