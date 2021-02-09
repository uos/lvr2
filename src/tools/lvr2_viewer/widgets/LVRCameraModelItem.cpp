#include "LVRCameraModelItem.hpp"

namespace lvr2
{

LVRCameraModelItem::LVRCameraModelItem(ScanCamera& cam) :
    QTreeWidgetItem(LVRCameraModelItemType)
{
    m_model = cam.camera;

    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_transform_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
    setText(0, "CamModel");

    //create items for camera parameters
    m_fxItem = std::make_shared<QTreeWidgetItem>(this);
    m_cxItem = std::make_shared<QTreeWidgetItem>(this);
    m_fyItem = std::make_shared<QTreeWidgetItem>(this);
    m_cyItem = std::make_shared<QTreeWidgetItem>(this);
    m_distortionItem = std::make_shared<QTreeWidgetItem>(this);
    
    //add created items as children
    addChild(m_fxItem.get());
    addChild(m_cxItem.get());
    addChild(m_fyItem.get());
    addChild(m_cyItem.get());
    addChild(m_distortionItem.get());
    
    for (int i = 0; i < 4; i++)
    {
        m_distortionCoef[i] = std::make_shared<QTreeWidgetItem>(this);
        m_distortionItem->addChild(m_distortionCoef[i].get());
    }

    setModel(m_model);
}

void LVRCameraModelItem::setModel(PinholeModeld& model)
{
    m_model = model;
    QString num;

    //fill child items with text
    m_fxItem->setText(0, "fx:");
    m_fxItem->setText(1, num.setNum(m_model.fx,'f'));

    m_cxItem->setText(0, "cx:");
    m_cxItem->setText(1, num.setNum(m_model.cx,'f'));

    m_fyItem->setText(0, "fy:");
    m_fyItem->setText(1, num.setNum(m_model.fy,'f'));

    m_cyItem->setText(0, "cy:");
    m_cyItem->setText(1, num.setNum(m_model.cy,'f'));

    m_distortionItem->setText(0, "Distortion Model:");

    m_distortionItem->setText(1, QString::fromStdString(m_model.distortionModel));

    for (int i = 0; i < 4; i++)
    {
        num = QString("k") + QString::number(i);
        m_distortionCoef[i]->setText(0, num);
        m_distortionCoef[i]->setText(1, num.setNum(m_model.k[i]));
    }
}


} //namespace lvr2