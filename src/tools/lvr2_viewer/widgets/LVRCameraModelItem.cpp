#include "LVRCameraModelItem.hpp"

namespace lvr2
{

LVRCameraModelItem::LVRCameraModelItem(ScanCamera& cam)
{
    m_model = cam.camera;

    QIcon icon;
    icon.addFile(QString::fromUtf8(":/qv_transform_tree_icon.png"), QSize(), QIcon::Normal, QIcon::Off);
    setIcon(0, icon);
    setText(0, "CamModel");


    m_fxItem = new QTreeWidgetItem(this);
    m_cxItem = new QTreeWidgetItem(this);
    m_fyItem = new QTreeWidgetItem(this);
    m_cyItem = new QTreeWidgetItem(this);
    m_distortionItem = new QTreeWidgetItem(this);

    addChild(m_fxItem);
    addChild(m_cxItem);
    addChild(m_fyItem);
    addChild(m_cyItem);
    addChild(m_distortionItem);
    setModel(m_model);
}

void LVRCameraModelItem::setModel(PinholeModeld& model)
{
    m_model = model;
    QString num;

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
}


} //namespace lvr2