/**
 * PointCloudTreeWidgetItem.cpp
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#include "PointCloudTreeWidgetItem.h"
#include <sstream>
using std::stringstream;

PointCloudTreeWidgetItem::PointCloudTreeWidgetItem(int type) : CustomTreeWidgetItem(type)
{
    m_name = "undefined";
    m_numPoints = 0;
    setText(0, QString(m_name.c_str()));
}

PointCloudTreeWidgetItem::PointCloudTreeWidgetItem(QTreeWidgetItem* parent, int type)
    : CustomTreeWidgetItem(parent, type)
{
    m_name = "undefined";
    m_numPoints = 0;

    setText(0, QString(m_name.c_str()));
}

void PointCloudTreeWidgetItem::setNumPoints(size_t numPoints)
{
    m_numPoints = numPoints;

    // Create new item
    QTreeWidgetItem* pointsItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Points: " << m_numPoints;

    // Set text and add child
    pointsItem->setText(0, pstream.str().c_str());
    addChild(pointsItem);
}
