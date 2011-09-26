/**
 * TriangleMeshTreeWidgetItem.cpp
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#include "TriangleMeshTreeWidgetItem.h"

#include <sstream>
using std::stringstream;

TriangleMeshTreeWidgetItem::TriangleMeshTreeWidgetItem(int type)
     : CustomTreeWidgetItem(type)
{
   m_numVertices = 0;
   m_numFaces = 0;
   setInitialState(Qt::Checked);
}

TriangleMeshTreeWidgetItem::TriangleMeshTreeWidgetItem(QTreeWidgetItem* parent, int type)
    : CustomTreeWidgetItem(parent, type)
{
    m_numVertices = 0;
    m_numFaces = 0;
    setInitialState(Qt::Checked);
}


void TriangleMeshTreeWidgetItem::setNumVertices(size_t n)
{
    m_numVertices = n;

    // Create new item
    QTreeWidgetItem* vertItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Vertices: " << m_numVertices;

    // Set text and add child
    vertItem->setText(0, pstream.str().c_str());
    addChild(vertItem);
}

void TriangleMeshTreeWidgetItem::setNumFaces(size_t n)
{
    m_numFaces = n;

    // Create new item
    QTreeWidgetItem* faceItem = new QTreeWidgetItem(this);

    // Create label text
    stringstream pstream;
    pstream << "Faces: " << m_numVertices;

    // Set text and add child
    faceItem->setText(0, pstream.str().c_str());
    addChild(faceItem);
}
