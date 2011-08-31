/*
 * CustomTreeWidgetItem.h
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef CUSTOMTREEWIDGETITEM_H_
#define CUSTOMTREEWIDGETITEM_H_

#include <QtGui>

#include "display/Renderable.hpp"

using lssr::Renderable;

enum
{
    ServerItem = 1001,
    InterfaceItem,
    PointCloudItem,
    TriangleMeshItem,
    MultiPointCloudItem
};

enum
{
    Mesh              = 0x01,
    Points            = 0x02,
    PointNormals      = 0x04,
    VertexNormals     = 0x08,
    Vertices          = 0x10,
    Wireframe         = 0x20
};


class CustomTreeWidgetItem : public QTreeWidgetItem
{
public:
	CustomTreeWidgetItem(int type);
	CustomTreeWidgetItem(QTreeWidgetItem* parent, int type);

	virtual ~CustomTreeWidgetItem();
	virtual void setRenderable(Renderable* renderable) {m_renderable = renderable;};

	bool toggled();
	void setInitialState(Qt::CheckState state);

	Renderable* renderable() { return m_renderable;}
	void setName(string name);

    bool centerOnClick()                { return m_centerOnClick;}
    void setViewCentering(bool center)  { m_centerOnClick = true;}

    string name() { return m_name;}

    void setSupportedRenderModes(int mode) {m_renderMode = mode;}

    bool supportsMode(int mode);
protected:

	Qt::CheckState      m_oldCheckState;
	Renderable*         m_renderable;
	string              m_name;
	bool                m_centerOnClick;
	int                 m_renderMode;

};

#endif /* CUSTOMTREEWIDGETITEM_H_ */
