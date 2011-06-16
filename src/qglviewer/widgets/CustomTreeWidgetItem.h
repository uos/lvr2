/*
 * CustomTreeWidgetItem.h
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef CUSTOMTREEWIDGETITEM_H_
#define CUSTOMTREEWIDGETITEM_H_

#include <QtGui>

#include "model3d/Renderable.h"

enum {ServerItem = 1001, InterfaceItem, PointCloudItem};

class CustomTreeWidgetItem : public QTreeWidgetItem
{
public:
	CustomTreeWidgetItem(int type);
	CustomTreeWidgetItem(QTreeWidgetItem* parent, int type);

	virtual ~CustomTreeWidgetItem();

	bool toggled();
	void setInitialState(Qt::CheckState state);
	void setRenderable(Renderable* renderable) {m_renderable = renderable;};
	Renderable* renderable() { return m_renderable;}

protected:
	Qt::CheckState      m_oldCheckState;
	Renderable*         m_renderable;
};

#endif /* CUSTOMTREEWIDGETITEM_H_ */
