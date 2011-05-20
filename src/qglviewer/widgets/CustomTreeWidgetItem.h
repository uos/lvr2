/*
 * CustomTreeWidgetItem.h
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef CUSTOMTREEWIDGETITEM_H_
#define CUSTOMTREEWIDGETITEM_H_

#include <QtGui>

enum {ServerItem = 1001, InterfaceItem};

class CustomTreeWidgetItem : public QTreeWidgetItem
{
public:
	CustomTreeWidgetItem(int type);
	CustomTreeWidgetItem(QTreeWidgetItem* parent, int type);

	virtual ~CustomTreeWidgetItem();

	bool toggled();
	void setInitialState(Qt::CheckState state);


protected:
	Qt::CheckState m_oldCheckState;
};

#endif /* CUSTOMTREEWIDGETITEM_H_ */
