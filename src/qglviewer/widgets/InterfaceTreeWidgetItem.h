/*
 * InterfaceTreeWidgetItem.h
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef INTERFACETREEWIDGETITEM_H_
#define INTERFACETREEWIDGETITEM_H_

#include "../data/PlayerConnectionManager.h"
#include "CustomTreeWidgetItem.h"

class InterfaceTreeWidgetItem : public CustomTreeWidgetItem
{
public:
	InterfaceTreeWidgetItem(InterfaceDcr dcr, QTreeWidgetItem* parent, int type = InterfaceItem);
	virtual ~InterfaceTreeWidgetItem();

	InterfaceDcr 	description();
	PlayerServer*	server();

private:
	InterfaceDcr m_dcr;
};

#endif /* INTERFACETREEWIDGETITEM_H_ */
