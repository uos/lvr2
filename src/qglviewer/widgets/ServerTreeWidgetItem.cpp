/*
 * ServerTreeWidgetItem.cpp
 *
 *  Created on: 12.10.2010
 *      Author: Thomas Wiemann
 */

#include "ServerTreeWidgetItem.h"

#include "../data/PlayerConnectionManager.h"
#include "InterfaceTreeWidgetItem.h"

ServerTreeWidgetItem::ServerTreeWidgetItem(PlayerServer* server, int type)
	: CustomTreeWidgetItem(type), m_server(server)
{

	// Setup widget item
	setText(0, QString(m_server->getHost().c_str()));

	// Add children
	InterfaceList proxys = PlayerConnectionManager::instance()->getInterfaces(m_server);
	InterfaceList::iterator it;

	for(it = proxys.begin(); it != proxys.end(); it++)
	{
		InterfaceDcr dcr = *it;
		InterfaceTreeWidgetItem* item = new InterfaceTreeWidgetItem(dcr, this);
		item->setInitialState(Qt::Unchecked);
		addChild(item);
	}
}

ServerTreeWidgetItem::~ServerTreeWidgetItem()
{
	// TODO Auto-generated destructor stub
}

PlayerServer* ServerTreeWidgetItem::server()
{
	return m_server;
}

