/*
 * ServerTreeWidgetItem.h
 *
 *  Created on: 12.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef SERVERTREEWIDGETITEM_H_
#define SERVERTREEWIDGETITEM_H_

#include <playerclient/PlayerServer.h>
#include "CustomTreeWidgetItem.h"

class ServerTreeWidgetItem : public CustomTreeWidgetItem
{
public:
	ServerTreeWidgetItem(PlayerServer* server, int type = ServerItem);
	virtual ~ServerTreeWidgetItem();

	PlayerServer* 	server();

private:
	PlayerServer* 	m_server;

};

#endif /* SERVERTREEWIDGETITEM_H_ */
