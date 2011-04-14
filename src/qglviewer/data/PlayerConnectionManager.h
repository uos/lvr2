/*
 * PlayerConnectionManager.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef PLAYERCONNECTIONMANAGER_H_
#define PLAYERCONNECTIONMANAGER_H_

/**
 * CAREFUL: Don't include any Qt-Headers here or your will get
 * a namespace collision with the Boost libraries that are yoused
 * by Player!
 */

#include <playerclient/PlayerServerManager.h>
#include <playerclient/PlayerServer.h>

struct InterfaceDcr
{
	int id;
	int index;
	string name;
};

typedef list<ClientProxy*> ProxyList;
typedef list<PlayerServer*> ServerList;
typedef list<InterfaceDcr> InterfaceList;

typedef map<PlayerServer*, InterfaceList> InterfaceMap;
typedef map<PlayerServer*, ProxyList> ProxyMap;
typedef map<string, PlayerServer*> ServerMap;

class PlayerConnectionManager
{

public:
	virtual 		~PlayerConnectionManager();
	static 			PlayerConnectionManager* instance();

	PlayerServer* 	addServer(string ip, int port);

	void			removeServer(PlayerServer* server);
	void 			subscribe(ClientProxy* &proxy, string ip, int port, int id, int index);
	void 			unsubscribe(string ip, int port, int id, int index);

	ProxyList 		getProxys(string ip, int port);
	ProxyList		getProxys(PlayerServer*);

	ServerList 		getServers();

	InterfaceList	getInterfaces(PlayerServer* server);

private:
	PlayerConnectionManager();

	PlayerServer*	getServer(string ip, int port);

	void 			createInterfaceList(PlayerServer* server);
	void 			unsubscribeProxys(PlayerServer* server);

	ServerList		m_playerServers;
	ProxyMap		m_proxys;
	InterfaceMap	m_interfaceDcrs;

	static PlayerConnectionManager* m_instance;


};

#endif /* PLAYERCONNECTIONMANAGER_H_ */
