/*
 * PlayerConnectionManager.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "PlayerConnectionManager.h"

#include <utility>
using std::make_pair;

PlayerConnectionManager* PlayerConnectionManager::m_instance = 0;

PlayerConnectionManager::PlayerConnectionManager() {}

PlayerConnectionManager::~PlayerConnectionManager() {}


PlayerConnectionManager* PlayerConnectionManager::instance()
{
	if(PlayerConnectionManager::m_instance == 0)
	{
		PlayerConnectionManager::m_instance = new PlayerConnectionManager;
	}
	return PlayerConnectionManager::m_instance;
}


PlayerServer* PlayerConnectionManager::addServer(string ip, int port)
{
	PlayerServer* server = PlayerServerManager::SubscribeServer(ip, port);

	// Check for double insertions
	ServerList::iterator it;
	it = find (m_playerServers.begin(), m_playerServers.end(), server);

	if(server != 0 && (it == m_playerServers.end()))
	{

		createInterfaceList(server);
		m_playerServers.push_back(server);
		return server;
	}
	return 0;
}

void PlayerConnectionManager::subscribe(ClientProxy* &proxy, string ip, int port, int id, int index)
{
	// Get server object from ip and port
	PlayerServer* server = getServer(ip, port);
	if(server != 0)
	{
		// Subscribe to server and create proxy
		proxy = PlayerServerManager::SubscribeProxy(ip, port, id, index);
		cout << "PROXY TO SUBSCRIBE:" << proxy << endl ;

		if(proxy != 0)
		{
			// Add proxy to the server's proxy list
			ProxyList p_list = getProxys(server);

			// Insert proxy to list if it wasn't already in it
			ProxyList::iterator it;
			it = find(p_list.begin(), p_list.end(), proxy);
			if(it == p_list.end())
			{
				p_list.push_back(proxy);
				m_proxys[server] = p_list;
			}
		}
	}
}

void PlayerConnectionManager::unsubscribe(string ip, int port, int id, int index)
{
	// Unsubscribe interface
	PlayerServerManager::Unsubscribe(ip, port, id, index);

	// Remove proxy from list
	PlayerServer* server = getServer(ip, port);
	if(server != 0)
	{
		ClientProxy* proxy = 0;
		ProxyList p_list = m_proxys[server];
		ProxyList::iterator it;
		for(it = p_list.begin(); it != p_list.end(); it++)
		{
			proxy = *it;
			int tmp_index = proxy->GetIndex();
			int tmp_id = proxy->GetInterface();
			if((index == tmp_index) && (id == tmp_id))
			{
				cout << "Removing Proxy" << endl;
				ProxyList::iterator it;
				it = find(p_list.begin(), p_list.end(), proxy);
				p_list.erase(it);
				m_proxys[server] = p_list;
				return;
			}
		}
	}
}


ProxyList PlayerConnectionManager::getProxys(string ip, int port)
{
	PlayerServer* server = getServer(ip, port);
	if(server != 0)
	{
		return getProxys(server);
	}
	return ProxyList();
}

ProxyList PlayerConnectionManager::getProxys(PlayerServer* server)
{
	ServerList::iterator it;
	for(it = m_playerServers.begin(); it != m_playerServers.end(); it++)
	{
		PlayerServer* srv = *it;
		if(srv == server && server != 0)
		{
			return m_proxys[srv];
		}
	}
	return ProxyList();
}

PlayerServer* PlayerConnectionManager::getServer(string ip, int port)
{
	ServerList::iterator it;

	for(it = m_playerServers.begin(); it != m_playerServers.end(); it++)
	{
		PlayerServer* tmp = *it;
		if(tmp == 0) return tmp;
		if(tmp->getHost() == ip && tmp->getPort() == port) return tmp;
	}

	return 0;
}

void PlayerConnectionManager::createInterfaceList(PlayerServer* server)
{
	// Create a player client instance
	PlayerClient* pc = PlayerServerManager::getPlayerClient(server->getHost(), server->getPort());

	// Request device list and create a interface descriptor list
	InterfaceList interface_list;
	pc->RequestDeviceList();
	list<playerc_device_info_t> device_list = pc->GetDeviceList();
	list<playerc_device_info_t>::iterator iter;
	for(iter = device_list.begin(); iter != device_list.end(); iter++)
	{
		InterfaceDcr dcr;
		dcr.id = iter->addr.interf;
		dcr.index = iter->addr.index;
		dcr.name = pc->LookupName(dcr.id);

		interface_list.push_back(dcr);
	}
	m_interfaceDcrs[server] = interface_list;
}

InterfaceList PlayerConnectionManager::getInterfaces(PlayerServer* server)
{
	InterfaceMap::iterator it = m_interfaceDcrs.find(server);
	if(it != m_interfaceDcrs.end())
	{
		return it->second;
	}
	return InterfaceList();
}

void PlayerConnectionManager::unsubscribeProxys(PlayerServer* server)
{
		// Unsubscribe all proxys
		InterfaceList dcrs = getInterfaces(server);
		InterfaceList::iterator it;

		for(it = dcrs.begin(); it != dcrs.end(); it++)
		{
			InterfaceDcr dcr;
			unsubscribe(server->getHost(), server->getPort(), dcr.id, dcr.index);
		}
}

void PlayerConnectionManager::removeServer(PlayerServer* server)
{
	unsubscribeProxys(server);
	m_proxys.erase(server);
	m_interfaceDcrs.erase(server);
	m_proxys.erase(server);

	ServerList::iterator it = find(m_playerServers.begin(), m_playerServers.end(), server);
	m_playerServers.erase(it);


}

