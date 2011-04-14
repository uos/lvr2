/*
 * DataCollector.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef DATACOLLECTOR_H_
#define DATACOLLECTOR_H_



#include <string>
using std::string;

#include "model3d/Renderable.h"
#include <playerclient/PlayerServerManager.h>

#include "../app/Types.h"
#include "../viewers/Viewer.h"

class DataManager;

class DataCollector
{
public:

	DataCollector(ClientProxy* proxy = 0, DataManager* manager = 0);
	DataCollector(Renderable* renderable, string name, DataManager* manager);
	virtual ~DataCollector();
	Renderable* renderable();
	string	name();

	virtual ViewerType supportedViewerType() = 0;

protected:

	ClientProxy*	m_proxy;
	DataManager* 	m_manager;
	Renderable*		m_renderable;
	string			m_name;


};

#endif /* DATACOLLECTOR_H_ */
