/*
 * DataCollector.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollector.h"

DataCollector::DataCollector(ClientProxy* proxy, DataManager* manager)
{
	cout << "DataCollector()" << proxy << endl;
	m_proxy = proxy;
	m_manager = manager;
	m_renderable = 0;
	m_name = "";
}

DataCollector::DataCollector(Renderable* renderable, string name, DataManager* manager)
{
	m_manager = manager;
	m_renderable = renderable;
	m_name = name;
}

DataCollector::~DataCollector()
{
	// TODO Auto-generated destructor stub
}

Renderable* DataCollector::renderable()
{
	return m_renderable;
}

string DataCollector::name()
{
	return m_name;
}
