/*
 * DataCollector.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataCollector.h"

DataCollector::DataCollector(Renderable* renderable, string name, DataManager* manager, QTreeWidgetItem* item)
{
	m_manager = manager;
	m_renderable = renderable;
	m_name = name;
	m_treeItem = item;
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
