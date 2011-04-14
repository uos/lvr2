/*
 * InterfaceTreeWidgetItem.cpp
 *
 *  Created on: 13.10.2010
 *      Author: Thomas Wiemann
 */
#include <sstream>
using std::stringstream;

#include "InterfaceTreeWidgetItem.h"
#include "ServerTreeWidgetItem.h"

InterfaceTreeWidgetItem::InterfaceTreeWidgetItem(InterfaceDcr dcr, QTreeWidgetItem* parent, int type)
	: CustomTreeWidgetItem(parent, type), m_dcr(dcr)
{
	// Create a QString from interface descriptor
	stringstream ss;
	ss << m_dcr.name << "(" << m_dcr.id << "):" << m_dcr.index;
	QString label(ss.str().c_str());

	// Set Text of this item
	setText(0, label);

}

PlayerServer* InterfaceTreeWidgetItem::server()
{
	if(parent()->type() == ServerItem)
	{
		ServerTreeWidgetItem* item = static_cast<ServerTreeWidgetItem*>(parent());
		return item->server();
	}
	return 0;
}

InterfaceTreeWidgetItem::~InterfaceTreeWidgetItem()
{
	// TODO Auto-generated destructor stub
}

InterfaceDcr InterfaceTreeWidgetItem::description()
{
	return m_dcr;
}
