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

#include "display/Renderable.hpp"

#include "../app/Types.h"
#include "../viewers/Viewer.h"
#include "../widgets/CustomTreeWidgetItem.h"

using lssr::Renderable;
using lssr::Vertex;

class DataManager;

using lssr::Renderable;
using lssr::BoundingBox;

class DataCollector
{
public:
	DataCollector(Renderable* renderable, string name, CustomTreeWidgetItem* item = 0);
	virtual ~DataCollector();
	Renderable* renderable();
	string	name();
	BoundingBox<Vertex<float> >* boundingBox() { return m_renderable->boundingBox();}

	virtual CustomTreeWidgetItem* treeItem() { return m_treeItem;}
	virtual ViewerType supportedViewerType() = 0;

protected:

	CustomTreeWidgetItem*   m_treeItem;
	Renderable*	            m_renderable;
	string		            m_name;


};

#endif /* DATACOLLECTOR_H_ */
