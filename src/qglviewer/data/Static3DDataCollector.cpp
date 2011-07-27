/*
 * Static3DDataCollector.cpp
 *
 *  Created on: 08.10.2010
 *      Author: Thomas Wiemann
 */

#include "Static3DDataCollector.h"

Static3DDataCollector::Static3DDataCollector(Renderable* renderable, string name, CustomTreeWidgetItem* item)
	: DataCollector(renderable, name, item) { m_renderable->setActive(true);}

ViewerType Static3DDataCollector::supportedViewerType()
{
	return PERSPECTIVE_VIEWER;
}
