/*
 * Static3DDataCollector.h
 *
 *  Created on: 08.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef STATIC3DDATACOLLECTOR_H_
#define STATIC3DDATACOLLECTOR_H_

#include "DataCollector.h"

class Static3DDataCollector : public DataCollector
{
public:

	Static3DDataCollector(Renderable* renderable, string name, DataManager* manager, CustomTreeWidgetItem* item = 0);
	virtual ViewerType supportedViewerType();
};

#endif /* STATIC3DDATACOLLECTOR_H_ */
