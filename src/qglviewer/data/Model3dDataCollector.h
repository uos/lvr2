/*
 * Model3dDataCollector.h
 *
 *  Created on: 14.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef MODEL3DDATACOLLECTOR_H_
#define MODEL3DDATACOLLECTOR_H_

#include "DataCollector.h"
#include <model3dproxy.h>

class Model3dDataCollector : public DataCollector
{
public:
	Model3dDataCollector(ClientProxy* proxy, DataManager* manager);
	virtual ~Model3dDataCollector();
	virtual ViewerType supportedViewerType();

};

#endif /* MODEL3DDATACOLLECTOR_H_ */
