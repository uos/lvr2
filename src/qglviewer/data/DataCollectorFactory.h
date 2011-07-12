/*
 * DataCollectorFactory.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef DATACOLLECTORFACTORY_H_
#define DATACOLLECTORFACTORY_H_

#include <string>
using std::string;

#include "DataCollector.h"

class DataManager;

class DataCollectorFactory
{
public:
	virtual ~DataCollectorFactory() {};

	static DataCollectorFactory* instance();
    DataCollector* create(string filename);

private:
	DataCollectorFactory() {};
	static DataCollectorFactory* m_instance;

};

#endif /* DATACOLLECTORFACTORY_H_ */
