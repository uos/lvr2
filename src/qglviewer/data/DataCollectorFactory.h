/*
 * DataCollectorFactory.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef DATACOLLECTORFACTORY_H_
#define DATACOLLECTORFACTORY_H_

#include <string>
#include <QtGui>

using std::string;

#include "DataCollector.h"

class DataManager;

class DataCollectorFactory : public QObject
{
    Q_OBJECT
public:
    DataCollectorFactory();

	virtual ~DataCollectorFactory() {};
	void create(string filename);

Q_SIGNALS:
    void dataCollectorCreated(DataCollector*);



};

#endif /* DATACOLLECTORFACTORY_H_ */
