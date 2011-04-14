/*
 * ThreadedDataCollector.h
 *
 *  Created on: 18.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef THREADEDDATACOLLECTOR_H_
#define THREADEDDATACOLLECTOR_H_

#include <QtGui>
#include "DataCollector.h"

class ThreadedDataCollector : public DataCollector, public QThread
{
public:
	ThreadedDataCollector(ClientProxy* proxy, DataManager* manager);
	virtual ~ThreadedDataCollector();

	virtual ViewerType supportedViewerType() = 0;
	virtual void run() = 0;

protected:
	QMutex				m_mutex;
};

#endif /* THREADEDDATACOLLECTOR_H_ */
