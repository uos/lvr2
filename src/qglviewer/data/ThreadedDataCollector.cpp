/*
 * ThreadedDataCollector.cpp
 *
 *  Created on: 18.10.2010
 *      Author: Thomas Wiemann
 */

#include "ThreadedDataCollector.h"

ThreadedDataCollector::ThreadedDataCollector(ClientProxy* proxy, DataManager* manager)
	: DataCollector(proxy, manager)
{
	cout << "THREADED COLLECTOR" << endl;
}

ThreadedDataCollector::~ThreadedDataCollector()
{

}
