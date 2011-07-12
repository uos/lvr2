/*
 * DataManager.h
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#ifndef DATAMANAGER_H_
#define DATAMANAGER_H_

#include <QtGui>

#include <string>
#include <map>
using std::string;
using std::map;

#include "DataCollector.h"

typedef map<string, DataCollector*> DataCollectorMap;

class DataManager : public QObject
{
	Q_OBJECT

public:
	DataManager();
	virtual ~DataManager();

	void update(DataCollector*);
	void loadFile( string filename );
	void exportData(string DataCollector);

public Q_SLOTS:
	void openFile();

Q_SIGNALS:
	void dataCollectorCreated(DataCollector*);
	void dataCollectorUpdate(DataCollector*);

private:
	DataCollectorMap		m_dataCollectorMap;
	QMutex					m_mutex;
};

#endif /* DATAMANAGER_H_ */
