/*
 * DataManager.cpp
 *
 *  Created on: 07.10.2010
 *      Author: Thomas Wiemann
 */

#include "DataManager.h"
#include "DataCollector.h"
#include "DataCollectorFactory.h"

#include <iostream>
using std::string;

DataManager::DataManager()
{
	// TODO Auto-generated constructor stub

}

DataManager::~DataManager()
{
	// TODO Auto-generated destructor stub
}

void DataManager::update(DataCollector* dc)
{
	QMutexLocker locker(&m_mutex);
	Q_EMIT dataCollectorUpdate(dc);
}

void DataManager::openFile()
{
	QFileDialog file_dialog;
	QStringList file_names;
	QStringList file_types;

	file_types << "Point Clouds (*.pts)"
//			   << "Points and Normals (*.nor)"
			   << "PLY Models (*.ply)"
//			   << "Polygonal Meshes (*.bor)"
		       << "All Files (*.*)";


	//Set Title
	file_dialog.setWindowTitle("Open File");
	file_dialog.setFileMode(QFileDialog::ExistingFile);
	file_dialog.setFilters(file_types);

	if(file_dialog.exec()){
		file_names = file_dialog.selectedFiles();
	} else {
		return;
	}

	//Get filename from list
	string file_name = file_names.constBegin()->toStdString();

	// Create a new data collector object and save it in the
	// collector map
	DataCollector* c = DataCollectorFactory::instance()->create(file_name, this);
	m_dataCollectorMap.insert(make_pair(c->name(), c));

	// Signal the creation of a new data collector
	Q_EMIT dataCollectorCreated(c);
}


void DataManager::loadFile( string filename ) {

	/* Create a new data collector object and save it in the collector map */
	DataCollector * c = DataCollectorFactory::instance()->create( filename, this );

	if(c != 0)
	{
	    m_dataCollectorMap.insert( make_pair( c->name(), c ) );

	    /* Signal the creation of a new data collector */
	    Q_EMIT dataCollectorCreated( c );
	}
}
