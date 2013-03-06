/*
 * mpi.cpp
 *
 *  Created on: 1.02.2013
 *      Author: Dominik Feldschnieders
 */

/*
 *
 * Programmierbeispiel der Stanfort Universität.
 *  http://www.slac.stanford.edu/comp/unix/farm/mpi.html
 */

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

// Las vegas Toolkit
#include "io/PointBuffer.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"
#include "mpi/KdTree.hpp"
#include "geometry/ColorVertex.hpp"

#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/PCLKSurface.hpp"

using namespace lssr;

// some easy typedefs
typedef ColorVertex<float, unsigned char>      cVertex;
typedef KdTree<cVertex>                        kd;
//später noch Unterscheidung machen, welches benutzt wird
typedef PointsetSurface<cVertex>                        psSurface;
typedef PCLKSurface<cVertex, cNormal>                   pclSurface;


int main (int argc , char *argv[]) {
	// Kd Tree
    // A list for all Nodes with less than MAX_POINTS
    std::list<KdNode<cVertex>*> m_nodelist;

	//Las Vegas Toolkit
    // m_ for Master
    // s_ for Slave

	// A shared-Pointer for the model, with the pointcloud in it
	ModelPtr 				m_model;

	// The pointcloud
	PointBufferPtr          m_loader;

	/// The currently stored points
	coord3fArr   			m_points;

	coord3fArr               c_normals;

	// Create a point loader object - Aus der Main (reconstruct)
	ModelFactory 			io_factory;

	// Number of points in the point cloud
	size_t 					m_numpoint;

	// Number of points in the point cloud (Child)
	int c_sizepackage;

	long int max_points = 10000;
	// for calculate normals
	int kd, kn, ki;
	kd = 40;
	kn = 40;
	ki = 40;


	// MPI

	// Anzahl an Processen
	int numprocs;
	// Rank / ID des Processes
	int rank;
	int i;

	int namelen = 0;
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	char idstring[32];
	char con_msg[128];
	char c_file_name[128];
	char c_end[1];
	int c_data_num;
	long int num_all_points;
	long int count_test = 0;

	// MPI-Typ zum Abfragen des Status der Übertragung
	MPI::Status stat;
	// Initialisiert die verbindung
	MPI::Init(argc, argv);

	//liefer die Anzahl an Prozessen
	numprocs = MPI::COMM_WORLD.Get_size();

	// liefert die Id / Rang des Prozesses
	rank = MPI::COMM_WORLD.Get_rank();

	// liefert den namen des Processors (Rechner auf dem es ausgeführt wird)
	MPI::Get_processor_name(processor_name, namelen);

	// Ausgabe welche Processe aktiv sind
	printf("Process %d on %s out of %d \n", rank, processor_name, numprocs);


	//auslesen der Datei
	m_model = io_factory.readModel( "flur3.pts" );

	// Aufbauen des Kd-Baums
	kd KDTree;

	m_nodelist = KDTree.GetList();


	// VaterProcess oder spaeter Master-Process
	if (rank == 0){
		// An alle eine Nachricht schicken zum Verbindungsaufbau
		for (i = 1; i < numprocs; i++)
		{
			sprintf(con_msg, "Hey Nummer %d...", i);
			MPI::COMM_WORLD.Send(con_msg, 128, MPI::CHAR, i, 0);

		}

		// wait for their answer
		for ( i = 1; i < numprocs; i++)
		{
			MPI::COMM_WORLD.Recv(con_msg, 128, MPI::CHAR, i, 0);
			printf("%s\n", con_msg);
		}

/*************************************** Connection is successful *****************/


		//dynamisches array für die Normalen anlegen
		float ** normals = new float*[numprocs - 1];

		//dynamisches array für die Indices anlegen
		long unsigned int ** Indizes = new long unsigned int*[numprocs - 1];

		// für laufvariable beim zusammenfügen
		int laufvariable[numprocs - 1];


	    sprintf(file_name, "scan%03d.3d", data_num);
	    string filename(file_name);

		//m_model = io_factory.readModel( file_name );

		typename std::list<KdNode<cVertex>*>::	iterator it= m_nodelist.begin();

		std::cout << "Init fertig, jetzt wird angefangen" << std::endl;
		// Anzahl aller Punkte zählen oder mit getNumPoints() von Pointbuffer aber das ist size_t
		num_all_points = 0;
		for ( it = m_nodelist.begin() ; it != m_nodelist.end(); ++it)
		{
			num_all_points += (*it)->getnumpoints();
		}

		it = m_nodelist.begin();
		//
		float * m_normal = new float[3 * num_all_points];

		// verschicken aller Datenpakete
		while ( it  != m_nodelist.end() )
		{

			std::cout << "Anzahl Durchlaeufe: " << count << std::endl;
			if (m_model != NULL)
			{
				m_loader = m_model->m_pointCloud;
			}
			else std::cout << " Model not existent" << std::endl;

			test_points = m_loader->getIndexedPointArray(m_numpoint);

			// Calculate bounding box
			//m_points = m_loader->getIndexedPointArray(m_numpoint);
			m_points = (*it)->getPoints();



			boost::shared_array<size_t> test = (*it)->getIndizes();
			std::cout << "jetzt" <<std::endl;
			std::cout << "Erster Wert des indexarray: " << test[0] << " zweiter: "<< test[1] << std::endl;




	// senden der Anzahl an Punkten die Folgen
			int * int_numpoint = new int[1];
			//int_numpoint[0] = static_cast<unsigned int>(m_numpoint);
			int_numpoint[0] = (*it)->getnumpoints();
			MPI::COMM_WORLD.Send(int_numpoint, 1, MPI::INT, client_serv_data, 2);
			// composed of the file name
			//char name[256];
			//sprintf(name, "scan%03d.3d", "Platzhalter");
			//string filename(name);

			// allokiere Normalenarray dynamisch
			normals[client_serv_data] = new float [3 * int_numpoint[0]];

			Indizes[client_serv_data] = (*it)->indizes.get();

			laufvariable[client_serv_data] = (*it)->getnumpoints();

			std::cout << "Punkt 1: " << m_points[1][0] << " Punkte aus Indizes: " << test_points[Indizes[client_serv_data][1]][0] << std::endl;


			std::cout << "Beim Host sind so viele Punkte drin: " << int_numpoint[0] << std::endl;

			std::cout << "Der Punkt an der stelle 1000: " << m_points[1000][0] << std::endl;

			data_num++;
			// sende dem Client, die Nummer der Datei, welche bearbeitet wird
			int int_datanum[1];
			int_datanum[0] = data_num;
			MPI::COMM_WORLD.Send(int_datanum, 1, MPI::INT, client_serv_data, 3);


			// sende Daten aus Scandatei
			MPI::Request req2 = MPI::COMM_WORLD.Isend(m_points.get(), 3 *  int_numpoint[0], MPI::FLOAT, client_serv_data, 1);
			req2.Wait();


			std::cout << "vorher" << std::endl;
			// empfange Normalen zurück
			MPI::Request tmp;
			 tmp = MPI::COMM_WORLD.Irecv(normals[client_serv_data], 3 * int_numpoint[0], MPI::FLOAT, client_serv_data, 4);
			 status[client_serv_data - 1] = tmp;
			std::cout << "danach" << std::endl;
	//		status[client_serv_data - 1].Wait();

/*******************************abspeichern  **************/

			if (count >= numprocs - 1)
			{
				client_serv_data = MPI::Request::Waitany( numprocs - 1, status);
				std::cout << "Der Process ist fertig und kann siene sachen abspeichern: " << client_serv_data << std::endl;
				client_serv_data++;


//für den test
				std::cout << "Neuer Test läuft an" << std::endl;
				//status[client_serv_data].Wait();
				// Punkte wieder in richtige Form fpr Pointbuffer bringen
//				boost::shared_array<float> norm (normals[client_serv_data]);

				std::cout << "Neuer Test läuft an" << std::endl;

				// The factory requires a model to save.
				// The model in turn requires a Pointbuffer and a Meshbuffer (can be emtpy).
				// The Pointbuffer contains the Indexlist.
//				PointBufferPtr pointcloud(new PointBuffer());
//				pointcloud->setPointNormalArray(norm, (*it)->getnumpoints() );
//				pointcloud->setIndexedPointArray((*it)->getPoints(), (*it)->getnumpoints());

				for (int x = 0; x < laufvariable[client_serv_data] ; x ++)
				{
					int n_buffer_pos = 3 * Indizes[client_serv_data][x];

					m_normal[n_buffer_pos]     = normals[client_serv_data][3 * x];
					m_normal[n_buffer_pos + 1] = normals[client_serv_data][ (3 * x) + 1];
					m_normal[n_buffer_pos + 2] = normals[client_serv_data][ (3 * x) + 2];

				}

//				boost::shared_array<float> norm (normal);
//
//				m_loader->setPointNormalArray(norm, num_all_points );




//				std::cout << "paar Punkte:" << norm[2] << " und " << m_points[2][0] << std::endl;
				//ModelPtr test_model( new Model);
//				m_model->m_pointCloud = m_loader;


//				char data_name[32];
//				sprintf(data_name, "Normals%03d.ply",count );


//				io_factory.saveModel(m_model, data_name);


				count++;
// ende test

				it++;
				client_serv_data++;
				client_serv_data = (client_serv_data % numprocs);
				if (client_serv_data == 0) client_serv_data++;

				std::cout << "Ende der Abfrage am Ende des Abspeicherfalls" << std::endl;

			}// ende if
			else
			{
				count++;
				it++;
				client_serv_data++;
				client_serv_data = (client_serv_data % numprocs);
				std::cout << "Ende der Abfrage am Ende" << std::endl;
			}
		}// Ende Dauerschleife

//test


		std::cout << "Stunde der Wahrheit, es wird abgespeichert" << std::endl;
		boost::shared_array<float> norm (m_normal);
		std::cout << "paar Punkte:" << norm[2] << " und " << norm[2000] << std::endl;

		long unsigned int tmp = static_cast<unsigned int>(m_loader->getNumPoints());

		std::cout << "groesse gerechnet:" << num_all_points << " und aus dem Pointbuffer " << tmp << std::endl;

		std::cout << "1" << std::endl;
		m_loader->setPointNormalArray(norm, m_loader->getNumPoints() );
		std::cout << "2" << std::endl;
		m_model->m_pointCloud = m_loader;
		std::cout << "3" << std::endl;
		io_factory.saveModel(m_model, "Normal.ply");
		std::cout << "4" << std::endl;

		// Beende Verbindung
		int end[1] = {-1};
		int j = 1;
		for (j = 1 ; j < numprocs ; j++)
		{
			std::cout << "Master schreibt das es zuende ist an: " << j << std::endl;
			MPI::COMM_WORLD.Send(end , 1, MPI::INT, j, 2);
		}
		std::cout << "Daten wurden gesendet" << std::endl;

// ist das so richtig, wirft manchmal fehler
		for (int i = 0 ; i < (numprocs -1) ; i++)
		{
			delete [] normals[i];
		}

		delete [] normals;

	}// Ende If
/**********************************************************************************************************/
	// Kinderprocesse
	else
	{
		//Abbruch Bedingung
		c_end[0] = 0;
		//MPI::COMM_WORLD.Irecv(c_end, 1, MPI::CHAR, 0, 3);


		// Warte im blockierendem Modus...
		MPI::COMM_WORLD.Recv(con_msg, 128, MPI::CHAR, 0,0);

		//erstelle Antwort
		sprintf(idstring, "Processor %d ", rank);
		strcat(con_msg,idstring);
		strcat(con_msg, "hat verstanden und hält sich bereit!");

		// Datei welche zeigt, ob der angesprochene Rechner auch richtig geantwortet hat. Liegt im gemeinsamen Speicher
		f.open("testclient.dat");
		f << con_msg << std::endl;


		MPI::COMM_WORLD.Send(con_msg, 128, MPI::CHAR, 0, 0);

/************************************ Connection is successful *******************/
		//int count = 1;


		// Schleife zum Empfangen der daten, -1 bricht Vorgang ab
		while(true)
		{
			MPI::COMM_WORLD.Recv( &c_sizepackage, 1, MPI::INT , 0,2);

			std::cout << "Abbruchssignal wenn gleich -1, sonst groesse der Punktwolke:" << c_sizepackage << std::endl;
			//Abbruchbedingung
			if (c_sizepackage == -1)
			{
				std::cout << "Abbruch hat gezuendet!!!!" << std::endl;
				break;
			}
			else
			{
				std::cout << "Die Datei ist so groß: " << c_sizepackage << std::endl;
		/**** Hier vorher noch schicken wie groß das Paket ist, in size_package speichern ***/


				MPI::COMM_WORLD.Recv(&c_data_num, 1, MPI::INT, 0, 3);

				// empfangen der scan-Dateien
				float * tmp = new float[3 * c_sizepackage];
				MPI::Request client_req2 = MPI::COMM_WORLD.Irecv(tmp, 3 * c_sizepackage, MPI::FLOAT, 0, 1);

				// warten bis Übertragung komplett ist
				client_req2.Wait();

				// Punkte wieder in richtige Form fpr Pointbuffer bringen
				boost::shared_array<float> punkte (tmp);


				// The factory requires a model to save.
				// The model in turn requires a Pointbuffer and a Meshbuffer (can be emtpy).
				// The Pointbuffer contains the Indexlist.
				PointBufferPtr pointcloud(new PointBuffer());
				pointcloud->setPointArray(punkte, c_sizepackage);

				psSurface::Ptr surface;
				surface = psSurface::Ptr( new pclSurface(pointcloud));

				// Set search options for normal estimation and distance evaluation
				//willkürliche Werte, eigentlich mnit option
				surface->setKd(kd);
				surface->setKi(ki);
				surface->setKn(kn);

				// calculate the normals
			    Timestamp ts;
				surface->calculateSurfaceNormals();
				cerr << ts.getElapsedTimeInMs() << endl;

				ModelPtr pn( new Model);
				pn->m_pointCloud = surface->pointBuffer();


				pointcloud = surface->pointBuffer();
				size_t size_normal;
				c_normals = pointcloud->getIndexedPointNormalArray(size_normal);

				// send the normals back to the Masterprocess
				MPI::COMM_WORLD.Send(c_normals.get(), 3 * c_sizepackage, MPI::FLOAT, 0, 4);

			}
		}

	}// End else

	MPI_Finalize();

}
