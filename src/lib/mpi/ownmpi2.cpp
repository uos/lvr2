/*
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

//Normals
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/PCLKSurface.hpp"

using namespace lssr;

#include "geometry/ColorVertex.hpp"

typedef ColorVertex<float, unsigned char>      cVertex;
typedef KdTree<cVertex>                        kd;
typedef PointsetSurface<cVertex>                        psSurface;
typedef PCLKSurface<cVertex, cNormal>                   pclSurface;

int main (int argc , char *argv[]) {
	// Kd Tree
    // A list for all Nodes with less than MAX_POINTS
    std::list<KdNode<cVertex>*> m_nodelist;

	//Las Vegas Toolkit

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

	// MPI
	std::ofstream f;

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
	
	// Aufbauen des Kd-Baums
	kd KDTree;

	m_nodelist = KDTree.GetList();

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


	// VaterProcess oder spaeter Master-Process
	if (rank == 0){
		// An alle eine Nachricht schicken zum Verbindungsaufbau
		for (i = 1; i < numprocs; i++)
		{
			sprintf(con_msg, "Hey Nummer %d...", i);
			MPI::COMM_WORLD.Send(con_msg, 128, MPI::CHAR, i, 0);

		}

		// dann blockierend darauf warten
		for ( i = 1; i < numprocs; i++)
		{
			MPI::COMM_WORLD.Recv(con_msg, 128, MPI::CHAR, i, 0);
			printf("%s\n", con_msg);
		}

/*************************************** Connection is successful *****************/
		int data_num = 1;
		int client_serv_data = 1;
		char file_name[256];
		MPI::Request status[numprocs-1];


	    sprintf(file_name, "scan%03d.3d", data_num);
	    string filename(file_name);

		m_model = io_factory.readModel( file_name );

		typename std::list<KdNode<cVertex>*>::	iterator it= m_nodelist.begin();
		int count = 1;
		// verschicken aller Datenpakete
		while ( m_model != NULL)
		{

			if (m_model != NULL)
			{
				m_loader = m_model->m_pointCloud;
			}
			else std::cout << " Model not existent" << std::endl;


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


			std::cout << "Beim Host sind so viele Punkte drin: " << int_numpoint[0] << std::endl;

			std::cout << "Der Punkt an der stelle 1000: " << m_points[1000][0] << std::endl;


			// sende dem Client, die Nummer der Datei, welche bearbeitet wird
			int int_datanum[1];
			int_datanum[0] = data_num;
			MPI::COMM_WORLD.Send(int_datanum, 1, MPI::INT, client_serv_data, 3);


			// sende Daten aus Scandatei
			MPI::Request req2 = MPI::COMM_WORLD.Isend(m_points.get(), 3 *  int_numpoint[0], MPI::FLOAT, client_serv_data, 1);
			req2.Wait();
			float * normals = new float[3 * int_numpoint[0]];
			// empfange Normalen zurück
			//status[client_serv_data] =
			MPI::COMM_WORLD.Recv(normals, 3 * int_numpoint[0], MPI::FLOAT, client_serv_data, 4);

			// go on to next datafile
			data_num++;
			sprintf(file_name, "scan%03d.3d", data_num);
			m_model = io_factory.readModel( file_name );

			// wait till message is send
			//req2.Wait();

//für den test
			std::cout << "Neuer Test läuft an" << std::endl;
			//status[client_serv_data].Wait();
			// Punkte wieder in richtige Form fpr Pointbuffer bringen
			boost::shared_array<float> norm (normals);

			std::cout << "Neuer Test läuft an" << std::endl;

			// The factory requires a model to save.
			// The model in turn requires a Pointbuffer and a Meshbuffer (can be emtpy).
			// The Pointbuffer contains the Indexlist.
			PointBufferPtr pointcloud(new PointBuffer());
			pointcloud->setPointNormalArray(norm, (*it)->getnumpoints() );
			pointcloud->setIndexedPointArray((*it)->getPoints(), (*it)->getnumpoints());

			std::cout << "paar Punkte:" << norm[2] << " und " << m_points[2][0] << std::endl;
			ModelPtr test_model( new Model);
			test_model->m_pointCloud = pointcloud;

			
			char data_name[32];
			sprintf(data_name, "Normals%03d.ply",count );


			io_factory.saveModel(test_model, data_name);


			count++;
// ende test

			std::cout << "Neuer Test ist fertig" << std::endl;
			// who is next and with witch file
			it++;
			client_serv_data++;
			client_serv_data = (client_serv_data % numprocs);
			if (client_serv_data == 0) client_serv_data++;
		}

		// Beende Verbindung
		int end[1] = {-1};
		int j = 1;
		for (j = 1 ; j < numprocs ; j++)
		{
			std::cout << "Master schreibt das es zuende ist an: " << j << std::endl;
			MPI::COMM_WORLD.Send(end , 1, MPI::INT, j, 2);
		}
		std::cout << "Daten wurden gesendet" << std::endl;

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

			std::cout << "sizepackage:" << c_sizepackage << std::endl;
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

// Normalentest
				psSurface::Ptr surface;
				surface = psSurface::Ptr( new pclSurface(pointcloud));

				// Set search options for normal estimation and distance evaluation
				//willkürliche Werte, eigentlich mnit option
				surface->setKd(40);
				surface->setKi(40);
				surface->setKn(40);

				// berechnen der Normalen mit Zeit
			    Timestamp ts;
				surface->calculateSurfaceNormals();
				cerr << ts.getElapsedTimeInMs() << endl;

				ModelPtr pn( new Model);
				pn->m_pointCloud = surface->pointBuffer();

				char pointnormals[32];
				sprintf(pointnormals, "pointnormals%03d%02d.ply",c_data_num, rank );


				ModelFactory::saveModel(pn, pointnormals);

				pointcloud = surface->pointBuffer();
				size_t size_normal;
				c_normals = pointcloud->getIndexedPointNormalArray(size_normal);

				MPI::COMM_WORLD.Send(c_normals.get(), 3 * c_sizepackage, MPI::FLOAT, 0, 4);

				//std::cout << "ist das: " << surface->pointBuffer()->m_points[] << "gleich dem:" << << "oder dem: " << << "oder dem: " << << std::endl;


//ende test
				MeshBufferPtr mesh;
				ModelPtr model( new Model( pointcloud, mesh ) );

				//Testweises speichern der übertragenen Datei auf dem lokalen Speichers des benutzten Rechners
				sprintf(c_file_name, "/mpitest/savedscan%03d.3d", c_data_num);

				io_factory.saveModel(model, c_file_name);

				std::cout << "In der Punktwolke sind es so viele Punkte: " << pointcloud->getNumPoints() << std::endl;
				//count++;
			}
		}

	}// Ende else fall

	std::cout << "Process Nr." << rank << " beendet jetzt seine Arbeit!" << std::endl;
	MPI_Finalize();

}

//namespace
