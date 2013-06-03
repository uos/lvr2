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
#include "src/mpi/KdTree.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/Normal.hpp"

#include "reconstruction/AdaptiveKSearchSurface.hpp"

#include <boost/program_options.hpp>

using namespace lssr;
namespace po = boost::program_options;


int main (int argc , char *argv[]) {
  int count_serv = 0;  
  fstream f;
  char test_aufgabe_name[64];
  
  
  Timestamp start;

    int kd, kn, ki; 
    long int max_points, min_points;
    bool ransac;
    
    
    po::options_description desc("Allowed options");
    desc.add_options()
      ("help"      , "produce help message")
      ("kd"        , po::value<int>(&kd)->default_value(40), "set kd")
      ("ki"        , po::value<int>(&ki)->default_value(40), "set ki")
      ("kn"        , po::value<int>(&kn)->default_value(40), "set kn")
      ("maxpoints" , po::value<long int>(&max_points)->default_value(100000), "set maxpoints, min 5000 and min 32 * kn bzw. ki")
      ("file"      , po::value<string>()->default_value("noinput"), "Inputfile")
      ("ransac"    , "Use RANSAC based normal estimation")
    ;
    
    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);    
  
    if (vm.count("help")) {
      cout << desc << "\n";
      return 1;
    }
        
        if(vm.count("ransac")) 
	{
	  ransac = true;
	} 
	else
	{
	  ransac = false;
	}

	// Kd Tree
    // A list for all Nodes with less than MAX_POINTS
    std::list<KdNode<ColorVertex<float, unsigned char>>*> m_nodelist;

	//Las Vegas Toolkit
    // m_ for Master
    // s_ for Slave

	// A shared-Pointer for the model, with the pointcloud in it
	ModelPtr 				m_model;

	// The pointcloud
	PointBufferPtr          m_loader;

	// The currently stored points
	coord3fArr   			m_points;

	coord3fArr               c_normals;

	// Create a point loader object - Aus der Main (reconstruct)
	ModelFactory 			io_factory;

	// Number of points in the point cloud
	size_t 					m_numpoint;

	// Number of points in the point cloud (Child)
	int c_sizepackage;


	// for calculate normals

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
	float expansion_bounding[6];

				

	//if send this, +1 because this the Master in MPI has the rank 0 but for all Indizes this is more pleasant
	int client_serv_data = 0;

	int count = 1;
	int progress = 0;


	// Initializes the connection
	MPI::Init(argc, argv);

	// gives the number of processes
	numprocs = MPI::COMM_WORLD.Get_size();

	// gives the Id / Rang of the processes
	rank = MPI::COMM_WORLD.Get_rank();

	// returns the name of the processor (computer on which it runs)
	MPI::Get_processor_name(processor_name, namelen);


	MPI::Request status[numprocs-1];
	// Master-Process
	if (rank == 0){
		int * int_numpoint = new int[numprocs - 1];
		int * tmp_int = new int[1];
	  
	/*temporär wird noch ausgelesen, später Übergabe durch das Programm */
		//Read the file and get the pointcloud
		
		if (  vm["file"].as<string>() != "noinput" ) m_model = io_factory.readModel( vm["file"].as<string>() );

		if (m_model != NULL)
		{
			m_loader = m_model->m_pointCloud;
		}
		else
		{
		  std::cout << "Model can´t be load!!!!" << std::endl;
		  MPI_Finalize();
		  return 1;
		  
		}
		
		//calc min_points
		if ( kd >= ki && kd >= kn) min_points = 50 + ( 32 * kd );
		else if ( ki >= kd && ki >= kn) min_points = 50 + ( 32 * ki );
		else min_points = 50 + ( 32 * kn );
		
		if (max_points < (  min_points )  ) max_points = ( 2 * min_points );
		// Building the Kd tree with max max_points in every packete
		KdTree<cVertex> KDTree(m_loader, max_points, min_points);

		// get the list with all Nodes with less than MAX_POINTS
		m_nodelist = KDTree.GetList();	

		BoundingBox<cVertex> tmp_BoundingBox = KDTree.GetBoundingBox();
		cVertex tmp_min = tmp_BoundingBox.getMin();
		cVertex tmp_max = tmp_BoundingBox.getMax();

		expansion_bounding[0] = tmp_min[0];
		expansion_bounding[1] = tmp_min[1];
		expansion_bounding[2] = tmp_min[2];
		expansion_bounding[3] = tmp_max[0];
		expansion_bounding[4] = tmp_max[1];
		expansion_bounding[5] = tmp_max[2];
		
		// Send an announcement to all other processes
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
		
		//send min and max of the Boundingbox to all other processes
		for ( i = 1; i < numprocs; i++)
		{
		    MPI::COMM_WORLD.Send(expansion_bounding, 6, MPI::FLOAT, i, 5);
		}
			

/*************************************** Connection is successful *****************/


		//dynamisches array für die Normalen anlegen
		float ** normals = new float*[numprocs - 1];

		//dynamisches array für die Indices anlegen
		long unsigned int ** Indizes = new long unsigned int*[numprocs - 1];

		// für laufvariable beim zusammenfügen
		int laufvariable[numprocs - 1];


		typename std::list<KdNode<cVertex>*>::	iterator it= m_nodelist.begin();


		// a buffer to store all the normals
		float * m_normal = new float[3 * static_cast<unsigned int>(m_loader->getNumPoints())];


		// Send all data packets
		while ( it  != m_nodelist.end() )
		{

		  // get the next points in row
			m_points = (*it)->getPoints();

			// send the number of points that will follow
			int_numpoint[client_serv_data] = (*it)->getnumpoints();
			tmp_int[0] = int_numpoint[client_serv_data];
			
			MPI::COMM_WORLD.Send(tmp_int, 1, MPI::INT, client_serv_data + 1, 2);
			
			// allokiere normal array dynamically
			normals[client_serv_data] = new float [3 * int_numpoint[client_serv_data]];

			// get the indices for the original sequence
			Indizes[client_serv_data] = (*it)->indizes.get();

			laufvariable[client_serv_data] = (*it)->getnumpoints();


			// send Data
			MPI::Request req2 = MPI::COMM_WORLD.Isend(m_points.get(), 3 *  int_numpoint[client_serv_data], MPI::FLOAT, client_serv_data + 1, 1);


			// Reciev Normals
			MPI::Request tmp;
			tmp = MPI::COMM_WORLD.Irecv(normals[client_serv_data], 3 * int_numpoint[client_serv_data], MPI::FLOAT, client_serv_data + 1, 4);
			status[client_serv_data] = tmp;


			req2.Wait();


/*******************************abspeichern  **************/


			if (count >= numprocs - 1)
			{
std::cout << "\n -------------Warten, bis wieder ein prozess bereit ist!\n" << std::endl;
			      client_serv_data = MPI::Request::Waitany( numprocs - 1, status);
std::cout << "\n -------------Abspeichern der Normalen eine Paketes mit so viel Inhalt: " << laufvariable[client_serv_data] << "\n" << std::endl;
				progress++;
				// store normals on correct position
				for (int x = 0; x < laufvariable[client_serv_data] ; x ++)
				{
					int n_buffer_pos = 3 * Indizes[client_serv_data][x];

					m_normal[n_buffer_pos]     = normals[client_serv_data][3 * x];
					m_normal[n_buffer_pos + 1] = normals[client_serv_data][ (3 * x) + 1];
					m_normal[n_buffer_pos + 2] = normals[client_serv_data][ (3 * x) + 2];

				}
				normals[client_serv_data][0] = 0;
std::cout << "Abspeichern der Normalen abgeschlossen!" << std::endl;

				count++;
				it++;
				
				std::cout << "\n------------" << progress << " von " << m_nodelist.size() << " Paketen erledigt!\n" << std::endl; 


			}// end if
			else
			{
				count++;
				it++;
				client_serv_data++;
				client_serv_data = (client_serv_data % (numprocs - 1));
			}
		}// End while

std::cout << "\n ES wird noch auf die letzten Ergebnisse gewartet" << std::endl;
		//store all data which is still not stored
		MPI::Request::Waitall( numprocs - 1, status);

		std::cout << "\n----------- Alle Pakete erledigt! Brechnungen abgeschlossen! \n" << std::endl;
		
		client_serv_data = 0;
		for (int y = 0 ; y < numprocs - 1 ; y++)
		{

			// check if some data is in there
			if (normals[client_serv_data][0] != 0)
			{
				// store normals on correct position
				for (int x = 0; x < laufvariable[client_serv_data] ; x ++)
				{
					int n_buffer_pos = 3 * Indizes[client_serv_data][x];

					m_normal[n_buffer_pos]     = normals[client_serv_data][3 * x];
					m_normal[n_buffer_pos + 1] = normals[client_serv_data][ (3 * x) + 1];
					m_normal[n_buffer_pos + 2] = normals[client_serv_data][ (3 * x) + 2];

				}
			}
			client_serv_data++;
		}

		

		//Points put back into proper shape for PointBufferPtr
		boost::shared_array<float> norm (m_normal);

		long unsigned int tmp = static_cast<unsigned int>(m_loader->getNumPoints());


		// set normals
		m_loader->setPointNormalArray(norm, m_loader->getNumPoints() );

		m_model->m_pointCloud = m_loader;

		// save data
		io_factory.saveModel(m_model, "Normal.ply");

		
		// Complete connection
		int end[1] = {-1};
		int j = 1;
		for (j = 1 ; j < numprocs ; j++)
		{
			MPI::COMM_WORLD.Send(end , 1, MPI::INT, j, 2);
		}


		
		for (int i = 0 ; i < (numprocs -1) ; i++)
		{
			delete [] normals[i];
		}

		delete [] normals;

	}// Ende If
/**********************************************************************************************************/
	// Slave-Process
	else
	{
//für die tests zum Zeitmessen
		


		// Wait for the first Message (INIT)
		MPI::COMM_WORLD.Recv(con_msg, 128, MPI::CHAR, 0,0);

		// create answer
		sprintf(idstring, "Processor %d on ", rank);
		strcat(idstring, processor_name);
		strcat(con_msg,idstring);
		strcat(con_msg, "...roger roger, we can go on!");

		MPI::COMM_WORLD.Send(con_msg, 128, MPI::CHAR, 0, 0);

		
		// Wait for the Boundingbox
		MPI::COMM_WORLD.Recv(expansion_bounding, 6, MPI::FLOAT, 0, 5);
/************************************ Connection is successful *******************/

		// Loop for receiving the data, -1 cancels operation
		while(true)
		{
			MPI::COMM_WORLD.Recv( &c_sizepackage, 1, MPI::INT , 0,2);

			//termination condition
			if (c_sizepackage == -1)
			{
				break;
			}
			else
			{
count_serv++;

			
				// Recv the data
				float * tmp = new float[3 * c_sizepackage];
				MPI::Request client_req2 = MPI::COMM_WORLD.Irecv(tmp, 3 * c_sizepackage, MPI::FLOAT, 0, 1);

				// wait till transmission complete
				client_req2.Wait();

				// Points put back into proper shape for PointBufferPtr
				boost::shared_array<float> punkte (tmp);


				// The factory requires a model to save.
				// The model in turn requires a Pointbuffer and a Meshbuffer (can be emtpy).
				// The Pointbuffer contains the Indexlist.
				PointBufferPtr pointcloud(new PointBuffer());
				pointcloud->setPointArray(punkte, c_sizepackage);

				PointsetSurface<ColorVertex<float, unsigned char> >* surface;
				surface = new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >(pointcloud, "STANN", kn, ki, kd, ransac);
				surface->expand_bounding(expansion_bounding[0], expansion_bounding[1], expansion_bounding[2],
							      expansion_bounding[3], expansion_bounding[4], expansion_bounding[5]);
				
				// Set search options for normal estimation and distance evaluation

				
				// calculate the normals
				//Timestamp ts;
std::cout << "\n++++++++++Client " << rank << " berechnet Normale mit so vielen Punkten: " << c_sizepackage << std::endl;
				surface->calculateSurfaceNormals();
				//cerr << ts.getElapsedTimeInMs() << endl;

//				ModelPtr pn( new Model);
//				pn->m_pointCloud = surface->pointBuffer();


				pointcloud = surface->pointBuffer();
				size_t size_normal;
				c_normals = pointcloud->getIndexedPointNormalArray(size_normal);
std::cout << "\n++++++++++Client " << rank << " hat seine Brechnung abgeschlossen!" << std::endl;
				// send the normals back to the Masterprocess
				MPI::COMM_WORLD.Send(c_normals.get(), 3 * c_sizepackage, MPI::FLOAT, 0, 4);

			}
		}

	}// End else

	sprintf(test_aufgabe_name, "Ausgabe%00d.dat" , rank);
	f.open(test_aufgabe_name, ios::out);
		if (rank == 0)
	{  
	  std::cout << "so lange hat es gebraucht: " << start.getElapsedTimeInMs() << std::endl;
	  f << "Beende den Durchlauf mit " << numprocs << " Prozessen, Mit den k-Werten: " << ki << " " << kn << " dieser hat " << start.getElapsedTimeInMs() << " Millisekunden gedauert." << std::endl;
	}
	else f << "Beende den Prozess: " << rank << ", dieser hat " << count_serv << " Pakete bearbeiet. Und hat mit kd, kn und ki gearbeitet:" << kd << kn << ki << endl; 
	f.close();
	MPI_Finalize();

}
