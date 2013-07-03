#include <iostream>
#include <vector>
#include "KdTree.hpp"
/**
 * @brief   Main entry point for the LSSR surface executable
 */
using namespace lssr;

#include "geometry/ColorVertex.hpp"
#include "io/PointBuffer.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"
#include "src/mpi/KdTree.hpp"
#include "geometry/ColorVertex.hpp"
#include "geometry/Normal.hpp"

#include "reconstruction/AdaptiveKSearchSurface.hpp"

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <mpi.h>
#include <unistd.h>

//typedef ColorVertex<float, unsigned char>      cVertex;
typedef KdTree<ColorVertex<float, unsigned char>>                        kd;
typedef Normal<float>                                   cNormal;
//typedef AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, cNormal>        akSurface;



int main(int argc, char** argv)
{
     std::cout << "!!!!!!!!!!!!!!Eigene Main gestartet!!!!!!!!!!!!!!!!!!!!" << endl;
     
    long size = atoi(argv[1]);
     
     // Anzahl an Processen
	int numprocs;
	// Rank / ID des Processes
	int rank;
	int namelen = 0;
	
	char processor_name[MPI_MAX_PROCESSOR_NAME];
	
     // Initializes the connection
	MPI::Init(argc, argv);

	// gives the number of processes
	numprocs = MPI::COMM_WORLD.Get_size();

	// gives the Id / Rang of the processes
	rank = MPI::COMM_WORLD.Get_rank();

	// returns the name of the processor (computer on which it runs)
	MPI::Get_processor_name(processor_name, namelen);
     
       double first;
       double second;

     
     if(rank == 0)
     {
            std::cout << "initialisieren der Arrays" << std::endl;
	  float * hin = new float[size];
	  float * back = new float[size];
     std::cout << "schleife folgt"  << std::endl;
	  for (long i = 0 ; i < size ; i++) hin[i] = 1;
	
	  
	  
	  Timestamp ts;
	  
	  MPI::COMM_WORLD.Send(hin, size, MPI::FLOAT, 1, 0);
 	  
	  first = ts.getElapsedTimeInMs();
	  
	  MPI::COMM_WORLD.Recv(back, size, MPI::FLOAT, 1, 1);
	  
	  second = ts.getElapsedTimeInMs();
	  
	  std::cout << "Das hin und herschicken hat so lange gedauert: " << second << std::endl;
     
          std::cout << "Die einzelne Übertragung von " << size << " Float hat gedauert: " << first << std::endl;

       
delete hin;
delete back;
    }
     else
     {
       
        float *first = new float[size];
	  
	  MPI::COMM_WORLD.Recv(first, size, MPI::FLOAT, 0, 0);
       
       
	   MPI::COMM_WORLD.Send(first, size, MPI::FLOAT, 0, 1);

       delete first;
    }
     

     
     MPI_Finalize();
     
     
     
     
     
     
     
     
     /*		// A shared-Pointer for the model, with the pointcloud in it
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
	
	ModelPtr model = io_factory.readModel( "horncolor_filtered.ply" );
	PointBufferPtr p_loader;
		// Parse loaded data
	if ( !model )
	{
		exit(-1);
	}
	p_loader = model->m_pointCloud;
		// Create a point cloud manager
	PointsetSurface<ColorVertex<float, unsigned char> >* surface;
	surface = new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >(p_loader, "STANN", 500, 500, 500, false);	
	

	surface->calculateSurfaceNormals();
	cerr << ts.getElapsedTimeInMs() << endl;
	
	ModelPtr pn( new Model);
	pn->m_pointCloud = surface->pointBuffer();
	ModelFactory::saveModel(pn, "pointnormals.ply");
	
	
*/
}
