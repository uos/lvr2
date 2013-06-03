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


//typedef ColorVertex<float, unsigned char>      cVertex;
typedef KdTree<ColorVertex<float, unsigned char>>                        kd;
typedef Normal<float>                                   cNormal;
//typedef AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, cNormal>        akSurface;



int main(int argc, char** argv)
{
	std::cout << "!!!!!!!!!!!!!!Eigene Main gestartet!!!!!!!!!!!!!!!!!!!!" << endl;
     Timestamp ts;
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
	
	ModelPtr model = io_factory.readModel( "polizei30M_cut.ply" );
	PointBufferPtr p_loader;
		// Parse loaded data
	if ( !model )
	{
		exit(-1);
	}
	p_loader = model->m_pointCloud;
		// Create a point cloud manager
	PointsetSurface<ColorVertex<float, unsigned char> >* surface;
	surface = new AdaptiveKSearchSurface<ColorVertex<float, unsigned char>, Normal<float> >(p_loader, "STANN", 500, 500, 500, true);	
	

	surface->calculateSurfaceNormals();
	cerr << ts.getElapsedTimeInMs() << endl;
	
	ModelPtr pn( new Model);
	pn->m_pointCloud = surface->pointBuffer();
	ModelFactory::saveModel(pn, "pointnormals.ply");
	
	
//aks->useRansac(true);
	
	std::cout << "!!!!!!!!!!!!!!Programm ist durchgelaufen, scan1 bis n müssten zur Verfuegung stehen!!!!!!!!!!!!!!!!!!!!" << endl;
}
