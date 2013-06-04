

// Program options for this tool
#include "Options.hpp"

// Local includes
#include "reconstruction/AdaptiveKSearchSurface.hpp"
#include "reconstruction/FastReconstruction.hpp"
#include "io/PLYIO.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/HalfEdgeMesh.hpp"
#include "texture/Texture.hpp"
#include "texture/Transform.hpp"
#include "texture/Texturizer.hpp"
#include "texture/Statistics.hpp"
#include "geometry/QuadricVertexCosts.hpp"
#include "reconstruction/SharpBox.hpp"

// PCL related includes
#ifdef _USE_PCL_
#include "reconstruction/PCLKSurface.hpp"
#endif


#include <iostream>

using namespace lssr;

typedef ColorVertex<float, unsigned char>               cVertex;
typedef Normal<float>                                   cNormal;
typedef PointsetSurface<cVertex>                        psSurface;
typedef AdaptiveKSearchSurface<cVertex, cNormal>        akSurface;

#ifdef _USE_PCL_
typedef PCLKSurface<cVertex, cNormal>                   pclSurface;
#endif

/**
 * @brief   Main entry point for the LSSR surface executable
 */
int main(int argc, char** argv)
{

	try
	{
		// Parse command line arguments
		reconstruct::Options options(argc, argv);

		// Exit if options had to generate a usage message
		// (this means required parameters are missing)
		if ( options.printUsage() )
		{
			return 0;
		}

		omp_set_num_threads(4);

		::std::cout << options << ::std::endl;



		// Create a point loader object
		ModelFactory io_factory;
		ModelPtr model = io_factory.readModel( options.getInputFileName());
		PointBufferPtr p_loader;

		// Parse loaded data
		if ( !model )
		{
			cout << timestamp << "IO Error: Unable to parse die Flur-kacke" << endl;
			exit(-1);
		}
		p_loader = model->m_pointCloud;

		// Create a point cloud manager
		string pcm_name = options.getPCM();
		psSurface::Ptr surface;

		// Create point set surface object
		if(pcm_name == "PCL")
		{
#ifdef _USE_PCL_
			surface = psSurface::Ptr( new pclSurface(p_loader));
#else 
			cout << timestamp << "Can't create a PCL point set surface without PCL installed." << endl;
			exit(-1);
#endif
		}
		else if(pcm_name == "STANN" || pcm_name == "FLANN" || pcm_name == "NABO" || pcm_name == "NANOFLANN")
		{
			akSurface* aks = new akSurface(
					p_loader, pcm_name,
					options.getKn(),
					options.getKi(),
					options.getKd()
			);

			surface = psSurface::Ptr(aks);
			// Set RANSAC flag
			if(options.useRansac())
			{
				aks->useRansac(true);
			}
		}
		else
		{
			cout << timestamp << "Unable to create PointCloudManager." << endl;
			cout << timestamp << "Unknown option '" << pcm_name << "'." << endl;
			cout << timestamp << "Available PCMs are: " << endl;
			cout << timestamp << "STANN, STANN_RANSAC";
#ifdef _USE_PCL_
			cout << ", PCL";
#endif
#ifdef _USE_NABO
			cout << ", Nabo";
#endif
			cout << endl;
			return 0;
		}

		// Set search options for normal estimation and distance evaluation
		surface->setKd(options.getKd());
		surface->setKi(options.getKi());
		surface->setKn(options.getKn());

		// Calculate normals if necessary
		if(!surface->pointBuffer()->hasPointNormals()
				|| (surface->pointBuffer()->hasPointNormals() && options.recalcNormals()))
		{
		    Timestamp ts;
			surface->calculateSurfaceNormals();
			cerr << ts.getElapsedTimeInMs() << endl;
		}
		else
		{
			cout << timestamp << "Using given normals." << endl;
		}

		// Save points and normals only
		if(options.savePointNormals())
		{
			ModelPtr pn( new Model);
			pn->m_pointCloud = surface->pointBuffer();
			ModelFactory::saveModel(pn, "pointnormals.ply");
		}

// <--------------------------------------------------------------
		// Create an empty mesh
		HalfEdgeMesh<cVertex, cNormal> mesh( surface );

		// Set recursion depth for region growing
		if(options.getDepth())
		{
			mesh.setDepth(options.getDepth());
		}

		if(options.getTexelSize())
		{
			Texture::m_texelSize = options.getTexelSize();
		}
		
		if(options.getTexturePack() != "")
		{
			Texturizer<cVertex, cNormal>::m_filename = options.getTexturePack();
			if(options.getStatsCoeffs())
			{	
				float* sc = options.getStatsCoeffs();
				for (int i = 0; i < 14; i++)
				{
					Statistics::m_coeffs[i] = sc[i];
				}
				delete sc;
			}
			if(options.getNumStatsColors())
			{
				Texturizer<cVertex, cNormal>::m_numStatsColors = options.getNumStatsColors();
			}
			if(options.getNumCCVColors())
			{
				Texturizer<cVertex, cNormal>::m_numCCVColors = options.getNumCCVColors();
			}
			if(options.getCoherenceThreshold())
			{
				Texturizer<cVertex, cNormal>::m_coherenceThreshold = options.getCoherenceThreshold();
			}

			if(options.getColorThreshold())
			{
				Texturizer<cVertex, cNormal>::m_colorThreshold = options.getColorThreshold();
			}
			if(options.getStatsThreshold())
			{
				Texturizer<cVertex, cNormal>::m_statsThreshold = options.getStatsThreshold();
			}
			if(options.getUseCrossCorr())
			{
				Texturizer<cVertex, cNormal>::m_useCrossCorr = options.getUseCrossCorr();
			}
			if(options.getFeatureThreshold())
			{
				Texturizer<cVertex, cNormal>::m_featureThreshold = options.getFeatureThreshold();
			}
			if(options.getPatternThreshold())
			{
				Texturizer<cVertex, cNormal>::m_patternThreshold = options.getPatternThreshold();
			}
			if(options.doTextureAnalysis())
			{
			    Texturizer<cVertex, cNormal>::m_doAnalysis = true;
			}
			if(options.getMinimumTransformationVotes())
			{
				Transform::m_minimumVotes = options.getMinimumTransformationVotes();
			}
		}

// <-----------------------pos2
		if(options.getSharpFeatureThreshold())
		{
			SharpBox<cVertex, cNormal>::m_theta_sharp = options.getSharpFeatureThreshold();
		}
		if(options.getSharpCornerThreshold())
		{
			SharpBox<cVertex, cNormal>::m_phi_corner = options.getSharpCornerThreshold();
		}

		// Determine whether to use intersections or voxelsize
		float resolution;
		bool useVoxelsize;
		if(options.getIntersections() > 0)
		{
			resolution = options.getIntersections();
			useVoxelsize = false;
		}
		else
		{
			resolution = options.getVoxelsize();
			useVoxelsize = true;
		}
		// <---- pos 3
		// ------------------------------- SPEICHER LECK HIER ---------------------------
		// Create a new reconstruction object
		


		FastReconstruction<cVertex, cNormal > reconstruction(
				surface,
				resolution,
				useVoxelsize,
				options.getDecomposition(),
				options.extrude());
		
		
		// ------------------------------- SPEICHER LECK HIER ENDE ---------------------------

		
		//<-------- pos 4 zweites Speicherleck
		// Create mesh
		reconstruction.getMesh(mesh);
		


		// Save grid to file
		if( options.saveGrid())
		{
			reconstruction.saveGrid("fastgrid.grid");
		}



		if(options.getDanglingArtifacts())
		{
			mesh.removeDanglingArtifacts(options.getDanglingArtifacts());
		}
		// Optimize mesh




		mesh.cleanContours(options.getCleanContourIterations());


		mesh.setClassifier(options.getClassifier());
		// <---------------------------------------pos 1
		if(options.optimizePlanes())
		{
			mesh.optimizePlanes(options.getPlaneIterations(),
					options.getNormalThreshold(),
					options.getMinPlaneSize(),
					options.getSmallRegionThreshold(),
					true);

			mesh.fillHoles(options.getFillHoles());

			mesh.optimizePlaneIntersections();

			mesh.restorePlanes(options.getMinPlaneSize());

			if(options.getNumEdgeCollapses())
			{
				QuadricVertexCosts<cVertex, cNormal> c = QuadricVertexCosts<cVertex, cNormal>(true);
				mesh.reduceMeshByCollapse(options.getNumEdgeCollapses(), c);
			}


		}
		else if(options.clusterPlanes())
		{
			mesh.clusterRegions(options.getNormalThreshold(), options.getMinPlaneSize());
			mesh.fillHoles(options.getFillHoles());
		}

		// Save triangle mesh
		if ( options.retesselate() )
		{
			mesh.finalizeAndRetesselate(options.generateTextures(),
					options.getLineFusionThreshold());
		}
		else
		{
			mesh.finalize();
		}

		// Create output model and save to file
		ModelPtr m( new Model( mesh.meshBuffer() ) );
		
		if(options.saveOriginalData())
		{
			m->m_pointCloud = model->m_pointCloud;
		}
		cout << timestamp << "Saving mesh." << endl;
		ModelFactory::saveModel( m, "triangle_mesh.ply");

		// Save obj model if textures were generated
		if(options.generateTextures())
		{
			ModelFactory::saveModel( m, "triangle_mesh.obj");
		}

		
		cout << timestamp << "Program end." << endl;

	}
	catch(...)
	{
		std::cout << "Unable to parse options. Call 'reconstruct --help' for more information." << std::endl;
	}

	return 0;
}

