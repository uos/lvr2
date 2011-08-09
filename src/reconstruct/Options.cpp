/*
 * Options.cpp
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#include "Options.hpp"

namespace reconstruct{

Options::Options(int argc, char** argv) : m_descr("Supported options")
{

	// Create option descriptions

	m_descr.add_options()
		("help", "Produce help message")
		("inputFile", value< vector<string> >(), "Input file name. Supported formats are ASCII (.pts, .xyz) and .ply")
		("voxelsize,v", value<float>(&m_voxelsize)->default_value(10), "Voxelsize of grid used for reconstruction.")
		("saveFaceNormals", "Writes all interpolated triangle normals together with triangle centroid to a file called 'face_normals.nor'")
		("cluster,c", "Extract planes and write result to 'planes.ply'")
		("optimizeCluster,o", "Shift all triangle vertices of a cluster onto their shared plane")
		("savePointsAndNormals,s", "Exports original point cloud data together with normals into a single file called 'points_and_normals.ply'")
		("recalcNormals,r", "Always estimate normals, even if given in .ply file.")
		("threads,t", value<int>(&m_numThreads)->default_value(4), "Number of threads")
		("saveNormals", "Writes all points and interpolated normals to a file called 'normals.nor'")
		("kd", value<int>(&m_kd)->default_value(5), "Number of normals used for distance function evaluation")
	    ("ki", value<int>(&m_ki)->default_value(10), "Number of normals used in the normal interpolation process")
	    ("kn", value<int>(&m_kn)->default_value(10), "Size of k-neighborhood used for normal estimation")
	    ("intersections,i", value<int>(&m_intersections)->default_value(-1), "Number of intersections used for reconstruction. If other than -1, voxelsize will calculated automatically.")
	    ("pcm,p", value<string>(&m_pcm)->default_value("STANN"), "Point cloud manager used for point handling and normal estimation. Choose from {STANN, PCL}.")
		;

	m_pdescr.add("inputFile", -1);

	// Parse command line and generate variables map
	store(command_line_parser(argc, argv).options(m_descr).positional(m_pdescr).run(), m_variables);
	notify(m_variables);

  if(m_variables.count("help")) {
    ::std::cout<< m_descr << ::std::endl;
    exit(-1);
  }


}

float Options::getVoxelsize() const
{
	return m_variables["voxelsize"].as<float>();
}

int Options::getNumThreads() const
{
	return m_variables["threads"].as<int>();
}

int Options::getKi() const
{
    return m_variables["ki"].as<int>();
}

int Options::getKd() const
{
    return m_variables["kd"].as<int>();
}

int Options::getKn() const
{
    return m_variables["kn"].as<int>();
}

int Options::getIntersections() const
{
    return m_variables["intersections"].as<int>();
}

string Options::getInputFileName() const
{
	return (m_variables["inputFile"].as< vector<string> >())[0];
}

string Options::getPCM() const
{
    return (m_variables["pcm"].as< string >());
}

bool Options::printUsage() const
{
  if(!m_variables.count("inputFile"))
    {
      cout << "Error: You must specify an input file." << endl;
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
  
  if(m_variables.count("help"))
    {
      cout << endl;
      cout << m_descr << endl;
      return true;
    }
  return false;
}

bool Options::saveFaceNormals() const
{
	return m_variables.count("saveFaceNormals");
}

bool Options::filenameSet() const
{
	return (m_variables["inputFile"].as< vector<string> >()).size() > 0;
}

bool Options::createClusters() const
{
	return (m_variables.count("cluster"));
}

bool Options::recalcNormals() const
{
	return (m_variables.count("recalcNormals"));
}

bool Options::savePointsAndNormals() const
{
	return (m_variables.count("savePointsAndNormals"));
}

bool Options::saveNormals() const
{
    return (m_variables.count("saveNormals"));
}

bool Options::optimizeClusters() const
{
	return createClusters() && m_variables.count("optimizeCluster");
}

Options::~Options() {
	// TODO Auto-generated destructor stub
}

} // namespace reconstruct
