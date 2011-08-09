/*
 * Options.h
 *
 *  Created on: Nov 21, 2010
 *      Author: Thomas Wiemann
 */

#ifndef OPTIONS_H_
#define OPTIONS_H_

#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>

using std::ostream;
using std::cout;
using std::endl;
using std::string;
using std::vector;


namespace reconstruct{

using namespace boost::program_options;

/**
 * @brief A class to parse the program options for the reconstruction
 * 		  executable.
 */
class Options {
public:

	/**
	 * @brief 	Ctor. Parses the command parameters given to the main
	 * 		  	function of the program
	 */
	Options(int argc, char** argv);
	virtual ~Options();

	/**
	 * @brief	Returns the given voxelsize
	 */
	float 	getVoxelsize()const;

	/**
	 * @brief	Returns the number of used threads
	 */
	int 	getNumThreads() const;

	/**
	 * @brief	Prints a usage message to stdout.
	 */
	bool	printUsage() const;

	/**
	 * @brief	Returns true if an output filen name was set
	 */
	bool	filenameSet() const;

	/**
	 * @brief 	Returns true if the face normals of the
	 * 			reconstructed mesh should be saved to an
	 * 			extra file ("face_normals.nor")
	 */
	bool	saveFaceNormals() const;

	/**
	 * @brief	Returns true if the interpolated normals
	 * 			should be saved in the putput file
	 */
	bool    saveNormals() const;

	/**
	 * @brief	Returns true if clustering is enabled
	 */
	bool	createClusters() const;

	/**
	 * @brief 	Returns true if cluster optimization is enabled
	 */
	bool 	optimizeClusters() const;

	/**
	 * @brief 	Indicates whether to save the used points
	 * 			together with the interpolated normals.
	 */
	bool 	savePointsAndNormals() const;

	/**
	 * @brief	If true, normals should be calculated even if
	 * 			they are already given in the input file
	 */
	bool	recalcNormals() const;

	/**
	 * @brief	Returns the number of neighbors
	 * 			for normal interpolation
	 */
	int     getKi() const;

	/**
	 * @brief	Returns the number of neighbors used for
	 * 			initial normal estimation
	 */
	int     getKn() const;

	/**
	 * @brief	Returns the number of neighbors used for distance
	 * 			function evaluation
	 */
	int     getKd() const;

	/**
	 * @brief	Returns the output file name
	 */
	string 	getInputFileName() const;

	/**
	 * @brief   Returns the number of intersections. If the return value
	 *          is positive it will be used for reconstruction instead of
	 *          absolute voxelsize.
	 */
	int     getIntersections() const;

	/**
	 * @brieg   Returns the name of the used point cloud handler.
	 */
	string getPCM() const;

private:

	/// The set voxelsize
	float 				            m_voxelsize;

	/// The number of uesed threads
	int				                m_numThreads;

	/// The internally used variable map
	variables_map			        m_variables;

	/// The internally used option description
	options_description 		    m_descr;

	/// The internally used positional option desription
	positional_options_description 	m_pdescr;

	/// The putput file name for face normals
	string 				            m_faceNormalFile;

	/// The number of used default values
	int                             m_numberOfDefaults;

	/// The number of neighbors for distance function evaluation
	int                             m_kd;

	/// The number of neighbors for normal estimation
	int                             m_kn;

	/// The number of neighbors for normal interpolation
	int                             m_ki;

	/// The number of intersections used for reconstruction
	int                             m_intersections;

	/// The used point cloud manager
	string                          m_pcm;
};


/// Overlaoeded outpur operator
inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " 	<< endl;
	if(o.getIntersections() > 0)
	{
	    cout << "##### Intersections \t\t: " << o.getIntersections() << endl;
	}
	else
	{
	    cout << "##### Voxelsize \t\t: " << o.getVoxelsize() << endl;
	}
	cout << "##### Output file \t\t: " 	<< o.getInputFileName() << endl;
	cout << "##### Number of threads \t: " << o.getNumThreads() << endl;
	cout << "##### Point cloud manager: \t: " << o.getPCM() << endl;
	cout << "##### k_n \t\t\t: " << o.getKn() << endl;
	cout << "##### k_i \t\t\t: " << o.getKi() << endl;
	cout << "##### k_d \t\t\t: " << o.getKd() << endl;
	if(o.saveFaceNormals())
	{
		cout << "##### Write Face Normals \t: YES" << endl;
	}
	if(o.createClusters())
	{
		cout << "##### Create cluster \t\t: YES" << endl;
	}
	if(o.optimizeClusters())
	{
		cout << "##### Optimize cluster \t\t: YES" << endl;
	}
	if(o.saveNormals())
	{
		cout << "##### Save normals \t\t: YES" << endl;
	}
	if(o.recalcNormals())
	{
		cout << "##### Recalc normals \t\t: YES" << endl;
	}
	if(o.savePointsAndNormals())
	{
	    cout << "##### Save points and normals \t: YES" << endl;
	}
	return os;
}

} // namespace reconstruct


#endif /* OPTIONS_H_ */
