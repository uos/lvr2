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

using namespace boost::program_options;

namespace lssr{

/**
 * @brief	A helper class to parse the given command
 * 			line parameters.
 */
class Options {
public:

	/**
	 * @brief	Creates a new options instance based on the
	 * 			provided command line arguments as given to the
	 * 			main function.
	 *
	 * @param	argc		The number of given arguments
	 * @param	argv		An array of command line parameters
	 */
	Options(int argc, char** argv);

	/**
	 * @brief	Destructor
	 */
	virtual ~Options();

	/**
	 * @brief	Returns the provided voxelsize of the grid used
	 * 			by the surface reconstruction algorithm
	 */
	float 	getVoxelsize()		const;

	/**
	 * @brief	Returns the number of threads used.
	 */
	float 	getNumThreads() 	const;

	/**
	 * @brief	Performs a simple test if all needed non-default
	 * 			parameters (e.g. an input file) are given. If
	 * 			something is missing or the 'help' paramters is
	 * 			set, a usage message is printed.
	 *
	 * @return	True, if all needed parameters are given.
	 */
	bool	printUsage() 		const;

	/**
	 * @brief	Returns true if an input filename is given.
	 */
	bool	filenameSet() 		const;

	/**
	 * @brief	Returns true if the face normals of each triangle
	 * 			in mesh should be saved to file.
	 */
	bool	saveFaceNormals() 	const;

	/**
	 * @brief	Returns true if clustering should be applied.
	 */
	bool	createClusters() 	const;

	/**
	 * @brief	If true, cluster optimization is enabled
	 */
	bool 	optimizeClusters() 	const;

	/**
	 * @brief	If true, the calculated point normals for the given
	 * 			input file should be saved.
	 */
	bool 	saveNormals() 		const;

	/**
	 * @brief	If true, normals are calculated even if the input
	 * 			file contains normal definitions.
	 * @return
	 */
	bool	recalcNormals() 	const;


	/**
	 * @brief	Returns the name of the specified output file name
	 */
	string 	getOutputFileName() const;

private:

	/// The voxelsize of the reconstruction grid
	float 							m_voxelsize;

	/// The number of used threads
	int								m_numThreads;

	///	The map containing the given command line variables
	variables_map					m_variables;

	///	An option descriptor object (see boost::program_options)
	options_description 			m_descr;

	/// An descriptor object for positional options
	positional_options_description 	m_pdescr;

	/// The output file name for face nromals
	string 							m_faceNormalFile;

	/// The number of given default arguments
	int                             m_numberOfDefaults;
};

/**
 * @brief	Output operator. Prints all enable features.
 *
 * @param 	os		The used output stream
 * @param	o 		An options object
 *
 * @return	The modified output stream.
 */
inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " 	<< endl;
	cout << "##### Voxelsize \t\t: " 		<< o.getVoxelsize() << endl;
	cout << "##### Output file \t\t: " 	<< o.getOutputFileName() << endl;
	cout << "##### Number of threads \t: " << o.getNumThreads() << endl;
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
	return os;
}

} //namespace lssr
#endif /* OPTIONS_H_ */
