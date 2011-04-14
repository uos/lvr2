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

namespace lssr{

using namespace boost::program_options;

class Options {
public:
	Options(int argc, char** argv);
	virtual ~Options();

	float 	getVoxelsize()const;
	int 	getNumThreads() const;

	bool	printUsage() const;
	bool	filenameSet() const;
	bool	saveFaceNormals() const;
	bool    saveNormals() const;
	bool	createClusters() const;
	bool 	optimizeClusters() const;
	bool 	savePointsAndNormals() const;
	bool	recalcNormals() const;

	int     getKi() const;
	int     getKn() const;
	int     getKd() const;

	string 	getOutputFileName() const;
private:
	float 				            m_voxelsize;
	int				                m_numThreads;
	variables_map			        m_variables;
	options_description 		    m_descr;
	positional_options_description 	m_pdescr;
	string 				            m_faceNormalFile;
	int                             m_numberOfDefaults;
	int                             m_kd;
	int                             m_kn;
	int                             m_ki;
};

inline ostream& operator<<(ostream& os, const Options &o)
{
	cout << "##### Program options: " 	<< endl;
	cout << "##### Voxelsize \t\t: " 		<< o.getVoxelsize() << endl;
	cout << "##### Output file \t\t: " 	<< o.getOutputFileName() << endl;
	cout << "##### Number of threads \t: " << o.getNumThreads() << endl;
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

} // namespace lssr

#endif /* OPTIONS_H_ */
