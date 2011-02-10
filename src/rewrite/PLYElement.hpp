/*
 * PLYELement.h
 *
 *  Created on: 25.11.2009
 *      Author: Thomas Wiemann
 */

#ifndef PLYELEMENT_H_
#define PLYELEMENT_H_

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::ofstream;

namespace lssr
{

class PLYProperty;

class PLYElement {
public:

	PLYElement();
	PLYElement(string name, size_t count);

	void addProperty(string name, string value_type, string count_type = "");

	vector<PLYProperty*>::iterator getFirstProperty();
	vector<PLYProperty*>::iterator getLastProperty();

	string getName();
	size_t getCount();

	virtual ~PLYElement();

	void printProperties(ofstream &str);

private:

	vector<PLYProperty*> m_properties;
	string 	m_name;
	size_t	m_count;


};

} // namespace lssr

#endif /* PLYELEMENT_H_ */
