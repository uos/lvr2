/*
 * PLYELement.h
 *
 *  Created on: 25.11.2009
 *      Author: twiemann
 */

#ifndef PLYELEMENT_H_
#define PLYELEMENT_H_

#include "PLYProperty.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>

using std::vector;
using std::cout;
using std::endl;
using std::string;
using std::ofstream;

class PLYElement {
public:

	PLYElement();
	PLYElement(string name, size_t count);

	void addProperty(string name, string value_type, string count_type = "");

	vector<Property*>::iterator getFirstProperty();
	vector<Property*>::iterator getLastProperty();

	string getName();
	size_t getCount();

	virtual ~PLYElement();

	void printProperties(ofstream &str);

private:

	vector<Property*> m_properties;
	string 	m_name;
	size_t	m_count;


};

#endif /* PLYELEMENT_H_ */
