/*
 * PointCloud.cpp
 *
 *  Created on: 02.09.2008
 *      Author: twiemann
 */

#include "PointCloud.h"

#include <string.h>

PointCloud::PointCloud()
{
}

PointCloud::PointCloud(string filename) : Renderable(filename) {

    ifstream in(filename.c_str());

    m_boundingBox = new BoundingBox;

    if ( !in.good() ) {
        cout << "##### Error: Could not open file " << filename << "." << endl;
        return;
    }

    /* Get number of data fields to ignore */
    int number_of_dummys = getFieldsPerLine(filename) - 3;
    int c = 0;

	 /* Use last three elements as color informations */
	 int read_color = ( number_of_dummys >= 3 );
	 
	 if ( read_color ) {
		 number_of_dummys -= 3;
	 }

    //Point coordinates
    float x, y, z, dummy;
	 unsigned int r, g, b;

    // Read file
    while ( in.good() ) {
		 /* Read y, x, z from first three values. */
    	in >> x >> y >> z;

		/* Ignore reflection, … */
    	for ( int i = 0; i < number_of_dummys; i++ ) {
    		in >> dummy;
    	}
		
		/* Read colors from last three values. */
		if ( read_color ) {
			in >> r >> g >> b;
			ColorVertex v( x, y, -z, (uchar) r, (uchar) g, (uchar) b );
			m_boundingBox->expand( v );
			points.push_back( v );
		} else {
			m_boundingBox->expand( x, y, -z );
			points.push_back( ColorVertex( x, y, -z ) );
		}

    	if ( ++c % 100000 == 0 ) {
			cout << "##### Reading Points... " << c << endl;
		}
    }

    cout << "Loaded Points: " << points.size() << endl;

    //initDisplayList();
}

void PointCloud::initDisplayList(){

    if(listIndex != -1) {
        cout<<"PointCloud::initDisplayList() delete display list"<<endl;
        glDeleteLists(listIndex,1);
    }
//    cout<<"PointCloud::initDisplayList() init display list "<<points.size()<<endl;
//    listIndex = glGenLists(1);
//    glNewList(listIndex, GL_COMPILE);
    glBegin(GL_POINTS);
    for(size_t i = 0; i < points.size(); i++){
        glVertex3f(points[i].x,
                   points[i].y,
                   points[i].z);
    }
    glEnd();
//    glEndList();
}

int PointCloud::getFieldsPerLine(string filename)
{
	ifstream in(filename.c_str());

	//Get first line from file
	char first_line[1024];
	in.getline(first_line, 1024);
	in.close();

	//Get number of blanks
	int c = 0;
	char* pch = strtok(first_line, " ");
	while(pch != NULL){
		c++;
		pch = strtok(NULL, " ");
	}

	in.close();

	return c;
}

PointCloud::~PointCloud() {
    // TODO Auto-generated destructor stub
}
