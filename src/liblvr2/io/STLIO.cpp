/*
 * STLIO.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: robot
 */


#include <lvr2/io/STLIO.hpp>
#include <lvr2/io/Timestamp.hpp>
#include <lvr2/geometry/BaseVector.hpp>
#include <lvr2/geometry/Vector.hpp>
#include <lvr2/geometry/Normal.hpp>

#include <iostream>
#include <cstdio>
#include <cstring>
#include <stdint.h>
#include <fstream>

using std::cout;
using std::endl;

namespace lvr2
{

STLIO::STLIO() {
	// TODO Auto-generated constructor stub

}

STLIO::~STLIO() {
	// TODO Auto-generated destructor stub
}

ModelPtr STLIO::read(string filename)
{
	return ModelPtr(new Model);
}

void STLIO::save( string filename )
{
	save(this->m_model, filename);
}

void STLIO::save( ModelPtr model, string filename )
{

	MeshBufferPtr mesh = model->m_mesh;
	size_t n_vert = mesh->numVertices();
	size_t n_faces = mesh->numFaces();
	floatArr vertices = mesh->getVertices();
	indexArray indices = mesh->getFaceIndices();

	std::string header_info = "Created by LVR";
	char head[80];
	std::strncpy(head,header_info.c_str(),sizeof(head)-1);
	char attribute[2] = "0";

	std::ofstream myfile(filename.c_str());

	myfile.write(head,sizeof(head));
	myfile.write((char*)&n_faces,4);

	if(myfile.good())
	{
		for(int i = 0; i < n_faces; i++)
		{
			int a = (int)indices[3 * i];
			int b = (int)indices[3 * i + 1];
			int c = (int)indices[3 * i + 2];

            using Vec = BaseVector<float>;
			Vector<Vec> v1;
			Vector<Vec> v2;
			Vector<Vec> v3;

			v1.x = vertices[3 * a];
			v1.y = vertices[3 * a + 1];
			v1.z = vertices[3 * a + 2];

			v2.x = vertices[3 * b];
			v2.y = vertices[3 * b + 1];
			v2.z = vertices[3 * b + 2];

			v3.x = vertices[3 * c];
			v3.y = vertices[3 * c + 1];
			v3.z = vertices[3 * c + 2];

			Normal<Vec> normal( (v1 - v2).cross(v1 - v3));

			myfile.write( (char*)&normal.x, 4);
			myfile.write( (char*)&normal.y, 4);
			myfile.write( (char*)&normal.z, 4);

			myfile.write( (char*)&v1.x, 4);
			myfile.write( (char*)&v1.y, 4);
			myfile.write( (char*)&v1.z, 4);

			myfile.write( (char*)&v2.x, 4);
			myfile.write( (char*)&v2.y, 4);
			myfile.write( (char*)&v2.z, 4);

			myfile.write( (char*)&v3.x, 4);
			myfile.write( (char*)&v3.y, 4);
			myfile.write( (char*)&v3.z, 4);

			uint16_t u = 0;
			myfile.write( (char*)attribute, 2 );
		}
	}
	else
	{
		cout << timestamp << "Could not open file " << filename << " for writing." << endl;
 	}
}

} /* namespace lvr2 */
