/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * STLIO.cpp
 *
 *  Created on: Dec 9, 2016
 *      Author: robot
 */


#include "lvr2/io/STLIO.hpp"
#include "lvr2/io/Timestamp.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Normal.hpp"

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
			Vec v1;
			Vec v2;
			Vec v3;

			v1.x = vertices[3 * a];
			v1.y = vertices[3 * a + 1];
			v1.z = vertices[3 * a + 2];

			v2.x = vertices[3 * b];
			v2.y = vertices[3 * b + 1];
			v2.z = vertices[3 * b + 2];

			v3.x = vertices[3 * c];
			v3.y = vertices[3 * c + 1];
			v3.z = vertices[3 * c + 2];

			Normal<float> normal( (v1 - v2).cross(v1 - v3));

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
