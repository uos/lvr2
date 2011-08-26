/*
 * PolygonMesh.cpp
 *
 *  Created on: 03.02.2009
 *      Author: twiemann
 */

#include "PolygonMesh.h"

void vertexCallback(GLvoid *vertex)
{
    GLdouble *ptr;
    ptr = (GLdouble *) vertex;
    glColor4dv((GLdouble *) ptr + 3);
    glVertex3dv((GLdouble *) ptr);
}

PolygonMesh::PolygonMesh(string filename) : Renderable(filename){
    load(filename);
    //drawFilled = false;
    compileDisplayList();
}

void PolygonMesh::load(string filename){

    char line[1024];
    Polygon p;

    ifstream in(filename.c_str());

    if(!in.good()){
        cout << "Warning: PolygonMesh::load(): Could not open file '"
             << filename << "'." << endl;
        return;
    }

    float x,y,z;
    float r,g,b;

    int c = 0;
    while(in.good()){
        in.getline(line, 1024);
        if(strstr(line, "BEGIN") != NULL){
            if(c % 1000 == 0){
                cout << "PolygonMesh: Read " << c << " polygons." << endl;
            }
            p = Polygon();
            c++;

        } else if(strstr(line, "END") != NULL) {
            polygons.push_back(p);
        } else {
            x = y = z = 0.0;
            r = g = b = 0.0;

            sscanf(line, "%f %f %f %f %f %f", &x, &y, &z, &r, &g, &b);
            p.addVertex(Vertex(x, y, z));
        }
    }

}

void PolygonMesh::save(string filename){

    cout << "TO DO: Implement PolygonMesh::save()!!!" << endl;

}

void PolygonMesh::compileDisplayList(){

    Polygon p;

    m_listIndex = glGenLists(1);
    glNewList(m_axesListIndex, GL_COMPILE);

    glDisable(GL_LIGHTING);
    glColor3f(0.0, 0.0, 0.0);
    
    list<Polygon>::iterator i;
    for(i=polygons.begin(); i != polygons.end(); ++i) {
			p = *i;
			glBegin(GL_LINES);
      for(size_t j = 0; j < p.vertices.size(); j++){
      	glVertex3f(p.vertices[j].x, p.vertices[j].y, p.vertices[j].z);
      }
      glEnd();
    }
    
//    for(size_t i = 0; i < polygons.size(); i++){
//        p = polygons[i];
//        glBegin(GL_LINES);
//        for(size_t j = 0; j < p.vertices.size(); j++){
//            glVertex3f(p.vertices[j].x, p.vertices[j].y, p.vertices[j].z);
//        }
//        glEnd();
//    }
    glEnable(GL_LIGHTING);
    glEndList();

}

PolygonMesh::~PolygonMesh() {
    // TODO Auto-generated destructor stub
}

bool comparePolygons(Polygon a, Polygon b) {
	double aZ = 0.0 , bZ = 0.0;
	for (size_t j = 0; j < a.vertices.size(); j++) {
		if (a.vertices[j].z > aZ) {
			aZ = a.vertices[j].z;
		} 
	}
	
	for (size_t j = 0; j < b.vertices.size(); j++) {
		if (b.vertices[j].z > aZ) {
			bZ = b.vertices[j].z;
		} 
	}
	
	return aZ < bZ; 
}
