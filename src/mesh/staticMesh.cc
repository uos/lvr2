#include "staticMesh.h"

StaticMesh::StaticMesh(){};

void StaticMesh::render(){
  glBegin(GL_TRIANGLES);
  for(size_t i = 0; i < indices.size(); i++){
    setColorMaterial(vertices[indices[i]].r,
				 vertices[indices[i]].g,
				 vertices[indices[i]].b);
				 
    normals[indices[i]].render();
    vertices[indices[i]].render();
  }
  glEnd();
}

void StaticMesh::interpolateNormal(int index, Normal n){

  if(index < (int)normals.size()){
    normals[index] += n;
  } else {
    cout << "Static Mesh::interpolateNormal: Warning: Normal index above array bounds "
	    << index << " / " << normals.size() << endl;
  }
  
}

void StaticMesh::setColorMaterial(uchar _r, uchar _g, uchar _b){

  //cout << "R: " << r << " G: " << g << " B: " << b << endl;

  float r = _r / 255.0;
  float g = _g / 255.0;
  float b = _b / 255.0;

  float mat_specular[4];
  float mat_ambient[4];
  float mat_diffuse[4];

  float mat_shininess = 50;
  
  mat_specular[0] = 0.7f; mat_ambient[0]  = 0.5f * r; mat_diffuse[0]  = r;
  mat_specular[1] = 0.7f; mat_ambient[1]  = 0.5f * g; mat_diffuse[1]  = g;
  mat_specular[2] = 0.7f; mat_ambient[2]  = 0.5f * b; mat_diffuse[2]  = b;
  mat_specular[3] = 1.0f; mat_ambient[3]  = 1.0f; mat_diffuse[3]  = 1.0f; 
  
  glMaterialfv(GL_FRONT, GL_SPECULAR,  mat_specular);
  glMaterialfv(GL_FRONT, GL_SHININESS, &mat_shininess);
  glMaterialfv(GL_FRONT, GL_AMBIENT,   mat_ambient);
  glMaterialfv(GL_FRONT, GL_DIFFUSE,   mat_diffuse); 
  
}
