#ifndef __STATIC_MESH_H__
#define __STATIC_MESH_H__

#include "colorVertex.h"
#include "normal.h"

#include <vector>
using namespace std;

class StaticMesh{

public:
  StaticMesh();
  StaticMesh(const StaticMesh& o);

  void addVertex(ColorVertex v){
    vertices.push_back(v);
  };
  void addNormal(Normal n){
    normals.push_back(n);
  }

  void interpolateNormal(int index, Normal n);
  void setColorMaterial(uchar r, uchar g, uchar b);
  void addFace(int a, int b, int c){
    indices.push_back(a);
    indices.push_back(b);
    indices.push_back(c);
  };
  
  void addIndex(int i) { indices.push_back(i);};

  void render();

  //private:
  vector<ColorVertex> vertices;
  vector<Normal> normals;
  vector<int> indices;
  
};


#endif
