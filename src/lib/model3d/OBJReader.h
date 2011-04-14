#ifndef __OBJREADER_H__
#define __OBJREADER_H__

#include <iostream>
#include <fstream>
#include <string>

#include "TriangleMesh.h"

using namespace std;

class ObjReader{

public:
  ObjReader(string);
  ~ObjReader();

  TriangleMesh* getMesh(){return mesh;};
    
private:

  void parseVertex(string);
  void parseNormal(string);
  void parseFace(string);
  
  TriangleMesh* mesh;
  
};

#endif
