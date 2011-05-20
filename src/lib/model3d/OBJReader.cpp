#include "OBJReader.h"

ObjReader::ObjReader(string filename){

  mesh = new TriangleMesh;
  
  ifstream in(filename.c_str());

  if(!in.good()) {
	cout << "ObjReader: Could not open file '" << filename << "'." << endl; 
	return;
  }

  char buffer[1024];
  
  while(!in.eof()){

    in.getline(buffer, 1024);
    string s(buffer);

    size_t pos = s.find_first_of(" ");

    string cmd = s.substr(0, pos);
    string data = s.substr(pos + 1, s.length() - pos - 1);

    if(cmd == "v") parseVertex(data);
    if(cmd == "vn") parseNormal(data);
    if(cmd == "f")  parseFace(data);
  }
  
}

void ObjReader::parseVertex(string s){

  float x, y, z;

  char* c_string = new char[s.size() + 1];
  strcpy(c_string, s.c_str());
  sscanf(c_string, "%f %f %f", &x, &y, &z);

  Vertex vertex(x, y, z);
  mesh->addVertex(vertex);
  delete[] c_string;
}

void ObjReader::parseNormal(string s){

  float x, y, z;  

  char* c_string = new char[s.size() + 1];
  strcpy(c_string, s.c_str());
  sscanf(c_string, "%f %f %f", &x, &y, &z);
  Normal normal(x, y, z);
  mesh->addNormal(normal);
  delete[] c_string;
  
}

void ObjReader::parseFace(string s){

  int a, b, c, dummy;
  a = b = c = dummy = 0;

  char* c_string = new char[s.size() + 1];
  strcpy(c_string, s.c_str());
  sscanf(c_string, "%d//%d %d//%d %d//%d", &a, &dummy, &b, &dummy, &c, &dummy);
  
  mesh->addTriangle(a - 1, b - 1, c - 1); //Numbering in OBJ-File starts with 1
  delete[] c_string;
  
}

ObjReader::~ObjReader(){
}
