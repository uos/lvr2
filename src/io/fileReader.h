#ifndef __FILEREADER_H__
#define __FILEREADER_H__

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <vector>

#include "../mesh/staticMesh.h"

using namespace std;

class FileReader{

public:
  FileReader(){};
  virtual ~FileReader();
  virtual void read(char* filename){};

  StaticMesh mesh;

};

#endif
