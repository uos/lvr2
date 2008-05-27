#ifndef __FILEWRITER_H_
#define __FILEWRITER_H_

#include <iostream>
#include <fstream>
#include <string.h>
#include <stdio.h>
#include <vector>

#include "../mesh/staticMesh.h"

class FileWriter{

public:
  FileWriter(char* fileName);
  virtual ~FileWriter();

  virtual void writeHeader();
  virtual void writeFooter();
  virtual void addMesh(StaticMesh &mesh);

protected:

  virtual void init(char* fileName);
  
  ofstream out;
  bool file_open;
  
};

#endif
