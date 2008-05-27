#ifndef __PLYREADER_H__
#define __PLYREADER_H__

#include "fileReader.h"
#include "plyDataTypes.h"

class PLYReader: public FileReader{

public:

  PLYReader(){};
  ~PLYReader(){};
  
  void read(char* filename);
  
};

#endif
