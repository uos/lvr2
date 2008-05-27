#ifndef __PLYWRITER_H__
#define __PLYWRITER_H__

#include "../mesh/colorVertex.h"
#include "../mesh/staticMesh.h"

#include "fileWriter.h"
#include "plyDataTypes.h"


class PLYWriter : public FileWriter{

public:
  PLYWriter(char* filename);

  void writeHeader();
  void writeFooter();
  void addMesh(StaticMesh &mesh);
 
  void setComment(char* c);
  
protected:
  void init(char* filename);

private:

  char* comment;
  
  PlyHeaderDescription header_dcr;
  PlyVertexDescription vertex_dcr;
  PlyFaceDescription face_dcr;
  
};



#endif
