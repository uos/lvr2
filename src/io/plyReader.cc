#include "plyReader.h"

void PLYReader::read(char* filename){
  
  ifstream in;
  
  PlyHeaderDescription head;
  PlyVertexDescription vertex_dcr;
  PlyFaceDescription face_dcr;

  PlyFace ply_face;
  PlyVertex ply_vertex;
  
  in.open(filename, fstream::in | fstream::binary);

  in.read( (char*)&head, sizeof(head));
  in.read( (char*)&vertex_dcr, sizeof(vertex_dcr));
  in.read( (char*)&face_dcr, sizeof(face_dcr));

  char* buffer = "end_header\n";
  char dummy[20];
  in.read( dummy, strlen(buffer));

  for(unsigned int i = 0; i < vertex_dcr.count; i++){

    in.read( (char*)&ply_vertex, sizeof(PlyVertex));

    ColorVertex v(ply_vertex.x, ply_vertex.y, ply_vertex.z,
			   (uchar)(ply_vertex.r * 255),
			   (uchar)(ply_vertex.g * 255),
			   (uchar)(ply_vertex.b * 255));
    
    Normal n(ply_vertex.nx, ply_vertex.ny, ply_vertex.nz);
  
    mesh.addVertex(v);
    mesh.addNormal(n);
  }

  for(unsigned int i = 0; i < face_dcr.count; i++){

    in.read( (char*)&ply_face, sizeof(ply_face));

    for(int j = 0; j < 3; j++)
	 mesh.addIndex(ply_face.indices[j]);
    
  }

  
}
