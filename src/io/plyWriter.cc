#include "plyWriter.h"

PLYWriter::PLYWriter(char* filename) : FileWriter(filename){

  comment = "";

}

void PLYWriter::setComment(char* c){

  char buffer[256];
  strcat(buffer, "comment ");
  strcat(buffer, "c");
  strcat(buffer, "\n");

  strcpy(comment, buffer);

}

void PLYWriter::writeHeader(){

  //Setting up header
  strcpy(header_dcr.ply, "ply");
  strcpy(header_dcr.format, PLY_LITTLE_ENDIAN);
  strcpy(header_dcr.comment, comment);

  //Setting up vertex descripten
  strcpy(vertex_dcr.element, "element vertex ");
  strcpy(vertex_dcr.property_x, "property float x\n");
  strcpy(vertex_dcr.property_y, "property float y\n");
  strcpy(vertex_dcr.property_z, "property float z\n");
  strcpy(vertex_dcr.property_nx, "property float nx\n");
  strcpy(vertex_dcr.property_ny, "property float ny\n");
  strcpy(vertex_dcr.property_nz, "property float nz\n");
  vertex_dcr.count = 0;

  //Setting up face desription
  strcpy(face_dcr.face, "face ");
  strcpy(face_dcr.property, "property list uchar int vertex_index\n");
  face_dcr.count = 3;

}

void PLYWriter::writeFooter(){
}

void PLYWriter::addMesh(StaticMesh &mesh){

  //Local variables
  PlyVertex ply_vertex;
  PlyFace ply_face;

  //Set vertex and face count
  vertex_dcr.count = (int)mesh.vertices.size();
  face_dcr.count = (int)mesh.indices.size() / 3;
  ply_face.vertexCount = 0;

  //Write header
  out.write( (char*)&header_dcr, sizeof(header_dcr));
  out.write( (char*)&vertex_dcr, sizeof(vertex_dcr));
  out.write( (char*)&face_dcr, sizeof(face_dcr));

  char* buffer = "end_header\n";
  out.write(buffer, strlen(buffer));

  //Write vertices and normals
  for(unsigned int i = 0; i < vertex_dcr.count; i++){

    //ColorVertex v = (ColorVertex)mesh.vertices[i];
    ColorVertex v = mesh.vertices[i];

    ply_vertex.x = v.x;
    ply_vertex.y = v.y;
    ply_vertex.z = v.z;

    ply_vertex.nx = mesh.normals[i].x;
    ply_vertex.ny = mesh.normals[i].y;
    ply_vertex.nz = mesh.normals[i].z;

    ply_vertex.r = v.r / 255.0;
    ply_vertex.g = v.g / 255.0;
    ply_vertex.b = v.b / 255.0;

    ply_vertex.u = 0.0;
    ply_vertex.v = 0.0;

    ply_vertex.texture = 1;

    out.write( (char*)&ply_vertex, sizeof(ply_vertex));

  }

  //Write faces
  for(unsigned int i = 0; i < mesh.indices.size(); i += 3){

    ply_face.indices[0] = mesh.indices[i];
    ply_face.indices[1] = mesh.indices[i+1];
    ply_face.indices[2] = mesh.indices[i+2];

    out.write( (char*)&ply_face, sizeof(ply_face));
  }

}


void PLYWriter::init(char* filename){

  out.open(filename, fstream::out | fstream::binary);

}
