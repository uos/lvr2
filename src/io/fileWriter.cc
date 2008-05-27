#include "fileWriter.h"

FileWriter::FileWriter(char* filename){

  init(filename);
  file_open = out.good();
}

FileWriter::~FileWriter(){

  if(file_open) out.close();
  
}

void FileWriter::writeHeader(){

}

void FileWriter::writeFooter(){
  
}

void FileWriter::addMesh(StaticMesh &mesh){
}

void FileWriter::init(char* fileName){

  out.open(fileName, ios::out);
  file_open = out.good();
  
}
