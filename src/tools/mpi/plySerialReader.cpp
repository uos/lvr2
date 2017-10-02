//
// Created by imitschke on 11.07.17.
//

#include "plySerialReader.h"
#include <fstream>
plySerialReader::plySerialReader(std::string filePath) : m_filePath(filePath)
{

}

size_t plySerialReader::getVertexCount()
{

}

plySerialReader::parseHeader()
{
  ifstream ifs(m_filePath);
  string readStr;
  while(readStr!="end")
  {

  }

}
