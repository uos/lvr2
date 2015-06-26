/*
 * MPITree.hpp
 *
 *  Created on: 15.01.2013
 *      Author: Dominik Feldschnieders
 */

#ifndef SERIALFILEREADER_HPP_
#define SERIALFILEREADER_HPP_

#include <fstream> /*ifstream*/
#include <iostream>
#include "geometry/Vertex.hpp"

namespace lvr{

/**
 * @brief A class for execution of a distribution using a very simple Kd-tree.
 * 	The classification criterion is to halve the longest axis.
 */
class SerialReader {
public:

	~SerialReader()
  {
    if(m_inputFile.is_open())
    {
      m_inputFile.close();
    }
  }



   SerialReader(string filePath)
   {
    try
    {
      m_inputFile.open(filePath, std::ifstream::in);
    }
    catch (std::ios_base::failure ex)
    {
      cout << "An exception occurred. " << ex.what() << '\n';
    }

   }

   Vertex<float> getVertexAt(unsigned long long int i)
   {
     
   }




private:
  std::ifstream m_inputFile;
};
}

#endif /* SERIALFILEREADER_HPP_ */
