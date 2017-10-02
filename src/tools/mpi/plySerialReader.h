//
// Created by imitschke on 11.07.17.
//

#ifndef LAS_VEGAS_PLYSERIALREADER_H
#define LAS_VEGAS_PLYSERIALREADER_H
#include <string>

class plySerialReader
{
    plySerialReader(std::string filePath);
    size_t getVertexCount();
private:
    m_filePath;
    parseHeader();
};


#endif //LAS_VEGAS_PLYSERIALREADER_H
