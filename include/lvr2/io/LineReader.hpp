/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
 * LineReader.hpp
 *
 *  Created on: Aug 15, 2017
 *      Author: Isaak Mitschke
 */

#ifndef LAS_VEGAS_LINEREADER_HPP
#define LAS_VEGAS_LINEREADER_HPP

#include "DataStruct.hpp"

#include <boost/shared_array.hpp>
#include <exception>
#include <string>

namespace lvr2
{

enum fileType
{
    XYZ,
    XYZRGB,
    XYZN,
    XYZNRGB
};

struct fileAttribut
{
    std::string m_filePath;
    size_t m_filePos;
    size_t m_elementAmount;
    fileType m_fileType;
    size_t m_PointBlockSize;
    bool m_ply;
    bool m_binary;
    size_t m_line_element_amount;
};

struct __attribute__((packed)) xyz
{
    lvr2::coord<float> point;
};

struct __attribute__((packed)) xyzn : xyz
{
    lvr2::coord<float> normal;
};

struct __attribute__((packed)) xyznc : xyzn
{
    lvr2::color<unsigned char> color;
};
struct __attribute__((packed)) xyzc : xyz
{
    lvr2::color<unsigned char> color;
};

/// TODO: Fixme!
/// Including this in BoundingBox leads to a very nasty circular 
/// include history when using the ModelFactory. Currently,
/// the only available solution is to inline all LineReader
/// methods (which is of course no suitable permanent solution).

class LineReader
{
  public:
    inline LineReader();
    inline LineReader(std::string filePath);
    inline LineReader(std::vector<std::string> filePaths);
    inline void open(std::string filePath);
    inline void open(std::vector<std::string> filePaths);
    inline size_t getNumPoints();
    inline bool getNextPoint(xyznc& point);
    //        boost::shared_array<xyzn> getNextPoints(size_t &return_amount, size_t amount =
    //        1000000); boost::shared_array<xyzc> getNextPoints(size_t &return_amount, size_t amount
    //        = 1000000); boost::shared_array<xyznc> getNextPoints(size_t &return_amount, size_t
    //        amount = 1000000);
    inline boost::shared_ptr<void> getNextPoints(size_t& return_amount, size_t amount = 1000000);
    inline fileType getFileType(size_t i);
    inline fileType getFileType();
    inline void rewind(size_t i);
    inline void rewind();
    inline bool ok();
    inline bool isPly() { return m_ply; }

    class readException : public std::exception
    {
      public:
        readException(std::string what) : error_msg(what) {}
        virtual const char* what() const throw() { return error_msg.c_str(); }

      private:
        std::string error_msg;
    };

  private:
    std::vector<std::string> m_filePaths;
    std::vector<size_t> m_filePos;
    size_t m_elementAmount;
    fileType m_fileType;
    size_t m_PointBlockSize;
    bool m_ply;
    bool m_binary;
    size_t m_line_element_amount;
    size_t m_numFiles;
    size_t m_currentReadFile;
    bool m_openNextFile;
    std::vector<fileAttribut> m_fileAttributes;
};

} // namespace lvr2

#include "lvr2/io/LineReader.tcc"

#endif // LAS_VEGAS_LINEREADER_H
