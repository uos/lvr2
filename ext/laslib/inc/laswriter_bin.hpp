/*
===============================================================================

  FILE:  laswriter_bin.hpp
  
  CONTENTS:
  
    Writes LIDAR points from to ASCII through on-the-fly conversion from LAS.

  PROGRAMMERS:

    martin.isenburg@gmail.com

  COPYRIGHT:

    (c) 2007-2011, Martin Isenburg, LASSO - tools to catch reality

    This is free software; you can redistribute and/or modify it under the
    terms of the GNU Lesser General Licence as published by the Free Software
    Foundation. See the COPYING file for more information.

    This software is distributed WITHOUT ANY WARRANTY and without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    5 November 2011 -- changed default IO buffer size with setvbuf() to 65536
    5 September 2011 -- created after sampling grapes in the sommerhausen hills  
  
===============================================================================
*/
#ifndef LAS_WRITER_BIN_HPP
#define LAS_WRITER_BIN_HPP

#include "laswriter.hpp"

#include <stdio.h>

class ByteStreamOut;

class LASwriterBIN : public LASwriter
{
public:

  BOOL refile(FILE* file);

  BOOL open(const char* file_name, const LASheader* header, const char* version, U32 io_buffer_size=65536);
  BOOL open(FILE* file, const LASheader* header, const char* version);
  BOOL open(ByteStreamOut* stream, const LASheader* header, const char* version);

  BOOL write_point(const LASpoint* point);
  BOOL chunk() { return FALSE; };

  BOOL update_header(const LASheader* header, BOOL use_inventory=TRUE, BOOL update_extra_bytes=FALSE);
  I64 close(BOOL update_npoints=true);

  LASwriterBIN();
  ~LASwriterBIN();

private:
  ByteStreamOut* stream;
  FILE* file;
  U32 version;
  I32 units;
  F64 origin_x;
  F64 origin_y;
  F64 origin_z;
};

#endif
