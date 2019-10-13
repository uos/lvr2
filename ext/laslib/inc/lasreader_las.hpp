/*
===============================================================================

  FILE:  lasreader_las.hpp
  
  CONTENTS:
  
    Reads LIDAR points from the LAS format (Version 1.x , April 29, 2008).

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
    21 January 2011 -- adapted from lasreader to create abstract reader class
    3 December 2010 -- updated to (somewhat) support LAS format 1.3
    7 September 2008 -- updated to support LAS format 1.2 
    18 February 2007 -- created after repairing 2 vacuum cleaners in the garden
  
===============================================================================
*/
#ifndef LAS_READER_LAS_HPP
#define LAS_READER_LAS_HPP

#include "lasreader.hpp"

#include <stdio.h>

#ifdef LZ_WIN32_VC6
#include <fstream.h>
#else
#include <istream>
#include <fstream>
using namespace std;
#endif

class LASreadPoint;

class LASreaderLAS : public LASreader
{
public:

  BOOL open(const char* file_name, U32 io_buffer_size=65536);
  BOOL open(FILE* file);
  BOOL open(istream& stream);

  I32 get_format() const;

  BOOL seek(const I64 p_index);

  ByteStreamIn* get_stream() const;
  void close(BOOL close_stream=TRUE);

  LASreaderLAS();
  virtual ~LASreaderLAS();

protected:
  virtual BOOL open(ByteStreamIn* stream);
  virtual BOOL read_point_default();

private:
  FILE* file;
  ByteStreamIn* stream;
  LASreadPoint* reader;
};

class LASreaderLASrescale : public virtual LASreaderLAS
{
public:
  LASreaderLASrescale(F64 x_scale_factor, F64 y_scale_factor, F64 z_scale_factor);

protected:
  virtual BOOL open(ByteStreamIn* stream);
  virtual BOOL read_point_default();
  BOOL rescale_x, rescale_y, rescale_z;
  F64 scale_factor[3];
  F64 orig_x_scale_factor, orig_y_scale_factor, orig_z_scale_factor;
};

class LASreaderLASreoffset : public virtual LASreaderLAS
{
public:
  LASreaderLASreoffset(F64 x_offset, F64 y_offset, F64 z_offset);

protected:
  virtual BOOL open(ByteStreamIn* stream);
  virtual BOOL read_point_default();
  BOOL reoffset_x, reoffset_y, reoffset_z;
  F64 offset[3];
  F64 orig_x_offset, orig_y_offset, orig_z_offset;
};

class LASreaderLASrescalereoffset : public LASreaderLASrescale, LASreaderLASreoffset
{
public:
  LASreaderLASrescalereoffset(F64 x_scale_factor, F64 y_scale_factor, F64 z_scale_factor, F64 x_offset, F64 y_offset, F64 z_offset);

protected:
  BOOL open(ByteStreamIn* stream);
  BOOL read_point_default();
};

#endif
