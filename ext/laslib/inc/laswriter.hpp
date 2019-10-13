/*
===============================================================================

  FILE:  laswriter.hpp
  
  CONTENTS:
  
    Interface to write LIDAR points to the LAS format versions 1.0 - 1.3 and
    per on-the-fly conversion to simple ASCII files.

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
  
    5 September 2011 -- support for writing Terrasolid's BIN format
    11 June 2011 -- billion point support: p_count & npoints are 64 bit counters
    8 May 2011 -- added an option for variable chunking via chunk()
    9 April 2011 -- added capability to write on-the-fly conversion to ASCII
    24 January 2011 -- introduced LASwriteOpener
    21 January 2011 -- turned into abstract reader to support multiple files
    3 December 2010 -- updated to (somewhat) support LAS format 1.3
    7 September 2008 -- updated to support LAS format 1.2 
    21 February 2007 -- created after eating Sarah's veggies with peanutsauce
  
===============================================================================
*/
#ifndef LAS_WRITER_HPP
#define LAS_WRITER_HPP

#include "lasdefinitions.hpp"

#include "lasutility.hpp"

class LASwriter
{
public:
  LASquantizer quantizer;
  I64 npoints;
  I64 p_count;
  LASinventory inventory;

  virtual BOOL write_point(const LASpoint* point) = 0;
  void update_inventory(const LASpoint* point) { inventory.add(point); };
  virtual BOOL chunk() = 0;

  virtual BOOL update_header(const LASheader* header, BOOL use_inventory=TRUE, BOOL update_extra_bytes=FALSE) = 0;
  virtual I64 close(BOOL update_npoints=TRUE) = 0;

  LASwriter() { npoints = 0; p_count = 0; };
  virtual ~LASwriter() {};
};

#include "laswaveform13writer.hpp"

class LASwriteOpener
{
public:
  void set_file_name(const char* file_name);
  void set_format(const char* format);
  void make_file_name(const char* file_name, I32 file_number=-1);
  const char* get_file_name() const;
  BOOL format_was_specified() const;
  I32 get_format() const;
  void set_parse_string(const char* parse_string);
  void set_separator(const char* separator);
  void usage() const;
  BOOL parse(int argc, char* argv[]);
  BOOL active() const;
  BOOL piped() const;
  LASwriter* open(LASheader* header);
  LASwaveform13writer* open_waveform13(const LASheader* lasheader);
  LASwriteOpener();
  ~LASwriteOpener();
private:
  char* file_name;
  char* parse_string;
  char* separator;
  U32 format;
  U32 chunk_size;
  BOOL use_chunking;
  BOOL use_stdout;
  BOOL use_nil;
  BOOL use_v1;
};

#endif
