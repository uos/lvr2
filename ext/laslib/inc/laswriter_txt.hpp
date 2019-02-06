/*
===============================================================================

  FILE:  laswriter_txt.hpp
  
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
  
    10 April 2011 -- created after a sunny weekend of biking to/from Buergel
  
===============================================================================
*/
#ifndef LAS_WRITER_TXT_HPP
#define LAS_WRITER_TXT_HPP

#include "laswriter.hpp"

#include <stdio.h>

class LASwriterTXT : public LASwriter
{
public:

  BOOL refile(FILE* file);

  BOOL open(const char* file_name, const LASheader* header, const char* parse_string=0, const char* separator=0);
  BOOL open(FILE* file, const LASheader* header, const char* parse_string=0, const char* separator=0);

  BOOL write_point(const LASpoint* point);
  BOOL chunk() { return FALSE; };

  BOOL update_header(const LASheader* header, BOOL use_inventory=TRUE, BOOL update_extra_bytes=FALSE);
  I64 close(BOOL update_npoints=true);

  LASwriterTXT();
  ~LASwriterTXT();

private:
  BOOL close_file;
  FILE* file;
  const LASheader* header;
  char* parse_string;
  char separator_sign;
  char printstring[512];
  I32 extra_attribute_array_offsets[10];
  BOOL check_parse_string(const char* parse_string);
  BOOL unparse_extra_attribute(const LASpoint* point, I32 index);
};

#endif
