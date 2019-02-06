/*
===============================================================================

  FILE:  lasreadermerged.hpp
  
  CONTENTS:
  
    Reads LIDAR points from the LAS format from more than one file.

  PROGRAMMERS:

    martin.isenburg@gmail.com

  COPYRIGHT:

    (c) 2011, Martin Isenburg, LASSO - tools to catch reality

    This is free software; you can redistribute and/or modify it under the
    terms of the GNU Lesser General Licence as published by the Free Software
    Foundation. See the COPYING file for more information.

    This software is distributed WITHOUT ANY WARRANTY and without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    20 January 2011 -- created missing Livermore and my Extra Virgin Olive Oil
  
===============================================================================
*/
#ifndef LAS_READER_MERGED_HPP
#define LAS_READER_MERGED_HPP

#include "lasreader_las.hpp"
#include "lasreader_bin.hpp"
#include "lasreader_shp.hpp"
#include "lasreader_qfit.hpp"
#include "lasreader_txt.hpp"

class LASreaderMerged : public LASreader
{
public:

  BOOL add_file_name(const char* file_name);
  void set_scale_factor(const F64* scale_factor);
  void set_offset(const F64* offset);
  void set_files_are_flightlines(BOOL flightlines);
  void set_translate_intensity(F32 translate_intensity);
  void set_scale_intensity(F32 scale_intensity);
  void set_translate_scan_angle(F32 translate_scan_angle);
  void set_scale_scan_angle(F32 scale_scan_angle);
  void set_parse_string(const char* parse_string);
  void set_skip_lines(I32 skip_lines);
  void set_populate_header(BOOL populate_header);
  BOOL open();
  BOOL reopen();

  void set_filter(LASfilter* filter);
  void set_transform(LAStransform* transform);

  BOOL inside_tile(const F32 ll_x, const F32 ll_y, const F32 size);
  BOOL inside_circle(const F64 center_x, const F64 center_y, const F64 radius);
  BOOL inside_rectangle(const F64 min_x, const F64 min_y, const F64 max_x, const F64 max_y);

  I32 get_format() const;

  BOOL seek(const I64 p_index){ return FALSE; };

  ByteStreamIn* get_stream() const { return 0; };
  void close(BOOL close_stream=TRUE);

  LASreaderMerged();
  ~LASreaderMerged();

protected:
  BOOL read_point_default();
  BOOL read_point_alternate();

private:
  BOOL open_next_file();
  void clean();

  LASreader* lasreader;
  LASreaderLAS* lasreaderlas;
  LASreaderBIN* lasreaderbin;
  LASreaderSHP* lasreadershp;
  LASreaderQFIT* lasreaderqfit;
  LASreaderTXT* lasreadertxt;
  BOOL point_type_change;
  BOOL point_size_change;
  BOOL rescale;
  BOOL reoffset;
  F64* scale_factor;
  F64* offset;
  F32 translate_intensity;
  F32 scale_intensity;
  F32 translate_scan_angle;
  F32 scale_scan_angle;
  char* parse_string;
  I32 skip_lines;
  BOOL populate_header;
  U32 file_name_current;
  U32 file_name_number;
  U32 file_name_allocated;
  char** file_names;
  F64* bounding_boxes;
  U32 inside; // 0 = none, 1 = tile, 2 = circle, 3 = rectangle
};

#endif
