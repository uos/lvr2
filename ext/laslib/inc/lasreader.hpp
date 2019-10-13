/*
===============================================================================

  FILE:  lasreader.hpp
  
  CONTENTS:
  
    Interface to read LIDAR points from the LAS format versions 1.0 - 1.3 and
    per on-the-fly conversion from simple ASCII files.

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
  
    5 September 2011 -- support for reading Terrasolid's BIN format
    11 June 2011 -- billion point support: p_count & npoints are 64 bit counters
    9 April 2011 -- added capability to read on-the-fly conversion from ASCII
    24 January 2011 -- introduced LASreadOpener
    21 January 2011 -- turned into abstract reader to support multiple files
    3 December 2010 -- updated to (somewhat) support LAS format 1.3
    7 September 2008 -- updated to support LAS format 1.2 
    18 February 2007 -- created after repairing 2 vacuum cleaners in the garden
  
===============================================================================
*/
#ifndef LAS_READER_HPP
#define LAS_READER_HPP

#include "lasdefinitions.hpp"

class LASindex;
class LASfilter;
class LAStransform;
class ByteStreamIn;

class LASreader
{
public:
  LASheader header;
  LASpoint point;

  I64 npoints;
  I64 p_count;

  virtual I32 get_format() const = 0;

  void set_index(LASindex* index);
  LASindex* get_index() const;
  virtual void set_filter(LASfilter* filter);
  virtual void set_transform(LAStransform* transform);
  void reset_filter();

  void use_default_reader() { read_simple = &LASreader::read_point_default; };
  void use_alternate_reader() { read_simple = &LASreader::read_point_alternate; };

  virtual BOOL inside_tile(const F32 ll_x, const F32 ll_y, const F32 size);
  virtual BOOL inside_circle(const F64 center_x, const F64 center_y, const F64 radius);
  virtual BOOL inside_rectangle(const F64 min_x, const F64 min_y, const F64 max_x, const F64 max_y);

  virtual BOOL seek(const I64 p_index) = 0;
  BOOL read_point() { return (this->*read_simple)(); };

  inline void compute_coordinates() { point.compute_coordinates(); };

  inline F64 get_min_x() const { return header.min_x; };
  inline F64 get_min_y() const { return header.min_y; };
  inline F64 get_min_z() const { return header.min_z; };

  inline F64 get_max_x() const { return header.max_x; };
  inline F64 get_max_y() const { return header.max_y; };
  inline F64 get_max_z() const { return header.max_z; };

  inline F64 get_x() const { return get_x(point.x); };
  inline F64 get_y() const { return get_y(point.y); };
  inline F64 get_z() const { return get_z(point.z); };

  inline F64 get_x(const I32 x) const { return header.get_x(x); };
  inline F64 get_y(const I32 y) const { return header.get_y(y); };
  inline F64 get_z(const I32 z) const { return header.get_z(z); };

  inline I32 get_x(const F64 x) const { return header.get_x(x); };
  inline I32 get_y(const F64 y) const { return header.get_y(y); };
  inline I32 get_z(const F64 z) const { return header.get_z(z); };

  virtual ByteStreamIn* get_stream() const = 0;
  virtual void close(BOOL close_stream=TRUE) = 0;

  LASreader();
  virtual ~LASreader();

protected:
  virtual BOOL read_point_default() = 0;
  virtual BOOL read_point_alternate() { return read_point_default(); };

  LASindex* index;
  LASfilter* filter;
  LAStransform* transform;

  F64 r_min_x, r_min_y, r_max_x, r_max_y;
  F32 t_ll_x, t_ll_y, t_size, t_ur_x, t_ur_y;
  F64 c_center_x, c_center_y, c_radius, c_radius_squared;

private:
  BOOL (LASreader::*read_simple)();
  BOOL (LASreader::*read_complex)();

  BOOL read_point_inside_tile();
  BOOL read_point_inside_tile_indexed();
  BOOL read_point_inside_circle();
  BOOL read_point_inside_circle_indexed();
  BOOL read_point_inside_rectangle();
  BOOL read_point_inside_rectangle_indexed();
  BOOL read_point_filtered();
  BOOL read_point_transformed();
  BOOL read_point_filtered_and_transformed();
};

#include "laswaveform13reader.hpp"

class LASreadOpener
{
public:
  const char* get_file_name() const;
  void set_file_name(const char* file_name, BOOL unique=FALSE);
  BOOL add_file_name(const char* file_name, BOOL unique=FALSE);
  void delete_file_name(U32 file_name_id);
  BOOL set_file_name_current(U32 file_name_id);
  U32 get_file_name_number() const;
  const char* get_file_name(U32 number) const;
  I32 get_file_format(U32 number) const;
  void set_merged(const BOOL merged);
  BOOL get_merged() const;
  void set_scale_factor(const F64* scale_factor);
  void set_offset(const F64* offset);
  void set_translate_intensity(F32 translate_intensity);
  void set_scale_intensity(F32 scale_intensity);
  void set_translate_scan_angle(F32 translate_scan_angle);
  void set_scale_scan_angle(F32 scale_scan_angle);
  void add_extra_attribute(I32 data_type, const char* name, const char* description=0, F64 scale=1.0, F64 offset=0.0);
  void set_parse_string(const char* parse_string);
  void set_skip_lines(I32 skip_lines);
  void set_populate_header(BOOL populate_header);
  const char* get_parse_string() const;
  void usage() const;
  BOOL parse(int argc, char* argv[]);
  BOOL piped() const;
  BOOL has_populated_header() const;
  BOOL active() const;
  const LASfilter* get_filter() { return filter; };
  const LAStransform* get_transform() { return transform; };
  void reset();
  LASreader* open();
  BOOL reopen(LASreader* lasreader);
  LASwaveform13reader* open_waveform13(const LASheader* lasheader);
  LASreadOpener();
  ~LASreadOpener();
private:
#ifdef _WIN32
  void add_file_name_windows(const char* file_name, BOOL unique=FALSE);
#endif
  char** file_names;
  char* file_name;
  BOOL merged;
  U32 file_name_number;
  U32 file_name_allocated;
  U32 file_name_current;
  F64* scale_factor;
  F64* offset;
  BOOL use_alternate;
  F32 translate_intensity;
  F32 scale_intensity;
  F32 translate_scan_angle;
  F32 scale_scan_angle;
  I32 number_extra_attributes;
  I32 extra_attribute_data_types[10];
  char* extra_attribute_names[10];
  char* extra_attribute_descriptions[10];
  F64 extra_attribute_scales[10];
  F64 extra_attribute_offsets[10];
  char* parse_string;
  I32 skip_lines;
  BOOL populate_header;
  BOOL use_stdin;

  // optional extras
  LASindex* index;
  LASfilter* filter;
  LAStransform* transform;

  // optional clipping
  F32* inside_tile;
  F64* inside_circle;
  F64* inside_rectangle;
};

#endif
