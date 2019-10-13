/*
===============================================================================

  FILE:  lasutility.hpp
  
  CONTENTS:
  
    Simple utilities that come in handy when using the laslib API.

  PROGRAMMERS:
  
    martin.isenburg@gmail.com
  
  COPYRIGHT:
  
    (c) 2010-2011, Martin Isenburg, LASSO - tools to catch reality

    This is free software; you can redistribute and/or modify it under the
    terms of the GNU Lesser General Licence as published by the Free Software
    Foundation. See the COPYING file for more information.

    This software is distributed WITHOUT ANY WARRANTY and without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
  
  CHANGE HISTORY:
  
    25 December 2010 -- created after swinging in Mara's hammock for hours
  
===============================================================================
*/
#ifndef LAS_UTILITY_HPP
#define LAS_UTILITY_HPP

#include "lasdefinitions.hpp"

class LASinventory
{
public:
  BOOL active() const { return (first == false); }; 
  U32 number_of_point_records;
  U32 number_of_points_by_return[8];
  I32 raw_max_x;
  I32 raw_min_x;
  I32 raw_max_y;
  I32 raw_min_y;
  I32 raw_max_z;
  I32 raw_min_z;
  BOOL add(const LASpoint* point);
  LASinventory();
private:
  BOOL first;
};

class LASsummary
{
public:
  BOOL active() const { return (first == false); }; 
  U32 number_of_point_records;
  U32 number_of_points_by_return[8];
  U32 number_of_returns_of_given_pulse[8];
  U32 classification[32];
  U32 classification_synthetic;
  U32 classification_keypoint;
  U32 classification_withheld;
  LASpoint min;
  LASpoint max;
  BOOL add(const LASpoint* point);
  LASsummary();
private:
  BOOL first;
};

class LASbin
{
public:
  void add(I32 item);
  void add(I64 item);
  void add(F64 item);
  void add(I32 item, I32 value);
  void report(FILE* file, const char* name=0, const char* name_avg=0) const;
  LASbin(F32 step);
  ~LASbin();
private:
  void add_to_bin(I32 bin);
  F64 total;
  I64 count;
  F32 one_over_step;
  BOOL first;
  I32 anker;
  I32 size_pos;
  I32 size_neg;
  U32* bins_pos;
  U32* bins_neg;
  F64* values_pos;
  F64* values_neg;
};

class LAShistogram
{
public:
  BOOL active() const { return is_active; }; 
  BOOL parse(int argc, char* argv[]);
  BOOL histo(const char* name, F32 step);
  BOOL histo_avg(const char* name, F32 step, const char* name_avg);
  void add(const LASpoint* point);
  void report(FILE* file) const;
  LAShistogram();
  ~LAShistogram();
private:
  BOOL is_active;
  // counter bins
  LASbin* x_bin;
  LASbin* y_bin;
  LASbin* z_bin;
  LASbin* intensity_bin;
  LASbin* classification_bin;
  LASbin* scan_angle_bin;
  LASbin* point_source_id_bin;
  LASbin* gps_time_bin;
  LASbin* wavepacket_index_bin;
  LASbin* wavepacket_offset_bin;
  LASbin* wavepacket_size_bin;
  LASbin* wavepacket_location_bin;
  // averages bins
  LASbin* classification_bin_intensity;
  LASbin* classification_bin_scan_angle;
  LASbin* scan_angle_bin_z;
  LASbin* scan_angle_bin_number_of_returns;
  LASbin* scan_angle_bin_intensity;
  LASbin* return_map_bin_intensity;
};

class LASoccupancyGrid
{
public:
  void reset();
  BOOL add(const LASpoint* point);
  BOOL add(I32 pos_x, I32 pos_y);
  BOOL occupied(const LASpoint* point) const;
  BOOL occupied(I32 pos_x, I32 pos_y) const;
  BOOL active() const;
  U32 get_num_occupied() const { return num_occupied; };
  BOOL write_asc_grid(const char* file_name) const;

  // read from file or write to file
//  BOOL read(ByteStreamIn* stream);
//  BOOL write(ByteStreamOut* stream) const;

  LASoccupancyGrid(F32 grid_spacing);
  ~LASoccupancyGrid();
  I32 min_x, min_y, max_x, max_y;
private:
  BOOL add_internal(I32 pos_x, I32 pos_y);
  F32 grid_spacing;
  I32 anker;
  I32* minus_ankers;
  U32 minus_minus_size;
  U32** minus_minus;
  U16* minus_minus_sizes;
  U32 minus_plus_size;
  U32** minus_plus;
  U16* minus_plus_sizes;
  I32* plus_ankers;
  U32 plus_minus_size;
  U32** plus_minus;
  U16* plus_minus_sizes;
  U32 plus_plus_size;
  U32** plus_plus;
  U16* plus_plus_sizes;
  U32 num_occupied;
};

#endif
