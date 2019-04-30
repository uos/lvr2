/*
===============================================================================

  FILE:  lasfilter.cpp
  
  CONTENTS:
  
    see corresponding header file
  
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
  
    see corresponding header file
  
===============================================================================
*/
#include "lasfilter.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

class LAScriterionClipTile : public LAScriterion
{
public:
  inline const char* name() const { return "clip_tile"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g %g ", name(), ll_x, ll_y, tile_size); };
  inline BOOL filter(const LASpoint* point) { return (!point->inside_tile(ll_x, ll_y, ur_x, ur_y)); };
  LAScriterionClipTile(F32 ll_x, F32 ll_y, F32 tile_size) { this->ll_x = ll_x; this->ll_y = ll_y; this->ur_x = ll_x+tile_size; this->ur_y = ll_y+tile_size; this->tile_size = tile_size; };
private:
  F32 ll_x, ll_y, ur_x, ur_y, tile_size;
};

class LAScriterionClipCircle : public LAScriterion
{
public:
  inline const char* name() const { return "clip_circle"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g %g ", name(), center_x, center_y, radius); };
  inline BOOL filter(const LASpoint* point) { return (!point->inside_circle(center_x, center_y, radius_squared)); };
  LAScriterionClipCircle(F64 x, F64 y, F64 radius) { this->center_x = x; this->center_y = y; this->radius = radius; this->radius_squared = radius*radius; };
private:
  F64 center_x, center_y, radius, radius_squared;
};

class LAScriterionClipXY : public LAScriterion
{
public:
  inline const char* name() const { return "clip"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g %g %g ", name(), below_x, below_y, above_x, above_y); };
  inline BOOL filter(const LASpoint* point) { return (!point->inside_rectangle(below_x, below_y, above_x, above_y)); };
  LAScriterionClipXY(F64 below_x, F64 below_y, F64 above_x, F64 above_y) { this->below_x = below_x; this->below_y = below_y; this->above_x = above_x; this->above_y = above_y; };
private:
  F64 below_x, below_y, above_x, above_y;
};

class LAScriterionClipZ : public LAScriterion
{
public:
  inline const char* name() const { return "clip_z"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g ", name(), below_z, above_z); };
  inline BOOL filter(const LASpoint* point) { F64 z = point->get_z(); return (z < below_z) || (z > above_z); };
  LAScriterionClipZ(F64 below_z, F64 above_z) { this->below_z = below_z; this->above_z = above_z; };
private:
  F64 below_z, above_z;
};

class LAScriterionClipXBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_x_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), below_x); };
  inline BOOL filter(const LASpoint* point) { return (point->get_x() < below_x); };
  LAScriterionClipXBelow(F64 below_x) { this->below_x = below_x; };
private:
  F64 below_x;
};

class LAScriterionClipXAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_x_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), above_x); };
  inline BOOL filter(const LASpoint* point) { return (point->get_x() > above_x); };
  LAScriterionClipXAbove(F64 above_x) { this->above_x = above_x; };
private:
  F64 above_x;
};

class LAScriterionClipYBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_y_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), below_y); };
  inline BOOL filter(const LASpoint* point) { return (point->get_y() < below_y); };
  LAScriterionClipYBelow(F64 below_y) { this->below_y = below_y; };
private:
  F64 below_y;
};

class LAScriterionClipYAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_y_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), above_y); };
  inline BOOL filter(const LASpoint* point) { return (point->get_y() > above_y); };
  LAScriterionClipYAbove(F64 above_y) { this->above_y = above_y; };
private:
  F64 above_y;
};

class LAScriterionClipZBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_z_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), below_z); };
  inline BOOL filter(const LASpoint* point) { return (point->get_z() < below_z); };
  LAScriterionClipZBelow(F64 below_z) { this->below_z = below_z; };
private:
  F64 below_z;
};

class LAScriterionClipZAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_z_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), above_z); };
  inline BOOL filter(const LASpoint* point) { return (point->get_z() > above_z); };
  LAScriterionClipZAbove(F64 above_z) { this->above_z = above_z; };
private:
  F64 above_z;
};

class LAScriterionClipRawXY : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_xy"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d %d %d ", name(), below_x, below_y, above_x, above_y); };
  inline BOOL filter(const LASpoint* point) { return (point->x < below_x) || (point->y < below_y) || (point->x > above_x) || (point->y > above_y); };
  LAScriterionClipRawXY(I32 below_x, I32 below_y, I32 above_x, I32 above_y) { this->below_x = below_x; this->below_y = below_y; this->above_x = above_x; this->above_y = above_y; };
private:
  I32 below_x, below_y, above_x, above_y;
};

class LAScriterionClipRawZ : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_z"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_z, above_z); };
  inline BOOL filter(const LASpoint* point) { return (point->z < below_z) || (point->z > above_z); };
  LAScriterionClipRawZ(I32 below_z, I32 above_z) { this->below_z = below_z; this->above_z = above_z; };
private:
  I32 below_z, above_z;
};

class LAScriterionClipRawXBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_x_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_x); };
  inline BOOL filter(const LASpoint* point) { return (point->x < below_x); };
  LAScriterionClipRawXBelow(I32 below_x) { this->below_x = below_x; };
private:
  I32 below_x;
};

class LAScriterionClipRawXAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_x_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_x); };
  inline BOOL filter(const LASpoint* point) { return (point->x > above_x); };
  LAScriterionClipRawXAbove(I32 above_x) { this->above_x = above_x; };
private:
  I32 above_x;
};

class LAScriterionClipRawYBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_y_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_y); };
  inline BOOL filter(const LASpoint* point) { return (point->y < below_y); };
  LAScriterionClipRawYBelow(I32 below_y) { this->below_y = below_y; };
private:
  I32 below_y;
};

class LAScriterionClipRawYAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_y_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_y); };
  inline BOOL filter(const LASpoint* point) { return (point->y > above_y); };
  LAScriterionClipRawYAbove(I32 above_y) { this->above_y = above_y; };
private:
  I32 above_y;
};

class LAScriterionClipRawZBelow : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_z_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_z); };
  inline BOOL filter(const LASpoint* point) { return (point->z < below_z); };
  LAScriterionClipRawZBelow(I32 below_z) { this->below_z = below_z; };
private:
  I32 below_z;
};

class LAScriterionClipRawZAbove : public LAScriterion
{
public:
  inline const char* name() const { return "clip_raw_z_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_z); };
  inline BOOL filter(const LASpoint* point) { return (point->z > above_z); };
  LAScriterionClipRawZAbove(I32 above_z) { this->above_z = above_z; };
private:
  I32 above_z;
};

class LAScriterionKeepFirstReturn : public LAScriterion
{
public:
  inline const char* name() const { return "keep_first"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->return_number > 1); };
};

class LAScriterionKeepMiddleReturn : public LAScriterion
{
public:
  inline const char* name() const { return "keep_middle"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return ((point->return_number == 1) || (point->return_number >= point->number_of_returns_of_given_pulse)); };
};

class LAScriterionKeepLastReturn : public LAScriterion
{
public:
  inline const char* name() const { return "keep_last"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->return_number < point->number_of_returns_of_given_pulse); };
};

class LAScriterionDropFirstReturn : public LAScriterion
{
public:
  inline const char* name() const { return "drop_first"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->return_number == 1); };
};

class LAScriterionDropMiddleReturn : public LAScriterion
{
public:
  inline const char* name() const { return "drop_middle"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return ((point->return_number > 1) && (point->return_number < point->number_of_returns_of_given_pulse)); };
};

class LAScriterionDropLastReturn : public LAScriterion
{
public:
  inline const char* name() const { return "drop_last"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->return_number >= point->number_of_returns_of_given_pulse); };
};

class LAScriterionKeepReturns : public LAScriterion
{
public:
  inline const char* name() const { return "keep_return_mask"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %u ", name(), ~drop_return_mask); };
  inline BOOL filter(const LASpoint* point) { return ((1 << point->return_number) & drop_return_mask); };
  LAScriterionKeepReturns(U32 keep_return_mask) { drop_return_mask = ~keep_return_mask; };
private:
  U32 drop_return_mask;
};

class LAScriterionKeepSpecificNumberOfReturns : public LAScriterion
{
public:
  inline const char* name() const { return (numberOfReturns == 1 ? "keep_single" : (numberOfReturns == 2 ? "keep_double" : (numberOfReturns == 3 ? "keep_triple" : (numberOfReturns == 4 ? "keep_quadruple" : "keep_quintuple")))); };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->number_of_returns_of_given_pulse != numberOfReturns); };
  LAScriterionKeepSpecificNumberOfReturns(U32 numberOfReturns) { this->numberOfReturns = numberOfReturns; };
private:
  U32 numberOfReturns;
};

class LAScriterionDropSpecificNumberOfReturns : public LAScriterion
{
public:
  inline const char* name() const { return (numberOfReturns == 1 ? "drop_single" : (numberOfReturns == 2 ? "drop_double" : (numberOfReturns == 3 ? "drop_triple" : (numberOfReturns == 4 ? "drop_quadruple" : "drop_quintuple")))); };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->number_of_returns_of_given_pulse == numberOfReturns); };
  LAScriterionDropSpecificNumberOfReturns(U32 numberOfReturns) { this->numberOfReturns = numberOfReturns; };
private:
  U32 numberOfReturns;
};

class LAScriterionDropScanDirection : public LAScriterion
{
public:
  inline const char* name() const { return "drop_scan_direction"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), scan_direction); };
  inline BOOL filter(const LASpoint* point) { return (scan_direction == point->scan_direction_flag); };
  LAScriterionDropScanDirection(I32 scan_direction) { this->scan_direction = scan_direction; };
private:
  I32 scan_direction;
};

class LAScriterionScanDirectionChangeOnly : public LAScriterion
{
public:
  inline const char* name() const { return "scan_direction_change_only"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { if (scan_direction_flag == point->scan_direction_flag) return TRUE; I32 s = scan_direction_flag; scan_direction_flag = point->scan_direction_flag; return s == -1; };
  void reset() { scan_direction_flag = -1; };
  LAScriterionScanDirectionChangeOnly() { reset(); };
private:
  I32 scan_direction_flag;
};

class LAScriterionEdgeOfFlightLineOnly : public LAScriterion
{
public:
  inline const char* name() const { return "edge_of_flight_line_only"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s ", name()); };
  inline BOOL filter(const LASpoint* point) { return (point->edge_of_flight_line == 0); };
};

class LAScriterionKeepScanAngle : public LAScriterion
{
public:
  inline const char* name() const { return "keep_scan_angle"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_scan, above_scan); };
  inline BOOL filter(const LASpoint* point) { return (point->scan_angle_rank < below_scan) || (above_scan < point->scan_angle_rank); };
  LAScriterionKeepScanAngle(I32 below_scan, I32 above_scan) { if (above_scan < below_scan) { this->below_scan = above_scan; this->above_scan = below_scan; } else { this->below_scan = below_scan; this->above_scan = above_scan; } };
private:
  I32 below_scan, above_scan;
};

class LAScriterionDropScanAngleBelow : public LAScriterion
{
public:
  inline const char* name() const { return "drop_scan_angle_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_scan); };
  inline BOOL filter(const LASpoint* point) { return (point->scan_angle_rank < below_scan); };
  LAScriterionDropScanAngleBelow(I32 below_scan) { this->below_scan = below_scan; };
private:
  I32 below_scan;
};

class LAScriterionDropScanAngleAbove : public LAScriterion
{
public:
  inline const char* name() const { return "drop_scan_angle_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_scan); };
  inline BOOL filter(const LASpoint* point) { return (point->scan_angle_rank > above_scan); };
  LAScriterionDropScanAngleAbove(I32 above_scan) { this->above_scan = above_scan; };
private:
  I32 above_scan;
};

class LAScriterionDropScanAngleBetween : public LAScriterion
{
public:
  inline const char* name() const { return "drop_scan_angle_between"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_scan, above_scan); };
  inline BOOL filter(const LASpoint* point) { return (below_scan <= point->scan_angle_rank) && (point->scan_angle_rank <= above_scan); };
  LAScriterionDropScanAngleBetween(I32 below_scan, I32 above_scan) { if (above_scan < below_scan) { this->below_scan = above_scan; this->above_scan = below_scan; } else { this->below_scan = below_scan; this->above_scan = above_scan; } };
private:
  I32 below_scan, above_scan;
};

class LAScriterionKeepIntensity : public LAScriterion
{
public:
  inline const char* name() const { return "keep_intensity"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_intensity, above_intensity); };
  inline BOOL filter(const LASpoint* point) { return (point->intensity < below_intensity) || (point->intensity > above_intensity); };
  LAScriterionKeepIntensity(I32 below_intensity, I32 above_intensity) { this->below_intensity = below_intensity; this->above_intensity = above_intensity; };
private:
  I32 below_intensity, above_intensity;
};

class LAScriterionDropIntensityBelow : public LAScriterion
{
public:
  inline const char* name() const { return "drop_intensity_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_intensity); };
  inline BOOL filter(const LASpoint* point) { return (point->intensity < below_intensity); };
  LAScriterionDropIntensityBelow(I32 below_intensity) { this->below_intensity = below_intensity; };
private:
  I32 below_intensity;
};

class LAScriterionDropIntensityAbove : public LAScriterion
{
public:
  inline const char* name() const { return "drop_intensity_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_intensity); };
  inline BOOL filter(const LASpoint* point) { return (point->intensity > above_intensity); };
  LAScriterionDropIntensityAbove(I32 above_intensity) { this->above_intensity = above_intensity; };
private:
  I32 above_intensity;
};

class LAScriterionDropIntensityBetween : public LAScriterion
{
public:
  inline const char* name() const { return "drop_intensity_between"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_intensity, above_intensity); };
  inline BOOL filter(const LASpoint* point) { return (below_intensity <= point->intensity) && (point->intensity <= above_intensity); };
  LAScriterionDropIntensityBetween(I32 below_intensity, I32 above_intensity) { this->below_intensity = below_intensity; this->above_intensity = above_intensity; };
private:
  I32 below_intensity, above_intensity;
};

class LAScriterionKeepClassifications : public LAScriterion
{
public:
  inline const char* name() const { return "keep_classification_mask"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %u ", name(), ~drop_classification_mask); };
  inline BOOL filter(const LASpoint* point) { return ((1 << point->classification) & drop_classification_mask); };
  LAScriterionKeepClassifications(U32 keep_classification_mask) { drop_classification_mask = ~keep_classification_mask; };
private:
  U32 drop_classification_mask;
};

class LAScriterionKeepPointSource : public LAScriterion
{
public:
  inline const char* name() const { return "keep_point_source"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), point_source_id); };
  inline BOOL filter(const LASpoint* point) { return (point->point_source_ID != point_source_id); };
  LAScriterionKeepPointSource(I32 point_source_id) { this->point_source_id = point_source_id; };
private:
  I32 point_source_id;
};

class LAScriterionKeepPointSourceBetween : public LAScriterion
{
public:
  inline const char* name() const { return "keep_point_source_between"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_point_source_id, above_point_source_id); };
  inline BOOL filter(const LASpoint* point) { return (point->point_source_ID < below_point_source_id) || (above_point_source_id < point->point_source_ID); };
  LAScriterionKeepPointSourceBetween(I32 below_point_source_id, I32 above_point_source_id) { this->below_point_source_id = below_point_source_id; this->above_point_source_id = above_point_source_id; };
private:
  I32 below_point_source_id, above_point_source_id;
};

class LAScriterionDropPointSourceBelow : public LAScriterion
{
public:
  inline const char* name() const { return "drop_point_source_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), below_point_source_id); };
  inline BOOL filter(const LASpoint* point) { return (point->point_source_ID < below_point_source_id) ; };
  LAScriterionDropPointSourceBelow(I32 below_point_source_id) { this->below_point_source_id = below_point_source_id; };
private:
  I32 below_point_source_id;
};

class LAScriterionDropPointSourceAbove : public LAScriterion
{
public:
  inline const char* name() const { return "drop_point_source_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), above_point_source_id); };
  inline BOOL filter(const LASpoint* point) { return (point->point_source_ID > above_point_source_id); };
  LAScriterionDropPointSourceAbove(I32 above_point_source_id) { this->above_point_source_id = above_point_source_id; };
private:
  I32 above_point_source_id;
};

class LAScriterionDropPointSourceBetween : public LAScriterion
{
public:
  inline const char* name() const { return "drop_point_source_between"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d %d ", name(), below_point_source_id, above_point_source_id); };
  inline BOOL filter(const LASpoint* point) { return (below_point_source_id <= point->point_source_ID) && (point->point_source_ID <= above_point_source_id); };
  LAScriterionDropPointSourceBetween(I32 below_point_source_id, I32 above_point_source_id) { this->below_point_source_id = below_point_source_id; this->above_point_source_id = above_point_source_id; };
private:
  I32 below_point_source_id, above_point_source_id;
};

class LAScriterionKeepGpsTime : public LAScriterion
{
public:
  inline const char* name() const { return "keep_gps_time"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g ", name(), below_gpstime, above_gpstime); };
  inline BOOL filter(const LASpoint* point) { return (point->have_gps_time && ((point->gps_time < below_gpstime) || (point->gps_time > above_gpstime))); };
  LAScriterionKeepGpsTime(F64 below_gpstime, F64 above_gpstime) { this->below_gpstime = below_gpstime; this->above_gpstime = above_gpstime; };
private:
  F64 below_gpstime, above_gpstime;
};

class LAScriterionDropGpsTimeBelow : public LAScriterion
{
public:
  inline const char* name() const { return "drop_gps_time_below"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), below_gpstime); };
  inline BOOL filter(const LASpoint* point) { return (point->have_gps_time && (point->gps_time < below_gpstime)); };
  LAScriterionDropGpsTimeBelow(F64 below_gpstime) { this->below_gpstime = below_gpstime; };
private:
  F64 below_gpstime;
};

class LAScriterionDropGpsTimeAbove : public LAScriterion
{
public:
  inline const char* name() const { return "drop_gps_time_above"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), above_gpstime); };
  inline BOOL filter(const LASpoint* point) { return (point->have_gps_time && (point->gps_time > above_gpstime)); };
  LAScriterionDropGpsTimeAbove(F64 above_gpstime) { this->above_gpstime = above_gpstime; };
private:
  F64 above_gpstime;
};

class LAScriterionDropGpsTimeBetween : public LAScriterion
{
public:
  inline const char* name() const { return "drop_gps_time_between"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g %g ", name(), below_gpstime, above_gpstime); };
  inline BOOL filter(const LASpoint* point) { return (point->have_gps_time && ((below_gpstime <= point->gps_time) && (point->gps_time <= above_gpstime))); };
  LAScriterionDropGpsTimeBetween(F64 below_gpstime, F64 above_gpstime) { this->below_gpstime = below_gpstime; this->above_gpstime = above_gpstime; };
private:
  F64 below_gpstime, above_gpstime;
};

class LAScriterionKeepWavepackets : public LAScriterion
{
public:
  inline const char* name() const { return "keep_wavepacket_mask"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %u ", name(), ~drop_wavepacket_mask); };
  inline BOOL filter(const LASpoint* point) { return ((1 << point->wavepacket.getIndex()) & drop_wavepacket_mask); };
  LAScriterionKeepWavepackets(U32 keep_wavepacket_mask) { drop_wavepacket_mask = ~keep_wavepacket_mask; };
private:
  U32 drop_wavepacket_mask;
};

class LAScriterionKeepEveryNth : public LAScriterion
{
public:
  inline const char* name() const { return "keep_every_nth"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %d ", name(), every); };
  inline BOOL filter(const LASpoint* point) { if (counter == every) { counter = 1; return FALSE; } else { counter++; return TRUE; } };
  LAScriterionKeepEveryNth(I32 every) { this->every = every; counter = 1; };
private:
  I32 counter;
  I32 every;
};

class LAScriterionKeepRandomFraction : public LAScriterion
{
public:
  inline const char* name() const { return "keep_random_fraction"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), fraction); };
  inline BOOL filter(const LASpoint* point) { F32 f = (F32)rand()/(F32)RAND_MAX; return f > fraction; };
  LAScriterionKeepRandomFraction(F32 fraction) { this->fraction = fraction; };
private:
  F32 fraction;
};

class LAScriterionThinWithGrid : public LAScriterion
{
public:
  inline const char* name() const { return "thin_with_grid"; };
  inline int get_command(char* string) const { return sprintf(string, "-%s %g ", name(), (grid_spacing > 0 ? grid_spacing : -grid_spacing)); };
  inline BOOL filter(const LASpoint* point)
  { 
    if (grid_spacing < 0)
    {
      grid_spacing = -grid_spacing;
      anker = I32_FLOOR(point->get_y() / grid_spacing);
    }
    I32 pos_x = I32_FLOOR(point->get_x() / grid_spacing);
    I32 pos_y = I32_FLOOR(point->get_y() / grid_spacing) - anker;
    BOOL no_x_anker = FALSE;
    U32* array_size;
    I32** ankers;
    U32*** array;
    U16** array_sizes;
    if (pos_y < 0)
    {
      pos_y = -pos_y - 1;
      ankers = &minus_ankers;
      if ((U32)pos_y < minus_plus_size && minus_plus_sizes[pos_y])
      {
        pos_x -= minus_ankers[pos_y];
        if (pos_x < 0)
        {
          pos_x = -pos_x - 1;
          array_size = &minus_minus_size;
          array = &minus_minus;
          array_sizes = &minus_minus_sizes;
        }
        else
        {
          array_size = &minus_plus_size;
          array = &minus_plus;
          array_sizes = &minus_plus_sizes;
        }
      }
      else
      {
        no_x_anker = TRUE;
        array_size = &minus_plus_size;
        array = &minus_plus;
        array_sizes = &minus_plus_sizes;
      }
    }
    else
    {
      ankers = &plus_ankers;
      if ((U32)pos_y < plus_plus_size && plus_plus_sizes[pos_y])
      {
        pos_x -= plus_ankers[pos_y];
        if (pos_x < 0)
        {
          pos_x = -pos_x - 1;
          array_size = &plus_minus_size;
          array = &plus_minus;
          array_sizes = &plus_minus_sizes;
        }
        else
        {
          array_size = &plus_plus_size;
          array = &plus_plus;
          array_sizes = &plus_plus_sizes;
        }
      }
      else
      {
        no_x_anker = TRUE;
        array_size = &plus_plus_size;
        array = &plus_plus;
        array_sizes = &plus_plus_sizes;
      }
    }
    // maybe grow banded grid in y direction
    if ((U32)pos_y >= *array_size)
    {
      U32 array_size_new = ((pos_y/1024)+1)*1024;
      if (*array_size)
      {
        if (array == &minus_plus || array == &plus_plus) *ankers = (I32*)realloc(*ankers, array_size_new*sizeof(I32));
        *array = (U32**)realloc(*array, array_size_new*sizeof(U32*));
        *array_sizes = (U16*)realloc(*array_sizes, array_size_new*sizeof(U16));
      }
      else
      {
        if (array == &minus_plus || array == &plus_plus) *ankers = (I32*)malloc(array_size_new*sizeof(I32));
        *array = (U32**)malloc(array_size_new*sizeof(U32*));
        *array_sizes = (U16*)malloc(array_size_new*sizeof(U16));
      }
      for (U32 i = *array_size; i < array_size_new; i++)
      {
        (*array)[i] = 0;
        (*array_sizes)[i] = 0;
      }
      *array_size = array_size_new;
    }
    // is this the first x anker for this y pos?
    if (no_x_anker)
    {
      (*ankers)[pos_y] = pos_x;
      pos_x = 0;
    }
    // maybe grow banded grid in x direction
    U32 pos_x_pos = pos_x/32;
    if (pos_x_pos >= (*array_sizes)[pos_y])
    {
      U32 array_sizes_new = ((pos_x_pos/256)+1)*256;
      if ((*array_sizes)[pos_y])
      {
        (*array)[pos_y] = (U32*)realloc((*array)[pos_y], array_sizes_new*sizeof(U32));
      }
      else
      {
        (*array)[pos_y] = (U32*)malloc(array_sizes_new*sizeof(U32));
      }
      for (U16 i = (*array_sizes)[pos_y]; i < array_sizes_new; i++)
      {
        (*array)[pos_y][i] = 0;
      }
      (*array_sizes)[pos_y] = array_sizes_new;
    }
    U32 pos_x_bit = 1 << (pos_x%32);
    if ((*array)[pos_y][pos_x_pos] & pos_x_bit) return TRUE;
    (*array)[pos_y][pos_x_pos] |= pos_x_bit;
    return FALSE;
  }
  void reset()
  {
    if (grid_spacing > 0) grid_spacing = -grid_spacing;
    if (minus_minus_size)
    {
      for (U32 i = 0; i < minus_minus_size; i++) if (minus_minus[i]) free(minus_minus[i]);
      free(minus_minus);
      minus_minus = 0;
      free(minus_minus_sizes);
      minus_minus_sizes = 0;
      minus_minus_size = 0;
    }
    if (minus_plus_size)
    {
      free(minus_ankers);
      minus_ankers = 0;
      for (U32 i = 0; i < minus_plus_size; i++) if (minus_plus[i]) free(minus_plus[i]);
      free(minus_plus);
      minus_plus = 0;
      free(minus_plus_sizes);
      minus_plus_sizes = 0;
      minus_plus_size = 0;
    }
    if (plus_minus_size)
    {
      for (U32 i = 0; i < plus_minus_size; i++) if (plus_minus[i]) free(plus_minus[i]);
      free(plus_minus);
      plus_minus = 0;
      free(plus_minus_sizes);
      plus_minus_sizes = 0;
      plus_minus_size = 0;
    }
    if (plus_plus_size)
    {
      free(plus_ankers);
      plus_ankers = 0;
      for (U32 i = 0; i < plus_plus_size; i++) if (plus_plus[i]) free(plus_plus[i]);
      free(plus_plus);
      plus_plus = 0;
      free(plus_plus_sizes);
      plus_plus_sizes = 0;
      plus_plus_size = 0;
    }
  };
  LAScriterionThinWithGrid(F32 grid_spacing)
  {
    this->grid_spacing = -grid_spacing;
    minus_ankers = 0;
    minus_minus_size = 0;
    minus_minus = 0;
    minus_minus_sizes = 0;
    minus_plus_size = 0;
    minus_plus = 0;
    minus_plus_sizes = 0;
    plus_ankers = 0;
    plus_minus_size = 0;
    plus_minus = 0;
    plus_minus_sizes = 0;
    plus_plus_size = 0;
    plus_plus = 0;
    plus_plus_sizes = 0;
  };
  ~LAScriterionThinWithGrid() { reset(); };
private:
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
};

void LASfilter::clean()
{
  U32 i;
  for (i = 0; i < num_criteria; i++)
  {
    delete criteria[i];
  }
  if (criteria) delete [] criteria;
  if (counters) delete [] counters;
  alloc_criteria = 0;
  num_criteria = 0;
  criteria = 0;
  counters = 0;
}

void LASfilter::usage() const
{
  fprintf(stderr,"Filter points based on their coordinates.\n");
  fprintf(stderr,"  -clip_tile 631000 4834000 1000 (ll_x, ll_y, size)\n");
  fprintf(stderr,"  -clip_circle 630250.00 4834750.00 100 (x, y, radius)\n");
  fprintf(stderr,"  -clip 630000 4834000 631000 4836000 (min_x, min_y, max_x, max_y)\n");
  fprintf(stderr,"  -clip_x_below 630000.50 (min_x)\n");
  fprintf(stderr,"  -clip_y_below 4834500.25 (min_y)\n");
  fprintf(stderr,"  -clip_x_above 630500.50 (max_x)\n");
  fprintf(stderr,"  -clip_y_above 4836000.75 (max_y)\n");
  fprintf(stderr,"  -clip_z 11.125 130.725 (min_z, max_z)\n");
  fprintf(stderr,"  -clip_z_below 11.125 (min_z)\n");
  fprintf(stderr,"  -clip_z_above 130.725 (max_z)\n");
  fprintf(stderr,"Filter points based on their return number.\n");
  fprintf(stderr,"  -first_only -keep_first -drop_first\n");
  fprintf(stderr,"  -last_only -keep_last -drop_last\n");
  fprintf(stderr,"  -keep_middle -drop_middle\n");
  fprintf(stderr,"  -keep_return 1 2 3\n");
  fprintf(stderr,"  -drop_return 3 4\n");
  fprintf(stderr,"  -keep_single -drop_single\n");
  fprintf(stderr,"  -keep_double -drop_double\n");
  fprintf(stderr,"  -keep_triple -drop_triple\n");
  fprintf(stderr,"  -keep_quadruple -drop_quadruple\n");
  fprintf(stderr,"  -keep_quintuple -drop_quintuple\n");
  fprintf(stderr,"Filter points based on the scanline flags.\n");
  fprintf(stderr,"  -drop_scan_direction 0\n");
  fprintf(stderr,"  -scan_direction_change_only\n");
  fprintf(stderr,"  -edge_of_flight_line_only\n");
  fprintf(stderr,"Filter points based on their intensity.\n");
  fprintf(stderr,"  -keep_intensity 20 380\n");
  fprintf(stderr,"  -drop_intensity_below 20\n");
  fprintf(stderr,"  -drop_intensity_above 380\n");
  fprintf(stderr,"  -drop_intensity_between 4000 5000\n");
  fprintf(stderr,"Filter points based on their classification.\n");
  fprintf(stderr,"  -keep_class 1 3 7\n");
  fprintf(stderr,"  -drop_class 4 2\n");
  fprintf(stderr,"Filter points based on their point source ID.\n");
  fprintf(stderr,"  -keep_point_source 3\n");
  fprintf(stderr,"  -keep_point_source_between 2 6\n");
  fprintf(stderr,"  -drop_point_source_below 6\n");
  fprintf(stderr,"  -drop_point_source_above 15\n");
  fprintf(stderr,"  -drop_point_source_between 17 21\n");
  fprintf(stderr,"Filter points based on their scan angle.\n");
  fprintf(stderr,"  -keep_scan_angle -15 15\n");
  fprintf(stderr,"  -drop_scan_angle_below -15\n");
  fprintf(stderr,"  -drop_scan_angle_above 15\n");
  fprintf(stderr,"  -drop_scan_angle_between -25 -23\n");
  fprintf(stderr,"Filter points based on their gps time.\n");
  fprintf(stderr,"  -keep_gps_time 11.125 130.725\n");
  fprintf(stderr,"  -drop_gps_time_below 11.125\n");
  fprintf(stderr,"  -drop_gps_time_above 130.725\n");
  fprintf(stderr,"  -drop_gps_time_between 22.0 48.0\n");
  fprintf(stderr,"Filter points based on their wavepacket.\n");
  fprintf(stderr,"  -keep_wavepacket 1 2\n");
  fprintf(stderr,"  -drop_wavepacket 0\n");
  fprintf(stderr,"Filter points with simple thinning.\n");
  fprintf(stderr,"  -keep_every_nth 2\n");
  fprintf(stderr,"  -keep_random_fraction 0.1\n");
  fprintf(stderr,"  -thin_with_grid 1.0\n");
}

BOOL LASfilter::parse(int argc, char* argv[])
{
  int i;

  U32 keep_return_mask = 0;
  U32 drop_return_mask = 0;

  U32 keep_classification_mask = 0;
  U32 drop_classification_mask = 0;

  U32 keep_wavepacket_mask = 0;
  U32 drop_wavepacket_mask = 0;

  for (i = 1; i < argc; i++)
  {
    if (argv[i][0] == '\0')
    {
      continue;
    }
    else if (strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"-help") == 0)
    {
      usage();
      return TRUE;
    }
    else if (strcmp(argv[i],"-clip_tile") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: llx lly size\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipTile((F32)atof(argv[i+1]), (F32)atof(argv[i+2]), (F32)atof(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3; 
    }
    else if (strcmp(argv[i],"-clip_circle") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: center_x center_y radius\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipCircle(atof(argv[i+1]), atof(argv[i+2]), atof(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3;
    }
    else if (strcmp(argv[i],"-clip") == 0 || strcmp(argv[i],"-clip_xy") == 0)
    {
      if ((i+4) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 4 arguments: min_x min_y max_x max_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipXY(atof(argv[i+1]), atof(argv[i+2]), atof(argv[i+3]), atof(argv[i+4])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; *argv[i+4]='\0'; i+=4; 
    }
    else if (strcmp(argv[i],"-clip_z") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min_z max_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipZ(atof(argv[i+1]), atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-clip_x_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_x\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipXBelow(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_y_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipYBelow(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_z_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipZBelow(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_x_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_x\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipXAbove(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_y_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipYAbove(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_z_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipZAbove(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw") == 0 || strcmp(argv[i],"-clip_raw_xy") == 0)
    {
      if ((i+4) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 4 arguments: min_raw_x min_raw_y max_raw_x max_raw_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawXY(atoi(argv[i+1]), atoi(argv[i+2]), atoi(argv[i+3]), atoi(argv[i+4])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; *argv[i+4]='\0'; i+=4; 
    }
    else if (strcmp(argv[i],"-clip_raw_z") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min_raw_z max_raw_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawZ(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2; 
    }
    else if (strcmp(argv[i],"-clip_raw_x_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_raw_x\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawXBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw_y_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_raw_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawYBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw_z_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_raw_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawZBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw_x_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_raw_x\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawXAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw_y_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_raw_y\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawYAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clip_raw_z_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_raw_z\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionClipRawZAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if ((strcmp(argv[i],"-first_only") == 0) || (strcmp(argv[i],"-keep_first") == 0))
    {
      add_criterion(new LAScriterionKeepFirstReturn());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_middle") == 0)
    {
      add_criterion(new LAScriterionKeepMiddleReturn());
      *argv[i]='\0';
    }
    else if ((strcmp(argv[i],"-last_only") == 0) || (strcmp(argv[i],"-keep_last") == 0))
    {
      add_criterion(new LAScriterionKeepLastReturn());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_first") == 0)
    {
      add_criterion(new LAScriterionDropFirstReturn());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_middle") == 0)
    {
      add_criterion(new LAScriterionDropMiddleReturn());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_last") == 0)
    {
      add_criterion(new LAScriterionDropLastReturn());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_return") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs at least 1 argument: return_number\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        keep_return_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-keep_return_mask") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: return_mask\n", argv[i]);
        return FALSE;
      }
      keep_return_mask = atoi(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_return") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs at least 1 argument: return_number\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        drop_return_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-keep_single") == 0 || strcmp(argv[i],"-single_only") == 0)
    {
      add_criterion(new LAScriterionKeepSpecificNumberOfReturns(1));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_double") == 0 || strcmp(argv[i],"-double_only") == 0)
    {
      add_criterion(new LAScriterionKeepSpecificNumberOfReturns(2));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_triple") == 0 || strcmp(argv[i],"-triple_only") == 0)
    {
      add_criterion(new LAScriterionKeepSpecificNumberOfReturns(3));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_quadruple") == 0 || strcmp(argv[i],"-quadruple_only") == 0)
    {
      add_criterion(new LAScriterionKeepSpecificNumberOfReturns(4));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_quintuple") == 0 || strcmp(argv[i],"-quintuple_only") == 0)
    {
      add_criterion(new LAScriterionKeepSpecificNumberOfReturns(5));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_single") == 0)
    {
      add_criterion(new LAScriterionDropSpecificNumberOfReturns(1));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_double") == 0)
    {
      add_criterion(new LAScriterionDropSpecificNumberOfReturns(2));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_triple") == 0)
    {
      add_criterion(new LAScriterionDropSpecificNumberOfReturns(3));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_quadruple") == 0)
    {
      add_criterion(new LAScriterionDropSpecificNumberOfReturns(4));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_quintuple") == 0)
    {
      add_criterion(new LAScriterionDropSpecificNumberOfReturns(5));
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-drop_scan_direction") == 0)
    {
      add_criterion(new LAScriterionDropScanDirection(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-scan_direction_change_only") == 0 || strcmp(argv[i],"-scan_direction_change") == 0)
    {
      add_criterion(new LAScriterionScanDirectionChangeOnly());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-edge_of_flight_line_only") == 0 || strcmp(argv[i],"-edge_of_flight_line") == 0)
    {
      add_criterion(new LAScriterionEdgeOfFlightLineOnly());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-keep_intensity") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepIntensity(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-drop_intensity_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropIntensityAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_intensity_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropIntensityBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_intensity_between") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropIntensityBetween(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-keep_scan_angle") == 0 || strcmp(argv[i],"-keep_scan") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepScanAngle(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-drop_scan_angle_above") == 0 || strcmp(argv[i],"-drop_scan_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropScanAngleAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_scan_angle_below") == 0 || strcmp(argv[i],"-drop_scan_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropScanAngleBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }    
    else if (strcmp(argv[i],"-drop_scan_angle_between") == 0 || strcmp(argv[i],"-drop_scan_between") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropScanAngleBetween(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-keep_classification") == 0 || strcmp(argv[i],"-keep_class") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 at least argument: classification\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        keep_classification_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-keep_classification_mask") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: classifications_mask\n", argv[i]);
        return FALSE;
      }
      keep_classification_mask = atoi(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_classification") == 0 || strcmp(argv[i],"-drop_class") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs at least 1 argument: classification\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        drop_classification_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-keep_wavepacket") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 at least argument: index\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        keep_wavepacket_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-drop_wavepacket") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs at least 1 argument: index\n", argv[i]);
        return FALSE;
      }
      *argv[i]='\0';
      i+=1;
      do
      {
        drop_wavepacket_mask |= (1 << atoi(argv[i]));
        *argv[i]='\0';
        i+=1;
      } while ((i < argc) && ('0' <= *argv[i]) && (*argv[i] <= '9'));
      i-=1;
    }
    else if (strcmp(argv[i],"-keep_point_source") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: ID\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepPointSource(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-keep_point_source_between") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min_ID max_ID\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepPointSourceBetween(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-drop_point_source_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_ID\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropPointSourceBelow(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_point_source_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_ID\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropPointSourceAbove(atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_point_source_between") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min_ID max_ID\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropPointSourceBetween(atoi(argv[i+1]), atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-keep_gps_time") == 0 || strcmp(argv[i],"-keep_gpstime") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepGpsTime(atof(argv[i+1]), atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-drop_gps_time_above") == 0 || strcmp(argv[i],"-drop_gpstime_above") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max_gps_time\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropGpsTimeAbove(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_gps_time_below") == 0 || strcmp(argv[i],"-drop_gpstime_below") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min_gps_time\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropGpsTimeBelow(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-drop_gps_time_between") == 0 || strcmp(argv[i],"-drop_gpstime_between") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min max\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionDropGpsTimeBetween(atof(argv[i+1]), atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-keep_every_nth") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: nth\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepEveryNth((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-keep_random_fraction") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: fraction\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionKeepRandomFraction((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-thin_with_grid") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: grid_spacing\n", argv[i]);
        return FALSE;
      }
      add_criterion(new LAScriterionThinWithGrid((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
  }

  if (drop_return_mask)
  {
    if (keep_return_mask == 0) keep_return_mask = 255 & ~drop_return_mask;
  }
  if (keep_return_mask) add_criterion(new LAScriterionKeepReturns(keep_return_mask));

  if (drop_classification_mask)
  {
    if (keep_classification_mask == 0) keep_classification_mask = ~drop_classification_mask;
  }
  if (keep_classification_mask) add_criterion(new LAScriterionKeepClassifications(keep_classification_mask));

  if (drop_wavepacket_mask)
  {
    if (keep_wavepacket_mask == 0) keep_wavepacket_mask = ~drop_wavepacket_mask;
  }
  if (keep_wavepacket_mask) add_criterion(new LAScriterionKeepWavepackets(keep_wavepacket_mask));

  return TRUE;
}

I32 LASfilter::unparse(char* string) const
{
  U32 i;
  I32 n = 0;
  for (i = 0; i < num_criteria; i++)
  {
    n += criteria[i]->get_command(&string[n]);
  }
  return n;
}

void LASfilter::addClipCircle(F64 x, F64 y, F64 radius)
{
  add_criterion(new LAScriterionClipCircle(x, y, radius));
}

void LASfilter::addScanDirectionChangeOnly()
{
  add_criterion(new LAScriterionScanDirectionChangeOnly());
}

BOOL LASfilter::filter(const LASpoint* point)
{
  U32 i;

  for (i = 0; i < num_criteria; i++)
  {
    if (criteria[i]->filter(point))
    {
      counters[i]++;
      return TRUE; // point was filtered
    }
  }
  return FALSE; // point survived
}

void LASfilter::reset()
{
  U32 i;
  for (i = 0; i < num_criteria; i++)
  {
    criteria[i]->reset();
  }
}

LASfilter::LASfilter()
{
  alloc_criteria = 0;
  num_criteria = 0;
  criteria = 0;
  counters = 0;
}

LASfilter::~LASfilter()
{
  if (criteria) clean();
}

void LASfilter::add_criterion(LAScriterion* filter_criterion)
{
  if (num_criteria == alloc_criteria)
  {
    U32 i;
    alloc_criteria += 16;
    LAScriterion** temp_criteria = new LAScriterion*[alloc_criteria];
    int* temp_counters = new int[alloc_criteria];
    if (criteria)
    {
      for (i = 0; i < num_criteria; i++)
      {
        temp_criteria[i] = criteria[i];
        temp_counters[i] = counters[i];
      }
      delete [] criteria;
      delete [] counters;
    }
    criteria = temp_criteria;
    counters = temp_counters;
  }
  criteria[num_criteria] = filter_criterion;
  counters[num_criteria] = 0;
  num_criteria++;
}
