/*
===============================================================================

  FILE:  lasutility.cpp
  
  CONTENTS:
  
    see corresponding header file
  
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
  
    see corresponding header file
  
===============================================================================
*/
#include "lasutility.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

LASinventory::LASinventory()
{
  U32 i;
  number_of_point_records = 0;
  for (i = 0; i < 8; i++) number_of_points_by_return[i] = 0;
  raw_max_x = raw_min_x = 0;
  raw_max_y = raw_min_y = 0;
  raw_max_z = raw_min_z = 0;
  first = true;
}

BOOL LASinventory::add(const LASpoint* point)
{
  number_of_point_records++;
  number_of_points_by_return[point->return_number]++;
  if (first)
  {
    raw_min_x = raw_max_x = point->x;
    raw_min_y = raw_max_y = point->y;
    raw_min_z = raw_max_z = point->z;
    first = false;
  }
  else
  {
    if (point->x < raw_min_x) raw_min_x = point->x;
    else if (point->x > raw_max_x) raw_max_x = point->x;
    if (point->y < raw_min_y) raw_min_y = point->y;
    else if (point->y > raw_max_y) raw_max_y = point->y;
    if (point->z < raw_min_z) raw_min_z = point->z;
    else if (point->z > raw_max_z) raw_max_z = point->z;
  }
  return TRUE;
}

LASsummary::LASsummary()
{
  U32 i;
  number_of_point_records = 0;
  for (i = 0; i < 8; i++) number_of_points_by_return[i] = 0;
  for (i = 0; i < 8; i++) number_of_returns_of_given_pulse[i] = 0;
  for (i = 0; i < 32; i++) classification[i] = 0;
  classification_synthetic = 0;
  classification_keypoint = 0;
  classification_withheld = 0;
  first = true;
}

BOOL LASsummary::add(const LASpoint* point)
{
  number_of_point_records++;
  number_of_points_by_return[point->return_number]++;
  number_of_returns_of_given_pulse[point->number_of_returns_of_given_pulse]++;
  classification[point->classification&31]++;
  if (point->classification & 32) classification_synthetic++;
  if (point->classification & 64) classification_keypoint++;
  if (point->classification & 128) classification_withheld++;
  if (first)
  {
    min = *point;
    max = *point;
    first = false;
  }
  else
  {
    if (point->x < min.x) min.x = point->x;
    else if (point->x > max.x) max.x = point->x;
    if (point->y < min.y) min.y = point->y;
    else if (point->y > max.y) max.y = point->y;
    if (point->z < min.z) min.z = point->z;
    else if (point->z > max.z) max.z = point->z;
    if (point->intensity < min.intensity) min.intensity = point->intensity;
    else if (point->intensity > max.intensity) max.intensity = point->intensity;
    if (point->edge_of_flight_line < min.edge_of_flight_line) min.edge_of_flight_line = point->edge_of_flight_line;
    else if (point->edge_of_flight_line > max.edge_of_flight_line) max.edge_of_flight_line = point->edge_of_flight_line;
    if (point->scan_direction_flag < min.scan_direction_flag) min.scan_direction_flag = point->scan_direction_flag;
    else if (point->scan_direction_flag > max.scan_direction_flag) max.scan_direction_flag = point->scan_direction_flag;
    if (point->number_of_returns_of_given_pulse < min.number_of_returns_of_given_pulse) min.number_of_returns_of_given_pulse = point->number_of_returns_of_given_pulse;
    else if (point->number_of_returns_of_given_pulse > max.number_of_returns_of_given_pulse) max.number_of_returns_of_given_pulse = point->number_of_returns_of_given_pulse;
    if (point->return_number < min.return_number) min.return_number = point->return_number;
    else if (point->return_number > max.return_number) max.return_number = point->return_number;
    if (point->classification < min.classification) min.classification = point->classification;
    else if (point->classification > max.classification) max.classification = point->classification;
    if (point->scan_angle_rank < min.scan_angle_rank) min.scan_angle_rank = point->scan_angle_rank;
    else if (point->scan_angle_rank > max.scan_angle_rank) max.scan_angle_rank = point->scan_angle_rank;
    if (point->user_data < min.user_data) min.user_data = point->user_data;
    else if (point->user_data > max.user_data) max.user_data = point->user_data;
    if (point->point_source_ID < min.point_source_ID) min.point_source_ID = point->point_source_ID;
    else if (point->point_source_ID > max.point_source_ID) max.point_source_ID = point->point_source_ID;
    if (point->point_source_ID < min.point_source_ID) min.point_source_ID = point->point_source_ID;
    else if (point->point_source_ID > max.point_source_ID) max.point_source_ID = point->point_source_ID;
    if (point->have_gps_time)
    {
      if (point->gps_time < min.gps_time) min.gps_time = point->gps_time;
      else if (point->gps_time > max.gps_time) max.gps_time = point->gps_time;
    }
    if (point->have_rgb)
    {
      if (point->rgb[0] < min.rgb[0]) min.rgb[0] = point->rgb[0];
      else if (point->rgb[0] > max.rgb[0]) max.rgb[0] = point->rgb[0];
      if (point->rgb[1] < min.rgb[1]) min.rgb[1] = point->rgb[1];
      else if (point->rgb[1] > max.rgb[1]) max.rgb[1] = point->rgb[1];
      if (point->rgb[2] < min.rgb[2]) min.rgb[2] = point->rgb[2];
      else if (point->rgb[2] > max.rgb[2]) max.rgb[2] = point->rgb[2];
    }
    if (point->have_wavepacket)
    {
      if (point->wavepacket.getIndex() < min.wavepacket.getIndex()) min.wavepacket.setIndex(point->wavepacket.getIndex());
      else if (point->wavepacket.getIndex() > max.wavepacket.getIndex()) max.wavepacket.setIndex(point->wavepacket.getIndex());
      if (point->wavepacket.getOffset() < min.wavepacket.getOffset()) min.wavepacket.setOffset(point->wavepacket.getOffset());
      else if (point->wavepacket.getOffset() > max.wavepacket.getOffset()) max.wavepacket.setOffset(point->wavepacket.getOffset());
      if (point->wavepacket.getSize() < min.wavepacket.getSize()) min.wavepacket.setSize(point->wavepacket.getSize());
      else if (point->wavepacket.getSize() > max.wavepacket.getSize()) max.wavepacket.setSize(point->wavepacket.getSize());
      if (point->wavepacket.getLocation() < min.wavepacket.getLocation()) min.wavepacket.setLocation(point->wavepacket.getLocation());
      else if (point->wavepacket.getLocation() > max.wavepacket.getLocation()) max.wavepacket.setLocation(point->wavepacket.getLocation());
      if (point->wavepacket.getXt() < min.wavepacket.getXt()) min.wavepacket.setXt(point->wavepacket.getXt());
      else if (point->wavepacket.getXt() > max.wavepacket.getXt()) max.wavepacket.setXt(point->wavepacket.getXt());
      if (point->wavepacket.getYt() < min.wavepacket.getYt()) min.wavepacket.setYt(point->wavepacket.getYt());
      else if (point->wavepacket.getYt() > max.wavepacket.getYt()) max.wavepacket.setYt(point->wavepacket.getYt());
      if (point->wavepacket.getZt() < min.wavepacket.getZt()) min.wavepacket.setZt(point->wavepacket.getZt());
      else if (point->wavepacket.getZt() > max.wavepacket.getZt()) max.wavepacket.setZt(point->wavepacket.getZt());
    }
  }
  return TRUE;
}

LASbin::LASbin(F32 step)
{
  total = 0;
  count = 0;
  this->one_over_step = 1.0f/step;
  first = TRUE;
  size_pos = 0;
  size_neg = 0;
  bins_pos = 0;
  bins_neg = 0;
  values_pos = 0;
  values_neg = 0;
}

LASbin::~LASbin()
{
  if (bins_pos) free(bins_pos);
  if (bins_neg) free(bins_neg);
  if (values_pos) free(values_pos);
  if (values_neg) free(values_neg);
}

void LASbin::add(I32 item)
{
  total += item;
  count++;
  I32 bin = I32_FLOOR(one_over_step*item);
  add_to_bin(bin);
}

void LASbin::add(F64 item)
{
  total += item;
  count++;
  I32 bin = I32_FLOOR(one_over_step*item);
  add_to_bin(bin);
}

void LASbin::add(I64 item)
{
  total += item;
  count++;
  I32 bin = I32_FLOOR(one_over_step*item);
  add_to_bin(bin);
}

void LASbin::add_to_bin(I32 bin)
{
  if (first)
  {
    anker = bin;
    first = FALSE;
  }
  bin = bin - anker;
  if (bin >= 0)
  {
    if (bin >= size_pos)
    {
      I32 i;
      if (size_pos == 0)
      {
        size_pos = 1024;
        bins_pos = (U32*)malloc(sizeof(U32)*size_pos);
        for (i = 0; i < size_pos; i++) bins_pos[i] = 0;
      }
      else
      {
        I32 new_size = bin + 1024;
        bins_pos = (U32*)realloc(bins_pos, sizeof(U32)*new_size);
        for (i = size_pos; i < new_size; i++) bins_pos[i] = 0;
        size_pos = new_size;
      }
    }
    bins_pos[bin]++;
  }
  else
  {
    bin = -(bin+1);
    if (bin >= size_neg)
    {
      I32 i;
      if (size_neg == 0)
      {
        size_neg = 1024;
        bins_neg = (U32*)malloc(sizeof(U32)*size_neg);
        for (i = 0; i < size_neg; i++) bins_neg[i] = 0;
      }
      else
      {
        I32 new_size = bin + 1024;
        bins_neg = (U32*)realloc(bins_neg, sizeof(U32)*new_size);
        for (i = size_neg; i < new_size; i++) bins_neg[i] = 0;
        size_neg = new_size;
      }
    }
    bins_neg[bin]++;
  }
}

void LASbin::add(I32 item, I32 value)
{
  total += item;
  count++;
  I32 bin = I32_FLOOR(one_over_step*item);
  if (first)
  {
    anker = bin;
    first = FALSE;
  }
  bin = bin - anker;
  if (bin >= 0)
  {
    if (bin >= size_pos)
    {
      I32 i;
      if (size_pos == 0)
      {
        size_pos = 1024;
        bins_pos = (U32*)malloc(sizeof(U32)*size_pos);
        values_pos = (F64*)malloc(sizeof(F64)*size_pos);
        for (i = 0; i < size_pos; i++) { bins_pos[i] = 0; values_pos[i] = 0; }
      }
      else
      {
        I32 new_size = bin + 1024;
        bins_pos = (U32*)realloc(bins_pos, sizeof(U32)*new_size);
        values_pos = (F64*)realloc(values_pos, sizeof(F64)*new_size);
        for (i = size_pos; i < new_size; i++) { bins_pos[i] = 0; values_pos[i] = 0; }
        size_pos = new_size;
      }
    }
    bins_pos[bin]++;
    values_pos[bin] += value;
  }
  else
  {
    bin = -(bin+1);
    if (bin >= size_neg)
    {
      I32 i;
      if (size_neg == 0)
      {
        size_neg = 1024;
        bins_neg = (U32*)malloc(sizeof(U32)*size_neg);
        values_neg = (F64*)malloc(sizeof(F64)*size_pos);
        for (i = 0; i < size_neg; i++) { bins_neg[i] = 0; values_neg[i] = 0; }
      }
      else
      {
        I32 new_size = bin + 1024;
        bins_neg = (U32*)realloc(bins_neg, sizeof(U32)*new_size);
        values_neg = (F64*)realloc(values_neg, sizeof(F64)*new_size);
        for (i = size_neg; i < new_size; i++) { bins_neg[i] = 0; values_neg[i] = 0; }
        size_neg = new_size;
      }
    }
    bins_neg[bin]++;
    values_neg[bin] += value;
  }
}

void LASbin::report(FILE* file, const char* name, const char* name_avg) const
{
  I32 i, bin;
  if (name)
  {
    if (values_pos)
    {
      if (name_avg)
        fprintf(file, "%s histogram of %s averages with bin size %g\012", name, name_avg, 1.0f/one_over_step);
      else
        fprintf(file, "%s histogram of averages with bin size %g\012", name, 1.0f/one_over_step);
    }
    else
      fprintf(file, "%s histogram with bin size %g\012", name, 1.0f/one_over_step);
  }
  if (size_neg)
  {
    for (i = size_neg-1; i >= 0; i--)
    {
      if (bins_neg[i])
      {
        bin = -(i+1) + anker;
        if (one_over_step == 1)
        {
          if (values_neg)
            fprintf(file, "  bin %d has average %g (of %d)\012", bin, values_neg[i]/bins_neg[i], bins_neg[i]);
          else
            fprintf(file, "  bin %d has %d\012", bin, bins_neg[i]);
        }
        else
        {
          if (values_neg)
            fprintf(file, "  bin [%g,%g) has average %g (of %d)\012", ((F32)bin)/one_over_step, ((F32)(bin+1))/one_over_step, values_neg[i]/bins_neg[i], bins_neg[i]);
          else
            fprintf(file, "  bin [%g,%g) has %d\012", ((F32)bin)/one_over_step, ((F32)(bin+1))/one_over_step, bins_neg[i]);
        }
      }
    }
  }
  if (size_pos)
  {
    for (i = 0; i < size_pos; i++)
    {
      if (bins_pos[i])
      {
        bin = i + anker;
        if (one_over_step == 1)
        {
          if (values_pos)
            fprintf(file, "  bin %d has average %g (of %d)\012", bin, values_pos[i]/bins_pos[i], bins_pos[i]);
          else
            fprintf(file, "  bin %d has %d\012", bin, bins_pos[i]);
        }
        else
        {
          if (values_pos)
            fprintf(file, "  bin [%g,%g) average has %g (of %d)\012", ((F32)bin)/one_over_step, ((F32)(bin+1))/one_over_step, values_pos[i]/bins_pos[i], bins_pos[i]);
          else
            fprintf(file, "  bin [%g,%g) has %d\012", ((F32)bin)/one_over_step, ((F32)(bin+1))/one_over_step, bins_pos[i]);
        }
      }
    }
  }
  if (name)
    fprintf(file, "  average %s %g\012", name, total/count);
  else
    fprintf(file, "  average %g\012", total/count);
}

LAShistogram::LAShistogram()
{
  is_active = FALSE;
  // counter bins
  x_bin = 0;
  y_bin = 0;
  z_bin = 0;
  intensity_bin = 0;
  classification_bin = 0;
  scan_angle_bin = 0;
  point_source_id_bin = 0;
  gps_time_bin = 0;
  wavepacket_index_bin = 0;
  wavepacket_offset_bin = 0;
  wavepacket_size_bin = 0;
  wavepacket_location_bin = 0;
  // averages bins
  classification_bin_intensity = 0;
  classification_bin_scan_angle = 0;
  scan_angle_bin_z = 0;
  scan_angle_bin_intensity = 0;
  scan_angle_bin_number_of_returns = 0;
  return_map_bin_intensity = 0;
}

LAShistogram::~LAShistogram()
{
  // counter bins
  if (x_bin) delete x_bin;
  if (y_bin) delete y_bin;
  if (z_bin) delete z_bin;
  if (intensity_bin) delete intensity_bin;
  if (classification_bin) delete classification_bin;
  if (scan_angle_bin) delete scan_angle_bin;
  if (point_source_id_bin) delete point_source_id_bin;
  if (gps_time_bin) delete gps_time_bin;
  if (wavepacket_index_bin) delete wavepacket_index_bin;
  if (wavepacket_offset_bin) delete wavepacket_offset_bin;
  if (wavepacket_size_bin) delete wavepacket_size_bin;
  if (wavepacket_location_bin) delete wavepacket_location_bin;
  // averages bins
  if (classification_bin_intensity) delete classification_bin_intensity;
  if (classification_bin_scan_angle) delete classification_bin_scan_angle;
  if (scan_angle_bin_z) delete scan_angle_bin_z;
  if (scan_angle_bin_intensity) delete scan_angle_bin_intensity;
  if (scan_angle_bin_number_of_returns) delete scan_angle_bin_number_of_returns;
  if (return_map_bin_intensity) delete return_map_bin_intensity;
}

BOOL LAShistogram::parse(int argc, char* argv[])
{
  int i;
  for (i = 1; i < argc; i++)
  {
    if (argv[i][0] == '\0')
    {
      continue;
    }
    else if (strcmp(argv[i],"-h") == 0 || strcmp(argv[i],"-help") == 0)
    {
      return TRUE;
    }
    else if (strcmp(argv[i],"-histo") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: name step\n", argv[i]);
        return FALSE;
      }
      if (!histo(argv[i+1], (F32)atof(argv[i+2]))) return FALSE;
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2; 
    }
    else if (strcmp(argv[i],"-histo_avg") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: name step name_avg\n", argv[i]);
        return FALSE;
      }
      if (!histo_avg(argv[i+1], (F32)atof(argv[i+2]), argv[i+3])) return FALSE;
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3; 
    }
  }
  return TRUE;
}

BOOL LAShistogram::histo(const char* name, F32 step)
{
  if (strcmp(name, "x") == 0)
    x_bin = new LASbin(step);
  else if (strcmp(name, "y") == 0)
    y_bin = new LASbin(step);
  else if (strcmp(name, "z") == 0)
    z_bin = new LASbin(step);
  else if (strcmp(name, "intensity") == 0)
    intensity_bin = new LASbin(step);
  else if (strcmp(name, "classification") == 0)
    classification_bin = new LASbin(step);
  else if (strstr(name, "scan_angle") != 0)
    scan_angle_bin = new LASbin(step);
  else if (strstr(name, "point_source") != 0)
    point_source_id_bin = new LASbin(step);
  else if (strstr(name, "gps_time") != 0)
    gps_time_bin = new LASbin(step);
  else if (strstr(name, "wavepacket_index") != 0)
    wavepacket_index_bin = new LASbin(step);
  else if (strstr(name, "wavepacket_offset") != 0)
    wavepacket_offset_bin = new LASbin(step);
  else if (strstr(name, "wavepacket_size") != 0)
    wavepacket_size_bin = new LASbin(step);
  else if (strstr(name, "wavepacket_location") != 0)
    wavepacket_location_bin = new LASbin(step);
  else
  {
    fprintf(stderr,"ERROR: histogram of '%s' not implemented\n", name);
    return FALSE;
  }
  is_active = TRUE;
  return TRUE;
}

BOOL LAShistogram::histo_avg(const char* name, F32 step, const char* name_avg)
{
  if (strcmp(name, "classification") == 0)
  {
    if (strcmp(name_avg, "intensity") == 0)
      classification_bin_intensity = new LASbin(step);
    else if (strstr(name_avg, "scan_angle") != 0)
      classification_bin_scan_angle = new LASbin(step);
    else
    {
      fprintf(stderr,"ERROR: histogram of '%s' with '%s' averages not implemented\n", name, name_avg);
      return FALSE;
    }
  }
  else if (strcmp(name, "scan_angle") == 0)
  {
    if (strcmp(name_avg, "z") == 0)
      scan_angle_bin_z = new LASbin(step);
    else if (strcmp(name_avg, "number_of_returns") == 0)
      scan_angle_bin_number_of_returns = new LASbin(step);
    else if (strcmp(name_avg, "intensity") == 0)
      scan_angle_bin_intensity = new LASbin(step);
    else
    {
      fprintf(stderr,"ERROR: histogram of '%s' with '%s' averages not implemented\n", name, name_avg);
      return FALSE;
    }
  }
  else if (strcmp(name, "return_map") == 0)
  {
    if (strcmp(name_avg, "intensity") == 0)
      return_map_bin_intensity = new LASbin(1);
    else
    {
      fprintf(stderr,"ERROR: histogram of '%s' with '%s' averages not implemented\n", name, name_avg);
      return FALSE;
    }
  }
  else
  {
    fprintf(stderr,"ERROR: histogram of '%s' not implemented\n", name);
    return FALSE;
  }
  is_active = TRUE;
  return TRUE;
}

void LAShistogram::add(const LASpoint* point)
{
  // counter bins
  if (x_bin) x_bin->add(point->x);
  if (y_bin) y_bin->add(point->y);
  if (z_bin) z_bin->add(point->z);
  if (intensity_bin) intensity_bin->add(point->intensity);
  if (classification_bin) classification_bin->add(point->classification);
  if (scan_angle_bin) scan_angle_bin->add(point->scan_angle_rank);
  if (point_source_id_bin) point_source_id_bin->add(point->point_source_ID);
  if (gps_time_bin) gps_time_bin->add(point->gps_time);
  if (wavepacket_index_bin) wavepacket_index_bin->add(point->wavepacket.getIndex());
  if (wavepacket_offset_bin) wavepacket_offset_bin->add((I64)point->wavepacket.getOffset());
  if (wavepacket_size_bin) wavepacket_size_bin->add((I32)point->wavepacket.getSize());
  if (wavepacket_location_bin) wavepacket_location_bin->add(point->wavepacket.getLocation());
  // averages bins
  if (classification_bin_intensity) classification_bin_intensity->add(point->classification, point->intensity);
  if (classification_bin_scan_angle) classification_bin_scan_angle->add(point->classification, point->scan_angle_rank);
  if (scan_angle_bin_z) scan_angle_bin_z->add(point->scan_angle_rank, point->z);
  if (scan_angle_bin_number_of_returns) scan_angle_bin_number_of_returns->add(point->scan_angle_rank, point->number_of_returns_of_given_pulse);
  if (scan_angle_bin_intensity) scan_angle_bin_intensity->add(point->scan_angle_rank, point->intensity);
  if (return_map_bin_intensity)
  {
    int n = point->number_of_returns_of_given_pulse;
    int r = point->return_number;
    return_map_bin_intensity->add((n == 1 ? 0 : (n == 2 ? r : (n == 3 ? r+2 : (n == 4 ? r+5 : (n == 5 ? r+9 : 15))))), point->intensity);
  }
}

void LAShistogram::report(FILE* file) const
{
  // counter bins
  if (x_bin) x_bin->report(file, "x coordinate");
  if (y_bin) y_bin->report(file, "y coordinate");
  if (z_bin) z_bin->report(file, "z coordinate");
  if (intensity_bin) intensity_bin->report(file, "intensity");
  if (classification_bin) classification_bin->report(file, "classification");
  if (scan_angle_bin) scan_angle_bin->report(file, "scan angle");
  if (point_source_id_bin) point_source_id_bin->report(file, "point source id");
  if (gps_time_bin) gps_time_bin->report(file, "gps_time");
  if (wavepacket_index_bin) wavepacket_index_bin->report(file, "wavepacket_index");
  if (wavepacket_offset_bin) wavepacket_offset_bin->report(file, "wavepacket_offset");
  if (wavepacket_size_bin) wavepacket_size_bin->report(file, "wavepacket_size");
  if (wavepacket_location_bin) wavepacket_location_bin->report(file, "wavepacket_location");
  // averages bins
  if (classification_bin_intensity) classification_bin_intensity->report(file, "classification", "intensity");
  if (classification_bin_scan_angle) classification_bin_scan_angle->report(file, "classification", "scan_angle");
  if (scan_angle_bin_z) scan_angle_bin_z->report(file, "scan angle", "z coordinate");
  if (scan_angle_bin_number_of_returns) scan_angle_bin_number_of_returns->report(file, "scan_angle", "number_of_returns");
  if (scan_angle_bin_intensity) scan_angle_bin_intensity->report(file, "scan angle", "intensity");
  if (return_map_bin_intensity) return_map_bin_intensity->report(file, "return map", "intensity");
}

BOOL LASoccupancyGrid::add(const LASpoint* point)
{
  I32 pos_x, pos_y;
  if (grid_spacing < 0)
  {
    grid_spacing = -grid_spacing;
    pos_x = I32_FLOOR(point->get_x() / grid_spacing);
    pos_y = I32_FLOOR(point->get_y() / grid_spacing);
    anker = pos_y;
    min_x = max_x = pos_x;
    min_y = max_y = pos_y;
  }
  else
  {
    pos_x = I32_FLOOR(point->get_x() / grid_spacing);
    pos_y = I32_FLOOR(point->get_y() / grid_spacing);
    if (pos_x < min_x) min_x = pos_x; else if (pos_x > max_x) max_x = pos_x;
    if (pos_y < min_y) min_y = pos_y; else if (pos_y > max_y) max_y = pos_y;
  }
  return add_internal(pos_x, pos_y);
}

BOOL LASoccupancyGrid::add(I32 pos_x, I32 pos_y)
{
  if (grid_spacing < 0)
  {
    grid_spacing = -grid_spacing;
    anker = pos_y;
    min_x = max_x = pos_x;
    min_y = max_y = pos_y;
  }
  else
  {
    if (pos_x < min_x) min_x = pos_x; else if (pos_x > max_x) max_x = pos_x;
    if (pos_y < min_y) min_y = pos_y; else if (pos_y > max_y) max_y = pos_y;
  }
  return add_internal(pos_x, pos_y);
}

BOOL LASoccupancyGrid::add_internal(I32 pos_x, I32 pos_y)
{
  pos_y = pos_y - anker;
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
  if ((*array)[pos_y][pos_x_pos] & pos_x_bit) return FALSE;
  (*array)[pos_y][pos_x_pos] |= pos_x_bit;
  num_occupied++;
  return TRUE;
}

BOOL LASoccupancyGrid::occupied(const LASpoint* point) const
{
  I32 pos_x = I32_FLOOR(point->get_x() / grid_spacing);
  I32 pos_y = I32_FLOOR(point->get_y() / grid_spacing);
  return occupied(pos_x, pos_y);
}

BOOL LASoccupancyGrid::occupied(I32 pos_x, I32 pos_y) const
{
  if (grid_spacing < 0)
  {
    return FALSE;
  }
  pos_y = pos_y - anker;
  U32 array_size;
  const I32* ankers;
  const U32* const * array;
  const U16* array_sizes;
  if (pos_y < 0)
  {
    pos_y = -pos_y - 1;
    ankers = minus_ankers;
    if ((U32)pos_y < minus_plus_size && minus_plus_sizes[pos_y])
    {
      pos_x -= minus_ankers[pos_y];
      if (pos_x < 0)
      {
        pos_x = -pos_x - 1;
        array_size = minus_minus_size;
        array = minus_minus;
        array_sizes = minus_minus_sizes;
      }
      else
      {
        array_size = minus_plus_size;
        array = minus_plus;
        array_sizes = minus_plus_sizes;
      }
    }
    else
    {
      return FALSE;
    }
  }
  else
  {
    ankers = plus_ankers;
    if ((U32)pos_y < plus_plus_size && plus_plus_sizes[pos_y])
    {
      pos_x -= plus_ankers[pos_y];
      if (pos_x < 0)
      {
        pos_x = -pos_x - 1;
        array_size = plus_minus_size;
        array = plus_minus;
        array_sizes = plus_minus_sizes;
      }
      else
      {
        array_size = plus_plus_size;
        array = plus_plus;
        array_sizes = plus_plus_sizes;
      }
    }
    else
    {
      return FALSE;
    }
  }
  // maybe out of bounds in y direction
  if ((U32)pos_y >= array_size)
  {
    return FALSE;
  }
  // maybe out of bounds in x direction
  U32 pos_x_pos = pos_x/32;
  if (pos_x_pos >= array_sizes[pos_y])
  {
    return FALSE;
  }
  U32 pos_x_bit = 1 << (pos_x%32);
  if (array[pos_y][pos_x_pos] & pos_x_bit) return TRUE;
  return FALSE;
}

BOOL LASoccupancyGrid::active() const
{
  if (grid_spacing < 0) return FALSE;
  return TRUE;
}

void LASoccupancyGrid::reset()
{
  min_x = min_y = max_x = max_y = 0;
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
  num_occupied = 0;
}

BOOL LASoccupancyGrid::write_asc_grid(const char* file_name) const
{
  FILE* file = fopen(file_name, "w");
  if (file == 0) return FALSE;
  fprintf(file, "ncols %d\012", max_x-min_x+1);
  fprintf(file, "nrows %d\012", max_y-min_y+1);
  fprintf(file, "xllcorner %f\012", grid_spacing*min_x);
  fprintf(file, "yllcorner %f\012", grid_spacing*min_y);
  fprintf(file, "cellsize %lf\012", grid_spacing);
  fprintf(file, "NODATA_value %d\012", 0);
  fprintf(file, "\012");
  I32 pos_x, pos_y;
  for (pos_y = min_y; pos_y <= max_y; pos_y++)
  {
    for (pos_x = min_x; pos_x <= max_x; pos_x++)
    {
      if (occupied(pos_x, pos_y))
      {
        fprintf(file, "1 ");
      }
      else
      {
        fprintf(file, "0 ");
      }
    }
    fprintf(file, "\012");
  }
  fclose(file);
  return TRUE;
}

LASoccupancyGrid::LASoccupancyGrid(F32 grid_spacing)
{
  min_x = min_y = max_x = max_y = 0;
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
  num_occupied = 0;
}

LASoccupancyGrid::~LASoccupancyGrid()
{
  reset();
}

