/*
===============================================================================

  FILE:  laswriter_txt.cpp
  
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
#include "laswriter_txt.hpp"

#include <stdlib.h>
#include <string.h>

BOOL LASwriterTXT::refile(FILE* file)
{
  this->file = file;
  return TRUE;
}

BOOL LASwriterTXT::open(const char* file_name, const LASheader* header, const char* parse_string, const char* separator)
{
  if (file_name == 0)
  {
    fprintf(stderr,"ERROR: file name pointer is zero\n");
    return FALSE;
  }

  file = fopen(file_name, "w");

  if (file == 0)
  {
    fprintf(stderr, "ERROR: cannot open file '%s'\n", file_name);
    return FALSE;
  }

  close_file = TRUE;

  return open(file, header, parse_string, separator);
}

BOOL LASwriterTXT::open(FILE* file, const LASheader* header, const char* parse_string, const char* separator)
{
  if (file == 0)
  {
    fprintf(stderr,"ERROR: file pointer is zero\n");
    return FALSE;
  }

  this->file = file;
  this->header = header;

  if (parse_string)
  {
    this->parse_string = strdup(parse_string);
  }
  else
  {
    if (header->point_data_format == 1 || header->point_data_format == 4)
    {
      this->parse_string = strdup("xyzt");
    }
    else if (header->point_data_format == 2)
    {
      this->parse_string = strdup("xyzRGB");
    }
    else if (header->point_data_format == 3 || header->point_data_format == 5)
    {
      this->parse_string = strdup("xyztRGB");
    }
    else
    {
      this->parse_string = strdup("xyz");
    }
  }

  if (!check_parse_string(this->parse_string))
  {
    return FALSE;
  }

  if (separator)
  {
    if (strcmp(separator, "comma") == 0)
    {
      separator_sign = ',';
    }
    else if (strcmp(separator, "tab") == 0)
    {
      separator_sign = '\t';
    }
    else if (strcmp(separator, "dot") == 0 || strcmp(separator, "period") == 0)
    {
      separator_sign = '.';
    }
    else if (strcmp(separator, "colon") == 0)
    {
      separator_sign = ':';
    }
    else if (strcmp(separator, "semicolon") == 0)
    {
      separator_sign = ';';
    }
    else if (strcmp(separator, "hyphen") == 0 || strcmp(separator, "minus") == 0)
    {
      separator_sign = '-';
    }
    else if (strcmp(separator, "space") == 0)
    {
      separator_sign = ' ';
    }
    else
    {
      fprintf(stderr, "ERROR: unknown seperator '%s'\n", separator);
      return FALSE;
    }
  }

  return TRUE;
}

static void lidardouble2string(char* string, double value)
{
  int len;
  len = sprintf(string, "%.15f", value) - 1;
  while (string[len] == '0') len--;
  if (string[len] != '.') len++;
  string[len] = '\0';
}

static void lidardouble2string(char* string, double value, double precision)
{
  if (precision == 0.1)
    sprintf(string, "%.1f", value);
  else if (precision == 0.01)
    sprintf(string, "%.2f", value);
  else if (precision == 0.001)
    sprintf(string, "%.3f", value);
  else if (precision == 0.0001)
    sprintf(string, "%.4f", value);
  else if (precision == 0.00001)
    sprintf(string, "%.5f", value);
  else if (precision == 0.000001)
    sprintf(string, "%.6f", value);
  else if (precision == 0.0000001)
    sprintf(string, "%.7f", value);
  else if (precision == 0.00000001)
    sprintf(string, "%.8f", value);
  else if (precision == 0.000000001)
    sprintf(string, "%.9f", value);
  else
    lidardouble2string(string, value);
}

BOOL LASwriterTXT::unparse_extra_attribute(const LASpoint* point, I32 index)
{
  if (index >= header->number_extra_attributes)
  {
    return FALSE;
  }
  if (header->extra_attributes[index].data_type == 1)
  {
    U8 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", (I32)value);
    }
  }
  else if (header->extra_attributes[index].data_type == 2)
  {
    I8 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", (I32)value);
    }
  }
  else if (header->extra_attributes[index].data_type == 3)
  {
    U16 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", (I32)value);
    }
  }
  else if (header->extra_attributes[index].data_type == 4)
  {
    I16 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", (I32)value);
    }
  }
  else if (header->extra_attributes[index].data_type == 5)
  {
    U32 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", (I32)value);
    }
  }
  else if (header->extra_attributes[index].data_type == 6)
  {
    I32 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%d", value);
    }
  }
  else if (header->extra_attributes[index].data_type == 9)
  {
    F32 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%g", value);
    }
  }
  else if (header->extra_attributes[index].data_type == 10)
  {
    F64 value;
    point->get_extra_attribute(extra_attribute_array_offsets[index], value);
    if (header->extra_attributes[index].has_scale() || header->extra_attributes[index].has_offset())
    {
      F64 temp_d = header->extra_attributes[index].scale[0]*value + header->extra_attributes[index].offset[0];
      fprintf(file, "%g", temp_d);
    }
    else
    {
      fprintf(file, "%g", value);
    }
  }
  else
  {
    fprintf(stderr, "WARNING: extra attribute %d not (yet) implemented.\n", index);
    return FALSE;
  }
  return TRUE;
}

BOOL LASwriterTXT::write_point(const LASpoint* point)
{
  p_count++;
  int i = 0;
  while (true)
  {
    switch (parse_string[i])
    {
    case 'x': // the x coordinate
      lidardouble2string(printstring, header->get_x(point->x), header->x_scale_factor); fprintf(file, "%s", printstring);
      break;
    case 'y': // the y coordinate
      lidardouble2string(printstring, header->get_y(point->y), header->y_scale_factor); fprintf(file, "%s", printstring);
      break;
    case 'z': // the z coordinate
      lidardouble2string(printstring, header->get_z(point->z), header->z_scale_factor); fprintf(file, "%s", printstring);
      break;
    case 't': // the gps-time
      lidardouble2string(printstring,point->gps_time); fprintf(file, "%s", printstring);
      break;
    case 'i': // the intensity
      fprintf(file, "%d", point->intensity);
      break;
    case 'a': // the scan angle
      fprintf(file, "%d", point->scan_angle_rank);
      break;
    case 'r': // the number of the return
      fprintf(file, "%d", point->return_number);
      break;
    case 'c': // the classification
      fprintf(file, "%d", point->classification);
      break;
    case 'u': // the user data
      fprintf(file, "%d", point->user_data);
      break;
    case 'n': // the number of returns of given pulse
      fprintf(file, "%d", point->number_of_returns_of_given_pulse);
      break;
    case 'p': // the point source ID
      fprintf(file, "%d", point->point_source_ID);
      break;
    case 'e': // the edge of flight line flag
      fprintf(file, "%d", point->edge_of_flight_line);
      break;
    case 'd': // the direction of scan flag
      fprintf(file, "%d", point->scan_direction_flag);
      break;
    case 'R': // the red channel of the RGB field
      fprintf(file, "%d", point->rgb[0]);
      break;
    case 'G': // the green channel of the RGB field
      fprintf(file, "%d", point->rgb[1]);
      break;
    case 'B': // the blue channel of the RGB field
      fprintf(file, "%d", point->rgb[2]);
      break;
    case 'm': // the index of the point (count starts at 0)
#ifdef _WIN32
      fprintf(file, "%I64d", p_count-1);
#else
      fprintf(file, "%lld", p_count-1);
#endif
      break;
    case 'M': // the index of the point (count starts at 1)
#ifdef _WIN32
      fprintf(file, "%I64d", p_count);
#else
      fprintf(file, "%lld", p_count);
#endif
      break;
    case 'w': // the wavepacket descriptor index
      fprintf(file, "%d", point->wavepacket.getIndex());
      break;
    case 'W': // all wavepacket attributes
      fprintf(file, "%d%c%d%c%d%c%g%c%.15g%c%.15g%c%.15g", point->wavepacket.getIndex(), separator_sign, (U32)point->wavepacket.getOffset(), separator_sign, point->wavepacket.getSize(), separator_sign, point->wavepacket.getLocation(), separator_sign, point->wavepacket.getXt(), separator_sign, point->wavepacket.getYt(), separator_sign, point->wavepacket.getZt());
      break;
    case 'X': // the unscaled and unoffset integer X coordinate
      fprintf(file, "%d", point->x);
      break;
    case 'Y': // the unscaled and unoffset integer Y coordinate
      fprintf(file, "%d", point->y);
      break;
    case 'Z': // the unscaled and unoffset integer Z coordinate
      fprintf(file, "%d", point->z);
      break;
    default:
      unparse_extra_attribute(point, (I32)(parse_string[i]-'0'));
    }
    i++;
    if (parse_string[i])
    {
      fprintf(file, "%c", separator_sign);
    }
    else
    {
      fprintf(file, "\012");
      break;
    }
  }
  return TRUE;
}

BOOL LASwriterTXT::update_header(const LASheader* header, BOOL use_inventory, BOOL update_extra_bytes)
{
  return TRUE;
}

I64 LASwriterTXT::close(BOOL update_header)
{
  U32 bytes = (U32)ftell(file);

  if (file)
  {
    if (close_file)
    {
      fclose(file);
      close_file = FALSE;
    }
    file = 0;
  }
  if (parse_string)
  {
    free(parse_string);
    parse_string = 0;
  }

  npoints = p_count;
  p_count = 0;

  return bytes;
}

LASwriterTXT::LASwriterTXT()
{
  close_file = FALSE;
  file = 0;
  parse_string = 0;
  separator_sign = ' ';
}

LASwriterTXT::~LASwriterTXT()
{
  if (file) close();
}

BOOL LASwriterTXT::check_parse_string(const char* parse_string)
{
  const char* p = parse_string;
  while (p[0])
  {
    if ((p[0] != 'x') && // the x coordinate
        (p[0] != 'y') && // the y coordinate
        (p[0] != 'z') && // the x coordinate
        (p[0] != 't') && // the gps time
        (p[0] != 'R') && // the red channel of the RGB field
        (p[0] != 'G') && // the green channel of the RGB field
        (p[0] != 'B') && // the blue channel of the RGB field
        (p[0] != 's') && // a string or a number that we don't care about
        (p[0] != 'i') && // the intensity
        (p[0] != 'a') && // the scan angle
        (p[0] != 'n') && // the number of returns of given pulse
        (p[0] != 'r') && // the number of the return
        (p[0] != 'c') && // the classification
        (p[0] != 'u') && // the user data
        (p[0] != 'p') && // the point source ID
        (p[0] != 'e') && // the edge of flight line flag
        (p[0] != 'd') && // the direction of scan flag
        (p[0] != 'm') && // the index of the point (count starts at 0)
        (p[0] != 'M') && // the index of the point (count starts at 1)
        (p[0] != 'w') && // the wavepacket descriptor index
        (p[0] != 'W') && // all wavepacket attributes
        (p[0] != 'X') && // the unscaled and unoffset integer x coordinate
        (p[0] != 'Y') && // the unscaled and unoffset integer y coordinate
        (p[0] != 'Z'))   // the unscaled and unoffset integer z coordinate
    {
      if (p[0] >= '0' && p[0] <= '9')
      {
        I32 index = (I32)(p[0] - '0');
        if (index >= header->number_extra_attributes)
        {
          fprintf(stderr, "ERROR: extra attribute '%d' does not exist.\n", index);
          return FALSE;
        }
        extra_attribute_array_offsets[index] = header->get_extra_attribute_array_offset(index);
      }
      else
      {
        fprintf(stderr, "ERROR: unknown symbol '%c' in parse string. valid are\n", p[0]);
        fprintf(stderr, "       'x' : the x coordinate\n");
        fprintf(stderr, "       'y' : the y coordinate\n");
        fprintf(stderr, "       'z' : the x coordinate\n");
        fprintf(stderr, "       't' : the gps time\n");
        fprintf(stderr, "       'R' : the red channel of the RGB field\n");
        fprintf(stderr, "       'G' : the green channel of the RGB field\n");
        fprintf(stderr, "       'B' : the blue channel of the RGB field\n");
        fprintf(stderr, "       's' : a string or a number that we don't care about\n");
        fprintf(stderr, "       'i' : the intensity\n");
        fprintf(stderr, "       'a' : the scan angle\n");
        fprintf(stderr, "       'n' : the number of returns of that given pulse\n");
        fprintf(stderr, "       'r' : the number of the return\n");
        fprintf(stderr, "       'c' : the classification\n");
        fprintf(stderr, "       'u' : the user data\n");
        fprintf(stderr, "       'p' : the point source ID\n");
        fprintf(stderr, "       'e' : the edge of flight line flag\n");
        fprintf(stderr, "       'd' : the direction of scan flag\n");
        fprintf(stderr, "       'M' : the index of the point\n");
        fprintf(stderr, "       'w' : the wavepacket descriptor index\n");
        fprintf(stderr, "       'W' : all wavepacket attributes\n");
        fprintf(stderr, "       'X' : the unscaled and unoffset integer x coordinate\n");
        fprintf(stderr, "       'Y' : the unscaled and unoffset integer y coordinate\n");
        fprintf(stderr, "       'Z' : the unscaled and unoffset integer z coordinate\n");
        return FALSE;
      }
    }
    p++;
  }
  return TRUE;
}
