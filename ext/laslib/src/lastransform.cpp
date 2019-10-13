/*
===============================================================================

  FILE:  lastransform.cpp

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
#include "lastransform.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

class LASoperationTranslateX : public LASoperation
{
public:
  inline const char* name() const { return "translate_x"; };
  inline void transform(LASpoint* point) const {
    point->set_x(point->get_x() + offset);
  };
  LASoperationTranslateX(F64 offset) { this->offset = offset; };
private:
  F64 offset;
};

class LASoperationTranslateY : public LASoperation
{
public:
  inline const char* name() const { return "translate_y"; };
  inline void transform(LASpoint* point) const {
    point->set_y(point->get_y() + offset);
  };
  LASoperationTranslateY(F64 offset) { this->offset = offset; };
private:
  F64 offset;
};

class LASoperationTranslateZ : public LASoperation
{
public:
  inline const char* name() const { return "translate_z"; };
  inline void transform(LASpoint* point) const {
    point->set_z(point->get_z() + offset);
  };
  LASoperationTranslateZ(F64 offset) { this->offset = offset; };
private:
  F64 offset;
};

class LASoperationTranslateXYZ : public LASoperation
{
public:
  inline const char* name() const { return "translate_xyz"; };
  inline void transform(LASpoint* point) const {
    point->set_x(point->get_x() + offset[0]);
    point->set_y(point->get_y() + offset[1]);
    point->set_z(point->get_z() + offset[2]);
  };
  LASoperationTranslateXYZ(F64 x_offset, F64 y_offset, F64 z_offset) { this->offset[0] = x_offset; this->offset[1] = y_offset; this->offset[2] = z_offset; };
private:
  F64 offset[3];
};

class LASoperationScaleX : public LASoperation
{
public:
  inline const char* name() const { return "scale_x"; };
  inline void transform(LASpoint* point) const {
    point->set_x(point->get_x() * scale);
  };
  LASoperationScaleX(F64 scale) { this->scale = scale; };
private:
  F64 scale;
};

class LASoperationScaleY : public LASoperation
{
public:
  inline const char* name() const { return "scale_y"; };
  inline void transform(LASpoint* point) const {
    point->set_y(point->get_y() * scale);
  };
  LASoperationScaleY(F64 scale) { this->scale = scale; };
private:
  F64 scale;
};

class LASoperationScaleZ : public LASoperation
{
public:
  inline const char* name() const { return "scale_z"; };
  inline void transform(LASpoint* point) const {
    point->set_z(point->get_z() * scale);
  };
  LASoperationScaleZ(F64 scale) { this->scale = scale; };
private:
  F64 scale;
};

class LASoperationScaleXYZ : public LASoperation
{
public:
  inline const char* name() const { return "scale_xyz"; };
  inline void transform(LASpoint* point) const {
    point->set_x(point->get_x() * scale[0]);
    point->set_y(point->get_y() * scale[1]);
    point->set_z(point->get_z() * scale[2]);
  };
  LASoperationScaleXYZ(F64 x_scale, F64 y_scale, F64 z_scale) { this->scale[0] = x_scale; this->scale[1] = y_scale; this->scale[2] = z_scale; };
private:
  F64 scale[3];
};

class LASoperationTranslateThenScaleX : public LASoperation
{
public:
  inline const char* name() const { return "translate_then_scale_x"; };
  inline void transform(LASpoint* point) const {
    point->set_x((point->get_x()+offset)*scale);
  };
  LASoperationTranslateThenScaleX(F64 offset, F64 scale_factor) { this->offset = offset; };
private:
  F64 offset;
  F64 scale;
};

class LASoperationTranslateThenScaleY : public LASoperation
{
public:
  inline const char* name() const { return "translate_then_scale_y"; };
  inline void transform(LASpoint* point) const {
    point->set_y((point->get_y()+offset)*scale);
  };
  LASoperationTranslateThenScaleY(F64 offset, F64 scale) { this->offset = offset; this->scale = scale; };
private:
  F64 offset;
  F64 scale;
};

class LASoperationTranslateThenScaleZ : public LASoperation
{
public:
  inline const char* name() const { return "translate_then_scale_z"; };
  inline void transform(LASpoint* point) const {
    point->set_z((point->get_z()+offset)*scale);
  };
  LASoperationTranslateThenScaleZ(F64 offset, F64 scale) { this->offset = offset; this->scale = scale; };
private:
  F64 offset;
  F64 scale;
};

class LASoperationRotateXY : public LASoperation
{
public:
  inline const char* name() const { return "rotate_xy"; };
  inline void transform(LASpoint* point) const {
    F64 x = point->get_x() - x_offset;
    F64 y = point->get_y() - y_offset;
    point->set_x(cos_angle*x - sin_angle*y + x_offset);
    point->set_y(cos_angle*y + sin_angle*x + y_offset);
  };
  LASoperationRotateXY(F64 angle, F64 x_offset, F64 y_offset) { this->x_offset = x_offset; this->y_offset = y_offset; cos_angle = cos(3.141592653589793238462643383279502884197169/180*angle); sin_angle = sin(3.141592653589793238462643383279502884197169/180*angle); };
private:
  F64 x_offset, y_offset;
  F64 cos_angle, sin_angle;
};

class LASoperationClampZ : public LASoperation
{
public:
  inline const char* name() const { return "clamp_z"; };
  inline void transform(LASpoint* point) const {
    F64 z = point->get_z();
    if (z < min) point->set_z(min);
    else if (z > max) point->set_z(max);
  };
  LASoperationClampZ(F64 min, F64 max) { this->min = min; this->max = max; };
private:
  F64 min, max;
};

class LASoperationClampZmin : public LASoperation
{
public:
  inline const char* name() const { return "clamp_z_min"; };
  inline void transform(LASpoint* point) const {
    F64 z = point->get_z();
    if (z < min) point->set_z(min);
  };
  LASoperationClampZmin(F64 min) { this->min = min; };
private:
  F64 min;
};

class LASoperationClampZmax : public LASoperation
{
public:
  inline const char* name() const { return "clamp_z_max"; };
  inline void transform(LASpoint* point) const {
    F64 z = point->get_z();
    if (z > max) point->set_z(max);
  };
  LASoperationClampZmax(F64 max) { this->max = max; };
private:
  F64 max;
};

class LASoperationTranslateRawX : public LASoperation
{
public:
  inline const char* name() const { return "translate_raw_x"; };
  inline void transform(LASpoint* point) const {
    point->x += offset;
  };
  LASoperationTranslateRawX(I32 offset) { this->offset = offset; };
private:
  I32 offset;
};

class LASoperationTranslateRawY : public LASoperation
{
public:
  inline const char* name() const { return "translate_raw_y"; };
  inline void transform(LASpoint* point) const {
    point->y += offset;
  };
  LASoperationTranslateRawY(I32 offset) { this->offset = offset; };
private:
  I32 offset;
};

class LASoperationTranslateRawZ : public LASoperation
{
public:
  inline const char* name() const { return "translate_raw_z"; };
  inline void transform(LASpoint* point) const {
    point->z += offset;
  };
  LASoperationTranslateRawZ(I32 offset) { this->offset = offset; };
private:
  I32 offset;
};

class LASoperationTranslateRawXYZ : public LASoperation
{
public:
  inline const char* name() const { return "translate_raw_xyz"; };
  inline void transform(LASpoint* point) const {
    point->x += offset[0];
    point->y += offset[1];
    point->z += offset[2];
  };
  LASoperationTranslateRawXYZ(I32 x_offset, I32 y_offset, I32 z_offset) { this->offset[0] = x_offset; this->offset[1] = y_offset; this->offset[2] = z_offset; };
private:
  I32 offset[3];
};

class LASoperationClampRawZ : public LASoperation
{
public:
  inline const char* name() const { return "clamp_raw_z"; };
  inline void transform(LASpoint* point) const {
    if (point->z < min) point->z = min;
    else if (point->z > max) point->z = max;
  };
  LASoperationClampRawZ(I32 min, I32 max) { this->min = min; this->max = max; };
private:
  I32 min, max;
};

class LASoperationScaleIntensity : public LASoperation
{
public:
  inline const char* name() const { return "scale_intensity"; };
  inline void transform(LASpoint* point) const {
    F32 intensity = scale*point->intensity;
    point->intensity = U16_CLAMP((I32)intensity);
  };
  LASoperationScaleIntensity(F32 scale) { this->scale = scale; };
private:
  F32 scale;
};

class LASoperationTranslateIntensity : public LASoperation
{
public:
  inline const char* name() const { return "translate_intensity"; };
  inline void transform(LASpoint* point) const {
    F32 intensity = offset+point->intensity;
    point->intensity = U16_CLAMP((I32)intensity);
  };
  LASoperationTranslateIntensity(F32 offset) { this->offset = offset; };
private:
  F32 offset;
};

class LASoperationTranslateThenScaleIntensity : public LASoperation
{
public:
  inline const char* name() const { return "translate_then_scale_intensity"; };
  inline void transform(LASpoint* point) const {
    F32 intensity = (offset+point->intensity)*scale;
    point->intensity = U16_CLAMP((I32)intensity);
  };
  LASoperationTranslateThenScaleIntensity(F32 offset, F32 scale) { this->offset = offset; this->scale = scale; };
private:
  F32 offset;
  F32 scale;
};

class LASoperationScaleScanAngle : public LASoperation
{
public:
  inline const char* name() const { return "scale_scan_angle"; };
  inline void transform(LASpoint* point) const {
    F32 scan_angle_rank = scale*point->scan_angle_rank;
    point->scan_angle_rank = I8_CLAMP(I32_QUANTIZE(scan_angle_rank));
  };
  LASoperationScaleScanAngle(F32 scale) { this->scale = scale; };
private:
  F32 scale;
};

class LASoperationTranslateScanAngle : public LASoperation
{
public:
  inline const char* name() const { return "translate_scan_angle"; };
  inline void transform(LASpoint* point) const {
    F32 scan_angle_rank = offset+point->scan_angle_rank;
    point->scan_angle_rank = I8_CLAMP(I32_QUANTIZE(scan_angle_rank));
  };
  LASoperationTranslateScanAngle(F32 offset) { this->offset = offset; };
private:
  F32 offset;
};

class LASoperationTranslateThenScaleScanAngle : public LASoperation
{
public:
  inline const char* name() const { return "translate_then_scale_scan_angle"; };
  inline void transform(LASpoint* point) const {
    F32 scan_angle_rank = (offset+point->scan_angle_rank)*scale;
    point->scan_angle_rank = I8_CLAMP(I32_QUANTIZE(scan_angle_rank));
  };
  LASoperationTranslateThenScaleScanAngle(F32 offset, F32 scale) { this->offset = offset; this->scale = scale; };
private:
  F32 offset;
  F32 scale;
};

class LASoperationChangeClassificationFromTo : public LASoperation
{
public:
  inline const char* name() const { return "change_classification_from_to"; };
  inline void transform(LASpoint* point) const { if ((point->classification & 31) == class_from) point->classification = (point->classification & 224) | class_to; };
  LASoperationChangeClassificationFromTo(U8 class_from, U8 class_to) { this->class_from = class_from; this->class_to = class_to; };
private:
  U8 class_from;
  U8 class_to;
};

class LASoperationChangePointSourceFromTo : public LASoperation
{
public:
  inline const char* name() const { return "change_point_source_from_to"; };
  inline void transform(LASpoint* point) const { if (point->point_source_ID == psid_from) point->point_source_ID = psid_to; };
  LASoperationChangePointSourceFromTo(U16 psid_from, U16 psid_to) { this->psid_from = psid_from; this->psid_to = psid_to; };
private:
  U16 psid_from;
  U16 psid_to;
};

class LASoperationRepairZeroReturns : public LASoperation
{
public:
  inline const char* name() const { return "repair_zero_returns"; };
  inline void transform(LASpoint* point) const { if (point->number_of_returns_of_given_pulse == 0) point->number_of_returns_of_given_pulse = 1; if (point->return_number == 0) point->return_number = 1; };
};

class LASoperationChangeReturnNumberFromTo : public LASoperation
{
public:
  inline const char* name() const { return "change_return_number_from_to"; };
  inline void transform(LASpoint* point) const { if (point->return_number == return_number_from) point->return_number = return_number_to; };
  LASoperationChangeReturnNumberFromTo(U8 return_number_from, U8 return_number_to) { this->return_number_from = return_number_from; this->return_number_to = return_number_to; };
private:
  U8 return_number_from;
  U8 return_number_to;
};

class LASoperationChangeNumberOfReturnsFromTo : public LASoperation
{
public:
  inline const char* name() const { return "change_number_of_returns_from_to"; };
  inline void transform(LASpoint* point) const { if (point->number_of_returns_of_given_pulse == number_of_returns_from) point->number_of_returns_of_given_pulse = number_of_returns_to; };
  LASoperationChangeNumberOfReturnsFromTo(U8 number_of_returns_from, U8 number_of_returns_to) { this->number_of_returns_from = number_of_returns_from; this->number_of_returns_to = number_of_returns_to; };
private:
  U8 number_of_returns_from;
  U8 number_of_returns_to;
};

class LASoperationTranslateGpsTime : public LASoperation
{
public:
  inline const char* name() const { return "translate_gps_time"; };
  inline void transform(LASpoint* point) const { point->gps_time += offset; };
  LASoperationTranslateGpsTime(F64 offset) { this->offset = offset; };
private:
  F64 offset;
};

class LASoperationScaleRGBdown : public LASoperation
{
public:
  inline const char* name() const { return "scale_rgb_down"; };
  inline void transform(LASpoint* point) const { point->rgb[0] = point->rgb[0]/256; point->rgb[1] = point->rgb[1]/256; point->rgb[2] = point->rgb[2]/256; };
};

class LASoperationScaleRGBup : public LASoperation
{
public:
  inline const char* name() const { return "scale_rgb_up"; };
  inline void transform(LASpoint* point) const { point->rgb[0] = point->rgb[0]*256; point->rgb[1] = point->rgb[1]*256; point->rgb[2] = point->rgb[2]*256; };
};

class LASoperationSwitchXY : public LASoperation
{
public:
  inline const char* name() const { return "switch_x_y"; };
  inline void transform(LASpoint* point) const { I32 temp = point->x; point->x = point->y; point->y = temp; };
};

class LASoperationSwitchXZ : public LASoperation
{
public:
  inline const char* name() const { return "switch_x_z"; };
  inline void transform(LASpoint* point) const { I32 temp = point->x; point->x = point->z; point->z = temp; };
};

class LASoperationSwitchYZ : public LASoperation
{
public:
  inline const char* name() const { return "switch_y_z"; };
  inline void transform(LASpoint* point) const { I32 temp = point->y; point->y = point->z; point->z = temp; };
};

class LASoperationFlipWaveformDirection : public LASoperation
{
public:
  inline const char* name() const { return "flip_waveform_direction"; };
  inline void transform(LASpoint* point) const { point->wavepacket.flipDirection(); };
};

void LAStransform::clean()
{
  U32 i;
  for (i = 0; i < num_operations; i++)
  {
    delete operations[i];
  }
  if (operations) delete [] operations;
  change_coordinates = FALSE;
  alloc_operations = 0;
  num_operations = 0;
  operations = 0;
}

void LAStransform::usage() const
{
  fprintf(stderr,"Transform coordinates.\n");
  fprintf(stderr,"  -translate_x -2.5\n");
  fprintf(stderr,"  -scale_z 0.3048\n");
  fprintf(stderr,"  -rotate_xy 15.0 620000 4100000 (angle + origin)\n");
  fprintf(stderr,"  -translate_xyz 0.5 0.5 0\n");
  fprintf(stderr,"  -translate_then_scale_y -0.5 1.001\n");
  fprintf(stderr,"  -clamp_z_min 70.5\n");
  fprintf(stderr,"  -clamp_z 70.5 72.5\n");
  fprintf(stderr,"Transform raw xyz integers.\n");
  fprintf(stderr,"  -translate_raw_z 20\n");
  fprintf(stderr,"  -translate_raw_xyz 1 1 0\n");
  fprintf(stderr,"  -clamp_raw_z 500 800\n");
  fprintf(stderr,"Transform intensity.\n");
  fprintf(stderr,"  -scale_intensity 2.5\n");
  fprintf(stderr,"  -translate_intensity 50\n");
  fprintf(stderr,"  -translate_then_scale_intensity 0.5 3.1\n");
  fprintf(stderr,"Transform scan_angle.\n");
  fprintf(stderr,"  -scale_scan_angle 1.944445\n");
  fprintf(stderr,"  -translate_scan_angle -5\n");
  fprintf(stderr,"  -translate_then_scale_scan_angle -0.5 2.1\n");
  fprintf(stderr,"Change the return number or return count of points.\n");
  fprintf(stderr,"  -repair_zero_returns\n");
  fprintf(stderr,"  -change_return_number_from_to 2 1\n");
  fprintf(stderr,"  -change_number_of_returns_from_to 0 2\n");
  fprintf(stderr,"Change classification by replacing one with another.\n");
  fprintf(stderr,"  -change_classification_from_to 2 4\n");
  fprintf(stderr,"Change point source ID by replacing one with another.\n");
  fprintf(stderr,"  -change_point_source_from_to 1023 1024\n");
  fprintf(stderr,"Transform gps_time.\n");
  fprintf(stderr,"  -translate_gps_time 40.50\n");
  fprintf(stderr,"Transform RGB colors.\n");
  fprintf(stderr,"  -scale_rgb_down (by 256)\n");
  fprintf(stderr,"  -scale_rgb_up (by 256)\n");
}

BOOL LAStransform::parse(int argc, char* argv[])
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
      usage();
      return TRUE;
    }
    else if (strcmp(argv[i],"-translate_x") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateX((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_y") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateY((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_z") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateZ((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_xyz") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: offset_x offset_y offset_z\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateXYZ((F64)atof(argv[i+1]), (F64)atof(argv[i+2]), (F64)atof(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3;
    }
    else if (strcmp(argv[i],"-scale_x") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationScaleX((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-scale_y") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationScaleY((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-scale_z") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationScaleZ((F64)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-scale_xyz") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: scale_x scale_y scale_z\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationScaleXYZ((F64)atof(argv[i+1]), (F64)atof(argv[i+2]), (F64)atof(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3;
    }
    else if (strcmp(argv[i],"-translate_then_scale_x") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: offset scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateThenScaleX((F64)atof(argv[i+1]), (F64)atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-translate_then_scale_y") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: offset scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateThenScaleY((F64)atof(argv[i+1]), (F64)atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-translate_then_scale_z") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: offset scale\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateThenScaleZ((F64)atof(argv[i+1]), (F64)atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-rotate_xy") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: angle, x, y\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationRotateXY((F64)atof(argv[i+1]), (F64)atof(argv[i+2]), (F64)atof(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3;
    }
    else if (strcmp(argv[i],"-clamp_z") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min, max\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationClampZ((I32)atoi(argv[i+1]), (I32)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-clamp_z_min") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: min\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationClampZmin((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-clamp_z_max") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: max\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationClampZmax((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_raw_x") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateRawX((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_raw_y") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateRawY((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_raw_z") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateRawZ((I32)atoi(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_raw_xyz") == 0)
    {
      if ((i+3) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 3 arguments: offset_x offset_y offset_z\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationTranslateRawXYZ((I32)atoi(argv[i+1]), (I32)atoi(argv[i+2]), (I32)atoi(argv[i+3])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; *argv[i+3]='\0'; i+=3;
    }
    else if (strcmp(argv[i],"-clamp_raw_z") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: min, max\n", argv[i]);
        return FALSE;
      }
      change_coordinates = TRUE;
      add_operation(new LASoperationClampRawZ((I32)atoi(argv[i+1]), (I32)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-scale_intensity") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: scale\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationScaleIntensity((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_intensity") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationTranslateIntensity((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_then_scale_intensity") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: offset scale\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationTranslateThenScaleIntensity((F32)atof(argv[i+1]), (F32)atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-scale_scan_angle") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: scale\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationScaleScanAngle((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_scan_angle") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationTranslateScanAngle((F32)atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-translate_then_scale_scan_angle") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: offset scale\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationTranslateThenScaleScanAngle((F32)atof(argv[i+1]), (F32)atof(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-change_classification_from_to") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: from_class to_class\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationChangeClassificationFromTo((U8)atoi(argv[i+1]), (U8)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-change_point_source_from_to") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: from_psid to_psid\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationChangePointSourceFromTo((U16)atoi(argv[i+1]), (U16)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-repair_zero_returns") == 0)
    {
      add_operation(new LASoperationRepairZeroReturns());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-change_return_number_from_to") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: from_return_number to_return_number\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationChangeReturnNumberFromTo((U8)atoi(argv[i+1]), (U8)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-change_number_of_returns_from_to") == 0)
    {
      if ((i+2) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 2 arguments: from_number_of_returns to_number_of_returns\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationChangeNumberOfReturnsFromTo((U8)atoi(argv[i+1]), (U8)atoi(argv[i+2])));
      *argv[i]='\0'; *argv[i+1]='\0'; *argv[i+2]='\0'; i+=2;
    }
    else if (strcmp(argv[i],"-translate_gps_time") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: offset\n", argv[i]);
        return FALSE;
      }
      add_operation(new LASoperationTranslateGpsTime(atof(argv[i+1])));
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-scale_rgb_down") == 0 || strcmp(argv[i],"-scale_rbg_down") == 0)
    {
      add_operation(new LASoperationScaleRGBdown());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-scale_rgb_up") == 0 || strcmp(argv[i],"-scale_rbg_up") == 0)
    {
      add_operation(new LASoperationScaleRGBup());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-switch_x_y") == 0)
    {
      add_operation(new LASoperationSwitchXY());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-switch_x_z") == 0)
    {
      add_operation(new LASoperationSwitchXZ());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-switch_y_z") == 0)
    {
      add_operation(new LASoperationSwitchYZ());
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-flip_waveform_direction") == 0)
    {
      add_operation(new LASoperationFlipWaveformDirection());
      *argv[i]='\0';
    }
  }
  return TRUE;
}

void LAStransform::transform(LASpoint* point) const
{
  U32 i;
  for (i = 0; i < num_operations; i++) operations[i]->transform(point);
}

LAStransform::LAStransform()
{
  change_coordinates = FALSE;
  alloc_operations = 0;
  num_operations = 0;
  operations = 0;
}

LAStransform::~LAStransform()
{
  if (operations) clean();
}

void LAStransform::add_operation(LASoperation* transform_operation)
{
  if (num_operations == alloc_operations)
  {
    U32 i;
    alloc_operations += 16;
    LASoperation** temp_operations = new LASoperation*[alloc_operations];
    if (operations)
    {
      for (i = 0; i < num_operations; i++)
      {
        temp_operations[i] = operations[i];
      }
      delete [] operations;
    }
    operations = temp_operations;
  }
  operations[num_operations] = transform_operation;
  num_operations++;
}
