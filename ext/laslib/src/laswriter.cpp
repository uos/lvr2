/*
===============================================================================

  FILE:  laswriter.cpp
  
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
#include "laswriter.hpp"

#include "laswriter_las.hpp"
#include "laswriter_bin.hpp"
#include "laswriter_qfit.hpp"
#include "laswriter_txt.hpp"

#include <stdlib.h>
#include <string.h>

BOOL LASwriteOpener::piped() const
{
  return ((file_name == 0) && use_stdout);
}

LASwriter* LASwriteOpener::open(LASheader* header)
{
  if (use_nil)
  {
    LASwriterLAS* laswriterlas = new LASwriterLAS();
    if (!laswriterlas->open(header, (format == LAS_TOOLS_FORMAT_LAZ ? (use_chunking ?  LASZIP_COMPRESSOR_CHUNKED : LASZIP_COMPRESSOR_NOT_CHUNKED) : LASZIP_COMPRESSOR_NONE), (use_v1 ? 1 : 2), chunk_size))
    {
      fprintf(stderr,"ERROR: cannot open laswriterlas to NULL\n");
      delete laswriterlas;
      return 0;
    }
    return laswriterlas;
  }
  else if (file_name)
  {
    if (format <= LAS_TOOLS_FORMAT_LAZ)
    {
      LASwriterLAS* laswriterlas = new LASwriterLAS();
      if (!laswriterlas->open(file_name, header, (format == LAS_TOOLS_FORMAT_LAZ ? (use_chunking ? LASZIP_COMPRESSOR_CHUNKED : LASZIP_COMPRESSOR_NOT_CHUNKED) : LASZIP_COMPRESSOR_NONE), (use_v1 ? 1 : 2), chunk_size))
      {
        fprintf(stderr,"ERROR: cannot open laswriterlas with file name '%s'\n", file_name);
        delete laswriterlas;
        return 0;
      }
      return laswriterlas;
    }
    else if (format == LAS_TOOLS_FORMAT_TXT)
    {
      LASwriterTXT* laswritertxt = new LASwriterTXT();
      if (!laswritertxt->open(file_name, header, parse_string, separator))
      {
        fprintf(stderr,"ERROR: cannot open laswritertxt with file name '%s'\n", file_name);
        delete laswritertxt;
        return 0;
      }
      return laswritertxt;
    }
    else if (format == LAS_TOOLS_FORMAT_BIN)
    {
      LASwriterBIN* laswriterbin = new LASwriterBIN();
      if (!laswriterbin->open(file_name, header, "ts8"))
      {
        fprintf(stderr,"ERROR: cannot open laswriterbin with file name '%s'\n", file_name);
        delete laswriterbin;
        return 0;
      }
      return laswriterbin;
    }
    else if (format == LAS_TOOLS_FORMAT_QFIT)
    {
      LASwriterQFIT* laswriterqfit = new LASwriterQFIT();
      if (!laswriterqfit->open(file_name, header, 40))
      {
        fprintf(stderr,"ERROR: cannot open laswriterqfit with file name '%s'\n", file_name);
        delete laswriterqfit;
        return 0;
      }
      return laswriterqfit;
    }
    else
    {
      fprintf(stderr,"ERROR: unknown format %d\n", format);
      return 0;
    }
  }
  else if (use_stdout)
  {
    if (format <= LAS_TOOLS_FORMAT_LAZ)
    {
      LASwriterLAS* laswriterlas = new LASwriterLAS();
      if (!laswriterlas->open(stdout, header, (format == LAS_TOOLS_FORMAT_LAZ ? (use_chunking ? LASZIP_COMPRESSOR_CHUNKED : LASZIP_COMPRESSOR_NOT_CHUNKED) : LASZIP_COMPRESSOR_NONE), (use_v1 ? 1 : 2), chunk_size))
      {
        fprintf(stderr,"ERROR: cannot open laswriterlas to stdout\n");
        delete laswriterlas;
        return 0;
      }
      return laswriterlas;
    }
    else if (format == LAS_TOOLS_FORMAT_TXT)
    {
      LASwriterTXT* laswritertxt = new LASwriterTXT();
      if (!laswritertxt->open(stdout, header, parse_string, separator))
      {
        fprintf(stderr,"ERROR: cannot open laswritertxt to stdout\n");
        delete laswritertxt;
        return 0;
      }
      return laswritertxt;
    }
    else if (format == LAS_TOOLS_FORMAT_BIN)
    {
      LASwriterBIN* laswriterbin = new LASwriterBIN();
      if (!laswriterbin->open(stdout, header, "ts8"))
      {
        fprintf(stderr,"ERROR: cannot open laswriterbin to stdout\n");
        delete laswriterbin;
        return 0;
      }
      return laswriterbin;
    }
    else if (format == LAS_TOOLS_FORMAT_QFIT)
    {
      LASwriterQFIT* laswriterqfit = new LASwriterQFIT();
      if (!laswriterqfit->open(stdout, header, 40))
      {
        fprintf(stderr,"ERROR: cannot open laswriterbin to stdout\n");
        delete laswriterqfit;
        return 0;
      }
      return laswriterqfit;
    }
    else
    {
      fprintf(stderr,"ERROR: unknown format %d\n", format);
      return 0;
    }
  }
  else
  {
    fprintf(stderr,"ERROR: no laswriter output specified\n");
    return 0;
  }
}

LASwaveform13writer* LASwriteOpener::open_waveform13(const LASheader* lasheader)
{
  if (lasheader->point_data_format < 4) return 0;
  if (lasheader->vlr_wave_packet_descr == 0) return 0;
  if (get_file_name() == 0) return 0;
  LASwaveform13writer* waveform13writer = new LASwaveform13writer();
  if (waveform13writer->open(get_file_name(), lasheader->vlr_wave_packet_descr))
  {
    return waveform13writer;
  }
  delete waveform13writer;
  return 0;
}

void LASwriteOpener::usage() const
{
  fprintf(stderr,"Supported LAS Outputs\n");
  fprintf(stderr,"  -o lidar.las\n");
  fprintf(stderr,"  -o lidar.laz\n");
  fprintf(stderr,"  -o xyzta.txt -oparse xyzta (on-the-fly to ASCII)\n");
  fprintf(stderr,"  -o terrasolid.bin\n");
  fprintf(stderr,"  -o nasa.qi\n");
  fprintf(stderr,"  -olas -olaz -otxt -obin -oqfit (specify format)\n");
  fprintf(stderr,"  -stdout (pipe to stdout)\n");
  fprintf(stderr,"  -nil    (pipe to NULL)\n");
}

BOOL LASwriteOpener::parse(int argc, char* argv[])
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
    else if (strcmp(argv[i],"-o") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: file_name\n", argv[i]);
        return FALSE;
      }
      set_file_name(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-olas") == 0)
    {
      format = LAS_TOOLS_FORMAT_LAS;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-olaz") == 0)
    {
      format = LAS_TOOLS_FORMAT_LAZ;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-otxt") == 0)
    {
      format = LAS_TOOLS_FORMAT_TXT;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-obin") == 0)
    {
      format = LAS_TOOLS_FORMAT_BIN;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-oqi") == 0 || strcmp(argv[i],"-oqfit") == 0)
    {
      format = LAS_TOOLS_FORMAT_QFIT;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-stdout") == 0)
    {
      use_stdout = TRUE;
      use_nil = FALSE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-nil") == 0)
    {
      use_nil = TRUE;
      use_stdout = FALSE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-v1") == 0)
    {
      use_v1 = TRUE;
      use_chunking = FALSE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-v2") == 0)
    {
      use_v1 = FALSE;
      use_chunking = TRUE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-no_chunk") == 0)
    {
      use_chunking = FALSE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-chunk") == 0)
    {
      use_chunking = TRUE;
      *argv[i]='\0';
    }
    else if (strcmp(argv[i],"-chunk_size") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: number_points\n", argv[i]);
        return FALSE;
      }
      use_chunking = TRUE;
      chunk_size = atoi(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-oparse") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: string\n", argv[i]);
        return FALSE;
      }
      set_parse_string(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
    else if (strcmp(argv[i],"-sep") == 0)
    {
      if ((i+1) >= argc)
      {
        fprintf(stderr,"ERROR: '%s' needs 1 argument: separator\n", argv[i]);
        return FALSE;
      }
      set_separator(argv[i+1]);
      *argv[i]='\0'; *argv[i+1]='\0'; i+=1;
    }
  }
  return TRUE;
}

void LASwriteOpener::set_file_name(const char* file_name)
{
  if (this->file_name) free(this->file_name);
  if (file_name)
  {
    this->file_name = strdup(file_name);
    if (strstr(file_name, ".laz") || strstr(file_name, ".LAZ"))
    {
      format = LAS_TOOLS_FORMAT_LAZ;
    }
    else if (strstr(file_name, ".las") || strstr(file_name, ".LAS"))
    {
      format = LAS_TOOLS_FORMAT_LAS;
    }
    else if (strstr(file_name, ".bin") || strstr(file_name, ".BIN")) // terrasolid
    {
      format = LAS_TOOLS_FORMAT_BIN;
    }
    else if (strstr(file_name, ".qi") || strstr(file_name, ".QI")) // QFIT
    {
      format = LAS_TOOLS_FORMAT_QFIT;
    }
    else // assume ascii output
    {
      format = LAS_TOOLS_FORMAT_TXT;
    }
  }
  else
  {
    this->file_name = 0;
  }
}

void LASwriteOpener::set_format(const char* format)
{
  if (format)
  {
    if (strstr(format, "laz") || strstr(format, "LAZ"))
    {
      this->format = LAS_TOOLS_FORMAT_LAZ;
    }
    else if (strstr(format, "las") || strstr(format, "LAS"))
    {
      this->format = LAS_TOOLS_FORMAT_LAS;
    }
    else if (strstr(format, "bin") || strstr(format, "BIN")) // terrasolid
    {
      this->format = LAS_TOOLS_FORMAT_BIN;
    }
    else if (strstr(format, "qi") || strstr(format, "QI")) // QFIT
    {
      this->format = LAS_TOOLS_FORMAT_QFIT;
    }
    else // assume ascii output
    {
      this->format = LAS_TOOLS_FORMAT_TXT;
    }
  }
  else
  {
    this->format = LAS_TOOLS_FORMAT_DEFAULT;
  }
}

void LASwriteOpener::make_file_name(const char* file_name, I32 file_number)
{
  size_t len;
  if (file_number > -1)
  {
    if (file_name)
    {
      if (this->file_name) free(this->file_name);
      len = strlen(file_name);
      this->file_name = (char*)malloc(len+10);
      strcpy(this->file_name, file_name);
    }
    else
    {
      len = strlen(this->file_name);
    }
    while (len > 0 && this->file_name[len] != '.') len--;
    len++;
    int num = len - 2;
    while (num > 0 && this->file_name[num] >= '0' && this->file_name[num] <= '9')
    {
      this->file_name[num] = '0' + (file_number%10);
      file_number = file_number/10;
      num--;
    }
  }
  else
  {
    if (this->file_name) free(this->file_name);
    if (file_name)
    {
      len = strlen(file_name);
      this->file_name = (char*)malloc(len+10);
      strcpy(this->file_name, file_name);
      while (len > 0 && this->file_name[len] != '.') len--;
      if (file_number == -1)
      {
        this->file_name[len] = '_';
        this->file_name[len+1] = '1';
        this->file_name[len+2] = '.';
        len += 3;
      }
      else
      {
        len++;
      }
    }
    else
    {
      len = 7;
      this->file_name = strdup("output.xxx");
    }
  }
  if (format <= LAS_TOOLS_FORMAT_LAS)
  {
    this->file_name[len] = 'l';
    this->file_name[len+1] = 'a';
    this->file_name[len+2] = 's';
  }
  else if (format == LAS_TOOLS_FORMAT_LAZ)
  {
    this->file_name[len] = 'l';
    this->file_name[len+1] = 'a';
    this->file_name[len+2] = 'z';
  }
  else if (format == LAS_TOOLS_FORMAT_BIN)
  {
    this->file_name[len] = 'b';
    this->file_name[len+1] = 'i';
    this->file_name[len+2] = 'n';
  }
  else // if (format == LAS_TOOLS_FORMAT_TXT)
  {
    this->file_name[len] = 't';
    this->file_name[len+1] = 'x';
    this->file_name[len+2] = 't';
  }
  this->file_name[len+3] = '\0';
}

const char* LASwriteOpener::get_file_name() const
{
  return file_name;
}

BOOL LASwriteOpener::format_was_specified() const
{
  return (format != LAS_TOOLS_FORMAT_DEFAULT);
}

I32 LASwriteOpener::get_format() const
{
  return format;
}

void LASwriteOpener::set_parse_string(const char* parse_string)
{
  if (this->parse_string) free(this->parse_string);
  this->parse_string = strdup(parse_string);
}

void LASwriteOpener::set_separator(const char* separator)
{
  if (this->separator) free(this->separator);
  this->separator = strdup(separator);
}

BOOL LASwriteOpener::active() const
{
  return (file_name != 0 || use_stdout || use_nil);
}

LASwriteOpener::LASwriteOpener()
{
  file_name = 0;
  parse_string = 0;
  separator = 0;
  format = LAS_TOOLS_FORMAT_DEFAULT;
  chunk_size = LASZIP_CHUNK_SIZE_DEFAULT;
  use_chunking = TRUE;
  use_stdout = FALSE;
  use_nil = FALSE;
  use_v1 = FALSE;
}

LASwriteOpener::~LASwriteOpener()
{
  if (file_name) free(file_name);
  if (parse_string) free(parse_string);
  if (separator) free(separator);
}
