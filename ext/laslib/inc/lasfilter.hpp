/*
===============================================================================

  FILE:  lasfilter.hpp
  
  CONTENTS:
  
    Filters LIDAR points based on certain criteria being true (or not).

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
  
    25 December 2010 -- created after swinging in Mara's hammock for hours
  
===============================================================================
*/
#ifndef LAS_FILTER_HPP
#define LAS_FILTER_HPP

#include "lasdefinitions.hpp"

class LAScriterion
{
public:
  virtual const char * name() const = 0;
  virtual int get_command(char* string) const = 0;
  virtual BOOL filter(const LASpoint* point) = 0;
  virtual void reset(){};
  virtual ~LAScriterion(){};
};

class LASfilter
{
public:

  void usage() const;
  void clean();
  BOOL parse(int argc, char* argv[]);
  int unparse(char* string) const;
  inline BOOL active() const { return (num_criteria != 0); };
  void addClipCircle(F64 x, F64 y, F64 radius);
  void addScanDirectionChangeOnly();

  BOOL filter(const LASpoint* point);
  void reset();

  LASfilter();
  ~LASfilter();

private:

  void add_criterion(LAScriterion* criterion);
  U32 num_criteria;
  U32 alloc_criteria;
  LAScriterion** criteria;
  int* counters;
};

#endif
