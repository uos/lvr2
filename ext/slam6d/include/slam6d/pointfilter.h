/**
 * @file
 * @brief IO filtering class for ScanIO to discard unwanted points.
 *
 * @author Thomas Escher
 */

#ifndef POINT_FILTER_H
#define POINT_FILTER_H

#include <string>
#include <map>

class Checker;


/**
 * Flexible filtering class for parsing a set of points.
 * 
 * This class is configurable with parameters for range and height and can be transferred via the use of a parameter string.
 * Use on a point set via repeated use of the check function, which creates the internal Checker structures once for each change to the parameters. The amount of tests is held as minimal as possible.
 */
class PointFilter
{
public:
  // Default empty constructor
  PointFilter();
  //! Deserialization constructor, forms parameters back from a string given by getParams
  PointFilter(const std::string& params);
  ~PointFilter();

  PointFilter& setRange(double maxDist, double minDist);
  PointFilter& setHeight(double top, double bottom);

  //! Serialization function to convert it into a string, usable in the constructor
  std::string getParams();

  //! Check a point, returning success if all contained Checker functions accept that point (implemented in .icc)
  inline bool check(double* point);
private:
  //! Storage for parameter keys and values
  std::map<std::string, std::string> m_params;

  //! If either not created yet or parameters get changed, this flag will cause check to create a new Checker chain
  bool m_changed;

  //! created in the first check call with the changed flag set
  Checker* m_checker;

  //! Allocation of the checkers
  void createCheckers();

  // factory magic
  template<typename T> friend struct CheckerFactory;
  static std::map<std::string, Checker* (*)(const std::string&)>* factory;
};

class Checker {
public:
  Checker();
  ~Checker();

  //! Testing function
  virtual bool test(double* point) = 0;

  //! Next test in chain
  Checker* m_next;
};

//! Factory integrating the create-function and adding it to the lookup map in an easy template
template<typename T>
struct CheckerFactory {
  //! Instanciate in the source code with the to be created class as template argument and associated key string as constructor argument
  CheckerFactory(const std::string& key) { (*PointFilter::factory)[key] = CheckerFactory<T>::create; }
  //! Automated create function, safely returning a nullpointer if the constructor throws for unwanted values
  static Checker* create(const std::string& value) { try { return new T(value); } catch(...) { return 0; } }
};

class CheckerRangeMax : public Checker {
public:
  CheckerRangeMax(const std::string& value);
  virtual bool test(double* point);
private:
  double m_max;
};

class CheckerRangeMin : public Checker {
public:
  CheckerRangeMin(const std::string& value);
  virtual bool test(double* point);
private:
  double m_min;
};

class CheckerHeightTop : public Checker {
public:
  CheckerHeightTop(const std::string& value);
  virtual bool test(double* point);
private:
  double m_top;
};

class CheckerHeightBottom : public Checker {
public:
  CheckerHeightBottom(const std::string& value);
  virtual bool test(double* point);
private:
  double m_bottom;
};

#include "pointfilter.icc"

#endif //POINT_FILTER_H
