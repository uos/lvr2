/**
 * @file
 * @brief Definition of point pairs
 *
 *  @author Kai Lingemann. Institute of Computer Science, University of Osnabrueck, Germany.
 *  @author Andreas Nuechter. Institute of Computer Science, University of Osnabrueck, Germany.
 */

#ifndef __PTPAIR_H__
#define __PTPAIR_H__

#include "slam6d/point.h"
#include <iostream>
using std::endl;
#include <fstream>
using std::ofstream;

/**
 * @brief Representing point pairs
 */
class PtPair {
public:
  /**
   * Constructor, by two 'point' pointers
   */
  inline PtPair(double *_p1, double *_p2);
  
  inline PtPair(Point &p1, Point &p2);
  
  inline PtPair();

  inline friend ostream& operator<<(ostream& os, const PtPair& pair);

  Point p1,  ///< The two points forming the pair
        p2;  ///< The two points forming the pair
};

#include "ptpair.icc"
#endif
