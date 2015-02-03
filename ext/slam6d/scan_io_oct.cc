/**
 * @file
 * @brief Implementation of reading 3D scans
 * @author Jan Elsberg. Automation Group, Jacobs University Bremen gGmbH, Germany. 
 */
#include "slam6d/Boctree.h"
#include "slam6d/scan_io_oct.h"
#include "slam6d/globals.icc"
#include <fstream>
using std::ifstream;
#include <iostream>
using std::cout;
using std::cerr;
using std::endl;


/**
 * Reads specified scans from given directory.
 *
 * Scan poses will NOT be initialized after a call
 * to this function.
 *
 * This function actually implements loading of 3D scans
 * in oct-file format and will be compiled as shared lib.
 * 
 * @param start Starts to read with this scan
 * @param end Stops with this scan
 * @param dir The directory from which to read
 * @param maxDist Reads only Points up to this Distance
 * @param minDist Reads only Points from this Distance
 * @param euler Initital pose estimates (will not be applied to the points
 * @param ptss Vector containing the read points
 */
int ScanIO_oct::readScans(int start, int end, string &dir, int maxDist, int mindist,
			  double *euler, vector<Point> &ptss)
{
  static int fileCounter = start;
  string scanFileName;
  string poseFileName;

  ifstream pose_in;

  //double maxDist2 = sqr(maxDist);

  int my_fileNr = fileCounter;
  
  if (end > -1 && fileCounter > end) return -1; // 'nuf read
  scanFileName = dir + "scan" + to_string(fileCounter,3) + ".oct";
  poseFileName = dir + "scan" + to_string(fileCounter,3) + ".pose";
  
//  scan_in.open(scanFileName.c_str());
  pose_in.open(poseFileName.c_str());

  // read 3D scan
  if (!pose_in.good() ) return -1; // no more files in the directory
  if (!pose_in.good()) { cerr << "ERROR: Missing file " << poseFileName << endl; exit(1); }
  cout << "Processing Scan " << scanFileName;
  
  for (unsigned int i = 0; i < 6; pose_in >> euler[i++]);

  cout << " @ pose (" << euler[0] << "," << euler[1] << "," << euler[2]
	  << "," << euler[3] << "," << euler[4] << ","  << euler[5] << ")" << endl;
  
  // convert angles from deg to rad
  for (unsigned int i = 3; i <= 5; i++) euler[i] = rad(euler[i]);

  BOctTree<float>::deserialize(scanFileName, ptss);

  pose_in.close();
  pose_in.clear();
  fileCounter++;
  
  return  my_fileNr;
}

