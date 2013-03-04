/**
 * @file
 * @brief IO of a 3D scan
 * @author Kai Lingemann. Institute of Computer Science, University of Osnabrueck, Germany.
 * @author Andreas Nuechter. Institute of Computer Science, University of Osnabrueck, Germany.
 * @author Thomas Escher
 */

#ifndef __SCAN_IO_H__
#define __SCAN_IO_H__

#include "slam6d/io_types.h"
#include "slam6d/pointfilter.h"

#include <string>
#include <list>
#include <map>
#include <vector>



/**
 * @brief IO of a 3D scan
 *
 * This class needs to be instantiated by a class loading 
 * 3D scans from different file formats.
 */
class ScanIO {
public:
  /**
   * Read a directory and return all possible scans in the [start,end] interval.
   *
   * @param dir_path The directory from which to read the scans
   * @param start Starting index
   * @param end Last index
   * @return List of IO-specific identifiers of scans, matching the search
   */
  virtual std::list<std::string> readDirectory(const char* dir_path, unsigned int start, unsigned int end) = 0;
  
  /**
   * Reads the pose from a dedicated pose file or from the scan file.
   *
   * @param dir_path The directory the scan is contained in
   * @param scan_identifier IO-specific identifier for the particular scan
   * @param pose Pointer to an existing double[6] array where the pose is saved in
   */
  virtual void readPose(const char* dir_path, const char* identifier, double* pose) = 0;

  /**
   * Given a scan identifier, load the contents of this particular scan.
   *
   * @param dir_path The directory the scan is contained in
   * @param identifier IO-specific identifier for the particular scan
   * @param filter Filter object which each point is tested on by its position
   */
  virtual void readScan(const char* dir_path, const char* identifier, PointFilter& filter, std::vector<double>* xyz = 0, std::vector<unsigned char>* rgb = 0, std::vector<float>* reflectance = 0, std::vector<float>* amplitude = 0, std::vector<int>* type = 0, std::vector<float>* deviation = 0) = 0;
  
  /**
   * Returns whether this ScanIO can load the requested data from a scan.
   *
   * @param type data channel request
   * @return whether it's supported or not
   */
  virtual bool supports(IODataType type) = 0;
  
  /**
   * @brief Global mapping of io_types to single instances of ScanIOs.
   *
   * If the ScanIO doesn't exist, it will be created and saved in a map.
   * Otherwise, the matching ScanIO will be returned.
   *
   * @param type Key identifying the ScanIO
   * @return The newly created or found ScanIO 
   */
  static ScanIO* getScanIO(IOType iotype);
  
  //! Delete all ScanIO instances and (lazy) try to unload the libraries.
  static void clearScanIOs();
private:
  static std::map<IOType, ScanIO *> m_scanIOs;
};

// Since the shared object files are loaded on the fly, we
// need class factories

// the types of the class factories
typedef ScanIO* create_sio();
typedef void destroy_sio(ScanIO*);

#endif
