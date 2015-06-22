/**
 * @file
 * @brief Scan types and mapping functions.
 *
 * @author Thomas Escher
 */

#ifndef IO_TYPES_H
#define IO_TYPES_H

//! IO types for file formats, distinguishing the use of ScanIOs
enum IOType {
  UOS, UOS_MAP, UOS_FRAMES, UOS_MAP_FRAMES, UOS_RGB, OLD, RTS, RTS_MAP, RIEGL_TXT, RIEGL_PROJECT, RIEGL_RGB, RIEGL_BIN, IFP, ZAHN, PLY, WRL, XYZ, ZUF, ASC, IAIS, FRONT, X3D, RXP, KIT, AIS, OCT, TXYZR, XYZR, XYZ_RGB, KS, KS_RGB, STL, LEICA, PCL, PCI, UOS_CAD, VELODYNE, VELODYNE_FRAMES
};

//! Data channels in the scans
enum IODataType {
  DATA_XYZ = 1<<0,
  DATA_RGB = 1<<1,
  DATA_REFLECTANCE = 1<<2,
  DATA_AMPLITUDE = 1<<3,
  DATA_TYPE = 1<<4,
  DATA_DEVIATION = 1<<5
};

IOType formatname_to_io_type(const char * string);

const char * io_type_to_libname(IOType type);

#endif //IO_TYPES_H
