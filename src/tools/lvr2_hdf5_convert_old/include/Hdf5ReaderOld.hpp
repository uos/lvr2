#ifndef HDF5_READER_OLD_HPP
#define HDF5_READER_OLD_HPP

#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2 {

std::string string_shift(size_t nspaces);

ScanPtr loadScanOld(HighFive::Group g);

ScanPositionPtr loadScanPositionOld(HighFive::Group g);

ScanProjectPtr loadScanProjectOld(HighFive::Group g);

ScanProjectPtr loadOldHDF5(std::string filename);

} // namespace lvr2

#endif // HDF5_READER_OLD_HPP