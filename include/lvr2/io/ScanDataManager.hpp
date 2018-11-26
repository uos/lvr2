#ifndef LVR2_IO_SCANDATAMANAGER_HPP
#define LVR2_IO_SCANDATAMANAGER_HPP

#include <vector>

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanData.hpp>

namespace lvr2
{

class ScanDataManager
{
    public:
        ScanDataManager(std::string filename);

        ScanData loadPointCloudData(size_t idx);

        std::vector<ScanData>& getScanData();

    private:
        std::vector<ScanData> m_scanData;
        HDF5IO m_io;
};

} // namespace lvr2

#endif
