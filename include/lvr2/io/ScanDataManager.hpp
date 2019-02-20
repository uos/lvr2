#ifndef LVR2_IO_SCANDATAMANAGER_HPP
#define LVR2_IO_SCANDATAMANAGER_HPP

#include <vector>

#include <lvr2/io/HDF5IO.hpp>
#include <lvr2/io/ScanData.hpp>
#include <lvr2/io/CamData.hpp>

namespace lvr2
{

class ScanDataManager
{
    public:
        ScanDataManager(std::string filename);

        void loadPointCloudData(ScanData &sd, bool preview = false);

        std::vector<ScanData> getScanData();

        std::vector<std::vector<CamData> > getCamData();

    private:
        HDF5IO m_io;
};

} // namespace lvr2

#endif
