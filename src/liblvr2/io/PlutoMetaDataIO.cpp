#include <yaml-cpp/yaml.h>
#include "lvr2/io/PlutoMetaDataIO.hpp"

namespace lvr2{
size_t PlutoMetaDataIO::readSpectralMetaData(const boost::filesystem::path& fn, floatArr& angles)
{
    std::vector<YAML::Node> root = YAML::LoadAllFromFile(fn.string());
    size_t size = 0;
    for(auto &n : root)
    {
        angles = floatArr(new float[n.size()]);
        std::cout << n.size() << std::endl;
        size = n.size();
        for(YAML::const_iterator it=n.begin();it!=n.end();++it)
        {
            // not sorted. key as index.
            angles[it->first.as<int>()] = it->second["angle"].as<float>() ;
        }
    }

    return size;
}

void PlutoMetaDataIO::readScanMetaData(const boost::filesystem::path& fn, ScanData& scan)
{
    // TODO parse.
    return;
}

} // namespace lvr2
