#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"

namespace lvr2
{

HDF5Kernel::HDF5Kernel(const std::string& rootFile) : FileKernel(rootFile)
{
    m_hdf5File = hdf5util::open(rootFile);
}

void HDF5Kernel::saveMeshBuffer(
    const std::string &group,
    const std::string &container,
    const MeshBufferPtr &buffer) const
{

}

void HDF5Kernel::savePointBuffer(
    const std::string &group,
    const std::string &container,
    const PointBufferPtr &buffer) const
{
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);

    for(auto elem : *buffer)
    {
        this->template save(g, elem.first, elem.second);
    }
}


void HDF5Kernel::saveImage(
    const std::string &group,
    const std::string &container,
    const cv::Mat &image) const
{

}

void HDF5Kernel::saveMetaYAML(
    const std::string &group,
    const std::string &metaName,
    const YAML::Node &node) const
{

}

MeshBufferPtr HDF5Kernel::loadMeshBuffer(
    const std::string &group,
    const std::string container) const
{
    return MeshBufferPtr(new MeshBuffer);
}

PointBufferPtr HDF5Kernel::loadPointBuffer(
    const std::string &group,
    const std::string &container) const
{
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);
    PointBufferPtr ret;

    // check if flags are correct
    // if(!isPointCloud(group) )
    // {
    //     std::cout << "[Hdf5IO - PointCloudIO] WARNING: flags of " << group.getId() << " are not correct." << std::endl;
    //     return ret;
    // }

    for(auto name : g.listObjectNames() )
    {
        std::unique_ptr<HighFive::DataSet> dataset;

        try {
            dataset = std::make_unique<HighFive::DataSet>(
                g.getDataSet(name)
            );
        } catch(HighFive::DataSetException& ex) {

        }

        if(dataset)
        {
            // name is dataset
            boost::optional<PointBuffer::val_type> opt_vchannel
                 = this->template load<PointBuffer::val_type>(group, name);
            
            if(opt_vchannel)
            {
                if(!ret)
                {
                    ret.reset(new PointBuffer);
                }
                ret->insert({
                    name,
                    *opt_vchannel
                });
            }
            
        }

    }

    return ret;
}

boost::optional<cv::Mat> HDF5Kernel::loadImage(
    const std::string &group,
    const std::string &container) const
{
    return boost::none;
}

YAML::Node HDF5Kernel::loadMetaYAML(
    const std::string &group,
    const std::string &container) const
{
    YAML::Node node;
    return node;
}

ucharArr HDF5Kernel::loadUCharArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<unsigned char>(group, container, dims);
}

floatArr HDF5Kernel::loadFloatArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<float>(group, container, dims);
}

doubleArr HDF5Kernel::loadDoubleArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<double>(group, container, dims);
}

void HDF5Kernel::saveFloatArray(
    const std::string &groupName,
    const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<float> &data) const
{
    this->template saveArray<float>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveDoubleArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<double> &data) const
{
    this->template saveArray<double>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveUCharArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<unsigned char> &data) const
{
    this->template saveArray<unsigned char>(groupName, datasetName, dimensions, data);
}

bool HDF5Kernel::exists(const std::string &group) const
{
    return hdf5util::exist(m_hdf5File, group);
}

bool HDF5Kernel::exists(const std::string &group, const std::string &container) const
{
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);
    return hdf5util::exist(g, container);
}

void HDF5Kernel::subGroupNames(const std::string &group, std::vector<string> &subGroupNames) const
{
    
}

void HDF5Kernel::subGroupNames(const std::string &group, const std::regex &filter, std::vector<string> &subGroupNames) const
{

}

} // namespace lvr2