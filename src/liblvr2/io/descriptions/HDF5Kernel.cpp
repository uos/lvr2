#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/yaml.hpp"
//#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

HDF5Kernel::HDF5Kernel(const std::string& rootFile, HDF5KernelConfig config) 
:FileKernel(rootFile)
,m_config(config)
{
    boost::filesystem::path p(rootFile);

    if(p.has_parent_path() && !boost::filesystem::exists(p.parent_path()))
    {
        boost::filesystem::create_directory(p.parent_path());
    }
    m_hdf5File = hdf5util::open(rootFile);
}

void HDF5Kernel::saveMeshBuffer(
    const std::string &group,
    const std::string &container,
    const MeshBufferPtr &buffer) const
{
    std::cout << "[HDF5Kernel - saveMeshBuffer] not implemented!" << std::endl; 
}

void HDF5Kernel::savePointBuffer(
    const std::string &group,
    const std::string &container,
    const PointBufferPtr &buffer) const
{
    // std::cout <<  "[HDF5Kernel - savePointBuffer]: " << group  <<  ", "  << container << std::endl;
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);

    // std::string tmp = "PointBuffer";
    // std::cout << "Set Attribute type='PointBuffer' at group '" << group << "'" << std::endl;
    // hdf5util::setAttribute(g, "type", tmp);

    for(auto elem : *buffer)
    {
        this->template save(g, elem.first, elem.second);
    }
}

void HDF5Kernel::saveImage(
    const std::string& group,
    const std::string& container,
    const cv::Mat &img) const
{
    if(m_hdf5File && m_hdf5File->isValid())
    {
        std::string groupName, datasetName;
        std::tie(groupName, datasetName) = hdf5util::validateGroupDataset(group, container);

        HighFive::Group group = hdf5util::getGroup(m_hdf5File, groupName, true);

        size_t w = img.cols;
        size_t h = img.rows;
        size_t c = img.channels();

        std::vector<size_t> dims = {h, w};
        std::vector<hsize_t> chunkSizes = {h, w};

        if(c > 1)
        {
            dims.push_back(c);
            chunkSizes.push_back(c);
        }

        HighFive::DataSpace space(dims);
        HighFive::DataSetCreateProps properties;

        if(m_config.compressionLevel > 0)
        {
            properties.add(HighFive::Chunking(chunkSizes));
            properties.add(HighFive::Deflate(m_config.compressionLevel));
        }

        // Single Channel Type
        const int SCTYPE = img.type() % 8;


        std::unique_ptr<HighFive::DataSet> dataset;

        if(SCTYPE == CV_8U) 
        {
            dataset = hdf5util::createDataset<unsigned char>(
                group, datasetName, space, properties
            );
            const unsigned char* ptr = reinterpret_cast<unsigned char*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_8S) 
        {
            dataset = hdf5util::createDataset<char>(
                group, datasetName, space, properties
            );
            const char* ptr = reinterpret_cast<char*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_16U) 
        {
            dataset = hdf5util::createDataset<unsigned short>(
                group, datasetName, space, properties
            );
            const unsigned short* ptr = reinterpret_cast<unsigned short*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_16S) 
        {
            dataset = hdf5util::createDataset<short>(
                group, datasetName, space, properties
            );
            const short* ptr = reinterpret_cast<short*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_32S) 
        {
            dataset = hdf5util::createDataset<int>(
                group, datasetName, space, properties
            );
            const int* ptr = reinterpret_cast<int*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_32F) 
        {
            dataset = hdf5util::createDataset<float>(
                group, datasetName, space, properties
            );
            const float* ptr = reinterpret_cast<float*>(img.data);
            dataset->write_raw(ptr);
        } 
        else if(SCTYPE == CV_64F) 
        {
            dataset = hdf5util::createDataset<double>(
                group, datasetName, space, properties
            );
            const double* ptr = reinterpret_cast<double*>(img.data);
            dataset->write_raw(ptr);
        } else 
        {
            std::cout << timestamp << "HDF5Kernel:SaveImage: Warning: unknown opencv type " << img.type() << std::endl;
        }

        if(dataset)
        {
            // Add additional attributes for specific images such that it can be displayed in HDF file viewer
            if(img.type() == CV_8U)
            {
                // HDFCompass cannot show this. But it is the correct way to store it
                // - H5IMmake_image_8bit(group.getId(), datasetName.c_str(), w, h, img.data)
                //     stores the same attributes
                hdf5util::setAttribute<std::string>(*dataset, "CLASS", "IMAGE");
                hdf5util::setAttribute<std::string>(*dataset, "IMAGE_VERSION", "1.2");
                hdf5util::setAttribute<std::string>(*dataset, "IMAGE_SUBCLASS", "IMAGE_INDEXED");
            }
            else if(img.type() == CV_8UC3) 
            {

                hdf5util::setAttribute<std::string>(*dataset, "CLASS", "IMAGE");
                hdf5util::setAttribute<std::string>(*dataset, "IMAGE_VERSION", "1.2");
                hdf5util::setAttribute<std::string>(*dataset, "IMAGE_SUBCLASS", "IMAGE_TRUECOLOR");
                hdf5util::setAttribute<std::string>(*dataset, "INTERLACE_MODE", "INTERLACE_PIXEL");
            } 
        }
        
        m_hdf5File->flush();
    } 
    else 
    {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
}

void HDF5Kernel::saveMetaYAML(
    const std::string &group_,
    const std::string &metaName,
    const YAML::Node &node) const
{
    // std::cout << "[HDF5Kernel - saveMetaYAML] " << group << ", " << metaName << std::endl;
    std::string group, container;
    std::tie(group, container) = hdf5util::validateGroupDataset(group_, metaName);

    // std::cout << "[HDF5Kernel - saveMetaYAML] checking " << group << ", " << container << std::endl;
    HighFive::Group hg = hdf5util::getGroup(m_hdf5File, group);
    
    if(hg.isValid())
    {
        // std::cout << "[HDF5Kernel - saveMetaYAML] Save META to " << group << ", " << container << std::endl;
        if(hg.exist(container))
        {
            HighFive::ObjectType h5type = hg.getObjectType(container);

            if(h5type == HighFive::ObjectType::Group)
            {
                HighFive::Group attgroup = hg.getGroup(container);
                hdf5util::setAttributeMeta(attgroup, node);
            }
            else if(h5type == HighFive::ObjectType::Dataset) 
            {
                HighFive::DataSet attds = hg.getDataSet(container);
                hdf5util::setAttributeMeta(attds, node);
            }

        } else {
            // Group or Dataset does not exist yet. 
            // Assuming it will be a group.
            // Create new group
            HighFive::Group attgroup = hdf5util::getGroup(hg, container);
            hdf5util::setAttributeMeta(attgroup, node);
        }
    } else {
        // Group not valid
        std::cout << "[HDF5Kernel - saveMetaYAML] ERROR - Group " << group << " not valid" << std::endl; 
    }
}

MeshBufferPtr HDF5Kernel::loadMeshBuffer(
    const std::string &group,
    const std::string container) const
{
    // old
    // return MeshBufferPtr(new MeshBuffer);
    MeshBufferPtr ret;

    return ret;
}

PointBufferPtr HDF5Kernel::loadPointBuffer(
    const std::string &group,
    const std::string &container) const
{
    // No:
    HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);
    PointBufferPtr ret;

    boost::shared_array<float> pointData;
    std::vector<size_t> pointDim;
    pointData = loadFloatArray(group, container, pointDim);    
    PointBufferPtr pb = PointBufferPtr(new PointBuffer(pointData, pointDim[0]));
    ret = pb;
    return ret;
}

boost::optional<cv::Mat> HDF5Kernel::loadImage(
    const std::string &groupName,
    const std::string &datasetName) const
{
    boost::optional<cv::Mat> ret;

    if(m_hdf5File && m_hdf5File->isValid())
    {
        HighFive::Group group = hdf5util::getGroup(m_hdf5File, groupName);
        if(group.exist(datasetName))
        {
            HighFive::DataSet dataset = group.getDataSet(datasetName);
            std::vector<size_t> dims = dataset.getSpace().getDimensions();
            HighFive::DataType dtype = dataset.getDataType();

            if(dtype == HighFive::AtomicType<unsigned char>()){
                ret = createMat<unsigned char>(dims);
                dataset.read(reinterpret_cast<unsigned char*>(ret->data));
            } else if(dtype == HighFive::AtomicType<char>()) {
                ret = createMat<char>(dims);
                dataset.read(reinterpret_cast<char*>(ret->data));
            } else if(dtype == HighFive::AtomicType<unsigned short>()) {
                ret = createMat<unsigned short>(dims);
                dataset.read(reinterpret_cast<unsigned short*>(ret->data));
            } else if(dtype == HighFive::AtomicType<short>()) {
                ret = createMat<short>(dims);
                dataset.read(reinterpret_cast<short*>(ret->data));
            } else if(dtype == HighFive::AtomicType<int>()) {
                ret = createMat<int>(dims);
                dataset.read(reinterpret_cast<int*>(ret->data));
            } else if(dtype == HighFive::AtomicType<float>()) {
                ret = createMat<float>(dims);
                dataset.read(reinterpret_cast<float*>(ret->data));
            } else if(dtype == HighFive::AtomicType<double>()) {
                ret = createMat<double>(dims);
                dataset.read(reinterpret_cast<double*>(ret->data));
            } else {
                std::cout << timestamp << "HDF5Kernel::loadImage(): Warning: Could'nt load blob. Datatype unkown." << std::endl;
            }
        }

    } 
    else 
    {
        throw std::runtime_error("[Hdf5 - ImageIO]: Hdf5 file not open.");
    }
    
    return ret;
}

bool HDF5Kernel::loadMetaYAML(
    const std::string &group_,
    const std::string &container_,
    YAML::Node& node) const
{
    std::string group, container;
    std::tie(group, container) = hdf5util::validateGroupDataset(group_, container_);

    HighFive::Group hg = hdf5util::getGroup(m_hdf5File, group, false);
    if(hg.isValid())
    {
        if(hg.exist(container))
        {
            HighFive::ObjectType h5type = hg.getObjectType(container);
            
            if(h5type == HighFive::ObjectType::Dataset)
            {
                HighFive::DataSet d = hg.getDataSet(container);
                node = hdf5util::getAttributeMeta(d);
            } 
            else if(h5type == HighFive::ObjectType::Group)
            {
                HighFive::Group g = hg.getGroup(container);
                node = hdf5util::getAttributeMeta(g);
            }
        } else {
            return false;
        }
    } else {
        return false;
    }

    return node.Type() != YAML::NodeType::Null && node.Type() != YAML::NodeType::Undefined;
}

charArr HDF5Kernel::loadCharArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<char>(group, container, dims);
}

ucharArr HDF5Kernel::loadUCharArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<unsigned char>(group, container, dims);
}

shortArr HDF5Kernel::loadShortArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<short>(group, container, dims);
}

ushortArr HDF5Kernel::loadUShortArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<unsigned short>(group, container, dims);
}

uint16Arr HDF5Kernel::loadUInt16Array(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<uint16_t>(group, container, dims);
}

intArr HDF5Kernel::loadIntArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<int>(group, container, dims);
}

uintArr HDF5Kernel::loadUIntArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<unsigned int>(group, container, dims);
}

lintArr HDF5Kernel::loadLIntArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<long int>(group, container, dims);
}

ulintArr HDF5Kernel::loadULIntArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<unsigned long int>(group, container, dims);
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

boolArr HDF5Kernel::loadBoolArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<bool>(group, container, dims);
}

void HDF5Kernel::saveCharArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<char> &data) const
{
    this->template saveArray<char>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveUCharArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<unsigned char> &data) const
{
    this->template saveArray<unsigned char>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveShortArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<short> &data) const
{
    this->template saveArray<short>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveUShortArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<unsigned short> &data) const
{
    this->template saveArray<unsigned short>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveUInt16Array(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<uint16_t> &data) const
{
    this->template saveArray<uint16_t>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveIntArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<int> &data) const
{
    this->template saveArray<int>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveUIntArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<unsigned int> &data) const
{
    this->template saveArray<unsigned int>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveLIntArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<long int> &data) const
{
    this->template saveArray<long int>(groupName, datasetName, dimensions, data);
}

void HDF5Kernel::saveULIntArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<unsigned long int> &data) const
{
    this->template saveArray<unsigned long int>(groupName, datasetName, dimensions, data);
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

void HDF5Kernel::saveBoolArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<bool> &data) const
{
    this->template saveArray<bool>(groupName, datasetName, dimensions, data);
}

bool HDF5Kernel::exists(const std::string &group) const
{
    return hdf5util::exist(m_hdf5File, group);
}

bool HDF5Kernel::exists(const std::string &group, const std::string &container) const
{
    //HighFive::Group g = hdf5util::getGroup(m_hdf5File, group);

    std::vector<std::string> ret;

    std::string remainder = group;
    size_t delimiter_pos = 0;

    while ((delimiter_pos = remainder.find('/', delimiter_pos)) != std::string::npos)
    {
        if (delimiter_pos > 0)
        {
            ret.push_back(remainder.substr(0, delimiter_pos));
        }

        remainder = remainder.substr(delimiter_pos + 1);

        delimiter_pos = 0;
    }

    if (remainder.size() > 0)
    {
        ret.push_back(remainder);
    }

    HighFive::Group g = m_hdf5File->getGroup("/");
    for( std::string substring : ret)
    {
        if(g.exist(substring))
        {   
            g = g.getGroup(substring);
        }else
        {
            return false;
        }
    } 
    //HighFive::Group g = m_hdf5File->getGroup(group);
    return hdf5util::exist(g, container);
}

void HDF5Kernel::subGroupNames(const std::string &group, std::vector<string> &subGroupNames) const
{
    HighFive::Group h5Group = hdf5util::getGroup(m_hdf5File, group);
    subGroupNames = h5Group.listObjectNames();
}

void HDF5Kernel::subGroupNames(const std::string &group, const std::regex &filter, std::vector<string> &subGroupNames) const
{

}

bool HDF5Kernel::getChannel(const std::string group, const std::string name, FloatChannelOptional& channel)  const
{
    return getChannel<float>(group, name, channel);
}


bool HDF5Kernel::getChannel(const std::string group, const std::string name, IndexChannelOptional& channel)  const
{
    return getChannel<unsigned int>(group, name, channel);
}


bool HDF5Kernel::getChannel(const std::string group, const std::string name, UCharChannelOptional& channel)  const
{
    return getChannel<unsigned char>(group, name, channel);
}


bool HDF5Kernel::addChannel(const std::string group, const std::string name, const FloatChannel& channel)  const
{
    return addChannel<float>(group, name, channel);
}


bool HDF5Kernel::addChannel(const std::string group, const std::string name, const IndexChannel& channel)  const
{
    return addChannel<unsigned int>(group, name, channel);
}


bool HDF5Kernel::addChannel(const std::string group, const std::string name, const UCharChannel& channel)  const
{
    return addChannel<unsigned char>(group, name, channel);
}


std::vector<std::string> HDF5Kernel::listDatasets(const std::string& group) const
{
    std::vector<std::string> ret;

    HighFive::Group h5Group = hdf5util::getGroup(m_hdf5File, group, false);

    for(std::string groupName : h5Group.listObjectNames())
    {
        HighFive::ObjectType h5type = h5Group.getObjectType(groupName);
        if(h5type == HighFive::ObjectType::Dataset)
        {
            ret.push_back(groupName);
        }
    }

    return ret;
}

std::unordered_map<std::string, YAML::Node> HDF5Kernel::metas(
    const std::string& group) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    

    
    return ret;
}


std::unordered_map<std::string, YAML::Node> HDF5Kernel::metas(
    const std::string& group, 
    const std::string& type) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    HighFive::Group h5Group = hdf5util::getGroup(m_hdf5File, group, false);
    for(std::string groupName : h5Group.listObjectNames())
    {
        HighFive::ObjectType h5type = h5Group.getObjectType(groupName);

        if(h5type == HighFive::ObjectType::Group)
        {
            HighFive::Group metaGroup = h5Group.getGroup(groupName);
            if(metaGroup.hasAttribute("type"))
            {
                std::string tmp = type;
                if(hdf5util::checkAttribute(metaGroup, "type", tmp))
                {
                    // Found a group with 'sensor_type' attribute: try to load yaml with loadMetaYAML
                    YAML::Node node = hdf5util::getAttributeMeta(metaGroup);
                    ret[groupName] = node;
                }
            }
        }

        if(h5type == HighFive::ObjectType::Dataset)
        {
            HighFive::DataSet metaDataset = h5Group.getDataSet(groupName);
            std::string tmp = type;

            if(metaDataset.hasAttribute("type"))
            {
                if(hdf5util::checkAttribute(metaDataset, "type", tmp))
                {
                    YAML::Node node = hdf5util::getAttributeMeta(metaDataset);
                    ret[groupName] = node;
                }
            }
        }
    }

    return ret;
}

} // namespace lvr2
