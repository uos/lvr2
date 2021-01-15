#include "lvr2/io/descriptions/HDF5Kernel.hpp"
#include "lvr2/io/yaml/MetaNodeDescriptions.hpp"
#include "lvr2/io/hdf5/Hdf5Util.hpp"
#include "lvr2/types/ScanTypes.hpp"

namespace lvr2
{

HDF5Kernel::HDF5Kernel(const std::string& rootFile) : FileKernel(rootFile)
{
    std::cout << "[HDF5Kernel - HDF5Kernel]: Open File" << std::endl;
    m_hdf5File = hdf5util::open(rootFile);
    m_metaDescription = new HDF5MetaDescriptionV2;
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

    std::string tmp = "PointBuffer";
    std::cout << "Set Attribute type='PointBuffer' at group '" << group << "'" << std::endl;
    hdf5util::setAttribute(g, "type", tmp);

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

        std::vector<std::string> names = hdf5util::splitGroupNames(datasetName);

        if(names.size() > 1)
        {
            std::cout << "Found group in dataset name" << std::endl;
            
            for(auto name : names)
            {
                std::cout << "-- " << name << std::endl;
            }
        }

        int w = img.cols;
        int h = img.rows;
        const char* interlace = "INTERLACE_PIXEL";

        // TODO: need to remove dataset if exists because of H5IMmake_image_8bit. do it better
        
        if(img.type() == CV_8U) 
        {
            if(group.exist(datasetName))
            {
                H5Ldelete(group.getId(), datasetName.data(), H5P_DEFAULT);
            }
            H5IMmake_image_8bit(group.getId(), datasetName.c_str(), w, h, img.data);
        } 
        else if(img.type() == CV_8UC3) 
        {
            if(group.exist(datasetName))
            {
                H5Ldelete(group.getId(), datasetName.data(), H5P_DEFAULT);
            }
            // bgr -> rgb?
            H5IMmake_image_24bit(group.getId(), datasetName.c_str(), w, h, interlace, img.data);
        } 
        else 
        {
            // std::cout << "[Hdf5IO - ImageIO] WARNING: OpenCV type not implemented -> " 
            //     << img.type() << ". Saving image as data blob." << std::endl;

            // Couldnt write as H5Image, write as blob

            std::vector<size_t> dims = {static_cast<size_t>(img.rows), static_cast<size_t>(img.cols)};
            std::vector<hsize_t> chunkSizes = {static_cast<hsize_t>(img.rows), static_cast<hsize_t>(img.cols)};

            if(img.channels() > 1)
            {
                dims.push_back(img.channels());
                chunkSizes.push_back(img.channels());
            }

            HighFive::DataSpace dataSpace(dims);
            HighFive::DataSetCreateProps properties;

            // Single Channel Type
            const int SCTYPE = img.type() % 8;

            if(SCTYPE == CV_8U) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<unsigned char>(
                    group, datasetName, dataSpace, properties
                );
                const unsigned char* ptr = reinterpret_cast<unsigned char*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_8S) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<char>(
                    group, datasetName, dataSpace, properties
                );
                const char* ptr = reinterpret_cast<char*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_16U) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<unsigned short>(
                    group, datasetName, dataSpace, properties
                );
                const unsigned short* ptr = reinterpret_cast<unsigned short*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_16S) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<short>(
                    group, datasetName, dataSpace, properties
                );
                const short* ptr = reinterpret_cast<short*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_32S) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<int>(
                    group, datasetName, dataSpace, properties
                );
                const int* ptr = reinterpret_cast<int*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_32F) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<float>(
                    group, datasetName, dataSpace, properties
                );
                const float* ptr = reinterpret_cast<float*>(img.data);
                dataset->write(ptr);
            } 
            else if(SCTYPE == CV_64F) 
            {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<double>(
                    group, datasetName, dataSpace, properties
                );
                const double* ptr = reinterpret_cast<double*>(img.data);
                dataset->write(ptr);
            } 
            else 
            {
                std::cout << timestamp << "HDF5Kernel:SaveImage: Warning: unknown opencv type " << img.type() << std::endl;
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
    const std::string &group,
    const std::string &metaName,
    const YAML::Node &node) const
{
    HighFive::Group hg = hdf5util::getGroup(m_hdf5File, group);

    if(hg.isValid() && node["sensor_type"] )
    {
        std::string sensor_type = node["sensor_type"].as<std::string>();

        hdf5util::setAttribute(hg, "sensor_type", sensor_type);

        if(sensor_type == "Channel")
        {
            // TODO: How to write meta for channel?
            // Example:
            // meta: "points" (Group)
            // data: "points" (Dataset)
            
            // 1. approach:
            // do not write meta data for channel since type and dimensions are 
            // already stored in h5
            // drawback: we need custom types that are stored as unsigned char buffer
            //     h5 type and dimensions do not correspond to the channels type and dimensions
            //     for this, we definatly require additional meta information

            // 2. approach:
            // - write to dataset attributes
            // drawback: what to do if dataset does not exist yes?
            // - write empty dataset sized by meta data
            // drawback: write a dataset two times?
            // -> NO if you use hdf5util::createDataset
            
            size_t num_elements = node["dims"][0].as<size_t>();
            size_t width = node["dims"][1].as<size_t>();
            int type = node["channel_type"].as<int>();
            std::cout << "TODO: Channel Meta: " << std::endl;
            std::cout << "-- size: " << num_elements << "x" << width << std::endl;
            std::cout << "-- type: " << type << std::endl;

            // HighFive::DataSpace space({num_elements, width});
            // HighFive::DataSetCreateProps properties;
            // auto dataset = hdf5util::createDataset<float>(hg, metaName, space, properties);


            return;
        }

        HighFive::Group meta_group = hdf5util::getGroup(hg, metaName);

        // mark as meta for corresponding sensor_type
        bool meta_flag = true;
        hdf5util::setAttribute(meta_group, "meta", meta_flag );
        hdf5util::setAttribute(meta_group, "sensor_type", sensor_type);

        ScanPosition sp;

        if(sensor_type == ScanProject::sensorType)
        {
            m_metaDescription->saveScanProject(meta_group, node);
        }
        else if(sensor_type == ScanPosition::sensorType)
        {
            m_metaDescription->saveScanPosition(meta_group, node);
        }
        else if(sensor_type == Scan::sensorType)
        {
            m_metaDescription->saveScan(meta_group, node);
        }
        else if(sensor_type == ScanCamera::sensorType)
        {
            m_metaDescription->saveScanCamera(meta_group, node);
        }
        else if(sensor_type == ScanImage::sensorType)
        {
            m_metaDescription->saveScanImage(meta_group, node);
        }
        else if(sensor_type == HyperspectralCamera::sensorType)
        {
            m_metaDescription->saveHyperspectralCamera(meta_group, node);
        }
        else if(sensor_type == HyperspectralPanoramaChannel::sensorType)
        {
            m_metaDescription->saveHyperspectralPanoramaChannel(meta_group, node);
        } else 
        {
            std::cout << timestamp
                      << "HDF5Kernel::SaveMetaYAML(): Warning: Sensor type '"
                      << sensor_type << "' is not defined." << std::endl;
        }
        m_hdf5File->flush();
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


    // check if flags are correct
    // if(!isPointCloud(group) )
    // {
    //     std::cout << "[Hdf5IO - PointCloudIO] WARNING: flags of " << group.getId() << " are not correct." << std::endl;
    //     return ret;
    // }

    /*
    for(auto name : g.listObjectNames() )
    {
        //TODO: FIX ME - Varaint Type Error
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



    return ret;*/
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
            if(H5IMis_image(group.getId(), datasetName.c_str()))
            {
                long long unsigned int w, h, planes;
                long long int npals;
                char interlace[256];

                int err = H5IMget_image_info(group.getId(), datasetName.c_str(), &w, &h, &planes, interlace, &npals);

                if(planes == 1) {
                    // 1 channel image
                    ret = cv::Mat(h, w, CV_8U);
                    H5IMread_image(group.getId(), datasetName.c_str(), ret->data);
                } else if(planes == 3) {
                    // 3 channel image
                    ret = cv::Mat(h, w, CV_8UC3);
                    H5IMread_image(group.getId(), datasetName.c_str(), ret->data);
                } else {
                    // ERROR
                }
            } else {
                // data blob
                // std::cout << "[Hdf5 - ImageIO] WARNING: Dataset is not formatted as image. Reading data as blob." << std::endl;
                // Data is not an image, load as blob

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

    } 
    else 
    {
        throw std::runtime_error("[Hdf5 - ImageIO]: Hdf5 file not open.");
    }
    

    return ret;
}

void HDF5Kernel::loadMetaYAML(
    const std::string &group,
    const std::string &container,
    YAML::Node& node) const
{
    std::cout << "[HDF5Kernel - loadMetaYAML]: Open Meta YAML '" << group << " , " << container << "'" << std::endl;
    HighFive::Group hg = hdf5util::getGroup(m_hdf5File, group);

    if(hg.isValid())
    {
        if(hg.exist(container))
        {
            HighFive::Group meta_group = hdf5util::getGroup(hg, container, false);

            // std::cout << "Loading '" << group + "/" + container << "/sensorType'" << std::endl;

            boost::optional<std::string> sensor_type_opt = hdf5util::getAtomic<std::string>(meta_group, "sensorType");

            if(sensor_type_opt)
            {
                // std::cout << "Meta contains sensorType: " << *sensor_type_opt << std::endl;
                std::string sensor_type = *sensor_type_opt;

                if(sensor_type == "ScanProject")
                {
                    node = m_metaDescription->scanProject(meta_group);
                }
                else if(sensor_type == "ScanPosition")
                {
                    node = m_metaDescription->scanPosition(meta_group);
                }
                else if(sensor_type == "Scan")
                {
                    node = m_metaDescription->scan(meta_group);
                }
                else if(sensor_type == "ScanCamera")
                {
                    node = m_metaDescription->scanCamera(meta_group);
                }
                else if(sensor_type == "HyperspectralCamera")
                {
                    node = m_metaDescription->hyperspectralCamera(meta_group);
                }
                else if(sensor_type == "HyperspectralPanoramaChannel")
                {
                    node = m_metaDescription->hyperspectralPanoramaChannel(meta_group);
                }
                else 
                {
                    std::cout << timestamp
                            << "HDF5Kernel::LoadMetaYAML(): Warning: Sensor type '"
                            << sensor_type << "' is not defined." << std::endl;
                }
            }
        }
        else
        {
            std::cout << timestamp 
                    << "HDF5Kernel::loadMetaYAML(): Warning: Sensor type field missing." 
                    << std::endl;
        }
    } else {
        throw std::runtime_error("[Hdf5Kernel - loadMetaYAML]: Hdf5 file not open.");
    }
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

intArr HDF5Kernel::loadIntArray(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<int>(group, container, dims);
}

uint16Arr HDF5Kernel::loadUInt16Array(
    const std::string &group,
    const std::string &container,
    std::vector<size_t> &dims) const
{
    return this->template loadArray<uint16_t>(group, container, dims);
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
void HDF5Kernel::saveIntArray(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<int> &data) const
{
    this->template saveArray<int>(groupName, datasetName, dimensions, data);
}
void HDF5Kernel::saveUInt16Array(
    const std::string &groupName, const std::string &datasetName,
    const std::vector<size_t> &dimensions,
    const boost::shared_array<uint16_t> &data) const
{
    this->template saveArray<uint16_t>(groupName, datasetName, dimensions, data);
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


std::unordered_map<std::string, YAML::Node> HDF5Kernel::metas(
    const std::string& group) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    

    
    return ret;
}


std::unordered_map<std::string, YAML::Node> HDF5Kernel::metas(
    const std::string& group, const std::string& sensor_type) const
{
    std::unordered_map<std::string, YAML::Node> ret;

    std::cout << "FIND METAS OF SENSOR TYPE " << sensor_type << std::endl;

    HighFive::Group h5Group = hdf5util::getGroup(m_hdf5File, group, false);
    for(std::string groupName : h5Group.listObjectNames())
    {
        HighFive::ObjectType h5type = h5Group.getObjectType(groupName);

        if(h5type == HighFive::ObjectType::Group)
        {
            HighFive::Group metaGroup = h5Group.getGroup(groupName);
            if(metaGroup.hasAttribute("sensor_type"))
            {
                std::string tmp = sensor_type;
                if(hdf5util::checkAttribute(metaGroup, "sensor_type", tmp))
                {
                    std::cout << "-- " << groupName << ": MetaGroup of type " << sensor_type << std::endl; 
                }
            }
        }

        if(h5type == HighFive::ObjectType::Dataset)
        {
            HighFive::DataSet metaDataset = h5Group.getDataSet(groupName);
            if(metaDataset.hasAttribute("sensor_type"))
            {
                std::string tmp = sensor_type;
                if(hdf5util::checkAttribute(metaDataset, "sensor_type", tmp))
                {
                    std::cout << "-- " << groupName << ": MetaDataset of type " << sensor_type << std::endl; 
                }
            }
        }

        // try{
        //     HighFive::Group metaGroup = h5Group.getGroup(groupName);
        //     std::cout << "-- " << groupName << ": MetaGroup" << std::endl;
        // } catch(HighFive::GroupException ex) {
        //     std::cout << "NOT A GROUP" << std::endl;
        // }
        
        // HighFive::DataSet dataset = h5Group.getDataSet(groupName);
        // if(dataset.hasAttribute("sensor_type"))
        // {
        //     std::cout << "Dataset with sensor type found!" << std::endl;
        // }
    }

    return ret;
}

} // namespace lvr2
