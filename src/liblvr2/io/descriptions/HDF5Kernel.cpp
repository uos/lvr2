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
    const std::string &groupName,
    const std::string &datasetName,
    const cv::Mat &img) const
{
    if(m_hdf5File && m_hdf5File->isValid())
    {
        HighFive::Group group = hdf5util::getGroup(m_hdf5File, groupName, true);

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

            // if(m_hdf5File->m_chunkSize)
            // {
            //     for(size_t i = 0; i < chunkSizes.size(); i++)
            //     {
            //         if(chunkSizes[i] > dims[i])
            //         {
            //             chunkSizes[i] = dims[i];
            //         }
            //     }
            //     properties.add(HighFive::Chunking(chunkSizes));
            // }
            // if(m_hdf5File->m_compress)
            // {
            //     properties.add(HighFive::Deflate(9));
            // }

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

    } else {
        throw std::runtime_error("[Hdf5 - ImageIO]: Hdf5 file not open.");
    }
    

    return ret;
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