
namespace lvr2 {

namespace hdf5features {

template<typename Derived>
void ImageIO<Derived>::save(std::string groupName,
    std::string datasetName,
    const cv::Mat& img)
{
    HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName);
    save(g, datasetName, img);
}

template<typename Derived>
void ImageIO<Derived>::save(HighFive::Group& group,
    std::string datasetName,
    const cv::Mat& img)
{

    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
        int w = img.cols;
        int h = img.rows;
        const char* interlace = "INTERLACE_PIXEL";

        // TODO: need to remove dataset if exists because of H5IMmake_image_8bit. do it better
        
        if(img.type() == CV_8U) {
            if(group.exist(datasetName)){H5Ldelete(group.getId(), datasetName.data(), H5P_DEFAULT);}
            H5IMmake_image_8bit(group.getId(), datasetName.c_str(), w, h, img.data);
        } else if(img.type() == CV_8UC3) {
            if(group.exist(datasetName)){H5Ldelete(group.getId(), datasetName.data(), H5P_DEFAULT);}
            // bgr -> rgb?
            H5IMmake_image_24bit(group.getId(), datasetName.c_str(), w, h, interlace, img.data);
        } else {
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

            if(m_file_access->m_chunkSize)
            {
                for(size_t i = 0; i < chunkSizes.size(); i++)
                {
                    if(chunkSizes[i] > dims[i])
                    {
                        chunkSizes[i] = dims[i];
                    }
                }
                properties.add(HighFive::Chunking(chunkSizes));
            }
            if(m_file_access->m_compress)
            {
                properties.add(HighFive::Deflate(9));
            }

            // Single Channel Type
            const int SCTYPE = img.type() % 8;
    

            if(SCTYPE == CV_8U) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<unsigned char>(
                    group, datasetName, dataSpace, properties
                );
                const unsigned char* ptr = reinterpret_cast<unsigned char*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_8S) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<char>(
                    group, datasetName, dataSpace, properties
                );
                const char* ptr = reinterpret_cast<char*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_16U) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<unsigned short>(
                    group, datasetName, dataSpace, properties
                );
                const unsigned short* ptr = reinterpret_cast<unsigned short*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_16S) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<short>(
                    group, datasetName, dataSpace, properties
                );
                const short* ptr = reinterpret_cast<short*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_32S) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<int>(
                    group, datasetName, dataSpace, properties
                );
                const int* ptr = reinterpret_cast<int*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_32F) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<float>(
                    group, datasetName, dataSpace, properties
                );
                const float* ptr = reinterpret_cast<float*>(img.data);
                dataset->write(ptr);
            } else if(SCTYPE == CV_64F) {
                std::unique_ptr<HighFive::DataSet> dataset = hdf5util::createDataset<double>(
                    group, datasetName, dataSpace, properties
                );
                const double* ptr = reinterpret_cast<double*>(img.data);
                dataset->write(ptr);
            } else {
                std::cout << "[Hdf5IO - ImageIO] WARNING: unknown opencv type " << img.type() << std::endl;
            }
        }
        m_file_access->m_hdf5_file->flush();
    
    } else {
        throw std::runtime_error("[Hdf5IO - ChannelIO]: Hdf5 file not open.");
    }
}

template<typename Derived>
boost::optional<cv::Mat> ImageIO<Derived>::load(HighFive::Group& group,
    std::string datasetName)
{
    boost::optional<cv::Mat> ret;

    if(m_file_access->m_hdf5_file && m_file_access->m_hdf5_file->isValid())
    {
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
                    std::cout << "[Hdf5 - ImageIO] WARNING: Couldnt load blob. Datatype unkown." << std::endl;
                }
            }
        }

    } else {
        throw std::runtime_error("[Hdf5 - ImageIO]: Hdf5 file not open.");
    }
    

    return ret;
}

template<typename Derived>
boost::optional<cv::Mat> ImageIO<Derived>::load(std::string groupName,
    std::string datasetName)
{
    boost::optional<cv::Mat> ret;

    if(hdf5util::exist(m_file_access->m_hdf5_file, groupName))
    {
        HighFive::Group g = hdf5util::getGroup(m_file_access->m_hdf5_file, groupName, false);
        ret = load(g, datasetName);
    } 

    return ret;
}

template<typename Derived>
boost::optional<cv::Mat> ImageIO<Derived>::loadImage(std::string groupName,
    std::string datasetName)
{
    return load(groupName, datasetName);
}

/// PROTECTED
template<typename Derived>
template<typename T>
cv::Mat ImageIO<Derived>::createMat(std::vector<size_t>& dims) {
    cv::Mat ret;

    // single channel type
    int cv_type = cv::DataType<T>::type;

    if(dims.size() > 2)
    {
        cv_type += (dims[2]-1) * 8;
    }

    if(dims.size() > 1)
    {
        ret = cv::Mat(dims[0], dims[1], cv_type);
    } else {
        ret = cv::Mat(dims[0], 1, cv_type);
    }

    return ret;
}


} // namespace hdf5features

} // namespace lvr2