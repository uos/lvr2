#ifndef HDF5IO_HPP_
#define HDF5IO_HPP_

#include "BaseIO.hpp"
#include "DataStruct.hpp"
#include "ScanData.hpp"
#include "lvr2/geometry/Matrix4.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <H5Tpublic.h>
#include <hdf5_hl.h>
#include <highfive/H5File.hpp>

#include <string>

namespace lvr2
{

class HDF5IO : public BaseIO
{
  public:
    /**
         * \brief Parse the given file and load supported elements.
         *
         * @param filename  The file to read.
         */
    virtual ModelPtr read(std::string filename);

    ModelPtr read(std::string filename, size_t scanNr);

    /**
         * \brief Save the loaded elements to the given file.
         *
         * @param filename Filename of the file to write.
         */
    virtual void save(std::string filename);

    HDF5IO(std::string filename);
    virtual ~HDF5IO();

    bool open(std::string filename);

    floatArr getFloatArray(
            std::string groupName, std::string datasetName,
            unsigned int& size);

    floatArr getFloatArray(
            std::string groupName, std::string datasetName,
            std::vector<size_t>& dim);

    ucharArr getUcharArray(
            std::string groupName, std::string datasetName,
            unsigned int& size);

    ucharArr getUcharArray(
            std::string groupName, std::string datasetName,
            std::vector<size_t>& dim);

    Texture getImage(std::string groupName, std::string datasetName);

    ScanData getRawScanData(int nr);

    floatArr getFloatChannelFromRawScanData(std::string name,
            int nr, unsigned int& n, unsigned& w);

    void addFloatArray(
            std::string groupName, std::string datasetName,
            unsigned int size, floatArr data);

    void addFloatArray(
            std::string groupName, std::string datasetName,
            std::vector<size_t>& dimensions, floatArr data);

    void addUcharArray(
            std::string groupName,
            std::string datasetName,
            unsigned int size, ucharArr data);

    void addUcharArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t> dimensions, ucharArr data);

    void addImage(
            std::string groupName, std::string name, cv::Mat& img);

    void addRawScanData(int nr, ScanData &scan);

    void addFloatChannelToRawScanData(std::string name, int nr, size_t n, unsigned w, floatArr data);

    void addRawDataHeader(std::string description, Matrix4<BaseVector<float>> &referenceFrame);




  private:
    floatArr getFloatArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim);
    ucharArr getUcharArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim);
    Texture getImage(HighFive::Group& g, std::string datasetName);
    void addFloatArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim, floatArr data);
    void addUcharArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim, ucharArr data);
    void addImage(HighFive::Group& g, std::string datasetName, cv::Mat& img);

    HighFive::Group getGroup(const std::string& groupName, bool create = true);

    std::vector<std::string> splitGroupNames(const std::string &groupName);

    bool exist(const std::string &groupName);

    void write_base_structure();

    HighFive::File*         m_hdf5_file;
};

} // namespace lvr2

#endif /* !HDF5IO_HPP_ */
