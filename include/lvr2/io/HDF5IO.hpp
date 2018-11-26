/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef HDF5IO_HPP_
#define HDF5IO_HPP_

#include "BaseIO.hpp"
#include "DataStruct.hpp"
#include "ScanData.hpp"
#include "CalibrationParameters.hpp"

#include "lvr2/geometry/Matrix4.hpp"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <H5Tpublic.h>
#include <hdf5_hl.h>


#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
#include <string>

namespace lvr2
{

class HDF5IO : public BaseIO
{
  public:
    /**
         * \brief Parse the given file and load supported elements.
         *#include <highfive/H5DataSet.hpp>
#include <highfive/H5DataSpace.hpp>
#include <highfive/H5File.hpp>
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
            std::string groupName,
            std::string datasetName,
            unsigned int size,
            floatArr data);

    void addFloatArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t>& dimensions,
            floatArr data);

    void addFloatArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t>& dimensions,
            std::vector<hsize_t>& chunkSize,
            floatArr data);

    void addUcharArray(
            std::string groupName,
            std::string datasetName,
            unsigned int size,
            ucharArr data);

    void addUcharArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t>& dimensions,
            ucharArr data);

    void addUcharArray(
            std::string group,
            std::string name,
            std::vector<size_t>& dim,
            std::vector<hsize_t>& chunks,
            ucharArr data);


    void addImage(
            std::string groupName, std::string name, cv::Mat& img);

    void addRawScanData(int nr, ScanData &scan);

    void addFloatChannelToRawScanData(std::string name, int nr, size_t n, unsigned w, floatArr data);

    void addRawDataHeader(std::string description, Matrix4<BaseVector<float>> &referenceFrame);

    void addHyperspectralCalibration(int position, const HyperspectralCalibration& calibration);

    void setCompress(bool compress);
    void setChunkSize(const size_t& size);

    bool compress();

    size_t chunkSize();


  private:
    floatArr getFloatArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim);
    ucharArr getUcharArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim);
    Texture getImage(HighFive::Group& g, std::string datasetName);

    void addFloatArray(
            HighFive::Group& g,
            std::string datasetName,
            std::vector<size_t>& dim,
            std::vector<hsize_t>& chunkSize,
            floatArr data);

    void addUcharArray(
            HighFive::Group& g,
            std::string datasetName,
            std::vector<size_t>& dim,
            std::vector<hsize_t>& chunkSize,
            ucharArr data);

    void addImage(HighFive::Group& g, std::string datasetName, cv::Mat& img);

    HighFive::Group getGroup(const std::string& groupName, bool create = true);

    std::vector<std::string> splitGroupNames(const std::string &groupName);

    bool exist(const std::string &groupName);

    void write_base_structure();

    HighFive::File*         m_hdf5_file;

    bool                    m_compress;
    size_t                  m_chunkSize;
};

} // namespace lvr2

#endif /* !HDF5IO_HPP_ */
