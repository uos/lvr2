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
#include <lvr2/io/MeshIOInterface.hpp>
#include <string>

namespace lvr2
{

class HDF5IO : public BaseIO, public MeshIOInterface
{
  public:

    static const std::string vertices_name;
    static const std::string indices_name;
    static const std::string meshes_group;

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

    HDF5IO(std::string filename, bool truncate = false);

    /**
     * @brief Constructs a HDFIO io object to read a HDF5 file with the given filename.
     * It accesses the mesh or point cloud with the given part name. The selected part can then be read and write
     * to and from a Model pointer. This will automatically load and store all provided channels. The channels will
     * be loaded if one accessing them in a lazy fashion.
     * @param filename  The HDF5 file filename
     * @param part_name The part in the HDF5 file which should be saved or loaded.
     * @param truncate  Open flag: Truncate the HDF5 file if already existing
     */
    HDF5IO(const std::string filename, const std::string part_name, bool truncate = false);

    virtual ~HDF5IO();

    bool open(std::string filename, bool truncate);

    template<typename T>
    boost::shared_array<T> getArray(
            std::string groupName, std::string datasetName,
            unsigned int& size);

    template<typename T>
    boost::shared_array<T> getArray(
            std::string groupName, std::string datasetName,
            std::vector<size_t>& dim);

    Texture getImage(std::string groupName, std::string datasetName);

    ScanData    getSingleRawScanData(int nr, bool load_points = true);

    std::vector<ScanData> getRawScanData(bool load_points = true);

    floatArr getFloatChannelFromRawScanData(std::string name,
            int nr, unsigned int& n, unsigned& w);

    template<typename T>
    void addArray(
            std::string groupName,
            std::string datasetName,
            unsigned int size,
            boost::shared_array<T> data);

    template<typename T>
    void addArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t>& dimensions,
            boost::shared_array<T> data);

    template<typename T>
    void addArray(
            std::string groupName,
            std::string datasetName,
            std::vector<size_t>& dimensions,
            std::vector<hsize_t>& chunkSize,
            boost::shared_array<T> data);


    void addImage(
            std::string groupName, std::string name, cv::Mat& img);

    void addRawScanData(int nr, ScanData &scan);

    void addFloatChannelToRawScanData(std::string name, int nr, size_t n, unsigned w, floatArr data);

    void addRawDataHeader(std::string description, Matrix4<BaseVector<float>> &referenceFrame);

    void addHyperspectralCalibration(int position, const HyperspectralCalibration& calibration);

    void setCompress(bool compress);
    void setChunkSize(const size_t& size);
    void setPreviewReductionFactor(const unsigned int factor);
    void setUsePreviews(bool use);

    bool compress();

    size_t chunkSize();


  private:

    /**
     * @brief Persistence layer interface, Accesses the vertices of the mesh in the persistence layer.
     * @return An optional float channel, the channel is valid if the mesh vertices have been read successfully
     */
    virtual FloatChannelOptional getVertices();

    /**
     * @brief Persistence layer interface, Accesses the face indices of the mesh in the persistence layer.
     * @return An optional index channel, the channel is valid if the mesh indices have been read successfully
     */
    virtual IndexChannelOptional getIndices();

    /**
     * @brief Persistence layer interface, Writes the vertices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    virtual bool addVertices(const FloatChannel& channel_ptr);

    /**
     * @brief Persistence layer interface, Writes the face indices of the mesh to the persistence layer.
     * @return true if the channel has been written successfully
     */
    virtual bool addIndices(const IndexChannel& channel_ptr);

    template <typename T>
    bool getChannel(const std::string group, const std::string name, boost::optional<AttributeChannel<T>>& channel);

    /**
     * @brief getChannel  Reads a float attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, FloatChannelOptional& channel);

    /**
     * @brief getChannel  Reads an index attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, IndexChannelOptional& channel);

    /**
     * @brief getChannel  Reads an unsigned char attribute channel in the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel
     * @return            true if the channel has been loaded successfully, false otherwise
     */
    virtual bool getChannel(const std::string group, const std::string name, UCharChannelOptional& channel);

    template <typename T>
    bool addChannel(const std::string group, const std::string name, const AttributeChannel<T>& channel);

    /**
     * @brief addChannel  Writes a float attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the float channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const FloatChannel& channel);

    /**
     * @brief addChannel  Writes an index attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the index channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const IndexChannel& channel);

    /**
     * @brief addChannel  Writes an unsigned char attribute channel from the given group with the given name
     * @param group       The associated attribute group
     * @param name        The associated attribute name
     * @param channel     The pointer to the unsigned char channel which should be written
     * @return            true if the channel has been written successfully, false otherwise
     */
    virtual bool addChannel(const std::string group, const std::string name, const UCharChannel& channel);

    boost::optional<HighFive::Group> getMeshGroup(bool create = false);

    template<typename T>
    boost::shared_array<T> getArray(HighFive::Group& g, std::string datasetName, std::vector<size_t>& dim);

    Texture getImage(HighFive::Group& g, std::string datasetName);

    template<typename T>
    void addArray(HighFive::Group& g,
            std::string datasetName,
            std::vector<size_t>& dim,
            std::vector<hsize_t>& chunkSize,
            boost::shared_array<T>& data);

    void addImage(HighFive::Group& g, std::string datasetName, cv::Mat& img);

    HighFive::Group getGroup(const std::string& groupName, bool create = true);

    std::vector<std::string> splitGroupNames(const std::string &groupName);

    bool exist(const std::string &groupName);

    void write_base_structure();

    bool isGroup(HighFive::Group grp, std::string objName);

    template <typename T>
    boost::shared_array<T> reduceData(boost::shared_array<T> data,
            size_t dataCount,
            size_t dataWidth,
            unsigned int reductionFactor,
            size_t *reducedDataCount);

    HighFive::File*         m_hdf5_file;

    bool                    m_compress;
    size_t                  m_chunkSize;
    bool                    m_usePreviews;
    unsigned int            m_previewReductionFactor;
    bool                    m_truncate;
    std::string             m_part_name;
    std::string             m_mesh_path;
};

} // namespace lvr2

#include "HDF5IO.tcc"

#endif /* !HDF5IO_HPP_ */
