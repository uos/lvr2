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

#ifndef __DIRECTORY_PARSER_HPP__
#define __DIRECTORY_PARSER_HPP__

#include <string>
#include <vector>

#include <boost/filesystem.hpp>
#include <Eigen/Dense>

#include "lvr2/io/Timestamp.hpp"
#include "lvr2/io/PointBuffer.hpp"
#include "lvr2/types/MatrixTypes.hpp"
#include "lvr2/registration/TransformUtils.hpp"

namespace lvr2
{

struct ScanInfo
{
    string              m_filename;
    size_t              m_numPoints;
    Transformd          m_pose;
};

class ScanDirectoryParser
{
   
public:
    ScanDirectoryParser(const std::string& directory) noexcept;

    void setPointCloudPrefix(const std::string& prefix);
    void setPointCloudExtension(const std::string& extension);
    void setPosePrefix(const std::string& prefix);
    void setPoseExtension(const std::string& extension);

    void setStart(int s);
    void setEnd(int e);

    void parseDirectory();

    PointBufferPtr randomSubSample(const size_t& targetSize);
    PointBufferPtr octreeSubSample(const double& voxelSize, const size_t& minPoints = 5);
    
    ~ScanDirectoryParser();

private:

    using Path = boost::filesystem::path;

    size_t examinePLY(const std::string& filename);
    size_t examineASCII(const std::string& filename);    

    size_t                  m_numPoints;
    std::string             m_pointPrefix;
    std::string             m_posePrefix;
    std::string             m_poseExtension;
    std::string             m_pointExtension;
    std::string             m_directory;

    size_t                  m_start;
    size_t                  m_end;

    std::vector<ScanInfo*>   m_scans;
};

} // namespace lvr2

#endif