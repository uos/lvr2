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

 /**
 * UosIO.h
 *
 *  @date 11.05.2011
 *  @author Thomas Wiemann
 */

#ifndef UOSIO_H_
#define UOSIO_H_

#define _USE_MATH_DEFINES

#include <string>
#include <iomanip>
#include <iostream>
#include <math.h>
#include <fstream>
#include <sstream>
#include <vector>

#include "lvr2/io/BaseIO.hpp"
#include "lvr2/io/AsciiIO.hpp"

#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"

using std::string;
using std::fstream;
using std::stringstream;

namespace lvr2
{

using Vec = BaseVector<float>;

/**
 * @brief An input class for laser scans in UOS 3d format.
 *
 * Per default, this class tries to read all scans in the given
 * directory. If you don't want to specify which scans to read,
 * use the \ref{setFirstScan} and \ref{setLastScan} methods to
 * specify an interval to read. If .frame files are present, the
 * single scans will be transformed according to the last
 * transformation in the file. If no .frame file are present, the
 * .pose files will be sued to transform the scans.
 */
class UosIO : public BaseIO
{

public:

    /**
     * @brief Contructor.
     */
    UosIO() :
        m_firstScan(-1),
        m_lastScan(-1),
        m_saveToDisk(false),
        m_reductionTarget(0),
        m_numScans(0),
        m_saveRemission(false),
        m_saveRemissionColor(false){}

    /**
     * @brief Reads all scans or an specified range of scans
     *        from the given directory
     *
     * @param dir       A directory containing scans in UOS format.
     * @return          An indexed array of scan points
     */
    ModelPtr read(string dir);


    /**
     * @brief Defines the first scan to read
     * @param n         The first scan to read
     */
    void setFirstScan(int n) { m_firstScan = n;}


    /**
     * @brief Defines the last scan to read
     * @param n         The last scan to read
     */
    void setLastScan(int n) {m_lastScan = n;}


    /**
     * Reduces the given point cloud and exports all points
     * into on single file.
     *
     * @param dir        The directory containg the scan data
     * @param reduction  Reduction factor (export only every n-th point)
     * @param target     A target file name
     */
    void reduce(string dir, string target, int reduction = 1);


    /**
     * \todo Implement this!
     * \warning This function is not yet implemented!
     **/
    void save(string filename) {}



    /**
     * @brief Convert remission value to pseudo RGB values while saving
     */
    void saveRemissionAsColor(bool convert) {m_saveRemissionColor = convert;}


    /**
     * @brief Save remission (if present)
     */
    void saveRemission(bool yes) { m_saveRemission= yes;}


    /**
     * @brief Creates a transformation matrix from given frame file
     *
     * @param frameFile
     * @return          A matrix representing the final transformation
     *                  from a frame file
     */
    Matrix4<Vec> parseFrameFile(ifstream& frameFile);

private:

    /**
     * @brief Reads scans from \ref{first} to \ref{last} in new UOS format.
     * @param dir       The directory path
     * @param first     The first scan to read
     * @param last      The last scan to read
     * @return          All read data points
     */
    void readNewFormat(ModelPtr &m, string dir, int first, int last, size_t &n);


    /**
     * @brief Reads scans from \ref{first} to \ref{last} in old UOS format.
     * @param dir       The directory path
     * @param first     The first scan to read
     * @param last      The last scan to read
     * @return          All read data points
     */
    void readOldFormat(ModelPtr &m, string dir, int first, int last, size_t &n);


    inline std::string to_string(const int& t, int width)
    {
      stringstream ss;
      ss << std::setfill('0') << std::setw(width) << t;
      return ss.str();
    }


    inline std::string to_string(const int& t)
    {
      stringstream ss;
      ss << t;
      return ss.str();
    }


    /**
     * Converts an angle (given in deg) to rad
     *
     * @param deg integer indicating, whether the figure to be drawn to show
     * the clusters should be circles (0) or rectangles(1)
     *
     * @return the clustered image, with the clusters marked by colored figures
     *
     */
    inline float rad(const float deg)
    {
		return (float)((2 * M_PI * deg) / 360);
    }


    /**
     * Converts an angle (given in rad) to deg
     *
     * @param rad  angle in rad
     * @return     angle in deg
     */
    inline float deg(const float rad)
    {
		return (float)((rad * 360) / (2 * M_PI));
    }


    /// The first scan to read (or -1 if all scans should be processed)
    int     m_firstScan;

    /// The last scan to read (or -1 if all scans should be processed)
    int     m_lastScan;

    /// If true, the read point will not be stored in local memory
    bool    m_saveToDisk;

    /// Filestream to save reduced data
    ofstream    m_outputFile;

    /// Number of loaded scans
    int     m_numScans;

    /// Number of targeted points for reduction
    int     m_reductionTarget;

    /// If true, remission values will be converted to color
    bool    m_saveRemissionColor;

    /// If true, the original remission information will be saved
    bool    m_saveRemission;

};

} // namespace lvr2

#endif /* UOSIO_H_ */
