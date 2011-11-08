/* Copyright (C) 2011 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


 /**
 * UosIO.h
 *
 *  @date 11.05.2011
 *  @author Thomas Wiemann
 */

#ifndef UOSIO_H_
#define UOSIO_H_

#include <string>
#include <iomanip>
#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <vector>

#include "BaseIO.hpp"
#include "AsciiIO.hpp"

#include "../geometry/Vertex.hpp"
#include "../geometry/Matrix4.hpp"

using std::string;
using std::fstream;
using std::stringstream;

namespace lssr
{
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
        m_reductionTarget(1),
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
    Model* read(string dir);

    void save(Model* m, string filename) {};

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

    //indexPair getScanRange( size_t num );

    //int getNumScans() { return m_numScans;}

    void save(string filename) {}

    void saveRemissionAsColor(bool convert) {m_saveRemissionColor = convert;}

    void saveRemission(bool yes) { m_saveRemission= yes;}

private:

    /**
     * @brief Reads scans from \ref{first} to \ref{last} in new UOS format.
     * @param dir       The directory path
     * @param first     The first scan to read
     * @param last      The last scan to read
     * @return          All read data points
     */
    void readNewFormat(Model* &m, string dir, int first, int last, size_t &n);

    /**
     * @brief Reads scans from \ref{first} to \ref{last} in old UOS format.
     * @param dir       The directory path
     * @param first     The first scan to read
     * @param last      The last scan to read
     * @return          All read data points
     */
    void readOldFormat(Model* &m, string dir, int first, int last, size_t &n);

    /**
     * @brief Creates a transformation matrix from given frame file
     *
     * @param frameFile
     * @return          A matrix representing the final transformation
     *                  from a frame file
     */
    Matrix4<float> parseFrameFile(ifstream& frameFile);

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
      return ( (2 * M_PI * deg) / 360 );
    }

    /**
     * Converts an angle (given in rad) to deg
     *
     * @param rad  angle in rad
     * @return     angle in deg
     */
    inline float deg(const float rad)
    {
      return ( (rad * 360) / (2 * M_PI) );
    }


    /// The first scan to read (or -1 if all scans should be processed)
    int     m_firstScan;

    /// The last scan to read (or -1 if all scans should be processed)
    int     m_lastScan;

    /// If true, the read point will not be stored in local memory
    bool    m_saveToDisk;

    /// Filestream to save reduced data
    ofstream    m_outputFile;

    /// Reduction factor
    int     m_reductionTarget;


    /// Number of loaded scans
    int     m_numScans;

    bool    m_saveRemissionColor;

    bool    m_saveRemission;
};

} // namespace lssr


#endif /* UOSIO_H_ */
