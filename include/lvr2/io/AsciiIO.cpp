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


 /*
 * AsciiIO.cpp
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#include <fstream>
#include <string.h>
#include <algorithm>

using std::ifstream;

#include <boost/filesystem.hpp>

#include <lvr2/io/AsciiIO.hpp>
#include <lvr2/io/Progress.hpp>
#include <lvr2/io/Timestamp.hpp>

namespace lvr2
{

template<typename BaseVecT>
ModelPtr<BaseVecT> AsciiIO<BaseVecT>::read(
        string filename,
        const int &xPos, const int& yPos, const int& zPos,
        const int &rPos, const int& gPos, const int& bPos, const int &iPos)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().string());

    if ( extension != ".pts" && extension != ".3d" && extension != ".xyz" && extension != ".txt" )
    {
        cout << "»" << extension << "« is not a valid file extension." << endl;
        return ModelPtr<BaseVecT>();
    }
    // Count lines in file to estimate the number of present points
    int lines_in_file = countLines(filename);

    if ( lines_in_file < 2 )
    {
        cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
        return ModelPtr<BaseVecT>();
    }

    // Open file
    ifstream in;
    in.open(filename.c_str());

    // Read first to lines, ignore the first one
    char buffer[2048];
    in.getline(buffer, 2048);
    in.getline(buffer, 2048);

    // Get number of entries in test line and analize
    int num_columns  = AsciiIO::getEntriesInLine(filename);

    // Reopen file and read data
    in.close();
    in.open(filename.c_str());

    // Again skip first line
    in.getline(buffer, 2048);

    // Buffer related variables
    size_t numPoints = 0;

    floatArr points;
    ucharArr pointColors;
    floatArr pointIntensities;

    // Alloc memory for points
    numPoints = lines_in_file - 1;
    points = floatArr( new float[ numPoints * 3 ] );
    ModelPtr<BaseVecT> model(new Model<BaseVecT>);
    model->m_pointCloud = PointBuffer2Ptr( new PointBuffer2);

    // (Some) sanity checks for given paramters
    if(rPos > num_columns || gPos > num_columns || bPos > num_columns || iPos > num_columns)
    {
        cout << timestamp << "Error: At least one attribute index is largen than the number of columns" << endl;
        // Retrun empty model
        return model;
    }

    // Check for color and intensity
    bool has_color = (rPos > -1 && gPos > -1 && bPos > -1);
    bool has_intensity = (iPos > -1);

    // Alloc buffer memory for additional attributes
    if ( has_color )
    {
        pointColors = ucharArr( new uint8_t[ numPoints * 3 ] );
    }

    if ( has_intensity )
    {
        pointIntensities = floatArr( new float[ numPoints ] );
    }

    // Read data form file
    size_t c = 0;
    while (in.good() && c < numPoints)
    {
        float x, y, z, intensity, dummy;
        unsigned int r, g, b;

        for(int i = 0; i < num_columns; i++)
        {
            if(i == xPos)
            {
                in >> x;
            }
            else if(i == yPos)
            {
                in >> y;
            }
            else if(i == zPos)
            {
                in >> z;
            }
            else if(i == rPos)
            {
                in >> r;
            }
            else if(i == gPos)
            {
                in >> g;
            }
            else if(i == bPos)
            {
                in >> b;
            }
            else if(i == iPos)
            {
                in >> intensity;
            }
            else
            {
                in >> dummy;
            }

        }

        // Read according to determined format
        if(has_color)
        {
            pointColors[ c * 3     ] = (unsigned char) r;
            pointColors[ c * 3 + 1 ] = (unsigned char) g;
            pointColors[ c * 3 + 2 ] = (unsigned char) b;

        }

        if (has_intensity)
        {
            pointIntensities[c] = intensity;

        }

        points[ c * 3     ] = x;
        points[ c * 3 + 1 ] = y;
        points[ c * 3 + 2 ] = z;
        c++;
    }

    // Sanity check
    if(c != numPoints)
    {
        cout << timestamp << "Warning: Point count / line count mismatch: "
             << numPoints << " / " << c << endl;
    }

    // Assign buffers
    size_t numColors = 0;
    size_t numIntensities = 0;

    if(has_color)
    {
        numColors = numPoints;
    }

    if(has_intensity)
    {
        numIntensities = numPoints;
    }

    model->m_pointCloud->addFloatChannel(points, "points", numPoints, 3);
    model->m_pointCloud->addUCharChannel(pointColors, "colors", numColors, 3);
    model->m_pointCloud->addFloatChannel(pointIntensities, "intensities", numIntensities, 1);

    this->m_model = model;
    return model;
}

template<typename BaseVecT>
ModelPtr<BaseVecT> AsciiIO<BaseVecT>::read(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().string());

    if ( extension != ".pts" && extension != ".3d" && extension != ".xyz" && extension != ".txt" )
    {
        cout << "»" << extension << "« is not a valid file extension." << endl;
        return ModelPtr<BaseVecT>();
    }
    // Count lines in file to estimate the number of present points
    int lines_in_file = countLines(filename);

    if ( lines_in_file < 2 )
    {
        cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
        return ModelPtr<BaseVecT>();
    }
    // Open the given file. Skip the first line (as it may
    // contain meta data in some formats). Then try to guess
    // the additional data using some heuristics that apply for
    // most data formats: If 4 values per point are, given
    // the 4th value usually is a reflectence information.
    // Six entries suggest RGB information, seven entries
    // intensity and RGB.

    // Open file
    ifstream in;
    in.open(filename.c_str());

    // Read first to lines, ignore the first one
    char buffer[2048];
    in.getline(buffer, 2048);
    in.getline(buffer, 2048);

    // Get number of entries in test line and analize
    int num_attributes  = AsciiIO::getEntriesInLine(filename) - 3;
    bool has_color      = (num_attributes == 3) || (num_attributes == 4);
    bool has_intensity  = (num_attributes == 1) || (num_attributes == 4);

    if(has_color || has_intensity)
    {
        cout << timestamp << "Autodetected the following attributes" << endl;
        cout << timestamp << "Color:     " << has_color << endl;
        cout << timestamp << "Intensity: " << has_intensity << endl;

        if(has_color && has_intensity)
        {
            return read(filename, 0, 1, 2, 3, 4, 5, 6);
        }
        else if(has_color)
        {
            return read(filename, 0, 1, 2, 3, 4, 5);
        }
        else if(has_intensity)
        {
            if(num_attributes == 1)
            {
                return read(filename, 0, 1, 2, -1, -1, -1, 3);
            }
            else
            {
                return read(filename, 0, 1, 2, -1, -1, -1, 6);
            }
        }
    }
    return read(filename, 0, 1, 2);

}


template<typename BaseVecT>
void AsciiIO<BaseVecT>::save( std::string filename )
{

    if ( !this->m_model->m_pointCloud ) {
        std::cerr << "No point buffer available for output." << std::endl;
        return;
    }

    size_t   pointcount( 0 ), buf ( 0 );

    floatArr   points;
    ucharArr   pointColors;
    floatArr   pointIntensities;

    unsigned w;
    points = this->m_model->m_pointCloud->getFloatArray(pointcount, w, "points");
//    points = this->m_model->m_pointCloud->getIndexedPointArray( pointcount );


//    pointColors = this->m_model->m_pointCloud->getIndexedPointColorArray( buf );
    /* We need the same amount of color information and points. */
    if ( pointcount != buf )
    {
        pointColors.reset();
        std::cerr << "Amount of points and color information is"
	  " not equal. Color information won't be written" << std::endl;
    }

 //   pointIntensities = this->m_model->m_pointCloud->getPointIntensityArray( buf );
    /* We need the same amount of intensity values and points. */
    if ( pointcount != buf )
    {
        pointIntensities.reset();
        std::cerr << "Amount of points and intensity values are"
            " not equal. Intensity information will not be written." << std::endl;
    }


    /* Prepare file for writing. */
    std::ofstream out( filename.c_str() );

    if ( !out.is_open() ) {
        std::cerr << "Could not open file »" << filename
            << "« for output." << std::endl;
        return;
    }

    for ( size_t i(0); i < pointcount; i++ )
    {
        out << points[i * 3] << " "
            << points[i * 3 + 1] << " "
            << points[i * 3 + 2];
        if ( pointIntensities )
        {
            out << " " << pointIntensities[i];
        }
        if ( pointColors )
        {
            /* Bad behaviour of C++ output streams: We have to cast the uchars
             * to unsigned integers. */
            out << " " << (unsigned int) pointColors[i * 3]
                << " " << (unsigned int) pointColors[i * 3 + 1]
                << " " << (unsigned int) pointColors[i * 3 + 2];
        }
        out << std::endl;
    }

    out.close();


}

template<typename BaseVecT>
size_t AsciiIO<BaseVecT>::countLines(string filename)
{
    // Open file for reading
    ifstream in(filename.c_str());

    // Count lines in file
    size_t c = 0;
    char line[2048];
    while(in.good())
    {
        in.getline(line, 1024);
        c++;
    }
    in.close();
    return c;
}

template<typename BaseVecT>
int AsciiIO<BaseVecT>::getEntriesInLine(string filename)
{

    ifstream in(filename.c_str());

    // Get first line from file and skip it (possibly metadata)
    char line[1024];
    in.getline(line, 1024);

    // Get second line -> hopefully point data
    in.getline(line, 1024);

    in.close();

    // Get number of blanks
    int c = 0;
    char* pch = strtok(line, " ");
    while(pch){
        c++;
        pch = strtok(NULL, " ");
    }

    return c;
}


} // namespace lvr
