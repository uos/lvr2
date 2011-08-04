/*
 * AsciiIO.cpp
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#include <fstream>
#include <string.h>
using std::ifstream;

#include <boost/filesystem.hpp>

#include "AsciiIO.hpp"
#include "Progress.hpp"
#include "Timestamp.hpp"

namespace lssr
{

AsciiIO::AsciiIO()
{
    m_points = 0;
    m_colors = 0;
    m_intensities = 0;
    m_numPoints = 0;
}

void AsciiIO::read(string filename)
{
    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().c_str());

    if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        // Count lines in file to estimate the number of present points
        int lines_in_file = countLines(filename);

        if(lines_in_file > 2)
        {

            // Open the given file. Skip the first line (as it may
            // contain meta data in some formats). Then try to guess
            // the additional data using some heuriscs that apply for
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

            // Get number of entries in test line and analiuze
            int num_attributes = AsciiIO::getEntriesInLine(filename) - 3;
            bool has_color = (num_attributes == 3) || (num_attributes == 4);
            bool has_intensity = (num_attributes == 1) || (num_attributes == 4);

            // Reopen file and read data
            in.close();
            in.open(filename.c_str());

            // Again skip first line
            in.getline(buffer, 2048);

            // Alloc memory for points
            m_numPoints = lines_in_file -1;
            m_points = new float*[m_numPoints];
            for(int i = 0; i < m_numPoints; i++) m_points[i] = new float[3];

            // Alloc buffer memory for additional attributes
            if(has_color)
            {
                m_colors = new unsigned char*[m_numPoints];
                for(int i = 0; i < m_numPoints; i++) m_colors[i] = new unsigned char[3];
            }

            if(has_intensity)
            {
                m_intensities = new float[m_numPoints];
            }

            // Read data form file
            size_t c = 0;
            while(in.good())
            {
                float x, y, z, i, dummy;
                unsigned char r, g, b;

                // Read according to determined format
                if(has_intensity && !has_color)
                {
                    in >> x >> y >> z >> i;
                    m_intensities[c] = i;
                }
                else if(has_intensity && has_color)
                {
                    in >> x >> y >> z >> i >> r >> g >> b;
                    m_intensities[c] = i;
                    m_colors[c][0] = r;
                    m_colors[c][1] = g;
                    m_colors[c][2] = b;
                }
                else if(has_color && !has_intensity)
                {
                    in >> x >> y >> z >> r >> g >> b;
                    m_colors[c][0] = r;
                    m_colors[c][1] = g;
                    m_colors[c][2] = b;
                }
                else
                {
                    in >> x >> y >> z;
                    for(int n_dummys = 0; n_dummys < num_attributes; n_dummys++) in >> dummy;
                }
                m_points[c][0] = x;
                m_points[c][1] = y;
                m_points[c][2] = z;
                c++;
            }


        }
        else
        {
            cout << timestamp << "AsciiIO: Too few lines in file (has to be > 2)." << endl;
        }

    }

}


size_t AsciiIO::countLines(string filename)
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


int AsciiIO::getEntriesInLine(string filename)
{

    ifstream in(filename.c_str());

    // Get first line from file and skip it (possibly metadata)
    char first_line[1024];
    in.getline(first_line, 1024);

    // Get second line -> hopefully point data
    char second_line[1024];
    in.getline(second_line, 1024);

    in.close();

    // Get number of blanks
    int c = 0;
    char* pch = strtok(second_line, " ");
    while(pch != NULL){
        c++;
        pch = strtok(NULL, " ");
    }

    in.close();

    return c;
}


} // namespace lssr
