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

#include "Progress.hpp"
#include "Timestamp.hpp"

namespace lssr
{

AsciiIO::AsciiIO()
{
    m_points = 0;
    m_numPoints = 0;
}

void AsciiIO::read(string filename)
{
    float** points = 0;

    // Check extension
    boost::filesystem::path selectedFile(filename);
    string extension(selectedFile.extension().c_str());

    if(extension == ".pts" || extension == ".3d" || extension == ".xyz" || extension == ".txt")
    {
        // Read the number of entries in the given file.
        // The fist three entries are considered to contain
        // the point definitions. The remaining columns are
        // skipped when reading the file.
        int skip = AsciiIO::getEntriesInLine(filename) - 3;

        if(skip < 0)
        {
            cout << timestamp << " Error: ASCII IO: File '"<<
                    filename  << "' contains less than three entries per line." << endl;
            return;
        }

        // Count points in given file
        size_t c = countLines(filename);

        // Alloc memory for points
        m_points = new float*[c];
        for(size_t i = 0; i < c; i++)  m_points[i] = new float[3];

        // Setup info output
        string comment = timestamp.getElapsedTime() + "Reading file " + filename;
        ProgressBar progress(c, comment);

        // Read point data
        ifstream in(filename.c_str());

        c = 0;
        float x, y, z, dummy;
        while(in.good() ){
            //in >> points[c][0] >> points[c][1] >> points[c][2];

            in >> x >> y >> z;
            m_points[c][0] = x;
            m_points[c][1] = y;
            m_points[c][2] = z;

            for(int i = 0; i < skip; i++)
            {
                in >> dummy;
            }
            c++;
            ++progress;
        }
        m_numPoints = c;
        cout << endl;
        cout << timestamp << "Read " << c << " data points" << endl;
        in.close();
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
