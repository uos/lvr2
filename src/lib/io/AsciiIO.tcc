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

template<typename T>
T** AsciiIO<T>::read(string filename, size_t &count)
{
    T** points = 0;

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
            return 0;
        }

        // Count points in given file
        size_t c = countLines(filename);

        // Alloc memory for points
        points = new T*[c];
        for(size_t i = 0; i < c; i++) points[i] = new T[3];

        // Setup info output
        string comment = timestamp.getElapsedTime() + "Reading file " + filename;
        ProgressBar progress(c, comment);

        // Read point data
        ifstream in(filename.c_str());

        c = 0;
        T x, y, z, dummy;
        while(in.good() ){
            //in >> points[c][0] >> points[c][1] >> points[c][2];
            in >> x >> y >> z;
            points[c][0] = x;
            points[c][1] = y;
            points[c][2] = z;

            for(int i = 0; i < skip; i++)
            {
                in >> dummy;
            }
            c++;
            ++progress;
        }
        cout << endl;
        cout << timestamp << "Read " << c << " data points" << endl;
        in.close();
    }
    return points;
}

template<typename T>
size_t AsciiIO<T>::countLines(string filename)
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

template<typename T>
int AsciiIO<T>::getEntriesInLine(string filename)
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
