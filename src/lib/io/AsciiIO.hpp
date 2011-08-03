/*
 * AsciiIO.h
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef ASCIIIO_H_
#define ASCIIIO_H_

namespace lssr
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *        .3d and .pts are supported.
 */
template<typename T>
class AsciiIO
{
public:

    /**
     * @brief Reads the given file and stores point and normal
     *        information in the given parameters
     *
     * @param filename      The file to read
     * @param count         The number of elements read
     * @return The read point cloud data or a zero pointer of reading failed.
     */
    static T** read(string filename, size_t &count);

    /// TODO: Coordinate mapping for ascii files
    static size_t countLines(string filename);

    /**
     * @brief Helper method. Returns the number of columns in the
     *        given file.
     */
    static int getEntriesInLine(string filename);
};


} // namespace lssr


#include "AsciiIO.tcc"

#endif /* ASCIIIO_H_ */
