/*
 * AsciiIO.h
 *
 *  Created on: 09.03.2011
 *      Author: Thomas Wiemann
 */

#ifndef ASCIIIO_H_
#define ASCIIIO_H_

#include "BaseIO.hpp"
#include "PointLoader.hpp"

namespace lssr
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *        .3d and .pts are supported.
 */
class AsciiIO : public BaseIO, public PointLoader
{
public:

    AsciiIO() {};

    /**
     * @brief Reads the given file and stores point and normal
     *        information in the given parameters
     *
     * @param filename      The file to read
     */
    virtual void read(string filename);

    /// TODO: Coordinate mapping for ascii files
    static size_t countLines(string filename);

    /**
     * @brief Helper method. Returns the number of columns in the
     *        given file.
     */
    static int getEntriesInLine(string filename);

    virtual void save(string filename)
    {

    }

};


} // namespace lssr


#endif /* ASCIIIO_H_ */
