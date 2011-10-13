/**
 *
 * @file      AsciiIO.hpp
 * @brief     Read and write pointclouds from .pts and .3d files.
 * @details   Read and write pointclouds from .pts and .3d files.
 * 
 * @author    Thomas Wiemann (twiemann), twiemann@uos.de, Universität Osnabrück
 * @author    Lars Kiesow (lkiesow), lkiesow@uos.de, Universität Osnabrück
 * @version   111001
 * @date      Created:       2011-03-09
 * @date      Last modified: 2011-10-01 19:49:24
 *
 **/

#ifndef ASCIIIO_H_
#define ASCIIIO_H_

#include "BaseIO.hpp"
#include "PointLoader.hpp"


#ifdef __GNUC__
#define WARN(msg) __attribute__ ((warning (msg)))
#else
#define WARN(msg)  
#endif


namespace lssr
{

/**
 * @brief A import / export class for point cloud data in plain
 *        text formats. Currently the file extensions .xyz, .txt,
 *        .3d and .pts are supported.
 * @todo  Implement save method
 */
class AsciiIO : public BaseIO, public PointLoader
{
public:

    /**
     * \brief Default constructor.
     **/
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

    /**
     * \brief   Save the loaded elements to the given file.
     * \todo    Implement this.
     * \warning This method is not yet implemented. It has no functionality.
     *
     * \param filename Filename of the file to write.
     **/
    virtual void save(string filename) WARN( "Warning: This method is not yet implemented." )
    {

    }

};


} // namespace lssr


#endif /* ASCIIIO_H_ */
