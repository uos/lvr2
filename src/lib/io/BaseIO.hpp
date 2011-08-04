/**
 * BaseIO.hpp
 *
 *  @date 03.08.2011
 *  @author Thomas Wiemann
 */

#ifndef BASEIO_HPP_
#define BASEIO_HPP_

#include <cstdlib>
#include <string>

using std::string;

/**
 * @brief Interface specification for low-level io. All read
 *        elements are stored in linear arrays.
 */
class BaseIO
{
public:
    BaseIO() {}

    /**
     * Parse the given file and load supported elements.
     *
     * @param filename  The file to read
     */
    virtual void read(string filename) = 0;

    /**
     * Save the loaded elements to the given file.
     *
     * @param filename
     */
    virtual void save(string filename) = 0;

};

#endif /* BASEIO_HPP_ */
