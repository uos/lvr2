/**
 * @file       BaseIO.hpp
 * @brief      Base interface for all I/O related classes.
 * @details    This file introduces a pure virtual class specifying some basic
 *             methods which must be implemented by all I/O classes in the lssr
 *             toolkit.
 * @author     Thomas Wiemann (twiemann), twiemann@uos.de
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version    110929
 * @date       Created:       2011-08-03
 * @date       Last modified: 2011-09-29 20:40:14
 */

#ifndef BASEIO_HPP_
#define BASEIO_HPP_

#include <cstdlib>
#include <string>

using std::string;

namespace lssr
{

    /**
     * @brief Interface specification for low-level io. All read
     *        elements are stored in linear arrays.
     */
    class BaseIO
    {
        public:
            BaseIO() {}

            /**
             * \brief Parse the given file and load supported elements.
             *
             * @param filename  The file to read.
             */
            virtual void read(string filename) = 0;

            /**
             * \brief Save the loaded elements to the given file.
             *
             * @param filename Filename of the file to write.
             */
            virtual void save(string filename) = 0;

    };

}

#endif /* BASEIO_HPP_ */
