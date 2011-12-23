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

#include "Model.hpp"

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
            virtual ModelPtr read(string filename ) = 0;


            /**
             * \brief Save the loaded elements to the given file.
             *
             * @param filename Filename of the file to write.
             */
            virtual void save( string filename ) = 0;


            /**
             * \brief Set the model and save the loaded elements to the given
             *        file.
             *
             * @param filename Filename of the file to write.
             */
            virtual void save( ModelPtr model, string filename )
            {
                m_model = model;
                save( filename );
            }

            virtual void setModel( ModelPtr m )
            {
                m_model = m;
            }

            virtual ModelPtr getModel() 
            {
                return m_model;
            }

        protected:
            ModelPtr m_model;

    };

}

#endif /* BASEIO_HPP_ */
