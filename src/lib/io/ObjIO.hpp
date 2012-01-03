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
 * ObjIO.hpp
 *
 *  @date 07.11.2011
 *  @author Florian Otte (fotte@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

#ifndef OBJIO_HPP_
#define OBJIO_HPP_
#include "BaseIO.hpp"
#include <fstream>
#include <set>

namespace lssr
{

/**
 * @brief A basic implementation of the obj file format.
 */
class ObjIO : public BaseIO
{
    public:

        /**
         * @brief Constructor.
         **/
        ObjIO()
        {
            m_model.reset();
        }

        ~ObjIO() { };

        /**
         * \brief   Parse the given file and load supported elements.
         * \warning This function is yet to be implemented.
         * \todo    Implement this function propperly.
         *
         * @param filename  The file to read.
         */
        ModelPtr read( string filename )
        {
            ModelPtr m( new Model );
            return m;
        };

        /**
         * @brief     Writes the mesh to an obj file.
         *
         * @param  model     The model containing all mesh data
         * @param  filename  The file name to use
         */
        void save( string filename );


};

}

#endif /* OBJIO_H_ */
