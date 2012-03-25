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
 * @file       lssr::BaseIO.cpp
 * @brief      Base interface for all I/O related classes.
 * @details    This file introduces a pure virtual class specifying some basic
 *             methods which must be implemented by all I/O classes in the lssr
 *             toolkit.
 * @author     Lars Kiesow (lkiesow), lkiesow@uos.de
 * @version    120103
 * @date       Created:       2012-01-03 12:34:25
 * @date       Last modified: 2012-01-03 12:34:28
 */

#include "BaseIO.hpp"



void lssr::BaseIO::save( lssr::ModelPtr m, string filename)
{
	m_model = m;
    save( filename);
}


void lssr::BaseIO::setModel( ModelPtr m )
{
    m_model = m;
}


lssr::ModelPtr lssr::BaseIO::getModel() 
{
    return m_model;
}


