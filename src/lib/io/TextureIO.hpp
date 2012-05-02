/* Copyright (C) 2012 Uni Osnabrück
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
 * @file       TextureIO.hpp
 * @brief      I/O support for Texture files
 * @details    I/O support for Texture files: Reading and writing textures
 *             from or to texture packages
 * @author     Kim Oliver Rinnewitz (krinnewitz), krinnewitz@uos.de
 * @date       Created:       2012-04-22 13:11:28
 */


#ifndef __TEXTURE_IO_H__
#define __TEXTURE_IO_H__

#include "geometry/Texture.hpp"

#include <stdint.h>
#include <cstdio>
#include <vector>
#include <fstream>

namespace lssr
{

/**
 *	TODO: Texture package format description
 *
 *
 *
 *
 *
**/

/**
 * \class TextureIO TextureIO.hpp "io/TextureIO.hpp"
 * \brief A class to read from or write to texture packages
 *
 */

class TextureIO
{
    public:
        /**
         * \brief Constructor.
         **/
        TextureIO(string filename);
        
	virtual ~TextureIO() {/*TODO*/}

	/**
	 * \brief Add the given texture to the texture package
	 * \param 	t	The texture to add
	 * \return 		The index of the texture in the texture package	
	**/
	virtual size_t add(Texture* t);
		
	/**
	 * \brief 	Remove the texture with the given index from the texture package
	 * \param	index	The index of the texture to remove
	**/
	virtual void remove (size_t index);

	/**
	 * \brief 	Update the texture with the given index in the texture package
	 * \param	index	The index of the texture to update
	 * \param	t	The new texture to replace the old one with
	**/
	virtual void update (size_t index, Texture* t);

	/**
	 * \brief 	Get the texture with the given index from the texture package
	 * \param	index	The index of the texture to get
	 * \return	The read texture 
	**/
	virtual Texture* get(size_t index);
	
	/**
	 * \brief 	Get the subsequent texture from the texture package
	 * \return	The read texture 
	**/
	virtual Texture* getNext();

	/**
	 * \brief (re-)write the file
	 *
	**/
	virtual void write();


	/**
	 * \brief resets the internal index
	 *
 	**/
	virtual void resetIndex(){m_currentIndex = 0;};	

	std::vector<Texture*> 	m_textures;
	string 			m_filename;
    private:
	size_t 			m_currentIndex;


};

}
#include "TextureIO.cpp"
#endif
