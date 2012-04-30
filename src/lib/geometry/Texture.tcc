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
 * Texture.tcc
 *
 *  @date 08.09.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 */

namespace lssr {

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::Texture()
{
	this->m_data   = 0;

	if(m_data == 0)
	{
		this->m_width = 1;
		this->m_height = 1;
		m_data = new PPMIO::ColorT*[this->m_height];
		for (int y = 0; y < this->m_height; y++)
		{
			m_data[y] = new PPMIO::ColorT[this->m_width];
			memset(m_data[y], 200, this->m_width * sizeof(PPMIO::ColorT));
		}
	}
}


template<typename VertexT, typename NormalT>
void Texture<VertexT, NormalT>::save()
{
	PPMIO ppm;
	ppm.setDataArray(this->m_data,this->m_width, this->m_height);
	ppm.write("texture_"+boost::lexical_cast<std::string>(0)+".ppm");
}

template<typename VertexT, typename NormalT>
Texture<VertexT, NormalT>::~Texture() {
	for(int y = 0; y < m_height; y++)
	{
		delete m_data[y];
	}
	delete m_data;
}

}
