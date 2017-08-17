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
 * TextureToken.tcc
 *
 *  @date 03.05.2012
 *  @author Kristin Schmidt (krschmidt@uos.de)
 *  @author Jan Philipp Vogtherr (jvogtherr@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

namespace lvr2 {

template<typename BaseVecT>
TextureToken<BaseVecT>::TextureToken(Texture* t, int index)
{
    this->m_texture = t;
    this->m_textureIndex = index;
}

template<typename BaseVecT>
TextureToken<BaseVecT>::TextureToken(
    BaseVecT vec1,
    BaseVecT vec2,
    BaseVecT supportVec,
    float minDistA,
    float minDistB,
    Texture* t,
    int index,
    float texelSize
) :
    m_vec1(vec1),
    m_vec2(vec2),
    m_supportVector(supportVec),
    m_minDistA(minDistA),
    m_minDistB(minDistB),
    m_texture(t),
    m_textureIndex(index),
    m_texelSize(texelSize)
{ }

template<typename BaseVecT>
TexCoords TextureToken<BaseVecT>::textureCoords(BaseVecT v) const
{
    // TODO: stimmt die berechnung?
    BaseVecT w =  v - ((m_vec1 * m_minDistA) + (m_vec2 * m_minDistB) + m_supportVector);
    float x = (m_vec1 * (w.dot(m_vec1))).length() / m_texelSize / m_texture->m_width;
    float y = (m_vec2 * (w.dot(m_vec2))).length() / m_texelSize / m_texture->m_height;

    return TexCoords(x,y);

}

}
