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
// TextureToken<BaseVecT>::TextureToken(BoundingRectangle<BaseVecT> boundingRect, Texture* t, int index)
TextureToken<BaseVecT>::TextureToken(Texture* t, int index)
{
    // this->v1    = v1;
    // this->v2    = v2;
    // this->p     = p;
    // this->a_min = a_min;
    // this->b_min = b_min;
    // this->m_boundingRect = boundingRect;
    this->m_texture = t;
    this->m_textureIndex = index;
    // this->m_transformationMatrix[0] = 1;
    // this->m_transformationMatrix[1] = 0;
    // this->m_transformationMatrix[2] = 0;
    // this->m_transformationMatrix[3] = 0;
    // this->m_transformationMatrix[4] = 1;
    // this->m_transformationMatrix[5] = 0;
    // this->m_mirrored        = 0;
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
TexCoords TextureToken<BaseVecT>::textureCoords(Vector<BaseVecT> v)
{
    Vector<BaseVecT> w =  v - ((m_vec1 * m_minDistA) + (m_vec2 * m_minDistB) + m_supportVector);
    float x = (m_vec1 * (w * m_vec1)).length() / m_texelSize / m_texture->m_width;
    float y = (m_vec2 * (w * m_vec2)).length() / m_texelSize / m_texture->m_height;

    // //Mirror y or x according to detected mirroring
    // if (this->m_mirrored == 1)
    // {
    //     y = m_texture->m_height - y;
    // }
    // else if (this->m_mirrored == 2)
    // {
    //     x = m_texture->m_width - x;
    // }

    // double tmp_x = x;
    // double tmp_y = y;

    // if (!this->m_texture->m_isPattern)
    // {
    //     //apply transformation
    //     tmp_x = m_transformationMatrix[0] * x + m_transformationMatrix[1] * y + m_transformationMatrix[2];
    //     tmp_y = m_transformationMatrix[3] * x + m_transformationMatrix[4] * y + m_transformationMatrix[5];
    // }

    return TexCoords(x,y);

}

}
