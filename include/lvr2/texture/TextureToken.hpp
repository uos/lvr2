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
 * TextureToken.hpp
 *
 *  @date 03.05.2012
 *  @author Kristin Schmidt (krschmidt@uos.de)
 *  @author Jan Philipp Vogtherr (jvogtherr@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 */

#ifndef LVR2_TEXTURE_TEXTURETOKEN_HPP_
#define LVR2_TEXTURE_TEXTURETOKEN_HPP_

#include <lvr2/algorithm/Materializer.hpp>
#include <lvr2/texture/ClusterTexCoordMapping.hpp>
#include <lvr2/texture/Texture.hpp>

namespace lvr2 {

/**
 * @brief   This class allows to calculate texture coordinates for the associated texture.
 */
template <typename BaseVecT>
class TextureToken {
public:

    /**
     * @brief   Constructor.
     *
     * @param   v1  The first vector of the texture coordinate system
     *
     * @param   v2  The second vector of the texture coordinate system
     *
     * @param   p   A point in the texture plane
     *
     * @param   a_min   This value defines the bounding box of the texture
     *
     * @param   b_min   This value defines the bounding box of the texture
     *
     * @param   t   The associated texture
     *
     * @param   index   The index of the texture in the texture package
     *
    **/
    TextureToken<BaseVecT>(
        BaseVecT vec1,
        BaseVecT vec2,
        BaseVecT supportVec,
        float minDistA,
        float minDistB,
        Texture* t,
        int index,
        float texelSize
    );

    /**
     * @brief Constructor
     * @param t Texture
     * @param index Texture index
     */
    TextureToken<BaseVecT>(Texture* t, int index);

    /**
     * @brief   computes texture coordinates corresponding to the given Vertex
     *
     * @param   v   the vertex to generate texture coordinates for
     *
     * @return  returns texture coordinates
     */
    TexCoords textureCoords(BaseVecT v) const;

    /**
     * @brief   Destructor.
     *
         */
    ~TextureToken(){/*delete m_texture;*/};

    ///The associated texture
    Texture* m_texture;

    ///The coordinate system of the texture plane
    BaseVecT m_vec1, m_vec2;

    ///A point in the texture plane
    BaseVecT m_supportVector;

    ///The bounding box of the texture plane
    float m_minDistA, m_minDistB;

    ///index of the texture in the texture pack
    size_t m_textureIndex;

    ///Texel size
    float m_texelSize;
};

}

#include <lvr2/texture/TextureToken.tcc>

#endif /* LVR2_TEXTURE_TEXTURETOKEN_HPP_ */
