/**
 * Copyright (c) 2022, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/**
 * @file       B3dmIO.hpp
 * @brief      I/O support for b3dm files.
 * @details    I/O support for b3dm files: Reading and writing meshes, including
 *             color information, textures and normals.
 * @author     Malte Hillmann <mhillmann@uni-osnabrueck.de>
 * @date       13.06.2022
 */


#pragma once

#ifdef LVR2_USE_3DTILES

#include "lvr2/io/modelio/ModelIOBase.hpp"

namespace lvr2
{

class B3dmIO : public ModelIOBase
{
public:
    /**
     * @brief Constructor.
     **/
    B3dmIO()
    {}

    virtual ~B3dmIO() = default;


    /**
     * @brief Save B3dm with previously specified data.
     *
     * @param filename Filename of the output file.
     **/
    void save(std::string filename) override;

    /**
     * @brief Save B3dm with the given data.
     * 
     * @param model    The model to save.
     * @param filename Filename of the output file.
     */
    void save(ModelPtr model, std::string filename) override
    {
        setModel(model);
        save(filename);
    }

    /**
     * @brief Save B3dm with previously specified data, compressed.
     *
     * @param filename Filename of the output file.
     */
    void saveCompressed(const std::string& filename);

    /**
     * @brief Save B3dm with the given data, compressed.
     * 
     * @param model    The model to save.
     * @param filename Filename of the output file.
     */
    void saveCompressed(ModelPtr model, std::string filename)
    {
        setModel(model);
        saveCompressed(filename);
    }

    /**
     * @brief Read specified B3dm file.
     *
     * @param filename           Filename of file to read.
     **/
    ModelPtr read(std::string filename) override;
};

} // namespace lvr2

#endif // LVR2_USE_3DTILES
