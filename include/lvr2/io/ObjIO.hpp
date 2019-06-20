/**
 * Copyright (c) 2018, University Osnabrück
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

/*
 * ObjIO.hpp
 *
 *  @date 07.11.2011
 *  @author Florian Otte (fotte@uos.de)
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Denis Meyer (denmeyer@uos.de)
 */

#ifndef LVR2_OBJIO_HPP_
#define LVR2_OBJIO_HPP_
#include "lvr2/io/BaseIO.hpp"
#include <fstream>
#include <set>
#include <map>

using namespace std;

namespace lvr2
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
     *
     * @param filename  The file to read.
     */
    ModelPtr read( string filename );

    /**
     * @brief     Writes the mesh to an obj file.
     *
     * @param  model     The model containing all mesh data
     * @param  filename  The file name to use
     */
    void save( string filename );


private:

    void parseMtlFile(map<string, int>& matNames,
            vector<Material>& materials,
            vector<Texture>& textures,
            string mtlname);

};

} // namespace lvr2

#endif /* OBJIO_H_ */
