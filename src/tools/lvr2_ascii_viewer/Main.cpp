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

/**
 * calcNormalsCuda.h
 *
 * @author Alexander Mock
 */

#include <boost/filesystem.hpp>

#include "lvr2/io/ModelFactory.hpp"
#include "Options.hpp"

#include "ascii_viewer/AsciiRenderer.hpp"

#include <chrono>
#include <stdio.h>
#include <stdlib.h>

#include <ncursesw/ncurses.h>
#include <memory>


#include "ascii_viewer/CursesHelper.hpp"

using namespace lvr2;


int main(int argc, char** argv)
{
    ascii_viewer::Options opt(argc, argv);
    ModelPtr m = ModelFactory::readModel(opt.inputFile());

    if(m->m_mesh)
    {
        AsciiRendererPtr renderer;
        renderer.reset(new AsciiRenderer(m->m_mesh) );
        bool finish = false;
        while(!finish)
        {
            int key;
            while( (key = getch() ) != ERR )
            {
                if( key == 27 )
                {
                    finish = true;
                    break;
                }
                renderer->processKey(key);
            }
            renderer->render();
        }
        renderer.reset();
    } else {
        // print error
        std::cout << opt.inputFile() << " is not a triangle mesh!" << std::endl;
    }
    
    return 0;
}


