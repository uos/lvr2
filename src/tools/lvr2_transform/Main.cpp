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
 * main.cpp
 *
 * @date 2012-08-23
 * @author Christian Wansart <cwansart@uos.de>
 * @author Thomas Wiemann
 */

#include "Options.hpp"
#include "lvr2/geometry/BaseVector.hpp"
#include "lvr2/geometry/Matrix4.hpp"
#include "lvr2/io/Model.hpp"
#include "lvr2/io/ModelFactory.hpp"
#include "lvr2/io/Timestamp.hpp"
#include <iostream>
#include <cmath>

using namespace lvr2;
using std::cout;
using std::endl;

using Vec = BaseVector<float>;


int main(int argc, char **argv)
{
  try {
    bool did_anything = false;
    float x, y, z, r1, r2, r3;
    Matrix4<Vec> mat;
    size_t num;

    // get options
    transform::Options options(argc, argv);

    // quit if usage was printed
    if(options.printUsage())
      return 0;

    // load model via ModelFactory
    ModelPtr model = ModelFactory::readModel(options.getInputFile());
    
    if(!model)
    {
      cout << timestamp << "IO Error: Unable to parse " << options.getInputFile() << endl;
      exit(-1);
    }

    if(options.anyTransformFile())
    {
      // Check if transformFile was given, check if it's a pose or frames file and
      // read in accordingly.
      ifstream in(options.getTransformFile().c_str());
      if(!in.good()){
        cout << timestamp << "Warning: Load transform file: File not found or corrupted." << endl;
        return -1;
      }

      if(options.getTransformFile().substr(options.getTransformFile().length()-5) == ".pose")
      {
        cout << timestamp << "Reading from .pose file" << endl;
        in >> x >> y >> z >> r1 >> r2 >> r3;

        r1 = r1 * 0.0174532925f;
        r2 = r2 * 0.0174532925f;
        r3 = r3 * 0.0174532925f;

        mat = Matrix4<Vec>(Vec(x, y, z), Vec(r1, r2, r3));
      }
      else //expect frames file instead
      {
        cout << timestamp << "Reading from .frames file" << endl;
        float t[17];
        ifstream in(options.getTransformFile().c_str());
        while(in.good())
        {
          in >>  t[0] >>  t[1] >>  t[2] >>  t[3] 
             >>  t[4] >>  t[5] >>  t[6] >>  t[7]
             >>  t[8] >>  t[9] >> t[10] >> t[11]
             >> t[12] >> t[13] >> t[14] >> t[15]
             >> t[16]; // we don't need this value but we need to skip it
        }

        for(int i = 0; i < 16; ++i)
          mat.set(i, t[i]);
      }
    }
    else // read from s, r or t
    {
      x = y = z = r1 = r2 = r3 = 0.0f;

      if(options.anyRotation())
      {
        if(options.anyRotationX())
          r1 = options.getRotationX();

        if(options.anyRotationY())
          r2 = options.getRotationY();

        if(options.anyRotationZ())
          r3 = options.getRotationZ();
      }

      if(options.anyTranslation())
      {
        if(options.anyTranslationX())
          x = options.getTranslationX();

        if(options.anyTranslationY())
          y = options.getTranslationY();

        if(options.anyTranslationZ())
          z = options.getTranslationZ();
      }

      // To radians
      r1 = r1 * 0.0174532925f;
      r2 = r2 * 0.0174532925f;
      r3 = r3 * 0.0174532925f;

      mat = Matrix4<Vec>(Vec(x, y, z), Vec(r1, r2, r3));
    }

    // Get point buffer
    if(model->m_pointCloud)
    {
      PointBufferPtr p_buffer = model->m_pointCloud;

      cout << timestamp << "Using points" << endl;
      did_anything = true;
      FloatChannelOptional points = p_buffer->getFloatChannel("points");
       
      cout << mat;
      for(size_t i = 0; i < points->numElements(); i++)
      {
        Vec v((*points)[i][0], (*points)[i][1], (*points)[i][2]);
        v = mat * v;
        v.x = options.anyScaleX() ? v[0] * options.getScaleX() : v[0];
        v.y = options.anyScaleX() ? v[1] * options.getScaleX() : v[1];
        v.z = options.anyScaleX() ? v[2] * options.getScaleX() : v[2];
        (*points)[i] = v;
      }
    }

    // Get mesh buffer
    if(model->m_mesh)
    {
      MeshBufferPtr m_buffer = model->m_mesh;

      cout << timestamp << "Using meshes" << endl;
      did_anything = true;
      FloatChannelOptional points = m_buffer->getFloatChannel("vertices");

      for(size_t i = 0; i < points->numElements(); i++)
      {
        Vec v((*points)[i][0], (*points)[i][1], (*points)[i][2]);
        v = mat * v;
        v.x = options.anyScaleX() ? v[0] * options.getScaleX() : v[0];
        v.y = options.anyScaleX() ? v[1] * options.getScaleX() : v[1];
        v.z = options.anyScaleX() ? v[2] * options.getScaleX() : v[2];
        (*points)[i] = v;
      }
    }

    if(!did_anything)
    {
      std::cerr << timestamp << "I had nothing to do. Terminating now..." << std::endl;
      return 0;
    }
    else
    {
      cout << timestamp << "Finished. Program end." << endl;
    }

    ModelFactory::saveModel(model, options.getOutputFile());
  } catch(...) {
    cout << "something went wrong..." << endl;
  }
}
