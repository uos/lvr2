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

/*
 * main.cpp
 *
 * @date 2012-08-23
 * @author Christian Wansart <cwansart@uos.de>
 * @author Thomas Wiemann
 */

#include "Options.hpp"
#include "geometry/Matrix4.hpp"
#include "geometry/Vertex.hpp"
#include "io/Model.hpp"
#include "io/ModelFactory.hpp"
#include "io/Timestamp.hpp"
#include <iostream>
#include <cmath>

using namespace lvr;
using std::cout;
using std::endl;

typedef Matrix4<float> Matrix4f;
typedef Vertex<float> Vertex3f;

int main(int argc, char **argv)
{
  try {
    bool did_anything = false;
    float x, y, z, r1, r2, r3;
    Matrix4f mat;
    size_t num;

    // get options
    transform::Options options(argc, argv);

    // quit if usage was printed
    if(options.printUsage())
      return 0;

    // load model via ModelFactory
    ModelFactory factory;
    ModelPtr model = factory.readModel(options.getInputFile());
    
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

        r1 = r1 * 0.0174532925;
        r2 = r2 * 0.0174532925;
        r3 = r3 * 0.0174532925;

        mat = Matrix4f(Vertex3f(x, y, z), Vertex3f(r1, r2, r3));
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
             >> t[17]; // we don't need this value but we need to skip it
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
      r1 = r1 * 0.0174532925;
      r2 = r2 * 0.0174532925;
      r3 = r3 * 0.0174532925;

      mat = Matrix4f(Vertex3f(x, y, z), Vertex3f(r1, r2, r3));
    }

    // Get point buffer
    if(model->m_pointCloud)
    {
      PointBufferPtr p_buffer = model->m_pointCloud;

      cout << timestamp << "Using points" << endl;
      did_anything = true;
      coord3fArr points = p_buffer->getIndexedPointArray(num);
      cout << mat;
      for(size_t i = 0; i < num; i++)
      {
        Vertex<float> v(points[i][0], points[i][1], points[i][2]);
        v = mat * v;
        points[i][0] = options.anyScaleX() ? v[0] * options.getScaleX() : v[0];
        points[i][1] = options.anyScaleY() ? v[1] * options.getScaleY() : v[1];
        points[i][2] = options.anyScaleZ() ? v[2] * options.getScaleZ() : v[2];
      }
      p_buffer->setIndexedPointArray(points, num);
    }

    // Get mesh buffer
    if(model->m_mesh)
    {
      MeshBufferPtr m_buffer = model->m_mesh;

      cout << timestamp << "Using meshes" << endl;
      did_anything = true;
      coord3fArr points = m_buffer->getIndexedVertexArray(num);

      for(size_t i = 0; i < num; i++)
      {
        Vertex<float> v(points[i][0], points[i][1], points[i][2]);
        v = mat * v;
        points[i][0] = options.anyScaleX() ? v[0] * options.getScaleX() : v[0];
        points[i][1] = options.anyScaleY() ? v[1] * options.getScaleY() : v[1];
        points[i][2] = options.anyScaleZ() ? v[2] * options.getScaleZ() : v[2];
      }
      m_buffer->setIndexedVertexArray(points, num);
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

    factory.saveModel(model, options.getOutputFile());
  } catch(...) {
    cout << "something went wrong..." << endl;
  }
}


