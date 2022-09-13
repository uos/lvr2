//
// Created by Nikolas on 13.09.22.
//

#include "lvr2/io/modelio/RdbxIO.hpp"

#include <iostream>

namespace lvr2
{
    RdbxIO::RdbxIO()
    {

    }
    RdbxIO::~RdbxIO()
    {

    }


    void RdbxIO::save(string filename)
    {
        if ( !this->m_model->m_pointCloud ) {
            std::cerr << "No point buffer available for output." << std::endl;
            return;
        }
    }


    void RdbxIO::save(ModelPtr model, string filename)
    {
        m_model = model;
        save(filename);
    }

    // ID -> x,y,z -> amplitude -> reflectance -> r,g,b -> pointclass?
    ModelPtr RdbxIO::read(string filename)
    {
        ModelPtr model (new Model);
        PointBufferPtr pointBuffer(new PointBuffer);

        // Allocate point buffer and read data from file
        int c = 0;
        std::ifstream in(filename.c_str(), std::ios::binary);
        if (!in)
        {
            std::cerr << "File:"
                      << " " << filename << " "
                      << "could not be read!" << std::endl;
            return ModelPtr(new Model());
        }

        int numPoints = 0;
        in.read((char*)&numPoints, sizeof(int));

        //TODO: reduktion? brauchen wir das auch?

        while(in.good())
        {

        }

        //ID
        //pointBuffer->setPointArray();
        //Amplitude
        //Reflectance
        //pointBuffer->setColorArray();
        //pointclass?

        return model;
    }

    /*
    ModelPtr RdbxIO::read(string filename, int n, int reduction)
    {
        return read(filename, 4);
    }
    */

} // lvr2