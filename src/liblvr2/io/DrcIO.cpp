/* Copyright (C) 2018 Uni Osnabrück
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

/**
 *
 * @file      DrcIO.cpp
 *
 * @author    Steffen Schupp (sschupp), sschupp@uos.de
 * @author	  Malte kl. Piening (mklpiening), mklpiening@uos.de
 *
 **/

#include <fstream>
#include <unistd.h>

#include "lvr2/io/DracoDecoder.hpp"
#include "lvr2/io/DracoEncoder.hpp"
#include "lvr2/io/DrcIO.hpp"

namespace lvr2
{

ModelPtr DrcIO::read(string filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "File:"
                  << " " << filename << " "
                  << "could not be read!" << std::endl;
        return ModelPtr(new Model());
    }

    std::streampos file_size = 0;
    file.seekg(0, std::ios::end);
    file_size = file.tellg() - file_size;
    file.seekg(0, std::ios::beg);
    std::vector<char> data(file_size);
    file.read(data.data(), file_size);

    if (data.empty())
    {
        std::cerr << "File:"
                  << " " << filename << " "
                  << "is empty!" << std::endl;
        return ModelPtr(new Model());
    }

    draco::DecoderBuffer buffer;
    buffer.Init(data.data(), data.size());

    auto type = draco::Decoder::GetEncodedGeometryType(&buffer);

    if (!type.ok())
    {
        std::cerr << "Content in"
                  << " " << filename << " "
                  << "is neither a Mesh nor a PointCloud!" << std::endl;
        return ModelPtr(new Model());
    }

    const draco::EncodedGeometryType geom_type = type.value();

    // decode
    ModelPtr modelPtr = decodeDraco(buffer, geom_type);
    m_model           = modelPtr;
    return modelPtr;
}

void DrcIO::save(string filename)
{
    // check for validity
    if (!m_model)
    {
        std::cerr << "no model set for export!" << std::endl;
        return;
    }

    if (!m_model->m_mesh && !m_model->m_pointCloud)
    {
        std::cerr << "model does not contain geometry data!" << std::endl;
        return;
    }

    // encode
    std::unique_ptr<draco::EncoderBuffer> buffer =
        encodeDraco(m_model, (m_model->m_pointCloud ? draco::EncodedGeometryType::POINT_CLOUD
                                                    : draco::EncodedGeometryType::TRIANGULAR_MESH));

    if (buffer)
    {
        std::ofstream file(filename, std::ios::binary);

        // write to file
        file.write(buffer->data(), buffer->size());
    }
}

void DrcIO::save(ModelPtr model, string filename)
{
    m_model = model;
    save(filename);
}

} /* namespace lvr */
