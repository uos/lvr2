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

#include "lvr2/io/Polygon.hpp"
#include "lvr2/io/Timestamp.hpp"
#include <highfive/H5Easy.hpp>
#include <iostream>

namespace lvr2
{

Polygon::Polygon()
:base()
{
   
}

    Polygon::Polygon(floatArr points, size_t n)
{
    // Generate channel object pointer and add it
    // to channel map
    FloatChannelPtr point_data(new FloatChannel(n, 3, points));
    this->addFloatChannel(point_data, "points");

}



void Polygon::setPointArray(floatArr points, size_t n)
{
    FloatChannelPtr pts(new FloatChannel(n, 3, points));
    this->addFloatChannel(pts, "points");
}


void Polygon::load(std::string file)
{
    HighFive::File hdf5_file(file, HighFive::File::ReadOnly);
    std::vector<std::vector<float> > data = H5Easy::load<std::vector<std::vector<float> > >(hdf5_file,"/field/outer_boundary/coordinates");
    std::cout << "data size: " << data.size() << std::endl;
    floatArr arr(new float[data.size()*3]);
    for(size_t i = 0 ; i < data.size() ; i++)
    {
        arr[i*3] = data[i][0];
        arr[i*3+1] = data[i][1];
        arr[i*3+2] = data[i][2];
    }
    this->addFloatChannel(arr, "points", data.size(), 3);
}

floatArr Polygon::getPointArray()
{
    typename Channel<float>::Optional opt = getChannel<float>("points");
    if(opt)
    {
        return opt->dataPtr();
    }

    return floatArr();
}


size_t Polygon::numPoints() const
{
    const typename Channel<float>::Optional opt = getChannel<float>("points");
    if(opt)
    {
        return opt->numElements();
    }
    else
    {
        return 0;
    }
    
}


    Polygon Polygon::clone() const
{
    Polygon pb;

    for(const auto& elem : *this)
    {
        pb.insert({elem.first, elem.second.clone()});
    }

    return pb;

}



}


