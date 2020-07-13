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
 * DMCVecPointHandle.hpp
 *
 *  @date 22.01.2019
 *  @author Benedikt Schumacher
 */

#ifndef DMCVecPointHandle_H_
#define DMCVecPointHandle_H_

#include <vector>
#include "lvr2/reconstruction/DMCPointHandle.hpp"
using std::vector;

namespace lvr2
{

template<typename BaseVecT>
class DMCVecPointHandle : public DMCPointHandle<BaseVecT>
{
public:

    /**
     * @brief Constructor.
     *
     * @param points vector of all points
     */
    DMCVecPointHandle(vector<coord<float>*> points);

    /**
     * @brief Destructor.
     */
    virtual ~DMCVecPointHandle() { };

    /**
     * @brief Get the Contained Points object at given index
     *
     * @param index Index of the octree cell
     * @return vector<coord<float>*> Vector of all points in the specififc cell
     */
    virtual vector<coord<float>*> getContainedPoints(int index);

    /**
     * @brief Splits the points of a specific cell into 8 subcelld
     *
     * @param index Index of splitted cell
     * @param splittedPoints Vector of the subcells points
     */
    virtual void split(int index,
        vector<coord<float>*> splittedPoints[8],
        bool dual);

    virtual void clear();

private:

    // Vector of vectors containing the cell specific points
    vector< vector<coord<float>*> > containedPoints;

};

} // namespace lvr2

#include "DMCVecPointHandle.tcc"

#endif /* DMCVecPointHandle_H_ */
