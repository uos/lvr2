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
 * DMCVecPointHandle.tcc
 *
 *  @date 22.01.2019
 *  @author Benedikt Schumacher
 */

namespace lvr2
{

template<typename BaseVecT>
DMCVecPointHandle<BaseVecT>::DMCVecPointHandle(vector<coord<float>*> points)
{
    containedPoints.push_back(points);
}

template<typename BaseVecT>
vector<coord<float>*> DMCVecPointHandle<BaseVecT>::getContainedPoints(int index)
{
    // temporarily error handling
    if((index - 7) > (containedPoints.size() - 1))
    {
        return vector<coord<float>*>();
        std::cout << "no points for current cell" << std::endl;
    }
    return containedPoints[index - 7];
}

template<typename BaseVecT>
void DMCVecPointHandle<BaseVecT>::split(int index,
    vector<coord<float>*> splittedPoints[8],
    bool dual)
{
    if(!dual || index == 7)
    {
        containedPoints[index - 7].clear();
        vector<coord<float>* >().swap(containedPoints[index - 7]);
    }
    for(unsigned char i = 0; i < 8; i++)
    {
        containedPoints.push_back(splittedPoints[i]);
    }
}

template<typename BaseVecT>
void DMCVecPointHandle<BaseVecT>::clear()
{
    for(int i = 0;i < containedPoints.size(); i++)
    {
        containedPoints[i].clear();
        for(int j = 0; j < containedPoints[i].size(); j++)
        {
            delete(containedPoints[i][j]);
        }
        vector<coord<float>* >().swap(containedPoints[i]);
    }
    containedPoints.clear();
    vector<vector<coord<float>* > >().swap(containedPoints);
}

} // namespace lvr2
