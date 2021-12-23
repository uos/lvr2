/**
 * Copyright (c) 2020, University Osnabrück
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
 * SymmetricHausdorffMetric.tcc
 *
 *  Created on: Dec 10, 2020
 *      Author: Martin ben Ahmed
 */



 namespace lvr2
 {
    
    SymmetricHausdorffMetric::SymmetricHausdorffMetric()
    {
        // do nothing yet
    }

    
    SymmetricHausdorffMetric::~SymmetricHausdorffMetric()
    {
        // do nothing yet
    }

    const double SymmetricHausdorffMetric::get_distance(HalfEdgeMesh<BaseVector<float> > a, HalfEdgeMesh<BaseVector<float> >  b)
    {
        // returning the max of both one sided hausdorff distances
        double distance = std::max(OneSidedHausdorffMetric::get_distance(a, b), OneSidedHausdorffMetric::get_distance(b, a));
        std::cout << std::endl;
        std::cout << distance << std::endl;
        std::cout << std::endl;
        return distance;
    }
    
}