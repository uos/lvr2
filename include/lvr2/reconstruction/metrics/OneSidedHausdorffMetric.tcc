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
 * OneSidedHausdorffMetric.tcc
 *
 *  Created on: Dec 10, 2020
 *      Author: Martin ben Ahmed
 */



 namespace lvr2
 {
    
    OneSidedHausdorffMetric::OneSidedHausdorffMetric()
    {
        // do nothing yet
    }

    
    OneSidedHausdorffMetric::~OneSidedHausdorffMetric()
    {
        // do nothing yet
    }

    const double OneSidedHausdorffMetric::get_distance(HalfEdgeMesh<BaseVector<float> > a, HalfEdgeMesh<BaseVector<float> >  b)
    {

        double max_distance = 0.0;
        double min_distance = std::numeric_limits<double>::max();
        double current_distance = 0.0;

        int i = 0;
        int j = 0;
        

        std::cout << "Size of A: " << a.numVertices() << std::endl;
        std::cout << "Size of B: " << b.numVertices() << std::endl;
        // iterating over vertex handles of a and b
        #pragma omp parallel
        for (auto vertexHandleA : a.vertices())
        {
            // getting vertex A
            auto vertexA = a.getVertexPosition(vertexHandleA);
            
            // resetting min distance after each run of the inner loop
            double min_distance = std::numeric_limits<double>::max();
            
            for(auto vertexHandleB : b.vertices())
            {
                
                // getting the vertex B    
                auto vertexB = b.getVertexPosition(vertexHandleB);

                // calculating the distance between to vertices
                current_distance = vertexA.distance(b.getVertexPosition(vertexHandleB));
    
                // cancel distance calculation if current_distance is smaller than max_distance
                if(current_distance < max_distance)
                {
                    min_distance = 0;
                    break;
                }
                min_distance = std::min(min_distance, current_distance);
            }
            max_distance = std::max(max_distance, min_distance);
        }
        
        std::cout <<  std::endl;
        std::cout << max_distance << std::endl;
        std::cout << std::endl;
        return max_distance;
    }
    
}