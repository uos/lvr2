

#ifndef LSNODE_HPP_
#define LSNODE_HPP_


// vorerst
#include <string>
#include <iostream>
#include "geometry/BoundingBox.hpp"

namespace lvr{
using namespace lvr;
/**
 * @brief A class for execution of a distribution using a very simple Kd-tree.
 * 	The classification criterion is to halve the longest axis.
 */
    class LSNode {
    public:
        LSNode()
        {
            this->m_points = vector<unsigned long long int>();
            again = 0;
        }

        BoundingBox<Vertex<float>> &getBoundingBox()  {
            return m_bbox;
        }

        void setBoundingBox( BoundingBox<Vertex<float>> &m_bbox) {
            this->m_bbox = m_bbox;
        }

         vector<unsigned long long int> &getPoints()  {
            return m_points;
        }

        void setPoints( vector<unsigned long long int> &m_points) {
            this->m_points = m_points;
        }
        int again;
    private:
        vector<unsigned long long int> m_points;
        BoundingBox<Vertex<float>> m_bbox;
    };
}


#endif /* LSNODE_HPP_ */
