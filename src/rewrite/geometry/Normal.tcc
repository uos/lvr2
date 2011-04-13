/*
 * Normal.tcc
 *
 *  Created on: 10.02.2011
 *      Author: Thomas Wiemann
 */

namespace lssr
{

template<typename CoordType>
void Normal<CoordType>::normalize()
{
    //Don't normalize if we don't have to
    float l_square =
            this->m_x * this->m_x
            + this->m_y * this->m_y
            + this->m_z * this->m_z;

    if( fabs(1 - l_square) > 0.001){

        float length = sqrt(l_square);
        if(length != 0){
            this->m_x /= length;
            this->m_y /= length;
            this->m_z /= length;
        }
    }
}

} // namespace lssr
