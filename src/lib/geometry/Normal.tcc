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
            this->x * this->x
            + this->y * this->y
            + this->z * this->z;

    if( fabs(1 - l_square) > 0.001){

        float length = sqrt(l_square);
        if(length != 0){
            this->x /= length;
            this->y /= length;
            this->z /= length;
        }
    }
}

} // namespace lssr
