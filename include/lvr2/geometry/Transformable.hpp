#ifndef __LVRGEOMETRYTRANSFORMABLE_HPP__
#define __LVRGEOMETRYTRANSFORMABLE_HPP__

#include "Matrix4.hpp"

namespace lvr2
{

/**
 * @brief Interface for transformable objects
 */
template <typename BaseVecT>
class TransformableBase {
public:

    Matrix4<BaseVecT> getTransform()
    {
        return m_transform;
    }

    void setTransform(Matrix4<BaseVecT> transform)
    {
        m_transform = transform;
    }

private:
    Matrix4<BaseVecT> m_transform;
};

/**
 * Object of types Transformable can be detected at runtime:
 * SomeClass* obj;
 * auto transform_obj = dynamic_cast< Transformable* >(obj);
 * if(transform_obj)
 * {
 *      auto T = transform_obj->getTransform();
 * } else {
 *      // obj is not derived from Transformable
 * }
 */
typedef TransformableBase<BaseVector<float> > Transformable;

} // namespace lvr2

# endif // __LVRGEOMETRYTRANSFORMABLE_HPP__