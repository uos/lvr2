#ifndef __LVRGEOMETRYTRANSFORMABLE_HPP__
#define __LVRGEOMETRYTRANSFORMABLE_HPP__

#include <Eigen/Dense>

namespace lvr2
{

/**
 * @brief Interface for transformable objects
 */
template <typename T>
class TransformableBase 
{
public:

    Eigen::Matrix<T, 4, 4> getTransform()
    {
        return m_transform;
    }

    void setTransform(Eigen::Matrix<T, 4, 4> transform)
    {
        m_transform = transform;
    }

private:
    Eigen::Matrix<T, 4, 4> m_transform;
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
typedef TransformableBase<double> Transformable;

} // namespace lvr2

# endif // __LVRGEOMETRYTRANSFORMABLE_HPP__