#pragma once

#ifndef LVR2_TYPES_ELEMENTPROXY
#define LVR2_TYPES_ELEMENTPROXY

#include <array>
#include <iostream>
#include <boost/optional.hpp>
#include <memory>
#include "lvr2/geometry/Handles.hpp"

namespace lvr2 {

// forward declaration for ElementProxyPtr
template<typename T>
class ElementProxy;

/**
 * @brief This class emulates a Pointer behaviour for an ElementProxy if its & operator is
 *        used. The arithmetic is based on the width of an ElementProxy. It was necessary for the
 *        Octree. USE WITH CARE.
 *
 * @tparam T the type of the underlying data array.
 */
template<typename T>
class ElementProxyPtr
{
  public:
    ElementProxyPtr(T* ptr = nullptr, size_t w = 0) : m_ptr(ptr), m_w(w) {}

    // "Pointer" arithmetic
    ElementProxyPtr operator+(const ElementProxyPtr&) = delete;
    
    ssize_t operator-(const ElementProxyPtr& p)
    {
      return (this->m_ptr - p.m_ptr) / m_w;
    }

    ElementProxyPtr& operator++()
    {
       m_ptr += m_w;
       return *this;
    }

    ElementProxyPtr operator+(size_t i)
    {
      ElementProxyPtr tmp(*this); 
      tmp += i;
      return tmp;
    }

    ElementProxyPtr operator++(int)
    { 
      ElementProxyPtr tmp(*this); 
      operator++();
      return tmp;
    }

    ElementProxyPtr& operator+=(size_t i)
    {
        m_ptr += (m_w * i);
        return *this;
    }

    ElementProxyPtr operator-=(size_t i)
    {
      m_ptr -= (m_w * i);
    }

    // comparison operators
    bool operator< (const ElementProxyPtr& rhs) const { return (*this).m_ptr < rhs.m_ptr;  }
    bool operator> (const ElementProxyPtr& rhs) const { return rhs < (*this); }
    bool operator<=(const ElementProxyPtr& rhs) const { return !((*this) > rhs); }
    bool operator>=(const ElementProxyPtr& rhs) const { return !((*this) < rhs); }
    bool operator==(const ElementProxyPtr& rhs) const { return (*this).m_ptr == rhs.m_ptr; }
    bool operator!=(const ElementProxyPtr& rhs) const { return !(*this == rhs); }

    // member access.
    ElementProxy<T> operator*() { return ElementProxy<T>(m_ptr, m_w); }

    // TODO [] operator ?!

    // I don't think we are able to return a useful raw pointer.
    // Array pointer breaks the abstraction.
    // Pointer to Elementproxy is pointless because we would need to create one with new.
    // We cannot overload the -> operator in ElementProxy for the same reasons.
    ElementProxy<T>* operator->() = delete;

  private:
    T* m_ptr;
    size_t m_w;

};

template<typename T>
class ElementProxy
{
public:
    friend class ElementProxyPtr<T>;
    /**
     * @brief look at ElementProxyPtr documentation. 
     *
     * @return 
     */
    ElementProxyPtr<T> operator&()
    {
      return ElementProxyPtr<T>(m_ptr, m_w);
    }

    ElementProxy operator=(const T& v)
    {
        if( m_ptr && (m_w == 1))
            m_ptr[0] = v;
        return *this;
    }

    template<typename BaseVecT>
    ElementProxy operator=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] = v.x;
            m_ptr[1] = v.y;
            m_ptr[2] = v.z;
        }
        return *this;
    }

    template<typename BaseVecT>
    ElementProxy operator+=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] += v.x;
            m_ptr[1] += v.y;
            m_ptr[2] += v.z;
        }
        return *this;
    }

    template<typename BaseVecT>
    ElementProxy operator-=(const BaseVecT& v)
    {
        if( m_ptr && (m_w > 2))
        {
            m_ptr[0] -= v.x;
            m_ptr[1] -= v.y;
            m_ptr[2] -= v.z;
        }
        return *this;
    }


    template<typename BaseVecT>
    BaseVecT operator+(const BaseVecT& v)
    {
        if(m_w > 2)
        {
            *this += v;
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        throw std::range_error("Element Proxy: Width to small for BaseVec addition");
    }

    template<typename BaseVecT>
    BaseVecT operator-(const BaseVecT& v)
    {
        if(m_w > 2)
        {
            *this -= v;
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        throw std::range_error("Element Proxy: Width to small for BaseVec subtraction");
    }

    ElementProxy(T* pos = nullptr, unsigned w = 0) : m_ptr(pos), m_w(w) {}

    T& operator[](int i) 
    {
        if(m_ptr && (i < m_w))
        {
            return m_ptr[i];
        }
        throw std::range_error("Element Proxy: Index larger than width");
    }

    const T& operator[](int i) const
    {
        if(m_ptr && (i < m_w))
        {
            return m_ptr[i];
        }
        throw std::range_error("Element Proxy: Index out of Bounds");
    }

    /// User defined conversion operator
    template<typename BaseVecT>
    operator BaseVecT() const
    {
        if(m_w == 3)
        {
            return BaseVecT(m_ptr[0], m_ptr[1], m_ptr[2]);
        }
        throw std::range_error("Element Proxy: Width != 3 in BaseVecT conversion");
    }

    operator std::array<VertexHandle, 3>() const
    {
        std::array<VertexHandle, 3> arr0 = {VertexHandle(0), VertexHandle(0), VertexHandle(0)};
        if(m_w == 3)
        {
            std::array<VertexHandle, 3> arr = {VertexHandle(m_ptr[0]), VertexHandle(m_ptr[1]), VertexHandle(m_ptr[2])};
            return  arr;
        }
        throw std::range_error("Element Proxy: Width != 3 in std::array conversion.");
    }

    operator EdgeHandle() const
    {
        if(m_w == 1)
        {
            return EdgeHandle(m_ptr[0]);
        }
        throw std::range_error("Element Proxy: Width != 1 in EdgeHandle conversion.");
    }

    operator FaceHandle() const
    {
        if(m_w == 1)
        {
            return FaceHandle(m_ptr[0]);
        }
        throw std::range_error("Element Proxy: Width != 1 in FaceHandle conversion.");
    }

    operator T() const
    {
        if(m_w == 1)
        {
            return m_ptr[0];
        }
        throw std::range_error("Element Proxy: Width != 1 in content type conversion.");
    }

private:

    T*              m_ptr;
    unsigned        m_w;
};

} // namespace lvr2

#endif // LVR2_TYPES_ELEMENTPROXY