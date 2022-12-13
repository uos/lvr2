// Copyright 2011-2020 the Polygon Mesh Processing Library developers.
// Copyright 2001-2005 by Computer Graphics Group, RWTH Aachen
// Distributed under a MIT-style license, see PMP_LICENSE.txt for details.

#pragma once

#include <cassert>

#include <string>
#include <vector>
#include <algorithm>
#include <typeinfo>
#include <iostream>

#include <highfive/H5Group.hpp>
#include <highfive/H5DataSet.hpp>

namespace pmp {

class BasePropertyArray
{
public:
    //! Default constructor
    BasePropertyArray(const std::string& name) : name_(name) {}

    //! Destructor.
    virtual ~BasePropertyArray() {}

    //! Reserve memory for n elements.
    virtual void reserve(size_t n) = 0;

    //! Resize storage to hold n elements.
    virtual void resize(size_t n) = 0;

    //! Free unused memory.
    virtual void free_memory() = 0;

    //! Erase the contents of the property without calling destructors.
    virtual void shallow_clear() = 0;

    //! Let two elements swap their storage place.
    virtual void swap(size_t i0, size_t i1) = 0;

    //! Copy an element from another array.
    virtual void copy_prop(const BasePropertyArray* src, size_t src_i, size_t dst_i) = 0;

    //! Return a deep copy of self.
    virtual BasePropertyArray* clone() const = 0;

    //! Return an empty copy of self.
    virtual BasePropertyArray* empty_copy() const = 0;

    //! Write contents into group and clear the array.
    void unload(HighFive::Group& group)
    {
        auto [ start, len ] = raw_data();
        std::vector<size_t> dims = { len };
        if (group.exist(name_))
        {
            auto dataset = group.getDataSet(name_);
            if (dataset.getDimensions() != dims)
            {
                dataset.resize(dims);
            }
            dataset.write_raw(start);
        }
        else
        {
            HighFive::DataSpace dataSpace(dims);
            HighFive::DataSetCreateProps properties;
            properties.add(HighFive::Chunking({ dims[0] }));
            group.createDataSet<uint8_t>(name_, dataSpace, properties).write_raw(start);
        }
        shallow_clear();
    }

    //! Restore contents from group after unload().
    void restore(const HighFive::Group& group)
    {
        auto [ start, len ] = raw_data();
        auto dataset = group.getDataSet(name_);
        if (dataset.getDimensions()[0] != len)
        {
            throw std::runtime_error("PropertyArray::restore: Dimension mismatch");
        }
        dataset.read(start);
    }

    //! Return the type_info of the property
    virtual const std::type_info& type() const = 0;

    //! Return the name of the property
    const std::string& name() const { return name_; }

protected:
    //! Return the raw data pointer and length.
    virtual std::pair<uint8_t*, size_t> raw_data() = 0;

    std::string name_;
};

template <class T>
class PropertyArray : public BasePropertyArray
{
public:
    typedef T ValueType;
    typedef std::vector<ValueType> VectorType;
    typedef typename VectorType::reference reference;
    typedef typename VectorType::const_reference const_reference;

    PropertyArray(const std::string& name, T t = T())
        : BasePropertyArray(name), value_(t)
    {
    }

public: // interface of BasePropertyArray
    void reserve(size_t n) override { data_.reserve(n); }

    void resize(size_t n) override { data_.resize(n, value_); }

    void free_memory() override { data_.shrink_to_fit(); }

    void shallow_clear() override
    {
        // free the memory without calling any destructors:
        // pretend like data is a trivially destructible type (like uint8_t)
        uint8_t* raw_ptr = (uint8_t*)data_.data();
        // replace data_ with a new empty vector in a way that assumes it was uninitialized memory
        new (&data_) VectorType();
        // free the memory
        delete[] raw_ptr;
    }

    void swap(size_t i0, size_t i1) override
    {
        std::swap(data_[i0], data_[i1]);
    }

    void copy_prop(const BasePropertyArray* src, size_t src_i, size_t dst_i) override
    {
        auto src_prop = dynamic_cast<const PropertyArray<T>*>(src);
        assert(src_prop);
        data_[dst_i] = src_prop->data_[src_i];
    }

    BasePropertyArray* clone() const override
    {
        auto p = new PropertyArray<T>(name_, value_);
        p->data_ = data_;
        return p;
    }

    BasePropertyArray* empty_copy() const override
    {
        return new PropertyArray<T>(name_, value_);
    }

    const std::type_info& type() const override { return typeid(T); }

public:
    //! Get pointer to array (does not work for T==bool)
    T* data() { return data_.data(); }

    //! Get pointer to array (does not work for T==bool)
    const T* data() const { return data_.data(); }

    //! Get reference to the underlying vector
    VectorType& vector() { return data_; }

    //! Get reference to the underlying vector
    const VectorType& vector() const { return data_; }

    //! Access the i'th element. No range check is performed!
    reference operator[](size_t idx)
    {
        assert(idx < data_.size());
        return data_[idx];
    }

    //! Const access to the i'th element. No range check is performed!
    const_reference operator[](size_t idx) const
    {
        assert(idx < data_.size());
        return data_[idx];
    }

private:
    std::pair<uint8_t*, size_t> raw_data() override
    {
        return std::make_pair((uint8_t*)data_.data(), data_.size() * sizeof(T));
    }
    VectorType data_;
    ValueType value_;
};

// specialization for bool properties
// std::vector<bool> is a specialization that uses one bit per element, which does not allow data() access
template <>
inline bool* PropertyArray<bool>::data()
{
    throw std::runtime_error("PropertyArray<bool>::data() not supported");
}
template <>
inline const bool* PropertyArray<bool>::data() const
{
    throw std::runtime_error("PropertyArray<bool>::data() not supported");
}
template <>
inline void PropertyArray<bool>::swap(size_t i0, size_t i1)
{
    data_.swap(data_[i0], data_[i1]);
}
template <>
inline std::pair<uint8_t*, size_t> PropertyArray<bool>::raw_data()
{
    auto begin = data_.begin()._M_p;
    auto end = (data_.end() - 1)._M_p + 1; // -1 + 1 to get a past-the-end whole pointer, not past-the-end bit
    return std::make_pair((uint8_t*)begin, (end - begin) * sizeof(*begin));
}
template <>
inline void PropertyArray<bool>::shallow_clear()
{
    VectorType().swap(data_);
}

template <class T>
class Property
{
public:
    typedef PropertyArray<T> ArrayType;
    typedef typename ArrayType::reference reference;
    typedef typename ArrayType::const_reference const_reference;

    friend class PropertyContainer;
    friend class SurfaceMesh;

public:
    Property(ArrayType* p = nullptr) : parray_(p) {}

    void reset() { parray_ = nullptr; }

    operator bool() const { return parray_ != nullptr; }

    reference operator[](size_t i)
    {
        assert(parray_ != nullptr);
        return (*parray_)[i];
    }

    const_reference operator[](size_t i) const
    {
        assert(parray_ != nullptr);
        return (*parray_)[i];
    }

    T* data()
    {
        assert(parray_ != nullptr);
        return parray_->data();
    }

    const T* data() const
    {
        assert(parray_ != nullptr);
        return parray_->data();
    }

    typename ArrayType::VectorType& vector()
    {
        assert(parray_ != nullptr);
        return parray_->vector();
    }

    const typename ArrayType::VectorType& vector() const
    {
        assert(parray_ != nullptr);
        return parray_->vector();
    }

    const std::string& name() const
    {
        assert(parray_ != nullptr);
        return parray_->name();
    }

    ArrayType& array()
    {
        assert(parray_ != nullptr);
        return *parray_;
    }

    const ArrayType& array() const
    {
        assert(parray_ != nullptr);
        return *parray_;
    }

private:
    ArrayType* parray_;
};

template <class T>
class ConstProperty
{
public:
    typedef PropertyArray<T> ArrayType;
    typedef typename ArrayType::const_reference const_reference;

    friend class PropertyContainer;
    friend class SurfaceMesh;

public:
    ConstProperty(const ArrayType* p = nullptr) : parray_(p) {}
    ConstProperty(const Property<T>& p) : ConstProperty(&p.array()) {}

    void reset() { parray_ = nullptr; }

    operator bool() const { return parray_ != nullptr; }

    const_reference operator[](size_t i) const
    {
        assert(parray_ != nullptr);
        return (*parray_)[i];
    }

    const T* data() const
    {
        assert(parray_ != nullptr);
        return parray_->data();
    }

    const typename ArrayType::VectorType& vector()
    {
        assert(parray_ != nullptr);
        return parray_->vector();
    }

    const std::string& name() const
    {
        assert(parray_ != nullptr);
        return parray_->name();
    }

    const ArrayType& array() const
    {
        assert(parray_ != nullptr);
        return *parray_;
    }

private:
    const ArrayType* parray_;
};

/**
 * @brief map between identical properties from different containers
 *
 * Semantics: Copy from property 'in' to property 'out'. The properties may be from different meshes.
 * output_mesh.property_out[index_out] = input_mesh.property_in[index_in]
 *
 * becomes:
 * property_map.add(input_mesh.property_in, output_mesh.property_out);
 * property_map.copy(index_in, index_out);
 *
 * @tparam HandleT type of the handle used by the properties
 */
template<typename HandleT>
class PropertyMap
{
public:
    /**
     * @brief Add two properties of the same type to the map.
     *
     * @param in The property to copy from
     * @param out The property to copy to
     */
    void add(const BasePropertyArray* in, BasePropertyArray* out)
    {
        if (in->type() != out->type())
            throw std::runtime_error("PropertyMap::add(): types do not match");
        properties_.emplace_back(in, out);
    }

    /**
     * @brief Copy from in_handle in all 'in' properties to out_handle in all 'out' properties.
     *
     * @param in_handle The index in the input properties to copy from
     * @param out_handle The index in the output properties to copy to
     */
    void copy(HandleT in_handle, HandleT out_handle)
    {
        for (auto& [ in, out ] : properties_)
        {
            out->copy_prop(in, in_handle.idx(), out_handle.idx());
        }
    }
private:
    std::vector<std::pair<const BasePropertyArray*, BasePropertyArray*>> properties_;
};

class PropertyContainer
{
public:
    // default constructor
    PropertyContainer() : size_(0) {}

    // destructor (deletes all property arrays)
    virtual ~PropertyContainer() { clear(); }

    // copy constructor: performs deep copy of property arrays
    PropertyContainer(const PropertyContainer& rhs) { operator=(rhs); }

    // move constructor
    PropertyContainer(PropertyContainer&& rhs) = default;

    // assignment: performs deep copy of property arrays
    PropertyContainer& operator=(const PropertyContainer& rhs)
    {
        if (this != &rhs)
        {
            clear();
            size_ = rhs.size();
            for (auto& src : rhs.parrays_)
            {
                auto p = src->clone();
                parrays_.push_back(p);
                map_[src->name()] = p;
            }
        }
        return *this;
    }

    // move assignment
    PropertyContainer& operator=(PropertyContainer&& rhs) = default;

    // returns the current size of the property arrays
    size_t size() const { return size_; }

    // returns the number of property arrays
    size_t n_properties() const { return parrays_.size(); }

    // returns a vector of all property names
    std::vector<std::string> properties() const
    {
        std::vector<std::string> names;
        for (auto& parray : parrays_)
            names.push_back(parray->name());
        return names;
    }

    // add a property with name \p name and default value \p t
    template <class T>
    Property<T> add(const std::string& name, const T t = T())
    {
        if (exists(name))
            throw std::runtime_error("PropertyContainer::add: property already exists");

        // otherwise add the property
        auto p = new PropertyArray<T>(name, t);
        p->resize(size_);
        parrays_.push_back(p);
        map_[name] = p;
        return Property<T>(p);
    }

    // do we have a property with a given name?
    bool exists(const std::string& name) const
    {
        return map_.find(name) != map_.end();
    }

    // get a property by its name. returns invalid property if it does not exist.
    template <class T>
    Property<T> get(const std::string& name)
    {
        auto it = map_.find(name);
        if (it == map_.end())
            return Property<T>();
        if (it->second->type() != typeid(T))
            throw std::runtime_error("PropertyContainer::get: type mismatch");

        return Property<T>(dynamic_cast<PropertyArray<T>*>(it->second));
    }

    // get a property by its name. returns invalid property if it does not exist.
    template <class T>
    ConstProperty<T> get(const std::string& name) const
    {
        auto it = map_.find(name);
        if (it == map_.end())
            return ConstProperty<T>();
        if (it->second->type() != typeid(T))
            throw std::runtime_error("PropertyContainer::get: type mismatch");

        return ConstProperty<T>(dynamic_cast<const PropertyArray<T>*>(it->second));
    }

    // returns a property if it exists, otherwise it creates it first.
    template <class T>
    Property<T> get_or_add(const std::string& name, const T t = T())
    {
        auto p = get<T>(name);
        if (!p)
            p = add<T>(name, t);
        return p;
    }

    // get the type of property by its name. returns typeid(void) if it does not exist.
    const std::type_info& get_type(const std::string& name)
    {
        auto it = map_.find(name);
        if (it == map_.end())
            return typeid(void);
        return it->second->type();
    }

    // delete a property
    template <class T>
    void remove(Property<T>& h)
    {
        for (auto it = parrays_.begin(); it != parrays_.end(); ++it)
        {
            if (*it == h.parray_)
            {
                map_.erase((*it)->name());
                delete *it;
                parrays_.erase(it);
                h.reset();
                break;
            }
        }
    }

    // delete all properties
    void clear()
    {
        for (auto& parray : parrays_)
            delete parray;
        parrays_.clear();
        map_.clear();
        size_ = 0;
    }

    // delete the content of all properties, but keep the properties themselves
    void shallow_clear()
    {
        for (auto& parray : parrays_)
            parray->shallow_clear();
    }

    // reserve memory for n entries in all arrays
    void reserve(size_t n)
    {
        for (auto& parray : parrays_)
            parray->reserve(n);
    }

    // resize all arrays to size n
    void resize(size_t n)
    {
        for (auto& parray : parrays_)
            parray->resize(n);
        size_ = n;
    }

    // free unused space in all arrays
    void free_memory()
    {
        for (auto& parray : parrays_)
            parray->free_memory();
    }

    // add new element(s) to each vector
    void push_back(size_t n = 1)
    {
        resize(size_ + n);
    }

    // swap elements a and b in all properties
    void swap(size_t a, size_t b)
    {
        for (auto& parray : parrays_)
            parray->swap(a, b);
    }

    // write contents to group and clear memory, but keep the properties themselves
    void unload(HighFive::Group& group)
    {
        for (auto& parray : parrays_)
            parray->unload(group);
    }
    // restore contents after a call to unload(group)
    void restore(const HighFive::Group& group)
    {
        for (auto& parray : parrays_)
        {
            parray->resize(size_);
            parray->restore(group);
        }
    }

    void copy(const PropertyContainer& src)
    {
        for (auto& src_prop : src.parrays_)
        {
            if (!exists(src_prop->name()))
            {
                BasePropertyArray* p = src_prop->empty_copy();
                p->resize(size_);
                parrays_.push_back(p);
                map_[src_prop->name()] = p;
            }
        }
    }

    template<typename HandleT>
    PropertyMap<HandleT> gen_copy_map(const PropertyContainer& src, size_t offset) const
    {
        PropertyMap<HandleT> ret;
        for (size_t src_i = offset; src_i < src.parrays_.size(); ++src_i)
        {
            auto it = map_.find(src.parrays_[src_i]->name());
            if (it != map_.end() && it->second->type() == src.parrays_[src_i]->type())
                ret.add(src.parrays_[src_i], it->second);
        }
        return ret;
    }

private:
    std::vector<BasePropertyArray*> parrays_;
    std::unordered_map<std::string, BasePropertyArray*> map_;
    size_t size_;
};

} // namespace pmp
