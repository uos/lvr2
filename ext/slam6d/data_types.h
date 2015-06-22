/**
 * @file
 * @brief Basic DataPointer class and its derivates SingleArray and TripleArray
 *
 * This file contains several classes for array-like access. The SingleArray
 * and TripleArray classes and their typedefs to DataXYZ/... overload
 * the operator[] and have a size function to act as their native arrays.
 * Similar to the array classes, SingleObject represents a whole object with
 * all its members in that allocated space.
 *
 * If an array of pointers to the elements of a TripleArray is required it can
 * create a temporary class PointerArray which holds creates and deletes a
 * native pointer array and follows the RAII-pattern.
 */

#ifndef DATA_TYPES_H
#define DATA_TYPES_H


/**
 * Representation of a pointer to a data field with no access methods.
 * 
 * Type-specialized access is gained by deriving from this class and
 * implementing access functions like operator[] and size().
 * 
 * The PrivateImplementation feature enables RAII-type locking mechanisms
 * used in the scanserver for holding CacheObject-locks. It is protected so
 * that scanserver-unaware code can only construct this class with a pointer
 * and size. Initialization of a derived class with these locking mechanisms
 * creates this class with the private implementation value, which will be
 * deleted in this dtor when it completely falls out of scope.
 */
class DataPointer {
protected:
  //! Subclass for storing further members and attaching an overloadable dtor
  class PrivateImplementation {
  public:
    virtual ~PrivateImplementation() {}
  };
public:
  //  DataPointer& operator=(const DataPointer&) = delete;
  //  DataPointer(const DataPointer&) = delete;

  /**
   * Ctor for the initial creation
   *
   * @param pointer base pointer to the data
   * @param size of the pointed data in bytes
   */
  DataPointer(unsigned char* pointer, unsigned int size,
    PrivateImplementation* private_impl = 0) :
    m_pointer(pointer), m_size(size), m_private_impl(private_impl) {
  }
  
  /**
   * Copy-Ctor for passing along via return by value
   *
   * The type-specialized classes (B) will be called with their
   * B(DataPointer&&) temporary ctor and call this constructor, so the private
   * imlementation has to be taken away. The temporary inside that constructor
   * isn't seen as temporary anymore, so we need a simple reference-ctor.
   */
  DataPointer(DataPointer& other) {
    m_pointer = other.m_pointer;
    m_size = other.m_size;
    // take ownership of this value, other is a temporary and will deconstruct
    m_private_impl = other.m_private_impl;
    other.m_private_impl = 0;
  };
  
  /**
   * Same as DataPointer(DataPointer&), except this is for functions returning
   * DataPointer instead of derived classes, so the temporary-ctor is used.
   */
  DataPointer(DataPointer&& other) {
    m_pointer = other.m_pointer;
    m_size = other.m_size;
    // take ownership of this value, other is a temporary and will deconstruct
    m_private_impl = other.m_private_impl;
    other.m_private_impl = 0;
  }
  
  //! Delete the private implementation with its derived dtor
  ~DataPointer() {
    if(m_private_impl != 0)
      delete m_private_impl;
  }
  
  //! Indicator for nullpointer / no data contained if false
  inline bool valid() {
    return m_size != 0;
  }
  
  inline unsigned char* get_raw_pointer() const { return m_pointer; }
  
protected:
  unsigned char* m_pointer;
  unsigned int m_size;

private:
  PrivateImplementation* m_private_impl;
};



template<typename T>
class SingleArray : public DataPointer {
public:
  //! Cast return-by-value temporary DataPointer to this type of array
  SingleArray(DataPointer&& temp) :
    DataPointer(temp)
  {
  }
  
  SingleArray(SingleArray&& temp) :
    DataPointer(temp)
  {
  }
  
  //! Represent the pointer as an array of T
  inline T& operator[](unsigned int i) const
  {
    return *(reinterpret_cast<T*>(m_pointer) + i);
  }
  
  //! The number of T instances in this array
  unsigned int size() {
    return m_size / sizeof(T);
  }
};



template<typename T>
class TripleArray : public DataPointer {
public:
  //! Cast return-by-value temporary DataPointer to this type of array
  TripleArray(DataPointer&& temp) :
    DataPointer(temp)
  {
  }
  
  TripleArray(TripleArray&& temp) :
    DataPointer(temp)
  {
  }
  
  //! Represent the pointer as an array of T[3]
  inline T* operator[](unsigned int i) const
  {
    return reinterpret_cast<T*>(m_pointer) + (i*3);
  }
  
  //! The number of T[3] instances in this array
  unsigned int size() const {
    return m_size / (3 * sizeof(T));
  }
};



template<typename T>
class SingleObject : public DataPointer {
public:
  //! Cast return-by-value temporary DataPointer to this type of object
  SingleObject(DataPointer&& temp) :
    DataPointer(temp)
  {
  }
  
  SingleObject(SingleObject&& temp) :
    DataPointer(temp)
  {
  }
  
  //! Type-cast
  inline T& get() const
  {
    return *reinterpret_cast<T*>(m_pointer);
  }
  
  //! There is only one object in here
  unsigned int size() const {
    return 1;
  }
};



/**
 * To simplify T** access patterns for an array of T[3] (points), this RAII-
 * type class helps creating and managing this pointer array on the stack.
 */
template<typename T>
class PointerArray {
public:
  //! Create a temporary array and fill it sequentially with pointers to points
  PointerArray(const TripleArray<T>& data) {
    unsigned int size = data.size();
    m_array = new T*[size];
    for(unsigned int i = 0; i < size; ++i)
      m_array[i] = data[i];
  }
  
  //! Removes the temporary array on destruction (RAII)
  ~PointerArray() {
    delete[] m_array;
  }
  
  //! Conversion operator to interface the TripleArray to a T** array
  inline T** get() const { return m_array; }
private:
  T** m_array;
};
  
// TODO: naming, see scan.h
typedef TripleArray<double> DataXYZ;
typedef TripleArray<float> DataXYZFloat;
typedef TripleArray<unsigned char> DataRGB;
typedef SingleArray<float> DataReflectance;
typedef SingleArray<float> DataAmplitude;
typedef SingleArray<int> DataType;
typedef SingleArray<float> DataDeviation;

#endif //DATA_TYPES_H
