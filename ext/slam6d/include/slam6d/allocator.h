/**
 * @file 
 * @brief allocator object that gets chunks of memory and then hands parts of them to a user
 * @author Jan Elsberg. Automation Group, Jacobs University Bremen gGmbH, Germany. 
 * @author Thomas Escher
 */
#ifndef ALLOCATOR_H
#define ALLOCATOR_H

#include <vector>

class Allocator {
public:
  virtual ~Allocator() {}

  template<typename T>
  T* allocate(unsigned int nr = 1) { return reinterpret_cast<T*>(allocate(nr*sizeof(T))); }
  
  virtual void printSize() const = 0;

protected:
  virtual unsigned char* allocate(unsigned int size) = 0;
};



class ChunkAllocator : public Allocator {
public:
  ChunkAllocator(unsigned int _csize = (1 << 20));
  ~ChunkAllocator();
  void printSize() const;
protected:
  unsigned char* allocate(unsigned int size);
private:
  std::vector<unsigned char *> mem;
  const unsigned int chunksize;
  unsigned int index;
  unsigned long int memsize;
  unsigned long int wastedspace;
};



class PackedChunkAllocator : public Allocator {
public:
  PackedChunkAllocator(unsigned int _csize = (1<<20));
  ~PackedChunkAllocator();
  void printSize() const;
protected:
  unsigned char* allocate(unsigned int size);
private:
  std::vector<unsigned char *> mem;
  std::vector<unsigned int > index;
  const unsigned int chunksize;
  unsigned long int memsize;
};



class SequentialAllocator : public Allocator {
public:
  //! Handle a preallocated memory up to \a max_size.
  SequentialAllocator(unsigned char* base_ptr, unsigned int max_size);
  ~SequentialAllocator();
  void printSize() const;
protected:
  unsigned char* allocate(unsigned int size);
private:
  unsigned char* m_base_ptr;
  unsigned int m_size, m_index;
};

#endif
