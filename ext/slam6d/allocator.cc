/*
 * allocator implementation
 *
 * Copyright (C) Thomas Escher, Kai Lingemann
 *
 * Released under the GPL version 3.
 *
 */

#include "slam6d/allocator.h"

using std::vector;
#include <iostream>
using std::cout;
using std::endl;
#include <stdexcept>
using std::runtime_error;

#include <cstring> //memset



ChunkAllocator::ChunkAllocator(unsigned int _csize) :
  chunksize(_csize), index(_csize), memsize(0), wastedspace(0)
{}

ChunkAllocator::~ChunkAllocator()
{
  for(unsigned int i = 0; i < mem.size(); i++) {
    delete[] mem[i];
  }
}

void ChunkAllocator::printSize() const
{
  cout << "Alloc'd  " << memsize/(1024*1024.0) << " Mb " << endl;
  cout << " wasted  " << wastedspace/(1024*1024.0) << " Mb " << endl;
}

unsigned char* ChunkAllocator::allocate(unsigned int size)
{
  unsigned char* chunk;
  if (size + index > chunksize) {
    // create new chunk
    wastedspace += (chunksize-index);
    // check for oversize
    if (chunksize > size) {
      chunk = new unsigned char[chunksize];
      memset(chunk, 0, chunksize);
      memsize+=chunksize;
    } else {
      chunk = new unsigned char[size];
      memset(chunk, 0, size);
      memsize+=size;
    }
    mem.push_back(chunk);
    index = 0;
  } else {
    // use last chunk
    chunk = mem.back();
    chunk = chunk + index;
  }
  index += size;
  return chunk;                 
}



PackedChunkAllocator::PackedChunkAllocator(unsigned int _csize) :
  chunksize(_csize), memsize(0)
{}

PackedChunkAllocator::~PackedChunkAllocator()
{
  for(unsigned int i = 0; i < mem.size(); i++) {
    delete[] mem[i];
  }
}

void PackedChunkAllocator::printSize() const
{
  cout << "Alloc'd  " << memsize/(1024*1024.0) << " Mb " << endl;
  
  unsigned long int wastedspace = 0;
  for(unsigned int i = 0; i < index.size(); i++) {
    if(index[i] < chunksize) {
      wastedspace += chunksize - index[i];
    }
  }
  cout << "wasted  " << wastedspace/(1024*1024.0) << " Mb " << endl;
}

unsigned char* PackedChunkAllocator::allocate(unsigned int size)
{
  unsigned char* chunk;
  for (unsigned int i = 0; i < index.size(); i++) {
    if ( !(size + index[i] > chunksize) ) {
      // found a suitable entry
      chunk = mem[i];
      chunk = chunk + index[i];          // pointer to free byte
      index[i] += size;                  // increment index
      return chunk;                 
    }
  }
  // no chunk is large enough... make new one
  if (chunksize > size) {
    chunk = new unsigned char[chunksize];
    memset(chunk, 0, chunksize);
    memsize += chunksize;
  } else {   // in case the requested memory is larger than our chunks, make a single chunk thats large enough
    chunk = new unsigned char[size];
    memset(chunk, 0, size);
    memsize += size;
  }
  mem.push_back(chunk);
  index.push_back(size);
  return chunk;
}

SequentialAllocator::SequentialAllocator(unsigned char* base_ptr, unsigned int max_size) :
  m_base_ptr(base_ptr), m_size(max_size), m_index(0)
{}

SequentialAllocator::~SequentialAllocator()
{}

void SequentialAllocator::printSize() const
{
  cout << "Using " << m_index << " of " << m_size << " bytes." << endl;
}

unsigned char* SequentialAllocator::allocate(unsigned int size)
{
  if(m_index + size > m_size) {
    throw runtime_error("SequentialAllocator memory overflow");
  }
  unsigned char* r = m_base_ptr + m_index;
  m_index += size;
  return r;
}
