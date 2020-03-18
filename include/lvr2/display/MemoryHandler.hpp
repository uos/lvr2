#ifndef MEMORY_HANDLER
#define MEMORY_HANDLER

#include <iostream>
#include <vector>
#include <string.h> //memset

class ChunkMemoryHandler
{  private:
  std::vector<unsigned char*> m_memory;
  std::vector<size_t> m_used;
  // chunk capacity;
  size_t m_capacity;


  public:
  ChunkMemoryHandler(size_t size = 131072) : m_used(0), m_capacity(size)
  {
    unsigned char* mem = (unsigned char*)malloc(size);
    m_memory.push_back(mem);
    memset(mem, 0, size);
    m_used.push_back(0);
  }


  virtual ~ChunkMemoryHandler()
  {
    for(size_t i = 0; i < m_memory.size(); ++i)
    {
      free(m_memory[i]);
    }
  }
  template <typename T>
    void* alloc(size_t required, long& offset)
    {
      required = (sizeof(T) * required);
      for(size_t i = 0; i < m_memory.size(); ++i)
      {
        size_t used = m_used[i];
        if((m_capacity - used) >= required)
        {
          m_used[i] += required;

          return m_memory[i] + used;
        }
      }
      

      ssize_t size = (required > m_capacity) ? required : m_capacity;
      
      unsigned char* mem = (unsigned char*)malloc(size);
      memset(mem, 0, size);
      m_memory.push_back(mem);
      m_used.push_back(required);
      return mem;
    }


};

class BlockMemoryHandler 
{
  public:
    // 1 MiB default
    BlockMemoryHandler(size_t size = 32) : m_size(size), m_used(0), m_capacity(size)
  {
    m_memory = (unsigned char*)malloc(m_size);
    m_address = m_memory;
  }

    size_t m_size;
    size_t m_used;
    size_t m_capacity;
    unsigned char* m_memory;
    unsigned char* m_address;

    template <typename T>
      void* alloc(size_t required, long& offset)
      {
        required = (sizeof(T) * required);

        if((m_capacity - m_used) < required)
        {
          long addSize = (required > m_size) ? required : m_size;
          m_capacity += addSize;

          unsigned char* tmp = (unsigned char*)realloc(m_memory, m_capacity);

          // check if realloc failed
          if(NULL == tmp)
          {
            return tmp;
          }

          m_memory = tmp;

          if(m_memory != m_address)
          {
            offset = m_memory - m_address;
            m_address = m_memory;
          }
        }

        unsigned char* ptr = &m_memory[m_used];
        memset(ptr, 0, required);
        m_used += required;
        return (void*)ptr;
      }

    void shrinkToFit()
    {
      m_memory = (unsigned char*)realloc(m_memory, m_used);
      if(m_memory != m_address)
      {
        std::cout << "FUCK WHY" << std::endl;
      }
    }

    virtual ~BlockMemoryHandler()
    {
      free(m_memory);
      m_memory = NULL;
      m_address = NULL;
    }

};

#endif
