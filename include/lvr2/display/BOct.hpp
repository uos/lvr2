#ifndef BOCT_HPP_
#define BOCT_HPP_


namespace lvr2
{

  struct BOct
  {
      long long m_child : 48;
      unsigned char m_valid : 8;
      unsigned char m_leaf : 8;
      BOct(): m_child(0), m_valid(0), m_leaf(0){}
  };
} 

#endif
