//
// Created by imitschke on 17.07.17.
//

#include "MPISlave.hpp"

#include <iostream>       // std::cout, std::endl
#include <thread>         // std::this_thread::sleep_for
#include <chrono>         // std::chrono::seconds

MPISlave::MPISlave(boost::mpi::environment &env, boost::mpi::communicator &world) : m_env(env), m_world(world)
{

}


void MPISlave::doSomething(int data )
{
  int result = data;


  m_world.send(0, 0, result);

}

void MPISlave::work()
{
  int tag;
  bool finished = false;
  while(!finished)
  {
    // std::cout << "WORK!" << std::endl;
    int rec;

    boost::mpi::status s = m_world.recv(0, boost::mpi::any_tag, rec);

    std::cout << "Tag " << s.tag() << std::endl;

    switch(s.tag())
    {
      case 0:
        doSomething(rec);
        break;
      default:
        finished = true;
        break;
    }

    // std::cout << "I am process " << m_world.rank() << " of " << m_world.size() << "." << std::endl;

    // std::cout << "INCREMENT " << rec << " -> " << rec+3 << std::endl;


  }

  std::cout << "Node " << m_world.rank() << " finished." << std::endl;
}
