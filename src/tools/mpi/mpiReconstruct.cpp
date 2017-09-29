#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
//
// Created by imitschke on 09.07.17.
//

int main(int argc, char* argv[])
{
  boost::mpi::environment env;
  boost::mpi::communicator world;

  //MASTER Node
  if (world.rank() == 0)
  {

  }
  //SLAVE Node
  else
  {

  }
}