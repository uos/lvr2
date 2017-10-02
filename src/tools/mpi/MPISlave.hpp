//
// Created by imitschke on 17.07.17.
//

#ifndef LAS_VEGAS_MPISLAVE_HPP
#define LAS_VEGAS_MPISLAVE_HPP
#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>
class MPISlave
{
public:
    MPISlave(boost::mpi::environment& env, boost::mpi::communicator& world);
    void doSomething(int data);
    void work();
private:
    boost::mpi::environment& m_env;
    boost::mpi::communicator& m_world;
};

#endif //LAS_VEGAS_MPISLAVE_HPP