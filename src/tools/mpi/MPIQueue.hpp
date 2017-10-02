//
// Created by imitschke on 17.07.17.
//

#ifndef LAS_VEGAS_MPIQUEUE_HPP
#define LAS_VEGAS_MPIQUEUE_HPP

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>

#include <map>
#include <set>

using namespace std;

class MPIQueue {

public:

    MPIQueue(boost::mpi::environment& env, boost::mpi::communicator& world);

    template< typename MpiDataT >
    int sendToFreeNode(int msg_id, const MpiDataT& data_in);

    virtual bool hasFreeNode();

    virtual bool receivedData(int i) = 0;

    virtual size_t nodeSize();

    virtual size_t size();

protected:

    void markAvailable(int node_id);
    void markWorking(int node_id);

    boost::mpi::environment& m_env;
    boost::mpi::communicator& m_world;

    std::set<int> m_nodes_avail;
    std::set<int> m_nodes_working;


};

#include "MPIQueue.tcc"

#endif //LAS_VEGAS_MPIQUEUE_HPP