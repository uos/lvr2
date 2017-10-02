//
// Created by imitschke on 05.08.17.
//

#ifndef LAS_VEGAS_MPISIMPLEQUEUE_HPP
#define LAS_VEGAS_MPISIMPLEQUEUE_HPP

#include "MPIQueue.hpp"

#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi.hpp>

#include <set>
#include <map>



using namespace std;

class MPISimpleQueue : public MPIQueue
{
public:
    MPISimpleQueue(boost::mpi::environment& env, boost::mpi::communicator& world);
    ~MPISimpleQueue();


    template< typename MpiDataT >
    int sendToFreeNode(int msg_id, const MpiDataT& data_in);

    bool receivedData(int i);

    template<typename MpiDataT>
    int receiveData(MpiDataT& data_out, int tag=-1);

    template<typename MpiDataT>
    std::map<int, MpiDataT> receiveAll(int tag=-1);

    void finish();

private:

    std::set<int> m_requests;
};

#include "MPISimpleQueue.tcc"
#endif //LAS_VEGAS_MPISIMPLEQUEUE_HPP