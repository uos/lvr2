//
// Created by imitschke on 05.08.17.
//

MPISimpleQueue::MPISimpleQueue(boost::mpi::environment& env, boost::mpi::communicator& world)
: MPIQueue(env, world)
{

}

MPISimpleQueue::~MPISimpleQueue()
{

}

void MPISimpleQueue::finish()
{
  for(int i=0; i< nodeSize(); i++)
  {
    m_world.send(i+1, 99);
  }

}

template< typename MpiDataT >
int MPISimpleQueue::sendToFreeNode(int msg_id, const MpiDataT & data_in)
{
  auto avail_it =  m_nodes_avail.begin();
  int mpi_node_id = *avail_it;

   // mark rank/id as working

  markWorking(mpi_node_id);

  m_requests.insert(mpi_node_id);
  m_world.send(mpi_node_id, msg_id, data_in);

  return mpi_node_id;
}

bool MPISimpleQueue::receivedData(int i)
{

  return false;

}

template<typename MpiDataT>
int MPISimpleQueue::receiveData(MpiDataT& data_out, int tag)
{

  boost::mpi::status s;
  if(tag < 0)
  {
    s = m_world.recv(boost::mpi::any_source, boost::mpi::any_tag, data_out);
  }else{
    s = m_world.recv(boost::mpi::any_source, tag, data_out);
  }

  int source = s.source();

  auto response_itr = m_requests.find(source);

  if(response_itr != m_requests.end())
  {
    m_requests.erase(response_itr);
    markAvailable(source);
  }

  std::cout << source << ": " << data_out << '\n';

  return source;
}

template<typename MpiDataT>
std::map<int, MpiDataT> MPISimpleQueue::receiveAll(int tag)
{
  boost::mpi::request requests[size()];

  std::map<int, MpiDataT> results;

  int idx = 0;
  for(auto itr = m_nodes_working.begin(); itr != m_nodes_working.end(); itr++, idx++)
  {
    if(tag < 0)
    {
      requests[idx] = m_world.irecv(*itr, boost::mpi::any_tag, results[*itr] );
    }else{
      requests[idx] = m_world.irecv(*itr, tag, results[*itr] );
    }

  }

  boost::mpi::wait_all(requests, requests + size());

  return results;

}
