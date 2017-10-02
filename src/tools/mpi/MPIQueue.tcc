MPIQueue::MPIQueue(boost::mpi::environment& env, boost::mpi::communicator& world) : m_env(env), m_world(world)
{
    for(int i = 1 ; i < world.size() ; i++)
    {
        m_nodes_avail.insert(i);
    }
}

bool MPIQueue::hasFreeNode()
{
    return ( m_nodes_avail.size() > 0);
}

size_t MPIQueue::nodeSize()
{
  return m_nodes_avail.size() + m_nodes_working.size();
}

size_t MPIQueue::size()
{
  return m_nodes_working.size();
}


void MPIQueue::markAvailable(int node_id)
{
  std::set<int>::const_iterator it = m_nodes_working.find(node_id);

  if(it != m_nodes_working.end() )
  {
    int value = *it;
    m_nodes_avail.insert(value);

    it = m_nodes_working.erase(it);
  } else {
    std::cout << "Error: Node " << node_id << " not working" << std::endl;
  }
}

void MPIQueue::markWorking(int node_id)
{
  std::set<int>::const_iterator it = m_nodes_avail.find(node_id);
  if(it != m_nodes_avail.end() )
  {
    int value = *it;
    m_nodes_working.insert(value);
    it = m_nodes_avail.erase(it);
  }else{
      std::cout << "Error: Node " << node_id << " not available" << std::endl;
  }
}
