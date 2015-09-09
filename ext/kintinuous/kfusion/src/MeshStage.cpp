#include <kfusion/MeshStage.hpp>

// default constructor
MeshStage::MeshStage() : AbstractStage()
{
	mesh_count_ = 0;
}

void MeshStage::firstStep() { /* skip */ };

void MeshStage::step()
{
	auto grid_work = boost::any_cast<pair<TGrid*, bool> >(getInQueue()->Take());
	unordered_map<size_t, size_t> verts_map;
	TGrid* act_grid = grid_work.first;
	bool last_shift = grid_work.second;
	MeshPtr meshPtr = new HMesh();
	string mesh_notice = ("#### B:        Mesh Creation " +  to_string(mesh_count_) + "    ####");
	ScopeTime* cube_time = new ScopeTime(mesh_notice.c_str());
	
	cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
	timestamp.setQuiet(true);
	// Create an empty mesh
	fast_recon->getMesh(*meshPtr);
	// mark all fusion vertices in the mesh
	for(auto cellPair : act_grid->getFusionCells())
	{
		cFastBox* box = cellPair.second;
		for( int i = 0; i < 12; i++)
		{
			uint inter = box->m_intersections[i];
			if(inter != cFastBox::INVALID_INDEX)
			{
				meshPtr->setFusionVertex(inter);
			}
		}
	}
	for(auto cellPair : act_grid->m_old_fusion_cells)
	{
		cFastBox* box = cellPair.second;
		for( int edge_index = 0; edge_index < 12; edge_index++)
		{
			uint inter = box->m_intersections[edge_index];
			uint inter2  = -1;
			if(inter != cFastBox::INVALID_INDEX)
			{
				for(int i = 0; i < 3; i++)
				{
					auto current_neighbor = box->m_neighbors[neighbor_table[edge_index][i]];
					if(current_neighbor != 0)
					{
						uint in2 = current_neighbor->m_intersections[neighbor_vertex_table[edge_index][i]];
						auto vert_it = verts_map.find(in2);
						if(vert_it == verts_map.end() && in2 != cFastBox::INVALID_INDEX && in2 != 0 && in2 != inter && current_neighbor->m_fusionNeighborBox)
						{
							inter2 = in2;
							
							verts_map.insert(pair<size_t, size_t>(inter, inter2));
							meshPtr->setOldFusionVertex(inter2);
							break;
						}
					}
				}
			}
			
		}
	}
	meshPtr->m_fusion_verts = verts_map;
	delete cube_time;
	delete fast_recon;
	getOutQueue()->Add(pair<MeshPtr, bool>(meshPtr, grid_work.second));
	if(last_shift)
		done(true);
}
void MeshStage::lastStep()	{ /* skip */ };
