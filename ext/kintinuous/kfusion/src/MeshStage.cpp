#include <kfusion/MeshStage.hpp>

// default constructor
MeshStage::MeshStage() : AbstractStage()
{
	mesh_count_ = 0;
}

void MeshStage::firstStep() { /* skip */ };

void MeshStage::step()
{
	auto grid_work = boost::any_cast<pair<pair<TGrid*, bool>, vector<ImgPose*> > >(getInQueue()->Take());
	unordered_map<size_t, size_t> verts_map;
	TGrid* act_grid = grid_work.first.first;
	bool last_shift = grid_work.first.second;
	MeshPtr meshPtr = new HMesh();
	ScopeTime* cube_time = new ScopeTime("Marching Cubes");
	cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
	
	// Create an empty mesh
	fast_recon->getMesh(*meshPtr);
	//meshPtr->m_fusionBoxes = act_grid->getFusionCells();
	//meshPtr->m_oldfusionBoxes = act_grid->m_old_fusion_cells;
	//meshPtr->m_fusionNeighborBoxes = act_grid->m_fusion_cells_neighbors;
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
	/*if(slice_count_ == 0)
	meshPtr_ = meshPtr;
	else
	{
		mesh_queue_.push(meshPtr);
	}*/
	delete cube_time;
	delete fast_recon;
	std::cout << "        ####     2 Finished reconstruction number: " << mesh_count_ << "   ####" << std::endl;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(meshPtr, last_shift), grid_work.second));
	if(last_shift)
		done(true);
}
void MeshStage::lastStep()	{ /* skip */ };
