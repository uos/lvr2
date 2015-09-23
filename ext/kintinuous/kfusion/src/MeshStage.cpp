#include <kfusion/MeshStage.hpp>

// default constructor
MeshStage::MeshStage(double camera_target_distance, double voxel_size, Options* options) : AbstractStage(),
					camera_target_distance_(camera_target_distance), voxel_size_(voxel_size), options_(options)
{
	mesh_count_ = 0;
	timestamp.setQuiet(!options->verbose());
}

void MeshStage::firstStep() { /* skip */ };

void MeshStage::step()
{
	auto grid_work = boost::any_cast<pair<pair<TGrid*, bool>, vector<ImgPose*> > >(getInQueue()->Take());
	unordered_map<size_t, size_t> verts_map;
	TGrid* act_grid = grid_work.first.first;
	bool last_shift = grid_work.first.second;
	MeshPtr meshPtr = new HMesh();
	string mesh_notice = ("#### B:        Mesh Creation " +  to_string(mesh_count_) + "    ####");
	ScopeTime* cube_time = new ScopeTime(mesh_notice.c_str());
	
	cFastReconstruction* fast_recon =  new cFastReconstruction(act_grid);
	timestamp.setQuiet(!options_->verbose());
	// Create an empty mesh
	fast_recon->getMesh(*meshPtr);
	if(meshPtr->meshSize() == 0)
		return;
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
						auto vert_it = verts_map.find(inter);
						if(current_neighbor->m_fusionNeighborBox && vert_it == verts_map.end() && in2 != cFastBox::INVALID_INDEX && in2 != inter)
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
	mesh_count_++;
	transformMeshBack(meshPtr);
	delete cube_time;
	delete fast_recon;
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<ImgPose*> >(
				pair<MeshPtr, bool>(meshPtr, last_shift), grid_work.second));
	if(last_shift)
		done(true);
}
void MeshStage::lastStep()	{ /* skip */ }

void MeshStage::transformMeshBack(MeshPtr mesh)
{
	for(auto vert : mesh->getVertices())
	{
		// calc in voxel
		vert->m_position.x 	*= voxel_size_;				
		vert->m_position.y 	*= voxel_size_;				
		vert->m_position.z 	*= voxel_size_;			
		//offset for cube coord to center coord
		vert->m_position.x 	-= 1.5;				
		vert->m_position.y 	-= 1.5;				
		vert->m_position.z 	-= 1.5 - camera_target_distance_;				
		
		//offset for cube coord to center coord
		vert->m_position.x 	-= 150;				
		vert->m_position.y 	-= 150;				
		vert->m_position.z 	-= 150;
	}
}
