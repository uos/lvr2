#include <kfusion/OptimizeStage.hpp>

// default constructor
OptimizeStage::OptimizeStage(double camera_target_distance, double voxel_size) : AbstractStage()
	,mesh_count_(0), camera_target_distance_(camera_target_distance), voxel_size_(voxel_size),textured(true),bounding_counter(0),pic_count_(0)
{
}

void OptimizeStage::firstStep() { optiMesh_ = NULL; };

void OptimizeStage::step()
{
	std::cout << "CamDistance: " << camera_target_distance_ << std::endl;
	auto mesh_work = boost::any_cast<pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> > >(getInQueue()->Take());
	bool last_shift = mesh_work.first.second;
	MeshPtr act_mesh = mesh_work.first.first;
    	transformMeshBack(act_mesh);
	//if(optiMesh_ == NULL)
		optiMesh_ = act_mesh;
	//else
		//optiMesh_->addMesh(act_mesh);
	std::vector<kfusion::ImgPose*> image_poses_buffer = mesh_work.second;
	std::cout << "Loaded " << image_poses_buffer.size() << " Images. " << std::endl;
	
	optiMesh_->optimizePlanes(3, 0.85, 7, 0);
	//act_mesh->optimizePlaneIntersections();
	MeshPtr tmp_pointer = optiMesh_->retesselateInHalfEdge(0.01,textured,bounding_counter);
	std::cout << "            ####     3 Finished optimisation number: " << mesh_count_ << "   ####" << std::endl;
	mesh_count_++;
	
	///texturing
	if(textured){
		int counter=0;
		for(int i=0;i<image_poses_buffer.size();i++){
			counter = optiMesh_->projectAndMapNewImage(*(image_poses_buffer[i]));
		}
		bounding_counter += counter;
		if(meshBufferPtr){
			std::cout << "addMesh" << std::endl;
			addMesh(optiMesh_->meshBuffer(),meshBufferPtr); 
			//meshBufferPtr = act_mesh->meshBuffer();
		} else {
			meshBufferPtr = optiMesh_->meshBuffer();
		}
		image_poses_buffer.resize(0);
	}
	
	
	getOutQueue()->Add(pair<pair<MeshPtr, bool>, vector<kfusion::ImgPose*> >(
				pair<MeshPtr, bool>(tmp_pointer, last_shift), mesh_work.second));
	//delete act_mesh;
	if(last_shift)
		done(true);
}
void OptimizeStage::lastStep()	{ 
	
	if(textured)
	{
		int i=0;
		//PointSave saver;
		//saver.saveBoundingRectangles("b_rects.ply",global_tmp_pointer->getBoundingRectangles(i),global_tmp_pointer->getBoundingRectangles(i).size()*4);
		//cout << i/4 << " Bounding Rectangles" << endl;
	
		ModelPtr dst_model(new Model(meshBufferPtr));
		ModelFactory::saveModel(dst_model, "./mesh_OUT.obj");
		cout << "File saved to mesh_OUT.obj" << endl;
	}
	
};

void OptimizeStage::transformMeshBack(MeshPtr mesh)
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

void OptimizeStage::addMesh(MeshBufferPtr src, MeshBufferPtr dst){
	size_t a,b,c,d,e,f,g;
	
	std::cout << "Begin adding Buffers" << std::endl;
	
	//define dst buffers
	std::vector<float> vertexBuffer_dst;
    std::vector<float> normalBuffer_dst;
    std::vector<uchar> colorBuffer_dst;
    std::vector<unsigned int> indexBuffer_dst;
    std::vector<float> textureCoordBuffer_dst;
    std::vector<unsigned int> materialIndexBuffer_dst;
    std::vector<Material*> materialBuffer_dst;
   
    
    //get src buffers
    //Vertices
    floatArr vertexBuffer_src = dst->getVertexArray(a);
    //std::cout << "a: " << a << std::endl;
    for(size_t i=0;i<(a*3);i++)
		vertexBuffer_dst.push_back(vertexBuffer_src[i]);
    vertexBuffer_src = src->getVertexArray(a);
    for(size_t i=0;i<a*3;i++)
		vertexBuffer_dst.push_back(vertexBuffer_src[i]);
		
	//Normals
	floatArr normalBuffer_src = dst->getVertexNormalArray(a);
    for(size_t i=0;i<a*3;i++)
		normalBuffer_dst.push_back(normalBuffer_src[i]);
    normalBuffer_src = src->getVertexNormalArray(a);
    for(size_t i=0;i<a*3;i++)
		normalBuffer_dst.push_back(normalBuffer_src[i]);
    
    //Colors
    ucharArr colorBuffer_src = dst->getVertexColorArray(a);
    for(size_t i=0;i<a*3;i++)
		colorBuffer_dst.push_back(colorBuffer_src[i]);
    colorBuffer_src = src->getVertexColorArray(a);
    for(size_t i=0;i<a*3;i++)
		colorBuffer_dst.push_back(colorBuffer_src[i]);
	
	//Faces
	uintArr indexBuffer_src = dst->getFaceArray(a);
	for(size_t i=0;i<a*3;i++)
		indexBuffer_dst.push_back(indexBuffer_src[i]);
	indexBuffer_src = src->getFaceArray(b);
	for(size_t i=0;i<b*3;i++)
		indexBuffer_dst.push_back(indexBuffer_src[i]+a);
		
	//TextureCoords
	floatArr textureCoordBuffer_src = dst->getVertexTextureCoordinateArray(a);
	for(size_t i=0;i<a*3;i++)
		textureCoordBuffer_dst.push_back(textureCoordBuffer_src[i]);
	textureCoordBuffer_src = src->getVertexTextureCoordinateArray(a);
	for(size_t i=0;i<a*3;i++)
		textureCoordBuffer_dst.push_back(textureCoordBuffer_src[i]);
		
	//Material
	materialArr materialBuffer_src = dst->getMaterialArray(a);
	for(size_t i=0;i<a;i++)
		materialBuffer_dst.push_back(materialBuffer_src[i]);
	materialBuffer_src = src->getMaterialArray(a);
	for(size_t i=0;i<a;i++)
		materialBuffer_dst.push_back(materialBuffer_src[i]);
		
	//MaterialIndices
	uintArr materialIndexBuffer_src = dst->getFaceMaterialIndexArray(a);
	for(size_t i=0;i<a;i++)
		materialIndexBuffer_dst.push_back(materialIndexBuffer_src[i]);
	materialIndexBuffer_src = src->getFaceMaterialIndexArray(b);
	for(size_t i=0;i<b;i++)
		materialIndexBuffer_dst.push_back(materialIndexBuffer_src[i]+a);
	
	
	
    dst->setVertexArray( vertexBuffer_dst );
    dst->setVertexNormalArray( normalBuffer_dst );
    dst->setVertexColorArray( colorBuffer_dst );
    dst->setFaceArray( indexBuffer_dst );
    dst->setVertexTextureCoordinateArray( textureCoordBuffer_dst );
	dst->setMaterialArray( materialBuffer_dst );
	dst->setFaceMaterialIndexArray( materialIndexBuffer_dst );
    
    std::cout << "success" << std::endl;
    
}

