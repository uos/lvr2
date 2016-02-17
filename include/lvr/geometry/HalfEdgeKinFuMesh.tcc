/* Copyright (C) 2015 Uni Osnabrück
 * This file is part of the LAS VEGAS Reconstruction Toolkit,
 *
 * LAS VEGAS is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * LAS VEGAS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA  02111-1307, USA
 */


/*
 * HalfEdgeKinFuMesh.tcc
 *
 *  @date 13.11.2015
 *  @author ristan Igelbrink (tigelbri@uos.de)
 */


namespace lvr
{

template<typename VertexT, typename NormalT>
HalfEdgeKinFuMesh<VertexT, NormalT>::HalfEdgeKinFuMesh( ) : HalfEdgeMesh<VertexT, NormalT >()
{
    start_texture_index=0;
}

template<typename VertexT, typename NormalT>
HalfEdgeKinFuMesh<VertexT, NormalT>::HalfEdgeKinFuMesh(
        typename PointsetSurface<VertexT>::Ptr pm ) : HalfEdgeMesh<VertexT, NormalT >(pm)
{

}

template<typename VertexT, typename NormalT>
HalfEdgeKinFuMesh<VertexT, NormalT>::HalfEdgeKinFuMesh(
        MeshBufferPtr mesh) : HalfEdgeMesh<VertexT, NormalT >(mesh)
{
    start_texture_index=0;
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::addMesh(HalfEdgeKinFuMesh<VertexT, NormalT>* slice, bool texture)
{
	size_t old_vert_size = this->m_vertices.size();
	this->m_vertices.resize(old_vert_size +  slice->m_vertices.size() - slice->m_fusion_verts.size());

    size_t count = 0;
    unordered_map<size_t, size_t> fused_verts;
    for(int i = 0; i < slice->m_vertices.size();i++)
    {
		size_t index = old_vert_size + i - count;
		if(!slice->m_vertices[i]->m_oldFused)
		{
			this->m_vertices[index] = slice->m_vertices[i];
			//fused_verts[i] = index;
			this->m_vertices[index]->m_actIndex = index;
		}
		else
			count++;
	}
	for(auto vert_it = slice->m_fusion_verts.begin(); vert_it != slice->m_fusion_verts.end(); vert_it++)
	{

		mergeVertex(vert_it->first, vert_it->second);
	}

	size_t old_size = this->m_faces.size();

	this->m_faces.resize(old_size +  slice->m_faces.size());
    for(int i = 0; i < slice->m_faces.size();i++)
    {
		size_t index = old_size + i;
		this->m_faces[index] = slice->m_faces[i];
	}

	//m_fused_verts = fused_verts;
	m_slice_verts = slice->m_slice_verts;
	this->m_globalIndex = this->meshSize();

    if(slice->m_meshBuffer && this->m_meshBuffer)
	{
        //Colors
        size_t a = 0,b = 0;
        std::vector<unsigned char> colorBuffer_dst;
        ucharArr colorBuffer_src_A = this->m_meshBuffer->getVertexColorArray(a);
        ucharArr colorBuffer_src_B = slice->m_meshBuffer->getVertexColorArray(b);
        colorBuffer_dst.resize(a*3+b*3);
        for(size_t i=0;i<a*3;i++)
            colorBuffer_dst[i] = colorBuffer_src_A[i];
        for(size_t i=0;i<b*3;i++)
            colorBuffer_dst[a*3+i] = colorBuffer_src_B[i];
        this->m_meshBuffer->setVertexColorArray(colorBuffer_dst);
    }
	if(slice->m_meshBuffer && this->m_meshBuffer && texture)
	{

		size_t a = 0,b = 0;

		//TextureCoords
		std::vector<float> textureCoordBuffer_dst;
		floatArr textureCoordBuffer_src_A = this->m_meshBuffer->getVertexTextureCoordinateArray(a);
		floatArr textureCoordBuffer_src_B = slice->m_meshBuffer->getVertexTextureCoordinateArray(b);
		textureCoordBuffer_dst.resize(a*3+b*3);
		for(size_t i=0;i<a*3;i++)
			textureCoordBuffer_dst[i] = textureCoordBuffer_src_A[i];
		for(size_t i=0;i<b*3;i++)
			textureCoordBuffer_dst[a*3+i] = textureCoordBuffer_src_B[i];

		//Material
		size_t offset = 0;
		std::vector<Material*> materialBuffer_dst;
		materialArr materialBuffer_src_A = this->m_meshBuffer->getMaterialArray(offset);
		materialArr materialBuffer_src_B = slice->m_meshBuffer->getMaterialArray(a);
		materialBuffer_dst.resize(offset+a);
		for(size_t i=0;i<offset;i++)
			materialBuffer_dst[i] = materialBuffer_src_A[i];
		for(size_t i=0;i<a;i++)
			materialBuffer_dst[offset+i] = materialBuffer_src_B[i];

		//MaterialIndices
		std::vector<unsigned int> materialIndexBuffer_dst;
		uintArr materialIndexBuffer_src_A = this->m_meshBuffer->getFaceMaterialIndexArray(a);
		uintArr materialIndexBuffer_src_B = slice->m_meshBuffer->getFaceMaterialIndexArray(b);
		materialIndexBuffer_dst.resize(a+b);
		for(size_t i=0;i<a;i++)
			materialIndexBuffer_dst[i] = materialIndexBuffer_src_A[i];
		for(size_t i=0;i<b;i++)
			materialIndexBuffer_dst[a+i] = materialIndexBuffer_src_B[i]+offset;


		this->m_meshBuffer->setVertexTextureCoordinateArray( textureCoordBuffer_dst );
		this->m_meshBuffer->setMaterialArray( materialBuffer_dst );
		this->m_meshBuffer->setFaceMaterialIndexArray( materialIndexBuffer_dst );




		///texture stuff
		int b_rect_size;
		std::vector<std::vector<cv::Point3f> > b_rects = slice->getBoundingRectangles(b_rect_size);;
		bounding_rectangles_3D.insert(bounding_rectangles_3D.end(),b_rects.begin(),b_rects.end());
		this->b_rect_size += b_rect_size;

		std::vector<std::pair<cv::Mat,float> > textures = slice->getTextures();
		this->textures.insert(this->textures.end(),textures.begin(),textures.end());
	}
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::mergeVertex(VertexPtr merge_vert, VertexPtr erase_vert)
{
	merge_vert->m_merged = true;
    if(erase_vert == NULL)
    {
        cout << "Merging NULL Pointer! " << endl;
        return;
    }

    //cout << "merge vertex " << merge_vert << endl;
    //cout << "erase vertex " << erase_vert << endl;
	if(merge_vert->m_position.x != erase_vert->m_position.x || merge_vert->m_position.y != erase_vert->m_position.y || merge_vert->m_position.z != erase_vert->m_position.z)
	{
		//cout << "Vertex missalignment! " << endl;
		float dist_x = merge_vert->m_position.x - erase_vert->m_position.x;
		float dist_y = merge_vert->m_position.y - erase_vert->m_position.y;
		float dist_z = merge_vert->m_position.z - erase_vert->m_position.z;
		float dist = sqrt(dist_x*dist_x + dist_y*dist_y + dist_z*dist_z);
		//cout << "distance " << dist << endl;*/


		if(dist > 0.1)
		{
			cout << "Big Vertex missalignment!!!!! " << endl;
			cout << "distance " << dist << endl;
            this->m_vertices.push_back(erase_vert);
            return;
		}
	}
	size_t old_size = merge_vert->in.size();
	merge_vert->in.resize(old_size + erase_vert->in.size());
	for(size_t i = 0; i < erase_vert->in.size(); i++)
	{
		size_t index = old_size + i;
		if(erase_vert->in[i]->isBorderEdge())
		{
			for(size_t j = 0; j < merge_vert->out.size(); j++)
			{
				if(merge_vert->out[j]->end() == erase_vert->in[i]->start())
				{
					erase_vert->in[i]->setPair(merge_vert->out[j]);
					merge_vert->out[j]->setPair(erase_vert->in[i]);
				}
			}
		}
		merge_vert->in[index] = erase_vert->in[i];
		erase_vert->in[i]->setEnd(merge_vert);
	}
    old_size = merge_vert->out.size();
	merge_vert->out.resize(old_size + erase_vert->out.size());
	for(size_t i = 0; i < erase_vert->out.size(); i++)
	{
		size_t index = old_size + i;
		merge_vert->out[index] = erase_vert->out[i];
		erase_vert->out[i]->setStart(merge_vert);
	}
	merge_vert->m_fused = false;
	delete erase_vert;
    erase_vert = NULL;
}

template<typename VertexT, typename NormalT>
HalfEdgeKinFuMesh<VertexT, NormalT>::~HalfEdgeKinFuMesh()
{

}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::setFusionVertex(uint v)
{
	auto vertice = this->m_vertices[v];
	vertice->m_fused = true;
	vertice->m_actIndex = v;
	m_fusionVertices.push_back(vertice);
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::setFusionNeighborVertex(uint v)
{
	auto vertice = this->m_vertices[v];
	vertice->m_fusedNeighbor = true;
	vertice->m_actIndex = v;
	m_oldFusionVertices.push_back(vertice);
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::setOldFusionVertex(uint v)
{
	auto vertice = this->m_vertices[v];
	if(!vertice->m_oldFused)
	{
		vertice->m_oldFused = true;
		vertice->m_actIndex = v;
		m_fusionNeighbors++;
	}
}

template<typename VertexT, typename NormalT>
int HalfEdgeKinFuMesh<VertexT, NormalT>::regionGrowing(FacePtr start_face, NormalT &normal, float &angle, RegionPtr region, vector<FacePtr> &leafs, unsigned int depth)
{
    //Mark face as used
    start_face->m_used = true;

    //Add face to region
    region->addFace(start_face);

    int neighbor_cnt = 0;

    //Get the unmarked neighbor faces and start the recursion
    try
    {
        for(int k = 0; k < 3; k++)
        {
            if((*start_face)[k]->pair()->face() != 0 && (*start_face)[k]->pair()->face()->m_used == false
                    && fabs((*start_face)[k]->pair()->face()->getFaceNormal() * normal) > angle )
            {
				if(start_face->m_fusion_face)
				{
					region->m_unfinished = true;
                }
                if(depth == 0)
                {
                    // if the maximum recursion depth is reached save the child faces to restart the recursion from
                    leafs.push_back((*start_face)[k]->pair()->face());
                }
                else
                {
                    // start the recursion
                    ++neighbor_cnt += regionGrowing((*start_face)[k]->pair()->face(), normal, angle, region, leafs, depth - 1);
                }
            }
        }
    }
    catch (HalfEdgeAccessException )
    {
        // Just ignore access to invalid elements
        //cout << "HalfEdgeMesh::regionGrowing(): " << e.what() << endl;
    }


    return neighbor_cnt;
}


template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT, NormalT>::optimizePlanes(
        int iterations,
        float angle,
        int min_region_size,
        int small_region_size,
        bool remove_flickering)
{
    cout << timestamp << "Starting plane optimization with threshold " << angle << endl;
    cout << timestamp << "Number of faces before optimization: " << this->m_faces.size() << endl;

    // Magic numbers
    int default_region_threshold = (int) 10 * log(this->m_faces.size());

    int region_size   = 0;
    int region_number = 0;
    this->m_regions.clear();
    for(int j = 0; j < iterations; j++)
    {
        cout << timestamp << "Optimizing planes. Iteration " <<  j + 1 << " / "  << iterations << endl;

        // Reset all used variables
        for(size_t i = 0; i < this->m_faces.size(); i++)
        {
            FacePtr face = this->m_faces[i];
			if((*face)(0)->m_fused || (*face)(1)->m_fused || (*face)(2)->m_fused)
			{
				face->m_fusion_face = true;
				if(j == iterations -1)
				{
					(*face)(0)->m_fused = false;
					(*face)(1)->m_fused = false;
					(*face)(2)->m_fused = false;
				}
			}
			else
				face->m_fusion_face = false;
			if(face->m_invalid)
				face->m_used = true;
			else
				face->m_used = false;
        }

        // Find all regions by regionGrowing with normal criteria
        for(size_t i = 0; i < this->m_faces.size(); i++)
        {
            if(this->m_faces[i]->m_used == false)
            {
                NormalT n = this->m_faces[i]->getFaceNormal();

                Region<VertexT, NormalT>* region = new Region<VertexT, NormalT>(region_number);
                m_garbageRegions.insert(region);
                region_size = this->stackSafeRegionGrowing(this->m_faces[i], n, angle, region) + 1;

                // Fit big regions into the regression plane
                if(region_size > max(min_region_size, default_region_threshold) && !region->m_unfinished)
                {
                    region->regressionPlane();
                }

                if(j == iterations - 1)
                {
                    // Save too small regions with size smaller than small_region_size
                    if (region_size < small_region_size)
                    {
                        region->m_toDelete = true;
                    }

                    // Save pointer to the region
                    this->m_regions.push_back(region);
                    region_number++;

                }
            }
        }
    }

    // Delete too small regions
    cout << timestamp << "Starting to delete small regions" << endl;
    if(small_region_size)
    {
        cout << timestamp << "Deleting small regions" << endl;
        this->deleteRegions();
    }

    //Delete flickering faces
    if(remove_flickering)
    {
   /*     vector<FacePtr> flickerer;
        for(size_t i = 0; i < this->m_faces.size(); i++)
        {
            if(this->m_faces[i]->m_region && this->m_faces[i]->m_region->detectFlicker(this->m_faces[i]))
            {
                flickerer.push_back(this->m_faces[i]);
            }
        }

        while(!flickerer.empty())
        {
            deleteFace(flickerer.back());
            flickerer.pop_back();
        }*/
    }

}


template<typename VertexT, typename NormalT>
HalfEdgeKinFuMesh<VertexT, NormalT>* HalfEdgeKinFuMesh<VertexT, NormalT>::retesselateInHalfEdge(float fusionThreshold , bool textured , int start_texture_index)
{
	std::cout << timestamp << "Retesselate mesh" << std::endl;

    // used Typedef's
    typedef std::vector<size_t>::iterator   intIterator;

    // default colors
    unsigned char r=(unsigned char)255, g=(unsigned char)255, b=(unsigned char)255;

    map<Vertex<uchar>, unsigned int> materialMap;

    // Since all buffer sizes are unknown when retesselating
    // all buffers are instantiated as vectors, to avoid manual reallocation
    std::vector<float> vertexBuffer;
    std::vector<float> normalBuffer;
    std::vector<uchar> colorBuffer;
    std::vector<unsigned int> indexBuffer;
    std::vector<unsigned int> materialIndexBuffer;
    std::vector<Material*> materialBuffer;
    std::vector<float> textureCoordBuffer;

    // Reset used variables. Otherwise the getContours() function might not work quite as expected.
    this->resetUsedFlags();

    // Take all regions that are not in an intersection plane
    std::vector<size_t> nonPlaneRegions;
    // Take all regions that were drawn into an intersection plane
    std::vector<size_t> planeRegions;
    for( size_t i = 0; i < this->m_regions.size(); ++i )
    {
		if(!this->m_regions[i]->m_unfinished)
		{
			if( !this->m_regions[i]->m_inPlane || this->m_regions[i]->m_regionNumber < 0)
			{

				nonPlaneRegions.push_back(i);
			}
			else
			{
				//cout << "plane region ! " << endl;
				planeRegions.push_back(i);
			}
		}
    }
    // keep track of used vertices to avoid doubles.
    map<Vertex<float>, unsigned int> vertexMap;
    Vertex<float> current;
    size_t vertexcount = 0;
    int globalMaterialIndex = 0;
    // Copy all regions that are non in an intersection plane directly to the buffers.
    for( intIterator nonPlane = nonPlaneRegions.begin(); nonPlane != nonPlaneRegions.end(); ++nonPlane )
    {
        size_t iRegion = *nonPlane;
        int surfaceClass = this->m_regions[iRegion]->m_regionNumber;

        // iterate over every face for the region number '*nonPlaneBegin'
        for( size_t i=0; i < this->m_regions[iRegion]->m_faces.size(); ++i )
        {
            size_t iFace=i;
            size_t pos;
            // loop over each vertex for this face
            for( int j=0; j < 3; j++ )
            {
                int iVertex = j;
                current = (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position;
                // look up the current vertex. If it was used before get the position for the indexBuffer.
                if( vertexMap.find(current) != vertexMap.end() )
                {
                    pos = vertexMap[current];
                }
                else
                {
					(*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_oldFused = false;
                    (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_fused = false;
					vertexcount++;
                    pos = vertexBuffer.size() / 3;
                    vertexMap.insert(pair<Vertex<float>, unsigned int>(current, pos));
                    vertexBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.x );
                    vertexBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.y );
                    vertexBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_position.z );

                    if((*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal.length() > 0.0001)
                    {
                    	normalBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[0] );
                    	normalBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[1] );
                    	normalBuffer.push_back( (*this->m_regions[iRegion]->m_faces[iFace])(iVertex)->m_normal[2] );
                    }
                    else
                    {
                    	normalBuffer.push_back((*this->m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[0]);
                    	normalBuffer.push_back((*this->m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[1]);
                    	normalBuffer.push_back((*this->m_regions[iRegion]->m_faces[iFace]).getFaceNormal()[2]);
                    }

                    //TODO: Color Vertex Traits stuff?
                    colorBuffer.push_back( r );
                    colorBuffer.push_back( g );
                    colorBuffer.push_back( b );

                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                    textureCoordBuffer.push_back( 0.0 );
                }

                indexBuffer.push_back( pos );
            }

            // Try to find a material with the same color
			std::map<Vertex<uchar>, unsigned int >::iterator it = materialMap.find(Vertex<uchar>(r, g, b));
			if(it != materialMap.end())
			{
				// If found, put material index into buffer
				//std::cout << "RE-USING MAT" << std::endl;
				unsigned int position = it->second;
				materialIndexBuffer.push_back(position);
			}
			else
			{
				Material* m = new Material;
				m->r = r;
				m->g = g;
				m->b = b;
				m->texture_index = -1;

				// Save material index
				materialBuffer.push_back(m);
				materialIndexBuffer.push_back(globalMaterialIndex);

				materialMap.insert(pair<Vertex<uchar>,unsigned int>(Vertex<uchar>(r,g,b),globalMaterialIndex));

				globalMaterialIndex++;
			}
            this->m_regions[iRegion]->m_faces[iFace]->m_invalid = true;
        }
        //this->m_regions[iRegion]->m_toDelete = true;
    }

    cout << timestamp << "Done copying non planar regions.";

    /*
         Done copying the simple stuff. Now the planes are going to be retesselated
         and the textures are generated if there are textures to generate at all.!
     */

     if( bounding_rectangles_3D.size() > 0 )
		bounding_rectangles_3D.resize(0);
	 if( textures.size() > 0 )
		textures.resize(0);
     vertexcount = 0;
     b_rect_size = 0;
     this->start_texture_index = start_texture_index;
     end_texture_index = start_texture_index;
     int texture_face_counter=0;
     size_t vertexBuffer_plane_start = vertexBuffer.size();

      ///Tells which texture belongs to which material
    map<unsigned int, unsigned int > textureMap;

    for(intIterator planeNr = planeRegions.begin(); planeNr != planeRegions.end(); ++planeNr )
    {
        try
        {
            size_t iRegion = *planeNr;

            int surface_class = this->m_regions[iRegion]->m_regionNumber;
            for( size_t i=0; i < this->m_regions[iRegion]->m_faces.size(); ++i )
			{
				size_t iFace=i;
				(*this->m_regions[iRegion]->m_faces[iFace]).m_invalid = true;
			}

            r = this->m_regionClassifier->r(surface_class);
            g = this->m_regionClassifier->g(surface_class);
            b = this->m_regionClassifier->b(surface_class);

            //textureBuffer.push_back( this->m_regions[iRegion]->m_regionNumber );

            // get the contours for this region
            vector<vector<VertexT> > contours = this->m_regions[iRegion]->getContours(fusionThreshold);

			///added
			vector<cv::Point3f> bounding_rectangle ;
			if(textured){
					bounding_rectangle = getBoundingRectangle(contours[0],this->m_regions[iRegion]->m_normal);
					bounding_rectangles_3D.push_back(bounding_rectangle);
					createInitialTexture(bounding_rectangle,end_texture_index,"",2000);
					b_rect_size+=4;
			}
			///

            // alocate a new texture
            TextureToken<VertexT, NormalT>* t = NULL;

            //retesselate these contours.
            std::vector<float> points;
            std::vector<unsigned int> indices;

            Tesselator<VertexT, NormalT>::getFinalizedTriangles(points, indices, contours);

			unordered_map<size_t, size_t> point_map;
			Vertex<float> current;
			size_t pos;
			for(size_t k = 0; k < points.size(); k+=3)
			{
				float u = 0.0;
				float v = 0.0;

				///added
				if(textured)
					getInitialUV(points[k+0],points[k+1],points[k+2],bounding_rectangle,u,v);
				///

				current = Vertex<float>(points[k], points[k + 1], points[k + 2]);
				auto it = vertexMap.find(current);
				if(it != vertexMap.end())
				{
					pos = vertexMap[current];

					//vertex of non planar region has no UV
					/// added
					if(textured)
					{
						if(pos*3 < vertexBuffer_plane_start)
						{
							textureCoordBuffer[pos*3]=u;
							textureCoordBuffer[pos*3+1]=v;
							textureCoordBuffer[pos*3+2]=0.0;
						}else if(textureCoordBuffer[pos*3] != u || textureCoordBuffer[pos*3+1] != v){
							pos = (vertexBuffer.size() / 3);
							vertexMap.insert(pair<Vertex<float>, unsigned int>(current, pos));
							vertexBuffer.push_back( points[k] );
							vertexBuffer.push_back( points[k + 1]);
							vertexBuffer.push_back( points[k + 2]);

							normalBuffer.push_back( this->m_regions[iRegion]->m_normal[0] );
							normalBuffer.push_back( this->m_regions[iRegion]->m_normal[1] );
							normalBuffer.push_back( this->m_regions[iRegion]->m_normal[2] );

							colorBuffer.push_back( static_cast<unsigned char>(255));
							colorBuffer.push_back( static_cast<unsigned char>(255));
							colorBuffer.push_back( static_cast<unsigned char>(255));

							textureCoordBuffer.push_back( u );
							textureCoordBuffer.push_back( v );
							textureCoordBuffer.push_back( 0 );
						}
					}
					///
				}
				else
				{
				    pos = (vertexBuffer.size() / 3);
				    vertexMap.insert(pair<Vertex<float>, unsigned int>(current, pos));
			        vertexBuffer.push_back( points[k] );
					vertexBuffer.push_back( points[k + 1]);
					vertexBuffer.push_back( points[k + 2]);

				    normalBuffer.push_back( this->m_regions[iRegion]->m_normal[0] );
					normalBuffer.push_back( this->m_regions[iRegion]->m_normal[1] );
					normalBuffer.push_back( this->m_regions[iRegion]->m_normal[2] );

					colorBuffer.push_back( static_cast<unsigned char>(255) );
					colorBuffer.push_back( static_cast<unsigned char>(255) );
					colorBuffer.push_back( static_cast<unsigned char>(255) );

					textureCoordBuffer.push_back( u );
					textureCoordBuffer.push_back( v );
					textureCoordBuffer.push_back(  0 );
				}
				point_map.insert(pair<size_t, size_t >(k/3, pos));
			}

            for(int j=0; j < indices.size(); j+=3)
            {
				auto it_a = point_map.find(indices[j + 0]);
				auto it_b = point_map.find(indices[j + 1]);
				auto it_c = point_map.find(indices[j + 2]);
                int a =  it_a->second;
                int b =  it_b->second;
                int c =  it_c->second;

                if(a != b && b != c && a != c)
                {
                    indexBuffer.push_back( a );
                    indexBuffer.push_back( b );
                    indexBuffer.push_back( c );
                }
            }

            ///added
            if(textured){
				map<unsigned int, unsigned int >::iterator it = textureMap.find(end_texture_index);
				//map<unsigned int, unsigned int >::iterator it = textureMap.find(start_texture_index);
			    if(it == textureMap.end())
			    {
			         //new texture -> create new material
			         Material* m = new Material;
			         m->r = r;
			         m->g = g;
			         m->b = b;
			         //m->texture_index=start_texture_index;
			         m->texture_index = end_texture_index;
			         materialBuffer.push_back(m);
			         for( int j = 0; j < indices.size() / 3; j++ )
			         {
						 texture_face_counter++;
			             materialIndexBuffer.push_back(globalMaterialIndex);
			         }
			         textureMap[end_texture_index] = globalMaterialIndex;
			         //textureMap[start_texture_index] = globalMaterialIndex;
			     }
			     else
			     {
			    	 //Texture already exists -> use old material
			         for( int j = 0; j < indices.size() / 3; j++ )
			         {
			             materialIndexBuffer.push_back(it->second);
			         }
			     }
			     globalMaterialIndex++;
			     end_texture_index++;

			 }
			  ///


			// iterate over every face for the region number '*nonPlaneBegin'
			/*for( size_t i=0; i < this->m_regions[iRegion]->m_faces.size(); ++i )
			{
				deleteFace(this->m_regions[iRegion]->m_faces[i]);
			}*/
			  this->m_regions[iRegion]->m_toDelete = true;

        }
        catch(...)
        {
            cout << timestamp << "Exception during finalization. Skipping triangle." << endl;
        };

    }

    if ( !this->m_meshBuffer )
    {
        this->m_meshBuffer = MeshBufferPtr( new MeshBuffer );
    }

    if(vertexBuffer.size() == 0 )
		return NULL;
    this->m_meshBuffer->setVertexArray( vertexBuffer );
    this->m_meshBuffer->setVertexColorArray( colorBuffer );
    this->m_meshBuffer->setVertexNormalArray( normalBuffer );
    this->m_meshBuffer->setFaceArray( indexBuffer );
    this->m_meshBuffer->setVertexTextureCoordinateArray( textureCoordBuffer );
	this->m_meshBuffer->setMaterialArray( materialBuffer );
	this->m_meshBuffer->setFaceMaterialIndexArray( materialIndexBuffer );
    cout << endl << timestamp << "Done retesselating." << ((textured)? " Done texturizing.": "") <<  endl;

	HalfEdgeKinFuMesh<VertexT, NormalT>* retased_mesh =  new HalfEdgeKinFuMesh(this->m_meshBuffer);
	retased_mesh->m_meshBuffer = this->m_meshBuffer;
	retased_mesh->bounding_rectangles_3D = bounding_rectangles_3D;
	retased_mesh->num_cams = num_cams;
	retased_mesh->textures = textures;
	retased_mesh->end_texture_index = end_texture_index;
	retased_mesh->start_texture_index = start_texture_index;

	size_t count_doubles = 0;
	retased_mesh->m_fusionNeighbors = 0;
	retased_mesh->m_fusion_verts.clear();
    Tesselator<VertexT, NormalT>::clear();
    this->m_meshBuffer = NULL;
    return retased_mesh;
}


template<typename VertexT, typename NormalT>
std::vector<cv::Point3f> HalfEdgeKinFuMesh<VertexT,NormalT>::getBoundingRectangle(std::vector<VertexT> act_contour, NormalT normale)
{
	vector<cv::Point3f> contour;
	for(auto it=act_contour.begin(); it != act_contour.end(); ++it){
		contour.push_back(cv::Point3f((*it)[0],(*it)[1],(*it)[2]));
	}

	std::vector<cv::Point3f> rect;
	rect.resize(4);

	if(contour.size()<3){
		// testloesung
		rect[0] = cv::Point3f(contour[0].x,contour[0].y,contour[0].z);
	    rect[1] = cv::Point3f(contour[0].x,contour[1].y,contour[0].z);
	    rect[2] = cv::Point3f(contour[1].x,contour[1].y,contour[1].z);
	    rect[3] = cv::Point3f(contour[1].x,contour[0].y,contour[1].z);

	    return rect;
	}

	float minArea = FLT_MAX;

	float best_a_min , best_a_max, best_b_min, best_b_max;
	cv::Vec3f best_v1, best_v2;

	//lvr normale manchmal 0
	cv::Vec3f n;
	if(normale[0]==0.0 && normale[1] == 0.0 && normale[2] == 0.0){
		n = (contour[1] - contour[0]).cross(contour[2] - contour[0]);
		if (n[0] < 0)
		{
			n *= -1;
		}
	}else{
		n = cv::Vec3f(normale[0],normale[1],normale[2]);
	}
	cv::normalize(n,n);


	 //store a stuetzvector for the bounding box
	 cv::Vec3f p(contour[0].x,contour[0].y,contour[0].z);

	 //calculate a vector in the plane of the bounding box
	 cv::Vec3f v1 = contour[1] - contour[0];
	 if (v1[0] < 0)
	 {
	      v1 *= -1;
	 }
	 cv::Vec3f v2 = v1.cross(n);

	//determines the resolution of iterative improvement steps
	float delta = M_PI/180.0;

	for(float theta = 0; theta < M_PI / 2; theta += delta)
	    {
	        //rotate the bounding box
	        v1 = v1 * cos(theta) + v2 * sin(theta);
	        v2 = v1.cross(n);

	        //calculate the bounding box
	        float a_min = FLT_MAX, a_max = FLT_MIN, b_min = FLT_MAX, b_max = FLT_MIN;
	        for(size_t c = 0; c < contour.size(); c++)
	        {
	            cv::Vec3f p_local = cv::Vec3f(contour[c]) - p;
	            float a_neu= p_local.dot(v1) * 1.0f/(cv::norm(v1)*cv::norm(v1));
	            float b_neu= p_local.dot(v2) * 1.0f/(cv::norm(v2)*cv::norm(v2));


	            if (a_neu > a_max) a_max = a_neu;
	            if (a_neu < a_min) a_min = a_neu;
	            if (b_neu > b_max) b_max = b_neu;
	            if (b_neu < b_min) b_min = b_neu;
	        }
	        float x = fabs(a_max - a_min);
	        float y = fabs(b_max - b_min);

	        //iterative improvement of the area
	        //sometimes wrong?
	        if(x * y < minArea && x * y > 0.01)
	        {

	            minArea = x * y;
	            //if(minArea < 0.4)
					//std::cout << "Bounding Rectangle short: " << minArea << std::endl;
	            best_a_min = a_min;
	            best_a_max = a_max;
	            best_b_min = b_min;
	            best_b_max = b_max;
	            best_v1 = v1;
	            best_v2 = v2;
	        }
	    }

	    rect[0] = cv::Point3f(p + best_a_min * best_v1 + best_b_min* best_v2);
	    rect[1] = cv::Point3f(p + best_a_min * best_v1 + best_b_max* best_v2);
	    rect[2] = cv::Point3f(p + best_a_max * best_v1 + best_b_max* best_v2);
	    rect[3] = cv::Point3f(p + best_a_max * best_v1 + best_b_min* best_v2);

	return rect;
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::createInitialTexture(std::vector<cv::Point3f> b_rect, int texture_index, const char* output_dir,float pic_size_factor)
{
	//auflösung?

	cv::Mat initial_texture = cv::Mat(cv::norm(b_rect[1]-b_rect[0])*pic_size_factor,cv::norm(b_rect[3]-b_rect[0])*pic_size_factor, CV_8UC3, cvScalar(0.0));

	//first version, more than one texture
	textures.push_back(std::pair<cv::Mat,float>(initial_texture,M_PI/2.0));

	//second version, one textur


		//cv::Size sz1 = texture.size();
		//cv::Size sz2 = initial_texture.size();

		//texture_stats.push_back(pair< size_t, cv::Size >(sz1.width,initial_texture.size()));

		//cv::Mat dst( (sz1.height > sz2.height ? sz1.height : sz2.height), sz1.width+sz2.width, CV_8UC3);
		//cv::Mat left(dst, cv::Rect(0, 0, sz1.width, sz1.height));
		//texture.copyTo(left);
		//cv::Mat right(dst, cv::Rect(sz1.width, 0, sz2.width, sz2.height));
		//initial_texture.copyTo(right);
		//dst.copyTo(texture);


	//cv::imwrite(std::string(output_dir)+"texture_"+std::to_string(texture_index)+".ppm",initial_texture);

}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::getInitialUV(float x,float y,float z,std::vector<cv::Point3f> b_rect,float& u, float& v){

	int stuetzpunkt=1;

	//vektor von oben rechts nach oben links
	cv::Vec3f u3D(b_rect[(stuetzpunkt+1)%4]-b_rect[stuetzpunkt]);
	//vektor von oben rechts nach unten rechts
	cv::Vec3f v3D(b_rect[(stuetzpunkt-1)%4]-b_rect[stuetzpunkt]);

	cv::Vec3f p_local(x-b_rect[stuetzpunkt].x,y-b_rect[stuetzpunkt].y,z-b_rect[stuetzpunkt].z);

	//projeziere p_local auf beide richtungsvektoren

	u= p_local.dot(u3D) * 1.0f/(cv::norm(u3D)*cv::norm(u3D));
	v= p_local.dot(v3D) * 1.0f/(cv::norm(v3D)*cv::norm(v3D));

	if(u<0.0) u=0.0;
	if(v<0.0) v=0.0;
	if(u>1.0) u=1.0;
	if(v>1.0) v=1.0;

}


template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::getInitialUV_b(float x,float y,float z,std::vector<std::vector<cv::Point3f> > b_rects,size_t b_rect_number,float& u, float& v){
	std::cout << "Initial UV for one texture version: " << b_rect_number << std::endl;
	getInitialUV(x,y,z,b_rects[b_rect_number],u,v);

	u = texture_stats[b_rect_number].first / (static_cast<float>(texture.cols)) + u * texture_stats[b_rect_number].second.width/(static_cast<float>(texture.cols));
	v = v * texture_stats[b_rect_number].second.height / (static_cast<float>(texture.rows));

}


template<typename VertexT, typename NormalT>
int HalfEdgeKinFuMesh<VertexT,NormalT>::projectAndMapNewImage(kfusion::ImgPose img_pose, const char* texture_output_dir)
{
	//meshlab doesnt show any colors for obj fileformat
	int num_colored_points = fillNonPlanarColors(img_pose);

	std::cout << timestamp <<  "Colorized " << num_colored_points << " Points." << std::endl;
	 //3) project plane to pic area.
				    				//calc transformation matrix with homography
				    				//warp perspective with trans-mat in the gevin texture file from planeindex
				    			//bislang werden boundingrectangles benutzen für homografieberechnungen
				    			//bislang werden boundingrectangles benutzen für homografieberechnungen

	fillInitialTextures(bounding_rectangles_3D,
			    					   img_pose,++num_cams,
									   texture_output_dir);
	//fillInitialTexture(bounding_rectangles_3D,
						//img_pose,++num_cams,
						//texture_output_dir);

	return bounding_rectangles_3D.size();
}

template<typename VertexT, typename NormalT>
int HalfEdgeKinFuMesh<VertexT,NormalT>::fillNonPlanarColors(kfusion::ImgPose img_pose)
{
	cv::Mat distCoeffs;
	if(img_pose.distortion.rows ==0){
	distCoeffs = cv::Mat(4,1,cv::DataType<float>::type);
					distCoeffs.at<float>(0) = 0.0;
					distCoeffs.at<float>(1) = 0.0;
					distCoeffs.at<float>(2) = 0.0;
					distCoeffs.at<float>(3) = 0.0;
	}else{
		distCoeffs=img_pose.distortion;
	}

	cv::Affine3f pose = img_pose.pose.inv();
	cv::Mat tvec(pose.translation());
	cv::Mat rvec(pose.rvec());
	cv::Mat cam = img_pose.intrinsics;


	size_t lenVertices,lenColors,lenTextureCoords;
	lenVertices=lenColors=lenTextureCoords=0;
	floatArr vertexBuffer = this->m_meshBuffer->getVertexArray(lenVertices);
	ucharArr colorBuffer = this->m_meshBuffer->getVertexColorArray(lenColors);
	floatArr textureCoordBuffer = this->m_meshBuffer->getVertexTextureCoordinateArray(lenTextureCoords);

	std::vector<cv::Point3f> uncolored_points;
	std::vector<unsigned int> uncolored_indices;

	//collect uncolored points
	for(unsigned int i=0;i<lenVertices;i++)
	{
		if(static_cast<unsigned int>(colorBuffer[i*3]) < 255 || static_cast<unsigned int>(colorBuffer[i*3+1]) < 255 || static_cast<unsigned int>(colorBuffer[i*3+2]) < 255)
			continue;
		else if(textureCoordBuffer[i*3]==0.0 && textureCoordBuffer[i*3+1]==0.0){
			uncolored_points.push_back(cv::Point3f(vertexBuffer[i*3],vertexBuffer[i*3+1],vertexBuffer[i*3+2]));
			uncolored_indices.push_back(i);
		}
	}

	//colorize
	std::vector<cv::Point2f> projected_uncolored_points;
	if(uncolored_points.size() == 0)
		return 0;

	cv::projectPoints(uncolored_points,rvec,tvec,cam,distCoeffs,projected_uncolored_points);

	float width = cam.at<float>(0,2);
	float height = cam.at<float>(1,2);

	size_t i=0;
	size_t counter=0;

	for(std::vector<cv::Point2f>::iterator it = projected_uncolored_points.begin(); it != projected_uncolored_points.end() ; ++it )
	{
		if(it->x >= 0 && it->x < width*2 && it->y >= 0 && it->y < height*2)
		{
			cv::Vec3b current_color(img_pose.image.at<cv::Vec3b>(it->y,it->x));

			colorBuffer[uncolored_indices[i]*3] = current_color[2];
			colorBuffer[uncolored_indices[i]*3+1] = current_color[1];
			colorBuffer[uncolored_indices[i]*3+2] = current_color[0];
			counter++;
		}
		i++;
	}

    return counter;
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::fillInitialTextures(std::vector<std::vector<cv::Point3f> > b_rects,
		   kfusion::ImgPose img_pose, int image_number,
		   const char* texture_output_dir)
{

	std::cout << timestamp << "Writing texture from ImagePose " << image_number << std::endl;
	cv::Mat distCoeffs;
	if(img_pose.distortion.rows ==0){
	distCoeffs = cv::Mat(4,1,cv::DataType<float>::type);
					distCoeffs.at<float>(0) = 0.0;
					distCoeffs.at<float>(1) = 0.0;
					distCoeffs.at<float>(2) = 0.0;
					distCoeffs.at<float>(3) = 0.0;
	}else{
		distCoeffs=img_pose.distortion;
	}

	cv::Affine3f pose = img_pose.pose.inv();
	cv::Mat tvec(pose.translation());
	cv::Mat rvec(pose.rvec());
	cv::Mat cam = img_pose.intrinsics;

	cv::Mat rotation(img_pose.pose.rotation());

	cv::Vec3f view_direction(img_pose.pose.rotation() * (cv::Vec3f(0,0,1)));

	//doppelte indizierung zum sortieren
	std::vector<int> sorted_indices;
	sorted_indices.resize(textures.size());
	for(int i=0;i<sorted_indices.size();i++) sorted_indices[i] = i;

	std::sort(sorted_indices.begin(),sorted_indices.end(),sort_indices(textures));
	//sortierung ende


	//for(size_t j=0;j<textures.size();j++){
	for(size_t z=0;z<sorted_indices.size();z++){
		size_t j = sorted_indices[z];
			//für jede region
			//project plane i to pic j -> cam_points
			//rvec tvec
		cv::Vec3f b_rect_x(b_rects[j][1] - b_rects[j][0]);
		cv::Vec3f b_rect_y(b_rects[j][3] - b_rects[j][0]);
		cv::Vec3f normal_current(cv::normalize(b_rect_x.cross(b_rect_y)));

		float angle_diff_current = acos(view_direction.dot(normal_current));
		if(angle_diff_current > M_PI/2.0 ) angle_diff_current -= M_PI;


				std::vector<cv::Point2f> image_points2D_br;
				std::vector<cv::Point3f> object_points_br(b_rects[j]);

				cv::projectPoints(object_points_br,rvec,tvec,cam,distCoeffs,image_points2D_br);

				//bounding rects ansatz
				std::vector<cv::Point2f> local_b_rect; //vorher: 00 01 11 10
								local_b_rect.push_back(cv::Point2f(0,0));
								local_b_rect.push_back(cv::Point2f(0,textures[j].first.rows));
								local_b_rect.push_back(cv::Point2f(textures[j].first.cols,textures[j].first.rows));
								local_b_rect.push_back(cv::Point2f(textures[j].first.cols,0));

				//cv::Mat H = cv::findHomography(image_points2D_br ,local_b_rect,CV_RANSAC );
				cv::Mat H = cv::getPerspectiveTransform(image_points2D_br,local_b_rect);



				cv::Mat texture_current = cv::Mat(textures[j].first.size(), CV_8UC3, cvScalar(0.0));
				cv::warpPerspective(img_pose.image,texture_current,H,textures[j].first.size(),cv::INTER_NEAREST | cv::BORDER_CONSTANT);
				//neue texture dst fuellt schwarze stellen der bisherigen textur -> dst hinter alte textur


				if(texture_current.size()==textures[j].first.size()){

					//cv::Mat shadow_mask(texture_current.size(), CV_8U, cv::Scalar(255.0));



					//angle calculation
					if(fabs(angle_diff_current) >= textures[j].second){
						//wenn bisheriger winkelunterschied kleiner gleich der aktuelle
						firstBehindSecondImage(texture_current,textures[j].first.clone(),textures[j].first);

						//cv::imwrite("shadow_mask_"+std::to_string(start_texture_index+j)+".jpg",shadow_mask);
					}else{
						//wenn bisheriger winkelunterschied groesser als der aktuelle
						firstBehindSecondImage(textures[j].first.clone(),texture_current,textures[j].first);
						textures[j].second = fabs(angle_diff_current);

						//cv::imwrite("shadow_mask_"+std::to_string(start_texture_index+j)+".jpg",shadow_mask);
					}

					//shadow magic

					cv::Point* pointarr = new cv::Point[image_points2D_br.size()];
					for(int i=0;i<image_points2D_br.size();i++)
						pointarr[i]=cv::Point(image_points2D_br[i].x,image_points2D_br[i].y);

					//fillImageWithBlackPolygon( img_pose.image , pointarr, (int)image_points2D_br.size());

				}
			}

		//saving
		for(int i=0;i<textures.size();i++)
			cv::imwrite(string(texture_output_dir)+"texture_"+std::to_string(start_texture_index+i)+".jpg",textures[i].first);

}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::fillImageWithBlackPolygon( cv::Mat& img , cv::Point* pointarr, int size)
{
  int lineType = 8;

  const cv::Point* ppt[1] = { pointarr };
  int npt[] = { size };
  cv::fillPoly( img,
            ppt,
            npt,
            1,
            cv::Scalar( 0, 0, 0 ),
            lineType );
 }

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::firstBehindSecondImage(cv::Mat first, cv::Mat second, cv::Mat& dst){
	cv::Mat gray_second;

	cvtColor(second,gray_second,CV_RGB2GRAY);
	cv::Mat mask_bin_second;
	//mask_bin: bisherige textur schwarz oder false
	cv::threshold(gray_second,mask_bin_second,1,255,cv::THRESH_BINARY_INV);

	//neue textur nur an stellen wo mask_bin true ist
	//cv::bitwise_or(first,second,dst,mask_bin_second);
	first.copyTo(second,mask_bin_second);
	second.copyTo(dst);
}

template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::firstBehindSecondImage(cv::Mat first, cv::Mat second, cv::Mat& dst, cv::Mat mask){
	cv::Mat gray_second;

	cvtColor(second,gray_second,CV_RGB2GRAY);
	cv::Mat mask_bin_second;
	//mask_bin: bisherige textur schwarz oder false
	cv::threshold(gray_second,mask_bin_second,1,255,cv::THRESH_BINARY_INV);

	//neue textur nur an stellen wo mask_bin true ist
	//cv::bitwise_or(first,second,dst,mask_bin_second);
	first.copyTo(second,mask_bin_second);
	second.copyTo(dst,mask);
}

template<typename VertexT, typename NormalT>
cv::Rect HalfEdgeKinFuMesh<VertexT,NormalT>::calcCvRect(std::vector<cv::Point2f> rect){
	float top=FLT_MAX,left=FLT_MAX;
	float width=cv::norm(rect[0]-rect[3]);
	float height=cv::norm(rect[0]-rect[1]);
	for(size_t i = 0; i< rect.size() ; i++){
		if(rect[i].x < left) left=rect[i].x;
		if(rect[i].y < top) top=rect[i].y;
	}

	return cv::Rect(left,top,width,height);
}

template<typename VertexT, typename NormalT>
std::pair<int, std::vector<int> > HalfEdgeKinFuMesh<VertexT,NormalT>::calcShadowTupel(std::vector<cv::Point3f> base_rect, std::vector<cv::Point3f> shadow_rect, int shadow_rect_index){
	std::vector<int> point_inlier_indices;

	return std::pair<int, std::vector<int> >(shadow_rect_index ,point_inlier_indices);
}

template<typename VertexT, typename NormalT>
bool HalfEdgeKinFuMesh<VertexT,NormalT>::calcShadowInliers(std::vector<cv::Point3f> base_rect, std::vector<cv::Point3f> shadow_rect, std::vector<int>& inliers){


	return true;
}


template<typename VertexT, typename NormalT>
void HalfEdgeKinFuMesh<VertexT,NormalT>::fillInitialTexture(std::vector<std::vector<cv::Point3f> > b_rects,
		   kfusion::ImgPose img_pose, int image_number,
		   const char* texture_output_dir)
{

	cv::Mat distCoeffs(4,1,cv::DataType<float>::type);
					distCoeffs.at<float>(0) = 0.042855;
					distCoeffs.at<float>(1) = -0.110419;
					distCoeffs.at<float>(2) = 0.004809;
					distCoeffs.at<float>(3) = 0.000398;

	cv::Affine3f pose = img_pose.pose.inv();
	cv::Mat tvec(pose.translation());


	cv::Mat rvec(pose.rvec());
	cv::Mat cam = img_pose.intrinsics;

	std::cout << "Writing texture from ImagePose " << image_number << std::endl;


	for(size_t j=0;j<texture_stats.size();j++){
		std::vector<cv::Point2f> image_points2D_br;
		std::vector<cv::Point3f> object_points_br(b_rects[j]);

		cv::projectPoints(object_points_br,rvec,tvec,cam,distCoeffs,image_points2D_br);

		//bounding rects ansatz
		std::vector<cv::Point2f> local_b_rect; //vorher: 00 01 11 10
								local_b_rect.push_back(cv::Point2f(0,0));
								local_b_rect.push_back(cv::Point2f(0,texture_stats[j].second.height));
								local_b_rect.push_back(cv::Point2f(texture_stats[j].second.width,texture_stats[j].second.height));
								local_b_rect.push_back(cv::Point2f(texture_stats[j].second.width,0));

		cv::Mat H = cv::getPerspectiveTransform(image_points2D_br,local_b_rect);

		cv::Mat dst ;
		cv::warpPerspective(img_pose.image,dst,H,texture_stats[j].second,cv::INTER_NEAREST | cv::BORDER_CONSTANT);

		if(dst.size()==texture_stats[j].second){
			cv::Mat gray_before,gray_now;

			cv::Mat texture_before = texture(cv::Rect(texture_stats[j].first,0,texture_stats[j].second.width,texture_stats[j].second.height));

			cvtColor(texture_before,gray_before,CV_RGB2GRAY);
			cvtColor(dst,gray_now,CV_RGB2GRAY);
			cv::Mat mask_bin_before,mask_bin_now;
			//mask_bin: bisherige textur schwarz oder false
			cv::threshold(gray_before,mask_bin_before,1,255,cv::THRESH_BINARY_INV);

			cv::bitwise_or(dst,texture_before,texture_before,mask_bin_before);

			texture_before.copyTo(texture(cv::Rect(texture_stats[j].first,0,texture_stats[j].second.width,texture_stats[j].second.height)));
		}
	}
	cv::imwrite(string(texture_output_dir)+"texture_"+std::to_string(start_texture_index)+".jpg",texture);
}

template<typename VertexT, typename NormalT>
std::vector<std::vector<cv::Point3f> > HalfEdgeKinFuMesh<VertexT,NormalT>::getBoundingRectangles(int& size){
	size = b_rect_size;
	return bounding_rectangles_3D;
}

template<typename VertexT, typename NormalT>
std::vector<std::pair<cv::Mat,float> > HalfEdgeKinFuMesh<VertexT,NormalT>::getTextures(){
	return textures;
}

} // namespace lvr
