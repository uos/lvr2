
namespace lssr
{

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::Region(int region_number)
{
	this->m_inPlane = false;
	this->m_region_number = region_number;
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::addFace(HFace* f)
{
	this->m_faces.push_back(f);
	f->m_region = this;
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::removeFace(HFace* f)
{
	m_faces.erase(find(m_faces.begin(), m_faces.end(), f));
}


template<typename VertexT, typename NormalT>
vector<vector<HalfEdgeVertex<VertexT, NormalT>* > > Region<VertexT, NormalT>::getContours(float epsilon)
{
	vector<vector<HVertex*> > result;

	//don't try to find contours of a region which wasn't dragged into a plane
	if (!this->m_inPlane) return result;

	for (int i = 0; i<this->m_faces.size(); i++)
	{
		for (int k = 0; k < 3; k++)
		{
			HEdge* current = (*m_faces[i])[k];
			if(!current->used && (current->pair->face == 0 || current->pair->face->m_region != current->face->m_region))
			{
				vector<HalfEdgeVertex<VertexT, NormalT>* > contour;
				Region<VertexT, NormalT>* region = current->face->m_region;

				HEdge* next = 0;
				while(current->used == false)
				{
					//mark edge as used
					current->used = true;
					next = 0;
					//push the next vertex
					contour.push_back(current->end);
					//find next edge
					for(int i = 0; i<current->end->out.size(); i++)
					{
						if(!current->end->out[i]->used
								&& current->end->out[i]->face && current->end->out[i]->face->m_region == region
								&& (current->end->out[i]->pair->face == 0
										||current->end->out[i]->pair->face  && current->end->out[i]->pair->face->m_region != region))

							next = current->end->out[i];
					}

					if(next)
					{
						//calculate direction of the current edge
						NormalT currentDirection(current->end->m_position - current->start->m_position);

						//calculate direction of the next edge
						NormalT nextDirection(next->end->m_position - next->start->m_position);
						//Check if we have to remove the top vertex
						if(    (    fabs(fabs(nextDirection[0]) - fabs(currentDirection[0])) <= epsilon
								&& fabs(fabs(nextDirection[1]) - fabs(currentDirection[1])) <= epsilon
								&& fabs(fabs(nextDirection[2]) - fabs(currentDirection[2])) <= epsilon
						)
						||
						(
								fabs(next->end->m_position[0] - current->end->m_position[0]) <= epsilon
								&& fabs(next->end->m_position[1] - current->end->m_position[1]) <= epsilon
								&& fabs(next->end->m_position[2] - current->end->m_position[2]) <= epsilon
						))
							contour.pop_back();
						current = next;
					}
				}
				result.push_back(contour);
			}
		}
	}

	//move outer contour to the first position
	float xmax = FLT_MIN;
	float ymax = FLT_MIN;
	float zmax = FLT_MIN;

	int outer = -1;
	for(int c=0; c<result.size(); c++)
	{
		for(int v=0; v<result[c].size(); v++)
		{
			if(result[c][v]->m_position.x > xmax)
			{
				xmax = result[c][v]->m_position.x;
				outer = c;
			}
			if(result[c][v]->m_position.y > ymax)
			{
				ymax = result[c][v]->m_position.y;
				outer = c;
			}
			if(result[c][v]->m_position.z > zmax)
			{
				zmax = result[c][v]->m_position.z;
				outer = c;
			}
		}
	}

	if(outer != -1)
	{
		result.insert(result.begin(), result[outer]);
		result.erase(result.begin()+outer+1);
	}
	else
	{
		cerr << "ERROR: could not find outer contour" << endl;
	}

	return result;
}

template<typename VertexT, typename NormalT>
NormalT Region<VertexT, NormalT>::calcNormal()
{
	NormalT result;
    //search for a valid normal of region
	int i = 0;
	do
	{
		result = m_faces[i++]->getFaceNormal();
	}
	while ((result.length() == 0 || isnan(result.length())) && i<m_faces.size());

	result.normalize();
	return result;
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::regressionPlane()
{
//  srand ( time(NULL) );

    VertexT point1;
    VertexT point2;
    VertexT point3;

    //representation of best regression plane by point and normal
    VertexT bestpoint;
    NormalT bestNorm;

    float bestdist = FLT_MAX;
    float dist = 0;

    int iterations = 0;
    int nonimproving_iterations = 0;

    while((nonimproving_iterations < 20) && (iterations < 200))
    {
        //randomly choose 3 disjoint points
        do{
            point1 = (*m_faces[rand() % m_faces.size()])(0)->m_position;
            point2 = (*m_faces[rand() % m_faces.size()])(1)->m_position;
            point3 = (*m_faces[rand() % m_faces.size()])(2)->m_position;
        }while(point1 == point2 || point2 == point3 || point3 == point1);

        //compute normal of the plane given by the 3 points
        NormalT n0 = (point1 - point2).cross(point1 - point3);
        n0.normalize();

        //compute error to at most 50 other randomly chosen points
        dist = 0;
        for(int i=0; i < min(50, (int)m_faces.size()); i++)
        {
            VertexT refpoint = (*m_faces[rand() % m_faces.size()])(0)->m_position;
            dist += fabs(refpoint * n0 - point1 * n0) / min(50, (int)m_faces.size());
        }

        //a new optimum is found
        if(dist < bestdist)
        {
            bestdist = dist;

            bestpoint = point1;
            bestNorm = n0;

            nonimproving_iterations = 0;
        }
        else
            nonimproving_iterations++;

        iterations++;
    }

    //drag points into the regression plane
    for(int i=0; i<m_faces.size(); i++)
    {
        for(int p=0; p<3; p++)
        {
            float v = ((bestpoint - (*m_faces[i])(p)->m_position) * bestNorm) / (bestNorm * bestNorm);
            if(v != 0)
                (*m_faces[i])(p)->m_position = (*m_faces[i])(p)->m_position + (VertexT)bestNorm * v;
        }
    }

    this->m_inPlane = true;
    this->m_normal = calcNormal();
}

template<typename VertexT, typename NormalT>
bool Region<VertexT, NormalT>::detectFlicker(HFace* f)
{
	if(this->m_inPlane)
		if ((VertexT(f->getFaceNormal())+VertexT(this->m_normal)).length() < 0.05)
		{
			return true;
		}
	return false;
}

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::~Region()
{
	for (int i = 0; i<m_faces.size(); i++)
		m_faces[i]->m_region = 0;
	m_faces.clear();
}

}
