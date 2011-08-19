
namespace lssr
{

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::addFace(HFace* f)
{
	this->m_faces.push_back(f);
}

template<typename VertexT, typename NormalT>
vector<stack<HalfEdgeVertex<VertexT, NormalT>* > > Region<VertexT, NormalT>::getContours(float epsilon)
{
	vector<stack<HVertex*> > result;
	//TODO: implement
	return result;
}

template<typename VertexT, typename NormalT>
NormalT Region<VertexT, NormalT>::getNormal()
{
	NormalT result;
	//TODO: implement
	return result;
}

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::~Region()
{
	m_faces.clear();
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
}


}
