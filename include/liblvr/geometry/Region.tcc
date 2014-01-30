/* Copyright (C) 2011 Uni Osnabrück
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
 * Region.tcc
 *
 *  @date 18.08.2011
 *  @author Kim Rinnewitz (krinnewitz@uos.de)
 *  @author Sven Schalk (sschalk@uos.de)
 *  @author Thomas Wiemann (twiemann@uos.de)
 */
 
#include <limits>

#include <psimpl.h>

namespace lvr
{

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::Region(int regionNumber)
{
	this->m_inPlane      = false;
	this->m_regionNumber = regionNumber;
	this->m_toDelete     = false;
	this->b_labeled      = false;
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::addFace(FacePtr f)
{
	this->m_faces.push_back(f);
	f->m_region = m_regionNumber;
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::deleteInvalidFaces()
{
    typename vector<FacePtr>::iterator it = m_faces.begin();
    while(it != m_faces.end())
    {
        if( (*it)->m_invalid)
        {
            it = m_faces.erase(it);
        }
        else
        {
            it++;
        }
    }
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::setLabel(std::string label)
{
	this->m_label = label;
	this->b_labeled = true;
}

template<typename VertexT, typename NormalT>
bool Region<VertexT, NormalT>::hasLabel()
{
	return this->b_labeled;
}

template<typename VertexT, typename NormalT>
std::string Region<VertexT, NormalT>::getLabel()
{
	return this->m_label;
}

template<typename VertexT, typename NormalT>
vector<vector<VertexT> > Region<VertexT, NormalT>::getContours(float epsilon)
{
    vector<vector<VertexT> > result;
    for (size_t i = 0; i < this->m_faces.size(); i++)
    {
        for (int k = 0; k < 3; k++)
        {
            EdgePtr current = (*m_faces[i])[k];
            if(!current->used && current->isBorderEdge())
            {
                std::deque<float> contour;
                //Region<VertexT, NormalT>* region = this;

                EdgePtr next;
                while(current->used == false)
                {
                    //mark edge as used
                    current->used = true;
                    next = 0;
                    //push the next vertex
                    contour.push_back(current->end()->m_position[0]);
                    contour.push_back(current->end()->m_position[1]);
                    contour.push_back(current->end()->m_position[2]);

                    for(size_t a = 0; a < current->end()->out.size(); a++)
                    {
                        try
                        {
                            if(current->end()->out[a]) // Don't know where the null pointers come from...
                            {
                                long int r1 = -1;
                                long int r2 = -1;

                                if(current->end()->out[a]->hasFace())
                                {
                                    r1 = current->end()->out[a]->face()->m_region;
                                }

                                if(current->end()->out[a]->hasPair() && current->end()->out[a]->pair()->hasFace())
                                {
                                    r2 = current->end()->out[a]->pair()->face()->m_region;
                                }

                                //find next edge
                                if( !((current->end()->out[a])->used)
                                        && current->end()->out[a]->hasFace() && r1 == m_regionNumber
                                        && (!current->end()->out[a]->hasNeighborFace() || r2 == -1
                                                || ( current->end()->out[a]->hasNeighborFace()  && r2 != m_regionNumber )) )
                                {
                                    next = current->end()->out[a];
                                }
                            }
                        }
                        catch(HalfEdgeAccessException &e)
                        {
                            //cout << e.what() << endl;
                        }
                    }

                    if(next)
                    {
                        current = next;
                    }
                }

                // Simplify contour
                vector<float> simple_contour;
                psimpl::simplify_reumann_witkam <3> (
                        contour.begin (), contour.end (),
                        epsilon, std::back_inserter(simple_contour));

                // Convert to VertexT
                vector<VertexT> tmp;
                for(int i = 0; i < simple_contour.size() / 3; i++)
                {
                    if(!tmp.size() || ! (tmp.back() == VertexT(simple_contour[i * 3], simple_contour[i * 3 + 1], simple_contour[i * 3 + 2])))
                    {
                        tmp.push_back(VertexT(simple_contour[i * 3], simple_contour[i * 3 + 1], simple_contour[i * 3 + 2]));
                    }
                }

                // Add contour
                result.push_back(tmp);
            }
        }
    }


    //  //don't try to find contours of a region which wasn't dragged into a plane
    //  if (!this->m_inPlane)
    //    {
    //        return result;
    //    }
    //
    //  for (size_t i = 0; i < this->m_faces.size(); i++)
    //  {
    //      for (int k = 0; k < 3; k++)
    //      {
    //          HEdge* current = (*m_faces[i])[k];
    //          if(!current->used && (current->pair->face == 0 || current->pair->face->m_region != current->face->m_region))
    //          {
    //              vector<VertexT> contour;
    //              //Region<VertexT, NormalT>* region = this;
    //
    //              HEdge* next = 0;
    //              while(current->used == false)
    //              {
    //                  //mark edge as used
    //                  current->used = true;
    //                  next = 0;
    //                  //push the next vertex
    //                  contour.push_back(current->end->m_position);
    //
    //                  //find next edge
    //                  for(size_t i = 0; i<current->end->out.size(); i++)
    //                  {
    //                      if( !current->end->out[i]->used
    //                              && current->end->out[i]->face && current->end->out[i]->face->m_region == this
    //                              && (current->end->out[i]->pair->face == 0
    //                                      || ( current->end->out[i]->pair->face  && current->end->out[i]->pair->face->m_region != this )))
    //                      {
    //                          next = current->end->out[i];
    //                      }
    //                  }
    //
    //                  if(next)
    //                  {
    //                      current = next;
    //                  }
    //              }
    //
    //              for(int kk = 0; kk < 1; kk++)
    //              {
    //                  // delete vertices due to direction
    //                  bool didSomething = true;
    //                  while(didSomething)
    //                  {
    //                      vector<VertexT> toDelete;
    //                      for(int c = 1; c < contour.size()-1; c++)
    //                      {
    //                          //calculate direction of the current edge
    //                          NormalT nextDirection(contour[c+1] - contour[c]);
    //
    //                          //calculate direction of the next edge
    //                          NormalT previousDirection(contour[c] - contour[c-1]);
    //
    //                          if
    //                          (
    //                                  fabs(fabs(previousDirection[0]) - fabs(nextDirection[0])) <= epsilon
    //                               && fabs(fabs(previousDirection[1]) - fabs(nextDirection[1])) <= epsilon
    //                               && fabs(fabs(previousDirection[2]) - fabs(nextDirection[2])) <= epsilon
    //                          )
    //                          {
    //                              toDelete.push_back(contour[c]);
    //                          }
    //                      }
    //                      didSomething = false;
    //                      for(int d = 0; d < toDelete.size(); d++)
    //                      {
    //                          contour.erase(find(contour.begin(), contour.end(), toDelete[d]));
    //                          didSomething = true;
    //                      }
    //                  }
    //                  // delete vertices due to distance
    //                  didSomething = true;
    //                  while(didSomething)
    //                  {
    //                      vector<VertexT> toDelete;
    //                      for(int c = 0; c < contour.size()-1; c++)
    //                      {
    //                          if
    //                          (
    //                                  fabs(contour[c+1][0] - contour[c][0]) <= epsilon
    //                               && fabs(contour[c+1][1] - contour[c][1]) <= epsilon
    //                               && fabs(contour[c+1][2] - contour[c][2]) <= epsilon
    //                          )
    //                          {
    //
    //                              toDelete.push_back(contour[c]);
    //                          }
    //                      }
    //                      didSomething = false;
    //                      for(int d = 0; d < toDelete.size(); d++)
    //                      {
    //                          contour.erase(find(contour.begin(), contour.end(), toDelete[d]));
    //                          didSomething = true;
    //                      }
    //                  }
    //
    //              }
    //              result.push_back(contour);
    //          }
    //      }
    //  }

    //move outer contour to the first position
    float xmax = std::numeric_limits<float>::min();
    float ymax = std::numeric_limits<float>::min();
    float zmax = std::numeric_limits<float>::min();

    int outer = -1;
    for(size_t c = 0; c < result.size(); c++)
    {
        for(size_t v = 0; v < result[c].size(); v++)
        {
            if(result[c][v].x > xmax)
            {
                xmax = result[c][v].x;
                outer = c;
            }
            if(result[c][v].y > ymax)
            {
                ymax = result[c][v].y;
                outer = c;
            }
            if(result[c][v].z > zmax)
            {
                zmax = result[c][v].z;
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
       // cerr << "ERROR: could not find outer contour" << endl;
    }

    return result;


}

template<typename VertexT, typename NormalT>
NormalT Region<VertexT, NormalT>::calcNormal()
{
    NormalT result;
    //search for a valid normal of region
    size_t i = 0;
    do
    {
        result = m_faces[i++]->getFaceNormal();
    }
    while ((result.length() == 0 || isnan(result.length())) && i < m_faces.size());

    result.normalize();

    //Check if this normal is representative for most of the others / it is not a flickering normal
    int fit   = 0;
    int nofit = 0;

    for(size_t i = 0; i < m_faces.size(); i++)
    {
        NormalT comp = m_faces[i]->getFaceNormal();
        comp.normalize();
        if(comp == result)
        {
            fit++;
        }
        else
        {
            if(comp == (result * -1))
            {
                nofit++;
            }
        }
    }

    if(fit>nofit)
    {
        return result;
    }
    else
    {
        return result * -1;
    }
}

template<typename VertexT, typename NormalT>
int Region<VertexT, NormalT>::size()
{
	return this->m_faces.size();
}

template<typename VertexT, typename NormalT>
void Region<VertexT, NormalT>::regressionPlane()
{
//    srand(time(NULL));

    // Quick and dirty fox to avoid hanging when
    // degenrated faces are in buffer
    if(m_faces.size() < 3)
    {
        m_normal = m_faces[0]->getFaceNormal();
        return;
    }

    VertexT point1;
    VertexT point2;
    VertexT point3;

    //representation of best regression plane by point and normal
    VertexT bestpoint;
    NormalT bestNorm;

    float bestdist = std::numeric_limits<float>::max();
    float dist     = 0;

    int iterations              = 0;
    int nonimproving_iterations = 0;

    while((nonimproving_iterations < 30) && (iterations < 200))
    {
    	NormalT n0;
        //randomly choose 3 disjoint points
        do{
            point1 = (*m_faces[rand() % m_faces.size()])(0)->m_position;
            point2 = (*m_faces[rand() % m_faces.size()])(1)->m_position;
            point3 = (*m_faces[rand() % m_faces.size()])(2)->m_position;

            //compute normal of the plane given by the 3 points
            n0 = (point1 - point2).cross(point1 - point3);
            n0.normalize();

        }while(point1 == point2 || point2 == point3 || point3 == point1 || n0.length() == 0);

        //compute error to at most 50 other randomly chosen points
        dist = 0;
        for(int i = 0; i < min(50, (int)m_faces.size()); i++)
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
        {
            nonimproving_iterations++;
        }

        iterations++;
    }

    //drag points into the regression plane
    for(size_t i = 0; i < m_faces.size(); i++)
    {
        for(int p = 0; p < 3; p++)
        {
            float v = ((bestpoint - (*m_faces[i])(p)->m_position) * bestNorm) / (bestNorm * bestNorm);
            if(v != 0)
            {
                (*m_faces[i])(p)->m_position = (*m_faces[i])(p)->m_position + (VertexT)bestNorm * v;
            }
        }
    }
    this->m_inPlane = true;
    this->m_normal = calcNormal();
    this->m_stuetzvektor = point1;
}

template<typename VertexT, typename NormalT>
bool Region<VertexT, NormalT>::detectFlicker(FacePtr f)
{
	if(this->m_inPlane)
	{
		if ((VertexT(f->getFaceNormal()) + VertexT(this->m_normal)).length() < 1.0)
		{
			return true;
		}
	}
	return false;
}

template<typename VertexT, typename NormalT>
Region<VertexT, NormalT>::~Region()
{
	for (size_t i = 0; i < m_faces.size(); i++)
	{
	    if(m_faces[i])
	    {
	        m_faces[i]->m_region = -1;
	    }
	}
	//m_faces.clear();
}

}
