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

/**
 * LVRVtkArrow.hpp
 *
 *  @date Feb 21, 2014
 *  @author Thomas Wiemann
 */
#ifndef LVRVTKARROW_HPP_
#define LVRVTKARROW_HPP_

#include <vtkSmartPointer.h>
#include <vtkActor.h>

#include "geometry/Vertex.hpp"

namespace lvr
{

/**
 * @brief   A wrapper class to generate arrow actors for vtk based on
 *          VTK's oriented arrow example.
 */
class LVRVtkArrow
{
public:

    LVRVtkArrow(Vertexf start, Vertexf end);

    vtkSmartPointer<vtkActor>   getArrowActor();
    vtkSmartPointer<vtkActor>   getStartActor();
    vtkSmartPointer<vtkActor>   getEndActor();

    void restoreColor();
    void setTmpColor(double r, double g, double b);

    virtual ~LVRVtkArrow();

private:
    vtkSmartPointer<vtkActor>   m_arrowActor;
    vtkSmartPointer<vtkActor>   m_startActor;
    vtkSmartPointer<vtkActor>   m_endActor;
    Vertexf                     m_start;
    Vertexf                     m_end;
    double                      m_r;
    double                      m_g;
    double                      m_b;

};

} /* namespace lvr */

#endif /* LVRVTKARROW_HPP_ */
