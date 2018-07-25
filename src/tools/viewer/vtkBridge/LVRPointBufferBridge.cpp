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
 * LVRPointBufferBridge.cpp
 *
 *  @date Feb 6, 2014
 *  @author Thomas Wiemann
 */
#include "LVRPointBufferBridge.hpp"
#include "LVRModelBridge.hpp"

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkCellArray.h>
#include <vtkPolyDataMapper.h>
#include <vtkPoints.h>
#include <vtkActor.h>
#include <vtkProperty.h>
#include <vtkPointData.h>

namespace lvr
{

LVRPointBufferBridge::LVRPointBufferBridge(PointBufferPtr pointCloud)
{
    // default: visible light
    m_SpectralChannels[0] = (612 - 400) / 4;
    m_SpectralChannels[1] = (552 - 400) / 4;
    m_SpectralChannels[2] = (462 - 400) / 4;

    // default: solid color gradient
    m_useGradient = false;
    m_useNormalizedGradient = false;
    m_SpectralGradientChannel = 0;
    m_SpectralGradient = SOLID;

    if(pointCloud)
    {
        // Save pc data
        m_pointBuffer = pointCloud;

        // Generate vtk actor representation
        computePointCloudActor(pointCloud);

        // Save meta information
        size_t numColors(0), numNormals(0);
        m_numPoints = pointCloud->getNumPoints();
        pointCloud->getPointNormalArray(numNormals);
        pointCloud->getPointColorArray(numColors);

        if(numColors > 0) m_hasColors = true;
        if(numNormals > 0) m_hasNormals = true;
    }
    else
    {
        m_numPoints = 0;
        m_hasNormals = false;
        m_hasColors = false;
    }
}

void LVRPointBufferBridge::setSpectralChannels(size_t r_channel, size_t g_channel, size_t b_channel)
{
    size_t n, n_channels;
    ucharArr spec = m_pointBuffer->getPointSpectralChannelsArray(n, n_channels);

    if (!n)
    {
        return;
    }

    m_SpectralChannels[0] = std::min((size_t)r_channel, n_channels - 1);
    m_SpectralChannels[1] = std::min((size_t)g_channel, n_channels - 1);
    m_SpectralChannels[2] = std::min((size_t)b_channel, n_channels - 1);

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetName("Colors");

    for (int i = 0; i < n; i++)
    {
        int specIndex = n_channels * i;
        unsigned char speccolor[3];
        speccolor[0] = spec[specIndex + m_SpectralChannels[0]];
        speccolor[1] = spec[specIndex + m_SpectralChannels[1]];
        speccolor[2] = spec[specIndex + m_SpectralChannels[2]];

#if VTK_MAJOR_VERSION < 7
        scalars->InsertNextTupleValue(speccolor);
#else
        scalars->InsertNextTypedTuple(speccolor);
#endif
    }

    m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetScalars(scalars);
}

void LVRPointBufferBridge::getSpectralChannels(size_t &r_channel, size_t &g_channel, size_t &b_channel) const
{
    r_channel = m_SpectralChannels[0];
    g_channel = m_SpectralChannels[1];
    b_channel = m_SpectralChannels[2];
}

void LVRPointBufferBridge::setSpectralColorGradient(GradientType gradient, size_t channel, bool normalized)
{
    size_t n, n_channels;
    ucharArr spec = m_pointBuffer->getPointSpectralChannelsArray(n, n_channels);

    if (!n)
    {
        return;
    }

    m_SpectralGradient = gradient;
    m_SpectralGradientChannel = channel;

    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetName("Colors");

    unsigned char max_val = spec[m_SpectralGradientChannel], min_val = spec[m_SpectralGradientChannel];

    ColorMap colorMap(255);
    if(normalized)
    {
        #pragma omp parallel for reduction(max : max_val), reduction(min : min_val)
        for (int i = 0; i < n; i++)
        {
            int specIndex = n_channels * i + m_SpectralGradientChannel;
            if(spec[specIndex] > max_val)
            {
                max_val = spec[specIndex];  
            }
            if(spec[specIndex] < min_val)
            {
                min_val = spec[specIndex];  
            }
        }
        colorMap = ColorMap(max_val - min_val);
    }

    for (int i = 0; i < n; i++)
    {
        int specIndex = n_channels * i;
        float color[3];

        if(normalized)
            colorMap.getColor(color, spec[specIndex + m_SpectralGradientChannel] - min_val, m_SpectralGradient);
        else
            colorMap.getColor(color, spec[specIndex + m_SpectralGradientChannel], m_SpectralGradient);

        unsigned char speccolor[3];
        speccolor[0] = color[0] * 255;
        speccolor[1] = color[1] * 255;
        speccolor[2] = color[2] * 255;

#if VTK_MAJOR_VERSION < 7
        scalars->InsertNextTupleValue(speccolor);
#else
        scalars->InsertNextTypedTuple(speccolor);
#endif
    }

    m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetScalars(scalars);
}

void LVRPointBufferBridge::getSpectralColorGradient(GradientType &gradient, size_t &channel, bool &normalized) const
{
    gradient = m_SpectralGradient;
    channel = m_SpectralGradientChannel;
    normalized = m_useNormalizedGradient;
}

void LVRPointBufferBridge::useGradient(bool useGradient)
{
    m_useGradient = useGradient;

    // update
    if(useGradient)
    {
        setSpectralColorGradient(m_SpectralGradient, m_SpectralGradientChannel, m_useNormalizedGradient);
    }
    else
    {
        setSpectralChannels(m_SpectralChannels[0], m_SpectralChannels[1], m_SpectralChannels[2]);
    }
}

PointBufferPtr LVRPointBufferBridge::getPointBuffer()
{
    return m_pointBuffer;
}

size_t  LVRPointBufferBridge::getNumPoints()
{
    return m_numPoints;
}

bool LVRPointBufferBridge::hasNormals()
{
    return m_hasNormals;
}

bool LVRPointBufferBridge::hasColors()
{
    return m_hasColors;
}

LVRPointBufferBridge::~LVRPointBufferBridge()
{
}

void LVRPointBufferBridge::computePointCloudActor(PointBufferPtr pc)
{
    if(pc)
    {
        m_pointCloudActor = vtkSmartPointer<vtkActor>::New();

        // Setup a poly data object
        vtkSmartPointer<vtkPolyData>    vtk_polyData = vtkSmartPointer<vtkPolyData>::New();
        vtkSmartPointer<vtkPoints>      vtk_points = vtkSmartPointer<vtkPoints>::New();
        vtkSmartPointer<vtkCellArray>   vtk_cells = vtkSmartPointer<vtkCellArray>::New();

        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");

        double point[3];
        size_t n, n_c, n_s_p, n_s_channels;
        floatArr points = pc->getPointArray(n);
        ucharArr colors = pc->getPointColorArray(n_c);
        ucharArr spec = pc->getPointSpectralChannelsArray(n_s_p, n_s_channels);

        for(vtkIdType i = 0; i < n; i++)
        {
        	int index = 3 * i;
            point[0] = points[index    ];
            point[1] = points[index + 1];
            point[2] = points[index + 2];

            if(n_s_p)
            {
                if (i >= n_s_p) // only take points with spectral information
                {
                    break;
                }
                int specIndex = n_s_channels * i;
                unsigned char speccolor[3];
            	speccolor[0] = spec[specIndex + m_SpectralChannels[0]];
            	speccolor[1] = spec[specIndex + m_SpectralChannels[1]];
            	speccolor[2] = spec[specIndex + m_SpectralChannels[2]];  

#if VTK_MAJOR_VERSION < 7
                scalars->InsertNextTupleValue(speccolor);
#else
	            scalars->InsertNextTypedTuple(speccolor);
#endif
            }
            else if(n_c)
            {
            	unsigned char color[3];
            	color[0] = colors[index];
            	color[1] = colors[index + 1];
            	color[2] = colors[index + 2];

#if VTK_MAJOR_VERSION < 7
                scalars->InsertNextTupleValue(color);
#else
	        scalars->InsertNextTypedTuple(color);
#endif
            }

            vtk_points->InsertNextPoint(point);
            vtk_cells->InsertNextCell(1, &i);
        }

        vtk_polyData->SetPoints(vtk_points);
        vtk_polyData->SetVerts(vtk_cells);

        if(n_c || n_s_p)
        {
        	vtk_polyData->GetPointData()->SetScalars(scalars);
        }

        // Create poly data mapper and generate actor
        //vtkSmartPointer<vtkPolyDataMapper> mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        vtkPolyDataMapper* mapper = vtkPolyDataMapper::New();
#ifdef LVR_USE_VTK5
        mapper->SetInput(vtk_polyData);
#else
        mapper->SetInputData(vtk_polyData);
#endif
        m_pointCloudActor->SetMapper(mapper);
    }
}

LVRPointBufferBridge::LVRPointBufferBridge(const LVRPointBufferBridge& b)
{
    m_pointCloudActor   = b.m_pointCloudActor;
    m_hasColors         = b.m_hasColors;
    m_hasNormals        = b.m_hasNormals;
    m_numPoints         = b.m_numPoints;
    memcpy(m_SpectralChannels, b.m_SpectralChannels, sizeof(b.m_SpectralChannels));
    m_useGradient       = b.m_useGradient;
    m_useNormalizedGradient = b.m_useNormalizedGradient;
    m_SpectralGradient = b.m_SpectralGradient;
    m_SpectralGradientChannel = b.m_SpectralGradientChannel;
}

void LVRPointBufferBridge::setBaseColor(float r, float g, float b)
{
	vtkSmartPointer<vtkProperty> p = m_pointCloudActor->GetProperty();
    p->SetColor(r, g, b);
    m_pointCloudActor->SetProperty(p);
}

void LVRPointBufferBridge::setPointSize(int pointSize)
{
	vtkSmartPointer<vtkProperty> p = m_pointCloudActor->GetProperty();
    p->SetPointSize(pointSize);
    m_pointCloudActor->SetProperty(p);
}

void LVRPointBufferBridge::setOpacity(float opacityValue)
{
	vtkSmartPointer<vtkProperty> p = m_pointCloudActor->GetProperty();
    p->SetOpacity(opacityValue);
    m_pointCloudActor->SetProperty(p);
}

void LVRPointBufferBridge::setVisibility(bool visible)
{
    if(visible) m_pointCloudActor->VisibilityOn();
    else m_pointCloudActor->VisibilityOff();
}

vtkSmartPointer<vtkActor> LVRPointBufferBridge::getPointCloudActor()
{
    return m_pointCloudActor;
}


} /* namespace lvr */
