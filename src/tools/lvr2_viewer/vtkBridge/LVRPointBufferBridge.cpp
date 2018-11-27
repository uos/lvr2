/**
 * Copyright (c) 2018, University Osnabrück
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the University Osnabrück nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL University Osnabrück BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

#include <lvr2/util/Util.hpp>

namespace lvr2
{

inline unsigned char floatToColor(float f)
{
    return f * 255;
}

LVRPointBufferBridge::LVRPointBufferBridge(PointBufferPtr pointCloud)
{
    // use all silders with channel 0
    m_useSpectralChannel.r = true;
    m_useSpectralChannel.g = true;
    m_useSpectralChannel.b = true;
    m_spectralChannels.r = 0;
    m_spectralChannels.g = 0;
    m_spectralChannels.b = 0;
    m_useGradient = false;
    m_useNDVI = false;
    m_useNormalizedGradient = false;
    m_spectralGradientChannel = 0;
    m_spectralGradient = HOT; // default gradientype: HOT

    m_numPoints = 0;
    m_hasNormals = false;
    m_hasColors = false;

    if(pointCloud)
    {
        // Save pc data
        m_pointBuffer = pointCloud;

        // default: visible light
        m_spectralChannels.r = Util::getSpectralChannel(612, pointCloud);
        m_spectralChannels.g = Util::getSpectralChannel(552, pointCloud);
        m_spectralChannels.b = Util::getSpectralChannel(462, pointCloud);

        // Generate vtk actor representation
        computePointCloudActor(pointCloud);

        // Save meta information
        m_numPoints = pointCloud->numPoints();

        if(pointCloud->hasColors()) m_hasColors = true;
        if(pointCloud->hasNormals()) m_hasNormals = true;
    }
}

template <typename T>
bool color_equal(const color<T> &col1, const color<T> &col2)
{
    return col1.r == col2.r && col1.g == col2.g && col1.b == col2.b;
}

void LVRPointBufferBridge::setSpectralChannels(color<size_t> channels, color<bool> use_channel)
{
    // do not update if nothing has changed
    if (color_equal(channels, m_spectralChannels) && color_equal(use_channel, m_useSpectralChannel))
    {
        return;
    }

    // set new values
    m_spectralChannels = channels;
    m_useSpectralChannel = use_channel;

    // update the view
    refreshSpectralChannel();
}


void LVRPointBufferBridge::refreshSpectralChannel()
{
    size_t n;
    unsigned n_channels;
    floatArr spec = m_pointBuffer->getFloatArray("spectral_channels", n, n_channels);

    // check if we have spectral data
    if (!spec)
    {
        return;
    }

    // create colorbuffer
    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetName("Colors");
    scalars->SetNumberOfTuples(n);

    #pragma omp parallel for
    for (vtkIdType i = 0; i < n; i++)
    {
        int specIndex = n_channels * i;
        unsigned char speccolor[3];
        // if the silder is not enabled the color get the value 0
        speccolor[0] = m_useSpectralChannel.r ? floatToColor(spec[specIndex + m_spectralChannels.r]) : 0;
        speccolor[1] = m_useSpectralChannel.g ? floatToColor(spec[specIndex + m_spectralChannels.g]) : 0;
        speccolor[2] = m_useSpectralChannel.b ? floatToColor(spec[specIndex + m_spectralChannels.b]) : 0;

#if VTK_MAJOR_VERSION < 7
        scalars->SetTupleValue(i, speccolor);
#else
        scalars->SetTypedTuple(i, speccolor); // no idea how the new method is called
#endif
    }

    // set new colors
    m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetScalars(scalars);
}

void LVRPointBufferBridge::getSpectralChannels(color<size_t> &channels, color<bool> &use_channel) const
{
    channels = m_spectralChannels;
    use_channel = m_useSpectralChannel;
}

void LVRPointBufferBridge::setSpectralColorGradient(GradientType gradient, size_t channel, bool normalized, bool useNDVI)
{
    // do not update if nothing has changed
    if (m_spectralGradient == gradient && m_spectralGradientChannel == channel
        && m_useNormalizedGradient == normalized && m_useNDVI == useNDVI)
    {
        return;
    }

    // set new values
    m_spectralGradient = gradient;
    m_spectralGradientChannel = channel;
    m_useNormalizedGradient = normalized;
    m_useNDVI = useNDVI;

    // update the view
    refreshSpectralGradient();
}

void LVRPointBufferBridge::refreshSpectralGradient()
{
    size_t n;
    unsigned n_channels;
    floatArr spec = m_pointBuffer->getFloatArray("spectral_channels", n, n_channels);

    // check if we have spectral data
    if (!spec)
    {
        return;
    }

    // calculate the ndvi values
    float ndviMax = 0;
    float ndviMin = 1;

    floatArr ndvi;
    if (m_useNDVI)
    {
        ndvi = floatArr(new float[n]);

        size_t redStart     = Util::getSpectralChannel(400, m_pointBuffer, 0);
        size_t redEnd       = Util::getSpectralChannel(700, m_pointBuffer, 1);
        size_t nearRedStart = Util::getSpectralChannel(700, m_pointBuffer, n_channels - 2);
        size_t nearRedEnd   = Util::getSpectralChannel(1100, m_pointBuffer, n_channels - 1);

        #pragma omp parallel for reduction(max : ndviMax), reduction(min : ndviMin)
        for (int i = 0; i < n; i++)
        {
            float redTotal = 0;
            float nearRedTotal = 0;
            float* specPixel = spec.get() + n_channels * i;

            // sum red and nir
            for (int channel = redStart; channel < redEnd; channel++)
            {
                redTotal += specPixel[channel];
            }
            for (int channel = nearRedStart; channel < nearRedEnd; channel++)
            {
                nearRedTotal += specPixel[channel];
            }

            // use NDVI formula:
            float red = redTotal / (redEnd - redStart);
            float nearRed = nearRedTotal / (nearRedEnd - nearRedStart);

            float val = (nearRed - red) / (nearRed + red);
            val = (val + 1) / 2; // NDVI is in range [-1, 1] => transform to [0, 1]
            ndvi[i] = val;

            // get min and max
            if (val < ndviMin) ndviMin = val;
            if (val > ndviMax) ndviMax = val;
        }
    }

    // create colorbuffer
    vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
    scalars->SetNumberOfComponents(3);
    scalars->SetName("Colors");
    scalars->SetNumberOfTuples(n);

    // normalize data
    unsigned char min = 0;
    unsigned char max = 255;
    if(m_useNormalizedGradient && !m_useNDVI)
    {
        // get min and max
        float max_val = spec[m_spectralGradientChannel], min_val = spec[m_spectralGradientChannel];
        #pragma omp parallel for reduction(max : max_val), reduction(min : min_val)
        for (int i = 0; i < n; i++)
        {
            int specIndex = n_channels * i + m_spectralGradientChannel;
            if(spec[specIndex] > max_val)
            {
                max_val = spec[specIndex];
            }
            if(spec[specIndex] < min_val)
            {
                min_val = spec[specIndex];
            }
        }
        min = floatToColor(min_val);
        max = floatToColor(max_val);
    }

    if(m_useNormalizedGradient && m_useNDVI)
    {
        min = floatToColor(ndviMin);
        max = floatToColor(ndviMax);
    }

    // Colormap is used to calculate gradients
    ColorMap colorMap(max - min);

    // update all colors
	#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        int specIndex = n_channels * i;
        float color[3];

        // get gradient colors
        if (m_useNDVI)
        {
            colorMap.getColor(color, floatToColor(ndvi[i]) - min, m_spectralGradient);
        }
        else
        {
            colorMap.getColor(color, floatToColor(spec[specIndex + m_spectralGradientChannel]) - min, m_spectralGradient);
        }

        unsigned char speccolor[3];
        speccolor[0] = color[0] * 255;
        speccolor[1] = color[1] * 255;
        speccolor[2] = color[2] * 255;

#if VTK_MAJOR_VERSION < 7
        scalars->SetTupleValue(i, speccolor);
#else
        scalars->SetTypedTuple(i, speccolor); // no idea how the new method is called
#endif
    }

    // set new colors
    m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetScalars(scalars);
}

void LVRPointBufferBridge::getSpectralColorGradient(GradientType &gradient, size_t &channel, bool &normalized, bool &useNDVI) const
{
    gradient = m_spectralGradient;
    channel = m_spectralGradientChannel;
    normalized = m_useNormalizedGradient;
    useNDVI = m_useNDVI;
}

void LVRPointBufferBridge::useGradient(bool useGradient)
{
    m_useGradient = useGradient;

    // update the view
    if(useGradient)
    {
        refreshSpectralGradient();
    }
    else
    {
        refreshSpectralChannel();
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
        m_vtk_normals = vtkSmartPointer<vtkDoubleArray>::New();
        m_vtk_normals->SetNumberOfComponents(3);
        m_vtk_normals->SetName("Normals");

        vtkSmartPointer<vtkUnsignedCharArray> scalars = vtkSmartPointer<vtkUnsignedCharArray>::New();
        scalars->SetNumberOfComponents(3);
        scalars->SetName("Colors");

        double point[3];
        double normal[3];
        size_t n, n_c, n_s_p;
        unsigned n_s_channels, w_color;
        n = pc->numPoints();
        n_c = n;

        floatArr points = pc->getPointArray();
        ucharArr colors = pc->getColorArray(w_color);
        floatArr spec = pc->getFloatArray("spectral_channels", n_s_p, n_s_channels);
        floatArr normals = pc->getNormalArray();

        scalars->SetNumberOfTuples(n_s_p ? n_s_p : n);
        vtk_points->SetNumberOfPoints(n_s_p ? n_s_p : n);

        if(normals)
        {
            m_vtk_normals->SetNumberOfTuples(n);
        }
        

        for(vtkIdType i = 0; i < n; i++)
        {
            int index = 3 * i;
            point[0] = points[index    ];
            point[1] = points[index + 1];
            point[2] = points[index + 2];

            if(normals)
            {
                normal[0] = normals[index    ];
                normal[1] = normals[index + 1];
                normal[2] = normals[index + 2];

                m_vtk_normals->SetTuple(i, normal);
            }


            // show spectral colors if we have spectral data
            if(spec)
            {
                if (i >= n_s_p) // only take points with spectral information
                {
                    break;
                }
                int specIndex = n_s_channels * i;
                unsigned char speccolor[3];
                speccolor[0] = floatToColor(spec[specIndex + m_spectralChannels.r]);
                speccolor[1] = floatToColor(spec[specIndex + m_spectralChannels.g]);
                speccolor[2] = floatToColor(spec[specIndex + m_spectralChannels.b]);

#if VTK_MAJOR_VERSION < 7
                scalars->SetTupleValue(i, speccolor);
#else
                scalars->SetTypedTuple(i, speccolor); // no idea how the new method is called
#endif
            }
            else if(colors)
            {
                size_t colorIndex = w_color * i;
                unsigned char color[3];
                color[0] = colors[colorIndex];
                color[1] = colors[colorIndex + 1];
                color[2] = colors[colorIndex + 2];

#if VTK_MAJOR_VERSION < 7
                scalars->SetTupleValue(i, color);
#else
                scalars->SetTypedTuple(i, color); // no idea how the new method is called
#endif
            }
            else
            {
            }

            vtk_points->SetPoint(i, point);
            vtk_cells->InsertNextCell(1, &i);
        }

        vtk_polyData->SetPoints(vtk_points);
        vtk_polyData->SetVerts(vtk_cells);

        if(hasColors() || n_s_p)
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
        m_pointCloudActor->GetProperty()->SetColor(1.0, 1.0, 1.0);
    }
}

LVRPointBufferBridge::LVRPointBufferBridge(const LVRPointBufferBridge& b)
{
    m_pointCloudActor   = b.m_pointCloudActor;
    m_hasColors         = b.m_hasColors;
    m_hasNormals        = b.m_hasNormals;
    m_numPoints         = b.m_numPoints;
    m_spectralChannels  = b.m_spectralChannels;
    m_useSpectralChannel= b.m_useSpectralChannel;
    m_useGradient       = b.m_useGradient;
    m_useNDVI           = b.m_useNDVI;
    m_spectralGradient  = b.m_spectralGradient;
    m_useNormalizedGradient = b.m_useNormalizedGradient;
    m_spectralGradientChannel = b.m_spectralGradientChannel;
}

void LVRPointBufferBridge::setBaseColor(float r, float g, float b)
{
    m_pointCloudActor->GetProperty()->SetColor(r, g, b);
}

void LVRPointBufferBridge::setPointSize(int pointSize)
{
    vtkSmartPointer<vtkProperty> p = m_pointCloudActor->GetProperty();
    p->SetPointSize(pointSize);
    //m_pointCloudActor->SetProperty(p);
}

void LVRPointBufferBridge::setOpacity(float opacityValue)
{
    vtkSmartPointer<vtkProperty> p = m_pointCloudActor->GetProperty();
    p->SetOpacity(opacityValue);
    //m_pointCloudActor->SetProperty(p);
}

void LVRPointBufferBridge::setVisibility(bool visible)
{
    if(visible) m_pointCloudActor->VisibilityOn();
    else m_pointCloudActor->VisibilityOff();
}

void LVRPointBufferBridge::setNormalsVisibility(bool visible)
{
    if(m_hasNormals)
    {
        if(visible)
        {
            m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetNormals(
                m_vtk_normals
            );
        } else {
            m_pointCloudActor->GetMapper()->GetInput()->GetPointData()->SetNormals(
                NULL
            );
        }
    }
    
    
}

vtkSmartPointer<vtkActor> LVRPointBufferBridge::getPointCloudActor()
{
    return m_pointCloudActor;
}


} /* namespace lvr2 */
