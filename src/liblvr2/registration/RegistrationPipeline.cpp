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

#include "lvr2/registration/RegistrationPipeline.hpp"
#include "lvr2/registration/SLAMAlign.hpp"
#include "lvr2/registration/SLAMScanWrapper.hpp"

#include "vector"
#include <math.h>

/**
 * RegistrationPipeline.cpp
 *
 *  @date Jan 7, 2020
 *  @author Timo Osterkamp (tosterkamp@uni-osnabrueck.de)
 *  @author Wilko Müller
 */

using namespace lvr2;

bool RegistrationPipeline::isToleratedDifference(Transformd a, Transformd b)
{
    Rotationd rotateA = a.block<3, 3>(0, 0);
    Rotationd rotateB = b.block<3, 3>(0, 0);
    Vector4d translateA = a.col(3);
    Vector4d translateB = b.col(3);
    assert(translateA[3] == 1.0);
    assert(translateB[3] == 1.0);

    // calculate translation distance
    double dist = (translateB - translateA).norm();

    // calculate angle difference
    Rotationd bTransposed = rotateB.transpose();
    Rotationd r = rotateA * bTransposed;

    double tmp = (r.trace() / 2.0) - 0.5;

    // fix so there will be no rounding errors, otherwise acos can be nan !!!
    if (tmp < -1.0)
    {
        tmp = -1.0;
    }
    else if (tmp > 1.0)
    {
        tmp = 1.0;
    }

    double angle = std::acos(tmp);

    if(m_options->verbose)
    {
        std::cout << "PoseDiff: " << dist << " ; AngleDiff: " << angle << std::endl;
    }

    return (angle < m_options->diffAngle && dist < m_options->diffPosition);
}

void RegistrationPipeline::rotateAroundYAxis(Transformd *inputMatrix4x4, double angle)
{
    // rotate only the upper left part of inputMatrix4x4
    Eigen::Matrix<double, 3, 3> tmp_mat(inputMatrix4x4->block<3,3>(0,0));
    // create y-vector (in scan coordinates) which can be transformed to world coordinates
    Eigen::Vector3d v(0, 1, 0);
    v = tmp_mat * v;

    double cosA = cos(angle);
    double sinA = sin(angle);

    // create rotation matrix
    Eigen::Matrix<double, 3, 3> mult_mat;
    mult_mat <<     v(0)*v(0)*(1.0-cosA)+cosA, v(0)*v(1)*(1.0-cosA)-v(2)*sinA, v(0)*v(2)*(1.0-cosA)+v(1)*sinA,
                    v(1)*v(0)*(1.0-cosA)+v(2)*sinA, v(1)*v(1)*(1.0-cosA)+cosA, v(1)*v(2)*(1.0-cosA)-v(0)*sinA,
                    v(2)*v(0)*(1.0-cosA)-v(1)*sinA, v(2)*v(1)*(1.0-cosA)+v(0)*sinA, v(2)*v(2)*(1.0-cosA)+cosA;
    tmp_mat = tmp_mat * mult_mat;

    // save result in original Matrix
    inputMatrix4x4[0](0,0) = tmp_mat(0,0);
    inputMatrix4x4[0](0,1) = tmp_mat(0,1);
    inputMatrix4x4[0](0,2) = tmp_mat(0,2);
    inputMatrix4x4[0](1,0) = tmp_mat(1,0);
    inputMatrix4x4[0](1,1) = tmp_mat(1,1);
    inputMatrix4x4[0](1,2) = tmp_mat(1,2);
    inputMatrix4x4[0](2,0) = tmp_mat(2,0);
    inputMatrix4x4[0](2,1) = tmp_mat(2,1);
    inputMatrix4x4[0](2,2) = tmp_mat(2,2);
}

RegistrationPipeline::RegistrationPipeline(const SLAMOptions* options, ScanProjectEditMarkPtr scans)
{
    m_options = options;
    m_scans = scans;
}


void RegistrationPipeline::doRegistration()
{
    // Create SLAMAlign object and add separate scans. The scans are not transferred via the constructor, because then they will not reduced.
    SLAMAlign align(*m_options);
    for (size_t i = 0; i < m_scans->project->positions.size(); i++)
    {
        if(m_scans->project->positions.at(i)->scans.size())
        {
            // if m_options->rotate_angle is not 0 -> all scans will be rotate around y axis
            rotateAroundYAxis(&(m_scans->project->positions[i]->scans[0]->poseEstimation), m_options->rotate_angle * M_PI / 180);

            // the SLAMAlign object needs a scan pointer 
            ScanPtr scptr = std::make_shared<Scan>(*(m_scans->project->positions[i]->scans[0]));
            align.addScan(scptr);
        }
    }

    if (m_options->verbose)
    {
        cout << "start SLAMAlign registration" << endl;
    }

    // start the registration (with params from m_options)
    align.finish();

    if (m_options->verbose)
    {
        cout << "end SLAMAlign registration" << endl;
    }

    // if all values are new, the second registration is not needed 
    bool all_values_new = true;
    for (int i = 0; i < m_scans->project->positions.size(); i++)
    {
        // check if the new pos different to old pos
        ScanPositionPtr posPtr = m_scans->project->positions.at(i);

        if (( !m_scans->changed.at(i)) && 
              !isToleratedDifference(
                  m_scans->project->positions.at(i)->scans[0]->registration,
                  align.scan(i)->pose()))
        {
            m_scans->changed.at(i) = true;
            cout << "New Values"<< endl;
        }
        // new pose of the first scan is same as the old pose
        else if (i != 0)
        {
            all_values_new = false;
        }
    }
    cout << "First registration done" << endl;

    // new align with fix old values only when not all poses new
    if (all_values_new)
    {
        cout << "no new registration" << endl;
    }
    else
    {
        cout << "start new registration with some fix poses" << endl;

        // do the same as above only with the m_scans->changed array wich says which scan is fix
        align = SLAMAlign(*m_options, m_scans->changed);

        for (size_t i = 0; i < m_scans->project->positions.size(); i++)
        {
            if(m_scans->project->positions.at(i)->scans.size())
            {
                ScanPtr scptr = std::make_shared<Scan>(*(m_scans->project->positions[i]->scans[0]));

                align.addScan(scptr);
            }
        }

        align.finish();
    }

    // transfer the calculated poses into the original data
    for (int i = 0; i < m_scans->project->positions.size(); i++)
    {
        ScanPositionPtr posPtr = m_scans->project->positions.at(i);

        if (m_scans->changed.at(i) || all_values_new)
        {
            posPtr->scans[0]->registration = align.scan(i)->pose();
            posPtr->registration = align.scan(i)->pose();
            cout << "Pose Scan Nummer " << i << endl << posPtr->scans[0]->registration << endl;
        }
    }
}
