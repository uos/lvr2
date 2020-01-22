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


using namespace lvr2;

double getDifference(Transformd a, Transformd b)
{

    double sum = 0;
    for (size_t i = 0; i < 4; i++)
    {
        for (size_t j = 0; j < 4; j++)
        {
            sum += std::abs(a(i,j) - b(i,j));
        }
    }
    return sum;
}  

RegistrationPipeline::RegistrationPipeline(const SLAMOptions* options, ScanProjectEditMarkPtr scans)
{
    m_options = options;
    m_scans = scans;
}


void RegistrationPipeline::doRegistration()
{
    SLAMAlign align(*m_options);
    
    for (size_t i = 0; i < m_scans->project->positions.size(); i++)
    {
        ScanOptional opt = m_scans->project->positions.at(i)->scan;
        if (opt)
        {
            ScanPtr scptr = std::make_shared<Scan>(*opt);
            align.addScan(scptr);
        }
    }
    cout << "Aus doRegistaration: vor finish" << endl;
    align.finish();
    cout << "Aus doRegistaration: nach finish" << endl;



    for (int i = 0; i < m_scans->project->positions.size(); i++)
    {
        // check if the new pos different to old pos
        // ToDo: make num to option

        ScanPositionPtr posPtr = m_scans->project->positions.at(i);

        cout << "Diff: " << getDifference(posPtr->scan->m_registration, align.scan(i)->pose()) << endl;
        
        if ((!m_scans->changed.at(i)) && (getDifference(posPtr->scan->m_registration, align.scan(i)->pose()) > m_options->diffPoseSum))
        {
            m_scans->changed.at(i) = true;
            cout << "New Values"<< endl;
        }
        else
        {
            posPtr->scan->m_registration = align.scan(i)->pose();
        }
        
        cout << "Pose Scan Nummer " << i << endl << posPtr->scan->m_registration << endl;
    }
    cout << "NAch der ersten For-Schleoife" << endl;
    // new align with fix old values 
    SLAMAlign align2(*m_options, m_scans->changed);
    
    for (size_t i = 0; i < m_scans->project->positions.size(); i++)
    {
        ScanOptional opt = m_scans->project->positions.at(i)->scan;
        if (opt)
        {
            ScanPtr scptr = std::make_shared<Scan>(*opt);
            align2.addScan(scptr);
        }
    }

    align2.finish();

    for (int i = 0; i < m_scans->project->positions.size(); i++)
    {
        // check if the new pos different to old pos
        // ToDo: make num to option

        ScanPositionPtr posPtr = m_scans->project->positions.at(i);

        cout << "Diff: " << getDifference(posPtr->scan->m_registration, align2.scan(i)->pose()) << endl;
        if ((!m_scans->changed.at(i)) && m_scans->changed.at(i) == true)
        {
            posPtr->scan->m_registration = align2.scan(i)->pose();
            cout << "Pose Scan Nummer " << i << endl << posPtr->scan->m_registration << endl;
        }
        m_scans->changed.at(i) = true; // ToDo: lsr test 
    }

}