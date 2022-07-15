#include "lvr2/io/scanio/ScanProjectManager.hpp"
#include "lvr2/util/ScanProjectUtils.hpp"
#include "lvr2/util/Timestamp.hpp"

namespace lvr2
{

ScanProjectManager::ScanProjectManager(ScanProjectPtr scanProject)
{
    // Mark all scan positions as new
    m_changed = std::vector<bool>(scanProject->positions.size(), true);
    m_scanProject = scanProject;
}

void ScanProjectManager::addScanPosition(ScanPositionPtr scanPosition)
{
    m_scanProject->positions.push_back(scanPosition);
    m_changed.push_back(true);
}

void ScanProjectManager::updateScanPosition(ScanPositionPtr position, size_t positionNo)
{
    if(positionNo < m_scanProject->positions.size())
    {
        m_scanProject->positions[positionNo] = position;
        m_changed[positionNo] = true;
    }
}


std::vector<size_t> ScanProjectManager::changedIndices()
{
    std::vector<size_t> changed;
    for(size_t i = 0; i < m_changed.size(); i++)
    {
        if(m_changed[i])
        {
            changed.push_back(i);
        }
    }
    return std::move(changed);
}

void ScanProjectManager::addScan(ScanPtr scan, size_t scanPosition, size_t lidar)
{
    if(scanPosition < m_changed.size())
    {
        if(lidar < m_scanProject->positions[scanPosition]->lidars.size())
        {
            m_scanProject->positions[scanPosition]->lidars[lidar]->scans.push_back(scan);
        }
    }
    std::cout << timestamp << "ScanprojectManager: Unable to add scan at position " 
              << scanPosition << "." << std::endl;
}

 
void ScanProjectManager::addScan( PointBufferPtr buffer, size_t scanPosition, size_t lidar, Transformd transformation)
{
    // Setup new scan
    ScanPtr scan(new Scan);
    scan->points = buffer;
    scan->transformation = transformation;
    
    // Add on corresponding position
    addScan(scan, scanPosition, lidar);
}

void ScanProjectManager::updateScan(ScanPtr scan, size_t scanPositionNo, size_t lidarNo, size_t scanNo)
{
    if(scanPositionNo < m_changed.size())
    {
        if(lidarNo < m_scanProject->positions[scanPositionNo]->lidars.size())
        {
            if(scanNo < m_scanProject->positions[scanPositionNo]->lidars[lidarNo]->scans.size())
            {
                m_scanProject->positions[scanPositionNo]->lidars[lidarNo]->scans[scanNo] = scan;
                m_changed[scanPositionNo] = true;
            }
        }
    }
    std::cout << timestamp << "ScanprojectManager: Unable to add scan at position " << scanPositionNo << "." << std::endl;
}

ScanProjectPtr ScanProjectManager::scanProject()
{
    return m_scanProject;
}

bool ScanProjectManager::changed(size_t scanPosition)
{
    if(scanPosition < m_changed.size())
    {
        return m_changed[scanPosition];
    }
    else
    {
        return false;
    }
}

ScanPositionPtr ScanProjectManager::loadScanPosition(size_t scanPosition, bool loadData)
{
    if(scanPosition < m_scanProject->positions.size())
    {
        m_changed[scanPosition] = false;
        return m_scanProject->positions[scanPosition];
    }
    return nullptr;
}



std::pair<ScanPtr, Transformd>  ScanProjectManager::loadScan(size_t scanPosition, size_t lidar, size_t scanNo)
{
    ScanPtr scan;
    Transformd transform;
    std::tie(scan, transform) = scanFromProject(m_scanProject, scanPosition, lidar, scanNo);
    
    // Check if access was successful and mark
    // that the data was loaded
    if(scan)
    {
        m_changed[scanPosition] = false;
    }

    return std::make_pair(scan, transform);
}

std::pair<ScanPtr, Transformd>  ScanProjectManager::loadNextChangedScan(size_t lidarNo, size_t scanNo)
{
    for(auto i : m_changed)
    {
        if(i)
        {
            return loadScan(i, lidarNo, scanNo);
        }
    }
    // Default values
    return std::make_pair(nullptr, Transformd::Identity());
}

ScanPositionPtr ScanProjectManager::loadNextChangedScanPosition()
{
    for(size_t i = 0; i < m_changed.size(); i++)
    {
        if(m_changed[i])
        {
            m_changed[i] = false;
            return m_scanProject->positions[i];
        }
    }
    return nullptr;
}




} // namespace lvr2