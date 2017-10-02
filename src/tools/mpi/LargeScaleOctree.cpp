//
// Created by eiseck on 08.12.15.
//

#include "LargeScaleOctree.hpp"
namespace lvr
{



LargeScaleOctree::LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints, size_t bufferSize) : m_center(center), m_size(size), m_maxPoints(maxPoints), m_data(bufferSize*3), m_parent(0), m_depth(0)
{
    m_root = this;

}
LargeScaleOctree::LargeScaleOctree(Vertexf center, float size, unsigned int maxPoints,  LargeScaleOctree* parent, LargeScaleOctree* root, int depth, size_t bufferSize) : m_center(center), m_size(size), m_maxPoints(maxPoints), m_data(bufferSize*3), m_parent(parent), m_depth(depth), m_root(root)
{

}


size_t LargeScaleOctree::getSize()
{
    return m_data.size();
}

Vertexf LargeScaleOctree::getCenter()
{
    return m_center;
}

float LargeScaleOctree::getWidth()
{
    return m_size;
}

void LargeScaleOctree::writeData()
{
    vector<LargeScaleOctree*> nodes = getNodes();
    for(int i = 0 ; i < nodes.size() ; i++)
    {

        nodes[i]->m_data.writeBuffer();

    }
}

void LargeScaleOctree::insert(Vertexf& pos, Vertexf normal)
{

    stack<std::pair<LargeScaleOctree*, std::pair<Vertexf, Vertexf> > > callStack;

    callStack.push(std::pair<LargeScaleOctree*, std::pair<Vertexf, Vertexf> > (this,  std::pair<Vertexf, Vertexf> (pos, normal)) );

    while(! callStack.empty())
    {
        auto currentCall = callStack.top();
        callStack.pop();

        auto currentNode = currentCall.first;
        auto currentPoint = currentCall.second.first;
        auto currentNormal = currentCall.second.second;
        if(currentNode->isLeaf())
        {
            currentNode->m_pointbb.expand(currentPoint[0], currentPoint[1], currentPoint[2]);
            currentNode->m_data.addBuffered(currentPoint);

                currentNode->m_data.addBufferedNormal(currentNormal);




            if(currentNode->m_data.size() == currentNode->m_maxPoints)
            {
                currentNode->m_data.writeBuffer();
                //cout << "NEW NODES" << endl;
                for(int i = 0 ; i<8 ; i++)
                {
                    Vertexf newCenter;
                    newCenter.x = currentNode->m_center.x + currentNode->m_size * 0.5 * (i&4 ? 0.5 : -0.5);
                    newCenter.y = currentNode->m_center.y + currentNode->m_size * 0.5 * (i&2 ? 0.5 : -0.5);
                    newCenter.z = currentNode->m_center.z + currentNode->m_size * 0.5 * (i&1 ? 0.5 : -0.5);
                    currentNode->m_children.push_back( new LargeScaleOctree(newCenter, currentNode->m_size * 0.5, currentNode->m_maxPoints, this, m_root, currentNode->m_depth+1) );
                }
                for(size_t i = 0 ; i<currentNode->m_data.size() ; i++)
                {
//                    cout << "size: " << currentNode->m_data.size() << " i: " << i << endl;
                    Vertexf v = currentNode->m_data.get(i);
                    Vertexf n = currentNode->m_data.getNormal(i);
                    callStack.push(std::pair<LargeScaleOctree*, std::pair<Vertexf, Vertexf> > (currentNode->m_children[currentNode->getOctant(v)], std::pair<Vertexf, Vertexf>(v,n)) );
                }
//                for(Vertexf v : currentNode->m_data)
//                {
//                    callStack.push(std::pair<LargeScaleOctree*, std::pair<Vertexf, Vertexf> > (currentNode->m_children[currentNode->getOctant(v)], v) );
//
//                }
                currentNode->m_data.remove();

            }

        }
        else
        {
            callStack.push(std::pair<LargeScaleOctree*, std::pair<Vertexf, Vertexf> > (currentNode->m_children[currentNode->getOctant(pos)], std::pair<Vertexf, Vertexf>(currentPoint, currentNormal)) );

        }
    }






}

bool LargeScaleOctree::isLeaf()
{
//    cout << "leaf check: " << m_children.size() << endl;
    return m_children.size() == 0;
}

inline int LargeScaleOctree::getOctant(const Vertexf& point) const
{
    int oct = 0;
    if(point.x >= m_center.x) oct |= 4;
    if(point.y >= m_center.y) oct |= 2;
    if(point.z >= m_center.z) oct |= 1;
    return oct;
}

string LargeScaleOctree::getFilePath()
{
    return m_data.getDataPath();
}


string LargeScaleOctree::getFolder()
{
    return m_data.getFolder();
}

vector<LargeScaleOctree*>& LargeScaleOctree::getChildren()
{
    return m_children;
}

vector<LargeScaleOctree*>& LargeScaleOctree::getSavedNeighbours()
{
    return m_neighbours;
}

void LargeScaleOctree::getNodes(LargeScaleOctree* root, vector<LargeScaleOctree*>& nodelist)
{
    nodelist.push_back(root);
    cout << "current node has: " << root->m_children.size() << " children" << endl;
    for(int i = 0 ; i < root->m_children.size() ; i++)
    {
        getNodes(root->m_children[i], nodelist);
    }
    return;

}

vector<LargeScaleOctree*> LargeScaleOctree::getNodes()
{
    vector<LargeScaleOctree*> nodes;
    getNodes(m_root, nodes);
    return nodes;

//    LargeScaleOctree* current_node;
//    std::stack<LargeScaleOctree*> s;
//    s.push(m_root);
//    std::vector<LargeScaleOctree*> nodes;
//    nodes.push_back(m_root);
//    while(!s.empty())
//    {
//        current_node = s.top();
//        s.pop();
//        for(int i = 0 ; i < current_node->m_children.size() ; i++)
//        {
//            nodes.push_back(current_node->m_children[i]);
//            s.push(current_node->m_children[i]);
//        }
//    }

}

bool LargeScaleOctree::operator<(  LargeScaleOctree& rhs )
{
    return getSize() < rhs.getSize();
}

vector<LargeScaleOctree*> LargeScaleOctree::getNeighbours()
{
    //Get neighbours on higher or same depth in tree
    vector<LargeScaleOctree*> finalNeighbours;
    vector<int> foundInDir;
    vector<LargeScaleOctree*> n;
    //cout << "center: "<< endl<< m_center <<endl << "width: " << m_size << endl;
    for(int i = 0 ; i<6 ; i++)
    {

        Vertexi currentDir = dirTable[i];
        Vertexf neighbourPoint = m_center;
        Vertexf dirAdd;
        dirAdd.x = m_size;
        dirAdd.y = m_size;
        dirAdd.z = m_size;
        dirAdd.x*=currentDir.x;
        dirAdd.y*=currentDir.y;
        dirAdd.z*=currentDir.z;
        neighbourPoint.x+=dirAdd.x;
        neighbourPoint.y+=dirAdd.y;
        neighbourPoint.z+=dirAdd.z;
        //cout << "checking neigbour at: "<< endl<< neighbourPoint <<endl;

        LargeScaleOctree* currentNode = m_root;

        //check if in BoundingBox

        Vertexf bbmax(m_root->m_center.x+(m_root->m_size/2),m_root->m_center.y+(m_root->m_size/2),m_root->m_center.z+(m_root->m_size/2));
        Vertexf bbmin(m_root->m_center.x-(m_root->m_size/2),m_root->m_center.y-(m_root->m_size/2),m_root->m_center.z-(m_root->m_size/2));
        //cout << "bb : " << bbmax << bbmin << endl;
        //cout << "is in bb?: "<< endl<< neighbourPoint <<endl;
        if(     (neighbourPoint.x <= bbmax.x && neighbourPoint.x >= bbmin.x &&
                neighbourPoint.y <= bbmax.y && neighbourPoint.y >= bbmin.y &&
                neighbourPoint.z <= bbmax.z && neighbourPoint.z >= bbmin.z ))
        {
          //  cout << "yes " <<endl;
            while( (! currentNode->isLeaf()) && (m_depth !=  currentNode->m_depth) )
            {
                currentNode = currentNode->getChildren()[currentNode->getOctant(neighbourPoint)];
            }
            //cout << "na : " << currentNode->getCenter() << endl;
            n.push_back(currentNode);
            foundInDir.push_back(i);


        }
        //else cout << "no" << endl;



    }
    for(int i = 0 ; i<n.size() ; i++)
    {
        if(n[i]->isLeaf()) finalNeighbours.push_back(n[i]);
        else
        {
            vector<LargeScaleOctree*> recChildren = getRecChildrenNeighbours(n[i], foundInDir[i]);
            finalNeighbours.insert(finalNeighbours.end(), recChildren.begin(), recChildren.end());
        }
    }
    finalNeighbours.erase(std::remove_if(finalNeighbours.begin(), finalNeighbours.end(),
                       [](LargeScaleOctree* i)
                       {
                           if(! boost::filesystem::exists(i->getFilePath())) return true;
                           return boost::filesystem::file_size(i->getFilePath()) == 0;
                    }), finalNeighbours.end());
    return finalNeighbours;
}

vector<LargeScaleOctree*> LargeScaleOctree::getRecChildrenNeighbours(LargeScaleOctree* octant, int dir)
{
    vector<LargeScaleOctree*> n;
    int newdir = dir;

    //get opposite direction
    if (dir%2 == 0) newdir++;
    else newdir--;
    stack<LargeScaleOctree*> chechChildren;
    for(int i = 0 ; i<4 ;i++) chechChildren.push(octant->getChildren()[dirChildrenTable[newdir][i]]);
    while(!chechChildren.empty())
    {
        LargeScaleOctree* current = chechChildren.top();
        chechChildren.pop();
        if(current->isLeaf()) n.push_back(current);
        else
        {
            for(int i = 0 ; i<4 ;i++) chechChildren.push(current->getChildren()[dirChildrenTable[newdir][i]]);
        }
    }
    //cout << this->getFilePath() << "( " << this->m_id << ") rec on side: " << newdir << endl;
    //for(int i = 0 ; i<n.size() ; i++)
    //{
    //    cout << n[i]->getFilePath() << "|" << n[i]->getID() << endl;
    //}
return n;

}

void LargeScaleOctree::generateNeighbourhood()
{
    m_neighbours = getNeighbours();
}

/*
vector<LargeScaleOctree*> getNoneLocalNeigbours(LargeScaleOctree* parent, int nodeID, Vertexi dir)
{
    vector<LargeScaleOctree*> tempNeighbours;
    for(int i = 0 ; i<3 ; i++)
    {
        if(parent->getChildren()[octreeLocalNeighborTable[nodeID][i]]!=0)
        {
            tempNeighbours.push_back(parent->getChildren()[octreeLocalNeighborTable[nodeID][i]]);
        }

    }

}
*/

}