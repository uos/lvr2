#ifndef CORRESPONDENCEVISUALIZER_H
#define CORRESPONDENCEVISUALIZER_H

#include "Visualizer.hpp"

class CorrespondenceVisualizer : public Visualizer
{
public:
    CorrespondenceVisualizer(string filename);
    virtual ~CorrespondenceVisualizer() {}
};

#endif // CORRESPONDENCEVISUALIZER_H
