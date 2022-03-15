#ifndef LARGESCALERECONSTRUCTIONBASE
#define LARGESCALERECONSTRUCTIONBASE

#include "lvr2/types/ScanTypes.hpp"

/**
 * @brief Base Class for all large scale reconstruction methods.
 */
class LargeScaleReconstructionBase
{
private:
    
public:

    /**
     * @brief   Construct a new Large Scale Reconstruction Base object 
     *          with default parameters and empty scan project.
     */
    LargeScaleReconstructionBase();

    /**
     * @brief   Destroy the Large Scale Reconstruction Base object
     * 
     */
    ~LargeScaleReconstructionBase();

    

protected:

    /// Scan project parsed for reconstruction
    ScanProjectPtr m_scanProject;
};



#endif // LARGESCALERECONSTRUCTIONBASE
