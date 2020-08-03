namespace lvr2
{

template <typename FeatureBase>
void LabelScanProjectIO<FeatureBase>::saveLabelScanProject(const LabeledScanProjectEditMarkPtr& labelScanProjectPtr)
{
    Description d = m_featureBase->m_description->scanProject();

    // Default names
    std::string group = "";

    if (labelScanProjectPtr->editMarkProject && labelScanProjectPtr->editMarkProject->project)
    {
        std::cout << "[LabelScanProjectIO] Save ScanProject" << std::endl;
        m_scanProjectIO->saveScanProject(labelScanProjectPtr->editMarkProject->project);
    }

    if (labelScanProjectPtr->labelRoot)
    {
        std::cout << "[LabelScanProjectIO] Save Labeles" << std::endl;
        m_labelIO->saveLabels(group, labelScanProjectPtr->labelRoot); 
    }
}

template <typename FeatureBase>
LabeledScanProjectEditMarkPtr LabelScanProjectIO<FeatureBase>::loadLabelScanProject()
{
    LabeledScanProjectEditMarkPtr ret(new LabeledScanProjectEditMark);

    ScanProjectEditMarkPtr editMarkPtr(new ScanProjectEditMark);
    ret->editMarkProject = editMarkPtr;
    
    editMarkPtr->project = m_scanProjectIO->loadScanProject();
    std::string pointCloud("/pointCloud");
    if (m_featureBase->m_kernel->exists(pointCloud))
    {
        std::cout << "[LabelScanProjectIO] Load Labeles" << std::endl;
        ret->labelRoot = m_labelIO->loadLabels(pointCloud);;
    }
    return ret;
}
template <typename FeatureBase>
ScanProjectPtr LabelScanProjectIO<FeatureBase>::loadScanProject()
{
    return m_scanProjectIO->loadScanProject();
}

} // namespace lvr2
