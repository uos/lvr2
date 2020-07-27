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
        m_scanProjectIO->saveScanProject(labelScanProjectPtr->editMarkProject->project);
    }

    if (labelScanProjectPtr->labelRoot)
    {
       m_labelIO->saveLabels(group, labelScanProjectPtr->labelRoot); 
    }

}

template <typename FeatureBase>
LabeledScanProjectEditMarkPtr LabelScanProjectIO<FeatureBase>::loadLabelScanProject()
{
    LabeledScanProjectEditMarkPtr ret(new LabeledScanProjectEditMark);

    ScanProjectEditMarkPtr editMarkPtr(new ScanProjectEditMark);
    editMarkPtr->project = m_scanProjectIO->loadScanProject();
    ret->editMarkProject = editMarkPtr;
    std::string pointCloud("pointcloud");
    if (m_featureBase->m_kernel->exists(pointCloud))
    {
        ret->labelRoot = m_labelIO->loadLabels("");;
    }
    return ret;
}
template <typename FeatureBase>
ScanProjectPtr LabelScanProjectIO<FeatureBase>::loadScanProject()
{
    return m_scanProjectIO->loadScanProject();
}

} // namespace lvr2
