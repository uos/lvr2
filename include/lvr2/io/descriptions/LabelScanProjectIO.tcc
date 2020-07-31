namespace lvr2
{

template <typename FeatureBase>
void LabelScanProjectIO<FeatureBase>::saveLabelScanProject(const LabeledScanProjectEditMarkPtr& labelScanProjectPtr)
{
    Description d = m_featureBase->m_description->scanProject();

    // Default names
    std::string group = "";

    std::cout << "Label Scan Project" << std::endl;
    if (labelScanProjectPtr->editMarkProject && labelScanProjectPtr->editMarkProject->project)
    {
        std::cout << "Label Scan Project -> Scan Project" << std::endl;
        m_scanProjectIO->saveScanProject(labelScanProjectPtr->editMarkProject->project);
    }

    if (labelScanProjectPtr->labelRoot)
    {
        std::cout << "Label Scan Project -> Label Root" << std::endl;
       m_labelIO->saveLabels(group, labelScanProjectPtr->labelRoot); 
    }

}

template <typename FeatureBase>
LabeledScanProjectEditMarkPtr LabelScanProjectIO<FeatureBase>::loadLabelScanProject()
{
    LabeledScanProjectEditMarkPtr ret(new LabeledScanProjectEditMark);

    ScanProjectEditMarkPtr editMarkPtr(new ScanProjectEditMark);
    ret->editMarkProject = editMarkPtr;
    std::string pointCloud("/pointCloud");
    if (m_featureBase->m_kernel->exists(pointCloud))
    {
        std::cout << "Label Scan Project -> Label Root" << std::endl;
        ret->labelRoot = m_labelIO->loadLabels(pointCloud);;
    }
    editMarkPtr->project = m_scanProjectIO->loadScanProject();
    return ret;
}
template <typename FeatureBase>
ScanProjectPtr LabelScanProjectIO<FeatureBase>::loadScanProject()
{
    return m_scanProjectIO->loadScanProject();
}

} // namespace lvr2
