namespace lvr2
{
namespace scanio
{

template <typename BaseIO>
void LabelScanProjectIO<BaseIO>::saveLabelScanProject(const LabeledScanProjectEditMarkPtr& labelScanProjectPtr)
{
    Description d = m_baseIO->m_description->scanProject();

    // Default names
    std::string group = "";

    if (labelScanProjectPtr->editMarkProject && labelScanProjectPtr->editMarkProject->project)
    {
        lvr2::logout::get() << lvr2::info << "[LabelScanProjectIO] Save ScanProject" << lvr2::endl;
        m_scanProjectIO->saveScanProject(labelScanProjectPtr->editMarkProject->project);
    }

    if (labelScanProjectPtr->labelRoot && labelScanProjectPtr->labelRoot->points)
    {
        lvr2::logout::get() << lvr2::info << "[LabelScanProjectIO] Save Labels" << lvr2::endl;
        m_labelIO->saveLabels(group, labelScanProjectPtr->labelRoot); 
    } else
    {
     lvr2::logout::get() << lvr2::warning << "[LabelScanProjectIO] No Labels" << lvr2::endl;
    }
}

template <typename BaseIO>
LabeledScanProjectEditMarkPtr LabelScanProjectIO<BaseIO>::loadLabelScanProject()
{
    LabeledScanProjectEditMarkPtr ret(new LabeledScanProjectEditMark);

    ScanProjectEditMarkPtr editMarkPtr(new ScanProjectEditMark);
    ret->editMarkProject = editMarkPtr;
    
    editMarkPtr->project = m_scanProjectIO->loadScanProject();
    std::string pointCloud("/pointCloud");
    if (m_baseIO->m_kernel->exists(pointCloud))
    {
        lvr2::logout::get() << "[LabelScanProjectIO] Load Labels" << lvr2::endl;
        ret->labelRoot = m_labelIO->loadLabels(pointCloud);;
    }
    return ret;
}
template <typename BaseIO>
ScanProjectPtr LabelScanProjectIO<BaseIO>::loadScanProject()
{
    return m_scanProjectIO->loadScanProject();
}

} // scanio

} // namespace lvr2
