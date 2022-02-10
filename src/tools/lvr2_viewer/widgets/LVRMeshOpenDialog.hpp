#ifndef LVRMESHOPENDIALOG_HPP
#define LVRMESHOPENDIALOG_HPP

#include <QDialog>

namespace Ui {
class LVRMeshOpenDialog;
}

class LVRMeshOpenDialog : public QDialog
{
    Q_OBJECT

public:
    /**
     * @brief Construct a new LVRMeshOpenDialog object
     * 
     * @param parent Parent Widget
     */
    explicit LVRMeshOpenDialog(QWidget *parent = nullptr);

    bool dialogSuccessful() const {return m_success;};

    /**
     * @brief Set the Available Meshes list
     * 
     * @param names The meshes available for loading
     */
    void setAvailableMeshes(const std::vector<std::string>& names);

    /**
     * @brief Get the Selected Mesh
     * 
     * @return std::string The mesh selected by the user for loading
     */
    std::string getSelectedMesh() const;

    std::string getFailureReason() const
    {
        return m_failureReason;
    }

    ~LVRMeshOpenDialog();

public Q_SLOTS:
    /// Called when the ok button is pressed
    void onAccept();

    /// Called when the cancel button is pressed
    void onCancel();

    /// Called when the selection is changed
    void onSelectedMeshNameChanged(QString);

// Private member functions
private:
    void connectSignalsAndSlots();

private:
    Ui::LVRMeshOpenDialog* m_ui;

    bool m_success;

    std::string m_failureReason;

    std::string m_selectedMeshName;
};

#endif // LVRMESHOPENDIALOG_HPP
