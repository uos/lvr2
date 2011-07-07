/**
 * TriangleMeshTreeWidgetItem.h
 *
 *  @date 16.06.2011
 *  @author Thomas Wiemann
 */

#ifndef TRIANGLEMESHTREEWIDGETITEM_H_
#define TRIANGLEMESHTREEWIDGETITEM_H_

#include "CustomTreeWidgetItem.h"

class TriangleMeshTreeWidgetItem : public CustomTreeWidgetItem
{
public:
    TriangleMeshTreeWidgetItem(int type);
    TriangleMeshTreeWidgetItem(QTreeWidgetItem* parent, int type);
    virtual ~TriangleMeshTreeWidgetItem() {}

    void setNumFaces(size_t n);
    void setNumVertices(size_t n);

private:
    size_t      m_numFaces;
    size_t      m_numVertices;
};

#endif /* TRIANGLEMESHTREEWIDGETITEM_H_ */
