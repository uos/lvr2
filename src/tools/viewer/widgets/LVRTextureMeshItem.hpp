/*
 * LVRTextureMeshItem.h
 *
 *  Created on: Dec 10, 2014
 *      Author: twiemann
 */

#ifndef SRC_TOOLS_VIEWER_WIDGETS_LVRTEXTUREMESHITEM_HPP_
#define SRC_TOOLS_VIEWER_WIDGETS_LVRTEXTUREMESHITEM_HPP_

#include "LVRMeshItem.hpp"

namespace lvr
{

class LVRTextureMeshItem : public LVRMeshItem
{
public:
	LVRTextureMeshItem(MeshBufferBridgePtr& ptr, QTreeWidgetItem* parent = 0);
	virtual ~LVRTextureMeshItem();

protected:
	virtual void addSubItems();
};

} /* namespace lvr */

#endif /* SRC_TOOLS_VIEWER_WIDGETS_LVRTEXTUREMESHITEM_HPP_ */
