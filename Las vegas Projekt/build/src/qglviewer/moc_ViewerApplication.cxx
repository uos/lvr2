/****************************************************************************
** Meta object code from reading C++ file 'ViewerApplication.h'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/qglviewer/app/ViewerApplication.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ViewerApplication.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ViewerApplication[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      27,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      19,   18,   18,   18, 0x0a,
      46,   18,   18,   18, 0x0a,
      69,   18,   18,   18, 0x0a,
      92,   18,   18,   18, 0x0a,
     115,   18,   18,   18, 0x0a,
     127,   18,   18,   18, 0x0a,
     156,  154,   18,   18, 0x0a,
     179,   18,   18,   18, 0x0a,
     191,   18,   18,   18, 0x0a,
     201,   18,   18,   18, 0x0a,
     210,   18,   18,   18, 0x0a,
     239,  237,   18,   18, 0x0a,
     278,  271,   18,   18, 0x0a,
     318,  316,   18,   18, 0x0a,
     356,   18,   18,   18, 0x0a,
     379,   18,   18,   18, 0x0a,
     412,   18,   18,   18, 0x0a,
     433,   18,   18,   18, 0x0a,
     454,   18,   18,   18, 0x0a,
     472,   18,   18,   18, 0x0a,
     490,   18,   18,   18, 0x0a,
     505,   18,   18,   18, 0x0a,
     516,   18,   18,   18, 0x0a,
     540,   18,   18,   18, 0x0a,
     565,   18,   18,   18, 0x0a,
     592,   18,   18,   18, 0x0a,
     612,   18,   18,   18, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ViewerApplication[] = {
    "ViewerApplication\0\0setViewerModePerspective()\0"
    "setViewerModeOrthoXY()\0setViewerModeOrthoXZ()\0"
    "setViewerModeOrthoYZ()\0toggleFog()\0"
    "displayFogSettingsDialog()\0i\0"
    "fogDensityChanged(int)\0fogLinear()\0"
    "fogExp2()\0fogExp()\0displayRenderingSettings()\0"
    "d\0dataCollectorAdded(Visualizer*)\0"
    "item,n\0treeItemClicked(QTreeWidgetItem*,int)\0"
    ",\0treeItemChanged(QTreeWidgetItem*,int)\0"
    "treeSelectionChanged()\0"
    "treeContextMenuRequested(QPoint)\0"
    "saveSelectedObject()\0changeSelectedName()\0"
    "transformObject()\0createAnimation()\0"
    "deleteObject()\0openFile()\0"
    "meshRenderModeChanged()\0"
    "pointRenderModeChanged()\0"
    "createMeshFromPointcloud()\0"
    "centerOnSelection()\0zoomChanged()\0"
};

const QMetaObject ViewerApplication::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_ViewerApplication,
      qt_meta_data_ViewerApplication, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ViewerApplication::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ViewerApplication::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ViewerApplication::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewerApplication))
        return static_cast<void*>(const_cast< ViewerApplication*>(this));
    return QObject::qt_metacast(_clname);
}

int ViewerApplication::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: setViewerModePerspective(); break;
        case 1: setViewerModeOrthoXY(); break;
        case 2: setViewerModeOrthoXZ(); break;
        case 3: setViewerModeOrthoYZ(); break;
        case 4: toggleFog(); break;
        case 5: displayFogSettingsDialog(); break;
        case 6: fogDensityChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: fogLinear(); break;
        case 8: fogExp2(); break;
        case 9: fogExp(); break;
        case 10: displayRenderingSettings(); break;
        case 11: dataCollectorAdded((*reinterpret_cast< Visualizer*(*)>(_a[1]))); break;
        case 12: treeItemClicked((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 13: treeItemChanged((*reinterpret_cast< QTreeWidgetItem*(*)>(_a[1])),(*reinterpret_cast< int(*)>(_a[2]))); break;
        case 14: treeSelectionChanged(); break;
        case 15: treeContextMenuRequested((*reinterpret_cast< const QPoint(*)>(_a[1]))); break;
        case 16: saveSelectedObject(); break;
        case 17: changeSelectedName(); break;
        case 18: transformObject(); break;
        case 19: createAnimation(); break;
        case 20: deleteObject(); break;
        case 21: openFile(); break;
        case 22: meshRenderModeChanged(); break;
        case 23: pointRenderModeChanged(); break;
        case 24: createMeshFromPointcloud(); break;
        case 25: centerOnSelection(); break;
        case 26: zoomChanged(); break;
        default: ;
        }
        _id -= 27;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
