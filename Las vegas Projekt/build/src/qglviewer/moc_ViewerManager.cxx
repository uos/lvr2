/****************************************************************************
** Meta object code from reading C++ file 'ViewerManager.h'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/qglviewer/viewers/ViewerManager.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ViewerManager.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ViewerManager[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       3,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      17,   15,   14,   14, 0x0a,
      51,   47,   14,   14, 0x0a,
      81,   47,   14,   14, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_ViewerManager[] = {
    "ViewerManager\0\0c\0addDataCollector(Visualizer*)\0"
    "obj\0updateDataObject(Visualizer*)\0"
    "removeDataCollector(Visualizer*)\0"
};

const QMetaObject ViewerManager::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_ViewerManager,
      qt_meta_data_ViewerManager, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &ViewerManager::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *ViewerManager::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *ViewerManager::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ViewerManager))
        return static_cast<void*>(const_cast< ViewerManager*>(this));
    return QObject::qt_metacast(_clname);
}

int ViewerManager::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: addDataCollector((*reinterpret_cast< Visualizer*(*)>(_a[1]))); break;
        case 1: updateDataObject((*reinterpret_cast< Visualizer*(*)>(_a[1]))); break;
        case 2: removeDataCollector((*reinterpret_cast< Visualizer*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 3;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
