/****************************************************************************
** Meta object code from reading C++ file 'TransformationDialog.h'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/qglviewer/widgets/TransformationDialog.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'TransformationDialog.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_TransformationDialog[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
      12,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      28,   22,   21,   21, 0x0a,
      49,   22,   21,   21, 0x0a,
      70,   22,   21,   21, 0x0a,
      91,   22,   21,   21, 0x0a,
     116,   22,   21,   21, 0x0a,
     141,   22,   21,   21, 0x0a,
     166,   22,   21,   21, 0x0a,
     194,   22,   21,   21, 0x0a,
     222,   22,   21,   21, 0x0a,
     250,   22,   21,   21, 0x0a,
     270,   21,   21,   21, 0x0a,
     278,   21,   21,   21, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_TransformationDialog[] = {
    "TransformationDialog\0\0value\0"
    "rotationXSlided(int)\0rotationYSlided(int)\0"
    "rotationZSlided(int)\0rotationXEntered(double)\0"
    "rotationYEntered(double)\0"
    "rotationZEntered(double)\0"
    "translationXEntered(double)\0"
    "translationYEntered(double)\0"
    "translationZEntered(double)\0"
    "stepChanged(double)\0reset()\0save()\0"
};

const QMetaObject TransformationDialog::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_TransformationDialog,
      qt_meta_data_TransformationDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &TransformationDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *TransformationDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *TransformationDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_TransformationDialog))
        return static_cast<void*>(const_cast< TransformationDialog*>(this));
    return QObject::qt_metacast(_clname);
}

int TransformationDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: rotationXSlided((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 1: rotationYSlided((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 2: rotationZSlided((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: rotationXEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 4: rotationYEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 5: rotationZEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 6: translationXEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 7: translationYEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 8: translationZEntered((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 9: stepChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 10: reset(); break;
        case 11: save(); break;
        default: ;
        }
        _id -= 12;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
