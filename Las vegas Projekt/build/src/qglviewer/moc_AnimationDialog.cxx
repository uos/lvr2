/****************************************************************************
** Meta object code from reading C++ file 'AnimationDialog.hpp'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/qglviewer/widgets/AnimationDialog.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'AnimationDialog.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_AnimationDialog[] = {

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
      17,   16,   16,   16, 0x0a,
      27,   16,   16,   16, 0x0a,
      42,   40,   16,   16, 0x0a,
      62,   16,   16,   16, 0x0a,
      75,   16,   16,   16, 0x0a,
      88,   16,   16,   16, 0x0a,
     102,   16,   16,   16, 0x0a,
     115,   16,   16,   16, 0x0a,
     122,   16,   16,   16, 0x0a,
     136,   16,   16,   16, 0x0a,
     147,   16,   16,   16, 0x0a,
     163,  158,   16,   16, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_AnimationDialog[] = {
    "AnimationDialog\0\0addItem()\0removeItem()\0"
    "d\0updateTimes(double)\0selectNext()\0"
    "selectPrev()\0selectFirst()\0selectLast()\0"
    "play()\0createVideo()\0savePath()\0"
    "loadPath()\0item\0updateSelectedItem(QListWidgetItem*)\0"
};

const QMetaObject AnimationDialog::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_AnimationDialog,
      qt_meta_data_AnimationDialog, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &AnimationDialog::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *AnimationDialog::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *AnimationDialog::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_AnimationDialog))
        return static_cast<void*>(const_cast< AnimationDialog*>(this));
    return QObject::qt_metacast(_clname);
}

int AnimationDialog::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: addItem(); break;
        case 1: removeItem(); break;
        case 2: updateTimes((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 3: selectNext(); break;
        case 4: selectPrev(); break;
        case 5: selectFirst(); break;
        case 6: selectLast(); break;
        case 7: play(); break;
        case 8: createVideo(); break;
        case 9: savePath(); break;
        case 10: loadPath(); break;
        case 11: updateSelectedItem((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 12;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
