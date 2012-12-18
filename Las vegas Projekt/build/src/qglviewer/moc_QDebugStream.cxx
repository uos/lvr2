/****************************************************************************
** Meta object code from reading C++ file 'QDebugStream.hpp'
**
** Created: Mon Oct 22 15:26:47 2012
**      by: The Qt Meta Object Compiler version 62 (Qt 4.7.4)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../../src/qglviewer/widgets/QDebugStream.hpp"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'QDebugStream.hpp' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 62
#error "This file was generated using the moc from 4.7.4. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_QDebugStream[] = {

 // content:
       5,       // revision
       0,       // classname
       0,    0, // classinfo
       1,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       1,       // signalCount

 // signals: signature, parameters, type, tag, flags
      19,   14,   13,   13, 0x05,

       0        // eod
};

static const char qt_meta_stringdata_QDebugStream[] = {
    "QDebugStream\0\0text\0sendString(QString)\0"
};

const QMetaObject QDebugStream::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_QDebugStream,
      qt_meta_data_QDebugStream, 0 }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &QDebugStream::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *QDebugStream::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *QDebugStream::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_QDebugStream))
        return static_cast<void*>(const_cast< QDebugStream*>(this));
    if (!strcmp(_clname, "std::basic_streambuf<char>"))
        return static_cast< std::basic_streambuf<char>*>(const_cast< QDebugStream*>(this));
    return QObject::qt_metacast(_clname);
}

int QDebugStream::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: sendString((*reinterpret_cast< QString(*)>(_a[1]))); break;
        default: ;
        }
        _id -= 1;
    }
    return _id;
}

// SIGNAL 0
void QDebugStream::sendString(QString _t1)
{
    void *_a[] = { 0, const_cast<void*>(reinterpret_cast<const void*>(&_t1)) };
    QMetaObject::activate(this, &staticMetaObject, 0, _a);
}
QT_END_MOC_NAMESPACE
