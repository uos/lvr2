/****************************************************************************
** Meta object code from reading C++ file 'ObjectHandler.h'
**
** Created: Tue Mar 24 12:40:37 2009
**      by: The Qt Meta Object Compiler version 59 (Qt 4.4.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'ObjectHandler.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 59
#error "This file was generated using the moc from 4.4.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_ObjectHandler[] = {

 // content:
       1,       // revision
       0,       // classname
       0,    0, // classinfo
       0,    0, // methods
       0,    0, // properties
       0,    0, // enums/sets

       0        // eod
};

static const char qt_meta_stringdata_ObjectHandler[] = {
    "ObjectHandler\0"
};

const QMetaObject ObjectHandler::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_ObjectHandler,
      qt_meta_data_ObjectHandler, 0 }
};

const QMetaObject *ObjectHandler::metaObject() const
{
    return &staticMetaObject;
}

void *ObjectHandler::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_ObjectHandler))
        return static_cast<void*>(const_cast< ObjectHandler*>(this));
    return QObject::qt_metacast(_clname);
}

int ObjectHandler::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    return _id;
}
QT_END_MOC_NAMESPACE
