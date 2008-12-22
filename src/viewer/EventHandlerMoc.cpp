/****************************************************************************
** Meta object code from reading C++ file 'EventHandler.h'
**
** Created: Thu Dec 18 13:46:33 2008
**      by: The Qt Meta Object Compiler version 59 (Qt 4.4.3)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'EventHandler.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 59
#error "This file was generated using the moc from 4.4.3. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_EventHandler[] = {

 // content:
       1,       // revision
       0,       // classname
       0,    0, // classinfo
       9,   10, // methods
       0,    0, // properties
       0,    0, // enums/sets

 // signals: signature, parameters, type, tag, flags
      14,   13,   13,   13, 0x05,

 // slots: signature, parameters, type, tag, flags
      31,   13,   13,   13, 0x0a,
      50,   13,   13,   13, 0x0a,
      67,   13,   13,   13, 0x0a,
      92,   13,   13,   13, 0x0a,
     118,  113,   13,   13, 0x0a,
     155,  113,   13,   13, 0x0a,
     195,   13,   13,   13, 0x0a,
     219,  217,   13,   13, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_EventHandler[] = {
    "EventHandler\0\0updateGLWidget()\0"
    "action_file_open()\0action_topView()\0"
    "action_perspectiveView()\0action_enterMatrix()\0"
    "item\0action_editObjects(QListWidgetItem*)\0"
    "action_objectSelected(QListWidgetItem*)\0"
    "transform_from_file()\0,\0"
    "touchpad_transform(int,double)\0"
};

const QMetaObject EventHandler::staticMetaObject = {
    { &QObject::staticMetaObject, qt_meta_stringdata_EventHandler,
      qt_meta_data_EventHandler, 0 }
};

const QMetaObject *EventHandler::metaObject() const
{
    return &staticMetaObject;
}

void *EventHandler::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_EventHandler))
        return static_cast<void*>(const_cast< EventHandler*>(this));
    return QObject::qt_metacast(_clname);
}

int EventHandler::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QObject::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        switch (_id) {
        case 0: updateGLWidget(); break;
        case 1: action_file_open(); break;
        case 2: action_topView(); break;
        case 3: action_perspectiveView(); break;
        case 4: action_enterMatrix(); break;
        case 5: action_editObjects((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 6: action_objectSelected((*reinterpret_cast< QListWidgetItem*(*)>(_a[1]))); break;
        case 7: transform_from_file(); break;
        case 8: touchpad_transform((*reinterpret_cast< int(*)>(_a[1])),(*reinterpret_cast< double(*)>(_a[2]))); break;
        }
        _id -= 9;
    }
    return _id;
}

// SIGNAL 0
void EventHandler::updateGLWidget()
{
    QMetaObject::activate(this, &staticMetaObject, 0, 0);
}
QT_END_MOC_NAMESPACE
