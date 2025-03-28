// generated from rosidl_generator_py/resource/_idl_support.c.em
// with input from zed_msgs:msg/Skeleton3D.idl
// generated code does not contain a copyright notice
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <stdbool.h>
#ifndef _WIN32
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunused-function"
#endif
#include "numpy/ndarrayobject.h"
#ifndef _WIN32
# pragma GCC diagnostic pop
#endif
#include "rosidl_runtime_c/visibility_control.h"
#include "zed_msgs/msg/detail/skeleton3_d__struct.h"
#include "zed_msgs/msg/detail/skeleton3_d__functions.h"

#include "rosidl_runtime_c/primitives_sequence.h"
#include "rosidl_runtime_c/primitives_sequence_functions.h"

// Nested array functions includes
#include "zed_msgs/msg/detail/keypoint3_d__functions.h"
// end nested array functions include
bool zed_msgs__msg__keypoint3_d__convert_from_py(PyObject * _pymsg, void * _ros_message);
PyObject * zed_msgs__msg__keypoint3_d__convert_to_py(void * raw_ros_message);

ROSIDL_GENERATOR_C_EXPORT
bool zed_msgs__msg__skeleton3_d__convert_from_py(PyObject * _pymsg, void * _ros_message)
{
  // check that the passed message is of the expected Python class
  {
    char full_classname_dest[37];
    {
      char * class_name = NULL;
      char * module_name = NULL;
      {
        PyObject * class_attr = PyObject_GetAttrString(_pymsg, "__class__");
        if (class_attr) {
          PyObject * name_attr = PyObject_GetAttrString(class_attr, "__name__");
          if (name_attr) {
            class_name = (char *)PyUnicode_1BYTE_DATA(name_attr);
            Py_DECREF(name_attr);
          }
          PyObject * module_attr = PyObject_GetAttrString(class_attr, "__module__");
          if (module_attr) {
            module_name = (char *)PyUnicode_1BYTE_DATA(module_attr);
            Py_DECREF(module_attr);
          }
          Py_DECREF(class_attr);
        }
      }
      if (!class_name || !module_name) {
        return false;
      }
      snprintf(full_classname_dest, sizeof(full_classname_dest), "%s.%s", module_name, class_name);
    }
    assert(strncmp("zed_msgs.msg._skeleton3_d.Skeleton3D", full_classname_dest, 36) == 0);
  }
  zed_msgs__msg__Skeleton3D * ros_message = _ros_message;
  {  // keypoints
    PyObject * field = PyObject_GetAttrString(_pymsg, "keypoints");
    if (!field) {
      return false;
    }
    PyObject * seq_field = PySequence_Fast(field, "expected a sequence in 'keypoints'");
    if (!seq_field) {
      Py_DECREF(field);
      return false;
    }
    Py_ssize_t size = 70;
    zed_msgs__msg__Keypoint3D * dest = ros_message->keypoints;
    for (Py_ssize_t i = 0; i < size; ++i) {
      if (!zed_msgs__msg__keypoint3_d__convert_from_py(PySequence_Fast_GET_ITEM(seq_field, i), &dest[i])) {
        Py_DECREF(seq_field);
        Py_DECREF(field);
        return false;
      }
    }
    Py_DECREF(seq_field);
    Py_DECREF(field);
  }

  return true;
}

ROSIDL_GENERATOR_C_EXPORT
PyObject * zed_msgs__msg__skeleton3_d__convert_to_py(void * raw_ros_message)
{
  /* NOTE(esteve): Call constructor of Skeleton3D */
  PyObject * _pymessage = NULL;
  {
    PyObject * pymessage_module = PyImport_ImportModule("zed_msgs.msg._skeleton3_d");
    assert(pymessage_module);
    PyObject * pymessage_class = PyObject_GetAttrString(pymessage_module, "Skeleton3D");
    assert(pymessage_class);
    Py_DECREF(pymessage_module);
    _pymessage = PyObject_CallObject(pymessage_class, NULL);
    Py_DECREF(pymessage_class);
    if (!_pymessage) {
      return NULL;
    }
  }
  zed_msgs__msg__Skeleton3D * ros_message = (zed_msgs__msg__Skeleton3D *)raw_ros_message;
  {  // keypoints
    PyObject * field = NULL;
    size_t size = 70;
    field = PyList_New(size);
    if (!field) {
      return NULL;
    }
    zed_msgs__msg__Keypoint3D * item;
    for (size_t i = 0; i < size; ++i) {
      item = &(ros_message->keypoints[i]);
      PyObject * pyitem = zed_msgs__msg__keypoint3_d__convert_to_py(item);
      if (!pyitem) {
        Py_DECREF(field);
        return NULL;
      }
      int rc = PyList_SetItem(field, i, pyitem);
      (void)rc;
      assert(rc == 0);
    }
    assert(PySequence_Check(field));
    {
      int rc = PyObject_SetAttrString(_pymessage, "keypoints", field);
      Py_DECREF(field);
      if (rc) {
        return NULL;
      }
    }
  }

  // ownership of _pymessage is transferred to the caller
  return _pymessage;
}
