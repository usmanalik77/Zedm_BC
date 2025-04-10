// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from zed_msgs:msg/Keypoint2Df.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "zed_msgs/msg/detail/keypoint2_df__struct.hpp"
#include "rosidl_typesupport_introspection_cpp/field_types.hpp"
#include "rosidl_typesupport_introspection_cpp/identifier.hpp"
#include "rosidl_typesupport_introspection_cpp/message_introspection.hpp"
#include "rosidl_typesupport_introspection_cpp/message_type_support_decl.hpp"
#include "rosidl_typesupport_introspection_cpp/visibility_control.h"

namespace zed_msgs
{

namespace msg
{

namespace rosidl_typesupport_introspection_cpp
{

void Keypoint2Df_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) zed_msgs::msg::Keypoint2Df(_init);
}

void Keypoint2Df_fini_function(void * message_memory)
{
  auto typed_message = static_cast<zed_msgs::msg::Keypoint2Df *>(message_memory);
  typed_message->~Keypoint2Df();
}

size_t size_function__Keypoint2Df__kp(const void * untyped_member)
{
  (void)untyped_member;
  return 2;
}

const void * get_const_function__Keypoint2Df__kp(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<float, 2> *>(untyped_member);
  return &member[index];
}

void * get_function__Keypoint2Df__kp(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<float, 2> *>(untyped_member);
  return &member[index];
}

void fetch_function__Keypoint2Df__kp(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const float *>(
    get_const_function__Keypoint2Df__kp(untyped_member, index));
  auto & value = *reinterpret_cast<float *>(untyped_value);
  value = item;
}

void assign_function__Keypoint2Df__kp(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<float *>(
    get_function__Keypoint2Df__kp(untyped_member, index));
  const auto & value = *reinterpret_cast<const float *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Keypoint2Df_message_member_array[1] = {
  {
    "kp",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_FLOAT,  // type
    0,  // upper bound of string
    nullptr,  // members of sub message
    true,  // is array
    2,  // array size
    false,  // is upper bound
    offsetof(zed_msgs::msg::Keypoint2Df, kp),  // bytes offset in struct
    nullptr,  // default value
    size_function__Keypoint2Df__kp,  // size() function pointer
    get_const_function__Keypoint2Df__kp,  // get_const(index) function pointer
    get_function__Keypoint2Df__kp,  // get(index) function pointer
    fetch_function__Keypoint2Df__kp,  // fetch(index, &value) function pointer
    assign_function__Keypoint2Df__kp,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Keypoint2Df_message_members = {
  "zed_msgs::msg",  // message namespace
  "Keypoint2Df",  // message name
  1,  // number of fields
  sizeof(zed_msgs::msg::Keypoint2Df),
  Keypoint2Df_message_member_array,  // message members
  Keypoint2Df_init_function,  // function to initialize message memory (memory has to be allocated)
  Keypoint2Df_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Keypoint2Df_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Keypoint2Df_message_members,
  get_message_typesupport_handle_function,
};

}  // namespace rosidl_typesupport_introspection_cpp

}  // namespace msg

}  // namespace zed_msgs


namespace rosidl_typesupport_introspection_cpp
{

template<>
ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
get_message_type_support_handle<zed_msgs::msg::Keypoint2Df>()
{
  return &::zed_msgs::msg::rosidl_typesupport_introspection_cpp::Keypoint2Df_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, zed_msgs, msg, Keypoint2Df)() {
  return &::zed_msgs::msg::rosidl_typesupport_introspection_cpp::Keypoint2Df_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
