// generated from rosidl_typesupport_introspection_cpp/resource/idl__type_support.cpp.em
// with input from zed_msgs:msg/Skeleton3D.idl
// generated code does not contain a copyright notice

#include "array"
#include "cstddef"
#include "string"
#include "vector"
#include "rosidl_runtime_c/message_type_support_struct.h"
#include "rosidl_typesupport_cpp/message_type_support.hpp"
#include "rosidl_typesupport_interface/macros.h"
#include "zed_msgs/msg/detail/skeleton3_d__struct.hpp"
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

void Skeleton3D_init_function(
  void * message_memory, rosidl_runtime_cpp::MessageInitialization _init)
{
  new (message_memory) zed_msgs::msg::Skeleton3D(_init);
}

void Skeleton3D_fini_function(void * message_memory)
{
  auto typed_message = static_cast<zed_msgs::msg::Skeleton3D *>(message_memory);
  typed_message->~Skeleton3D();
}

size_t size_function__Skeleton3D__keypoints(const void * untyped_member)
{
  (void)untyped_member;
  return 70;
}

const void * get_const_function__Skeleton3D__keypoints(const void * untyped_member, size_t index)
{
  const auto & member =
    *reinterpret_cast<const std::array<zed_msgs::msg::Keypoint3D, 70> *>(untyped_member);
  return &member[index];
}

void * get_function__Skeleton3D__keypoints(void * untyped_member, size_t index)
{
  auto & member =
    *reinterpret_cast<std::array<zed_msgs::msg::Keypoint3D, 70> *>(untyped_member);
  return &member[index];
}

void fetch_function__Skeleton3D__keypoints(
  const void * untyped_member, size_t index, void * untyped_value)
{
  const auto & item = *reinterpret_cast<const zed_msgs::msg::Keypoint3D *>(
    get_const_function__Skeleton3D__keypoints(untyped_member, index));
  auto & value = *reinterpret_cast<zed_msgs::msg::Keypoint3D *>(untyped_value);
  value = item;
}

void assign_function__Skeleton3D__keypoints(
  void * untyped_member, size_t index, const void * untyped_value)
{
  auto & item = *reinterpret_cast<zed_msgs::msg::Keypoint3D *>(
    get_function__Skeleton3D__keypoints(untyped_member, index));
  const auto & value = *reinterpret_cast<const zed_msgs::msg::Keypoint3D *>(untyped_value);
  item = value;
}

static const ::rosidl_typesupport_introspection_cpp::MessageMember Skeleton3D_message_member_array[1] = {
  {
    "keypoints",  // name
    ::rosidl_typesupport_introspection_cpp::ROS_TYPE_MESSAGE,  // type
    0,  // upper bound of string
    ::rosidl_typesupport_introspection_cpp::get_message_type_support_handle<zed_msgs::msg::Keypoint3D>(),  // members of sub message
    true,  // is array
    70,  // array size
    false,  // is upper bound
    offsetof(zed_msgs::msg::Skeleton3D, keypoints),  // bytes offset in struct
    nullptr,  // default value
    size_function__Skeleton3D__keypoints,  // size() function pointer
    get_const_function__Skeleton3D__keypoints,  // get_const(index) function pointer
    get_function__Skeleton3D__keypoints,  // get(index) function pointer
    fetch_function__Skeleton3D__keypoints,  // fetch(index, &value) function pointer
    assign_function__Skeleton3D__keypoints,  // assign(index, value) function pointer
    nullptr  // resize(index) function pointer
  }
};

static const ::rosidl_typesupport_introspection_cpp::MessageMembers Skeleton3D_message_members = {
  "zed_msgs::msg",  // message namespace
  "Skeleton3D",  // message name
  1,  // number of fields
  sizeof(zed_msgs::msg::Skeleton3D),
  Skeleton3D_message_member_array,  // message members
  Skeleton3D_init_function,  // function to initialize message memory (memory has to be allocated)
  Skeleton3D_fini_function  // function to terminate message instance (will not free memory)
};

static const rosidl_message_type_support_t Skeleton3D_message_type_support_handle = {
  ::rosidl_typesupport_introspection_cpp::typesupport_identifier,
  &Skeleton3D_message_members,
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
get_message_type_support_handle<zed_msgs::msg::Skeleton3D>()
{
  return &::zed_msgs::msg::rosidl_typesupport_introspection_cpp::Skeleton3D_message_type_support_handle;
}

}  // namespace rosidl_typesupport_introspection_cpp

#ifdef __cplusplus
extern "C"
{
#endif

ROSIDL_TYPESUPPORT_INTROSPECTION_CPP_PUBLIC
const rosidl_message_type_support_t *
ROSIDL_TYPESUPPORT_INTERFACE__MESSAGE_SYMBOL_NAME(rosidl_typesupport_introspection_cpp, zed_msgs, msg, Skeleton3D)() {
  return &::zed_msgs::msg::rosidl_typesupport_introspection_cpp::Skeleton3D_message_type_support_handle;
}

#ifdef __cplusplus
}
#endif
