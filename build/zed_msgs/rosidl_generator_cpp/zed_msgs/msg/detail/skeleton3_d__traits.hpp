// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from zed_msgs:msg/Skeleton3D.idl
// generated code does not contain a copyright notice

#ifndef ZED_MSGS__MSG__DETAIL__SKELETON3_D__TRAITS_HPP_
#define ZED_MSGS__MSG__DETAIL__SKELETON3_D__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "zed_msgs/msg/detail/skeleton3_d__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'keypoints'
#include "zed_msgs/msg/detail/keypoint3_d__traits.hpp"

namespace zed_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const Skeleton3D & msg,
  std::ostream & out)
{
  out << "{";
  // member: keypoints
  {
    if (msg.keypoints.size() == 0) {
      out << "keypoints: []";
    } else {
      out << "keypoints: [";
      size_t pending_items = msg.keypoints.size();
      for (auto item : msg.keypoints) {
        to_flow_style_yaml(item, out);
        if (--pending_items > 0) {
          out << ", ";
        }
      }
      out << "]";
    }
  }
  out << "}";
}  // NOLINT(readability/fn_size)

inline void to_block_style_yaml(
  const Skeleton3D & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: keypoints
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.keypoints.size() == 0) {
      out << "keypoints: []\n";
    } else {
      out << "keypoints:\n";
      for (auto item : msg.keypoints) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const Skeleton3D & msg, bool use_flow_style = false)
{
  std::ostringstream out;
  if (use_flow_style) {
    to_flow_style_yaml(msg, out);
  } else {
    to_block_style_yaml(msg, out);
  }
  return out.str();
}

}  // namespace msg

}  // namespace zed_msgs

namespace rosidl_generator_traits
{

[[deprecated("use zed_msgs::msg::to_block_style_yaml() instead")]]
inline void to_yaml(
  const zed_msgs::msg::Skeleton3D & msg,
  std::ostream & out, size_t indentation = 0)
{
  zed_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use zed_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const zed_msgs::msg::Skeleton3D & msg)
{
  return zed_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<zed_msgs::msg::Skeleton3D>()
{
  return "zed_msgs::msg::Skeleton3D";
}

template<>
inline const char * name<zed_msgs::msg::Skeleton3D>()
{
  return "zed_msgs/msg/Skeleton3D";
}

template<>
struct has_fixed_size<zed_msgs::msg::Skeleton3D>
  : std::integral_constant<bool, has_fixed_size<zed_msgs::msg::Keypoint3D>::value> {};

template<>
struct has_bounded_size<zed_msgs::msg::Skeleton3D>
  : std::integral_constant<bool, has_bounded_size<zed_msgs::msg::Keypoint3D>::value> {};

template<>
struct is_message<zed_msgs::msg::Skeleton3D>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ZED_MSGS__MSG__DETAIL__SKELETON3_D__TRAITS_HPP_
