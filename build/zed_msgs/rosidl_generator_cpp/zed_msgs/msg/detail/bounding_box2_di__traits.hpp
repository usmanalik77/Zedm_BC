// generated from rosidl_generator_cpp/resource/idl__traits.hpp.em
// with input from zed_msgs:msg/BoundingBox2Di.idl
// generated code does not contain a copyright notice

#ifndef ZED_MSGS__MSG__DETAIL__BOUNDING_BOX2_DI__TRAITS_HPP_
#define ZED_MSGS__MSG__DETAIL__BOUNDING_BOX2_DI__TRAITS_HPP_

#include <stdint.h>

#include <sstream>
#include <string>
#include <type_traits>

#include "zed_msgs/msg/detail/bounding_box2_di__struct.hpp"
#include "rosidl_runtime_cpp/traits.hpp"

// Include directives for member types
// Member 'corners'
#include "zed_msgs/msg/detail/keypoint2_di__traits.hpp"

namespace zed_msgs
{

namespace msg
{

inline void to_flow_style_yaml(
  const BoundingBox2Di & msg,
  std::ostream & out)
{
  out << "{";
  // member: corners
  {
    if (msg.corners.size() == 0) {
      out << "corners: []";
    } else {
      out << "corners: [";
      size_t pending_items = msg.corners.size();
      for (auto item : msg.corners) {
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
  const BoundingBox2Di & msg,
  std::ostream & out, size_t indentation = 0)
{
  // member: corners
  {
    if (indentation > 0) {
      out << std::string(indentation, ' ');
    }
    if (msg.corners.size() == 0) {
      out << "corners: []\n";
    } else {
      out << "corners:\n";
      for (auto item : msg.corners) {
        if (indentation > 0) {
          out << std::string(indentation, ' ');
        }
        out << "-\n";
        to_block_style_yaml(item, out, indentation + 2);
      }
    }
  }
}  // NOLINT(readability/fn_size)

inline std::string to_yaml(const BoundingBox2Di & msg, bool use_flow_style = false)
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
  const zed_msgs::msg::BoundingBox2Di & msg,
  std::ostream & out, size_t indentation = 0)
{
  zed_msgs::msg::to_block_style_yaml(msg, out, indentation);
}

[[deprecated("use zed_msgs::msg::to_yaml() instead")]]
inline std::string to_yaml(const zed_msgs::msg::BoundingBox2Di & msg)
{
  return zed_msgs::msg::to_yaml(msg);
}

template<>
inline const char * data_type<zed_msgs::msg::BoundingBox2Di>()
{
  return "zed_msgs::msg::BoundingBox2Di";
}

template<>
inline const char * name<zed_msgs::msg::BoundingBox2Di>()
{
  return "zed_msgs/msg/BoundingBox2Di";
}

template<>
struct has_fixed_size<zed_msgs::msg::BoundingBox2Di>
  : std::integral_constant<bool, has_fixed_size<zed_msgs::msg::Keypoint2Di>::value> {};

template<>
struct has_bounded_size<zed_msgs::msg::BoundingBox2Di>
  : std::integral_constant<bool, has_bounded_size<zed_msgs::msg::Keypoint2Di>::value> {};

template<>
struct is_message<zed_msgs::msg::BoundingBox2Di>
  : std::true_type {};

}  // namespace rosidl_generator_traits

#endif  // ZED_MSGS__MSG__DETAIL__BOUNDING_BOX2_DI__TRAITS_HPP_
