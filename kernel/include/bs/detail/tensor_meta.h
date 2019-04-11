/// @file
/// @author uentity
/// @date 11.04.2019
/// @brief Set compile-time flags that describe Eigen::Tensor family types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <type_traits>
#include <unsupported/Eigen/CXX11/Tensor>

namespace blue_sky::meta {
namespace detail {

/// Matches Eigen::TensorMap
template<typename T>
struct is_tensor_map : std::false_type {};
template<typename T_, int O_, template<typename> typename M_>
struct is_tensor_map<Eigen::TensorMap<T_, O_, M_>> : std::true_type {};

/// Matches Eigen::TensorRef
template<typename T>
struct is_tensor_ref : std::false_type {};
template<typename T_>
struct is_tensor_ref<Eigen::TensorRef<T_>> : std::true_type {};

/// Matches Eigen::Tensor
template<typename T>
struct is_tensor_plain : std::false_type {};
template<typename S_, int N_, int O_, typename I_>
struct is_tensor_plain<Eigen::Tensor<S_, N_, O_, I_>> : std::true_type {};

} // eof blue_sky::meta::detail

template<typename T> struct tensor {
	static constexpr bool
		is_base    = std::is_base_of_v<Eigen::TensorBase<T, Eigen::ReadOnlyAccessors>, T>,
		is_mutable = std::is_base_of_v<Eigen::TensorBase<T, Eigen::WriteAccessors>, T>,
		is_map     = detail::is_tensor_map<T>::value,
		is_ref     = detail::is_tensor_ref<T>::value,
		is_tensor  = detail::is_tensor_plain<T>::value;

	static constexpr bool
		is_handle  = is_map || is_ref,
		is_mutable_map = is_map && is_mutable,
		is_mutable_ref = is_ref && is_mutable,
		is_mutable_tensor = is_tensor && is_mutable;

	static constexpr bool
		is_mutable_handle = is_handle && is_mutable;
};

} // eof blue_sky::meta
