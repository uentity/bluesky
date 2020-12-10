/// @file
/// @author uentity
/// @date 28.06.2018
/// @brief Serialization support for C-style arrays
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <cereal/cereal.hpp>

/*-----------------------------------------------------------------------------
 *  serialization support for C arrays
 *  [NOTE] saved array can be loaded into std::vector, because format is fully compatible
 *-----------------------------------------------------------------------------*/

namespace cereal {
namespace detail {

template<typename T>
inline constexpr auto noop_size_callback = [](T* data, std::size_t) { return data; };

// test if we serialize arithmetic types & archive supports binary format
template<typename Archive, typename T>
inline constexpr auto binary_carray_support = (
	traits::is_output_serializable<BinaryData<T>, Archive>::value
	|| traits::is_input_serializable<BinaryData<T>, Archive>::value
) && (
	std::is_arithmetic_v<std::remove_all_extents_t<T>>
	|| std::is_enum_v<std::remove_all_extents_t<T>>
);

inline constexpr char const* data_nvp_name = "data";

template<size_type N, typename Archive, typename Array>
auto serialize_array(Archive& ar, Array& array, const char* name) -> void {
	using T = typename Array::value_type;
	constexpr auto size = sizeof(T) * N;
	if constexpr(detail::binary_carray_support<Archive, T>)
		ar(binary_data( array.begin(), size ));
	else {
		if constexpr(Archive::is_saving::value)
			ar.saveBinaryValue(array.begin(), size, name);
		else
			ar.loadBinaryValue(array.begin(), size, name);
	}
}

} // eof detail

///////////////////////////////////////////////////////////////////////////////
//  load/save paths for C arrays
//
template<typename Archive, typename T>
auto save_carray(
	Archive& ar, T* array, const std::size_t size, const char* name = detail::data_nvp_name
) -> void {
	ar(make_nvp("size", size));
	if constexpr(detail::binary_carray_support<Archive, T>)
		ar(binary_data( array, size * sizeof(T) ));
	else
		ar.saveBinaryValue(array, size * sizeof(T), name);
}

// [NOTE] F is a functor that will be called with size passed in before elements are loaded
template< typename Archive, typename T, typename F = decltype((detail::noop_size_callback<T>)) >
auto load_carray(
	Archive& ar, T* array, F&& f = detail::noop_size_callback<T>, const char* name = detail::data_nvp_name
) -> void {
	// read number of elements and invoke f(size)
	std::size_t size;
	ar(make_nvp("size", size));
	array = f(array, size);
	// read data
	if constexpr(detail::binary_carray_support<Archive, T>)
		ar(binary_data( array, size * sizeof(T) ));
	else
		ar.loadBinaryValue(array, size * sizeof(T), name);
}

///////////////////////////////////////////////////////////////////////////////
//  united serialize for C arrays
//
template< typename Archive, typename T, typename F = decltype((detail::noop_size_callback<T>)) >
auto serialize_carray(
	Archive& ar, T* array, [[maybe_unused]] const std::size_t size,
	[[maybe_unused]] F&& f = detail::noop_size_callback<T>, const char* name = detail::data_nvp_name
) -> void {
	// invoke serialization op
	if constexpr(Archive::is_saving::value)
		save_carray(ar, array, size, name);
	else
		load_carray(ar, array, std::forward<F>(f), name);
}

///////////////////////////////////////////////////////////////////////////////
//  std::array support
//
template<typename Archive, typename T, std::size_t N>
auto serialize_carray(
	Archive& ar, const std::array<T, N>& array, const char* name = detail::data_nvp_name
) -> std::enable_if_t<Archive::is_saving::value, void> {
	detail::serialize_array<N>(ar, array, name);
}

template<typename Archive, typename T, std::size_t N>
auto serialize_carray(
	Archive& ar, std::array<T, N>& array, const char* name = detail::data_nvp_name
) -> std::enable_if_t<Archive::is_loading::value, void> {
	detail::serialize_array<N>(ar, array, name);
}

///////////////////////////////////////////////////////////////////////////////
//  array descriptor proxy
//
namespace detail {

template<typename T, typename F>
struct carray_view {
	T* data;
	const std::size_t size;
	F&& f;

	// main serialization functions
	template<typename Archive>
	auto serialize(Archive& ar) const -> void {
		serialize_carray(ar, data, size, std::forward<F>(f));
	}
};

} // eof detail

///////////////////////////////////////////////////////////////////////////////
//  return proxy class for saving C array as standalone 'unit'
//
template< typename T, typename F = decltype((detail::noop_size_callback<T>)) >
auto make_carray_view(T* array, std::size_t size, F f = detail::noop_size_callback<T>) {
	return detail::carray_view<T, F>{array, size, std::forward<F>(f)};
}

} // eof namespace cereal
