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

// dummy size callback for deserialization
constexpr auto noop_size_callback(std::size_t) -> void {}

// test if we serialize arithmetic types & archive supports binary format
template<typename Archive, typename T>
inline constexpr auto binary_carray_support =
	traits::is_output_serializable<BinaryData<T>, Archive>::value &&
	std::is_arithmetic_v<std::remove_all_extents_t<T>>;

} // eof detail

///////////////////////////////////////////////////////////////////////////////
//  load/save paths for C arrays
//
template<typename Archive, typename T>
inline auto save_carray(Archive& ar, T* array, const std::size_t size) -> void {
	ar( make_size_tag(size) );
	if constexpr(detail::binary_carray_support<Archive, T>)
		ar( binary_data(array, sizeof(T) * size) );
	else {
		for(std::size_t i = 0; i < size; ++i)
			ar(array[i]);
	}
}

// [NOTE] F is a functor that will be called with size passed in before elements are loaded
template<typename Archive, typename T, typename F = decltype(detail::noop_size_callback)>
inline auto load_carray(Archive& ar, T* array, const F& f = detail::noop_size_callback) -> void {
	// read number of elements and invoke f(size)
	std::size_t size;
	ar( make_size_tag(size) );
	f(size);
	// read data
	if constexpr(detail::binary_carray_support<Archive, T>)
		ar( binary_data(array, sizeof(T) * size) );
	else {
		for(std::size_t i = 0; i < size; ++i)
			ar(array[i]);
	}
}

///////////////////////////////////////////////////////////////////////////////
//  united serialize for C arrays
//
template< typename Archive, typename T, typename F = decltype(detail::noop_size_callback) > inline
auto serialize_carray(
	Archive& ar, T* array, const std::size_t size, const F& size_callback = detail::noop_size_callback
) -> void {
	// invoke serialization op
	if constexpr(Archive::is_saving::value)
		save_carray(ar, array, size);
	else
		load_carray(ar, array, size_callback);
}

namespace detail {

///////////////////////////////////////////////////////////////////////////////
//  proxy view class for C arrays
//
template<typename T, typename F = decltype(noop_size_callback)>
struct carray_view {
	T* data_;
	const std::size_t size_;
	const std::decay_t<F> size_cb_;

	explicit carray_view(T* data, std::size_t size, F f = noop_size_callback)
		: data_(data), size_(size),  size_cb_(std::move(f))
	{}

	// main serialization functions
	template<typename Archive>
	auto save(Archive& ar) const -> void {
		save_carray(ar, data_, size_);
	}

	template<typename Archive>
	auto load(Archive& ar) -> void {
		load_carray(ar, data_, size_cb_);
	}
};

} // eof namespace detail

///////////////////////////////////////////////////////////////////////////////
//  return proxy class for saving C array as standalone 'unit'
//
template< typename T, typename F = decltype(detail::noop_size_callback) > inline
auto make_carray_view(T* array, std::size_t size, F size_callback = detail::noop_size_callback) {
	return detail::carray_view<T>(array, size, std::move(size_callback));
}

} // eof namespace cereal
