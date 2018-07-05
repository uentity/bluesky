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

namespace {
	// dummy size callback for deserialization
	auto noop_size_callback(std::size_t) -> void {}
}

///////////////////////////////////////////////////////////////////////////////
// Serialization for C arrays if BinaryData is supported and we are arithmetic
//
template< typename Archive, typename T > inline
auto save_carray(
	Archive& ar, T* array, const std::size_t size, std::true_type
) -> void {
	ar( make_size_tag(size) ); // number of elements
	ar( binary_data(array, sizeof(T) * size) );
}

// [NOTE] F is a functor that will be called with size passed in before elements are loaded
template< typename Archive, typename T, typename F = decltype(detail::noop_size_callback) > inline
auto load_carray(
	Archive& ar, T* array, std::true_type, F f = noop_size_callback
) -> void {
	// read number of elements and invoke f(size)
	std::size_t size;
	ar( make_size_tag(size) );
	f(size);
	// read data
	ar( binary_data(array, sizeof(T) * size) );
}

///////////////////////////////////////////////////////////////////////////////
// Serialization for arrays if BinaryData is not supported or we are not arithmetic
//
template< class Archive, class T > inline
auto save_carray(Archive& ar, T* array, const std::size_t size, std::false_type /* binary_supported */ )
-> void {
	ar( make_size_tag(size) ); // number of elements
	for(std::size_t i = 0; i < size; ++i)
		ar(array[i]);
}

// [NOTE] F is a functor that will be called with size passed in before elements are loaded
template< typename Archive, typename T, typename F = decltype(detail::noop_size_callback) > inline
auto load_carray(
	Archive& ar, T* array, std::false_type, F f = noop_size_callback
) -> void {
	// read number of elements and invoke f(size)
	std::size_t size;
	ar( make_size_tag(size) );
	f(size);
	// read data
	for(std::size_t i = 0; i < size; ++i)
		ar(array[i]);
}

///////////////////////////////////////////////////////////////////////////////
//  proxy view class for C arrays
//
template<typename T>
struct carray_view {
	T* data_;
	std::size_t size_;
	using size_callback_f = void (*)(std::size_t);
	const size_callback_f size_cb_;

	template<typename F = decltype(detail::noop_size_callback)>
	explicit carray_view(T* data, std::size_t size, F f = noop_size_callback)
		: data_(data), size_(size),  size_cb_(f)
	{}

	// test if archive supports binary data serialization
	template<typename Archive> using binary_supported = std::integral_constant<
		bool, traits::is_output_serializable<BinaryData<T>, Archive>::value &&
		std::is_arithmetic<typename std::remove_all_extents<T>::type>::value
	>;

	// main serialization functions
	template<typename Archive>
	auto save(Archive& ar) const -> void {
		save_carray(ar, data_, size_, binary_supported<Archive>());
	}

	template<typename Archive>
	auto load(Archive& ar) -> void {
		load_carray(ar, data_, binary_supported<Archive>(), size_cb_);
	}
};

} // eof namespace detail

///////////////////////////////////////////////////////////////////////////////
//  load/save paths for C arrays
//
template< typename Archive, typename T > inline
auto save_carray(Archive& ar, T* array, const std::size_t size) -> void {
	detail::save_carray(ar, array, size, detail::carray_view<T>::template binary_supported<Archive>());
}

template< typename Archive, typename T, typename F = decltype(detail::noop_size_callback) > inline
auto load_carray(Archive& ar, T* array, F size_callback = detail::noop_size_callback) -> void {
	detail::load_carray(
		ar, array, detail::carray_view<T>::template binary_supported<Archive>(), size_callback
	);
}

///////////////////////////////////////////////////////////////////////////////
//  single serialize for C arrays
//
template< typename Archive, typename T, typename F = decltype(detail::noop_size_callback) > inline
auto serialize_carray(
	Archive& ar, T* array, const std::size_t size, F size_callback = detail::noop_size_callback
) -> void {
	// helper to select save or load op depending on archive
	struct select_saveload {
		// save
		static auto go(Archive& ar_, T* array_, const std::size_t size_, F, std::true_type)
		-> void {
			detail::save_carray(
				ar_, array_, size_, detail::carray_view<T>::template binary_supported<Archive>())
			;
		}
		// load
		static auto go(Archive& ar_, T* array_, const std::size_t size_, F f, std::false_type)
		-> void{
			detail::load_carray(
				ar_, array_, detail::carray_view<T>::template binary_supported<Archive>(), f
			);
		}
	};
	// invoke serialization op
	select_saveload::go(ar, array, size, size_callback, typename Archive::is_saving());
}

///////////////////////////////////////////////////////////////////////////////
//  return proxy class for saving C array as standalone 'unit'
//
template< typename T, typename F = decltype(detail::noop_size_callback) > inline
auto make_carray_view(T* array, std::size_t size, F size_callback = detail::noop_size_callback) {
	return detail::carray_view<T>(array, size, size_callback);
}

} // eof namespace cereal

