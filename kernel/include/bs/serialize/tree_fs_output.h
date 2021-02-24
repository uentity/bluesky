/// @file
/// @author uentity
/// @date 29.05.2019
/// @brief Archive that represents BS tree in filesystem
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../error.h"
#include "../meta.h"
#include "../timetypes.h"
#include "serialize_decl.h"
#include "object_formatter.h"

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>

NAMESPACE_BEGIN(blue_sky)

class BS_API tree_fs_output :
	public cereal::OutputArchive<tree_fs_output>, public cereal::traits::TextArchive
{
	// main API
public:
	using Base = cereal::OutputArchive<tree_fs_output>;
	friend Base;
	// indicate that polymorphic names will always be emitted
	static constexpr auto always_emit_polymorphic_name = true;
	static constexpr auto always_emit_class_version = true;
	static constexpr auto custom_node_serialization = true;
	static constexpr auto default_opts = TFSOpts::ClearDirs;

	tree_fs_output(std::string root_fname, TFSOpts opts = default_opts);
	~tree_fs_output();

	// retrive stream for archive's head
	auto head() -> result_or_err<cereal::JSONOutputArchive*>;

	auto begin_link(const tree::link& L) -> error;
	auto end_link(const tree::link& L) -> error;

	auto begin_node(const tree::node& N) -> error;
	auto end_node(const tree::node& N) -> error;

	auto save_object(const objbase& obj, bool has_node) -> error;
	auto wait_objects_saved(timespan how_long = infinite) const -> std::vector<error>;

	auto get_active_formatter(std::string_view obj_type_id) -> object_formatter*;
	auto select_active_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> bool;

	auto saveBinaryValue(const void* data, size_t size, const char* name = nullptr) -> void;

	// detect types that have empty prologue/epilogue
private:
	template<typename T> struct has_empty {
		static constexpr auto prologue = false;
		static constexpr auto epilogue = std::is_same_v<T, std::nullptr_t> |
			std::is_arithmetic_v<T>;
	};

	template<template<typename...> typename T, typename U, typename... Us>
	struct has_empty< T<U, Us...> > {
		using type = T<U, Us...>;

		// detect more types to silence prologue & epilogue
		template<typename X, typename = void>
		struct dispatch {
			static constexpr auto prologue = false;
			static constexpr auto epilogue = false;
		};
		// strings
		template<typename Char, typename Traits, typename Alloc, typename Partial>
		struct dispatch< std::basic_string<Char, Traits, Alloc>, Partial > {
			static constexpr auto prologue = false;
			static constexpr auto epilogue = true;
		};

		static constexpr auto prologue =
			std::is_same_v<type, cereal::NameValuePair<U>> |
			std::is_same_v<type, cereal::DeferredData<U>> |
			dispatch<type>::prologue;

		static constexpr auto epilogue = prologue |
			std::is_same_v<type, cereal::SizeTag<U>> |
			dispatch<type>::epilogue;
	};

public:
	template<typename T>
	static constexpr auto has_empty_prologue = has_empty<T>::prologue;

	template<typename T>
	static constexpr auto has_empty_epilogue = has_empty<T>::epilogue;

	// details
private:
	friend class ::blue_sky::atomizer;

	// detect objects, but skips exactly `objbase` & `objnode`
	template<typename T>
	static constexpr auto is_object_v = std::is_base_of_v<objbase, T> &&
		!std::is_same_v<objbase, T> && !std::is_same_v<objnode, T>;

	// calls `save_object()` for every BS object, except `objbase` & `objnode`
	template<typename... Ts>
	inline auto process(Ts&&... ts) -> void {
		const auto dispatch_process = [this](auto&& x) {
			using Tx = decltype(x);
			if constexpr(is_object_v<meta::remove_cvref_t<Tx>>)
				this->save_object(
					std::forward<Tx>(x), std::is_base_of_v<objnode, meta::remove_cvref_t<Tx>>
				);
			else
				Base::process(std::forward<Tx>(x));
		};
		(dispatch_process(std::forward<Ts>(ts)), ...);
	}

	struct impl;
	std::unique_ptr<impl> pimpl_;
};

BS_API auto prologue(tree_fs_output& ar, tree::link const& L) -> void;
BS_API auto epilogue(tree_fs_output& ar, tree::link const& L) -> void;

BS_API auto prologue(tree_fs_output& ar, tree::node const& N) -> void;
BS_API auto epilogue(tree_fs_output& ar, tree::node const& N) -> void;

NAMESPACE_END(blue_sky)

/*-----------------------------------------------------------------------------
 *  save overloads for different types - repeat JSONOutputArchive code
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(cereal)

template<typename T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline auto save(blue_sky::tree_fs_output& ar, T const & t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->saveValue(t);
	});
}

template<typename T>
inline auto save(blue_sky::tree_fs_output& ar, NameValuePair<T> const& t ) -> void {
	ar.head().map([&](auto* jar) {
		jar->setNextName(t.name);
		ar(t.value);
	});

}

inline auto save(blue_sky::tree_fs_output& ar, std::nullptr_t const& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->saveValue(t);
	});
}

template<typename... Args>
inline auto save(blue_sky::tree_fs_output& ar, std::basic_string<Args...> const& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->saveValue(t);
	});
}

// nothing to do here, we don't explicitly save the size
template<typename T>
inline auto save(blue_sky::tree_fs_output&, SizeTag<T> const&) -> void {}

///////////////////////////////////////////////////////////////////////////////
//  prologue/epilogue for misc types - repeat JSONOutputArchive
//

//! Prologue for SizeTags for JSON archives
/*! SizeTags are strictly ignored for JSON, they just indicate
	that the current node should be made into an array */
template< typename T>
inline void prologue(blue_sky::tree_fs_output& ar, SizeTag<T> const&) {
	ar.head().map([](auto* jar) {
		jar->makeArray();
	});
}

//! Prologue for all other types for JSON archives (except minimal types)
/*! Starts a new node, named either automatically or by some NVP,
  that may be given data by the type about to be archived
  Minimal types do not start or finish nodes */
template< typename T, traits::EnableIf<
	!blue_sky::tree_fs_output::has_empty_prologue<T>, !std::is_arithmetic_v<T>,
	!traits::has_minimal_base_class_serialization<
		T, traits::has_minimal_output_serialization, blue_sky::tree_fs_output
	>::value,
	!traits::has_minimal_output_serialization<T, blue_sky::tree_fs_output>::value
> = traits::sfinae >
inline void prologue(blue_sky::tree_fs_output& ar, T const &) {
	ar.head().map([](auto* jar) {
		jar->startNode();
	});
}

//! Epilogue for all other types other for JSON archives (except minimal types)
/*! Finishes the node created in the prologue
  Minimal types do not start or finish nodes */
template< typename T, traits::EnableIf<
	!blue_sky::tree_fs_output::has_empty_epilogue<T>, !std::is_arithmetic_v<T>,
	!traits::has_minimal_base_class_serialization<
		T, traits::has_minimal_output_serialization, blue_sky::tree_fs_output
	>::value,
	!traits::has_minimal_output_serialization<T, blue_sky::tree_fs_output>::value
> = traits::sfinae >
inline void epilogue( blue_sky::tree_fs_output& ar, T const &) {
	ar.head().map([](auto* jar) {
		jar->finishNode();
	});
}

//! Prologue for arithmetic types for JSON archives
inline void prologue(blue_sky::tree_fs_output & ar, std::nullptr_t const &) {
	ar.head().map([](auto* jar) {
		jar->writeName();
	});
}

//! Prologue for arithmetic types for JSON archives
template<typename T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline void prologue( blue_sky::tree_fs_output& ar, T const &) {
	ar.head().map([](auto* jar) {
		jar->writeName();
	});
}

//! Prologue for strings for JSON archives
template<class CharT, class Traits, class Alloc>
inline void prologue(blue_sky::tree_fs_output& ar, std::basic_string<CharT, Traits, Alloc> const &) {
	ar.head().map([](auto* jar) {
		jar->writeName();
	});
}

NAMESPACE_END(cereal)

CEREAL_REGISTER_ARCHIVE(blue_sky::tree_fs_output)
