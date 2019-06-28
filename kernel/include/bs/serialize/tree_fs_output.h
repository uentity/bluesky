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
#include "atomizer.h"

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/archives/portable_binary.hpp>

#include <optional>
#include <fstream>

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

	tree_fs_output(
		std::string root_dir, std::string root_fname = ".data",
		std::string objects_dir = ".objects"
	);
	~tree_fs_output();

	// retrive stream for archive's head (if any)
	auto head() -> result_or_err<cereal::JSONOutputArchive*>;

	auto begin_link(const tree::sp_link& L) -> error;
	auto end_link() -> error;

	auto begin_node(const tree::node& N) -> error;
	auto end_node() -> error;

	using object_saver_fn = std::function<error(std::ofstream&, const objbase&)>;
	auto install_object_saver(std::string obj_type_id, std::string fmt_descr, object_saver_fn f) -> bool;
	auto can_save_object(std::string_view obj_type_id) const -> bool;

	auto save_object(const objbase& obj) -> error;

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

	// detect specific processing overloads
	template<typename T, typename = void> struct has_specific_processing : std::false_type {};
	template<typename T>
	static constexpr auto has_specific_processing_v = has_specific_processing<std::decay_t<T>>::value;

	template<typename T> struct has_specific_processing<T, std::enable_if_t<
		std::is_base_of_v<objbase, T> && !std::is_same_v<tree::node, T>
	>> : std::true_type {};

	// generic specialization that forwards everything to base archive
	template<typename... Ts>
	inline auto process(Ts&&... ts) -> void {
		const auto dispatch_process = [this](auto&& x) {
			using Tx = decltype(x);
			if constexpr(has_specific_processing_v<Tx>)
				this->process_specific(std::forward<Tx>(x));
			else
				Base::process(std::forward<Tx>(x));
		};
		(dispatch_process(std::forward<Ts>(ts)), ...);
	}

	// specialization for objects
	template<typename T>
	inline auto process_specific(T&& t) -> void {
		if(!can_save_object(t.type_id())) {
			install_object_saver(t.type_id(), "bin", [](std::ofstream& os, const objbase& obj) -> error {
				cereal::PortableBinaryOutputArchive binar(os);
				//cereal::JSONOutputArchive binar(os);
				binar(static_cast< std::add_lvalue_reference_t<const std::decay_t<T>> >(obj));
				return success();
			});
		}

		auto cur_head = head();
		cur_head.map([](auto* jar) { jar->startNode(); });
		save_object(std::forward<T>(t));
		cur_head.map([](auto* jar) { jar->finishNode(); });
	}

	struct impl;
	std::unique_ptr<impl> pimpl_;
};

BS_API auto prologue(tree_fs_output& ar, tree::sp_link const& L) -> void;
BS_API auto epilogue(tree_fs_output& ar, tree::sp_link const&) -> void;

BS_API auto prologue(tree_fs_output& ar, tree::node const& N) -> void;
BS_API auto epilogue(tree_fs_output& ar, tree::node const&) -> void;

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

// empty prologue/epilogue for corresponding types
template< typename T>
inline auto prologue(blue_sky::tree_fs_output& ar, T const&)
-> std::enable_if_t<blue_sky::tree_fs_output::has_empty_prologue<T>>
{}

template< typename T>
inline auto epilogue(blue_sky::tree_fs_output& ar, T const&)
-> std::enable_if_t<blue_sky::tree_fs_output::has_empty_epilogue<T>>
{}

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
