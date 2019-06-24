/// @file
/// @author uentity
/// @date 20.06.2019
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

#include <fstream>

NAMESPACE_BEGIN(blue_sky)

class BS_API tree_fs_input :
	public cereal::InputArchive<tree_fs_input>, public cereal::traits::TextArchive
{
	// main API
public:
	using Base = cereal::InputArchive<tree_fs_input>;
	friend Base;
	// indicate that polymorphic names will always be emitted
	static constexpr auto always_emit_polymorphic_name = true;
	static constexpr auto always_emit_class_version = true;

	tree_fs_input(
		std::string root_dir, std::string root_fname = ".data",
		std::string objects_dir = ".objects"
	);
	~tree_fs_input();

	// retrive stream for archive's head (if any)
	auto head() -> result_or_err<cereal::JSONInputArchive*>;

	auto begin_node(const tree::node& N) -> error;
	auto end_node(const tree::node& N) -> error;

	using object_loader_fn = std::function<error(std::ifstream&, objbase&)>;
	auto install_object_loader(std::string obj_type_id, std::string fmt_descr, object_loader_fn f) -> bool;
	auto can_load_object(std::string_view obj_type_id) const -> bool;

	auto load_object(objbase& obj) -> error;

	auto loadBinaryValue(void* data, size_t size, const char* name = nullptr) -> void;

	// detect types that have empty prologue/epilogue
private:
	template<typename T> struct has_empty {
		static constexpr auto prologue = std::is_same_v<T, std::nullptr_t> |
			std::is_arithmetic_v<T>;
		static constexpr auto epilogue = prologue;
	};

	template<template<typename...> typename T, typename U, typename... Us>
	struct has_empty< T<U, Us...> > {
		using type = T<U, Us...>;

		// detect more types to silence prologue & epilogue
		template<typename X, typename = void>
		struct dispatch { static constexpr auto value = false; };
		// strings
		template<typename Char, typename Traits, typename Alloc, typename Partial>
		struct dispatch< std::basic_string<Char, Traits, Alloc>, Partial > {
			static constexpr auto value = true;
		};
		// unique ptrs
		template<typename X, typename D, typename Partial>
		struct dispatch< std::unique_ptr<X, D>, Partial > {
			static constexpr auto value = std::is_base_of_v<tree::link, std::remove_cv_t<X>>;
		};

		static constexpr auto prologue =
			std::is_same_v<type, cereal::NameValuePair<U>> |
			std::is_same_v<type, cereal::DeferredData<U>> |
			std::is_same_v<type, cereal::SizeTag<U>> |
			// detect shared_ptr to link and derived friends
			(std::is_same_v<type, std::shared_ptr<U>> && std::is_base_of_v<tree::link, std::remove_cv_t<U>>) |
			dispatch<type>::value;

		static constexpr auto epilogue = prologue;
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

	// generic specialization that dispatch args to base or specific processing
	template<typename... Ts>
	inline auto process(Ts&&... ts) -> void {
		const auto dispatch_process = [this](auto&& x) {
			using Tx = decltype(x);
			if constexpr(has_specific_processing_v<Tx>)
				process_specific(std::forward<Tx>(x));
			else
				Base::process(std::forward<Tx>(x));
		};
		(dispatch_process(std::forward<Ts>(ts)), ...);
	}

	// specialization for objects
	template<typename T>
	inline auto process_specific(T&& t) -> void {
	//-> std::enable_if_t<!is_objbase_ptr_v<T>> {
		if(!can_load_object(t.type_id())) {
			install_object_loader(t.type_id(), "bin", [](std::ifstream& os, objbase& obj) -> error {
				cereal::PortableBinaryInputArchive binar(os);
				binar(static_cast< std::add_lvalue_reference_t<std::decay_t<T>> >(obj));
				return success();
			});
		}

		auto cur_head = head();
		cur_head.map([](auto* jar) { jar->startNode(); });
		load_object(std::forward<T>(t));
		cur_head.map([](auto* jar) { jar->finishNode(); });
	}

	struct impl;
	std::unique_ptr<impl> pimpl_;
};

BS_API auto prologue(tree_fs_input& ar, tree::node const& N) -> void;
BS_API auto epilogue(tree_fs_input& ar, tree::node const& N) -> void;

NAMESPACE_END(blue_sky)

/*-----------------------------------------------------------------------------
 *  load overloads for different types - repeat JSONOutputArchive code
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN(cereal)

template<typename T>
inline auto load(blue_sky::tree_fs_input& ar, NameValuePair<T>& t) -> void {
	ar.head().map([&](auto* jar) {
		jar->setNextName(t.name);
		ar(t.value);
	});

}

template<typename T, traits::EnableIf<std::is_arithmetic<T>::value> = traits::sfinae>
inline auto load(blue_sky::tree_fs_input& ar, T& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->loadValue(t);
	});
}

inline auto load(blue_sky::tree_fs_input& ar, std::nullptr_t& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->loadValue(t);
	});
}

template<typename... Args>
inline auto load(blue_sky::tree_fs_input& ar, std::basic_string<Args...>& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->loadValue(t);
	});
}

template<typename T>
inline auto load(blue_sky::tree_fs_input& ar, SizeTag<T>& t) -> void {
	ar.head().map([&t](auto* jar) {
		jar->loadSize(t.size);
	});
}

///////////////////////////////////////////////////////////////////////////////
//  prologue/epilogue for misc types - repeat JSONOutputArchive
//

// write empty prologue/epilogue for corresponding types
template< typename T>
inline auto prologue(blue_sky::tree_fs_input& ar, T const&)
-> std::enable_if_t<blue_sky::tree_fs_input::has_empty_prologue<T>>
{}

template< typename T>
inline auto epilogue(blue_sky::tree_fs_input& ar, T const&)
-> std::enable_if_t<blue_sky::tree_fs_input::has_empty_epilogue<T>>
{}

//! Prologue for all other types for JSON archives (except minimal types)
/*! Starts a new node, named either automatically or by some NVP,
  that may be given data by the type about to be archived
  Minimal types do not start or finish nodes */
template< typename T, traits::EnableIf<
	!blue_sky::tree_fs_input::has_empty_prologue<T>, !std::is_arithmetic_v<T>,
	!traits::has_minimal_base_class_serialization<
		T, traits::has_minimal_output_serialization, blue_sky::tree_fs_input
	>::value,
	!traits::has_minimal_input_serialization<T, blue_sky::tree_fs_input>::value
> = traits::sfinae >
inline void prologue(blue_sky::tree_fs_input& ar, T const&) {
	ar.head().map([](auto* jar) {
		jar->startNode();
	});
}

//! Epilogue for all other types other for JSON archives (except minimal types)
/*! Finishes the node created in the prologue
  Minimal types do not start or finish nodes */
template< typename T, traits::EnableIf<
	!blue_sky::tree_fs_input::has_empty_epilogue<T>, !std::is_arithmetic_v<T>,
	!traits::has_minimal_base_class_serialization<
		T, traits::has_minimal_output_serialization, blue_sky::tree_fs_input
	>::value,
	!traits::has_minimal_input_serialization<T, blue_sky::tree_fs_input>::value
> = traits::sfinae >
inline void epilogue( blue_sky::tree_fs_input& ar, T const&) {
	ar.head().map([](auto* jar) {
		jar->finishNode();
	});
}

NAMESPACE_END(cereal)

CEREAL_REGISTER_ARCHIVE(blue_sky::tree_fs_input)

