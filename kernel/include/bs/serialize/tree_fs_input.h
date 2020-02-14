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

NAMESPACE_BEGIN(blue_sky)

class BS_API tree_fs_input :
	public cereal::InputArchive<tree_fs_input>, public cereal::traits::TextArchive
{
	// main API
public:
	using Base = cereal::InputArchive<tree_fs_input>;
	friend Base;
	// tweak serialization behaviour to better support out-of-order loading
	static constexpr auto always_emit_class_version = true;
	static constexpr auto custom_node_serialization = true;

	// 'Normal' - node is reconstructed exactly from leafs stored with it's handle link.
	// 'Recover' - for any node N all link files found inside N's directory
	// will be loaded into N.
	enum class NodeLoad { Normal, Recover };

	tree_fs_input(std::string root_fname, NodeLoad mode = NodeLoad::Normal);
	~tree_fs_input();

	// retrive stream for archive's head (if any)
	auto head() -> result_or_err<cereal::JSONInputArchive*>;

	auto begin_node() -> error;
	auto end_node(const tree::node& N) -> error;

	auto load_object(objbase& obj) -> error;

	auto loadBinaryValue(void* data, size_t size, const char* name = nullptr) -> void;

	// detect types that have empty prologue/epilogue
private:
	template<typename T> struct has_empty {
		static constexpr auto prologue = std::is_same_v<T, std::nullptr_t> |
			std::is_arithmetic_v<T> |
			// match any link
			std::is_base_of_v<tree::link, std::remove_cv_t<T>>;
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
			// match shared_ptr to link and derived friends
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

	// detect pure objects (not nodes)
	template<typename T>
	static constexpr auto is_object_v = std::is_base_of_v<objbase, T> && !std::is_same_v<tree::node, T>;

	// generic specialization that dispatch args to base or specific processing
	template<typename... Ts>
	inline auto process(Ts&&... ts) -> void {
		const auto dispatch_process = [this](auto&& x) {
			using Tx = decltype(x);
			if constexpr(is_object_v<std::decay_t<Tx>>)
				this->load_object(std::forward<Tx>(x));
			else
				Base::process(std::forward<Tx>(x));
		};
		(dispatch_process(std::forward<Ts>(ts)), ...);
	}

	struct impl;
	std::unique_ptr<impl> pimpl_;
};

// make prologue/epilogue for different possible node load impl
// either 1st or 2nd set will be called
// 1. without `load_and_construct`
BS_API auto prologue(tree_fs_input& ar, tree::node const& N) -> void;
BS_API auto epilogue(tree_fs_input& ar, tree::node const& N) -> void;
// 2. via `load_and construct`
BS_API auto prologue(
	tree_fs_input& ar, cereal::memory_detail::LoadAndConstructLoadWrapper<tree_fs_input, tree::node> const& N
) -> void;
BS_API auto epilogue(
	tree_fs_input& ar, cereal::memory_detail::LoadAndConstructLoadWrapper<tree_fs_input, tree::node> const& N
) -> void;

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

// empty prologue/epilogue for corresponding types
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

