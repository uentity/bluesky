/// @file
/// @author uentity
/// @date 21.12.2018
/// @brief BS kernel helper type tuple (type_descriptor + plugin_descriptor)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "../type_descriptor.h"
#include "../plugin_descriptor.h"

NAMESPACE_BEGIN(blue_sky::kernel::tfactory)

// type tuple - contains type information coupled with plugin information
struct BS_API type_tuple : public std::tuple< const plugin_descriptor*, const type_descriptor* > {
	using base_t = std::tuple< const plugin_descriptor*, const type_descriptor* >;
	using pd_t = const plugin_descriptor&;
	using td_t = const type_descriptor&;

	// construct from lvalue refs to plugin_descriptor & type_descriptor
	template<
		typename P, typename T,
		typename = std::enable_if_t<
			!std::is_rvalue_reference<P&&>::value && !std::is_rvalue_reference<T&&>::value
		>
	>
	type_tuple(P&& plug, T&& type) : base_t(&plug, &type) {}

	// ctors accepting only plugin_descriptor or only type_descriptor
	// uninitialized value will be nil
	type_tuple(pd_t plug) : base_t(&plug, &type_descriptor::nil()) {};
	type_tuple(td_t type) : base_t(&plugin_descriptor::nil(), &type) {}
	// deny constructing from rfavlue refs
	type_tuple(plugin_descriptor&&) = delete;
	type_tuple(type_descriptor&&) = delete;

	// empty ctor creates nil type_tuple
	type_tuple() : base_t(&plugin_descriptor::nil(), &type_descriptor::nil()) {}

	bool is_nil() const {
		return pd().is_nil() && td().is_nil();
	}

	pd_t pd() const {
		return *std::get< 0 >(*this);
	}
	td_t td() const {
		return *std::get< 1 >(*this);
	}

	// direct access to plugin & type names
	// simplifies boost::multi_index_container key specification
	const std::string& plug_name() const {
		return pd().name;
	}
	const std::string& type_name() const {
		return td().name;
	}
};

NAMESPACE_END(blue_sky::kernel::tfactory)
