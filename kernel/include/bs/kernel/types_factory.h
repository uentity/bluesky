/// @file
/// @author uentity
/// @date 21.12.2018
/// @brief BS kernel types factory API
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "../common.h"
#include "type_tuple.h"

/// `tfactory` = types factory
NAMESPACE_BEGIN(blue_sky::kernel::tfactory)

NAMESPACE_BEGIN(detail)

// extract type_descriptor for given type from internal kernel storage
BS_API auto demand_type(const type_descriptor& obj_type) -> const type_descriptor&;

NAMESPACE_END(detail)

BS_API auto register_type(const type_descriptor& td, const plugin_descriptor* pd = nullptr) -> bool;
// register type with plugin_descriptor specified by name
// types will bind to real plugin descriptor when it will be loaded
BS_API auto register_type(const type_descriptor& td, const std::string& plug_name) -> bool;
// no way to register rvalue (temp) type_descriptor
auto register_type(type_descriptor&&, const plugin_descriptor* = nullptr) -> bool = delete;
auto register_type(type_descriptor&&, const std::string&) -> bool = delete;
// Find type by type name
BS_API auto find_type(const std::string& type_name) -> type_tuple;

// create instance of object
template< typename Obj_type_spec, typename... Args >
auto create_object(Obj_type_spec&& obj_type, Args&&... ctor_args) {
	auto& td = detail::demand_type(type_descriptor(std::forward< Obj_type_spec >(obj_type)));
	return td.construct(std::forward< Args >(ctor_args)...);
}

// clone object
BS_API auto clone_object(bs_type_copy_param source) -> type_descriptor::shared_ptr_cast;

// store and access instances of BS types
BS_API auto register_instance(sp_cobj obj) -> int;

template< class T >
auto register_instance(T* obj) {
	// check that T is inherited from objbase
	static_assert(
		std::is_base_of< objbase, T >::value,
		"Only blue_sky::objbase derived instances can be registered!"
	);
	return register_instance(obj->shared_from_this());
}

BS_API auto free_instance(sp_cobj obj) -> int;

template< class T >
auto free_instance(T* obj) {
	// check that T is inherited from objbase
	static_assert(
		std::is_base_of< objbase, T >::value,
		"Only blue_sky::objbase derived instances can be registered!"
	);
	return free_instance(obj->shared_from_this());
}

using instances_enum = std::vector< sp_cobj >;
BS_API auto instances(const type_descriptor& obj_t) -> instances_enum;
BS_API auto instances(std::string_view type_id) -> instances_enum;

template< typename T >
auto instances() {
	return instances(T::bs_type());
}

NAMESPACE_END(blue_sky::kernel::tfactory)
