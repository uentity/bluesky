/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief Kernel types factory API impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/kernel/types_factory.h>
#include <bs/objbase.h>
#include "kimpl.h"

NAMESPACE_BEGIN(blue_sky::kernel::tfactory)

NAMESPACE_BEGIN(detail)

// extract type_descriptor for given type from internal kernel storage
auto demand_type(const type_descriptor& obj_type) -> const type_descriptor& {
	return KIMPL.demand_type({obj_type}).td();
}

NAMESPACE_END(detail)

auto register_type(const type_descriptor& td, const plugin_descriptor* pd) -> bool {
	return KIMPL.register_type(td, pd);
}

auto register_type(const type_descriptor& td, const std::string& plug_name) -> bool {
	return KIMPL.register_type(td, plug_name);
}

auto find_type(const std::string& type_name) -> type_tuple {
	return KIMPL.find_type(type_name);
}

auto clone_object(bs_type_copy_param source) -> type_descriptor::shared_ptr_cast {
	return source->bs_resolve_type().clone(source);
}

auto register_instance(sp_cobj obj) -> int {
	return KIMPL.register_instance(std::move(obj));
}

auto free_instance(sp_cobj obj) -> int {
	return KIMPL.free_instance(std::move(obj));
}

auto instances(const BS_TYPE_INFO& ti) -> instances_enum {
	return KIMPL.instances(ti);
}

NAMESPACE_END(blue_sky::kernel::tfactory)
