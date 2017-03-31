/// @file
/// @author uentity
/// @date 21.10.2016
/// @brief Kernel types factory functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/exception.h>
#include <bs/log.h>
#include "kernel_plugins_subsyst.h"

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(detail)

using namespace std;

bool kernel_plugins_subsyst::register_type(
	const type_descriptor& td, const plugin_descriptor* pd, type_tuple* tt_ref
) {
	if(td.is_nil()) return false;

	// if no plugin descriptor provided - register as runtime type
	const plugin_descriptor* pdp = pd ? pd : &runtime_pd_;
	if(!is_inner_pd(*pdp)) {
		// case for external plugin descriptor
		// try to register it or find a match with existing pd
		pdp = register_plugin(pdp, lib_descriptor()).first;
	}

	// register obj in factory
	auto res = types_.insert(type_tuple{*pdp, td});
	const type_tuple& tar_tt = *res.first;

	// replace plugin_descriptor if passed one is not nil and differ from existing
	if(!res.second && !pd->is_nil() && *pd != tar_tt.pd()) {
		types_.replace(res.first, {*pd, tar_tt.td()});
		res.second = true;
	}
	if(!res.second) {
		// dump warning
		bserr() << log::W("[kernel:factory] type '{}' already registered") << td.name << bs_end;
	}

	// save registered type if asked for
	if(tt_ref) *tt_ref = tar_tt;
	return res.second;
}

bool kernel_plugins_subsyst::register_type(
	const type_descriptor& td, const std::string& plug_name, type_tuple* tt_ref
) {
	if(!plug_name.size()) return false;

	// check if plugin with given name already registered
	plugin_descriptor tplug(plug_name);
	auto reg_plug = loaded_plugins_.find(&tplug);
	if(reg_plug != loaded_plugins_.end()) {
		return register_type(td, reg_plug->first, tt_ref);
	}
	// register temp plugin with given name
	auto ttplug = temp_plugins_.insert(std::move(tplug));
	// and finally register type
	return register_type(td, &*ttplug.first, tt_ref);
}

type_tuple kernel_plugins_subsyst::demand_type(const type_tuple& obj_t) {
	//type_tuple tt_ref(obj_t);
	type_tuple tt_ref;
	if(obj_t.td().is_nil()) {
		// if type is nil try to find it by name
		auto tt = types_.get< type_name_key >().find(obj_t.td());
		if(tt != types_.get< type_name_key >().end())
			tt_ref = *tt;
	}
	else {
		// otherwise try to find requested type using fast search in factory
		auto tt = types_.get< type_key >().find(obj_t.td());
		if(tt != types_.get< type_key >().end())
			tt_ref = *tt;
	}
	if(tt_ref.td().is_nil()) {
		//type wasn't found - try to register it first
		if(obj_t.pd().is_nil())
			register_rt_type(obj_t.td(), &tt_ref);
		else
			register_type(obj_t.td(), &obj_t.pd(), &tt_ref);
		// still nil td means that serious error happened - type cannot be registered
		if(tt_ref.is_nil()) {
			throw bs_kexception(boost::format(
				"Type (%s) is nil or cannot be registered!") % obj_t.td().name,
				"kernel:factory"
			);
		}
	}
	return tt_ref;
}

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

