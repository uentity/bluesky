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

//bool kernel_plugins_subsyst::register_kernel_type(const type_descriptor& td, fab_elem* tp_ref) {
//	return register_type(td, &kernel_pd_, tp_ref);
//}
//
//bool kernel_plugins_subsyst::register_rt_type(const type_descriptor& td, fab_elem* tp_ref) {
//	return register_type(td, &runtime_pd_, tp_ref);
//}

bool kernel_plugins_subsyst::register_type(
	const type_descriptor& td, const plugin_descriptor* pd, type_tuple* tt_ref
) {
	if(td.is_nil()) return false;

	//pair< factory_t::iterator, bool > res;

	// find correct plugin_descriptor
	// pointer to registered plugin_descriptor
	const plugin_descriptor* pdp = pd;
	// if now plugin descriptor provided - register as runtime type
	if(pdp->is_nil())
		pdp = &runtime_pd_;
	if(!is_inner_pd(*pdp)) {
		// case for external plugin descriptor
		// try to register it or find a match with existing pd
		pdp = register_plugin(pdp, lib_descriptor()).first;
	}
	// register obj in factory
	auto res = obj_fab_.insert(type_tuple(*pdp, td));
	// save registered type if asked for
	if(tt_ref) *tt_ref = *res.first;

	//register type in dictionaries
	if(res.second) {
		auto res_ref = types_resolver_.insert(*res.first);
		if(!res_ref.second) {
			//probably duplicating type name found
			// dump error
			bserr() << log::W("[kernel:factory] type '") << td.type_name()
				<< "' cannot be registered because type with such name already exist" << bs_end;

			obj_fab_.erase(res.first);
			if(tt_ref) {
				if(res_ref.first != types_resolver_.end())
					*tt_ref = *res_ref.first;
				else
					//some unknown bad error happened
					*tt_ref = type_tuple();
			}
			return false;
		}

		plugin_types_.insert(*res.first);
		return true;
	}
	else if(*pdp != res.first->pd() && is_inner_pd(res.first->pd())) {
		// current type was previously registered as inner
		// replace with correct plugin d-tor now
		// remove first from types_resolver_
		types_resolver_.erase(*res.first);
		// remove inner-type association
		plugin_types_.erase(*res.first);
		// now delete factory entry
		obj_fab_.erase(res.first);

		// register type with passed plugin_descriptor
		res = obj_fab_.insert(type_tuple({*pd, td}));
		types_resolver_.insert(*res.first);
		plugin_types_.insert(*res.first);
		return true;
	}

	// dump warning
	bserr() << log::W("[kernel:factory] type '{}' already registered") << td.type_name() << bs_end;
	return false;
}

type_tuple kernel_plugins_subsyst::demand_type(const type_tuple& obj_t) {
	type_tuple tt_ref(obj_t);
	if(obj_t.td().is_nil()) {
		// if type is nil try to find it by name
		auto tt = types_resolver_.find(tt_ref);
		if(tt != types_resolver_.end())
			tt_ref = *tt;
	}
	else {
		// otherwise try to find requested type using fast search in factory
		auto tt = obj_fab_.find(tt_ref);
		if(tt != obj_fab_.end())
			tt_ref = *tt;
		else
			tt_ref = type_tuple();
	}
	if(tt_ref.td().is_nil()) {
		//type wasn't found - try to register it first
		if(obj_t.pd().is_nil())
			register_rt_type(obj_t.td(), &tt_ref);
		else
			register_type(obj_t.td(), &obj_t.pd(), &tt_ref);
		//still nil td means that serious error happened - type cannot be registered
		if(tt_ref.is_nil()) {
			throw bs_kexception(boost::format(
				"Type (%s) is nil or cannot be registered!") % obj_t.td().type_name(),
				"kernel:factory"
			);
		}
	}
	return tt_ref;
}

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

