/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Plugins subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/kernel.h>
#include <bs/detail/lib_descriptor.h>
#include <boost/pool/pool_alloc.hpp>
#include <map>
#include <set>

#include <boost/multi_index_container.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index/mem_fun.hpp>
//#include <boost/multi_index/member.hpp>
//#include <boost/multi_index/identity.hpp>

namespace blue_sky { namespace detail {
namespace mi = boost::multi_index;

// plugin descriptors order by name
struct BS_HIDDEN_API pdp_comp_name {
	bool operator()(const plugin_descriptor* lhs, const plugin_descriptor* rhs) const {
		return lhs->name < rhs->name;
	}
};

struct BS_HIDDEN_API pd_comp_name {
	bool operator()(const plugin_descriptor& lhs, const plugin_descriptor& rhs) const {
		return lhs.name < rhs.name;
	}
};

/*-----------------------------------------------------------------
 * Kernel plugins loading subsystem definition
 *----------------------------------------------------------------*/
struct BS_HIDDEN_API kernel_plugins_subsyst {

	static const plugin_descriptor& kernel_pd();
	static const plugin_descriptor& runtime_pd();

	// plugin_descriptor -> lib_descriptor 1-to-1 relation
	// IMPORTANT: descriptors are stored by string name - thus allowing us
	// to store multiple nil plugins with different names
	// and later rebind to valid descriptor
	using plugins_enum_t = std::map<
		const plugin_descriptor*, lib_descriptor, pdp_comp_name,
		boost::fast_pool_allocator<
			typename std::map<const plugin_descriptor*, lib_descriptor>::value_type,
			//std::pair< const plugin_descriptor*, lib_descriptor >,
			boost::default_user_allocator_new_delete,
			boost::details::pool::null_mutex
		>
	>;
	plugins_enum_t loaded_plugins_;

	// storage for temp plugin descriptors created when only plugin name specified
	std::set< plugin_descriptor, pd_comp_name > temp_plugins_;

	// type_tuple allocator
	using types_alloc_t = boost::fast_pool_allocator<
		type_tuple, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
	>;
	// tags for type_tuples storage keys
	using type_key = mi::const_mem_fun< type_tuple, const type_descriptor&, &type_tuple::td >;
	using plug_key = mi::const_mem_fun< type_tuple, const plugin_descriptor&, &type_tuple::pd >;
	using type_name_key = mi::const_mem_fun< type_tuple, const std::string&, &type_tuple::type_name >;
	using plug_name_key = mi::const_mem_fun< type_tuple, const std::string&, &type_tuple::plug_name >;

	// type tuples storage type
	using types_container_t = mi::multi_index_container<
		type_tuple,
		mi::indexed_by<
			mi::ordered_unique< mi::tag< type_key >, type_key >,
			mi::ordered_unique< mi::tag< type_name_key >, type_name_key >,
			mi::ordered_non_unique< mi::tag< plug_key >, plug_key >,
			mi::ordered_non_unique< mi::tag< plug_name_key >, plug_name_key >
		>,
		types_alloc_t
	>;
	types_container_t types_;

#ifdef BSPY_EXPORTING
	struct bspy_module;
	std::unique_ptr< bspy_module > pymod_;
#endif

	// ctor
	kernel_plugins_subsyst();
	~kernel_plugins_subsyst();

	// allow obtain kernel's plugin descriptor
	friend const plugin_descriptor* bs_get_plugin_descriptor();

	/*-----------------------------------------------------------------
	 * plugins managing
	 *----------------------------------------------------------------*/
	void clean_plugin_types(const plugin_descriptor& pd);

	void unload_plugin(const plugin_descriptor& pd);

	void unload_plugins();

	std::pair< const plugin_descriptor*, bool >
	register_plugin(const plugin_descriptor* pd, const lib_descriptor& ld);

	int load_plugin(const std::string& fname, bool init_py_subsyst);

	int load_plugins(void* py_root_module);

	/*-----------------------------------------------------------------
	 * types management
	 *----------------------------------------------------------------*/
	bool is_inner_pd(const plugin_descriptor& pd) {
		return (pd == kernel_pd() || pd == runtime_pd());
	}

	bool register_kernel_type(const type_descriptor& td, type_tuple* tt_ref = nullptr) {
		return register_type(td, &kernel_pd(), tt_ref);
	}

	bool register_rt_type(const type_descriptor& td, type_tuple* tt_ref = nullptr) {
		return register_type(td, &runtime_pd(), tt_ref);
	}

	// register given type
	// return registered type in tp_ref, if not null
	bool register_type(
		const type_descriptor& td, const plugin_descriptor* pd = nullptr, type_tuple* tp_ref = nullptr
	);
	bool register_type(
		const type_descriptor& td, const std::string& plug_name, type_tuple* tp_ref = nullptr
	);
	// deny rvalue type descriptors registering
	bool register_type(type_descriptor&&, const plugin_descriptor* = nullptr, type_tuple* = nullptr) = delete;
	bool register_type(type_descriptor&&, const std::string&, type_tuple* = nullptr) = delete;

	// extract type information from stored factory, register as runtime type if wasn't found
	type_tuple demand_type(const type_tuple& obj_t);

	// unite serialization code among all loaded plugins
	void unify_serialization() const;
};

}} // eof blue_sky::detail namespace

