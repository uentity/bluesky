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

namespace blue_sky { namespace detail {

// plugin descriptors order by name
struct BS_HIDDEN_API pdp_comp_name {
	bool operator()(const plugin_descriptor* lhs, const plugin_descriptor* rhs) const {
		return lhs->name < rhs->name;
	}
};

// fab_elem sorting order by bs_type_info
struct BS_HIDDEN_API tt_comp_ti {
	bool operator()(const type_tuple& lhs, const type_tuple& rhs) const {
		return lhs.td() < rhs.td();
	}
};

// fab element comparison by string typename
struct BS_HIDDEN_API tt_comp_typename {
	bool operator()(const type_tuple& lhs, const type_tuple& rhs) const {
		return lhs.td().type_name() < rhs.td().type_name();
	}
};

// fab elements comparison by plugin name
struct BS_HIDDEN_API tt_comp_pd {
	bool operator()(const type_tuple& lhs, const type_tuple& rhs) const {
		return (lhs.pd().name < rhs.pd().name);
	}
};

/*-----------------------------------------------------------------
 * Kernel plugins loading subsystem definition
 *----------------------------------------------------------------*/
struct BS_HIDDEN_API kernel_plugins_subsyst {

	const plugin_descriptor& kernel_pd_;       //! plugin descriptor for kernel types
	const plugin_descriptor runtime_pd_;      //! plugin descriptor for runtime types

	//! plugin_descriptor -> lib_descriptor 1-to-1 relation
	using plugins_enum_t = std::map<
		const plugin_descriptor*, lib_descriptor, pdp_comp_name,
		boost::fast_pool_allocator<
			std::pair< const plugin_descriptor*, lib_descriptor >,
			boost::default_user_allocator_new_delete,
			boost::details::pool::null_mutex
		>
	>;
	plugins_enum_t loaded_plugins_;

	// uentity: replaced fab_elem -> type_tuple
	//! types factory: fab_elements sorted by BS_TYPE_INFO
	using types_alloc_t = boost::fast_pool_allocator<
		type_tuple, boost::default_user_allocator_new_delete, boost::details::pool::null_mutex
	>;
	using factory_t = std::set<
		type_tuple, tt_comp_ti, types_alloc_t
	>;
	factory_t obj_fab_;

	//! types dictionary: fe_ptrs sorted by string type
	using types_dict_t = std::set<
		type_tuple, tt_comp_typename, types_alloc_t
	>;
	types_dict_t types_resolver_;

	//! loaded plugins: pointers to type_tuples sorted by plugins names
	using plugin_types_enum_t = std::multiset<
		type_tuple, tt_comp_pd, types_alloc_t
	>;
	plugin_types_enum_t plugin_types_;

#ifdef BSPY_EXPORTING
	struct bspy_module;
	std::unique_ptr< bspy_module > pymod_;
#endif

	// ctor
	kernel_plugins_subsyst();
	~kernel_plugins_subsyst();

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
		return (pd == kernel_pd_ || pd == runtime_pd_);
	}

	bool register_kernel_type(const type_descriptor& td, type_tuple* tt_ref = NULL) {
		return register_type(td, &kernel_pd_, tt_ref);
	}

	bool register_rt_type(const type_descriptor& td, type_tuple* tt_ref = NULL) {
		return register_type(td, &runtime_pd_, tt_ref);
	}

	// register given type
	// return registered type in tp_ref, if not null
	bool register_type(
		const type_descriptor& td, const plugin_descriptor* pd = nullptr, type_tuple* tp_ref = nullptr
	);
	// deny rvalue type descriptors registering
	bool register_type(type_descriptor&&, const plugin_descriptor& = nullptr, type_tuple* = nullptr) = delete;

	// extract type information from stored factory, register as runtime type if wasn't found
	type_tuple demand_type(const type_tuple& obj_t);
};

}} // eof blue_sky::detail namespace

