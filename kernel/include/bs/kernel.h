/// @file
/// @author uentity
/// @date 10.08.2016
/// @brief BlueSky kernel class declaration (API)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "common.h"
#include "objbase.h"
#include "log.h"

//#define BS_AUTOLOAD_PLUGINS

/*!
	\brief Macro for registering your objects in blue-sky.

	Generates proper call to kernel's register_type function.

	\param plugin_descr - descriptor of your plugin. Can be taken, for example, from plugin_initializer that is passed to bs_register_plugin
	\param T = class name
	\param creation_fun = creation function pointer
 */
#define BLUE_SKY_REGISTER_TYPE(plugin_descr, T) \
blue_sky::give_kernel::Instance().register_type(plugin_descr, T::bs_type())

//! \defgroup kernel_group kernel group - kernel of blue-sky
namespace blue_sky {

namespace detail {
	struct wrapper_kernel;
}

// type tuple - contains type information coupled with plugin information
struct type_tuple {
	// typedefs to look like std::pair
	typedef plugin_descriptor first_type;
	typedef type_descriptor second_type;

	plugin_descriptor pd;
	type_descriptor td;

	type_tuple(const plugin_descriptor& pd_, const type_descriptor& td_)
		: pd(pd_), td(td_)
	{}
};

inline bool operator ==(const type_tuple& tl, const type_tuple& tr) {
	return (tl.pd == tr.pd && tl.td == tl.td);
}

inline bool operator !=(const type_tuple& tl, const type_tuple& tr) {
	return !(tl.pd == tr.pd && tl.td == tl.td);
}

/*! \class kernel
	*  \ingroup kernel_group
	*  \brief makes access to blue-sky structures
	*
	*         blue-sky kernel makes access to main blue-sky
	*         factory of objects and factory of commands
	*/
class BS_API kernel {
	friend struct detail::wrapper_kernel;

public:
	typedef std::vector< plugin_descriptor > plugins_enum;
	typedef std::vector< type_descriptor > types_enum;

	//access to plugins & types from them
	//! \brief loaded plugins
	plugins_enum loaded_plugins() const;

	//! \brief Deletes all dangling objects that aren't contained in data storage
	//! This is a garbage collection method
	//! Call it to force system cleaning, although all should be cleaned automatically
	ulong tree_gc() const;

	//! \brief Dynamically loads particular plugin
	int load_plugin(const std::string& fname, bool init_py_subsyst);
	//! \brief Load blue-sky plugins method
	int load_plugins(void* py_root_module = nullptr);
	//! \brief Unloads plugin
	void unload_plugin(const plugin_descriptor& pd);
	//! \brief Unload blue-sky plugins method
	void unload_plugins();

	std::string last_error() const;

	static spdlog::logger& get_log(const char* name);

private:
	//! \brief Constructor of kernel
	kernel();
	//! \brief Copy constructor of kernel
	kernel(const kernel& k);
	//! \brief Destructor.
	~kernel();

	//! PIMPL for kernel
	class kernel_impl;
	std::unique_ptr< kernel_impl > pimpl_;

	// kernel initialization routine
	void init();
	// called before kernel dies
	void cleanup();
};

//! \brief singleton for accessing the instance of kernel
typedef singleton< kernel > give_kernel;

} // eof blue_sky namespace

