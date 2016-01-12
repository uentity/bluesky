/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief BlueSky kernel class declaration (API)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_KERNEL_H_
#define BS_KERNEL_H_

#include "bs_common.h"
#include "bs_object_base.h"
#include "bs_abstract_storage.h"
#include "bs_link.h"
#include "bs_report.h"
#include "memory_manager.h"
#include "throw_exception.h"

#include "kernel_signals.h"

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

namespace bs_private {
	struct wrapper_kernel;
}

//type tuple - contains type information coupled with plugin information
struct type_tuple {
	// typedefs to look like std::pair
	typedef plugin_descriptor first_type;
	typedef type_descriptor second_type;

	plugin_descriptor pd_;
	type_descriptor td_;

	type_tuple(const plugin_descriptor& pd, const type_descriptor& td);
};

/*! \class kernel
	*  \ingroup kernel_group
	*  \brief makes access to blue-sky structures
	*
	*         blue-sky kernel makes access to main blue-sky
	*         factory of objects and factory of commands
	*/
class BS_API kernel
{
	friend struct bs_private::wrapper_kernel;

public:
	typedef std::vector< plugin_descriptor > plugins_enum;
	typedef std::vector< type_descriptor > types_enum;

	//! data table with string key
	typedef mt_ptr< data_table< bs_map, str_val_traits > > str_dt_ptr;
	//! indexed data table
	typedef mt_ptr< data_table< bs_array, vector_traits > > idx_dt_ptr;

	//access to plugins & types from them
	//! \brief loaded plugins
	plugins_enum loaded_plugins() const;
	//! \brief types of plugin (by plugin descriptor)
	types_enum plugin_types(const plugin_descriptor& pd) const;
	types_enum plugin_types(const std::string& plugin_name) const;
	//! \brief Find type by type_str
	type_tuple find_type(const std::string& type_str) const;

	//! \brief registered type infos of objects
	std::vector< type_tuple > registered_types() const;

	//! \brief register type in kernel with this function
	bool register_type(const plugin_descriptor& pd, const type_descriptor& td) const;

	//! \brief object creation method
	//! Contains auto-registration for unknown types
	sp_obj create_object(const type_descriptor& obj_t, bool unmanaged = unmanaged_def_val(), bs_type_ctor_param param = NULL) const;
	//! Supply maximum info about type to create (useful for auto-registering with proper plugin_descriptor)
	//sp_obj create_object(const type_descriptor& td, const plugin_descriptor& pd,
	//	bool unmanaged = false, bs_type_ctor_param param = NULL) const;
	//! Use this method if type_descriptor is unknown by any reason
	sp_obj create_object(const std::string& obj_t, bool unmanaged = unmanaged_def_val(), bs_type_ctor_param param = NULL) const;

	//! \brief Objects copying method
	sp_obj create_object_copy(const sp_obj& src, bool unmanaged = unmanaged_def_val()) const;

	//! \brief Registers object in managed instances lists
	int register_instance(const sp_obj&) const;
	//! \brief Removes object from managed instances lists
	int free_instance(const sp_obj&) const;

	//! \brief Deletes all dangling objects that aren't contained in data storage
	//! This is a garbage collection method
	//! Call it to force system cleaning, although all should be cleaned automatically
	ulong tree_gc() const;

	//! \brief access to list of given type's instances
	bs_objinst_holder::const_iterator objinst_begin(const type_descriptor& td) const;
	bs_objinst_holder::const_iterator objinst_end(const type_descriptor& td) const;
	ulong objinst_cnt(const type_descriptor& td) const;

	sp_storage create_storage(const std::string &filename, const std::string &format, int flags) const;
	void close_storage(const sp_storage &storage) const;

	//! \brief Load blue-sky plugins method
	error_code LoadPlugins(bool init_py_subsyst = false) const;
	//! \brief Unload blue-sky plugins method
	void UnloadPlugins() const;
	//! \brief Dynamically loads particular plugin
	//! \param version - desired plugin version, if left blank no version checking is performed
	error_code load_plugin(const std::string& fname, const std::string version, bool init_py_subsyst = false);
	//! \brief Unloads plugin
	void unload_plugin(const plugin_descriptor& pd);

	//! access to global table of different types with string key
	//str_dt_ptr global_dt() const;
	//! access to per-type tables
	str_dt_ptr pert_str_dt(const type_descriptor& obj_t) const;
	idx_dt_ptr pert_idx_dt(const type_descriptor& obj_t) const;

	std::string get_last_error() const;

	void test() const;

	//push task to the FIFO queue for processing
	void add_task(const sp_com& task);
	//
	bool is_tq_empty() const;
	void wait_tq_empty() const;
	//

	//access to the root of BlueSky tree
	sp_link bs_root() const;

	//! Method for registering and quering signals with given code for given object type
	//! returns true if signal was really created
	std::pair< sp_signal, bool > reg_signal(const BS_TYPE_INFO& obj_t, int signal_code) const;

	//! Method for erasing given signal from given type
	bool rem_signal(const BS_TYPE_INFO& obj_t, int signal_code) const;

	memory_manager& get_memory_manager () {
		return memory_manager_;
	}

	static bs_log &
	get_log ();

	static thread_log &
	get_tlog ();

	void
	register_disconnector (signals_disconnector *d);

	void
	unregister_disconnector (signals_disconnector *d);

private:
	//! \brief Constructor of kernel
	kernel();
	//! \brief Copy constructor of kernel
	kernel(const kernel& k);
	//! \brief Destructor.
	~kernel();

	static bool unmanaged_def_val();

	//! PIMPL for kernel
	class kernel_impl;
	typedef mt_ptr< kernel_impl > pimpl_t;

	// don't change line order. never.
	pimpl_t         pimpl_;
	memory_manager  memory_manager_;

	std::vector <signals_disconnector *> disconnectors_;

	// kernel initialization routine
	void init();
	// called before kernel dies
	void cleanup();
};

//! \brief singleton for accessing the instance of kernel
typedef singleton< kernel > give_kernel;

inline bool operator ==(const type_tuple& tl, const type_tuple& tr) {
	return (tl.pd_ == tr.pd_ && tl.td_ == tl.td_);
}

inline bool operator !=(const type_tuple& tl, const type_tuple& tr) {
		return !(tl.pd_ == tr.pd_ && tl.td_ == tl.td_);
}

} // eof blue_sky namespace

#endif

