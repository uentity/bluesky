// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
 * \file bs_kernel.cpp
 * \brief Contains blue-sky kernel implementation.
 * \author uentity
 */

#include "bs_kernel.h"
#include "bs_misc.h"
#include "bs_exception.h"
#include "thread_pool.h"
#include "bs_prop_base.h"
#include "bs_tree.h"
#include "bs_report.h"
#include "bs_log_scribers.h"

#include <stdio.h>
//#include <iostream>
#include <list>
#include <map>
#include <set>

#ifdef _WIN32
#include <windows.h>
#include <Psapi.h>
#elif defined(UNIX)
#include <dlfcn.h>
#endif

//boost library
//#include "boost/throw_exception.hpp"
#include "boost/filesystem/operations.hpp"
#include "boost/filesystem/exception.hpp"
#include "boost/filesystem/path.hpp"
#include "boost/pool/pool_alloc.hpp"
//boost::python
#include "boost/python/module_init.hpp"
#include "boost/python/scope.hpp"
#include "boost/python/class.hpp"
#include "boost/python/str.hpp"
#include "boost/python/exec.hpp"
#include "boost/python/import.hpp"

//Loki
//#include "loki/AssocVector.h"
//#include "loki/Factory.h"
#include "loki/Singleton.h"


using namespace std;
using namespace boost;
using namespace boost::filesystem;
//using namespace blue_sky;

#define KERNEL_VERSION "0.1" //!< version of blue-sky kernel

#define COUT_LOG_INST log::Instance()[COUT_LOG] //!< blue-sky log for simple cout output
#define XPN_LOG_INST log::Instance()[XPN_LOG] //!< blue-sky log for errors output
#define MAIN_LOG_INST log::Instance()[MAIN_LOG] //!< blue-sky log for every output

//export specific BlueSky kernel's plugin_descriptor
BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("BlueSky kernel", "1.0RC4", "Plugin tag for BlueSky kernel", "", "bs")

// define default value for unmanaged parameter to create_object
bool blue_sky::kernel::unmanaged_def_val() {
#ifdef BS_CREATE_UNMANAGED_OBJECTS
	return true;
#else
	return false;
#endif
}
//std::less specialization for plugin_descriptor
namespace std {
	template< >
	bool less< blue_sky::plugin_descriptor >::operator ()(
		const blue_sky::plugin_descriptor& lhs,
		const blue_sky::plugin_descriptor& rhs) const
	{
		return (lhs.name_ < rhs.name_);
	}
}

//all implementation contained in blue_sky namespace
namespace blue_sky {

type_tuple::type_tuple(const plugin_descriptor& pd, const type_descriptor& td)
	: pd_(pd), td_(td)
{}

//---------------helper structures--------------------------------------------------------------------
/*!
	\struct lib
	\brief Necessary for dynamic libraries handlers
 */
struct lib_descriptor
{
	string fname_; //!< path to dynamic library
#ifdef UNIX
	void * handle_; //!< handler of dynamic library object
#elif defined(_WIN32)
	HMODULE handle_;
#endif

	lib_descriptor()
		: fname_(""), handle_(NULL)
	{}

	bool load(const char* fname) {
		fname_ = fname;
	#ifdef UNIX
		handle_ = dlopen(fname, RTLD_GLOBAL | RTLD_NOW);
	#else
		handle_ = LoadLibrary(LPCSTR(fname));
	#endif
    if (!handle_)
      {
        throw bs_dynamic_lib_exception ("LoadPlugin");
      }

		return (bool)handle_;
	}

	void unload() {
		//unload library
		if(handle_) {
	#ifdef UNIX
			dlclose(handle_);
	#elif defined(_WIN32)
			FreeLibrary(handle_);
	#endif
		}
		handle_ = NULL;
	}

	template< typename fn_t >
	bool load_sym(const char* sym_name, fn_t& fn) const {
		if(handle_) {
	#ifdef UNIX
			fn = (fn_t)dlsym(handle_, sym_name);
	#else
			fn = (fn_t)GetProcAddress(handle_, LPCSTR(sym_name));
	#endif
		}
		else fn = NULL;
		return (bool)fn;
	}

	template< class sym_t >
	static int load_sym_glob(const char* sym_name, sym_t& sym) {
#ifdef UNIX
		sym = (sym_t)dlsym(RTLD_DEFAULT, sym_name);
		return 0;
#elif defined(_WIN32)
		//helper struct to find BlueSky kernel among all loaded modules
		struct find_kernel_module {
			static HMODULE go() {
				HANDLE hproc;
				HMODULE hmods[1024];
				DWORD cbNeeded;
				BS_GET_PLUGIN_DESCRIPTOR pdf = NULL;
				plugin_descriptor* pd = NULL;
				ulong m_ind = 0;

				//get handle of current process
				hproc = GetCurrentProcess();

				//enumerate all modules of current process
				HMODULE res = NULL;
				if(hproc && EnumProcessModules(hproc, hmods, sizeof(hmods), &cbNeeded))	{
					ulong cnt = cbNeeded / sizeof(HMODULE);
					for (ulong i = 0; i < cnt; ++i) {
						//search for given symbol in i-th module
						if(pdf = (BS_GET_PLUGIN_DESCRIPTOR)GetProcAddress(hmods[i], "bs_get_plugin_descriptor")) {
							//get pointer to plugin_descriptor & check if this is a kernel
							if((pd = pdf()) && pd->name_.compare("BlueSky kernel") == 0) {
								res = hmods[i];
								break;
							}
						}
					}
					CloseHandle(hproc);
				}
				return res;
			}
		};

		//get kernel module handle
		static HMODULE km = find_kernel_module::go();

		sym = NULL;
		if(!km) return 1;

		//search for given symbol
		sym = (sym_t)GetProcAddress(km, sym_name);
		return 0;
#endif
	}

	static string lib_sys_msg() {
	#ifdef UNIX
		return dlerror();
	#else
		return last_system_message();
	#endif
	}
};

//lib_descriptors are comparable by file name
bool operator <(const lib_descriptor& lhs, const lib_descriptor& rhs) {
	return lhs.fname_ < rhs.fname_;
}

// equality operator for lib's handlers
bool operator ==(const lib_descriptor& left, const lib_descriptor& right) {
	return left.fname_ == right.fname_;
}

template< class T >
struct elem_ptr {
	typedef T elem_t;
	//nil ctor
	elem_ptr() : p_(&nil_el) {}

	elem_ptr(const T& el) : p_(&el) {}

	elem_ptr& operator =(const T& el) {
		p_ = &el;
		return *this;
	}

	const T* operator ->() const { return p_; }

	//conversion to elem_t
	const T& elem() const { return *p_; }
	const T& operator *() const { return *p_; }

	bool is_nil() const {
		return p_->is_nil();
	}

	operator bool() const {
		return !is_nil();
	}

	static const T nil_el;

private:
	const T* p_;
};
//alias
typedef elem_ptr< plugin_descriptor > pd_ptr;

//typedef type_tuple fab_elem;
struct fab_elem {
	// nil ctor
	fab_elem() {}

	fab_elem(const pd_ptr& pd, const type_descriptor& td)
		: pd_(pd), td_(td)
	{}
	//the same but from tuple
	fab_elem(const type_tuple& tt)
		: pd_(tt.pd_), td_(tt.td_)
	{}

	fab_elem(const pd_ptr& pd)
		: pd_(pd)
	{}

	fab_elem(const type_descriptor& td)
		: td_(td)
	{}

	bool is_nil() const {
		return (td_.is_nil() && pd_.is_nil());
	}

	operator bool() const {
		return !is_nil();
	}

	operator type_tuple() const {
		return type_tuple(*pd_, td_);
	};

	pd_ptr pd_;
	type_descriptor td_;
};
//alias
typedef elem_ptr< fab_elem > fe_ptr;

//static nil elements for elem_ptrs
template< > const plugin_descriptor pd_ptr::nil_el = plugin_descriptor();
template< > const fab_elem fe_ptr::nil_el = fab_elem();

//pd_ptrs comparison by name
struct pdp_comp_name {
	bool operator()(const pd_ptr& lhs, const pd_ptr& rhs) const {
		return (lhs->name_ < rhs->name_);
	}
};

//fab_elem sorting order by bs_type_info
struct fe_comp_ti {
	bool operator()(const fab_elem& lhs, const fab_elem& rhs) const {
		return (lhs.td_ < rhs.td_);
	}
};

//fab_elem pointers comparison functions for sorting in containers
struct fep_comp_ti {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->td_ < rhs->td_);
	}
};

// fab element comparison by string typename
struct fep_comp_stype {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->td_.stype_ < rhs->td_.stype_);
	}
};

//fab elements comparison by plugin name
struct fep_comp_pd {
	bool operator()(const fe_ptr& lhs, const fe_ptr& rhs) const {
		return (lhs->pd_->name_ < rhs->pd_->name_);
	}
};

//signals comparator
struct sp_sig_comp {
	bool operator()(const sp_signal& lhs, const sp_signal& rhs) const {
		return (lhs->get_code() < rhs->get_code());
	}
};

//----------------------kernel_impl----------------------------------------------------------------------------
/*!
	\class kernel::kernel_impl
	\ingroup kernel_group
	\brief kernel's some methods implementor
 */
class BS_HIDDEN_API kernel::kernel_impl
{
public:
	//typedefs

	//! type of lsmart_ptr<pimpl_t>
	typedef lsmart_ptr< pimpl_t > impl_locker;
	//! mutex for kernel_impl locking
	static bs_mutex guard_;

	//! plugin_descriptor <-> lib_descriptor 1-to-1 relation
	typedef fast_pool_allocator< pair< plugin_descriptor, lib_descriptor >, default_user_allocator_new_delete,
		details::pool::null_mutex > libs_allocator;
	typedef map< plugin_descriptor, lib_descriptor, less< plugin_descriptor >, libs_allocator > pl_enum;

	//! plugin_descriptors sorted by name
	typedef fast_pool_allocator< pd_ptr, default_user_allocator_new_delete, details::pool::null_mutex > pl_dict_allocator;
	typedef multiset< pd_ptr, pdp_comp_name, pl_dict_allocator > pl_dict;

	//! types factory: fab_elements sorted by BS_TYPE_INFO
	typedef fast_pool_allocator< fab_elem, default_user_allocator_new_delete,
		details::pool::null_mutex > fab_allocator;
	typedef set< fab_elem, fe_comp_ti, fab_allocator > factory;

	//! types dictionary: fe_ptrs sorted by string type
	typedef fast_pool_allocator< fe_ptr, default_user_allocator_new_delete,
		details::pool::null_mutex > dict_allocator;
	typedef set< fe_ptr, fep_comp_stype, dict_allocator > types_dict;

	//! loaded plugins: pointers to type_tuples sorted by plugins names
	typedef multiset< fe_ptr, fep_comp_pd, dict_allocator > plt_enum;

	pl_enum loaded_plugins_;
	pl_dict pl_dict_;
	plt_enum plugin_types_;

	factory obj_fab_;
	types_dict types_resolver_;

	//! data storage for pairs string key - value
	//str_data_table data_tbl_;
	//per-type data storages
	typedef fast_pool_allocator< pair< fe_ptr, sp_obj >, default_user_allocator_new_delete,
		details::pool::null_mutex > tbl_allocator;
	map< fe_ptr, smart_ptr< str_data_table >, fep_comp_ti, tbl_allocator > pert_str_tbl_;
	map< fe_ptr, smart_ptr< idx_data_table >, fep_comp_ti, tbl_allocator > pert_idx_tbl_;

	//! instances list for every registered type
	//typedef fast_pool_allocator< pair< fe_ptr, sp_objinst >, default_user_allocator_new_delete,
	//	details::pool::null_mutex > sp_objinst_allocator;
	typedef map< BS_TYPE_INFO, bs_objinst_holder > objinst_map_t;
	objinst_map_t instances_;

	//! list of signal registered per type
	// allcoator for sp_signal
	typedef fast_pool_allocator< pair< int, sp_signal >, default_user_allocator_new_delete, details::pool::null_mutex > sig_map_alloc;
	// signals map
	typedef map< int, sp_signal, std::less<int>, sig_map_alloc > signals_map;
	// 1-1 mapping type --> signals map
	typedef map< BS_TYPE_INFO, signals_map > sig_storage_t;
	// at last a storage itself
	sig_storage_t sig_storage_;

	//! plugin descriptor tag fot kernel types
	static plugin_descriptor& kernel_pd_;
	//! plugin descriptor tag for runtime types
	static plugin_descriptor runtime_pd_;

	//last error message stored here
	string last_err_;

	string lib_dir_; //!< Current library loading directory
	list< lload > cft_; //!< list of depth-search graph of loading order

	//! thread pool of workers
	worker_thread_pool wtp_;

	//root of objects tree
	sp_link root_;

/*-----------------------------------------------------------------------------
 * kernel_impl ctor
 *-----------------------------------------------------------------------------*/

	//constructor
	kernel_impl()
	{
		//register inner plugin_descriptors in dictionary
		pl_dict_.insert(kernel_pd_);
		pl_dict_.insert(runtime_pd_);
		//register str_data_table object
		register_kernel_type(str_data_table::bs_type());
		//register idx_data_table object
		register_kernel_type(idx_data_table::bs_type());

		register_kernel_type(empty_storage::bs_type());
		//get_lib_list(cft_);
		//cleanup dictionaries
		//clean_plugins();

		//register BlueSky signal type
		//register_kernel_type(bs_signal::bs_type());
	}

	//Myers Singleton for kernel_impl
	//static kernel_impl& self() {
	//	static kernel_impl ki;
	//	return ki;
	//}

	~kernel_impl() {
		string s = "~kernel_impl called";
		cout << s << endl;
	}

	//kernel initialization routine
	void init() {
		//create root of object tree
		root_ = bs_link::create(bs_node::create_node(), "/");
		//create system subdirectories of root
		lsmart_ptr< sp_node > lp_root(root_->node());
		//hidden .system dir
		lp_root->insert(bs_node::create_node(), ".system", true);
		//etc dir
		lp_root->insert(bs_node::create_node(), "etc", true);
		//temp dir
		lp_root->insert(bs_node::create_node(), "temp", true);
		//mnt dir
		lp_root->insert(bs_node::create_node(), "mnt", true);
		//var dir
		lp_root->insert(bs_node::create_node(), "var", true);
		//misc dir
		lp_root->insert(bs_node::create_node(), "misc", true);
	}

	//lock method
	bs_locker< kernel_impl > lock() const {
		return bs_locker< kernel_impl >(this, guard_);
	}

	//access to instances list
	bs_objinst_holder::const_iterator objinst_begin(const BS_TYPE_INFO& ti) {
		return instances_[ti].begin();
	}

	bs_objinst_holder::const_iterator objinst_end(const BS_TYPE_INFO& ti) {
		return instances_[ti].end();
	}

	ulong objinst_cnt(const BS_TYPE_INFO& ti) const {
		objinst_map_t::const_iterator ilist = instances_.find(ti);
		if(ilist != instances_.end())
			return static_cast< ulong >(ilist->second.size());
		else
			return 0;
	}

	//register instance of any BlueSky type
	int register_instance(const sp_obj& obj) {
		if(!obj) return 0;
		type_descriptor td = obj->bs_resolve_type();
		//go through chain of type_descriptors up to objbase
		int arity = 0;
		while(!td.is_nil()) {
			arity += instances_[td.type()].insert(obj).second;
			td = td.parent_td();
		}
		return arity;
	}

	int free_instance(const sp_obj& obj) {
		if(!obj) return 0;

		//if object is dangling, ie it's inode has zero hard links,
		//delete the inode (remove reference to obj)
		//if(obj->inode_ && obj->inode_->links_count() == 0)
		//	obj.lock()->inode_.release();

		//go through chain of type_descriptors up to objbase
		type_descriptor td = obj->bs_resolve_type();
		int arity = 0;
		while(!td.is_nil()) {
			arity += (int)instances_[td.type()].erase(obj);
			td = td.parent_td();
		}
		return arity;
	}

	ulong tree_gc() {
		bs_objinst_holder::iterator p_obj = instances_[objbase::bs_type().type()].begin(),
			end = instances_[objbase::bs_type().type()].end(),
			tmp;
		ulong cnt = 0;
		while(p_obj != end) {
			if(!(*p_obj)->inode() || (*p_obj)->inode()->links_count() == 0) {
				tmp = p_obj;
				++p_obj;
				free_instance(*tmp);
				++cnt;
			}
			else ++p_obj;
		}
		return cnt;
	}

	void clean_plugin_tails(const plugin_descriptor& pd) {
		//we cannot clear kernel internal types
		if(pd == kernel_pd_) return;
		//get all types of given plugin
		pair< plt_enum::iterator, plt_enum::iterator > pl_types =
			plugin_types_.equal_range(fab_elem(pd));
		//clear these types
		for(plt_enum::const_iterator p = pl_types.first; p != pl_types.second; ++p) {
			//const fab_elem& type2kill = tp->elem();
			types_resolver_.erase(*p);
			obj_fab_.erase(**p);
		}
		plugin_types_.erase(pl_types.first, pl_types.second);
	}

	void clean_plugins() {
		pair< plt_enum::iterator, plt_enum::iterator > pl_types;
		//iterate among loaded plugins
		for(pl_enum::const_iterator p = loaded_plugins_.begin(), p_end = loaded_plugins_.end(); p != p_end; ++p) {
			//for each plugin clear it's types
			pl_types = plugin_types_.equal_range(fab_elem(p->first));
			for(plt_enum::const_iterator t = pl_types.first; t != pl_types.second; ++t) {
				//const fab_elem& type2kill = tp->elem();
				types_resolver_.erase(*t);
				obj_fab_.erase(**t);
			}
			plugin_types_.erase(pl_types.first, pl_types.second);
			//clear plugin's dictionary
			pl_dict_.erase(p->first);
		}
	}

	//loads plugin from given library
	error_code load_plugin(const string& fname, const string& version, bool init_py_subsyst = false);
	//loads all found plugins
	error_code load_plugins(bool init_py_subsyst = false);
	//unloads given plugin
	void unload_plugin(const plugin_descriptor& pd) {
		clean_plugin_tails(pd);
		pl_dict_.erase(pd);
		loaded_plugins_[pd].unload();
		loaded_plugins_.erase(pd);
	}
	//unloads all plugins
	void unload_plugins() {
		//clean all types registered by plugins
		clean_plugins();

		// close all loaded libraries
		for (pl_enum::iterator p = loaded_plugins_.begin(), end = loaded_plugins_.end(); p != end; ++p)
			p->second.unload();

		//clear loaded plugins dictionary
		pl_dict_.clear();
		loaded_plugins_.clear();
	}

	pd_ptr register_plugin(const plugin_descriptor& pd, const lib_descriptor& ld) {
		//enumerate plugin first
		pair< pl_enum::iterator, bool > res = loaded_plugins_.insert(pl_enum::value_type(pd, ld));
		pd_ptr ret = res.first->first;
		//register plugin_descriptor in dictionary
		if(res.second)
			pl_dict_.insert(ret);
		return ret;
	}


	bool is_inner_pd(const plugin_descriptor& pd) {
		return (pd != kernel_pd_ && pd != runtime_pd_);
	}

	bool register_type(const plugin_descriptor& pd, const type_descriptor& td, bool inner_type = false,
		fe_ptr* tp_ref = NULL)
	{
		if(td.type().is_nil()) return false;

		pair< factory::iterator, bool > res;

		//find correct plugin_descriptor
		// pointer to registered plugin_descriptor
		pd_ptr pdp(pd);
		if(!is_inner_pd(pd)) {
			// case for external plugin descriptor
			// try to register it or find a match with existing pd
			pdp = register_plugin(pd, lib_descriptor());
		}
		//register obj in factory
		res = obj_fab_.insert(fab_elem(pdp, td));
		//save registered type if asked for
		if(tp_ref) *tp_ref = *res.first;

		//register type in dictionaries
		if(res.second) {
			pair< types_dict::const_iterator, bool > res_ref = types_resolver_.insert(*res.first);
			if(!res_ref.second) {
				//probably duplicating type name found
				obj_fab_.erase(res.first);
				if(tp_ref) {
					if(res_ref.first != types_resolver_.end())
						*tp_ref = *res_ref.first;
					else
						//some unknown bad error happened
						*tp_ref = fe_ptr::nil_el;
				}
				return false;
			}

			plugin_types_.insert(*res.first);
			return true;
		}
		else if(pd != *res.first->pd_ && is_inner_pd(*res.first->pd_)) {
			// current type was previously registered as inner
			// replace with correct plugin d-tor now
			// remove first from types_resolver_
			types_resolver_.erase(*res.first);
			// remove inner-type association
			plugin_types_.erase(*res.first);
			// now delete factory entry
			obj_fab_.erase(res.first);

			// register type with passed plugin_descriptor
			res = obj_fab_.insert(fab_elem(pd, td));
			types_resolver_.insert(*res.first);
			plugin_types_.insert(*res.first);
		}

		return false;
	}

	const fab_elem& demand_type(const fab_elem& obj_t) const {
		fe_ptr tt_ref(obj_t);
		if(obj_t.td_.is_nil()) {
			//if type is nil try to find it by name
			types_dict::const_iterator tt = types_resolver_.find(tt_ref);
			if(tt != types_resolver_.end())
				tt_ref = *tt;
		}
		else {
			//otherwise try to find requested type using fast search in factory
			factory::const_iterator tt = obj_fab_.find(*tt_ref);
			if(tt != obj_fab_.end())
				tt_ref = *tt;
			else
				tt_ref = fe_ptr::nil_el;
		}
		if(tt_ref->td_.is_nil()) {
			//type wasn't found - try to register it first
			if(obj_t.pd_.is_nil())
				lock()->register_rt_type(obj_t.td_, &tt_ref);
			else
				lock()->register_type(*obj_t.pd_, obj_t.td_, false, &tt_ref);
			//still nil td means that serious error happened - type cannot be registered
			if(tt_ref.is_nil())
        {
#ifdef BS_EXCEPTION_USE_BOOST_FORMAT
          throw bs_kernel_exception ("BlueSkt kernel", no_type, boost::format ("Unknown error. Type (%s) cannot be registered.") % obj_t.td_.name ());
#else
          throw bs_kernel_exception ("BlueSkt kernel", no_type, "Unknown error. Type " + obj_t.td_.name () + " cannot be registered.");
#endif
        }
		}
		return *tt_ref;
	}

	sp_obj create_object(const fab_elem& obj_t, bool unmanaged, bs_type_ctor_param param) {
		BS_TYPE_CREATION_FUN crfn = *demand_type(obj_t).td_.creation_fun_;
		if(!crfn) return NULL;
		// invoke creation function and make a smrt_ptr reference to new object
		sp_obj res((*crfn)(param), bs_dynamic_cast());
		if(res) {
			// if we successfully created an objbase instance
			// decrement reference counter
			res->del_ref();
			// register instance if needed
			if(!unmanaged) register_instance(res);
		}
		return res;
	}

	sp_obj create_object_copy(const sp_obj& src, bool unmanaged) {
		if(!src) {
			throw bs_kernel_exception ("BlueSky kernel", no_error, "Source object for copying is not defined");
		}

		BS_TYPE_COPY_FUN cpyfn = *demand_type(src->bs_resolve_type()).td_.copy_fun_;
		BS_ERROR (cpyfn, "kernel::create_object_copy: copy_fun is null");

		// invoke copy creation function and make a smrt_ptr reference to new object
		sp_obj res((*cpyfn)(src), bs_dynamic_cast());
		if(res) {
			// if we successfully created an objbase instance
			// decrement reference counter
			res->del_ref();
			// register instance if needed
			if(!unmanaged) register_instance(res);
		}
		return res;
	}

	bool register_kernel_type(const type_descriptor& td, fe_ptr* tp_ref = NULL) {
		return register_type(kernel_pd_, td, true, tp_ref);
	}

	bool register_rt_type(const type_descriptor& td, fe_ptr* tp_ref = NULL) {
		return register_type(runtime_pd_, td, true, tp_ref);
	}

	plugins_enum loaded_plugins() const {
		plugins_enum res;
		for(plt_enum::const_iterator p = plugin_types_.begin(); p != plugin_types_.end();
			p = plugin_types_.upper_bound(*p))
		{
			res.push_back(*p->elem().pd_);
		}
		return res;
	}

	types_enum plugin_types(const plugin_descriptor& pd) const {
		types_enum res;
		pair< plt_enum::const_iterator, plt_enum::const_iterator > plt =
			plugin_types_.equal_range(fab_elem(pd));
		for(plt_enum::const_iterator pos = plt.first; pos != plt.second; ++pos)
			res.push_back(pos->elem().td_);

		return res;
	}

	fab_elem find_type(const std::string& type_str) const {
		return *set_at(types_resolver_, fab_elem(type_descriptor(type_str.c_str())));
	}

	str_dt_ptr pert_str_dt(const type_descriptor& obj_t) {
		//ensure that given type is registered
		fe_ptr tp;
		register_rt_type(obj_t, &tp);
		if(tp.is_nil()) 
      {
        throw bs_kernel_exception ("BlueSky Kernel", blue_sky::no_type, "Cannot create str_data_table for unknown type: " + obj_t.name ());
      }

		//create or return data_table for it
		smart_ptr< str_data_table >& p_tbl = pert_str_tbl_[tp];
		if(!p_tbl) p_tbl = create_object(str_data_table::bs_type(), false, NULL);
		return str_dt_ptr(p_tbl, p_tbl->mutex());
	}

	idx_dt_ptr pert_idx_dt(const type_descriptor& obj_t) {
		//ensure that given type is registered
		fe_ptr tp;
		register_rt_type(obj_t, &tp);
		if(tp.is_nil()) 
      {
        throw bs_kernel_exception ("BlueSky Kernel", blue_sky::no_type, "Cannot create str_data_table for unknown type: " + obj_t.name ());
      }

		//create or return data_table for it
		smart_ptr< idx_data_table >& p_tbl = pert_idx_tbl_[tp];
		if(!p_tbl) p_tbl = create_object(idx_data_table::bs_type(), false, NULL);
		return idx_dt_ptr(p_tbl, p_tbl->mutex());
	}

	 //! Error processor (For load plugins in kernel::LoadPlugins() )
	void error_processor(const blue_sky::error_code&, const char* lib_dir = NULL) const;

	template< class cont_t >
	static bool erase_elem(cont_t& m, const typename cont_t::key_type& id) {
		return m.erase(id) != 0;
	}


/*-----------------------------------------------------------------------------
 *  templated functions in order to simplify access to kernel arrays
 *-----------------------------------------------------------------------------*/

	template< class cont_t, class elem_t = typename cont_t::key_type >
	struct keys_getter {
		std::vector< elem_t > operator()(const cont_t& m) {
			std::vector< elem_t > ids;
			for(typename cont_t::const_iterator it = m.begin(); it != m.end(); ++it) {
				ids.push_back(*it);
			}
			return ids;
		}
	};

	template< class cont_t >
	static const typename cont_t::mapped_type& map_at(const cont_t& m, const typename cont_t::key_type& id,
		const char* err_msg = "Type not found")
	{
		typename cont_t::const_iterator i = m.find(id);
		if (i != m.end())
			return (m->second);
		throw bs_exception("BlueSky kernel", blue_sky::no_type, err_msg, false);
	}

	template< class cont_t >
	static const typename cont_t::key_type& set_at(const cont_t& m, const typename cont_t::key_type& id,
		const char* err_msg = "Type not found")
	{
		typename cont_t::const_iterator i = m.find(id);
		if (i != m.end())
			return (*i);
		throw bs_kernel_exception ("BlueSky kernel", blue_sky::no_type, err_msg);
	}

/*-----------------------------------------------------------------------------
 *  signals register and creation function
 *-----------------------------------------------------------------------------*/

	pair< sp_signal, bool > reg_signal(const BS_TYPE_INFO& obj_t, int signal_code) {
//		factory::const_iterator fe = obj_fab_.find(obj_t);
//		if(fe == obj_fab_.end()) {
//			// Type wasn't registered eaarlier - throw exception
//			throw bs_exception("bs_kernel::reg_signal", no_type,
//				(string("Type ") + obj_t.stype_ + string(" wasn't registered")).c_str());
//		}
//
		// find|create corresponding signals map
		signals_map& sm = sig_storage_[obj_t];

		// check if signal with given code already exists
		pair< sp_signal, bool > res;
		signals_map::iterator p_sig = sm.find(signal_code);
		res.second = (p_sig == sm.end());
		if(!res.second)
			res.first = p_sig->second;
		else {
			// create signal
			res.first = new bs_signal(signal_code);
			sm[signal_code] = res.first;
		}

		return res;
	}

	bool rem_signal(const BS_TYPE_INFO& obj_t, int signal_code) {
		// find corresponding signals map
		if(sig_storage_.find(obj_t) == sig_storage_.end()) return false;
		signals_map& sm = sig_storage_[obj_t];

		// check if signal with given code exists
		signals_map::iterator p_sig = sm.find(signal_code);
		if(p_sig == sm.end()) return false;

		//delete signal
		sm.erase(signal_code);
		return true;
	}

};	//end of kernel_impl declaration

namespace {
	//tags for runtime types
	class _bs_runtime_types_tag_ {};
	//class _bs_kernel_types_tag_ {};
}

//kernel_impl guard
bs_mutex kernel::kernel_impl::guard_;
//descriptors for kernel & runtime types
plugin_descriptor& kernel::kernel_impl::kernel_pd_(plugin_info);
//plugin_descriptor kernel::kernel_impl::kernel_pd_(BS_GET_TI(bs_private::_bs_kernel_types_tag_), "BlueSky kernel",
//												  "0.1", "Plugin tag for BlueSky kernel");
plugin_descriptor kernel::kernel_impl::runtime_pd_(BS_GET_TI(_bs_runtime_types_tag_), "Runtime types",
												   "0.1", "Plugin tag for runtime types");

void kernel::kernel_impl::error_processor(const blue_sky::error_code& e, const char* lib_dir) const
{
	if(e != blue_sky::no_error)
	{
		if (e == blue_sky::wrong_path)
			//log::Instance()["main_log"] << "Wrong path: " << lib_dir << endl << "No plugins loaded." << endl;
			BSERROR << "Wrong path: " << lib_dir << " : No plugins loaded." << bs_end;
		else if (e == blue_sky::no_plugins)
			BSERROR << "No plugins found in " << lib_dir << bs_end;
	}
}

namespace {
//some global functions in hidden namespace
using namespace boost::python;
using namespace boost::python::detail;

//dumb struct to create new Python scope
struct py_scope_plug {};

////the following code is originaly taken from boost::python::detail::init_module
////and slightly modified to allow scope changing BEFORE plugin's python subsystem is initialized
PyMethodDef initial_methods[] = { { 0, 0, 0, 0 } };

void bspy_init_plugin(const string& nested_scope, void(*init_function)())
{
    static PyObject* m
        = Py_InitModule(const_cast< char* >(plugin_info.py_namespace_.c_str()), initial_methods);
	// Create the current module scope
	static scope current_module(object(((borrowed_reference_t*)m)));
	static object exp_scope_plug = class_< py_scope_plug >("bs_scope");

    if (m != 0)
    {
        // Create the current module scope
//        object m_obj(((borrowed_reference_t*)m));
//        scope current_module(m_obj);

        //make nested scope
		current_module.attr(nested_scope.c_str()) = exp_scope_plug();
		boost::python::scope outer = //exp_scope_plug();
			object(current_module.attr(nested_scope.c_str()));
			//object(current_module.attr(nested_scope.c_str()));
			//boost::python::object(nested_scope);
			//boost::python::class_< py_scope_plug >(nested_scope.c_str());



        handle_exception(init_function);
    }
}

string extract_root_name(const string& full_name) {
	//extract user-friendly lib name from full name
	string pyl_name = full_name;
	//cut path
	string::size_type pos = pyl_name.rfind('/');
	if(pos != string::npos) pyl_name = pyl_name.substr(pos + 1, string::npos);
	pos = pyl_name.rfind('\\');
	if(pos != string::npos) pyl_name = pyl_name.substr(pos + 1, string::npos);
	//cut extension
	pyl_name = pyl_name.substr(0, pyl_name.find('.'));
	//cut "lib" prefix if exists
	if(pyl_name.compare(0, 3, string("lib")) == 0)
		pyl_name = pyl_name.substr(3, string::npos);

	return pyl_name;
}

}	//end of hidden namespace

error_code kernel::kernel_impl::load_plugin(const string& fname, const string& version, bool init_py_subsyst)
{

	lib_descriptor lib; //temp DLL-pointer
	BS_REGISTER_PLUGIN bs_register_plugin; //pointer to fun_register (from temp opened DLL)
	BS_GET_PLUGIN_DESCRIPTOR bs_plugin_descriptor;
	bs_init_py_fn init_py_fn;

	//message formatter
	string msg;
	msg.clear();
	//msg.
	//log stream
	//	bs_log::stream_wrapper& xpn_log = XPN_LOG_INST

	//plugin initializer
	plugin_initializer plugin_init;
	//pointer to returned plugin descriptor
	plugin_descriptor* p_descr = NULL;
	//fully qualified python namespace
	string py_scope = "";

	try {
		//load library
		lib.load(fname.c_str());

		//check for plugin descriptor presence
		lib.load_sym("bs_get_plugin_descriptor", bs_plugin_descriptor);
		if(!bs_plugin_descriptor) {
			throw bs_exception ("LoadPlugins", lib.fname_ + " is not a BlueSky plugin (bs_get_plugin_descriptor wasn't found)");
		}
		//retrieve descriptor from plugin
		if(!(p_descr = dynamic_cast< plugin_descriptor* >(bs_plugin_descriptor()))) {
			throw bs_exception ("LoadPlugins", "No plugin descriptor found in module " + lib.fname_);
		}
		//check if loaded lib is really a blue-sky kernel
		if(*p_descr == kernel_pd_)
			return blue_sky::no_library;

		//enumerate plugin
		register_plugin(*p_descr, lib);

		//pass plugin descriptor to registering function
		plugin_init.pd_ = p_descr;

		//check version
		//int plugin_ver = version_comparator(p_descr->version_.c_str(), version.c_str());
		if(version.size() && version_comparator(p_descr->version_.c_str(), version.c_str()) < 0) { // !plugin_ver || plugin_ver == 1) {
			msg = "BlueSky plugin " + lib.fname_ + " has wrong version";
			throw bs_exception("LoadPlugins", msg.c_str());
		}

		//check if bs_register_plugin function present in library
		lib.load_sym("bs_register_plugin", bs_register_plugin);
		if(!bs_register_plugin) {
			msg = lib.fname_ + " is not a BlueSky plugin (bs_register_plugin wasn't found)";
			throw bs_exception("LoadPlugins", msg.c_str());
		}

		//invoke bs_register_plugin
		if(!bs_register_plugin(plugin_init)) {
			msg = "Plugin " + lib.fname_ + " was unable to register itself and will be unloaded";
			throw bs_exception("LoadPlugins", msg.c_str());
		}

		//init Python subsystem if asked for
		if(init_py_subsyst) {
			lib.load_sym("bs_init_py_subsystem", init_py_fn);
			if(!init_py_fn)
				BSERROR << "LoadPlugins: Python subsystem wasn't found in plugin " << lib.fname_ << bs_end;
			else {
				//DEBUG
				//cout << "Python subsystem of plugin " << lib.fname_ << " is to be initiaized" << endl;
				if(p_descr->py_namespace_ == "") {
					p_descr->py_namespace_ = extract_root_name(lib.fname_);
					//update reference information
					loaded_plugins_.erase(*p_descr);
					register_plugin(*p_descr, lib);
				}

				//init python subsystem
				bspy_init_plugin(p_descr->py_namespace_, init_py_fn);
				py_scope = plugin_info.py_namespace_ + "." + p_descr->py_namespace_;
				//boost::python::detail::init_module(py_scope.c_str(), init_py_fn);
				//BSERROR << "LoadPlugins: Error during initialization of Python sybsystem in plugin " << lib.fname_ << bs_end;
			}
		}

		//finally everything ok now
		BSOUT << "BlueSky plugin " << lib.fname_.c_str() << " loaded";
		if(py_scope.size())
			BSOUT << ", Python subsystem initialized (namespace " << py_scope << ")";
		BSOUT << bs_end;
	}
	catch(const bs_exception& /*ex*/) {
		//unload library
		if(p_descr)
			unload_plugin(*p_descr);
		else
			lib.unload();
		return blue_sky::no_library;
	}
	catch(const std::exception& ex) {
		BSERROR << "LoadPlugins: " << ex.what();
		return blue_sky::system_error;
	}
	catch(...) {
		//something really serious happened
		BSERROR << "Unknown error happened during plugins loading. Terminating.";
		terminate();
	}

	return blue_sky::no_error;
}

/*!
Loading blue-sky plugins method and register them in the blue-sky kernel.
\return blue_sky::system_error if system error rised,
or blue_sky::no_plugins if not plugins in folder
or blue_sky::no_error if all is ok.
*/
blue_sky::error_code kernel::kernel_impl::load_plugins(bool init_py_subsyst) {
	//unload all plugins
	unload_plugins();
	//get order of plugins loading
	get_lib_list(cft_);

	//PyObject* bspy_module = NULL;
	if(init_py_subsyst) {
		//initialize Python module for kernel (from boost::python::detail::init_module
		//bspy_module	= Py_InitModule(const_cast<char*>(plugin_info.py_namespace_.c_str()), initial_methods);
		//find kernel's Python initialization function
		bs_init_py_fn init_py;
		lib_descriptor::load_sym_glob("bs_init_py_subsystem", init_py);
		//init kernel's py subsyst
		if(init_py) {
			boost::python::detail::init_module(plugin_info.py_namespace_.c_str(), init_py);
			//create bs_scope exporting
			BSOUT << "BlueSky kernel Python subsystem initialized successfully under namespace ";
			BSOUT << plugin_info.py_namespace_ << bs_end;
		}
		else {
			BSERROR << "Python subsystem wasn't found in BlueSky kernel" << bs_end;
			init_py_subsyst = false;
		}
	}

	ulong lib_cnt = 0;
	for(std::list< lload >::const_iterator i = cft_.begin(); i != cft_.end(); ++i) {
		if(load_plugin(lload(*i).first, lload(*i).second, init_py_subsyst) == blue_sky::no_error)
			++lib_cnt;
	} //main loading cycle

	if (lib_cnt == 0) {
		BSERROR << "BlueSky: no plugins were loaded" << bs_end;
		return blue_sky::no_plugins;
	}
	return blue_sky::no_error;
}

//===================================== kernel implementation ==========================================================
void kernel::test() const
{
  //log::Instance()["cout1"].echo("this message for exception",-1);

	cout << "kernel.test entered" << std::endl;
  try
  {
	  path("~");
  }
  catch(const boost::filesystem::filesystem_error& e)
  {
    throw bs_kernel_exception ("path", blue_sky::boost_error, e.what());
  }
}

kernel::kernel()
	: pimpl_(new kernel_impl, kernel_impl::guard_)
{
	//initialize logs
	//log::Instance().init_logs();
}

kernel::~kernel()
{
	UnloadPlugins();

	// WTF?? 
	if(pimpl_.get()) delete pimpl_.get();
}

// kernel::str_dt_ptr kernel::global_dt() const {
// 	return str_dt_ptr(&pimpl_->data_tbl_, pimpl_->data_tbl_.mutex());
// }

void kernel::init() 
{
  //detail::bs_log_holder::Instance ().register_signals ();
  //detail::thread_log_holder::Instance ().register_signals ();

	pimpl_.lock()->init();
}

int kernel::free_instance(const sp_obj& p_obj) const
{
	return pimpl_.lock()->free_instance(p_obj);
}

int kernel::register_instance(const sp_obj& p_obj) const
{
	return pimpl_.lock()->register_instance(p_obj);
}

//access to object's instances
bs_objinst_holder::const_iterator kernel::objinst_begin(const type_descriptor& td) const {
	return pimpl_.lock()->objinst_begin(td.type());
}

bs_objinst_holder::const_iterator kernel::objinst_end(const type_descriptor& td) const {
	return pimpl_.lock()->objinst_end(td.type());
}

ulong kernel::objinst_cnt(const type_descriptor& td) const {
	return pimpl_->objinst_cnt(td.type());
}

//per-storage data tables access
kernel::str_dt_ptr kernel::pert_str_dt(const type_descriptor& obj_t) const {
	return pimpl_.lock()->pert_str_dt(obj_t);
}

kernel::idx_dt_ptr kernel::pert_idx_dt(const type_descriptor& obj_t) const {
	return pimpl_.lock()->pert_idx_dt(obj_t);
}

//loaded plugins enumeration
kernel::plugins_enum kernel::loaded_plugins() const {
	return pimpl_->loaded_plugins();
}

//types came from particular plugin enumeration
kernel::types_enum kernel::plugin_types(const plugin_descriptor& pd) const {
	return pimpl_->plugin_types(pd);
}

kernel::types_enum kernel::plugin_types(const std::string& plugin_name) const
{
	kernel_impl::pl_dict::const_iterator p = pimpl_->pl_dict_.find(plugin_descriptor(plugin_name.c_str()));
	if(p != pimpl_->pl_dict_.end())
		return pimpl_->plugin_types(**p);
	else
		return types_enum();
}

type_tuple kernel::find_type(const std::string& type_str) const {
	return pimpl_->find_type(type_str);
}

bool kernel::register_type(const plugin_descriptor& pd, const type_descriptor& td) const
{
	return pimpl_.lock()->register_type(pd, td);
}

sp_obj kernel::create_object(const type_descriptor& obj_t, bool unmanaged, bs_type_ctor_param param) const
{
	return pimpl_.lock()->create_object(obj_t, unmanaged, param);
}

//sp_obj kernel::create_object(const type_descriptor& td, const plugin_descriptor& pd,
//							 bool unmanaged, bs_type_ctor_param param) const
//{
//	return pimpl_->create_object(type_tuple(pd, td), unmanaged, param);
//}

sp_obj kernel::create_object(const std::string& obj_t, bool unmanaged, bs_type_ctor_param param) const
{
	return pimpl_.lock()->create_object(type_descriptor(obj_t.c_str()), unmanaged, param);
}

std::vector< type_tuple > kernel::registered_types() const
{
	return kernel_impl::keys_getter< kernel_impl::factory, type_tuple >()(pimpl_->obj_fab_);
}

blue_sky::sp_obj kernel::create_object_copy( const sp_obj& src, bool unmanaged /*= false*/ ) const
{
	return pimpl_.lock()->create_object_copy(src, unmanaged);
}

sp_storage kernel::create_storage(const std::string &filename, const std::string &format, int flags) const
{
	sp_storage new_storage;
	try {
		new_storage = create_object(find_type(format).td_);
		new_storage.lock()->open(filename, flags);
	}
	catch(const bs_exception& /*ex*/) {
		// if no proper storage found
		new_storage = create_object(empty_storage::bs_type());
	}
	return new_storage;
}

void kernel::close_storage(const sp_storage& storage) const {
  //storages.pop_back();
  free_instance( storage );
}

blue_sky::error_code kernel::LoadPlugins(bool init_py_subsyst) const {
	return pimpl_.lock()->load_plugins(init_py_subsyst);
}

void kernel::UnloadPlugins() const {
	pimpl_.lock()->unload_plugins();
}

error_code kernel::load_plugin(const std::string& fname, const std::string version, bool init_py_subsyst) {
	return pimpl_.lock()->load_plugin(fname, version, init_py_subsyst);
}

void kernel::unload_plugin(const plugin_descriptor& pd) {
	pimpl_.lock()->unload_plugin(pd);
}

std::string kernel::get_last_error() const {
	return pimpl_->last_err_;
}

void kernel::add_task(const blue_sky::sp_com& task)
{
	pimpl_.lock()->wtp_.add_command(task);
}

bool kernel::is_tq_empty() const {
	return pimpl_->wtp_.is_queue_empty();
}

void kernel::wait_tq_empty() const {
	return pimpl_->wtp_.wait_queue_empty();
}

sp_link kernel::bs_root() const {
	return pimpl_->root_;
}

ulong kernel::tree_gc() const {
	return pimpl_.lock()->tree_gc();
}

std::pair< sp_signal, bool > kernel::reg_signal(const BS_TYPE_INFO& obj_t, int signal_code) const {
	return pimpl_.lock()->reg_signal(obj_t, signal_code);
}

bool kernel::rem_signal(const BS_TYPE_INFO& obj_t, int signal_code) const {
	return pimpl_.lock()->rem_signal(obj_t, signal_code);
}

}	//end of blue_sky namespace
