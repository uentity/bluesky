/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Plugins subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifdef BSPY_EXPORTING
#include <pybind11/pybind11.h>
#endif

#include "kernel_plugins_subsyst.h"
#include <bs/exception.h>
#include <bs/log.h>

#define KERNEL_VERSION "0.1" //!< version of blue-sky kernel

/*-----------------------------------------------------------------------------
 *  BS kernel plugin descriptor
 *-----------------------------------------------------------------------------*/
BLUE_SKY_PLUGIN_DESCRIPTOR_EXT("BlueSky kernel", KERNEL_VERSION, "BlueSky kernel types tag", "", "bs");

// hide implementation
namespace blue_sky { namespace {

// tags for kernel & runtime types plugin_descriptor
struct __kernel_types_pd_tag__ {};
struct __runtime_types_pd_tag__ {};

std::string extract_root_name(const std::string& full_name) {
	using namespace std;

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

#ifdef BSPY_EXPORTING
struct bspy_module {
	typedef void (*init_function_t)();

	bspy_module() : root_(NULL) {}

	bool init_kernel_subsyst(const std::string& root_ns) {
		// find kernel's Python initialization function
		init_function_t init_py;
		lib_descriptor::load_sym_glob("bs_init_py_subsystem", init_py);
		if(init_py) {
			root_ = boost::python::detail::init_module(root_ns.c_str(), init_py);
			//create bs_scope exporting
			BSOUT << "BlueSky kernel Python subsystem initialized successfully under namespace "
				<< root_ns << bs_end;
			// save name
			root_ns_ = root_ns;
			return true;
		}
		else {
			BSERROR << "Python subsystem wasn't found in BlueSky kernel" << bs_end;
		   return false;
		}
	}

	scope root_scope() {
		if(!root_) {
			bs_throw_exception (boost::format ("load_plugin: No BS kernel module in namespace %s")
					% root_ns_);
		}
		return scope(object(((borrowed_reference_t*)root_)));
	}

	string init_plugin_subsyst(const string& nested_scope, init_function_t f) {
		//the following code is originaly taken from boost::python::detail::init_module
		//and slightly modified to allow scope changing BEFORE plugin's python subsystem is initialized
		scope bs_root = root_scope();

		// create plugin's module & scope
		string nested_namespace = root_ns_ + "." + nested_scope;
		PyObject *nested_module = Py_InitModule (nested_namespace.c_str(), initial_methods());
		if (!nested_module) {
			bs_throw_exception (boost::format ("bspy_init_plugin: Can't create plugin module in namespace %s") % nested_namespace);
		}
		scope nested(object (((borrowed_reference_t *)nested_module)));
		bs_root.attr(nested_scope.c_str ()) = nested;

		handle_exception(f);
		return nested_namespace;
	}

	static PyMethodDef* initial_methods() {
		static PyMethodDef m[] = { { 0, 0, 0, 0 } };
		return m;
	}

	PyObject* root_;
	string root_ns_;
};
#endif

} // eof hidden namespace

namespace detail {

kernel_plugins_subsyst::kernel_plugins_subsyst()
	: kernel_pd_(
		BS_GET_TI(__kernel_types_pd_tag__), "Kernel types", KERNEL_VERSION,
		"BlueSky kernel types tag", "", "bs")
	, runtime_pd_(
		BS_GET_TI(__runtime_types_pd_tag__), "Runtime types", KERNEL_VERSION,
		"BlueSky runtime types tag", "", "bs")
{}

std::pair< pd_ptr, bool >
kernel_plugins_subsyst::register_plugin(const plugin_descriptor& pd, const lib_descriptor& ld) {
	// enumerate plugin first
	auto res = loaded_plugins_.insert(std::make_pair(pd, ld));
	pd_ptr ret = res.first->first;
	// register plugin_descriptor in dictionary
	if(res.second)
		plugins_dict_.insert(ret);
	return std::make_pair(ret, res.second);
}

void kernel_plugins_subsyst::clean_plugin_types(const plugin_descriptor& pd) {
	// we cannot clear kernel internal types
	if(pd == kernel_pd_) return;

	auto plug_types = plugin_types_.equal_range(fab_elem(pd));
	for(auto ptype = plug_types.first; ptype != plug_types.second; ++ptype) {
		types_resolver_.erase(ptype);
		obj_fab_.erase(**ptype);
	}
	plugin_types_.erase(plug_types.first, plug_types.second);
}

// unloas given plugin
void kernel_plugins_subsyst::unload_plugin(const plugin_descriptor& pd) {
	// check if given plugin was registered
	auto plug = loaded_plugins_.find(pd);
	if(plug == loaded_plugins_.end()) return;

	clean_plugin_types(pd);
	plugins_dict_.erase(pd);
	loaded_plugins_[pd].unload();
	loaded_plugins_.erase(pd);
}

// unloads all plugins
void kernel_plugins_subsyst::unload_plugins() {
	for(auto p : loaded_plugins_) {
		unload_plugin(p.first);
	}
}

int kernel_plugins_subsyst::load_plugin(
	const std::string& fname, bool init_py_subsyst
) {
	using namespace std;

	lib_descriptor lib; // temp DLL-pointer
	// RAII object for plugin lib unload
	auto LU = [](lib_descriptor* plib) { plib->unload(); };
	using delay_unload = std::unique_ptr< lib_descriptor, decltype(LU) >;
	// RAII for plugin cleanup (including registered types, etc..)
	auto PU = [=](plugin_descriptor* pd) { this->unload_plugin(*pd); };
	using delay_unplug = std::unique_ptr< plugin_descriptor, decltype(PU) >;

	BS_REGISTER_PLUGIN bs_register_plugin; // pointer to fun_register (from temp DLL)
	BS_GET_PLUGIN_DESCRIPTOR bs_plugin_descriptor;

	// error message formatter
	static const char* who = "load_plugins";
	string msg;

	// plugin initializer
	plugin_initializer plugin_init;
	// pointer to returned plugin descriptor
	plugin_descriptor* p_descr = nullptr;
	// fully qualified python namespace
	string py_scope = "";
	int retval = -1;

	try {
		// load library
		lib.load(fname.c_str());

		// check for plugin descriptor presence
		lib.load_sym("bs_get_plugin_descriptor", bs_plugin_descriptor);
		if(!bs_plugin_descriptor) {
			delay_unload killer(&lib, LU);
			bsout() << "{}: {} is not a BlueSky plugin (bs_get_plugin_descriptor wasn't found)"
				<< who << lib.fname_ << log::end;
			return retval;
		}
		// retrieve descriptor from plugin
		if(!(p_descr = dynamic_cast< plugin_descriptor* >(bs_plugin_descriptor()))) {
			delay_unload killer(&lib, LU);
			bsout() << log::W("{}: No plugin descriptor found in module {}") << who << lib.fname_ << log::end;
			return retval;
		}
		// check if loaded lib is really a blue-sky kernel
		if(*p_descr == kernel_pd_)
			return retval;
			// TODO: do something with err code
			//return blue_sky::no_library;

		// check if bs_register_plugin function present in library
		lib.load_sym("bs_register_plugin", bs_register_plugin);
		if(!bs_register_plugin) {
			delay_unload killer(&lib, LU);
			bsout() << "{}: {} is not a BlueSky plugin (bs_register_plugin wasn't found)"
				<< who << lib.fname_ << log::end;
			return retval;
		}

		// enumerate plugin
		if(!register_plugin(*p_descr, lib).second) {
			// plugin was already registered earlier
			delay_unload killer(&lib, LU);
			bsout() << log::W("{}: {} plugin is already registred, skipping...")
				<< who << lib.fname_ << log::end;
			// TODO: do something with err code
			return retval;
		}

		// pass plugin descriptor to registering function
		plugin_init.pd = p_descr;

		// TODO: enable versions
		// check version
		//if(version.size() && version_comparator(p_descr->version.c_str(), version.c_str()) < 0) {
		//	delay_unload killer(&lib, LU);
		//	bsout() << log::E("{}: BlueSky plugin {} has wrong version") << who << lib.fname_ << log::end;
		//	return retval;
		//}

		//invoke bs_register_plugin
		if(!bs_register_plugin(plugin_init)) {
			delay_unplug killer(p_descr, PU);
			bsout() << log::E("{}: {} plugin was unable to register itself and will be unloaded")
				<< who << lib.fname_ << log::end;
			return retval;
		}

#ifdef BSPY_EXPORTING
		//init Python subsystem if asked for
		if(init_py_subsyst) {
			bs_init_py_fn init_py_fn;
			lib.load_sym("bs_init_py_subsystem", init_py_fn);
			if(!init_py_fn)
				// TODO: enable logging
				bserr() << E("{}: Python subsystem wasn't found in plugin {}"
					<< who << lib.fname_ << log::end;
			else {
				//DEBUG
				//cout << "Python subsystem of plugin " << lib.fname_ << " is to be initiaized" << endl;
				if(p_descr->py_namespace_ == "") {
					p_descr->py_namespace_ = extract_root_name(lib.fname_);
					// update reference information
					loaded_plugins_.erase(*p_descr);
					register_plugin(*p_descr, lib);
				}

				// init python subsystem
				py_scope = pymod_.init_plugin_subsyst(p_descr->py_namespace_, init_py_fn);
			}
		}
#else
		// supress warning
		(void)init_py_subsyst;
#endif

		// finally everything is OK now
		msg = "BlueSky plugin {} loaded";
		if(py_scope.size())
			msg += ", Python subsystem initialized (namespace {})";
		auto log_msg = bsout() << log::I(msg.c_str()) << lib.fname_;
		if(py_scope.size())
			log_msg << py_scope << log::end;
		else
			log_msg << log::end;
	}
	catch(const bs_exception& ex) {
		// print error information
		ex.print();
		retval = ex.err_code();
	}
	catch(const std::exception& ex) {
		BSERROR << "[Std Exception] {}: {}" << who << ex.what() << bs_end;
	}
	catch(...) {
		//something really serious happened
		BSERROR << "[Unknown Exception] {}: Unknown error happened during plugins loading"
			<< who << bs_end;
		throw;
	}

	return retval;
}

}} // eof blue_sky::detail namespace

