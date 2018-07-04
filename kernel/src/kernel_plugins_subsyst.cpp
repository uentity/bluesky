/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Plugins subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifdef BSPY_EXPORTING
#include <bs/python/common.h>
#endif

#include "kernel_plugins_subsyst.h"
#include "kernel_plugins_discover.h"
#include <bs/error.h>
#include <bs/log.h>

#define KERNEL_VERSION "0.1" //!< version of blue-sky kernel

/*-----------------------------------------------------------------------------
 *  BS kernel plugin descriptor
 *-----------------------------------------------------------------------------*/
BS_C_API const blue_sky::plugin_descriptor* bs_get_plugin_descriptor() {
	return &blue_sky::detail::kernel_plugins_subsyst::kernel_pd();
}

NAMESPACE_BEGIN(blue_sky)
// hide implementation
namespace {

// tags for kernel & runtime types plugin_descriptors
struct BS_HIDDEN_API __kernel_types_pd_tag__ {};
struct BS_HIDDEN_API __runtime_types_pd_tag__ {};

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

} // eof hidden namespace

NAMESPACE_BEGIN(detail)

// init kernel plugin descriptors
const plugin_descriptor& kernel_plugins_subsyst::kernel_pd() {
	static const plugin_descriptor kernel_pd(
		BS_GET_TI(__kernel_types_pd_tag__), "kernel", KERNEL_VERSION,
		"BlueSky virtual kernel plugin", "bs",
		(void*)&cereal::detail::StaticObject<cereal::detail::InputBindingMap>::getInstance(),
		(void*)&cereal::detail::StaticObject<cereal::detail::OutputBindingMap>::getInstance()
	);
	return kernel_pd;
}

const plugin_descriptor& kernel_plugins_subsyst::runtime_pd() {
	static const plugin_descriptor runtime_pd(
		BS_GET_TI(__runtime_types_pd_tag__), "runtime", KERNEL_VERSION,
		"BlueSky virtual plugin for runtime types", "bs"
	);
	return runtime_pd;
}

kernel_plugins_subsyst::kernel_plugins_subsyst() {
	// register kernel virtual plugins
	register_plugin(&kernel_pd(), lib_descriptor());
	register_plugin(&runtime_pd(), lib_descriptor());
}

kernel_plugins_subsyst::~kernel_plugins_subsyst() {}

#ifdef BSPY_EXPORTING
struct kernel_plugins_subsyst::bspy_module {
	bspy_module(pybind11::module& root_mod) : root_module_(root_mod) {}

	bool init_kernel_subsyst() {
		namespace py = pybind11;
		// find kernel's Python initialization function
		bs_init_py_fn init_py;
		detail::lib_descriptor::load_sym_glob("bs_init_py_subsystem", init_py);
		if(init_py) {
			init_py(&root_module_);

			BSOUT << "BlueSky kernel Python subsystem initialized successfully under namespace: {}"
				<< PyModule_GetName(root_module_.ptr()) << bs_end;
			return true;
		}
		else {
			BSERROR << "Python subsystem wasn't found in BlueSky kernel" << bs_end;
		   return false;
		}
	}

	void init_plugin_subsyst(const char* plugin_namespace, const char* doc, bs_init_py_fn f) {
		// create submodule
		auto plugin_mod = root_module_.def_submodule(plugin_namespace, doc);
		// invoke init function
		f(&plugin_mod);
	}

	pybind11::module& root_module_;
};
#endif

std::pair< const plugin_descriptor*, bool >
kernel_plugins_subsyst::register_plugin(const plugin_descriptor* pd, const lib_descriptor& ld) {
	// deny registering nil plugin descriptors
	if(!pd) return {&plugin_descriptor::nil(), false};

	// find or insert passed plugin_descriptor
	auto res = loaded_plugins_.insert(std::make_pair(pd, ld));
	auto pplug = res.first->first;

	// if plugin with same name was already registered
	if(!res.second) {
		if(pplug->is_nil() && !pd->is_nil()) {
			// replace temp nil plugin with same name
			loaded_plugins_.erase(res.first);
			res = loaded_plugins_.insert(std::make_pair(pd, ld));
			pplug = res.first->first;
		}
		else if(!res.first->second.handle_ && ld.handle_) {
			// update lib descriptor to valid one
			res.first->second = ld;
			return {pplug, true};
		}
	}

	// for newly inserted valid (non-nill) plugin descriptor
	// update types that previousely referenced it by name
	if(res.second && !pplug->is_nil()) {
		auto& plug_name_view = types_.get< plug_name_key >();
		auto ptypes = plug_name_view.equal_range(pplug->name);
		for(auto ptype = ptypes.first; ptype != ptypes.second; ++ptype) {
			plug_name_view.replace(ptype, type_tuple{*pplug, ptype->td()});
		}

		// remove entry from temp plugins if any
		// temp plugins are always nill
		auto tplug = temp_plugins_.find(*pplug);
		if(tplug != temp_plugins_.end())
			temp_plugins_.erase(tplug);

		// merge serialization bindings maps
		unify_serialization();
	}

	return {pplug, res.second};
}

void kernel_plugins_subsyst::clean_plugin_types(const plugin_descriptor& pd) {
	// we cannot clear kernel internal types
	if(pd == kernel_pd()) return;

	types_.get< plug_key >().erase(pd);
}

// unloas given plugin
void kernel_plugins_subsyst::unload_plugin(const plugin_descriptor& pd) {
	// check if given plugin was registered
	auto plug = loaded_plugins_.find(&pd);
	if(plug == loaded_plugins_.end()) return;

	clean_plugin_types(pd);
	// unload and erase plugin
	plug->second.unload();
	loaded_plugins_.erase(plug);
}

// unloads all plugins
void kernel_plugins_subsyst::unload_plugins() {
	for(auto p : loaded_plugins_) {
		unload_plugin(*p.first);
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

	bs_register_plugin_fn bs_register_plugin; // pointer to fun_register (from temp DLL)
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
		if(*p_descr == kernel_pd())
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
		if(!register_plugin(p_descr, lib).second) {
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
				bserr() << log::E("{}: Python subsystem wasn't found in plugin {}")
					<< who << lib.fname_ << log::end;
			else {
				//DEBUG
				//cout << "Python subsystem of plugin " << lib.fname_ << " is to be initiaized" << endl;
				if(p_descr->py_namespace == "") {
					p_descr->py_namespace = extract_root_name(lib.fname_);
					// update reference information
					loaded_plugins_.erase(p_descr);
					register_plugin(p_descr, lib);
				}

				// init python subsystem
				pymod_->init_plugin_subsyst(
					p_descr->py_namespace.c_str(), p_descr->description.c_str(), init_py_fn
				);
				py_scope = kernel_pd().py_namespace + '.' + p_descr->py_namespace;
			}
		}
#else
		// supress warning
		(void)init_py_subsyst;
#endif

		// finally everything is OK now
		msg = "BlueSky plugin {} loaded";
		if(py_scope.size())
			msg += ", Python subsystem initialized, namespace: {}";
		auto log_msg = bsout() << log::I(msg.c_str()) << lib.fname_;
		if(py_scope.size())
			log_msg << py_scope << log::end;
		else
			log_msg << log::end;
	}
	catch(const error& ex) {
		retval = ex.code.value();
	}
	catch(const std::exception& ex) {
		BSERROR << log::E("[Std Exception] {}: {}") << who << ex.what() << bs_end;
	}
	catch(...) {
		//something really serious happened
		BSERROR << log::E("[Unknown Exception] {}: Unknown error happened during plugins loading")
			<< who << bs_end;
		throw;
	}

	return retval;
}

int kernel_plugins_subsyst::load_plugins(void* py_root_module) {
	// discover plugins
	auto plugins = plugins_discover().go();
	BSOUT << "--------" << bs_end;
	if(!plugins.size())
		BSOUT << "No plugins were found!" << bs_end;
	else {
		BSOUT << "Found plugins:" << bs_end;
		for(const auto& plugin : plugins)
			BSOUT << "{}" << plugin << bs_end;
		BSOUT << "--------" << bs_end;
	}

	// init kernel Python subsystem
#ifdef BSPY_EXPORTING
	if(py_root_module) {
		pymod_.reset(new bspy_module(*static_cast< pybind11::module* >(py_root_module)));
		pymod_->init_kernel_subsyst();
	}
#endif

	std::size_t plugin_cnt = 0;
	for(const auto& plugin_fname : plugins) {
		load_plugin(plugin_fname, bool(py_root_module));
		++plugin_cnt;
	}

#ifdef BSPY_EXPORTING
	// register constructor of std::error_code from arbitrary int value
	if(py_root_module) {
		auto err_class = (pybind11::class_<std::error_code>)pymod_->root_module_.attr("error_code");
		pybind11::init([](int ec) {
			return std::error_code(static_cast<Error>(ec));
		}).execute(err_class);
	}
#endif

	return 0;
}

namespace {
// helper to unify input or output bindings
template<typename bindings_t>
auto unify_bindings(const kernel_plugins_subsyst& K, void *const plugin_descriptor::*binding_var) {
	using Serializers_map = typename bindings_t::Serializers_map;
	using Archives_map = typename bindings_t::Archives_map;
	Archives_map united;
	bindings_t* plug_bnd = nullptr;

	// lambda helper to merge bindings from source to destination
	auto merge_bindings = [](const auto& src, auto& dest) {
		// loop over archive entries
		for(const auto& ar : src) {
			auto ar_dest = dest.find(ar.first);
			// if entries for archive not found at all - insert 'em all at once
			// otherwise go deeper and loop over entries per selected archive
			if(ar_dest == dest.end())
				dest.insert(ar);
			else {
				// loop over bindings for selected archive
				// and insert bindings that are missing in destination
				auto& dest_binds = ar_dest->second;
				for(const auto& src_bind : ar.second) {
					if(dest_binds.find(src_bind.first) == dest_binds.end())
						dest_binds.insert(src_bind);
				}
			}
		}
	};

	// first pass -- collect all bindings into single map
	for(const auto& plugin : K.loaded_plugins_) {
		// extract bindings global from plugin descriptor
		if( !(plug_bnd = reinterpret_cast<bindings_t*>(plugin.first->*binding_var)) )
			continue;
		// merge plugin bindings into united map
		merge_bindings(plug_bnd->archives_map, united);
	}

	// 2nd pass - merge united into plugins' bindings
	for(const auto& plugin : K.loaded_plugins_) {
		// extract bindings global from plugin descriptor
		if( !(plug_bnd = reinterpret_cast<bindings_t*>(plugin.first->*binding_var)) )
			continue;
		// merge all entries from plugin into united
		merge_bindings(united, plug_bnd->archives_map);
	}
}

} // eof hidden namespace

void kernel_plugins_subsyst::unify_serialization() const {
	unify_bindings<cereal::detail::InputBindingMap>(
		*this, &plugin_descriptor::serial_input_bindings
	);
	unify_bindings<cereal::detail::OutputBindingMap>(
		*this, &plugin_descriptor::serial_output_bindings
	);
}

NAMESPACE_END(detail)
NAMESPACE_END(blue_sky)

