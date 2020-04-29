/// @file
/// @author uentity
/// @date 24.08.2016
/// @brief Plugins subsystem of BS kernel
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/kernel/errors.h>
#include <bs/defaults.h>
#include <bs/log.h>
#include <bs/detail/scope_guard.h>
#include <bs/detail/is_container.h>
#include <bs/detail/function_view.h>

#include "plugins_subsyst.h"
#include "python_subsyst.h"

#define PYSS singleton<python_subsyst>::Instance()

/*-----------------------------------------------------------------------------
 *  BS kernel plugin descriptor
 *-----------------------------------------------------------------------------*/
BS_C_API blue_sky::plugin_descriptor* bs_get_plugin_descriptor() {
	return &blue_sky::kernel::detail::plugins_subsyst::kernel_pd();
}

NAMESPACE_BEGIN(blue_sky::kernel::detail)
// import defaults
using defaults::kernel::version;

// hide implementation
NAMESPACE_BEGIN()

// tags for kernel & runtime types plugin_descriptors
struct __kernel_types_pd_tag__ {};
struct __runtime_types_pd_tag__ {};

NAMESPACE_END() // eof hidden namespace

// init kernel plugin descriptors
auto plugins_subsyst::kernel_pd() -> plugin_descriptor& {
	static plugin_descriptor kernel_pd(
		BS_GET_TI(__kernel_types_pd_tag__), defaults::kernel::plugin_name, defaults::kernel::version,
		"BlueSky kernel", defaults::kernel::py_namespace,
		(void*)&cereal::detail::StaticObject<cereal::detail::InputBindingMap>::getInstance(),
		(void*)&cereal::detail::StaticObject<cereal::detail::OutputBindingMap>::getInstance(),
		(void*)&cereal::detail::StaticObject<cereal::detail::PolymorphicCasters>::getInstance()
	);
	return kernel_pd;
}

auto plugins_subsyst::runtime_pd() -> plugin_descriptor& {
	static plugin_descriptor runtime_pd(
		BS_GET_TI(__runtime_types_pd_tag__), defaults::kernel::rt_plugin_name, defaults::kernel::version,
		"BlueSky runtime types", defaults::kernel::py_namespace
	);
	return runtime_pd;
}

plugins_subsyst::plugins_subsyst() {
	// register kernel virtual plugins
	register_plugin(&kernel_pd(), lib_descriptor());
	register_plugin(&runtime_pd(), lib_descriptor());
}

plugins_subsyst::~plugins_subsyst() {}

void plugins_subsyst::clean_plugin_types(const plugin_descriptor& pd) {
	// we cannot clear kernel internal types
	if(pd == kernel_pd()) return;

	types_.get< plug_name_key >().erase(pd.name);
}

// unloas given plugin
void plugins_subsyst::unload_plugin(const plugin_descriptor& pd) {
	// check if given plugin was registered
	auto plug = loaded_plugins_.find(&pd);
	if(plug == loaded_plugins_.end()) return;

	clean_plugin_types(pd);
	// unload and erase plugin
	plug->second.unload();
	loaded_plugins_.erase(plug);
}

// unloads all plugins
void plugins_subsyst::unload_plugins() {
	for(const auto& p : loaded_plugins_) {
		unload_plugin(*p.first);
	}
}

auto plugins_subsyst::load_plugin(const std::string& fname) -> error {
return error::eval_safe([&]() -> error {
	// DLL handle
	lib_descriptor lib;
	auto unload_on_error = scope_guard{ [&lib]{ lib.unload(); } };

	plugin_initializer plugin_init;
	BS_GET_PLUGIN_DESCRIPTOR bs_plugin_descriptor;
	bs_register_plugin_fn bs_register_plugin;
	plugin_descriptor* p_descr = nullptr;

	// load library
	if(!lib.load(fname.c_str()))
		return {lib.dll_error_message(), kernel::Error::CantLoadDLL};

	// check for plugin descriptor presence
	lib.load_sym("bs_get_plugin_descriptor", bs_plugin_descriptor);
	if(!bs_plugin_descriptor)
		return {lib.fname_, kernel::Error::BadBSplugin};

	// retrieve descriptor from plugin
	if(!(p_descr = dynamic_cast< plugin_descriptor* >(bs_plugin_descriptor())))
		return {lib.fname_, kernel::Error::BadPluginDescriptor};

	// check if loaded lib is really a blue-sky kernel
	if(*p_descr == kernel_pd())
		return error::quiet("load_plugin: cannot load kernel (already loaded)");

	// check if bs_register_plugin function present in library
	lib.load_sym("bs_register_plugin", bs_register_plugin);
	if(!bs_register_plugin)
		return {lib.fname_ + ": missing bs_register_plugin)", kernel::Error::BadBSplugin};

	// check if plugin was already registered earlier
	if(!register_plugin(p_descr, lib).second)
		return {lib.fname_, kernel::Error::PluginAlreadyRegistered};

	// TODO: enable versions checking
	// check version
	//if(version.size() && version_comparator(p_descr->version.c_str(), version.c_str()) < 0) {
	//	delay_unload killer(&lib, LU);
	//	bsout() << log::E("{}: BlueSky plugin {} has wrong version") << who << lib.fname_ << log::end;
	//	return retval;
	//}

	// invoke bs_register_plugin
	plugin_init.pd = p_descr;
	if(!bs_register_plugin(plugin_init)) {
		unload_plugin(*p_descr);
		return {lib.fname_, kernel::Error::PluginRegisterFail};
	}

	// init Python subsystem
	auto py_scope = PYSS.py_init_plugin(lib, *p_descr).value_or("");

	// print status
	std::string msg = "BlueSky plugin {} loaded";
	if(py_scope.size())
		msg += ", Python subsystem initialized, namespace: {}";
	auto log_msg = bsout() << log::I(msg.c_str()) << lib.fname_;
	if(py_scope.size())
		log_msg << py_scope << log::end;
	else
		log_msg << log::end;

	// don't unload sucessfully loaded plugin
	unload_on_error.disable();
	return perfect;
}); }

auto plugins_subsyst::load_plugins() -> error {
	// discover plugins
	auto plugins = discover_plugins();
	BSOUT << "--------> [load plugins]" << bs_end;

	// collect error messages
	std::string err_messages;
	for(const auto& plugin_fname : plugins)
		if(auto er = load_plugin(plugin_fname)) {
			if(!err_messages.empty()) err_messages += '\n';
			err_messages += er.what();
		}

	// register closure ctor for error from any int value
	PYSS.py_add_error_closure();

	return err_messages.empty() ? success() : error::quiet(err_messages);
}

NAMESPACE_BEGIN()

// [DEBUG] print types registered in passed bindings map
template<typename Class, typename Target>
void print_serial_map(
	void* pB, Target Class::*class_member, std::string_view domain
) {
	auto B = reinterpret_cast<Class*>(pB);
	bsout() << "----> [{} at {}]" << domain << pB << bs_end;
	if(!B)
		bsout() << "Empty" << bs_end;
	else {
		for(const auto& ar : B->*class_member) {
			bsout() << "[{}]" << ar.first.name() << bs_end;
			if constexpr(meta::is_map_v<decltype(ar.second)>) {
				for(const auto& ar_bnd : ar.second) {
					if constexpr(std::is_same_v<std::type_index, std::decay_t<decltype(ar_bnd.first)>>)
						bsout() << "  {}" << ar_bnd.first.name() << bs_end;
					else
						bsout() << "  {}" << ar_bnd.first << bs_end;
				}
			}
			else bsout() << "  {}" << ar.second.name() << bs_end;
		}
	}
	bsout() << "<----" << bs_end;
}

// helper to unify input or output bindings
template<typename Class, typename Target>
auto unify_serial_globals(
	const plugins_subsyst& K, void* const plugin_descriptor::*class_storage,
	Target Class::*class_member,
	function_view<void (Class*)> postprocess_united_fn = [](Class*){}
) {
	// lambda helper to merge bindings from source to destination
	auto merge_targets = [](const auto& src, auto& dest) {
		auto merger = [](const auto& src, auto& dest, auto self) {
			// supports map, vector, list, etc
			auto search = [](auto& where, auto& what) {
				if constexpr(meta::is_map_v<decltype(where)>)
					return where.find(what.first);
				else return std::find(where.begin(), where.end(), what);
			};
			auto append = [](auto& where, auto& what) {
				if constexpr(meta::is_map_v<decltype(where)>)
					where.insert(what);
				else where.push_back(what);
			};

			for(const auto& S : src) {
				// if entries for archive not found at all - insert 'em all at once
				// otherwise go deeper and loop over entries per selected archive
				if(auto D = search(dest, S); D == dest.end())
					append(dest, S);
				else if constexpr(meta::is_map_v<decltype(src)>) {
					// recursively merge mapped values if they're containers
					if constexpr(meta::is_container_v<decltype(S.second)>)
						self(S.second, D->second, self);
				}
			}
		};

		merger(src, dest, merger);
	};

	Target united;
	Class* storage = nullptr;

	// first pass -- collect all bindings into `united` container
	for(const auto& plugin : K.loaded_plugins_) {
		// extract bindings static variable from plugin descriptor
		if( !(storage = reinterpret_cast<Class*>(plugin.first->*class_storage)) )
			continue;

		// merge plugin bindings into united map
		merge_targets(storage->*class_member, united);
	}

	// assign united map to kernel's plugin descriptor
	auto* kstor = reinterpret_cast<Class*>(K.kernel_pd().*class_storage);
	kstor->*class_member = std::move(united);
	// invoke postprocessing
	postprocess_united_fn(kstor);

	// 2nd pass -- assign back to plugin's maps from kernel's sample
	for(const auto& plugin : K.loaded_plugins_) {
		if( *plugin.first == K.kernel_pd() ||
			!(storage = reinterpret_cast<Class*>(plugin.first->*class_storage))
		)
			continue;

		storage->*class_member = kstor->*class_member;
		// merge all entries from plugin into united
		//merge_targets(united, storage->*class_member);
	}
}

NAMESPACE_END()

void plugins_subsyst::unify_serialization() const {
	using IBM = cereal::detail::InputBindingMap;
	unify_serial_globals(*this, &plugin_descriptor::serial_input_bindings, &IBM::archives_map);

	using OBM = cereal::detail::OutputBindingMap;
	unify_serial_globals(*this, &plugin_descriptor::serial_output_bindings, &OBM::archives_map);

	using PCs = cereal::detail::PolymorphicCasters;
	// [NOTE] important to unify reverseMap first, because it's used on next stage
	unify_serial_globals(*this, &plugin_descriptor::serial_polycasters, &PCs::reverseMap);
	// unify main forward map with postprocessing
	unify_serial_globals(
		*this, &plugin_descriptor::serial_polycasters, &PCs::map,
		{ [this](PCs* kcasters) -> void {
			// refresh polycaster maps (forward and reverse), because new relations are merged
			kcasters->refresh_relations();
			// because `reverseMap` refreshed, spread it among plugins
			for(const auto& plugin : loaded_plugins_) {
				if(auto storage = reinterpret_cast<PCs*>(plugin.first->serial_polycasters);
					*plugin.first != kernel_pd() && storage
				)
					storage->reverseMap = kcasters->reverseMap;
			}
		} }
	);

	// [DEBUG]
	//print_serial_map(kernel_pd().serial_polycasters, &PCs::map, "Polycasters map");
	//print_serial_map(kernel_pd().serial_polycasters, &PCs::reverseMap, "Polycasters reverse map");
}

std::pair< const plugin_descriptor*, bool >
plugins_subsyst::register_plugin(const plugin_descriptor* pd, const lib_descriptor& ld) {
	// deny registering nil plugin descriptors
	if(!pd) return {&plugin_descriptor::nil(), false};

	// find or insert passed plugin_descriptor
	auto res = loaded_plugins_.emplace(pd, ld);
	auto pplug = res.first->first;

	// if plugin with same name was already registered
	if(!res.second) {
		if(pplug->is_nil() && !pd->is_nil()) {
			// replace temp nil plugin with same name
			loaded_plugins_.erase(res.first);
			res = loaded_plugins_.emplace(pd, ld);
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

NAMESPACE_END(blue_sky::kernel::detail)
