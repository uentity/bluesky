/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief kernel singleton definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/kernel/misc.h>
#include "kimpl.h"
#ifdef BSPY_EXPORTING
#include "python_subsyst_impl.h"
#endif

#include <caf/actor_system.hpp>
#include <fmt/format.h>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(kernel)
NAMESPACE_BEGIN()

struct python_subsyt_dumb : public detail::python_subsyst {
	auto py_init_kernel() -> error { return success(); }

	auto py_init_plugin(
		const blue_sky::detail::lib_descriptor&, plugin_descriptor&
	) -> result_or_err<std::string> {
		return "";
	}

	// construct `error` from any int value -- call after all modules initialized
	auto py_add_error_closure() -> void {}

	auto setup_py_kmod(void*) -> void {};
};

NAMESPACE_END()

kimpl::kimpl()
	: init_state_(InitState::NonInitialized)
{
	// [NOTE] We can't create `caf::actor_system` here.
	// `actor_system` starts worker and other service threads in constructor.
	// At the same time kernel singleton is constructed most of the time during
	// initialization of kernel shared library. And on Windows it is PROHIBITED to start threads
	// in `DllMain()`, because that cause a deadlock.
	// Solution: delay construction of actor_system until first usage, don't use CAf in kernel ctor.

	// setup Python support
#ifdef BSPY_EXPORTING
	pysupport_ = std::make_unique<detail::python_subsyst_impl>();
#else
	pysupport_ = std::make_unique<python_subsyt_dumb>();
#endif
}

kimpl::~kimpl() = default;

auto kimpl::find_type(const std::string& key) const -> type_tuple {
	using search_key = plugins_subsyst::type_name_key;

	auto& I = types_.get<search_key>();
	auto tp = I.find(key);
	return tp != I.end() ? *tp : type_tuple();
}

auto kimpl::pert_str_any_array(const std::string& key) -> str_any_array& {
	return str_any_map_[key];
}

auto kimpl::pert_idx_any_array(const std::string& key) -> idx_any_array& {
	return idx_any_map_[key];
}

auto kimpl::actor_system() -> caf::actor_system& {
	// delayed actor system initialization
	// [TODO] write safer code
	static auto* actor_sys = [this]{
		init();
		return actor_sys_.get();
	}();
	return *actor_sys;
}

NAMESPACE_END(kernel)

/*-----------------------------------------------------------------------------
 *  kernel internal signleton instantiation
 *-----------------------------------------------------------------------------*/
using namespace kernel::detail;

template<> auto singleton<kernel::kimpl>::Instance() -> kernel::kimpl& {
	static kernel::kimpl K;
	return K;
}

// access individual subsystems
template<> auto singleton<config_subsyst>::Instance() -> config_subsyst& {
	return static_cast<config_subsyst&>(KIMPL);
}

template<> auto singleton<plugins_subsyst>::Instance() -> plugins_subsyst& {
	return static_cast<plugins_subsyst&>(KIMPL);
}

template<> auto singleton<instance_subsyst>::Instance() -> instance_subsyst& {
	return static_cast<instance_subsyst&>(KIMPL);
}

template<> auto singleton<logging_subsyst>::Instance() -> logging_subsyst& {
	return static_cast<logging_subsyst&>(KIMPL);
}

template<> auto singleton<python_subsyst>::Instance() -> python_subsyst& {
	return static_cast<python_subsyst&>(*KIMPL.pysupport_);
}

NAMESPACE_END(blue_sky)
