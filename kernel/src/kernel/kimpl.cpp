/// @file
/// @author uentity
/// @date 14.01.2019
/// @brief kernel singleton definition
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/log.h>
#include <bs/kernel/misc.h>

#include "kimpl.h"
#include "radio_subsyst.h"
#include "config_subsyst.h"

#include <fmt/format.h>
#include <spdlog/spdlog.h>

#ifdef BSPY_EXPORTING
#include "python_subsyst_impl.h"
#else
#include "python_subsyst.h"

NAMESPACE_BEGIN()

struct python_subsyt_dumb : public blue_sky::kernel::detail::python_subsyst {
	auto py_init_plugin(
		const blue_sky::detail::lib_descriptor&, plugin_descriptor&
	) -> result_or_err<std::string> override {
		return "";
	}

	auto py_add_error_closure() -> void override {}
	auto setup_py_kmod(void*) -> void override {};
	auto py_kmod() const -> void* override { return nullptr; }
};

NAMESPACE_END()

#endif

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(kernel)

kimpl::kimpl() : init_state_(InitState::NonInitialized) {}

kimpl::~kimpl() = default;

auto kimpl::get_radio() -> detail::radio_subsyst* {
	std::call_once(radio_up_, [&] { radio_ss_ = std::make_unique<detail::radio_subsyst>(); });
	return radio_ss_.get();
}

auto kimpl::pysupport() -> detail::python_subsyst* {
	std::call_once(py_up_, [&] {
		// setup Python support
#ifdef BSPY_EXPORTING
		pysupport_ = std::make_unique<detail::python_subsyst_impl>();
#else
		pysupport_ = std::make_unique<python_subsyt_dumb>();
#endif
	});
	return pysupport_.get();
}

auto kimpl::init() -> error {
	// do initialization only once from non-initialized state
	auto expected_state = InitState::NonInitialized;
	if(init_state_.compare_exchange_strong(expected_state, InitState::Initialized)) {
		// if init wasn't finished - return kernel to non-initialized status
		auto init_ok = false;
		auto finally = scope_guard{ [&]{ if(!init_ok) init_state_ = InitState::NonInitialized; } };

		// configure kernel
		configure();
		// switch to mt logs
		toggle_mt_logs(true);
		// init kernel radio subsystem
		auto er = get_radio()->init();
		init_ok = er.ok();
		return er;
	}
	return perfect;
}

auto kimpl::shutdown() -> void {
	// shut down if not already Down
	auto expected_state = InitState::Initialized;
	if(init_state_.compare_exchange_strong(expected_state, InitState::Down)) {
		// turn off radio subsystem
		if(radio_ss_) radio_ss_->shutdown();
		// shutdown mt logs
		toggle_mt_logs(false);
		spdlog::shutdown();
	}
}

auto kimpl::find_type(const std::string& key) const -> type_tuple {
	using search_key = plugins_subsyst::type_name_key;

	auto& I = types_.get<search_key>();
	auto tp = I.find(key);
	return tp != I.end() ? *tp : type_tuple();
}

auto kimpl::str_key_storage(const std::string& key) -> str_any_array& {
	return str_key_storage_[key];
}

auto kimpl::idx_key_storage(const std::string& key) -> idx_any_array& {
	return idx_key_storage_[key];
}

NAMESPACE_END(kernel)

/*-----------------------------------------------------------------------------
 *  kernel subsystems impl instantiation
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

template<> auto singleton<logging_subsyst>::Instance() -> logging_subsyst& {
	return static_cast<logging_subsyst&>(KIMPL);
}

template<> auto singleton<radio_subsyst>::Instance() -> radio_subsyst& {
	return *KIMPL.get_radio();
}

template<> auto singleton<python_subsyst>::Instance() -> python_subsyst& {
	return *KIMPL.pysupport();
}

NAMESPACE_END(blue_sky::kernel)
