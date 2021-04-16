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
#include "../tree/private_common.h"

#include <bs/serialize/cafbind.h>
#include <bs/serialize/propdict.h>
#include <bs/serialize/tree.h>
#include "../serialize/tree_impl.h"

#include <caf/init_global_meta_objects.hpp>
#include <caf/io/middleman.hpp>

#include <fmt/format.h>

#ifdef BSPY_EXPORTING
#include "python_subsyst_impl.h"
#else
#include "python_subsyst.h"

NAMESPACE_BEGIN()

struct python_subsyt_dumb : public blue_sky::kernel::detail::python_subsyst {
	auto py_init_plugin(
		const blue_sky::detail::lib_descriptor&, blue_sky::plugin_descriptor&
	) -> blue_sky::result_or_err<std::string> override {
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

kimpl::~kimpl() {
	// [NOTE] shutdown is required to preperly terminate logs, etc
	shutdown();
}

auto kimpl::get_radio() -> detail::radio_subsyst* {
	std::call_once(radio_up_, [&] { radio_ss_ = std::make_unique<detail::radio_subsyst>(); });
	return radio_ss_.get();
}

auto kimpl::get_config() -> detail::config_subsyst* {
	std::call_once(config_up_, [&] {
		// instantiate global meta objects
		caf::init_global_meta_objects<caf::id_block::bs_atoms>();
		caf::init_global_meta_objects<caf::id_block::bs>();
		caf::init_global_meta_objects<caf::id_block::bs_private>();
		caf::init_global_meta_objects<caf::id_block::bs_props>();
		caf::init_global_meta_objects<caf::id_block::bs_tr>();
		caf::init_global_meta_objects<caf::id_block::bs_tree>();
		caf::core::init_global_meta_objects();
		caf::io::middleman::init_global_meta_objects();

		// construct config subsystem
		config_ss_ = std::make_unique<detail::config_subsyst>();
	});
	return config_ss_.get();
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
		get_config()->configure();
		// switch to mt logs
		logging_subsyst::toggle_async(true);
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
		// shutdown logging subsyst
		logging_subsyst::shutdown();
	}
}

auto kimpl::find_type(const std::string& key) const -> type_tuple {
	using search_key = plugins_subsyst::type_name_key;

	auto& I = types_.get<search_key>();
	auto tp = I.find(key);
	return tp != I.end() ? *tp : type_tuple();
}

auto kimpl::str_key_storage(const std::string& key) -> str_any_array& {
	auto solo = std::lock_guard{ sync_storage_ };
	return str_key_storage_[key];
}

auto kimpl::idx_key_storage(const std::string& key) -> idx_any_array& {
	auto solo = std::lock_guard{ sync_storage_ };
	return idx_key_storage_[key];
}

auto kimpl::gen_uuid() -> boost::uuids::uuid {
	auto guard = std::lock_guard{ sync_uuid_ };
	return uuid_gen_();
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
template<> auto singleton<plugins_subsyst>::Instance() -> plugins_subsyst& {
	return static_cast<plugins_subsyst&>(KIMPL);
}

template<> auto singleton<logging_subsyst>::Instance() -> logging_subsyst& {
	return static_cast<logging_subsyst&>(KIMPL);
}

template<> auto singleton<config_subsyst>::Instance() -> config_subsyst& {
	return *KIMPL.get_config();
}

template<> auto singleton<radio_subsyst>::Instance() -> radio_subsyst& {
	return *KIMPL.get_radio();
}

template<> auto singleton<python_subsyst>::Instance() -> python_subsyst& {
	return *KIMPL.pysupport();
}

NAMESPACE_END(blue_sky::kernel)
