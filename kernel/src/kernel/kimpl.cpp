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
#include "radio_subsyst.h"

#include <fmt/format.h>

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

	// construct `error` from any int value -- call after all modules initialized
	auto py_add_error_closure() -> void override {}

	auto setup_py_kmod(void*) -> void override {};
	auto py_kmod() const -> void* override { return nullptr; }
};

NAMESPACE_END()
#endif

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(kernel)

kimpl::kimpl()
	: init_state_(InitState::NonInitialized)
{
	// setup Python support
#ifdef BSPY_EXPORTING
	pysupport_ = std::make_unique<detail::python_subsyst_impl>();
#else
	pysupport_ = std::make_unique<python_subsyt_dumb>();
#endif
}

kimpl::~kimpl() = default;

auto kimpl::init_radio() -> error {
	try {
		if(radio_ss_)
			return radio_ss_->init();
		else
			radio_ss_ = std::make_unique<detail::radio_subsyst>();
	}
	catch(error& er) {
		radio_ss_.reset();
		return std::move(er);
	}
	return perfect;
}

auto kimpl::shutdown_radio() -> void {
	if(radio_ss_) radio_ss_->shutdown();
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
	// ensure radio subsystem is initialized
	static auto& radio = []() -> radio_subsyst& {
		KIMPL.init_radio();
		return *KIMPL.radio_ss_;
	}();
	return radio;
}

template<> auto singleton<python_subsyst>::Instance() -> python_subsyst& {
	return static_cast<python_subsyst&>(*KIMPL.pysupport_);
}

NAMESPACE_END(blue_sky)
