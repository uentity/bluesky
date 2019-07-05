/// @file
/// @author uentity
/// @date 18.08.2016	
/// @brief Contains implimentations of BlueSky exceptions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/tree/errors.h>
#include <bs/misc.h>
#include <bs/log.h>
#include <bs/kernel/tools.h>
#include <bs/kernel/misc.h>

#include <fmt/format.h>
#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <cstring>
#include <unordered_map>
#include <mutex>
#include <iostream>

using namespace std;

NAMESPACE_BEGIN(blue_sky)
/*-----------------------------------------------------------------------------
 *  hidden details
 *-----------------------------------------------------------------------------*/
namespace {

// not throwing error message formatter
inline std::string format_errmsg(std::string ec_message, std::string custom_message) {
	return ec_message.size() ? (
			custom_message.size() ?
				std::move(ec_message) + ": " + std::move(custom_message) : std::move(ec_message)
		) :
		std::move(custom_message);
};

// implements categories registry
struct cat_registry {
	using cat_dict_t = std::unordered_map<std::string_view, std::error_category const*>;
	cat_dict_t cat_dict;
	// sync access to dict
	std::mutex cat_sync;

	cat_registry() = default;
	cat_registry(const cat_registry& rhs) : cat_dict(rhs.cat_dict) {}
	cat_registry(cat_registry&& rhs) : cat_dict(std::move(rhs.cat_dict)) {}

	static auto self() -> cat_registry& {
		static cat_registry& self = []() -> cat_registry& {
			// generate random key
			auto& kstorage = kernel::idx_key_storage(to_string( boost::uuids::random_generator()() ));
			auto r = kstorage.insert_element(0, cat_registry{});
			if(!r.first) throw error("Failed to instantiate error categories registry in kernel storage!");
			return *r.first;
		}();

		return self;
	}

	auto register_category(std::error_category const* cat) -> void {
		if(!cat) return;
		auto solo = std::lock_guard{ cat_sync };
		cat_dict[cat->name()] = cat;
	}

	auto lookup_category(std::string_view name) const -> std::error_category const* {
		if(auto pcat = cat_dict.find(name); pcat != cat_dict.end())
			return pcat->second;
		return nullptr;
	}

	auto make_error_code(int ec, std::string_view cat_name) -> std::error_code {
		if(auto pcat = lookup_category(cat_name); pcat)
			return { ec, *pcat };
		// fallback to default category
		return { static_cast<Error>(ec) };
	}
};

#define ECR cat_registry::self()

} // hidden namespace

/*-----------------------------------------------------------------------------
 *  Default error category for generic exception
 *-----------------------------------------------------------------------------*/
std::error_code make_error_code(Error e) {
	// implement error categiry for default error code
	struct default_category : error::category<default_category> {
		const char* name() const noexcept override {
			return "blue_sky";
		}

		std::string message(int ec) const override {
			// in any case we should just substitute custom error message
				return "";
		}
	};

	return { static_cast<int>(e), default_category::self() };
}

/*-----------------------------------------------------------------------------
 *  error implementation
 *-----------------------------------------------------------------------------*/
error::error(IsQuiet quiet, std::string message, std::error_code ec)
	: runtime_error(format_errmsg(ec.message(), std::move(message))),
	  code(ec == Error::Undefined ? (quiet == IsQuiet::Yes ? Error::OK : Error::Happened) : std::move(ec))
{
	if(quiet == IsQuiet::No) dump();
}

error::error(IsQuiet quiet, std::error_code ec) : error(quiet, "", std::move(ec)) {}

error::error(IsQuiet quiet, std::string message, int ec, std::string_view cat_name)
	: error(quiet, std::move(message), ECR.make_error_code(ec, cat_name))
{}

error::error(IsQuiet quiet, int ec, std::string_view cat_name)
	: error(quiet,
		ECR.lookup_category(cat_name) ? "" : fmt::format("Unknown error from category '{}'", cat_name),
		ECR.make_error_code(ec, cat_name)
	)
{}

error::error(success_tag) : error(IsQuiet::Yes, Error::OK) {}

// copy & move ctors are default
error::error(const error& rhs) noexcept = default;
error::error(error&& rhs) noexcept = default;

// put passed error_category into registry
auto error::register_category(std::error_category const* cat) -> void {
	ECR.register_category(cat);
}

const char* error::domain() const noexcept {
	return code.category().name();
}

std::string error::to_string() const {
	std::string s = fmt::format("[{}] [{}] {}", domain(), code.value(), what());
#if defined(_DEBUG) && !defined(_MSC_VER)
	if(!ok()) s += kernel::tools::get_backtrace(20, 4);
#endif
	return s;
}

void error::dump() const {
	//const auto msg = fmt::format("[{}] [{}] {}", domain(), code.value(), what());
	if(code)
		bserr() << log::E(to_string()) << log::end;
	else
		bsout() << log::I(to_string()) << log::end;
}

bool error::ok() const {
	static const auto tree_extra_ok = tree::make_error_code(tree::Error::OKOK);

	return !(bool)code || (code == tree_extra_ok);
}

// error printing
BS_API std::ostream& operator <<(std::ostream& os, const error& ec) {
	return os << ec.to_string();
}

NAMESPACE_END(blue_sky)

