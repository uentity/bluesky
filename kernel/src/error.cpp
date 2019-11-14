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

using namespace std;

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN()

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

	auto lookup_sys_category(std::string_view name) const -> std::error_category const* {
		static std::error_category const* syscats[] = {
			&std::system_category(), &std::generic_category(), &std::iostream_category()
		};

		for(auto cat : syscats) { if(cat->name() == name) return cat; }
		return nullptr;
	}

	auto lookup_category(std::string_view name) const -> std::error_category const* {
		if(auto pcat = cat_dict.find(name); pcat != cat_dict.end())
			return pcat->second;
		else
			return lookup_sys_category(name);
	}

	auto make_error_code(int ec, std::string_view cat_name) -> std::error_code {
		if(auto pcat = lookup_category(cat_name); pcat)
			return { ec, *pcat };
		// fallback to default category
		return { static_cast<Error>(ec) };
	}
};

#define ECR cat_registry::self()

NAMESPACE_END()

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
			return "Generic blue-sky error";
		}
	};

	return { static_cast<int>(e), default_category::self() };
}

/*-----------------------------------------------------------------------------
 *  error implementation
 *-----------------------------------------------------------------------------*/
error::error(IsQuiet quiet, std::string message, std::error_code ec)
	: runtime_error([ec_msg = ec.message(), msg = std::move(message)]() mutable {
		auto res = std::move(ec_msg);
		if(!msg.empty()) {
			res += res.empty() ? "|> " : " |> ";
			res += std::move(msg);
		}
		return res;
	}()),
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

error::error(IsQuiet quiet, const std::system_error& er)
	: runtime_error(er.what()), code(er.code())
{}

// [NOTE] unpacking is always quiet
error::error(IsQuiet quiet, box b) : error(IsQuiet::Yes, std::move(b.message), b.ec, b.domain) {}

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

const char* error::message() const noexcept {
	if(ECR.lookup_sys_category(domain()))
		return what();
	// custom message is tail part delimited by semicolon and space
	else if(auto pos = strstr(what(), "|>"); pos)
		return pos + 3;
	return "";
}

bool error::ok() const {
	static const auto tree_extra_ok = tree::make_error_code(tree::Error::OKOK);

	return !(bool)code || (code == tree_extra_ok);
}

BS_API std::string to_string(const error& er) {
	std::string s = fmt::format("[{}] [{}] {}", er.domain(), er.code.value(), er.what());
#if defined(_DEBUG) && !defined(_MSC_VER)
	if(!er.ok()) s += kernel::tools::get_backtrace(16, 4);
#endif
	return s;
}

auto error::pack() const -> box {
	return { code.value(), message(), domain() };
}

auto error::unpack(box b) -> error {
	return error{ IsQuiet::Yes, std::move(b) };
}

void error::dump() const {
	if(code)
		bserr() << log::E(to_string(*this)) << log::end;
	else
		bsout() << log::I(to_string(*this)) << log::end;
}

// error printing
BS_API std::ostream& operator <<(std::ostream& os, const error& ec) {
	return os << to_string(ec);
}

///////////////////////////////////////////////////////////////////////////////
//  error::box
//

error::box::box(const error& er) {
	*this = er.pack();
}

error::box::box(int ec_, std::string message_, std::string domain_)
	: ec(ec_), message(std::move(message_)), domain(std::move(domain_))
{}

NAMESPACE_END(blue_sky)
