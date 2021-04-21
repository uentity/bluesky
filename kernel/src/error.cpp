/// @file
/// @author uentity
/// @date 18.08.2016
/// @brief Contains implimentations of BlueSky exceptions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/error.h>
#include <bs/actor_common.h>
#include <bs/tree/errors.h>
#include <bs/misc.h>
#include <bs/log.h>
#include <bs/uuid.h>
#include <bs/kernel/tools.h>
#include <bs/kernel/misc.h>
#include <bs/kernel/radio.h>

#include <fmt/format.h>
#include <fmt/compile.h>

#include <ostream>
#include <string_view>
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
			auto& kstorage = kernel::idx_key_storage(to_string(gen_uuid()));
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

const auto& ok_err_code() {
	static const auto v = make_error_code(Error::OK);
	return v;
};

const auto& fail_err_code() {
   static const auto v = make_error_code(Error::Happened);
   return v;
}

#define ECR cat_registry::self()

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  Default error category for generic exception
 *-----------------------------------------------------------------------------*/
std::error_code make_error_code(Error e) {
	// implement error categiry for default error code
	struct default_category : error::category<default_category> {
		const char* name() const noexcept override {
			return "BS";
		}

		std::string message(int ec) const override {
			switch(static_cast<Error>(ec)) {
			case Error::TrEmptyTarget:
				return "Transaction target is empty";

			case Error::Happened:
				return "runtime error";

			default:
				return {};
			};
		}
	};

	return { static_cast<int>(e), default_category::self() };
}

/*-----------------------------------------------------------------------------
 *  error implementation
 *-----------------------------------------------------------------------------*/
error::error(IsQuiet quiet, const char* message, std::error_code ec) noexcept :
	code(ec == Error::Undefined ?
		(quiet == IsQuiet::Yes ? ok_err_code() : fail_err_code()) :
		std::move(ec)
	)
{
	if(message) if(auto msg_v = std::string_view{message}; !msg_v.empty())
		info.emplace(message);
	try {
		if(quiet == IsQuiet::No) dump();
	} catch(...) {}
}
error::error(IsQuiet quiet, const std::string& message, std::error_code ec) noexcept :
	error(quiet, message.c_str(), ec)
{}

error::error(IsQuiet quiet, const char* message, int ec, std::string_view cat_name) noexcept
	: error(quiet, message, ECR.make_error_code(ec, cat_name))
{}

error::error(IsQuiet quiet, const std::string& message, int ec, std::string_view cat_name) noexcept
	: error(quiet, message.c_str(), ECR.make_error_code(ec, cat_name))
{}

error::error(IsQuiet quiet, std::error_code ec) noexcept
	: error(quiet, nullptr, std::move(ec))
{}

error::error(IsQuiet quiet, const std::system_error& er) noexcept
	: error(quiet, er.what(), er.code())
{}

error::error(IsQuiet quiet, raw_code ec, std::string_view cat_name) noexcept
	: error(quiet, nullptr, ECR.make_error_code(ec.value, cat_name))
{}

// [NOTE] unpacking is quiet
error::error(IsQuiet, box b) noexcept
	: error(IsQuiet::Yes, b.message, ECR.make_error_code(b.ec, b.domain))
{}

error::error(IsQuiet, quiet_flag f) noexcept : code(f.value ? ok_err_code() : fail_err_code())
{}

// put passed error_category into registry
auto error::register_category(std::error_category const* cat) -> void {
	ECR.register_category(cat);
}

auto error::what() const -> std::string {
	std::string res = code.message();
	if(info) {
		res += res.empty() ? "|> " : " |> ";
		res += info->what();
	}
	return res;
}

const char* error::domain() const noexcept {
	return code.category().name();
}

const char* error::message() const noexcept {
	return info ? info->what() : "";
}

bool error::ok() const noexcept {
	return !(bool)code;
}

BS_API std::string to_string(const error& er) {
	std::string s = fmt::format(FMT_COMPILE("[{}] [{}] {}"), er.domain(), er.code.value(), er.what());
#if defined(_DEBUG) && !defined(_MSC_VER)
	if(!er.ok()) s += kernel::tools::get_backtrace(16, 4);
#endif
	return s;
}

auto error::pack() const -> box {
	return *this;
}

auto error::unpack(box b) noexcept -> error {
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

error::box::box(const error& er)
	: ec(er.code.value()), domain(er.domain())
{
	if(er.info) message = er.info->what();
}

error::box::box(int ec_, std::string domain_, std::string message_) noexcept
	: ec(ec_), domain(std::move(domain_)), message(std::move(message_))
{}

///////////////////////////////////////////////////////////////////////////////
//  CAF errors passthrough
//
auto forward_caf_error(const caf::error& er, std::string_view msg) -> error {
	// produce std::error_code from CAF error
	static const auto caf_error_code = [](const caf::error& er) -> std::error_code {
		struct caf_category : error::category<caf_category> {
			const char* name() const noexcept override { return "CAF"; }

			std::string message(int ec) const override { return ""; }
		};

		return { er.code(), caf_category::self() };
	};

	auto ermsg = to_string(er);
	if(!msg.empty()) {
		ermsg += " |> ";
		ermsg += msg;
	}
	return { std::move(ermsg), caf_error_code(er) };
}

NAMESPACE_END(blue_sky)
