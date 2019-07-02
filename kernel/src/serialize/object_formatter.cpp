/// @file
/// @author uentity
/// @date 01.07.2019
/// @brief Implements manipulations with object formatters
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/object_formatter.h>
#include <bs/type_descriptor.h>

//#include <bs/kernel/misc.h>
//#include <boost/uuid/random_generator.hpp>
//#include <boost/uuid/uuid_io.hpp>

#include <map>
#include <set>
#include <mutex>

NAMESPACE_BEGIN(blue_sky)

// generate random UUID key for storing formatters inside kernel
//const auto impl_key = to_string(boost::uuids::random_generator()());

// compare formatters by name
auto operator<(const object_formatter& lhs, const object_formatter& rhs) {
	return lhs.name < rhs.name;
}
// ... and with arbitrary string key
auto operator<(const object_formatter& lhs, std::string_view rhs) {
	return lhs.name < rhs;
}
auto operator<(std::string_view lhs, const object_formatter& rhs) {
	return lhs < rhs.name;
}

NAMESPACE_BEGIN()
/*-----------------------------------------------------------------------------
 *  Formatters manip impl
 *-----------------------------------------------------------------------------*/
struct fmaster {
	// obj_type_id -> set of unique object_formatter instances sorted by name
	// [NOTE] type_id is stored as string view!
	using fmt_storage_t = std::map< std::string_view, std::set<object_formatter, std::less<>>, std::less<> >;
	fmt_storage_t fmt_storage;

	// sync access to storage above
	std::mutex fmt_guard;

	static auto self() -> fmaster& {
		static fmaster self_;
		return self_;
		//auto& kstorage = kernel::str_key_storage(impl_key);
		//kstorage.insert_element("fmaster", fmaster());
		//return kstorage.ss<fmaster>("fmaster");
	}

	auto install_formatter(const type_descriptor& obj_type, object_formatter&& of) -> bool {
		auto solo = std::lock_guard{ fmt_guard };
		return fmt_storage[obj_type.name].insert(std::move(of)).second;
	}

	auto uninstall_formatter(std::string_view obj_type_id, std::string fmt_name) -> bool {
		// deny removing fallback binary formatter
		if(fmt_name == detail::bin_fmt_name) return false;

		if(auto fmts = fmt_storage.find(obj_type_id); fmts != fmt_storage.end()) {
			auto& fmt_set = fmts->second;
			if(auto pfmt = fmt_set.find(fmt_name); pfmt != fmt_set.end()) {
				// erase format
				auto solo = std::lock_guard{ fmt_guard };
				fmt_set.erase(pfmt);
				return true;
			}
		}
		return false;
	}

	auto formatter_installed(std::string_view obj_type_id, std::string_view fmt_name) -> bool {
		if(auto fmts = fmt_storage.find(obj_type_id); fmts != fmt_storage.end())
			return fmts->second.find(fmt_name) != fmts->second.end();
		return false;
	}

	auto list_installed_formatters(std::string_view obj_type_id) -> std::vector<std::string> {
		auto res = std::vector<std::string>{};
		if(auto fmts = fmt_storage.find(obj_type_id); fmts != fmt_storage.end()) {
			res.reserve(fmts->second.size());
			for(const auto& f : fmts->second)
				res.emplace_back(f.name);
		}
		return res;
	}

	auto get_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> object_formatter* {
		if(auto fmts = fmt_storage.find(obj_type_id); fmts != fmt_storage.end()) {
			auto& fmt_set = fmts->second;
			if(auto pfmt = fmt_set.find(fmt_name); pfmt != fmt_set.end())
				return &const_cast<object_formatter&>(*pfmt);
		}
		return nullptr;
	}
};

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  Formatters manip public API
 *-----------------------------------------------------------------------------*/
#define FM fmaster::self()

auto install_formatter(const type_descriptor& obj_type, object_formatter of) -> bool {
	return FM.install_formatter(obj_type, std::move(of));
}

auto uninstall_formatter(std::string_view obj_type_id, std::string fmt_name) -> bool {
	return FM.uninstall_formatter(obj_type_id, std::move(fmt_name));
}

auto formatter_installed(std::string_view obj_type_id, std::string_view fmt_name) -> bool {
	return FM.formatter_installed(obj_type_id, fmt_name);
}

auto list_installed_formatters(std::string_view obj_type_id) -> std::vector<std::string> {
	return FM.list_installed_formatters(obj_type_id);
}

auto get_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> object_formatter* {
	return FM.get_formatter(obj_type_id, fmt_name);
}

NAMESPACE_END(blue_sky)
