/// @file
/// @author uentity
/// @date 01.07.2019
/// @brief Implements manipulations with object formatters
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/object_formatter.h>
#include <bs/objbase.h>
#include <bs/kernel/misc.h>

#include <boost/uuid/random_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

#include <map>
#include <set>
#include <mutex>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN()

/*-----------------------------------------------------------------------------
 *  Formatters manip impl
 *-----------------------------------------------------------------------------*/
struct fmaster {
	// obj_type_id -> set of unique object_formatter instances sorted by name
	// [NOTE] type_id is stored as string view!
	using fmt_storage_t = std::map< std::string_view, std::set<object_formatter, std::less<>>, std::less<> >;
	fmt_storage_t fmt_storage;

	using registry_t = std::map<const objbase*, std::string_view>;
	registry_t registry;

	// sync access to storage above
	std::mutex fmt_guard;

	fmaster() {}
	fmaster(const fmaster& rhs) : fmt_storage(rhs.fmt_storage) {}
	fmaster(fmaster&& rhs) : fmt_storage(std::move(rhs.fmt_storage)) {}

	static auto self() -> fmaster& {
		static fmaster& self = []() -> fmaster& {
			// generate random key
			auto& kstorage = kernel::idx_key_storage(to_string( boost::uuids::random_generator()() ));
			auto r = kstorage.insert_element(0, fmaster());
			if(!r.first) throw error("Failed to make impl of object formatters in kernel storage!");
			return *r.first;
		}();

		return self;
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

	auto register_formatter(const objbase& obj, std::string_view fmt_name) -> bool {
		if(auto frm = get_formatter(obj.bs_resolve_type().name, fmt_name)) {
			auto solo = std::lock_guard{ fmt_guard };
			registry[&obj] = frm->name;
			return true;
		}
		return false;
	}

	auto deregister_formatter(const objbase& obj) -> bool {
		if(auto r = registry.find(&obj); r != registry.end()) {
			auto solo = std::lock_guard{ fmt_guard };
			registry.erase(r);
			return true;
		}
		return false;
	}

	auto get_obj_formatter(const objbase* obj) -> object_formatter* {
		if(auto r = registry.find(obj); r != registry.end())
			return get_formatter(obj->bs_resolve_type().name, r->second);
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

auto get_obj_formatter(const objbase* obj) -> object_formatter* {
	return FM.get_obj_formatter(obj);
}

/*-----------------------------------------------------------------------------
 *  object_formatter
 *-----------------------------------------------------------------------------*/
object_formatter::object_formatter(
	std::string fmt_name, object_saver_fn saver, object_loader_fn loader, bool stores_node_
) : base_t{std::move(saver), std::move(loader)}, name(std::move(fmt_name)), stores_node(stores_node_)
{}

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

auto object_formatter::save(const objbase& obj, std::string obj_fname) const -> error {
	FM.register_formatter(obj, name);
	auto res = error::eval_safe([&]{ first(obj, std::move(obj_fname), name); });
	FM.deregister_formatter(obj);
	return res;
}

auto object_formatter::load(objbase& obj, std::string obj_fname) const -> error {
	FM.register_formatter(obj, name);
	auto res = error::eval_safe([&]{ second(obj, std::move(obj_fname), name); });
	FM.deregister_formatter(obj);
	return res;
}

NAMESPACE_END(blue_sky)
