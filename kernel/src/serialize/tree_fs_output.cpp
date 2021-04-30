/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "tree_fs_impl.h"

#include <bs/actor_common.h>
#include <bs/log.h>
#include <bs/tree/errors.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree_fs_output.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>

#include <cereal/types/vector.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

NAMESPACE_BEGIN(blue_sky)
namespace fs = std::filesystem;

using detail::objects_dirname;
using detail::links_dirname;

///////////////////////////////////////////////////////////////////////////////
//  tree_fs_output::impl
//
struct tree_fs_output::impl : detail::file_heads_manager<true> {
	using heads_mgr_t = detail::file_heads_manager<true>;
	using fmanager_t = detail::objfrm_manager;
	using Error = tree::Error;

	impl(std::string root_fname, TFSOpts opts) :
		heads_mgr_t{opts, std::move(root_fname)}
	{}

	auto begin_link(const tree::link& L) -> error {
		if(root_path_.empty()) {
			// add root link head & write correct rel path to objects dir
			auto res = head().map([&](auto* ar) { return error::eval_safe([&] {
				const auto links_rel_path = fs::path{ L.home_id() } / links_dirname;
				const auto objects_rel_path = fs::path{ L.home_id() } / objects_dirname;
				// can skip UTF-8 conversion as dirnames are guaranteed ASCII
				(*ar)(
					cereal::make_nvp("links_dir", links_rel_path.generic_string()),
					cereal::make_nvp("objects_dir", objects_rel_path.generic_string())
				);
				// construct full links/objects path (create nessessary dirs)
				enter_dir(root_path_ / links_rel_path, links_path_, opts_);
				// [NOTE] need to set specific flag to force objects directory cleanup
				enter_dir(
					root_path_ / objects_rel_path, objects_path_,
					enumval(opts_ & TFSOpts::ClearObjectsDir) ? TFSOpts::ClearDirs : TFSOpts::None
				);
			}); });
			return res ? res.value() : res.error();
		}

		return add_head(links_path_ / prehash_stem(to_string(L.id()) + link_file_ext));
	}

	auto begin_node(const tree::node& N) -> error {
		return error::eval_safe(
			[&]{ return head().map( [&](auto* ar) { prologue(*ar, N); }); }
		);
	}

	auto end_node(const tree::node& N) -> error {
		return error::eval_safe(
			// write down node's metadata nessessary to load it later
			[&]{ return head().map( [&](auto* ar) {
				if(N) {
					// custom leafs order
					std::vector<std::string> leafs_order = N.skeys(tree::Key::ID, tree::Key::AnyOrder);
					(*ar)(cereal::make_nvp("leafs_order", leafs_order));
				}
				// and finish
				epilogue(*ar, N);
			}); }
		);
	}

	auto save_object(tree_fs_output& ar, const objbase& obj, bool has_node) -> error {
	return error::eval_safe([&]() -> error {
		std::string obj_fmt;
		bool fmt_ok = false;

		auto finally = scope_guard{ [&]{
			// if error happened
			if(!fmt_ok) ar(cereal::make_nvp("fmt", "<error>"));
		} };

		// 1. obtain & write down formatter
		auto F = get_active_formatter(obj.type_id());
		if(!F) return { obj.type_id(), Error::MissingFormatter };
		obj_fmt = F->name;
		// write down object formatter name
		ar(cereal::make_nvp("fmt", obj_fmt));
		fmt_ok = true;

		// 2. write down `objbase` or `objnode` subobject
		if(has_node) {
			if(!F->stores_node)
				ar(cereal::make_nvp( "object", static_cast<const objnode&>(obj) ));
		}
		else
			ar(cereal::make_nvp( "object", obj ));

		// 3. if object is pure node - we're done and can skip data processing
		if(obj.bs_resolve_type() == objnode::bs_type()) return perfect;

		// 4. save object data to file
		auto abs_obj_path = fs::path{};
		EVAL_SAFE
			[&] {
				abs_obj_path = fs::absolute(
					objects_path_ / prehash_stem(obj.home_id() + '.' + obj_fmt)
				);
			},
			// ensure intermediate dirs are created
			[&] { return enter_dir(abs_obj_path.parent_path()); }
		RETURN_EVAL_ERR

		caf::anon_send(
			manager_, const_cast<objbase&>(obj).shared_from_this(), obj_fmt, abs_obj_path.string()
		);
		// defer wait until save completes
		if(!has_wait_deferred_) {
			ar(cereal::defer(cereal::Functor{ [](auto& ar){ ar.wait_objects_saved(); } }));
			has_wait_deferred_ = true;
		}
		return perfect;
	}); }

	auto wait_objects_saved(timespan how_long) -> std::vector<error> {
		auto res = fmanager_t::wait_jobs_done(manager_, how_long);
		has_wait_deferred_ = false;
		return res;
	}

	auto get_active_formatter(std::string_view obj_type_id) -> object_formatter* {
		if(auto paf = active_fmt_.find(obj_type_id); paf != active_fmt_.end())
			return get_formatter(obj_type_id, paf->second);
		else {
			// if bin format installed - select it
			if(select_active_formatter(obj_type_id, detail::bin_fmt_name))
				return get_active_formatter(obj_type_id);
			else if(
				auto frms = list_installed_formatters(obj_type_id);
				!frms.empty() && select_active_formatter(obj_type_id, *frms.begin())
			)
				return get_active_formatter(obj_type_id);

		}
		return nullptr;
	}

	auto select_active_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> bool {
		if(auto pfmt = get_formatter(obj_type_id, fmt_name); pfmt) {
			active_fmt_[obj_type_id] = fmt_name;
			return true;
		}
		return false;
	}

	///////////////////////////////////////////////////////////////////////////////
	//  data
	//
	// obj_type_id -> formatter name
	using active_fmt_t = std::map<std::string_view, std::string, std::less<>>;
	active_fmt_t active_fmt_;

	bool has_wait_deferred_ = false;
};

///////////////////////////////////////////////////////////////////////////////
//  output archive
//
tree_fs_output::tree_fs_output(std::string root_fname, TFSOpts opts)
	: Base(this), pimpl_{ std::make_unique<impl>(std::move(root_fname), opts) }
{}

tree_fs_output::~tree_fs_output() = default;

auto tree_fs_output::head() -> result_or_err<cereal::JSONOutputArchive*> {
	return pimpl_->head();
}

auto tree_fs_output::begin_link(const tree::link& L) -> error {
	return pimpl_->begin_link(L);
}

auto tree_fs_output::end_link(const tree::link& L) -> error {
	return pimpl_->end_link(L);
}

auto tree_fs_output::begin_node(const tree::node& N) -> error {
	return pimpl_->begin_node(N);
}

auto tree_fs_output::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(N);
}

auto tree_fs_output::save_object(const objbase& obj, bool has_node) -> error {
	return pimpl_->save_object(*this, obj, has_node);
}

auto tree_fs_output::wait_objects_saved(timespan how_long) const -> std::vector<error> {
	return pimpl_->wait_objects_saved(how_long);
}

auto tree_fs_output::saveBinaryValue(const void* data, size_t size, const char* name) -> void {
	head().map([=](cereal::JSONOutputArchive* jar) {
		jar->saveBinaryValue(data, size, name);
	});
}

auto tree_fs_output::get_active_formatter(std::string_view obj_type_id) -> object_formatter* {
	return pimpl_->get_active_formatter(obj_type_id);
}

auto tree_fs_output::select_active_formatter(std::string_view obj_type_id, std::string_view fmt_name) -> bool {
	return pimpl_->select_active_formatter(obj_type_id, fmt_name);
}

///////////////////////////////////////////////////////////////////////////////
//  prologue, epilogue
//

auto prologue(tree_fs_output& ar, tree::link const& L) -> void {
	ar.begin_link(L);
}

auto epilogue(tree_fs_output& ar, tree::link const& L) -> void {
	ar.end_link(L);
}

auto prologue(tree_fs_output& ar, tree::node const& N) -> void {
	ar.begin_node(N);
}

auto epilogue(tree_fs_output& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
