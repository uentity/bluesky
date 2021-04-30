/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "tree_fs_impl.h"
#include "../objbase_actor.h"

#include <bs/uuid.h>
#include <bs/tree/errors.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree_fs_input.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>

#include <cereal/types/vector.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <optional>

NAMESPACE_BEGIN(blue_sky)
namespace fs = std::filesystem;

///////////////////////////////////////////////////////////////////////////////
//  tree_fs_input::impl
//
struct tree_fs_input::impl : detail::file_heads_manager<false> {
	using heads_mgr_t = detail::file_heads_manager<false>;
	using fmanager_t = detail::objfrm_manager;
	using Error = tree::Error;

	impl(std::string root_fname, TFSOpts opts) :
		heads_mgr_t{opts, std::move(root_fname)}
	{}

	auto begin_node(tree_fs_input& ar) -> error {
		const auto sentinel = std::optional<tree::node>{};
		return error::eval_safe(
			// sentinel is ONLY used for template matching
			[&] { return head().map([&](auto* ar) { prologue(*ar, *sentinel); }); }
		);
	}

	auto end_node(tree_fs_input& ar, tree::node& N) -> error {
		std::vector<std::string> leafs_order;
		return error::eval_safe(
			// read node's metadata
			[&]{ return head().map( [&](auto* ar) {
				if(N) {
					(*ar)(cereal::make_nvp("leafs_order", leafs_order));
				}
				// we finished reading node
				epilogue(*ar, N);
			}); },
			// load leafs
			[&]{ return N ? load_node(ar, N, std::move(leafs_order)) : perfect; }
		);
	}

	auto load_node(
		tree_fs_input& ar, tree::node& N, std::vector<std::string> leafs_order
	) -> error {
		using namespace allow_enumops;
		using namespace tree;
		using Options = fs::directory_options;

		// skip empty dirs in normal mode
		if(leafs_order.empty()) return perfect;
		// enter node's dir
		if(auto er = enter_dir(links_path_, cur_path_)) return er;

		std::string united_err_msg;
		auto push_error = [&](auto er) {
			if(er.ok()) return;
			if(!united_err_msg.empty()) united_err_msg += " | ";
			united_err_msg += er.what();
		};

		// links to be inserted are collected here first
		auto babies = links_v{};
		babies.reserve(leafs_order.size());

		// fill leafs by scanning directory and loading link files
		std::for_each(
			leafs_order.begin(), leafs_order.end(),
			[&](auto& f) {
				push_error(error::eval_safe(
					[&] { return add_head(cur_path_ / prehash_stem(std::move(f) + link_file_ext)); },
					[&] { // head is removed later by epilogue()
						tree::link L;
						ar(L);
						babies.push_back(std::move(L));
					}
				));
			}
		);

		// insert loaded leafs in one transaction
		push_error(N.apply([&](bare_node N) {
			N.insert(unsafe, std::move(babies));
			return perfect;
		}));

		if(united_err_msg.empty()) return perfect;
		else return united_err_msg;
	}

	auto load_object(tree_fs_input& ar, objbase& obj, bool has_node) -> error {
	return error::eval_safe([&]() -> error {
		// 1, read object format & obtain formatter filename
		std::string obj_frm;
		ar(cereal::make_nvp("fmt", obj_frm));
		auto F = get_formatter(obj.type_id(), obj_frm);
		if(!F) return { fmt::format("{} -> {}", obj.type_id(), obj_frm), Error::MissingFormatter };

		// 2. read `objbase` or `objnode` subobject
		if(has_node) {
			if(!F->stores_node)
				ar(cereal::make_nvp( "object", static_cast<objnode&>(obj) ));
		}
		else
			ar(cereal::make_nvp( "object", obj ));

		// 3. if object is pure node - we're done and can skip data processing
		// [TODO] resolve this via formatter
		if(obj.bs_resolve_type() == objnode::bs_type()) return perfect;

		// 4. format absolute object data file path
		auto abs_obj_path = fs::path{};
		SCOPE_EVAL_SAFE
			// [NOTE] assume objects dir is stored in generic format
			abs_obj_path = fs::absolute(
				objects_path_ / prehash_stem(obj.home_id() + '.' + obj_frm)
			);
		RETURN_SCOPE_ERR

		// 4. read object data
		// instead of posting save job to manager, setup delayed read job
		const auto read_node = has_node && F->stores_node;
		if(auto r = actorf<bool>(
			objbase_actor::actor(obj), kernel::radio::timeout(),
			a_lazy(), a_load(), obj_frm, abs_obj_path.string(), read_node
		); !r)
			return r.error();

		//caf::anon_send(
		//	manager_, obj.shared_from_this(), obj_frm, abs_obj_path.string()
		//);
		//// defer wait until save completes
		//if(!has_wait_deferred_) {
		//	ar(cereal::defer(cereal::Functor{ [](auto& ar){ ar.wait_objects_loaded(); } }));
		//	has_wait_deferred_ = true;
		//}
		return perfect;
	}); }

	auto wait_objects_loaded(timespan how_long) -> std::vector<error> {
		auto res = fmanager_t::wait_jobs_done(manager_, how_long);
		has_wait_deferred_ = false;
		return res;
	}

	// async loaders manager
	bool has_wait_deferred_ = false;
};

///////////////////////////////////////////////////////////////////////////////
//  input archive
//
tree_fs_input::tree_fs_input(std::string root_fname, TFSOpts opts)
	: Base(this), pimpl_{ std::make_unique<impl>(std::move(root_fname), opts) }
{}

tree_fs_input::~tree_fs_input() = default;

auto tree_fs_input::head() -> result_or_err<cereal::JSONInputArchive*> {
	return pimpl_->head();
}

auto tree_fs_input::end_link(const tree::link& L) -> error {
	return pimpl_->end_link(L);
}

auto tree_fs_input::begin_node() -> error {
	return pimpl_->begin_node(*this);
}

auto tree_fs_input::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(*this, const_cast<tree::node&>(N));
}

auto tree_fs_input::load_object(objbase& obj, bool has_node) -> error {
	return pimpl_->load_object(*this, obj, has_node);
}

auto tree_fs_input::wait_objects_loaded(timespan how_long) const -> std::vector<error> {
	return pimpl_->wait_objects_loaded(how_long);
}

auto tree_fs_input::loadBinaryValue(void* data, size_t size, const char* name) -> void {
	head().map([=](auto* jar) {
		jar->loadBinaryValue(data, size, name);
	});
}

///////////////////////////////////////////////////////////////////////////////
//  prologue, epilogue
//
// noop - link head is added explicitly by `impl::load_node()`
auto prologue(tree_fs_input& ar, tree::link const&) -> void {}

// remove head after link is read
auto epilogue(tree_fs_input& ar, tree::link const& L) -> void {
	ar.end_link(L);
}

auto prologue(tree_fs_input& ar, tree::node const&) -> void {
	ar.begin_node();
}

auto epilogue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
