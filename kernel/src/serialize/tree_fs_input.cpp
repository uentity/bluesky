/// @file
/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "tree_fs_impl.h"

#include <bs/uuid.h>
#include <bs/tree/errors.h>
#include <bs/tree/node.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree_fs_input.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <cereal/types/vector.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <optional>

NAMESPACE_BEGIN(blue_sky)
namespace fs = std::filesystem;
using NodeLoad = tree_fs_input::NodeLoad;

///////////////////////////////////////////////////////////////////////////////
//  tree_fs_input::impl
//
struct tree_fs_input::impl : detail::file_heads_manager<false> {
	using heads_mgr_t = detail::file_heads_manager<false>;
	using fmanager_t = detail::objfrm_manager;
	using Error = tree::Error;

	impl(std::string root_fname, NodeLoad mode) :
		heads_mgr_t{std::move(root_fname)}, mode_(mode)
	{}

	auto begin_node(tree_fs_input& ar) -> error {
		const auto sentinel = std::optional<tree::node>{};
		return error::eval_safe(
			// sentinel is ONLY used for template matching
			[&] { return head().map([&](auto* ar) { prologue(*ar, *sentinel); }); },
			[&] { return enter_root(); }
		);
	}

	auto end_node(tree_fs_input& ar, tree::node& N) -> error {
		if(cur_path_.empty()) return Error::NodeWasntStarted;

		// always return to parent dir after node is loaded
		auto finally = scope_guard{[=, p = cur_path_] {
			if(auto er = enter_dir(p, cur_path_)) throw er;
		}};

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
		using Options = fs::directory_options;

		// skip empty dirs in normal mode
		if(mode_ == NodeLoad::Normal && leafs_order.empty()) return perfect;
		// enter node's dir
		if(auto er = enter_dir(cur_path_ / N.home_id(), cur_path_)) return er;

		std::string united_err_msg;
		auto push_error = [&](auto er) {
			if(er.ok()) return;
			if(!united_err_msg.empty()) united_err_msg += " | ";
			united_err_msg += er.what();
		};

		// fill leafs by scanning directory and loading link files
		const auto normal_load = [&] { std::for_each(
			leafs_order.begin(), leafs_order.end(),
			[&](auto& f) {
				push_error(error::eval_safe(
					[&] { add_head(cur_path_ / std::move(f)); },
					[&] { // head is removed later by epilogue()
						tree::link L;
						ar(L);
						N.insert(std::move(L));
					}
				));
			}
		); };

		const auto recover_load = [&] {
			// setup node iterator
			auto Niter = fs::directory_iterator{};
			if(auto er = error::eval_safe([&] {
				Niter = fs::directory_iterator(cur_path_, Options::skip_permission_denied);
			})) {
				push_error(std::move(er));
				return;
			}

			// read links
			for(auto& f : Niter) {
				// skip directories
				if(error::eval_safe([&]{ return !fs::is_directory(f); })) continue;

				// try load file as a link
				push_error(error::eval_safe(
					[&] { add_head(f); },
					[&] { // head is removed later by epilogue()
						tree::link L;
						ar(L);
						N.insert(std::move(L));
					}
				));
			}

			// [NOTE] implementation with parallel STL
			//std::for_each(
			//	std::execution::par, begin(Niter), end(Niter),
			//	[&](auto& f) {
			//		// skip directories
			//		if(error::eval_safe([&]{ return !fs::is_directory(f); })) return;

			//		// try load file as a link
			//		push_error(error::eval_safe(
			//			[&] { add_head(f); },
			//			[&] {
			//				tree::link L;
			//				ar(L);
			//				N.insert(std::move(L));
			//			}
			//		));
			//	}
			//);

			// restore links order
			using namespace tree;
			if(N.size() < 2 || leafs_order.size() < 2) return;

			// convert string uids to UUIDs
			auto wanted_order = lids_v(leafs_order.size());
			std::transform(
				leafs_order.cbegin(), leafs_order.cend(), wanted_order.begin(),
				[](const auto& s_uid) { return to_uuid_raw(s_uid); }
			);

			// extract current order of link IDs
			auto res_order = N.keys(Key::AnyOrder);

			// sort according to passed `leafs_order`
			const auto lo_begin = wanted_order.begin(), lo_end = wanted_order.end();
			std::sort(res_order.begin(), res_order.end(), [&](auto i1, auto i2) {
				return std::find(lo_begin, lo_end, i1) < std::find(lo_begin, lo_end, i2);
			});
			// apply custom order
			N.rearrange(std::move(res_order));
		};

		// invoke laod
		if(mode_ == NodeLoad::Normal)
			normal_load();
		else
			recover_load();

		if(united_err_msg.empty()) return perfect;
		else return united_err_msg;
	}

	auto load_object(tree_fs_input& ar, objbase& obj) -> error {
	return error::eval_safe([&]() -> error {
		auto cur_head = head();
		if(!cur_head) return cur_head.error();
		// open node & close on exit
		prologue(*cur_head.value(), obj);
		auto finally = scope_guard{ [&]{ epilogue(*cur_head.value(), obj); } };

		// 1, read object format & obtain formatter filename
		std::string obj_frm;
		ar(cereal::make_nvp("fmt", obj_frm));
		auto F = get_formatter(obj.type_id(), obj_frm);
		if(!F) return { fmt::format("{} -> {}", obj.type_id(), obj_frm), Error::MissingFormatter };

		// 2. read object's metadata (objbase)
		(**cur_head)(cereal::make_nvp( "objbase", obj ));
		// if object is node and formatter don't store leafs, then load 'em explicitly
		if(!F->stores_node)
			ar(cereal::make_nvp( "node", obj.data_node() ));

		// 3. if object is pure node - we're done and can skip data processing
		if(obj.bs_resolve_type() == objnode::bs_type()) return perfect;

		// 4. read object data from specified file
		EVAL
			[&]{ return enter_root(); },
			[&]{ return objects_path_.empty() ?
				enter_dir(root_path_ / objects_dname_, objects_path_) : perfect;
			}
		RETURN_EVAL_ERR

		auto obj_path = objects_path_ / (obj.home_id() + '.' + obj_frm);
		auto abs_obj_path = fs::path{};
		SCOPE_EVAL_SAFE
			abs_obj_path = fs::absolute(obj_path);
		RETURN_SCOPE_ERR

		caf::anon_send(
			manager_, obj.shared_from_this(), obj_frm, abs_obj_path.u8string()
		);
		// defer wait until save completes
		if(!has_wait_deferred_) {
			ar(cereal::defer(cereal::Functor{ [](auto& ar){ ar.wait_objects_loaded(); } }));
			has_wait_deferred_ = true;
		}
		return perfect;
	}); }

	auto wait_objects_loaded(timespan how_long) -> std::vector<error> {
		auto res = fmanager_t::wait_jobs_done(manager_, how_long);
		has_wait_deferred_ = false;
		return res;
	}

	NodeLoad mode_;
	// async loaders manager
	bool has_wait_deferred_ = false;
};

///////////////////////////////////////////////////////////////////////////////
//  input archive
//
tree_fs_input::tree_fs_input(std::string root_fname, NodeLoad mode)
	: Base(this), pimpl_{ std::make_unique<impl>(std::move(root_fname), mode) }
{}

tree_fs_input::~tree_fs_input() = default;

auto tree_fs_input::head() -> result_or_err<cereal::JSONInputArchive*> {
	return pimpl_->head();
}

auto tree_fs_input::end_link() -> void {
	return pimpl_->end_link();
}

auto tree_fs_input::begin_node() -> error {
	return pimpl_->begin_node(*this);
}

auto tree_fs_input::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(*this, const_cast<tree::node&>(N));
}

auto tree_fs_input::load_object(objbase& obj) -> error {
	return pimpl_->load_object(*this, obj);
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
auto epilogue(tree_fs_input& ar, tree::link const&) -> void {
	ar.end_link();
}

auto prologue(tree_fs_input& ar, tree::node const&) -> void {
	ar.begin_node();
}

auto epilogue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
