/// @file
/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree_fs_input.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/serialize_decl.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>
#include <bs/tree/node.h>

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>

#include <cereal/types/vector.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <caf/all.hpp>

#include <filesystem>
#include <fstream>
#include <list>
#include <algorithm>
//#include <execution>

NAMESPACE_BEGIN(blue_sky)
namespace fs = std::filesystem;
using NodeLoad = tree_fs_input::NodeLoad;

const auto uuid_from_str = boost::uuids::string_generator{};

///////////////////////////////////////////////////////////////////////////////
//  tree_fs_input::impl
//
struct tree_fs_input::impl {

	impl(std::string root_fname, NodeLoad mode) :
		mode_(mode), root_fname_(std::move(root_fname))
	{
		// try convert root filename to absolute
		auto root_path = fs::path(root_fname_);
		auto abs_root = fs::path{};
		auto er = error::eval_safe([&]{ abs_root = fs::absolute(root_path); });
		if(!er) {
			// extract root dir from absolute filename
			root_dname_ = abs_root.parent_path().string();
			root_fname_ = abs_root.filename().string();
		}
		else if(root_path.has_parent_path()) {
			// could not make abs path
			root_dname_ = root_path.parent_path().string();
			root_fname_ = root_path.filename().string();
		}
		else {
			// best we can do
			error::eval_safe([&]{ root_dname_ = fs::current_path().string(); });
		}
	}

	// if entering `src_path` is successfull, set `tar_path` to src_path
	// if `tar_path` is nonempty, return success immediately
	template<typename Path>
	auto enter_dir(Path src_path, fs::path& tar_path) -> error {
		auto path = fs::path(std::move(src_path));
		//if(path == tar_path) return perfect;
		if(path.empty()) return error{"Cannot load tree from empty path"};

		EVAL_SAFE
			// check that path exists
			[&]{
				return fs::exists(path) ?
					perfect : error{ fmt::format("Can't enter {}: does not exist", path) };
			},
			// check that path is a directory
			[&]{
				return fs::is_directory(path) ?
					perfect : error{ fmt::format("Can't enter {}: not a directory", path) };
			}
		RETURN_EVAL_ERR

		tar_path = std::move(path);
		return perfect;
	}

	auto enter_root() -> error {
		if(root_path_.empty())
			if(auto er = enter_dir(root_dname_, root_path_)) return er;
		if(cur_path_.empty()) cur_path_ = root_path_;
		return perfect;
	}

	auto add_head(const fs::path& head_path) -> error {
	return error::eval_safe(
		// enter parent dir
		// [NOTE] - disabled, because in all usage conditions we already entered parent dir
		//[&] { return enter_dir(head_path.parent_path(), cur_path_); },
		// open head file
		[&] {
			if(auto neck = std::ifstream(head_path, std::ios::in)) {
				necks_.push_back(std::move(neck));
				heads_.emplace_back(necks_.back());
				return success();
			}
			return error{ fmt::format("Can't open file '{}' for reading", head_path) };
		}
	); }

	auto pop_head() -> void {
		if(!heads_.empty()) {
			heads_.pop_back();
			necks_.pop_back();
		}
	}

	auto head() -> result_or_err<cereal::JSONInputArchive*> {
		if(heads_.empty()) {
			if(auto er = error::eval(
				[&]{ return enter_root(); },
				[&]{ return add_head(root_path_ / root_fname_); },
				[&] { // read objects directory
					heads_.back()( cereal::make_nvp("objects_dir", objects_dname_) );
				}
			)) return tl::make_unexpected(std::move(er));
		}
		return &heads_.back();
	}

	auto begin_node(tree_fs_input& ar) -> error {
		// node reference is ONLY used for template matching
		static constexpr tree::node* sentinel = nullptr;
		return error::eval_safe(
			[&]{ return head().map( [&](auto* ar) { prologue(*ar, *sentinel); }); },
			[&]{ return enter_root(); }
		);
	}

	auto end_node(tree_fs_input& ar, tree::node& N) -> error {
		if(cur_path_.empty()) return {"Node loading wasn't started"};

		// always return to parent dir after node is loaded
		auto finally = scope_guard{[=, p = cur_path_] {
			if(auto er = enter_dir(p, cur_path_)) throw er;
		}};

		std::string node_dir;
		std::vector<std::string> leafs_order;
		return error::eval_safe(
			// read node's metadata
			[&]{ return head().map( [&](auto* ar) {
				(*ar)(cereal::make_nvp("node_dir", node_dir));
				(*ar)(cereal::make_nvp("leafs_order", leafs_order));
				// we finished reading node
				epilogue(*ar, N);
			}); },
			// load leafs
			[&]{ return load_node(ar, N, std::move(node_dir), std::move(leafs_order)); }
		);
	}

	auto load_node(
		tree_fs_input& ar, tree::node& N, std::string node_dir, std::vector<std::string> leafs_order
	) -> error {
		using Options = fs::directory_options;

		// skip empty dirs in normal mode
		if(mode_ == NodeLoad::Normal && leafs_order.empty()) return perfect;
		// enter node's dir
		if(auto er = enter_dir(cur_path_ / node_dir, cur_path_)) return er;

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
					[&] {
						auto finally = detail::scope_guard{[this]{ pop_head(); }};
						tree::sp_link L;
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
					[&] {
						auto finally = detail::scope_guard{[this]{ pop_head(); }};

						tree::sp_link L;
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
			//				auto finally = detail::scope_guard{[this]{ pop_head(); }};

			//				tree::sp_link L;
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
			auto wanted_order = std::vector<link::id_type>(leafs_order.size());
			std::transform(
				leafs_order.cbegin(), leafs_order.cend(), wanted_order.begin(),
				[](const auto& s_uid) { return uuid_from_str(s_uid); }
			);

			// extract current order of link IDs
			auto res_order = std::vector<link::id_type>();
			res_order.reserve(N.size());
			for(auto i = N.begin(), end = N.end(); i != end; ++i)
				res_order.push_back((*i)->id());

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
		auto cur_head = head();
		if(!cur_head) return cur_head.error();
		// open node & close on exit
		prologue(*cur_head.value(), obj);
		auto finally = scope_guard{ [&]{ epilogue(*cur_head.value(), obj); } };

		// read object format & filename
		std::string obj_filename, obj_frm;
		ar(cereal::make_nvp("fmt", obj_frm), cereal::make_nvp("filename", obj_filename));

		// obtain formatter
		auto F = get_formatter(obj.type_id(), obj_frm);
		if(!F) return { fmt::format(
			"Cannot load '{}' - missing formatter '{}'", obj.type_id(), obj_frm
		) };

		// if object is node and formatter don't store leafs, then load 'em explicitly
		if(obj.is_node() && !F->stores_node)
			ar(cereal::make_nvp( "node", static_cast<tree::node&>(obj) ));

		// read object data from specified file
		EVAL
			[&]{ return enter_root(); },
			[&]{ return objects_path_.empty() ?
				enter_dir(root_path_ / objects_dname_, objects_path_) : perfect;
			}
		RETURN_EVAL_ERR

		auto obj_path = objects_path_ / obj_filename;
		auto abs_obj_path = fs::path{};
		SCOPE_EVAL_SAFE
			abs_obj_path = fs::absolute(obj_path);
		RETURN_SCOPE_ERR
		return F->load(obj, abs_obj_path.string(), obj_frm);
	}

	NodeLoad mode_;
	std::string root_fname_, root_dname_, objects_dname_;
	fs::path root_path_, cur_path_, objects_path_;

	std::list<std::ifstream> necks_;
	std::list<cereal::JSONInputArchive> heads_;
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

auto tree_fs_input::begin_node() -> error {
	return pimpl_->begin_node(*this);
}

auto tree_fs_input::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(*this, const_cast<tree::node&>(N));
}

auto tree_fs_input::load_object(objbase& obj) -> error {
	return pimpl_->load_object(*this, obj);
}

auto tree_fs_input::loadBinaryValue(void* data, size_t size, const char* name) -> void {
	head().map([=](auto* jar) {
		jar->loadBinaryValue(data, size, name);
	});
}

///////////////////////////////////////////////////////////////////////////////
//  prologue, epilogue
//
auto prologue(tree_fs_input& ar, tree::node const&) -> void {
	ar.begin_node();
}

auto epilogue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

auto prologue(
	tree_fs_input& ar, cereal::memory_detail::LoadAndConstructLoadWrapper<tree_fs_input, tree::node> const&
) -> void {
	ar.begin_node();
}

auto epilogue(
	tree_fs_input& ar, cereal::memory_detail::LoadAndConstructLoadWrapper<tree_fs_input, tree::node> const& N
) -> void {
	ar.end_node( *const_cast<cereal::construct<tree::node>&>(N.construct).ptr() );
}

NAMESPACE_END(blue_sky)
