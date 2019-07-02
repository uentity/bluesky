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

#include <cereal/types/vector.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <caf/all.hpp>

#include <filesystem>
#include <fstream>
#include <list>

namespace fs = std::filesystem;

NAMESPACE_BEGIN(blue_sky)
///////////////////////////////////////////////////////////////////////////////
//  tree_fs_input::impl
//
struct tree_fs_input::impl {

	impl(std::string root_fname) :
		root_fname_(std::move(root_fname))
	{
		// try convert root filename to absolute
		auto root_path = fs::path(root_fname_);
		auto abs_root = fs::absolute(root_path, file_er_);
		if(!file_er_) {
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
			root_dname_ = fs::current_path(file_er_).string();
		}
	}

	auto make_error(error&& cust_er = perfect) -> error {
		return file_er_ ?
			error{cust_er ? cust_er.what() : "", file_er_} :
			std::move(cust_er);
	};

	// if entering `src_path` is successfull, set `tar_path` to src_path
	// if `tar_path` is nonempty, return success immediately
	template<typename Path>
	auto enter_dir(Path src_path, fs::path& tar_path) -> error {
		auto path = fs::path(std::move(src_path));
		if(path.empty()) return error{"Cannot load tree from empty path"};

		// check that path exists
		file_er_.clear();
		if(!fs::exists(path, file_er_))
			return make_error("Specified directory does not exist");

		// check that path is a directory
		if(!fs::is_directory(path, file_er_))
			return make_error("Tree path is not a directory");

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
		if(auto neck = std::ifstream(head_path, std::ios::in)) {
			necks_.emplace_back(std::move(neck));
			heads_.emplace_back(necks_.back());
			return perfect;
		}
		else return { fmt::format("Cannot open file '{}' for reading", head_path.string()) };
	}

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
				[&]{ return add_head(fs::path(root_path_) / root_fname_); }
			))
				return tl::make_unexpected(std::move(er));

			// read objects directory
			heads_.back()( cereal::make_nvp("objects_dir", objects_dname_) );
		}
		return &heads_.back();
	}

	auto load_node(tree_fs_input& ar, tree::node& N, const std::vector<std::string>& leafs_order) -> error {
		// loaded node in most cases will be empty (leafs are serialized to individual files)
		// fill leafs by scanning directory and loading link files
		file_er_.clear();
		using Options = fs::directory_options;
		auto Niter = fs::directory_iterator(cur_path_, Options::skip_permission_denied, file_er_);
		if(file_er_) return make_error();

		std::string united_err_msg;
		auto dump_error = [&](auto& ex) {
			if(!united_err_msg.empty()) united_err_msg += " | ";
			united_err_msg += ex.what();
		};
		for(auto& f : Niter) {
			// skip directories
			if(fs::is_directory(f, file_er_)) continue;

			// try load file as a link
			try {
				if(auto er = add_head(f)) {
					dump_error(er);
					continue;
				}
				auto finally = detail::scope_guard{[this]{ pop_head(); }};

				tree::sp_link L;
				ar(L);
				N.insert(std::move(L));
			}
			catch(cereal::Exception& ex) {
				dump_error(ex);
			}
		}

		if(united_err_msg.empty()) return perfect;
		else return united_err_msg;
	}

	auto begin_node(tree_fs_input& ar, const tree::node& N) -> error {
		return error::eval(
			[&]{ return head().map( [&](auto* ar) { prologue(*ar, N); }); },
			[&]{ return enter_root(); }
		);
	}

	auto end_node(tree_fs_input& ar, tree::node& N) -> error {
		if(cur_path_.empty()) return {"No node loading were started"};

		std::string node_dir;
		std::vector<std::string> leafs_order;
		return error::eval(
			// read node's metadata
			[&]{ return head().map( [&](auto* ar){
				(*ar)(cereal::make_nvp("node_dir", node_dir));
				(*ar)(cereal::make_nvp("leafs_order", leafs_order));
				// we finished reading node
				epilogue(*ar, N);
			}); },
			// enter node's directory
			[&]{ return enter_dir(cur_path_ / node_dir, cur_path_); },
			// load leafs
			[&]{ return load_node(ar, N, leafs_order); },
			// enter parent dir
			[&] { return enter_dir(cur_path_.parent_path(), cur_path_); }
		);
	}

	auto load_object(tree_fs_input& ar, objbase& obj) -> error {
		auto cur_head = head();
		if(!cur_head) return cur_head.error();
		// open node & close on exit
		prologue(*cur_head.value(), obj);
		auto finally = scope_guard{ [&]{ epilogue(*cur_head.value(), obj); } };

		// read object format & filename
		std::string obj_filename;
		ar(cereal::make_nvp("fmt", obj_frm_), cereal::make_nvp("filename", obj_filename));

		// obtain formatter
		auto F = get_formatter(obj.type_id(), obj_frm_);
		if(!F) return { fmt::format(
			"Cannot load '{}' - missing object formatter '{}'", obj.type_id(), obj_frm_
		) };

		// if object is node and formatter don't store leafs, then load 'em explicitly
		if(obj.is_node() && !F->stores_node)
			ar(static_cast<tree::node&>(obj));

		// read object data from specified file
		if(auto er = enter_root()) return er;
		if(objects_path_.empty())
			if(auto er = enter_dir(root_path_ / objects_dname_, objects_path_)) return er;

		auto obj_path = objects_path_ / obj_filename;
		auto objf = std::ifstream{obj_path, std::ios::in | std::ios::binary};

		if(objf) return F->second(obj, objf, obj_frm_);
		else return { fmt::format(
			"Cannot load '{}' - unable to open file '{}' for reading", obj.type_id(), obj_path
		) };
	}

	std::string root_fname_, root_dname_, objects_dname_, obj_frm_;
	std::error_code file_er_;
	fs::path root_path_, cur_path_, objects_path_;

	std::list<std::ifstream> necks_;
	std::list<cereal::JSONInputArchive> heads_;
};

///////////////////////////////////////////////////////////////////////////////
//  input archive
//
tree_fs_input::tree_fs_input(std::string root_fname)
	: Base(this), pimpl_{ std::make_unique<impl>(std::move(root_fname)) }
{}

tree_fs_input::~tree_fs_input() = default;

auto tree_fs_input::head() -> result_or_err<cereal::JSONInputArchive*> {
	return pimpl_->head();
}

auto tree_fs_input::begin_node(const tree::node& N) -> error {
	return pimpl_->begin_node(*this, N);
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

auto tree_fs_input::will_serialize_node(objbase const* obj) const -> bool {
	if(auto pfmt = get_formatter(obj->bs_type().name, pimpl_->obj_frm_); obj->is_node() && pfmt)
		return pfmt->stores_node;
	return true;
}

///////////////////////////////////////////////////////////////////////////////
//  prologue, epilogue
//
auto prologue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.begin_node(N);
}

auto epilogue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
