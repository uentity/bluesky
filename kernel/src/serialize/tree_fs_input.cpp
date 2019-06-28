/// @file
/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree_fs_input.h>
#include <bs/serialize/serialize_decl.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>
#include <bs/tree/node.h>

#include <cereal/types/vector.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
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

	impl(std::string root_dirname, std::string data_fname) :
		root_dname_(std::move(root_dirname)), data_fname_(std::move(data_fname))
	{}

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
		else return { fmt::format("Cannot open file '{}' for writing", head_path.string()) };
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
				[&]{ return add_head(fs::path(root_path_) / data_fname_); }
			))
				return tl::make_unexpected(std::move(er));

			// read objects directory
			heads_.back()( cereal::make_nvp("objects_dir", objects_dname_) );
		}
		return &heads_.back();
	}

	//auto begin_link(const tree::sp_link& L) -> error {
	//	if(auto er = enter_root()) return er;
	//	if(cur_path_ == root_path_) return perfect;

	//	return add_head(cur_path_ / to_string(L->id()));
	//}

	//auto end_link() -> error {
	//	if(heads_.size() == 1) return error::quiet("No link file started");
	//	pop_head();
	//	return perfect;
	//}

	auto begin_node(const tree::node& N) -> error {
		// read node's relative path
		std::string node_dir;
		return error::eval(
			// read node's relative path
			[&]{ return head().map( [&](auto* ar){
				// read node's directory
				(*ar)(cereal::make_nvp("node_dir", node_dir));
				// read leafs order
				std::vector<std::string> leafs_order;
				(*ar)(cereal::make_nvp("leafs_order", leafs_order));
			}); },
			[&]{ return enter_root(); },
			[&]{ return enter_dir(root_path_ / node_dir, cur_path_); }
		);
	}

	auto end_node(tree_fs_input& ar, tree::node& N) -> error {
		if(cur_path_.empty() || cur_path_ == root_path_) return {"No node saving were started"};

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
		// correct owner of all loaded links
		N.propagate_owner();

		auto er = enter_dir(cur_path_.parent_path(), cur_path_);
		return united_err_msg.empty() ?
			(er ? er : perfect) :
			(er ?
				error{fmt::format("{} | {}", er.what(), std::move(united_err_msg)), er.code} :
				std::move(united_err_msg)
			);
	}

	auto install_object_loader(std::string obj_type_id, std::string fmt_descr, object_loader_fn f) -> bool {
		obj_loaders_[std::move(obj_type_id)] = std::pair{ std::move(fmt_descr), std::move(f) };
		return true;
	}

	auto can_load_object(std::string_view obj_type_id) -> bool {
		return obj_loaders_.find(obj_type_id) != obj_loaders_.end();
	}

	auto load_object(tree_fs_input& ar, objbase& obj) -> error {
		// write down object format and filename from archive
		std::string obj_fmt, obj_filename;
		ar(cereal::make_nvp("fmt", obj_fmt), cereal::make_nvp("filename", obj_filename));
		if(obj_fmt.size() + obj_filename.size() == 0)
			return { fmt::format("Cannot load object of type {} - missing data") };

		auto S = obj_loaders_.find(obj.type_id());
		if(S == obj_loaders_.end() || obj_fmt != S->second.first)
			return { fmt::format("Cannot load object of type {} - missing loader", obj.type_id()) };

		if(auto er = enter_root()) return er;
		if(objects_path_.empty())
			if(auto er = enter_dir(root_path_ / objects_dname_, objects_path_)) return er;

		auto obj_path = objects_path_ / obj_filename;
		auto objf = std::ifstream{obj_path, std::ios::in | std::ios::binary};

		if(objf) S->second.second(objf, obj);
		else return { fmt::format("Cannot open file '{}' for reading", obj_path) };

		return perfect;
	}

	std::string root_dname_, data_fname_, objects_dname_;
	std::error_code file_er_;
	fs::path root_path_, cur_path_, objects_path_;

	std::list<std::ifstream> necks_;
	std::list<cereal::JSONInputArchive> heads_;

	using loader_descr = std::pair<std::string, object_loader_fn>;
	std::map<std::string, loader_descr, std::less<>> obj_loaders_;
};

///////////////////////////////////////////////////////////////////////////////
//  input archive
//
tree_fs_input::tree_fs_input(
	std::string root_path, std::string data_fname, std::string objects_dir
)
	: Base(this), pimpl_{std::make_unique<impl>(
		std::move(root_path), std::move(data_fname)
	)}
{}

tree_fs_input::~tree_fs_input() = default;

auto tree_fs_input::head() -> result_or_err<cereal::JSONInputArchive*> {
	return pimpl_->head();
}

//auto tree_fs_input::begin_link(const tree::sp_link& L) -> error {
//	return pimpl_->begin_link(L);
//}
//
//auto tree_fs_input::end_link() -> error {
//	return pimpl_->end_link();
//}

auto tree_fs_input::begin_node(const tree::node& N) -> error {
	return pimpl_->begin_node(N);
}

auto tree_fs_input::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(*this, const_cast<tree::node&>(N));
}

auto tree_fs_input::load_object(objbase& obj) -> error {
	return pimpl_->load_object(*this, obj);
}

auto tree_fs_input::install_object_loader(
	std::string obj_type_id, std::string fmt_descr, object_loader_fn f
) -> bool {
	return pimpl_->install_object_loader(std::move(obj_type_id), std::move(fmt_descr), std::move(f));
}

auto tree_fs_input::can_load_object(std::string_view obj_type_id) const -> bool {
	return pimpl_->can_load_object(obj_type_id);
}

auto tree_fs_input::loadBinaryValue(void* data, size_t size, const char* name) -> void {
	head().map([=](auto* jar) {
		jar->loadBinaryValue(data, size, name);
	});
}

///////////////////////////////////////////////////////////////////////////////
//  prologue, epilogue
//

//auto prologue(tree_fs_input& ar, tree::sp_link const& L) -> void {
//	ar.begin_link(L);
//}
//
//auto epilogue(tree_fs_input& ar, tree::sp_link const&) -> void {
//	ar.end_link();
//}

auto prologue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.begin_node(N);
}

auto epilogue(tree_fs_input& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
