/// @file
/// @author uentity
/// @date 19.02.2020
/// @brief Tree FS impl details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/errors.h>
#include <bs/serialize/serialize_decl.h>

#include <cereal/archives/json.hpp>

#include <filesystem>
#include <fstream>
#include <list>

NAMESPACE_BEGIN(blue_sky::detail)
namespace fs = std::filesystem;

template<bool Saving>
struct file_heads_manager {
	using Error = tree::Error;

	// setup traits depending on save/load mode
	template<bool Saving_, typename = void>
	struct trait {
		using neck_t = std::ofstream;
		using head_t = cereal::JSONOutputArchive;

		static constexpr auto neck_mode = std::ios::out | std::ios::trunc;
	};

	template<typename _>
	struct trait<false, _> {
		using neck_t = std::ifstream;
		using head_t = cereal::JSONInputArchive;

		static constexpr auto neck_mode = std::ios::in;
	};

	using trait_t = trait<Saving>;
	using neck_t = typename trait_t::neck_t;
	using head_t = typename trait_t::head_t;
	static constexpr auto neck_mode = trait_t::neck_mode;


	// ctor
	file_heads_manager(std::string root_fname, std::string objects_dirname = {})
		: root_fname_(std::move(root_fname)), objects_dname_(std::move(objects_dirname))
	{
		// for Windows add '\\?\' prefix for long names support
#ifdef _WIN32
		static constexpr auto magic_prefix = std::string_view{ "\\\\?\\" };
		if(magic_prefix.compare(0, magic_prefix.size(), root_fname_.data()) != 0)
			root_fname_.insert(0, magic_prefix);
#endif
		// try convert root filename to absolute
		auto root_path = fs::path(root_fname_);
		auto abs_root = fs::path{};
		auto er = error::eval_safe([&]{ abs_root = fs::absolute(root_path); });
		if(!er) {
			// extract root dir from absolute filename
			root_dname_ = abs_root.parent_path().u8string();
			root_fname_ = abs_root.filename().u8string();
		}
		else if(root_path.has_parent_path()) {
			// could not make abs path
			root_dname_ = root_path.parent_path().u8string();
			root_fname_ = root_path.filename().u8string();
		}
		else {
			// best we can do
			error::eval_safe([&]{ root_dname_ = fs::current_path().u8string(); });
		}
	}

	// if entering `src_path` is successfull, set `tar_path` to src_path
	// if `tar_path` already equals `src_path` return success
	template<typename Path>
	auto enter_dir(Path src_path, fs::path& tar_path) -> error {
		auto path = fs::path(std::move(src_path));
		//if(path == tar_path) return perfect;
		if(path.empty()) return { path.u8string(), Error::EmptyPath };

		EVAL_SAFE
			// do something when path doesn't exist
			[&] {
				if(!fs::exists(path)) {
					if constexpr(Saving)
						fs::create_directories(path);
					else
						return error{ path.u8string(), Error::PathNotExists };
				}
				return success();
			},
			// check that path is a directory
			[&] {
				return fs::is_directory(path) ?
					success() : error{ path.u8string(), Error::PathNotDirectory };
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
		// [NOTE] explicit capture `head_path` because VS doesn't capture it with simple '&'
		// because of constexpr if?
		[this, &head_path] {
			// [NOTE] don't enter parent dir (and reset `cur_path_`) when loading because:
			// 1. we already antered parent dir on all usage conditions
			// 2. resetting `cur_path_` is an error, because ther's an optimization for not creating
			// empty dirs for empty nodes
			if constexpr(Saving)
				return enter_dir(head_path.parent_path(), cur_path_);
		},
		// open head file
		[&] {
			if(auto neck = neck_t(head_path, neck_mode)) {
				necks_.push_back(std::move(neck));
				heads_.emplace_back(necks_.back());
				return success();
			}
			return error{ head_path.u8string(), Saving ? Error::CantWriteFile : Error::CantReadFile };
		}
	); }

	auto pop_head() -> void {
		if(!heads_.empty()) {
			heads_.pop_back();
			necks_.pop_back();
		}
	}

	auto head() -> result_or_err<head_t*> {
		if(heads_.empty()) {
			if(auto er = error::eval_safe(
				[&] { return enter_root(); },
				[&] { return add_head(fs::path(root_path_) / root_fname_); },
				[&] { // read/write objects directory
					heads_.back()( cereal::make_nvp("objects_dir", objects_dname_) );
				}
			)) return tl::make_unexpected(std::move(er));
		}
		return &heads_.back();
	}

	std::string root_fname_, objects_dname_, root_dname_;
	fs::path root_path_, cur_path_, objects_path_;

	std::list<neck_t> necks_;
	std::list<head_t> heads_;
};

NAMESPACE_END(blue_sky::detail)
