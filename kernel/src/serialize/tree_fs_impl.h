/// @file
/// @author uentity
/// @date 19.02.2020
/// @brief Tree FS impl details
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/actor_common.h>
#include <bs/timetypes.h>
#include <bs/kernel/radio.h>
#include <bs/tree/errors.h>
#include <bs/tree/link.h>
#include <bs/detail/str_utils.h>
#include <bs/serialize/serialize_decl.h>

#include "../tree/link_impl.h"

#include <caf/typed_event_based_actor.hpp>

#include <cereal/archives/json.hpp>

#include <filesystem>
#include <fstream>
#include <list>

NAMESPACE_BEGIN(blue_sky::detail)
namespace fs = std::filesystem;

/*-----------------------------------------------------------------------------
 *  handle objects save/load jobs
 *-----------------------------------------------------------------------------*/
using objfrm_manager_t = caf::typed_actor<
	// launch object formatting job
	caf::reacts_to<
		sp_obj /*what*/, std::string /*formatter name*/, std::string /*filename*/
	>,
	// end formatting session
	caf::reacts_to<a_bye>,
	// return collected job errors
	caf::replies_to< a_ack >::with< std::vector<error::box> >
>;

// [NOTE] manager is valid only for one save/load session!
// on next session just spawn new manager
struct BS_HIDDEN_API objfrm_manager : objfrm_manager_t::base {
	using actor_type = objfrm_manager_t;

	objfrm_manager(caf::actor_config& cfg, bool is_saving);

	auto make_behavior() -> behavior_type override;

	static auto wait_jobs_done(objfrm_manager_t self, timespan how_long) -> std::vector<error>;

private:
	// deliver session results back to requester
	auto session_ack() -> void;

	const bool is_saving_;
	bool session_finished_ = false;

	// errors collection
	std::vector<error::box> er_stack_;
	caf::response_promise boxed_errs_;
	// track running savers
	size_t nstarted_ = 0, nfinished_ = 0;
};

/*-----------------------------------------------------------------------------
 *  Manage link file streams and dirs during tree save/load
 *-----------------------------------------------------------------------------*/
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
	// [NOTE] assume that paths come in UTF-8
	file_heads_manager(const std::string& root_fname, const std::string& objects_dirname = {})
		: root_fname_(ustr2str(root_fname)), objects_dname_(ustr2str(objects_dirname))
	{
		// [NOTE] `root_fname_`, `root_dname_` & `objects_dname_` will be converted to native encoding
#ifdef _WIN32
		// for Windows add '\\?\' prefix for long names support
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
			// [TODO] throw error here
			error::eval_safe([&]{ root_dname_ = fs::current_path().string(); });
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
			// 1. we already antered parent dir in all usage conditions
			// 2. resetting `cur_path_` is an error, because there's an optimization for not creating
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
				[&] { // read/write objects directory encoded in UTF-8
					if constexpr(Saving)
						heads_.back()( cereal::make_nvp("objects_dir", str2ustr(objects_dname_)) );
					else {
						// read in UTF-8 & converto to native
						auto objects_dname = std::string{};
						heads_.back()( cereal::make_nvp("objects_dir", objects_dname) );
						objects_dname_ = ustr2str(objects_dname);
					}
				}
			)) return tl::make_unexpected(std::move(er));
			// start new formatters manager
			manager_ = kernel::radio::system().spawn<objfrm_manager>(Saving);
		}
		return &heads_.back();
	}

	auto end_link(const tree::link& L) -> error {
		pop_head();
		// tell manager that session finished when very first head (root_fname_) is popped
		if(heads_.empty())
			caf::anon_send(manager_, a_bye());
		// setup link to trigger dealyed object load
		if constexpr(!Saving) {
			if(auto r = tree::link_impl::actorf<bool>(L, a_lazy(), a_load()); !r)
				return std::move(r.error());
		}
		return perfect;
	}

	std::string root_fname_, objects_dname_, root_dname_;
	fs::path root_path_, cur_path_, objects_path_;

	objfrm_manager_t manager_;

	std::list<neck_t> necks_;
	std::list<head_t> heads_;
};

NAMESPACE_END(blue_sky::detail)
