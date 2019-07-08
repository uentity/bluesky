/// @file
/// @author uentity
/// @date 29.05.2019
/// @brief Tree filesystem archive implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/serialize/tree_fs_output.h>
#include <bs/serialize/object_formatter.h>
#include <bs/serialize/serialize_decl.h>
#include <bs/serialize/base_types.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <bs/tree/node.h>
#include <bs/log.h>
#include <bs/kernel/config.h>

#include <cereal/types/vector.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <fmt/format.h>
#include <fmt/ostream.h>

#include <caf/all.hpp>

#include <filesystem>
#include <fstream>
#include <list>

namespace fs = std::filesystem;

CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::sp_cobj)
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::error::box)

template<typename T> struct TD;

NAMESPACE_BEGIN(blue_sky)

///////////////////////////////////////////////////////////////////////////////
//  tree_fs_output::impl
//
struct tree_fs_output::impl {

	impl(std::string root_fname, std::string objects_dirname) :
		root_fname_(std::move(root_fname)), objects_dname_(std::move(objects_dirname)),
		manager_(kernel::config::actor_system().spawn<savers_manager>())
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
		if(path.empty()) return error{"Cannot save tree to empty path"};

		// create folders along specified path if not created yet
		file_er_.clear();
		if(!fs::exists(path, file_er_) && !file_er_)
			fs::create_directories(path, file_er_);
		if(file_er_) return make_error();

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
		auto flags = std::ios::out | std::ios::trunc;
		if(auto neck = std::ofstream(head_path, flags)) {
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

	auto head() -> result_or_err<cereal::JSONOutputArchive*> {
		if(heads_.empty()) {
			if(auto er = enter_root()) return tl::make_unexpected(std::move(er));
			if(auto er = add_head(fs::path(root_path_) / root_fname_))
				return tl::make_unexpected(std::move(er));
			else {
				// write objects directory
				heads_.back()( cereal::make_nvp("objects_dir", objects_dname_) );
			}
		}
		return &heads_.back();
	}

	auto begin_link(const tree::sp_link& L) -> error {
		if(auto er = enter_root()) return er;
		if(cur_path_ == root_path_) return perfect;

		return add_head(cur_path_ / to_string(L->id()));
	}

	auto end_link() -> error {
		if(heads_.size() == 1) return error::quiet("No link file started");
		pop_head();
		return perfect;
	}

	auto begin_node(const tree::node& N) -> error {
		return error::eval(
			[&]{ return head().map( [&](auto* ar) { prologue(*ar, N); }); },
			[&]{ return enter_root(); },
			[&]{ return enter_dir(cur_path_ / N.id(), cur_path_); }
		);
	}

	auto end_node(const tree::node& N) -> error {
		if(cur_path_.empty() || cur_path_ == root_path_) return {"No node saving were started"};
		return error::eval(
			// write down node's metadata nessessary to load it later
			[&]{ return head().map( [&](auto* ar) {
				// node directory
				(*ar)(cereal::make_nvp("node_dir", N.id()));

				// cusstom leafs order
				std::vector<std::string> leafs_order;
				leafs_order.reserve(N.size());
				for(const auto& L : N)
					leafs_order.emplace_back(to_string(L->id()));
				(*ar)(cereal::make_nvp("leafs_order", leafs_order));
				// and finish
				epilogue(*ar, N);
			}); },
			// enter parent dir
			[&] { return enter_dir(cur_path_.parent_path(), cur_path_); }
		);
	}

	auto save_object(tree_fs_output& ar, const objbase& obj) -> error {
		// open node
		auto cur_head = head();
		if(!cur_head) return cur_head.error();
		prologue(*cur_head.value(), obj);

		std::string obj_fmt, obj_filename;
		bool fmt_ok = false, filename_ok = false;

		auto finally = scope_guard{ [&]{
			// if error happened we still need to write values
			if(!fmt_ok) ar(cereal::make_nvp("fmt", obj_fmt));
			if(!filename_ok) cereal::make_nvp("filename", obj_filename);
			// ... and close node
			epilogue(*cur_head.value(), obj);
		} };

		// obtain formatter
		auto F = get_active_formatter(obj.type_id());
		if(!F) {
			// output error to format
			obj_fmt = fmt::format("Cannot save '{}' - no formatters installed", obj.type_id());
			return { obj_fmt };
		}
		obj_fmt = F->name;
		// write down object formatter name
		ar(cereal::make_nvp("fmt", obj_fmt));
		fmt_ok = true;

		if(auto er = enter_root()) return er;
		if(objects_path_.empty())
			if(auto er = enter_dir(root_path_ / objects_dname_, objects_path_)) return er;

		auto obj_path = objects_path_ / obj.id();
		obj_path += std::string(".") + obj_fmt;
		obj_filename = obj_path.filename().string();
		// write down object filename
		ar(cereal::make_nvp("filename", obj_filename));
		filename_ok = true;

		// if object is node and formatter don't store leafs, then save 'em explicitly
		if(obj.is_node() && !F->stores_node)
			ar(static_cast<const tree::node&>(obj));

		// and actually save object data to file
		auto abs_obj_path = fs::absolute(obj_path, file_er_);
		if(file_er_) return make_error();

		// spawn object save actor
		auto A = kernel::config::actor_system().spawn(async_saver, F, manager_);
		auto pm = caf::actor_cast<savers_manager_ptr>(manager_);
		// defer wait until save completes
		if(!has_wait_deferred_) {
			ar(cereal::defer(cereal::Functor{ [](auto& ar){ ar.wait_objects_saved(); } }));
			has_wait_deferred_ = true;
		}
		// inc started counter
		++pm->state.nstarted_;
		// post invoke save mesage
		pm->send(A, obj.shared_from_this(), abs_obj_path.string());
		return perfect;
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
	//  async saver API
	//
	using saver_actor_t = caf::typed_actor<
		caf::reacts_to< sp_cobj, std::string > // invoke save
	>;

	using savers_manager_t = caf::typed_actor<
		caf::reacts_to< error::box > // store error from saver and increment finished counter
	>;

	// saver
	static auto async_saver(
		saver_actor_t::pointer self, object_formatter* F, savers_manager_t manager
	) -> saver_actor_t::behavior_type {
		return {
			[=](const sp_cobj& obj, const std::string& fname) {
				self->send( manager, F->save(*obj, fname, F->name).pack() );
			}
		};
	}

	// savers manager
	struct manager_state {
		// errors collection
		std::vector<error::box> er_stack_;
		std::mutex er_sync_;
		// track running savers
		size_t nstarted_ = 0;
		std::atomic<size_t> nfinished_ = 0;
		std::mutex running_mtx_;
		std::condition_variable running_cv_;
	};

	using savers_manager_ptr = typename savers_manager_t::stateful_pointer<manager_state>;

	struct savers_manager : savers_manager_t::stateful_base<manager_state> {

		savers_manager(caf::actor_config& cfg) : savers_manager_t::stateful_base<manager_state>(cfg) {}

		behavior_type make_behavior() override {
			return {
				[this](error::box er) {
					{
						auto solo = std::lock_guard{ state.er_sync_ };
						state.er_stack_.push_back(std::move(er));
					}
					// dec counter
					finished();
				},
			};
		}

		void finished() {
			std::unique_lock guard{ state.running_mtx_ };
			if(++state.nfinished_ == state.nstarted_)
				state.running_cv_.notify_all();
		}
	};

	auto wait_objects_saved(timespan how_long) -> std::vector<error> {
		auto pm = caf::actor_cast<savers_manager_ptr>(manager_);
		auto& S = pm->state;
		//auto res = std::move(S.er_stack_);
		// unpack error boxes -> result vector of errors
		auto res = std::vector<error>{};
		for(auto& er_box : S.er_stack_)
			res.emplace_back( error::unpack(std::move(er_box)) );

		// reset state on exit
		auto finally = scope_guard{ [&S, this]{
			S.nstarted_ = 0;
			S.nfinished_ = 0;
			S.er_stack_.clear();
			has_wait_deferred_ = false;
		}};

		std::unique_lock guard{ S.running_mtx_ };
		if(!S.running_cv_.wait_for( guard, how_long, [&S]{ return S.nfinished_ == S.nstarted_; }))
			res.emplace_back("Timeout waiting for Tree FS save to complete");
		return res;
	}

	///////////////////////////////////////////////////////////////////////////////
	//  data
	//
	std::string root_fname_, objects_dname_, root_dname_;
	std::error_code file_er_;
	fs::path root_path_, cur_path_, objects_path_;

	std::list<std::ofstream> necks_;
	std::list<cereal::JSONOutputArchive> heads_;

	// obj_type_id -> formatter name
	using active_fmt_t = std::map<std::string_view, std::string, std::less<>>;
	active_fmt_t active_fmt_;

	// async savers manager
	savers_manager_t manager_;
	bool has_wait_deferred_ = false;
};

///////////////////////////////////////////////////////////////////////////////
//  output archive
//
tree_fs_output::tree_fs_output(
	std::string root_fname, std::string objects_dir
)
	: Base(this), pimpl_{ std::make_unique<impl>(std::move(root_fname), std::move(objects_dir)) }
{}

tree_fs_output::~tree_fs_output() = default;

auto tree_fs_output::head() -> result_or_err<cereal::JSONOutputArchive*> {
	return pimpl_->head();
}

auto tree_fs_output::begin_link(const tree::sp_link& L) -> error {
	return pimpl_->begin_link(L);
}

auto tree_fs_output::end_link() -> error {
	return pimpl_->end_link();
}

auto tree_fs_output::begin_node(const tree::node& N) -> error {
	return pimpl_->begin_node(N);
}

auto tree_fs_output::end_node(const tree::node& N) -> error {
	return pimpl_->end_node(N);
}

auto tree_fs_output::save_object(const objbase& obj) -> error {
	return pimpl_->save_object(*this, obj);
}

auto tree_fs_output::wait_objects_saved(timespan how_long) const -> std::vector<error> {
	return pimpl_->wait_objects_saved(how_long);
}

auto tree_fs_output::saveBinaryValue(const void* data, size_t size, const char* name) -> void {
	head().map([=](cereal::JSONOutputArchive* jar) {
		jar->saveBinaryValue(data, size, name);
	});
}

auto tree_fs_output::will_serialize_node(objbase const* obj) -> bool {
	if(auto pfmt = get_active_formatter(obj->bs_type().name); obj->is_node() && pfmt)
		return pfmt->stores_node;
	return true;
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

auto prologue(tree_fs_output& ar, tree::sp_link const& L) -> void {
	ar.begin_link(L);
}

auto epilogue(tree_fs_output& ar, tree::sp_link const&) -> void {
	ar.end_link();
}

auto prologue(tree_fs_output& ar, tree::node const& N) -> void {
	ar.begin_node(N);
}

auto epilogue(tree_fs_output& ar, tree::node const& N) -> void {
	ar.end_node(N);
}

NAMESPACE_END(blue_sky)
