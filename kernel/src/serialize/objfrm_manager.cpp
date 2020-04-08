/// @file
/// @author uentity
/// @date 08.04.2020
/// @brief Objects save/load jobs manager impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "tree_fs_impl.h"

#include <bs/actor_common.h>
#include <bs/objbase.h>
#include <bs/tree/errors.h>
#include <bs/serialize/cafbind.h>

#include <cereal/types/vector.hpp>

NAMESPACE_BEGIN(blue_sky::detail)
using errb_vector = std::vector<error::box>;
using blue_sky::tree::Error;

objfrm_manager::objfrm_manager(caf::actor_config& cfg, bool is_saving) :
	objfrm_manager_t::base(cfg), is_saving_(is_saving)
{}

auto objfrm_manager::cut_save_errors() {
	auto finally = scope_guard{[=]{ er_stack_.clear(); }};
	return std::move(er_stack_);
}

template<bool Saving, typename Master>
static auto make_frm_job(object_formatter* F, std::string fname, Master master) {
	return [F, fname = std::move(fname), master = std::move(master)](const sp_obj& obj) {
		auto er = [&] {
			if constexpr(Saving)
				return F->save(*obj, std::move(fname));
			else
				return F->load(*obj, std::move(fname));
		}();
		caf::anon_send(master, er.pack());
		return er;
	};
}

auto objfrm_manager::make_behavior() -> behavior_type {
return {
	// save given object
	[=](caf::actor self, const sp_cobj& obj, const std::string& fmt_name, std::string fname) {
		auto res = error::box{};
		if(auto F = get_formatter(obj->type_id(), fmt_name)) {
			// run save in object's queue
			if(is_saving_)
				obj->apply(launch_async, make_frm_job<true>(F, std::move(fname), std::move(self)));
			else
				obj->apply(launch_async, make_frm_job<false>(F, std::move(fname), std::move(self)));
			// inc counter of started save jobs
			++nstarted_;
		}
		else
			er_stack_.emplace_back(error{obj->type_id(), Error::MissingFormatter});
	},

	// store error from finished saver, deliver errors stack when save is done
	[=](error::box er) {
		if(er.ec) er_stack_.push_back(std::move(er));
		if(++nfinished_ == nstarted_)
			boxed_errs_.deliver(cut_save_errors());
	},

	[=](a_ack) -> caf::result<errb_vector> {
		if(nfinished_ == nstarted_)
			return cut_save_errors();
		else {
			boxed_errs_ = make_response_promise<errb_vector>();
			return boxed_errs_;
		}
	}
}; }

auto objfrm_manager::wait_jobs_done(objfrm_manager_t self, timespan how_long) -> std::vector<error> {
	auto fmanager = caf::make_function_view(
		self, how_long == infinite ? caf::infinite : caf::duration{how_long}
	);

	auto res = std::vector<error>{};
	auto boxed_res = actorf<std::vector<error::box>>(fmanager, a_ack());
	if(boxed_res) {
		auto& boxed_errs = *boxed_res;
		res.reserve(boxed_errs.size());
		for(auto& er_box : boxed_errs)
			res.push_back( error::unpack(std::move(er_box)) );
	}
	else
		res.push_back(std::move(boxed_res.error()));
	return res;
}

NAMESPACE_END(blue_sky::detail)
