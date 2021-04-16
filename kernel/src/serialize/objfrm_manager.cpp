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

#include "../objbase_actor.h"

#include <fmt/format.h>

#include <algorithm>

NAMESPACE_BEGIN(blue_sky::detail)
using errb_vector = std::vector<error::box>;
using blue_sky::tree::Error;

objfrm_manager::objfrm_manager(caf::actor_config& cfg, bool is_saving) :
	objfrm_manager_t::base(cfg), is_saving_(is_saving)
{}

auto objfrm_manager::session_ack() -> void {
	if(session_finished_ && (nstarted_ == nfinished_)) {
		boxed_errs_.deliver(er_stack_);
	}
}

auto objfrm_manager::make_behavior() -> behavior_type {
return {
	// stop session
	[=](a_bye) {
		if(!session_finished_) {
			session_finished_ = true;
			session_ack();
		}
	},

	// process given object
	[=](const sp_obj& obj, std::string fmt_name, std::string fname) {
		// sanity
		if(!obj) er_stack_.emplace_back(error{Error::EmptyData});
		// run job in object's queue
		++nstarted_;
		auto objA = objbase_actor::actor(*obj);
		auto frm_job = is_saving_ ?
			request(objA, caf::infinite, a_save(), std::move(fmt_name), std::move(fname)) :
			request(objA, caf::infinite, a_load(), std::move(fmt_name), std::move(fname))
		;
		// process result
		frm_job.then([=](error::box er) {
			// save result of finished job & inc finished counter
			++nfinished_;
			if(er.ec) er_stack_.push_back(std::move(er));
			session_ack();
		}, [=](const caf::error& er) {
			// in case smth went wrong with job posting
			--nstarted_;
			er_stack_.emplace_back(forward_caf_error(er, fmt::format(
				"failed to enqueue {} job: object[{}, {}] <-> {}",
				(is_saving_ ? "save" : "load"), obj->type_id(), obj->id(), fname
			)));
		});
	},

	[=](a_ack) -> caf::result<errb_vector> {
		boxed_errs_ = make_response_promise<errb_vector>();
		session_ack();
		return boxed_errs_;
	}
}; }

auto objfrm_manager::wait_jobs_done(objfrm_manager_t self, timespan how_long) -> std::vector<error> {
	auto fmanager = caf::make_function_view(
		self, how_long == infinite ? caf::infinite : how_long
	);

	auto res = std::vector<error>{};
	auto boxed_res = actorf<errb_vector>(fmanager, a_ack());
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
