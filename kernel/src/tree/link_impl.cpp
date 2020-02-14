/// @file
/// @author uentity
/// @date 22.07.2019
/// @brief Link implementation that contain link state
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_impl.h"
#include "link_actor.h"
#include "actor_common.h"
#include <bs/atoms.h>
#include <bs/log.h>

#include <boost/uuid/random_generator.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

// global random UUID generator for BS links
static boost::uuids::random_generator idgen;

/*-----------------------------------------------------------------------------
 *  link_impl
 *-----------------------------------------------------------------------------*/
link_impl::link_impl(std::string name, Flags f)
	: id_(idgen()), name_(std::move(name)), flags_(f), timeout(def_timeout(true))
{}

link_impl::link_impl()
	: link_impl("", Flags::Plain)
{}

link_impl::~link_impl() = default;

auto link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<link_actor>(std::move(limpl));
}

auto link_impl::propagate_handle(const link& super) -> result_or_err<sp_node> {
	return actorf<result_or_errbox<sp_node>>(super, a_lnk_dnode(), false)
	.and_then( [&](sp_node&& N) -> result_or_err<sp_node> {
		super.self_handle_node(N);
		return std::move(N);
	} );
}

auto link_impl::set_node_handle(const link& h, const sp_node& N) -> void {
	h.self_handle_node(N);
}

auto link_impl::reset_owner(const sp_node& new_owner) -> void {
	auto guard = lock();
	owner_ = new_owner;
}

auto link_impl::req_status(Req request) const -> ReqStatus {
	if(const auto i = (unsigned)request; i < 2) {
		auto guard = std::lock_guard{ status_[i].guard };
		return status_[i].value;
	}
	return ReqStatus::Void;
}

auto link_impl::rs_reset(
	Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs,
	on_rs_changed_fn on_rs_changed
) -> ReqStatus {
	const auto i = (unsigned)request;
	if(i >= 2) return ReqStatus::Error;

	// remove possible extra flags from cond
	cond &= 3;
	auto& S = status_[i];

	// atomic set value
	S.guard.lock();
	const auto self = S.value;
	if( cond == ReqReset::Always ||
		(cond == ReqReset::IfEq && self == old_rs) ||
		(cond == ReqReset::IfNeq && self != old_rs)
	) {
		S.value = new_rs;
		S.guard.unlock();
		// Data = OK will always fire (work as 'data changed' signal)
		if(new_rs != self || (request == Req::Data && new_rs == ReqStatus::OK))
			on_rs_changed(request, new_rs, self);
	}
	else
		S.guard.unlock();

	return self;
}

///////////////////////////////////////////////////////////////////////////////
//  inode
//
auto link_impl::get_inode() -> result_or_err<inodeptr> {
	// default implementation obtains inode from `data()->inode_`
	return data().and_then([](const sp_obj& obj) {
		return obj ?
			result_or_err<inodeptr>(obj->inode_.lock()) :
			tl::make_unexpected(error::quiet(Error::EmptyData));
	});
}

auto link_impl::make_inode(const sp_obj& obj, inodeptr new_i) -> inodeptr {
	auto obj_i = obj ? obj->inode_.lock() : nullptr;
	if(!obj_i)
		obj_i = new_i ? std::move(new_i) : std::make_shared<inode>();

	if(obj)
		obj->inode_ = obj_i;
	return obj_i;
}

/*-----------------------------------------------------------------------------
 *  ilink_impl
 *-----------------------------------------------------------------------------*/
ilink_impl::ilink_impl(std::string name, const sp_obj& data, Flags f)
	: super(std::move(name), f), inode_(make_inode(data))
{}

ilink_impl::ilink_impl()
	: super()
{}

auto ilink_impl::get_inode() -> result_or_err<inodeptr> {
	return inode_;
};

/*-----------------------------------------------------------------------------
 *  misc
 *-----------------------------------------------------------------------------*/
auto to_string(Req r) -> const char* {
	switch(r) {
	case Req::Data : return "Data";
	case Req::DataNode : return "DataNode";
	default: return "<bad request>";
	}
}

auto to_string(ReqStatus s) -> const char* {
	switch(s) {
	case ReqStatus::Void : return "Void";
	case ReqStatus::Busy : return "Busy";
	case ReqStatus::OK : return "OK";
	case ReqStatus::Error : return "Error";
	default: return "<bad request status>";
	}
}

NAMESPACE_END(blue_sky::tree)
