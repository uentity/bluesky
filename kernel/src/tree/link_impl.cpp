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

#include <bs/atoms.h>
#include <bs/actor_common.h>
#include <bs/log.h>
#include <bs/uuid.h>
#include <bs/kernel/tools.h>

#include <optional>
#include <variant>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  link_impl
 *-----------------------------------------------------------------------------*/
link_impl::link_impl(std::string name, Flags f)
	: id_(gen_uuid()), name_(std::move(name)), flags_(f)
{}

link_impl::link_impl()
	: link_impl("", Flags::Plain)
{}

link_impl::~link_impl() = default;

auto link_impl::spawn_actor(sp_limpl limpl) const -> caf::actor {
	return spawn_lactor<link_actor>(std::move(limpl));
}

auto link_impl::data(unsafe_t) const -> sp_obj { return nullptr; }

auto link_impl::data_node(unsafe_t) const -> node {
	if(auto obj = data(unsafe))
		return obj->data_node();
	return node::nil();
}

auto link_impl::propagate_handle(node& N) const -> node& {
	if(N) N.pimpl()->set_handle(super_engine());
	return N;
}

auto link_impl::propagate_handle() -> node_or_err {
	if(auto obj = data(unsafe)) {
		if(auto N = obj->data_node())
			return propagate_handle(N);
		return unexpected_err_quiet(Error::NotANode);
	}
	return unexpected_err_quiet(Error::EmptyData);
}

auto link_impl::owner() const -> node {
	auto guard = lock(blue_sky::detail::shared);
	return owner_.lock();
}

auto link_impl::reset_owner(const node& new_owner) -> void {
	auto guard = lock();
	owner_ = new_owner;
}

auto link_impl::req_status(Req request) const -> ReqStatus {
	if(const auto i = enumval(request); i < 2) {
		auto guard = std::shared_lock{ status_[i].guard };
		return status_[i].value;
	}
	return ReqStatus::Error;
}

auto link_impl::req_status_handle(Req request) -> status_handle& {
	return status_[enumval(request) & 1];
}

auto link_impl::rs_apply(Req req, function_view< void() > f, ReqReset cond, ReqStatus cond_value) -> bool {
	const auto i = enumval<unsigned>(req);

	// obtain either single lock or global lock depending on `ReqReset::Broadcast` flag
	using single_lock_t = std::scoped_lock<engine_impl_mutex>;
	using global_lock_t = std::scoped_lock<engine_impl_mutex, engine_impl_mutex>;
	using either_lock_t = std::optional<std::variant<single_lock_t, global_lock_t>>;
	auto locker = [&] {
		if(enumval(cond & ReqReset::Broadcast))
			return either_lock_t{
				std::in_place, std::in_place_type_t<global_lock_t>(), status_[i].guard, status_[1 - i].guard
			};
		else
			return either_lock_t{
				std::in_place, std::in_place_type_t<single_lock_t>(), status_[i].guard
			};
	}();

	// run f
	// remove possible extra flags from cond
	cond &= 3;
	// atomic value set
	const auto self = status_[i].value;
	if( cond == ReqReset::Always ||
		(cond == ReqReset::IfEq && self == cond_value) ||
		(cond == ReqReset::IfNeq && self != cond_value)
	) {
		f();
		return true;
	}
	return false;
}

auto link_impl::rs_reset(
	Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs,
	on_rs_changed_fn on_rs_changed
) -> ReqStatus {
	const auto i = enumval<unsigned>(request);
	if(i > 1) return ReqStatus::Error;
	auto& S = status_[i];

	bool trigger_event = false;
	auto self = ReqStatus::Void;
	rs_apply(request, [&] {
		self = S.value;
		S.value = new_rs;
		if(enumval(cond & ReqReset::Broadcast))
			status_[1 - i].value = new_rs;
		// status = OK will always fire (works as 'dirty' signal)
		if(new_rs != self || new_rs == ReqStatus::OK)
			trigger_event = true;
	}, cond, old_rs);

	if(trigger_event)
		on_rs_changed(request, new_rs, self);
	return self;
}

auto link_impl::rs_reset(Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs) -> ReqStatus {
	return rs_reset(request, cond, new_rs, old_rs, [this](auto req, auto new_rs, auto prev_rs) {
		// send notification to link's home group
		checked_send<link_impl::home_actor_type, high_prio>(
			home, a_ack(), a_lnk_status(), req, new_rs, prev_rs
		);
	});
}

auto link_impl::rename(std::string new_name)-> void {
	if(new_name == name_) return;
	auto old_name = std::move(name_);
	name_ = std::move(new_name);
	// notify home group
	checked_send<home_actor_type, high_prio>(
		home, a_ack(), a_lnk_rename(), name_, std::move(old_name)
	);
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
