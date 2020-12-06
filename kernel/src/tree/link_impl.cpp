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

#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

#include <optional>

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

auto link_impl::rs_apply(Req req, function_view< void() > f, ReqReset cond, ReqStatus cond_value)
-> ReqStatus {
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

	// remove possible extra flags from cond
	cond &= 3;
	// atomic value set
	const auto self = status_[i].value;
	if( cond == ReqReset::Always ||
		(cond == ReqReset::IfEq && self == cond_value) ||
		(cond == ReqReset::IfNeq && self != cond_value)
	) {
		f();
	}
	return self;
}

auto link_impl::rs_reset(
	Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs, on_rs_changed_fn on_rs_changed
) -> ReqStatus {
	const auto i = enumval<unsigned>(request);
	if(i > 1) return ReqStatus::Error;

	return rs_apply(request, [&] {
		// update status value
		auto& S = status_[i];
		const auto self = S.value;
		S.value = new_rs;
		if(enumval(cond & ReqReset::Broadcast))
			status_[1 - i].value = new_rs;

		// new status = OK will always trigger callback
		if(new_rs != self || new_rs == ReqStatus::OK) {
			// if postcondition failed - restore prev status
			if(!on_rs_changed(request, new_rs, self)) {
				S.value = self;
				if(enumval(cond & ReqReset::Broadcast))
					status_[1 - i].value = self;
			}
		}
	}, cond, old_rs);
}

auto link_impl::rs_reset(Req request, ReqReset cond, ReqStatus new_rs, ReqStatus old_rs) -> ReqStatus {
	return rs_reset(request, cond, new_rs, old_rs, [this](auto req, auto new_rs, auto prev_rs) {
		// send notification to link's home group
		send_home<high_prio>(*this, a_ack(), a_lnk_status(), req, new_rs, prev_rs);
		return true;
	});
}

auto link_impl::rs_update_from_data(req_result rdata, bool broadcast) -> void {
	std::visit([&](auto&& obj) {
		// infer static request type & result
		using res_t = meta::remove_cvref_t<decltype(obj)>;
		static constexpr auto req = [&] {
			if constexpr(std::is_same_v<res_t, obj_or_errbox>)
				return Req::Data;
			else
				return Req::DataNode;
		}();

		// convert req result to result of another req
		const auto convert_req_res = [&](const res_t& obj) {
			if constexpr(req == Req::Data)
				// feed DataNode waiters by extracting node from object
				return obj.and_then([&](const auto& obj) -> node_or_errbox {
					return obj->data_node();
				});
			else
				// if uniform DataNode request is completed => we already have valid object
				// get it using data(unsafe)
				return obj.and_then([&](auto&) -> obj_or_errbox {
					if(auto obj = data(unsafe))
						return obj;
					return unexpected_err_quiet(Error::EmptyData);
				});
		};

		// release all waiers with given value
		const auto feed_waiters = [&](Req who, const auto& value) {
			auto& rsh = req_status_handle(who);
			for(auto& w : rsh.waiters)
				caf::anon_send(w, a_apply(), value);
			rsh.waiters.clear();
		};

		// prepare transaction that will run after status is updated
		const auto req_transaction = [&](Req, ReqStatus new_rs, ReqStatus old_rs) {
			// send notification strictly if status changes
			if(new_rs != old_rs)
				send_home<high_prio>(*this, a_ack(), a_lnk_status(), req, new_rs, old_rs);

			// for Data request if object is nil -> return error
			if constexpr(req == Req::Data) {
				obj = obj.and_then([](auto&& obj) -> res_t {
					return obj ? res_t{std::move(obj)} : unexpected_err_quiet(Error::EmptyData);
				});
			}

			// feed primary request waiters
			feed_waiters(req, obj);
			// for broadcasted request feed waiters on another queue
			if(broadcast)
				feed_waiters(enumval<Req>(1 - enumval(req)), convert_req_res(obj));

			return true;
		};

		// update status & run transaction
		auto cond = ReqReset::Always;
		if(broadcast) cond |= ReqReset::Broadcast;
		rs_reset(
			req, cond, obj ? (*obj ? ReqStatus::OK : ReqStatus::Void) : ReqStatus::Error,
			ReqStatus::Void, req_transaction
		);
	}, std::move(rdata));
}

auto link_impl::rs_add_waiter(Req req, caf::actor w) -> void {
	rs_apply(req, [&] { req_status_handle(req).waiters.push_back(std::move(w)); });
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
