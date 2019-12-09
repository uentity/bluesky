/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/kernel/radio.h>
#include "link_actor.h"

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

OMIT_OBJ_SERIALIZATION
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::link::modificator_f)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using bs_detail::shared;

link::link(std::shared_ptr<link_impl> impl, bool start_actor)
	: pimpl_(std::move(impl))
{
	if(!pimpl_) throw error{ "Trying to construct tree::link with invalid (null) impl" };
	if(start_actor) start_engine();
}

link::~link() {
	//caf::anon_send(actor_, a_bye());
	caf::anon_send_exit(actor(*this), caf::exit_reason::user_shutdown);
}

auto link::start_engine() -> bool {
	if(!actor_) {
		actor_ = pimpl_->spawn_actor(pimpl_);
		return true;
	}
	return false;
}

auto link::pimpl() const -> link_impl* {
	return pimpl_.get();
}

/// access link's unique ID
auto link::id() const -> id_type {
	// ID cannot change
	return pimpl_->id_;
}

auto link::rename(std::string new_name) -> void {
	caf::anon_send(actor(*this), a_lnk_rename(), std::move(new_name), false);
}

auto link::rename_silent(std::string new_name) -> void {
	caf::anon_send(actor(*this), a_lnk_rename(), std::move(new_name), true);
}

/// get link's container
auto link::owner() const -> sp_node {
	auto guard = pimpl_->lock(shared);
	return pimpl_->owner_.lock();
}

void link::reset_owner(const sp_node& new_owner) {
	pimpl_->reset_owner(new_owner);
}

auto link::info() const -> result_or_err<inode> {
	return pimpl_->actorf<result_or_errbox<inodeptr>>(*this, a_lnk_inode())
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

link::Flags link::flags() const {
	return pimpl_->actorf<Flags>(*this, a_lnk_flags()).value_or(Flags::Plain);
}

void link::set_flags(Flags new_flags) {
	caf::anon_send(actor(*this), a_lnk_flags(), new_flags);
	//pimpl_->flags_ = new_flags;
}

auto link::req_status(Req request) const -> ReqStatus {
	return pimpl_->req_status(request);
}

auto link::rs_reset(Req request, ReqStatus new_rs) -> ReqStatus {
	return pimpl_->actorf<ReqStatus>(
		*this, a_lnk_status(), request, ReqReset::Always, new_rs, ReqStatus::Void
	).value_or(ReqStatus::Error);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) -> ReqStatus {
	return pimpl_->actorf<ReqStatus>(
		*this, a_lnk_status(), request, ReqReset::IfEq, new_rs, self
	).value_or(ReqStatus::Error);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) -> ReqStatus {
	return pimpl_->actorf<ReqStatus>(
		*this, a_lnk_status(), request, ReqReset::IfNeq, new_rs, self
	).value_or(ReqStatus::Error);
}

/*-----------------------------------------------------------------------------
 *  sync API
 *-----------------------------------------------------------------------------*/
/// obtain link's human-readable name
auto link::name() const -> std::string {
	return pimpl_->actorf<std::string>(*this, a_lnk_name()).value_or("");
	//return pimpl_->name_;
}

// get link's object ID
std::string link::oid() const {
	return pimpl_->actorf<std::string>(*this, a_lnk_oid())
		.value_or(nil_oid);
	//return pimpl_->data()
	//	.map([](const sp_obj& obj) { return obj ? obj->id() : nil_oid; })
	//	.value_or(nil_oid);
}

std::string link::obj_type_id() const {
	return pimpl_->actorf<std::string>(*this, a_lnk_otid())
		.value_or( type_descriptor::nil().name );
	//return pimpl_->data()
	//	.map([](const sp_obj& obj) { return obj ? obj->type_id() : type_descriptor::nil().name; })
	//	.value_or(type_descriptor::nil().name);
}

auto link::data_ex(bool wait_if_busy) const -> result_or_err<sp_obj> {
	return pimpl_->actorf<result_or_errbox<sp_obj>>(*this, a_lnk_data(), wait_if_busy);
}

auto link::data_node_ex(bool wait_if_busy) const -> result_or_err<sp_node> {
	return pimpl_->actorf<result_or_errbox<sp_node>>(*this, a_lnk_dnode(), wait_if_busy);
}

auto link::modify_data(modificator_f m, bool silent) const -> error {
	auto res = pimpl_->actorf<error::box>(*this, a_apply(), std::move(m), silent);
	return res ? error::unpack(res.value()) : res.error();
}
auto link::modify_data(launch_async_t, modificator_f m, bool silent) const -> void {
	caf::anon_send(actor(), a_apply(), std::move(m), silent);
}

auto link::data_node_gid() const -> result_or_err<std::string> {
	return pimpl_->actorf<result_or_errbox<std::string>>(*this, a_node_gid());
}

auto link::is_node() const -> bool {
	return !data_node_gid().value_or("").empty();
}

void link::self_handle_node(const sp_node& N) {
	if(N) N->set_handle(shared_from_this());
}

result_or_err<sp_node> link::propagate_handle() {
	return is_node() ? data_node_ex()
	.and_then( [this](sp_node&& N) -> result_or_err<sp_node> {
		N->set_handle(shared_from_this());
		return std::move(N);
	} )
	: tl::make_unexpected( error::quiet(Error::NotANode) );
}

/*-----------------------------------------------------------------------------
 *  async API
 *-----------------------------------------------------------------------------*/

auto link::data(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		actor(*this), def_timeout(true), high_priority,
		[f = std::move(f), self = shared_from_this()](result_or_errbox<sp_obj> eobj) {
			f(std::move(eobj), std::move(self));
		},
		a_lnk_data(), true
	);
}

auto link::data_node(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		actor(*this), def_timeout(true), high_priority,
		[f = std::move(f), self = shared_from_this()](result_or_errbox<sp_node> eobj) {
			f(std::move(eobj), std::move(self));
		},
		a_lnk_dnode(), true
	);
}

/*-----------------------------------------------------------------------------
 *  subscribers management
 *-----------------------------------------------------------------------------*/
auto link::subscribe(handle_event_cb f, Event listen_to) const -> std::uint64_t {
	struct ev_state { handle_event_cb f; };

	// produce event bhavior that calls passed callback with proper params
	auto make_ev_character = [L = shared_from_this(), listen_to, f = std::move(f)](
		caf::stateful_actor<ev_state>* self
	) {
		auto res = caf::message_handler{};
		self->state.f = std::move(f);

		if(enumval(listen_to & Event::LinkRenamed))
			res = res.or_else(
				[self, wL = std::weak_ptr{L}] (
					a_ack, a_lnk_rename, std::string new_name, std::string old_name
				) {
					if(auto lnk = wL.lock())
						self->state.f(std::move(lnk), Event::LinkRenamed, {
							{"new_name", std::move(new_name)},
							{"prev_name", std::move(old_name)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkStatusChanged))
			res = res.or_else(
				[self, wL = std::weak_ptr{L}] (
					a_ack, a_lnk_status, Req request, ReqStatus new_v, ReqStatus prev_v
				) {
					if(auto lnk = wL.lock())
						self->state.f(std::move(lnk), Event::LinkStatusChanged, {
							{"request", prop::integer(new_v)},
							{"new_status", prop::integer(new_v)},
							{"prev_status", prop::integer(prev_v)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkDeleted))
			res = res.or_else(
				[self, lid = L->id()](a_bye) {
					self->state.f(sp_link{}, Event::LinkDeleted, {{ "lid", to_string(lid) }});
				}
			);

		return res;
	};

	// make shiny new subscriber actor, place into parent's room and return it's ID
	auto baby = system().spawn(ev_listener_actor<ev_state>, pimpl_->self_grp, std::move(make_ev_character));
	// and return ID
	return baby.id();
}

auto link::unsubscribe(std::uint64_t event_cb_id) const -> void {
	const auto ev_actor = system().registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	caf::anon_send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
}

NAMESPACE_END(blue_sky::tree)
