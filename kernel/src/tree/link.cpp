/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "link_actor.h"
#include "ev_listener_actor.h"
#include "nil_link.h"

#include <bs/log.h>
#include <bs/kernel/radio.h>

#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

OMIT_OBJ_SERIALIZATION
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(blue_sky::tree::data_modificator_f)

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;
using bs_detail::shared;

///////////////////////////////////////////////////////////////////////////////
//  actor_handle
//
link::actor_handle::actor_handle(caf::actor Lactor)
	: actor_(std::move(Lactor))
{}

// destructor of actor handle terminates wrapped actor
link::actor_handle::~actor_handle() {
	caf::anon_send_exit(actor_, caf::exit_reason::user_shutdown);
}

///////////////////////////////////////////////////////////////////////////////
//  link_weak_ptr
//
link::weak_ptr::weak_ptr(const link& src)
	: actor_(src.actor_), pimpl_(src.pimpl_)
{}

auto link::weak_ptr::operator=(const link& rhs) -> weak_ptr& {
	pimpl_ = rhs.pimpl_;
	actor_ = rhs.actor_;
	return *this;
}

auto link::weak_ptr::lock() const -> link {
	auto res = link{ pimpl_.lock(), false };
	if(res) res.actor_ = actor_.lock();
	// if handle actor is dead -> self = nil
	if(!res.actor_) res = link{};
	return res;
}

auto link::weak_ptr::expired() const -> bool {
	return actor_.expired();
}

auto link::weak_ptr::reset() -> void {
	pimpl_.reset();
	actor_.reset();
}

auto link::weak_ptr::operator==(const weak_ptr& rhs) const -> bool {
	return pimpl_.lock() == rhs.pimpl_.lock();
}

auto link::weak_ptr::operator!=(const weak_ptr& rhs) const -> bool {
	return !(*this == rhs);
}

auto link::weak_ptr::operator==(const link& rhs) const -> bool {
	return pimpl_.lock() == rhs.pimpl_;
}

auto link::weak_ptr::operator!=(const link& rhs) const -> bool {
	return !(*this == rhs);
}

auto link::weak_ptr::operator<(const weak_ptr& rhs) const -> bool {
	return pimpl_.lock() < rhs.pimpl_.lock();
}

///////////////////////////////////////////////////////////////////////////////
//  link
//
link::link()
	: factor_(system(), true), actor_(nil_link::actor()), pimpl_(nil_link::pimpl())
{}

link::link(std::string name, sp_obj data, Flags f)
	: link(hard_link{std::move(name), std::move(data), f})
{}

link::link(std::shared_ptr<link_impl> impl, bool start_actor)
	: factor_(system(), true), actor_(nil_link::actor()),
	pimpl_(impl ? std::move(impl) : nil_link::pimpl())
{
	if(start_actor) start_engine();
}

link::link(const link& rhs)
	: factor_(system(), true), actor_(rhs.actor_), pimpl_(rhs.pimpl_)
{}

link::link(const link& rhs, std::string_view rhs_type_id)
	: factor_(system(), true),
	actor_(rhs.type_id() == rhs_type_id ? rhs.actor_ : nil_link::actor()),
	pimpl_(rhs.type_id() == rhs_type_id ? rhs.pimpl_ : nil_link::pimpl())
{}

link::~link() = default;

auto link::operator=(const link& rhs) -> link& {
	actor_ = rhs.actor_;
	pimpl_ = rhs.pimpl_;
	return *this;
}

auto link::reset() -> void {
	actor_ = nil_link::actor();
	pimpl_ = nil_link::pimpl();
}

auto link::start_engine() -> bool {
	if(actor_ == nil_link::self().actor()) {
		actor_ = std::make_shared<actor_handle>(pimpl_->spawn_actor(pimpl_));
		return true;
	}
	return false;
}

auto link::operator==(const link& rhs) const -> bool {
	return pimpl_ == rhs.pimpl_;
}

auto link::operator!=(const link& rhs) const -> bool {
	return !(*this == rhs);
}

auto link::operator<(const link& rhs) const -> bool {
	return id() < rhs.id();
}

auto link::is_nil() const -> bool {
	return pimpl_ == nil_link::pimpl();
}

auto link::clone(bool deep) const -> link {
	return { pimpl_->clone(deep) };
}

auto link::pimpl() const -> link_impl* {
	return pimpl_.get();
}

auto link::factor() const -> const caf::scoped_actor& {
	return factor_;
}

auto link::raw_actor() const -> const caf::actor& {
	return actor_->actor_;
}

auto link::type_id() const -> std::string_view {
	return pimpl_->type_id();
}

/// access link's unique ID
auto link::id() const -> lid_type {
	// ID cannot change
	return pimpl_->id_;
}

auto link::rename(std::string new_name) const -> void {
	caf::anon_send(actor(*this), a_lnk_rename(), std::move(new_name), false);
}

auto link::rename_silent(std::string new_name) const -> void {
	caf::anon_send(actor(*this), a_lnk_rename(), std::move(new_name), true);
}

/// get link's container
auto link::owner() const -> sp_node {
	auto guard = pimpl_->lock(shared);
	return pimpl_->owner_.lock();
}

auto link::reset_owner(const sp_node& new_owner) const -> void {
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

auto link::info(unsafe_t) const -> result_or_err<inode> {
	return pimpl_->get_inode()
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

auto link::flags() const -> Flags {
	return pimpl_->actorf<Flags>(*this, a_lnk_flags()).value_or(Flags::Plain);
}

auto link::flags(unsafe_t) const -> Flags {
	return pimpl_->flags_;
}

auto link::set_flags(Flags new_flags) const -> void {
	caf::anon_send(actor(*this), a_lnk_flags(), new_flags);
}

auto link::req_status(Req request) const -> ReqStatus {
	return pimpl_->req_status(request);
}

auto link::rs_reset(Req request, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->actorf<ReqStatus>(
		*this, a_lnk_status(), request, ReqReset::Always, new_rs, ReqStatus::Void
	).value_or(ReqStatus::Error);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->actorf<ReqStatus>(
		*this, a_lnk_status(), request, ReqReset::IfEq, new_rs, self
	).value_or(ReqStatus::Error);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
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
}

auto link::name(unsafe_t) const -> std::string {
	return pimpl_->name_;
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
		.value_or( nil_otid );
	//return pimpl_->data()
	//	.map([](const sp_obj& obj) { return obj ? obj->type_id() : nil_otid; })
	//	.value_or(nil_otid);
}

auto link::data_ex(bool wait_if_busy) const -> result_or_err<sp_obj> {
	return pimpl_->actorf<result_or_errbox<sp_obj>>(*this, a_lnk_data(), wait_if_busy);
}

auto link::data_node_ex(bool wait_if_busy) const -> result_or_err<sp_node> {
	return pimpl_->actorf<result_or_errbox<sp_node>>(*this, a_lnk_dnode(), wait_if_busy);
}

auto link::data_apply(data_modificator_f m, bool silent) const -> error {
	auto res = pimpl_->actorf<error::box>(*this, a_apply(), std::move(m), silent);
	return res ? error::unpack(res.value()) : res.error();
}
auto link::data_apply(launch_async_t, data_modificator_f m, bool silent) const -> void {
	caf::anon_send(actor(*this), a_apply(), std::move(m), silent);
}

auto link::data_node_gid() const -> result_or_err<std::string> {
	return pimpl_->actorf<result_or_errbox<std::string>>(*this, a_node_gid());
}

auto link::is_node() const -> bool {
	return !data_node_gid().value_or("").empty();
}

void link::self_handle_node(const sp_node& N) const {
	if(N) N->set_handle(*this);
}

auto link::propagate_handle() const -> result_or_err<sp_node> {
	return pimpl_->propagate_handle(*this);
}

/*-----------------------------------------------------------------------------
 *  async API
 *-----------------------------------------------------------------------------*/

auto link::data(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		actor(*this), def_timeout(true), high_priority,
		[f = std::move(f), self_impl = pimpl_](result_or_errbox<sp_obj> eobj) {
			f(std::move(eobj), link(self_impl));
		},
		a_lnk_data(), true
	);
}

auto link::data_node(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		actor(*this), def_timeout(true), high_priority,
		[f = std::move(f), self_impl = pimpl_](result_or_errbox<sp_node> eobj) {
			f(std::move(eobj), link(self_impl));
		},
		a_lnk_dnode(), true
	);
}

/*-----------------------------------------------------------------------------
 *  subscribers management
 *-----------------------------------------------------------------------------*/
auto link::subscribe(handle_event_cb f, Event listen_to) const -> std::uint64_t {
	using baby_t = ev_listener_actor<link>;

	// produce event bhavior that calls passed callback with proper params
	auto make_ev_character = [weak_src = weak_ptr{*this}, src_id = id(), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed))
			res = res.or_else(
				[=](a_ack, a_lnk_rename, std::string new_name, std::string old_name) {
					if(auto src = weak_src.lock())
						self->f(link{std::move(src)}, Event::LinkRenamed, {
							{"new_name", std::move(new_name)},
							{"prev_name", std::move(old_name)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkStatusChanged))
			res = res.or_else(
				[=](a_ack, a_lnk_status, Req request, ReqStatus new_v, ReqStatus prev_v) {
					if(auto src = weak_src.lock())
						self->f(link{std::move(src)}, Event::LinkStatusChanged, {
							{"request", prop::integer(new_v)},
							{"new_status", prop::integer(new_v)},
							{"prev_status", prop::integer(prev_v)}
						});
				}
			);

		if(enumval(listen_to & Event::LinkDeleted))
			res = res.or_else(
				[=](a_bye) {
					// when overriding this we must call `disconnect()` explicitly
					self->disconnect();
					self->f( weak_src.lock(), Event::LinkDeleted, {{ "lid", to_string(src_id) }} );
				}
			);

		return res;
	};

	// make baby event handler actor
	auto baby = system().spawn_in_group<baby_t>(
		pimpl_->home, pimpl_->home, std::move(f), std::move(make_ev_character)
	);
	// return baby ID
	return baby.id();
}

auto link::unsubscribe(std::uint64_t event_cb_id) const -> void {
	const auto ev_actor = system().registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	caf::anon_send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
}

NAMESPACE_END(blue_sky::tree)
