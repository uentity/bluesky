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

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky::tree)
using namespace kernel::radio;

link::link(caf::actor impl_a) : aimpl_(std::move(impl_a)) {
	pimpl_ = caf::actor_cast<link_actor*>(aimpl_);
	if(!pimpl_) throw error{ "Trying to construct tree::link with invalid actor" };
	fimpl_ = caf::make_function_view(aimpl_, caf::duration{pimpl_->timeout_});
	//pimpl_->master_ = this;
}

link::~link() {
	pimpl_->goodbye();
}

auto link::pimpl() const -> link_actor* {
	return pimpl_;
}

auto link::actor() const -> caf::actor {
	return aimpl_;
}

/// access link's unique ID
auto link::id() const -> id_type {
	return actorf<id_type>(fimpl_, a_lnk_id()).value_or(nil_uid);
}

/// obtain link's human-readable name
auto link::name() const -> std::string {
	return actorf<std::string>(fimpl_, a_lnk_name()).value_or("");
}

/// get link's container
auto link::owner() const -> sp_node {
	return pimpl_->owner_.lock();
}

void link::reset_owner(const sp_node& new_owner) {
	pimpl_->reset_owner(new_owner);
}

auto link::info() const -> result_or_err<inode> {
	//return pimpl_->get_inode()
	return actorf<result_or_errbox<inodeptr>>(fimpl_, a_lnk_inode())
	.and_then([](const inodeptr& i) {
		return i ?
			result_or_err<inode>(*i) :
			tl::make_unexpected(error::quiet(Error::EmptyInode));
	});
}

link::Flags link::flags() const {
	return pimpl_->flags_;
}

void link::set_flags(Flags new_flags) {
	std::lock_guard<std::mutex> g(pimpl_->solo_);
	pimpl_->flags_ = new_flags;
}

auto link::rename(std::string new_name) -> void {
	pimpl_->send(pimpl_, a_lnk_rename(), std::move(new_name), false);
}

auto link::rename_silent(std::string new_name) -> void {
	pimpl_->send(pimpl_, a_lnk_rename(), std::move(new_name), true);
}

auto link::req_status(Req request) const -> ReqStatus {
	return pimpl_->req_status(request);
}

auto link::rs_reset(Req request, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset(request, ReqReset::Always, new_rs);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset(request, ReqReset::IfEq, new_rs, self);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset(request, ReqReset::IfNeq, new_rs, self);
}

auto link::rs_data_changed() const -> void {
	// unconditionally sends Data = OK, ack
	pimpl_->send(pimpl_->self_grp, a_lnk_status(), a_ack(), Req::Data, ReqStatus::OK, req_status(Req::Data));
}


/*-----------------------------------------------------------------------------
 *  sync API
 *-----------------------------------------------------------------------------*/
// get link's object ID
std::string link::oid() const {
	//pimpl_->pdbg() << "link: oid()" << std::endl;
	return actorf<std::string>(fimpl_, a_lnk_oid())
		.value_or( to_string(boost::uuids::nil_uuid()) );
}

std::string link::obj_type_id() const {
	//pimpl_->pdbg() << "link: obj_type_id()" << std::endl;
	return actorf<std::string>(fimpl_, a_lnk_otid())
		.value_or( type_descriptor::nil().name );
}

result_or_err<sp_obj> link::data_ex(bool wait_if_busy) const {
	return pimpl_->data_ex(wait_if_busy);
}

result_or_err<sp_node> link::data_node_ex(bool wait_if_busy) const {
	return pimpl_->data_node_ex(wait_if_busy);
}

void link::self_handle_node(const sp_node& N) {
	if(N) N->set_handle(shared_from_this());
}

result_or_err<sp_node> link::propagate_handle() {
	return data_node_ex()
	.and_then( [this](sp_node&& N) -> result_or_err<sp_node> {
		N->set_handle(shared_from_this());
		return std::move(N);
	} );
}

/*-----------------------------------------------------------------------------
 *  async API
 *-----------------------------------------------------------------------------*/

auto link::data(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		aimpl_, pimpl_->timeout_, high_priority,
		[f = std::move(f), self = shared_from_this()](result_or_errbox<sp_obj> eobj) {
			f(std::move(eobj), std::move(self));
		},
		a_lnk_data(), true
	);
}

auto link::data_node(process_data_cb f, bool high_priority) const -> void {
	anon_request(
		aimpl_, pimpl_->timeout_, high_priority,
		[f = std::move(f), self = shared_from_this()](result_or_errbox<sp_obj> eobj) {
			f(std::move(eobj), std::move(self));
		},
		a_lnk_dnode(), true
	);
}

/*-----------------------------------------------------------------------------
 *  subscribers management
 *-----------------------------------------------------------------------------*/
// event actor
auto ev_listener_actor(
	caf::event_based_actor* self, caf::group tgt_grp, caf::message_handler character
) -> caf::behavior {
	// silently drop all other messages not in my character
	self->set_default_handler([](caf::scheduled_actor* self, caf::message_view& mv) {
		return caf::drop(self, mv);
	});
	// come to mummy
	self->join(tgt_grp);
	auto& Reg = system().registry();
	Reg.put(self->id(), self);

	// unsubscribe when parent leaves its group
	return character.or_else(
		[self, grp = std::move(tgt_grp), &Reg](a_bye) {
			self->leave(grp);
			Reg.erase(self->id());
		}
	);
};

auto link::subscribe(handle_event_cb f, Event listen_to) -> std::uint64_t {
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
					a_lnk_rename, a_ack, std::string new_name, std::string old_name
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
					a_lnk_status, a_ack, Req request, ReqStatus new_v, ReqStatus prev_v
				) {
					if(auto lnk = wL.lock())
						self->state.f(std::move(lnk), Event::LinkStatusChanged, {
							{"request", prop::integer(new_v)},
							{"new_status", prop::integer(new_v)},
							{"prev_status", prop::integer(prev_v)}
						});
				}
			);

		return res;
	};

	// make shiny new subscriber actor, place into parent's room and return it's ID
	auto& AS = system();
	auto baby = AS.spawn(ev_listener_actor<ev_state>, pimpl_->self_grp, std::move(make_ev_character));
	// and return ID
	return baby.id();
}

auto link::unsubscribe(std::uint64_t event_cb_id) -> void {
	auto& AS = system();
	const auto ev_actor = AS.registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	pimpl_->send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
}

NAMESPACE_END(blue_sky::tree)
