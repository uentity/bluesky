/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/kernel/config.h>
#include "link_actor.h"

#include <boost/uuid/uuid_generators.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky::tree)

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

/// access link's unique ID
auto link::id() const -> const id_type& {
	return pimpl_->id_;
}

/// obtain link's symbolic name
auto link::name() const -> std::string {
	return pimpl_->name_;
}

/// get link's container
auto link::owner() const -> sp_node {
	return pimpl_->owner_.lock();
}

void link::reset_owner(const sp_node& new_owner) {
	pimpl_->reset_owner(new_owner);
}

auto link::info() const -> result_or_err<inode> {
	return pimpl_->get_inode().and_then([](const inodeptr& i) {
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
	return pimpl_->rs_reset(request, new_rs);
}

auto link::rs_reset_if_eq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset_if_eq(request, self, new_rs);
}

auto link::rs_reset_if_neq(Req request, ReqStatus self, ReqStatus new_rs) const -> ReqStatus {
	return pimpl_->rs_reset_if_neq(request, self, new_rs);
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
	auto lazy_f = [ f = std::move(f), self = shared_from_this()]( result_or_errbox<sp_obj> eobj) {
		f(std::move(eobj), std::move(self));
	};

	if(high_priority)
		pimpl_->request<caf::message_priority::high>(aimpl_, pimpl_->timeout_, a_lnk_data(), true)
		.then(std::move(lazy_f));
	else
		pimpl_->request<caf::message_priority::normal>(aimpl_, pimpl_->timeout_, a_lnk_data(), true)
		.then(std::move(lazy_f));
}

auto link::data_node(process_data_cb f, bool high_priority) const -> void {
	auto lazy_f = [ f = std::move(f), this ]( result_or_errbox<sp_node> eobj) {
		f(std::move(eobj), shared_from_this());
	};

	if(high_priority)
		pimpl_->request<caf::message_priority::high>(aimpl_, pimpl_->timeout_, a_lnk_dnode(), true)
		.then(std::move(lazy_f));
	else
		pimpl_->request<caf::message_priority::normal>(aimpl_, pimpl_->timeout_, a_lnk_dnode(), true)
		.then(std::move(lazy_f));
}

/*-----------------------------------------------------------------------------
 *  subscribers management
 *-----------------------------------------------------------------------------*/
auto link::subscribe(Event listen_to, handle_event_cb f) -> std::uint64_t {
	// event actor
	static constexpr auto listener_actor = [](
		caf::event_based_actor* self, caf::group link_grp, caf::message_handler character
	) {
		// silently drop all other messages not in my character
		self->set_default_handler([](caf::scheduled_actor* self, caf::message_view& mv) {
			return caf::drop(self, mv);
		});
		// come to mummy
		self->join(link_grp);
		// unsubscribe when parent leaves its group
		return character.or_else(
			[self, grp = std::move(link_grp)](a_bye) {
				self->leave(grp);
				kernel::config::actor_system().registry().erase(self->id());
			}
		);
	};

	// produce event bhavior that calls passed callback with proper params
	static constexpr auto make_ev_character = [](const sp_link& self, Event listen_to_, handle_event_cb& f_)
	-> caf::message_handler {
		switch(listen_to_) {
		case Event::Renamed :
			return [f = std::move(f_), self = std::weak_ptr{self}] (
				a_lnk_rename, a_ack, std::string new_name, std::string old_name
			) {
				if(auto lnk = self.lock())
					f(std::move(lnk), {
						{"new_name", std::move(new_name)},
						{"prev_name", std::move(old_name)}
					});
			};

		case Event::StatusChanged :
			return [f = std::move(f_), self = std::weak_ptr{self}](
				a_lnk_status, a_ack, Req request, ReqStatus new_v, ReqStatus prev_v
			) {
				if(auto lnk = self.lock())
					f(std::move(lnk), {
						{"request", prop::integer(new_v)},
						{"new_status", prop::integer(new_v)},
						{"prev_status", prop::integer(prev_v)}
					});
			};

		default: return [] {};
		}
	};

	// make shiny new subscriber actor and place into parent's room
	auto& AS = kernel::config::actor_system();
	auto baby = AS.spawn(listener_actor, pimpl_->self_grp, make_ev_character(shared_from_this(), listen_to, f));
	// register him
	AS.registry().put(baby.id(), baby);
	// and return ID
	return baby.id();
}

auto link::unsubscribe(std::uint64_t event_cb_id) -> void {
	auto& AS = kernel::config::actor_system();
	const auto ev_actor = AS.registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	pimpl_->send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
}

NAMESPACE_END(blue_sky::tree)
