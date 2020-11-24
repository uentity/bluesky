/// @date 05.10.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "map_engine.h"
#include "node_impl.h"

#include <bs/tree/tree.h>
#include <bs/detail/enumops.h>
#include <bs/serialize/propdict.h>
#include <bs/serialize/tree.h>
#include <bs/serialize/cafbind.h>

#include <caf/stateful_actor.hpp>

#include <algorithm>

#define DEBUG_ACTOR 0
#include "actor_debug.h"

NAMESPACE_BEGIN(blue_sky::tree)
NAMESPACE_BEGIN()
///////////////////////////////////////////////////////////////////////////////
//  helper retranslator that forwards acks from input node to parent `map_link`
//
struct iar_state {
	map_link_impl_base::map_actor_type parent;
	node::actor_type input;
};

auto input_ack_retranslator(
	caf::stateful_actor<iar_state>* self, map_link_impl_base::map_actor_type parent, node::actor_type input,
	Event update_on, TreeOpts opts
) {
	using namespace allow_enumops;

	self->state = { std::move(parent), std::move(input) };

	// ignore unexpected messages
	self->set_default_handler([](auto*, auto&) -> caf::result<caf::message> {
		return caf::none;
	});

	const auto send_parent = [self, update_on, opts]
	(Event src_ev, const lid_type& src_id, const caf::actor& origin = {}) {
		// skip unsibscribed events
		if(!enumval(update_on & src_ev)) return;

		const auto& [parent, input] = self->state;
		const auto& src_node = origin ? caf::actor_cast<node::actor_type>(origin) : input;

		// on erase event we don't search for source link
		if(src_ev == Event::LinkErased && (enumval(opts & TreeOpts::Deep) || src_node == input)) {
			self->send(parent, a_ack(), a_node_erase(), src_id);
			return;
		}

		// find source link (depending on deep flag)
		auto notify_parent = [self, &parent = self->state.parent](const link& src_link) {
			if(src_link)
				self->send(parent, a_ack(), a_apply(), src_link);
		};

		if(src_node == input)
			self->request(src_node, caf::infinite, a_node_find(), src_id).then(std::move(notify_parent));
		else if(enumval(opts & TreeOpts::Deep))
			self->request(src_node, caf::infinite, a_node_deep_search(), src_id)
			.then([notify_parent = std::move(notify_parent)](links_v ls) {
				if(!ls.empty()) notify_parent(ls.front());
			});
	};

	// build behavior
	// 1. Base behavior
	auto res = first_then_second( node_impl::leafs_ack_actor_type::behavior_type{

		// Input node direct leafs acks are always retranslated
		// leaf rename
		[=](a_ack, const lid_type& lid, a_lnk_rename, const std::string&, const std::string&)
		{ send_parent(Event::LinkRenamed, lid); },
		// leaf status
		[=](a_ack, const lid_type& lid, a_lnk_status, Req, ReqStatus, ReqStatus)
		{ send_parent(Event::LinkStatusChanged, lid); },
		// leaf data altered by transaction
		[=](a_ack, const lid_type& lid, a_data, const tr_result::box&)
		{ send_parent(Event::DataModified, lid); }

	}, node_impl::self_ack_actor_type::behavior_type{

		// ack on insert
		[=](a_ack, const caf::actor& N, a_node_insert, const lid_type& lid, size_t)
		{ send_parent(Event::LinkInserted, lid, N); },
		// ignore moves within same node
		[=](a_ack, const caf::actor& N, a_node_insert, const lid_type&, size_t, size_t) {},
		// ack on erase
		[=](a_ack, const caf::actor& N, a_node_erase, const lids_v& erased_leafs) {
			if(erased_leafs.empty()) return;
			send_parent(Event::LinkErased, erased_leafs.front(), N);
			if(enumval(opts & TreeOpts::Deep)) {
				std::for_each(erased_leafs.begin()++, erased_leafs.end(), [&](auto& cut_leaf) {
					send_parent(Event::LinkErased, cut_leaf, N);
				});
			}
		}

	}).unbox();

	// 2. Deeper layers - retranslate only if deep processign enabled
	if(enumval(opts & TreeOpts::Deep))
		return first_then_second(std::move(res), link_impl::deep_ack_actor_type::behavior_type{
			// leaf rename
			[=](
				a_ack, const caf::actor& N, const lid_type& lid,
				a_lnk_rename, const std::string&, const std::string&
			) { send_parent(Event::LinkRenamed, lid, N); },
			// leaf status
			[=](a_ack, caf::actor N, const lid_type& lid, a_lnk_status, Req, ReqStatus, ReqStatus)
			{ send_parent(Event::LinkStatusChanged, lid, N); },
			// leaf data altered by transaction
			[=](a_ack, caf::actor N, const lid_type& lid, a_data, const tr_result::box&)
			{ send_parent(Event::DataModified, lid, N); }
		});
	else
		return res;
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  map_link_actor
 *-----------------------------------------------------------------------------*/
map_link_actor::map_link_actor(caf::actor_config& cfg, caf::group self_grp, sp_limpl Limpl) :
	super(cfg, std::move(self_grp), std::move(Limpl))
{
	using namespace allow_enumops;
	const auto& simpl = mimpl();
	// set detached flag if requested
	if(enumval(simpl.opts_ & TreeOpts::DetachedWorkers))
		ropts_.data_node |= ReqOpts::Detached;
	// start input node events tracker
	if(enumval(simpl.update_on_))
		reset_input_listener(simpl.update_on_, simpl.opts_);
}

auto map_link_actor::reset_input_listener(Event update_on, TreeOpts opts) -> void {
	const auto& simpl = mimpl();
	adbg(this) << "starting listener on input node, is_nil = " << simpl.in_.is_nil() << std::endl;

	if(!simpl.in_) return;
	inp_listener_ = spawn_in_group(
		simpl.in_.home(), input_ack_retranslator,
		caf::actor_cast<map_link_impl_base::map_actor_type>(this), simpl.in_.actor(), update_on, opts
	);
}

auto map_link_actor::on_exit() -> void {
	super::on_exit();
	// explicitly terminate listener
	send_exit(inp_listener_, caf::exit_reason::user_shutdown);
}

auto map_link_actor::name() const -> const char* { return "map_link_actor"; }

auto map_link_actor::make_casual_behavior() -> typed_behavior {
	return first_then_second(typed_behavior_overload{

		[=](a_data, bool) -> obj_or_errbox {
			adbg(this) << "<- a_data" << std::endl;
			return unexpected_err_quiet(Error::EmptyData);
		},

		[=](a_data_node, bool) -> node_or_errbox {
			adbg(this) << "<- a_data_node (casual)" << std::endl;
			return mimpl().out_;
		},

		[=](a_lazy, a_node_clear) {
			adbg(this) << "<- lazy clear" << std::endl;
 			become(make_behavior().unbox());
		},

		[=](a_node_clear) -> caf::result<node_or_errbox> {
			adbg(this) << "<- immediate clear" << std::endl;
			become(make_behavior().unbox());
			// imediately invoke DataNode request on refresh behavior
			return delegate(caf::actor_cast<actor_type>(this), a_data_node(), true);
		},

		[=](a_ack, a_apply, const link& inp_lnk) {
			adbg(this) << "<- update (casual)" << std::endl;
			mimpl().update(this, inp_lnk);
		},

		[=](a_ack, a_node_erase, const lid_type& src_id) {
			adbg(this) << "<- erase (casual)" << std::endl;
			mimpl().erase(this, src_id);
		}

	}, super::make_typed_behavior());
}

auto map_link_actor::make_refresh_behavior() -> refresh_behavior_overload {
	// invoke refresh once on DataNode request
	return refresh_behavior_overload{

		[=, casual_bhv = make_casual_behavior().unbox()](a_data_node, bool) mutable
		-> caf::result<node_or_errbox> {
			adbg(this) << "<- a_data_node (refresh)" << std::endl;
			// install casual behavior
			become(casual_bhv);
			// invoke refresh
			return mimpl().refresh(this);
		},

		[=](a_ack, a_apply, const link&) {
			adbg(this) << "<- update (refresh)" << std::endl;
			request(caf::actor_cast<actor_type>(this), caf::infinite, a_data_node(), true)
			.then([](const node_or_errbox&) {});
		},

		[=](a_ack, a_node_erase, const lid_type& src_id) {
			adbg(this) << "<- erase (refresh)" << std::endl;
			request(caf::actor_cast<actor_type>(this), caf::infinite, a_data_node(), true)
			.then([](const node_or_errbox&) {});
		}

	};
}

auto map_link_actor::make_behavior() -> behavior_type {
	return first_then_second(make_refresh_behavior(), make_casual_behavior()).unbox();
}

NAMESPACE_END(blue_sky::tree)
