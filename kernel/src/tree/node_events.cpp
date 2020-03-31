/// @file
/// @author uentity
/// @date 30.03.2020
/// @brief BS tree node events handling
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include "ev_listener_actor.h"
#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky::tree)

auto node::subscribe(handle_event_cb f, Event listen_to) const -> std::uint64_t {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using baby_t = ev_listener_actor<sp_cnode>;

	auto make_ev_character = [N = bs_shared_this<node>(), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed)) {
			static const auto event_impl = [](
				auto& self, auto& wN, auto& lid, auto& new_name, auto& old_name
			) {
				bsout() << "*-* node: fired LinkRenamed event" << bs_end;
				if(auto N = wN.lock())
					self->f(std::move(N), Event::LinkRenamed, {
						{"link_id", to_string(lid)},
						{"new_name", std::move(new_name)},
						{"prev_name", std::move(old_name)}
					});
			};

			// this node leaf was renamed
			res = res.or_else(
				[self, wN = std::weak_ptr{N}] (
					a_ack, const lid_type& lid, a_lnk_rename, std::string new_name, std::string old_name
				) {
					event_impl(self, wN, lid, new_name, old_name);
				}
			);
			// deeper subtree leaf was renamed
			// [TODO] extract correct node from source actor
			res = res.or_else(
				[self, wN = std::weak_ptr{N}] (
					a_ack, const caf::actor& src, const lid_type& lid,
					a_lnk_rename, std::string new_name, std::string old_name
				) {
					event_impl(self, wN, lid, new_name, old_name);
				}
			);
			bsout() << "*-* node: subscribed to LinkRenamed event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkStatusChanged)) {
			static const auto event_impl = [](
				auto& self, auto& wN, auto& lid, auto req, auto new_s, auto prev_s
			) {
				bsout() << "*-* node: fired LinkStatusChanged event" << bs_end;
				if(auto N = wN.lock())
					self->f(std::move(N), Event::LinkStatusChanged, {
						{"link_id", to_string(lid)},
						{"request", prop::integer(req)},
						{"new_status", prop::integer(new_s)},
						{"prev_status", prop::integer(prev_s)}
					});
			};

			// this node leaf status changed
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_ack, const lid_type& lid,
					a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					event_impl(self, wN, lid, req, new_s, prev_s);
				}
			);
			// deeper subtree leaf status changed
			// [TODO] extract correct node from source actor
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_ack, const caf::actor& src, const lid_type& lid,
					a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					event_impl(self, wN, lid, req, new_s, prev_s);
				}
			);
			bsout() << "*-* node: subscribed to LinkStatusChanged event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkInserted)) {
			res = res.or_else(
				// insert
				[self, wN = std::weak_ptr{N}](
					a_ack, const caf::actor& src, a_node_insert,
					const lid_type& lid, std::size_t pos, InsertPolicy pol
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->f(std::move(N), Event::LinkInserted, {
							{"link_id", to_string(lid)},
							{"pos", (prop::integer)pos}
						});
				},
				// move
				[self, wN = std::weak_ptr{N}](
					a_ack, const caf::actor& src, a_node_insert,
					const lid_type& lid, std::size_t to_idx, std::size_t from_idx
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->f(std::move(N), Event::LinkInserted, {
							{"link_id", to_string(lid)},
							{"to_idx", (prop::integer)to_idx},
							{"from_idx", (prop::integer)from_idx}
						});
				}
			);
			bsout() << "*-* node: subscribed to LinkInserted event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkErased)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_ack, const caf::actor& src, a_node_erase, const lids_v& lids
				) {
					bsout() << "*-* node: fired LinkErased event" << bs_end;
					auto N = wN.lock();
					if(!N) return;

					// convert link IDs to strings
					std::vector<std::string> slids(lids.size());
					std::transform(
						lids.begin(), lids.end(), slids.begin(),
						[](const auto& lid) { return to_string(lid); }
					);

					self->f(std::move(N), Event::LinkErased, {
						{"lids", std::move(slids)}
					});
				}
			);
			bsout() << "*-* node: subscribed to LinkErased event" << bs_end;
		}

		return res;
	};

	// make shiny new subscriber actor and place into parent's room
	auto baby = system().spawn_in_group<baby_t>(
		pimpl_->home_, pimpl_->home_, std::move(f), std::move(make_ev_character)
	);
	// and return ID
	return baby.id();
}

auto node::unsubscribe(std::uint64_t event_cb_id) -> void {
	kernel::radio::bye_actor(event_cb_id);
}

NAMESPACE_END(blue_sky::tree)
