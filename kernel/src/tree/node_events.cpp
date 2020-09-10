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
#include "../serialize/tree_impl.h"

#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/serialize/cafbind.h>
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky::tree)

auto node::subscribe(event_handler f, Event listen_to) const -> std::uint64_t {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using baby_t = ev_listener_actor<node>;

	static const auto handler_impl = [](
		baby_t* self, auto& weak_root, const caf::actor& subn_actor, Event ev, prop::propdict params
	) {
		if(auto r = weak_root.lock()) {
			// reconstruct subnode via node impl
			auto subnode = node::nil();
			if(subn_actor) {
				actorf<sp_nimpl>(subn_actor, kernel::radio::timeout(), a_impl())
				.map([&](const sp_nimpl& subn_impl) { subnode = subn_impl->super_engine(); });
			}
			// invoke callback
			self->f(std::move(r), std::move(subnode), ev, std::move(params));
		}
		else // if root source is dead, quit
			self->quit();
	};

	auto make_ev_character = [weak_root = weak_ptr(*this), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed)) {
			const auto renamed_impl = [=](
				const caf::actor& subn_actor, auto& lid, auto& new_name, auto& old_name
			) {
				bsout() << "*-* node: fired LinkRenamed event" << bs_end;
				handler_impl(self, weak_root, subn_actor, Event::LinkRenamed, {
					{"link_id", to_string(lid)},
					{"new_name", std::move(new_name)},
					{"prev_name", std::move(old_name)}
				});
			};

			// this node leaf was renamed
			res = res.or_else(
				[=] (
					a_ack, const lid_type& lid, a_lnk_rename, std::string new_name, std::string old_name
				) {
					renamed_impl({}, lid, new_name, old_name);
				}
			);
			// deeper subtree leaf was renamed
			res = res.or_else(
				[=] (
					a_ack, const caf::actor& src, const lid_type& lid,
					a_lnk_rename, std::string new_name, std::string old_name
				) {
					renamed_impl(src, lid, new_name, old_name);
				}
			);
			bsout() << "*-* node: subscribed to LinkRenamed event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkStatusChanged)) {
			const auto status_impl = [=](
				const caf::actor& subn_actor, auto& lid, auto req, auto new_s, auto prev_s
			) {
				bsout() << "*-* node: fired LinkStatusChanged event" << bs_end;
				handler_impl(self, weak_root, subn_actor, Event::LinkStatusChanged, {
					{"link_id", to_string(lid)},
					{"request", prop::integer(req)},
					{"new_status", prop::integer(new_s)},
					{"prev_status", prop::integer(prev_s)}
				});
			};

			// this node leaf status changed
			res = res.or_else(
				[=](
					a_ack, const lid_type& lid,
					a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					status_impl({}, lid, req, new_s, prev_s);
				}
			);
			// deeper subtree leaf status changed
			res = res.or_else(
				[=](
					a_ack, const caf::actor& src, const lid_type& lid,
					a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					status_impl(src, lid, req, new_s, prev_s);
				}
			);
			bsout() << "*-* node: subscribed to LinkStatusChanged event" << bs_end;
		}

		if(enumval(listen_to & Event::DataModified)) {
			const auto datamod_impl = [=](
				const caf::actor& subn_actor, auto& lid, tr_result::box&& tres_box
			) {
				bsout() << "*-* node: fired DataModified event" << bs_end;
				auto params = prop::propdict{{ "link_id", to_string(lid) }};
				if(auto tres = tr_result{std::move(tres_box)})
					params = extract_info(std::move(tres));
				else
					params["error"] = to_string(extract_err(std::move(tres)));
				handler_impl(self, weak_root, subn_actor, Event::DataModified, std::move(params));
			};

			// this node leaf data changed
			res = res.or_else(
				[=](a_ack, const lid_type& lid, a_data, tr_result::box trbox) {
					datamod_impl({}, lid, std::move(trbox));
				}
			);
			// deeper subtree leaf status changed
			res = res.or_else(
				[=](a_ack, const caf::actor& src, const lid_type& lid, a_data, tr_result::box trbox) {
					datamod_impl(src, lid, std::move(trbox));
				}
			);
			bsout() << "*-* node: subscribed to DataModified event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkInserted)) {
			res = res.or_else(
				// insert
				[=](
					a_ack, const caf::actor& src, a_node_insert,
					const lid_type& lid, std::size_t pos, InsertPolicy pol
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					handler_impl(self, weak_root, src, Event::LinkInserted, {
						{"link_id", to_string(lid)},
						{"pos", (prop::integer)pos}
					});
				},
				// move
				[=](
					a_ack, const caf::actor& src, a_node_insert,
					const lid_type& lid, std::size_t to_idx, std::size_t from_idx
				) {
					bsout() << "*-* node: fired LinkInserted event (move)" << bs_end;
					handler_impl(self, weak_root, src, Event::LinkInserted, {
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
				[=](
					a_ack, const caf::actor& src, a_node_erase, const lids_v& lids
				) {
					bsout() << "*-* node: fired LinkErased event" << bs_end;

					// convert link IDs to strings
					std::vector<std::string> slids(lids.size());
					std::transform(
						lids.begin(), lids.end(), slids.begin(),
						[](const auto& lid) { return to_string(lid); }
					);

					handler_impl(self, weak_root, src, Event::LinkErased, {
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
		pimpl()->home, pimpl()->home, std::move(f), std::move(make_ev_character)
	);
	// and return ID
	return baby.id();
}

auto node::unsubscribe(std::uint64_t event_cb_id) -> void {
	kernel::radio::bye_actor(event_cb_id);
}

NAMESPACE_END(blue_sky::tree)
