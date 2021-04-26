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

NAMESPACE_BEGIN(blue_sky::tree)
using event_handler = node::event_handler;

static auto make_listener(const node& self, event_handler f, Event listen_to) {
	using namespace kernel::radio;
	using namespace allow_enumops;
	using baby_t = ev_listener_actor<node>;

	static const auto handler_impl = [](
		baby_t* self, auto& weak_root, caf::actor origin, Event ev, prop::propdict params
	) {
		if(auto r = weak_root.lock()) {
			if(!origin) origin = caf::actor_cast<caf::actor>(r.actor());
			self->f(std::move(r), {std::move(origin), std::move(params), ev});
		}
		else // if root source is dead, quit
			self->quit();
	};

	auto make_ev_character = [weak_root = node::weak_ptr(self), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed)) {
			const auto renamed_impl = [=](
				caf::actor origin, auto& lid, auto& new_name, auto& old_name
			) {
				handler_impl(self, weak_root, std::move(origin), Event::LinkRenamed, {
					{"link_id", lid},
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
					a_ack, caf::actor src, const lid_type& lid,
					a_lnk_rename, std::string new_name, std::string old_name
				) {
					renamed_impl(std::move(src), lid, new_name, old_name);
				}
			);
		}

		if(enumval(listen_to & Event::LinkStatusChanged)) {
			const auto status_impl = [=](
				caf::actor origin, auto& lid, auto req, auto new_s, auto prev_s
			) {
				//bsout() << "*-* node: fired LinkStatusChanged event" << bs_end;
				handler_impl(self, weak_root, std::move(origin), Event::LinkStatusChanged, {
					{"link_id", lid},
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
					a_ack, caf::actor src, const lid_type& lid,
					a_lnk_status, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					status_impl(std::move(src), lid, req, new_s, prev_s);
				}
			);
		}

		if(enumval(listen_to & Event::DataModified)) {
			const auto datamod_impl = [=](
				caf::actor origin, auto& lid, tr_result::box&& tres_box
			) {
				//bsout() << "*-* node: fired DataModified event" << bs_end;
				auto params = prop::propdict{{ "link_id", lid }};
				if(auto tres = tr_result{std::move(tres_box)})
					params.merge_props(extract_info(std::move(tres)));
				else
					params["error"] = to_string(extract_err(std::move(tres)));
				handler_impl(self, weak_root, std::move(origin), Event::DataModified, std::move(params));
			};

			// this node leaf data changed
			res = res.or_else(
				[=](a_ack, const lid_type& lid, a_data, tr_result::box trbox) {
					datamod_impl({}, lid, std::move(trbox));
				}
			);
			// deeper subtree leaf status changed
			res = res.or_else(
				[=](a_ack, caf::actor src, const lid_type& lid, a_data, tr_result::box trbox) {
					datamod_impl(std::move(src), lid, std::move(trbox));
				}
			);
		}

		if(enumval(listen_to & Event::LinkInserted)) {
			res = res.or_else(
				// insert
				[=](
					a_ack, caf::actor src, a_node_insert,
					const lid_type& lid, std::size_t pos
				) {
					//bsout() << "*-* node: fired LinkInserted event" << bs_end;
					handler_impl(self, weak_root, std::move(src), Event::LinkInserted, {
						{"link_id", lid},
						{"pos", (prop::integer)pos}
					});
				},
				// move
				[=](
					a_ack, caf::actor src, a_node_insert,
					const lid_type& lid, std::size_t to_idx, std::size_t from_idx
				) {
					//bsout() << "*-* node: fired LinkInserted event (move)" << bs_end;
					handler_impl(self, weak_root, std::move(src), Event::LinkInserted, {
						{"link_id", lid},
						{"to_idx", (prop::integer)to_idx},
						{"from_idx", (prop::integer)from_idx}
					});
				}
			);
		}

		if(enumval(listen_to & Event::LinkErased)) {
			res = res.or_else(
				[=](
					a_ack, caf::actor src, a_node_erase, lids_v lids
				) {
					//bsout() << "*-* node: fired LinkErased event" << bs_end;
					auto context = prop::propdict{};
					if(!lids.empty()) {
						context["link_id"] = lids[0];
						context["lids"] = std::move(lids);
					}
					handler_impl(self, weak_root, std::move(src), Event::LinkErased, std::move(context));
				}
			);
		}

		return res;
	};

	// make shiny new subscriber actor and place into parent's room
	return system().spawn<baby_t, caf::lazy_init>(
		self.actor().address(), std::move(f), std::move(make_ev_character)
	);
}

auto node::subscribe(event_handler f, Event listen_to) const -> std::uint64_t {
	// ensure it has started & properly initialized
	// throw exception otherwise
	if(auto res = actorf<std::uint64_t>(
		pimpl()->actor(*this), infinite, a_subscribe(),
		make_listener(*this, std::move(f), listen_to)
	))
		return *res;
	else
		throw res.error();
}

auto node::subscribe(launch_async_t, event_handler f, Event listen_to) const -> std::uint64_t {
	auto baby = make_listener(*this, std::move(f), listen_to);
	auto baby_id = baby.id();
	caf::anon_send(pimpl()->actor(*this), a_subscribe{}, std::move(baby));
	return baby_id;
}

auto node::unsubscribe(deep_t) const -> void {
	if(is_nil()) return;
	unsubscribe();
	apply(launch_async, [](bare_node self) {
		for(const auto& L : self.leafs())
			L.unsubscribe(deep);
		return perfect;
	});
}

NAMESPACE_END(blue_sky::tree)
