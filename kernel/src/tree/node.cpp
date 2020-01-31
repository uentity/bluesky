/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include "ev_listener_actor.h"
#include <bs/kernel/types_factory.h>
#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/tree/tree.h>
#include <bs/serialize/cafbind.h>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/random_generator.hpp>

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
node::node(bool start_actor, std::string custom_id)
	: objbase(true, std::move(custom_id)), pimpl_(std::make_shared<node_impl>(this))
{
	if(start_actor) start_engine();
}

node::node(std::string custom_id)
	: node(true, std::move(custom_id))
{}

node::node(const node& src)
	: objbase(src), pimpl_(std::make_shared<node_impl>(*src.pimpl_, this))
{
	start_engine();
}

node::~node() {
	// mark self as deleted for impl & actor
	pimpl_->super_ = nullptr;
	// then kill self actor
	caf::anon_send_exit(actor_, caf::exit_reason::user_shutdown);
}

auto node::start_engine(const std::string& gid) -> bool {
	static auto uuid_gen = boost::uuids::random_generator{};
	if(!actor_) {
		// generate random UUID for actor groud if passed GID is empty
		actor_ = pimpl_->spawn_actor(pimpl_, gid.empty() ? to_string(uuid_gen()) : gid);
		return true;
	}
	return false;
}

auto node::gid() const -> std::string { return pimpl_->gid(); }

auto node::disconnect(bool deep) -> void {
	// disconnect self
	caf::anon_send(actor_, a_node_disconnect());

	if(deep) {
		// don't follow symlinks & lazy links
		walk(bs_shared_this<node>(), [](const sp_node&, std::list<sp_node>& subnodes, std::vector<sp_link>&) {
			for(const auto& N : subnodes)
				N->disconnect(false);
		}, true, false, false);
	}
}

auto node::propagate_owner(bool deep) -> void {
	caf::anon_send(actor_, a_node_propagate_owner(), deep);
}

auto node::handle() const -> sp_link {
	auto guard = pimpl_->lock<node_impl::Metadata>(shared);
	return pimpl_->handle_.lock();
}

auto node::set_handle(const sp_link& handle) -> void {
	pimpl_->set_handle(handle);
}

///////////////////////////////////////////////////////////////////////////////
//  leafs container
//
auto node::size() const -> std::size_t {
	return pimpl_->size();
}

auto node::empty() const -> bool {
	return pimpl_->links_.empty();
}

auto node::leafs(Key order) const -> links_v {
	return pimpl_->actorf<links_v>(
		*this, a_node_leafs(), order
	).value_or(links_v{});
}

///////////////////////////////////////////////////////////////////////////////
//  keys
//
auto node::keys(Key ordering) const -> lids_v {
	return node_impl::keys<Key::ID>(leafs(ordering));
}

auto node::skeys(Key key_meaning, Key ordering) const -> std::vector<std::string> {
	static const auto slids = [](const links_v& Ls) {
		return range_t{ Ls.begin(), Ls.end() }.extract<std::string>(
			[](const auto& L) { return to_string(L->id()); }
		);
	};

	auto Ls = leafs(ordering);
	switch(key_meaning) {
	default:
	case Key::ID:
		return slids(Ls);
	case Key::Name:
		return node_impl::keys<Key::Name>(Ls);
	case Key::OID:
		return node_impl::keys<Key::OID>(Ls);
	case Key::Type:
		return node_impl::keys<Key::Type>(Ls);
	case Key::AnyOrder:
		return {};
	}
}

///////////////////////////////////////////////////////////////////////////////
//  find
//
auto node::find(std::size_t idx) const -> sp_link {
	return pimpl_->actorf<sp_link>(
		*this, a_node_find(), idx
	).value_or(nullptr);
}

auto node::find(lid_type id) const -> sp_link {
	return pimpl_->actorf<sp_link>(
		*this, a_node_find(), std::move(id)
	).value_or(nullptr);
}

auto node::find(std::string key, Key key_meaning) const -> sp_link {
	return pimpl_->actorf<sp_link>(
		*this, a_node_find(), std::move(key), key_meaning
	).value_or(nullptr);
}

// ---- deep_search
auto node::deep_search(lid_type id) const -> sp_link {
	return pimpl_->actorf<sp_link>(
		*this, a_node_deep_search(), std::move(id)
	).value_or(nullptr);
}

auto node::deep_search(std::string key, Key key_meaning) const -> sp_link {
	return pimpl_->actorf<sp_link>(
		*this, a_node_deep_search(), std::move(key), key_meaning
	).value_or(nullptr);
}

// ---- index
auto node::index(lid_type lid) const -> existing_index {
	return pimpl_->actorf<existing_index>(
		*this, a_node_index(), std::move(lid)
	).value_or(existing_index{});
}

auto node::index(std::string key, Key key_meaning) const -> existing_index {
	return pimpl_->actorf<existing_index>(
		*this, a_node_index(), std::move(key), key_meaning
	).value_or(existing_index{});
}

// ---- equal_range
auto node::equal_range(std::string key, Key key_meaning) const -> links_v {
	return pimpl_->actorf<links_v>(
		*this, a_node_equal_range(), std::move(key), key_meaning
	).value_or(links_v{});
}

///////////////////////////////////////////////////////////////////////////////
//  insert
//
auto node::insert(sp_link l, InsertPolicy pol) -> insert_status {
	return pimpl_->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(sp_link l, std::size_t idx, InsertPolicy pol) -> insert_status {
	return pimpl_->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), idx, pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(links_v ls, InsertPolicy pol) -> std::size_t {
	return pimpl_->actorf<std::size_t>(
		*this, a_node_insert(), std::move(ls), pol
	).value_or(0);
}

auto node::insert(std::string name, sp_obj obj, InsertPolicy pol) -> insert_status {
	return insert(
		std::make_shared<hard_link>(std::move(name), std::move(obj)), pol
	);
}

///////////////////////////////////////////////////////////////////////////////
//  erase
//
auto node::erase(std::size_t idx) -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), idx
	).value_or(0);
}

auto node::erase(lid_type lid) -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), std::move(lid), EraseOpts::Normal
	).value_or(0);
}

auto node::erase(std::string key, Key key_meaning) -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), std::move(key), key_meaning
	).value_or(0);
}

auto node::clear() -> void {
	caf::anon_send(actor_, a_node_clear());
}

///////////////////////////////////////////////////////////////////////////////
//  rename
//
auto node::rename(std::size_t idx, std::string new_name) -> bool {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), idx, std::move(new_name)
	).value_or(false);
}

auto node::rename(lid_type lid, std::string new_name) -> bool {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(lid), std::move(new_name)
	).value_or(false);
}

auto node::rename(std::string old_name, std::string new_name) -> std::size_t {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(old_name), std::move(new_name)
	).value_or(false);
}

///////////////////////////////////////////////////////////////////////////////
//  rearrrange
//
auto node::rearrange(lids_v new_order) -> void {
	caf::anon_send(actor(), a_node_rearrange(), std::move(new_order));
}

auto node::rearrange(std::vector<std::size_t> new_order) -> void {
	caf::anon_send(actor(), a_node_rearrange(), std::move(new_order));
}

///////////////////////////////////////////////////////////////////////////////
//  events handling
//
auto node::subscribe(handle_event_cb f, Event listen_to) -> std::uint64_t {
	using baby_t = ev_listener_actor<sp_node>;

	auto make_ev_character = [N = bs_shared_this<node>(), listen_to](baby_t* self) {
		auto res = caf::message_handler{};
		if(enumval(listen_to & Event::LinkRenamed)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}] (
					a_ack, a_lnk_rename, const lid_type& lid, std::string new_name, std::string old_name
				) {
					bsout() << "*-* node: fired LinkRenamed event" << bs_end;
					if(auto N = wN.lock())
						self->f(std::move(N), Event::LinkRenamed, {
							{"link_id", to_string(lid)},
							{"new_name", std::move(new_name)},
							{"prev_name", std::move(old_name)}
						});
				}
			);
			bsout() << "*-* node: subscribed to LinkRenamed event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkStatusChanged)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_ack, a_lnk_status, const lid_type& lid, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					bsout() << "*-* node: fired LinkStatusChanged event" << bs_end;
					if(auto N = wN.lock())
						self->f(std::move(N), Event::LinkStatusChanged, {
							{"link_id", to_string(lid)},
							{"request", prop::integer(req)},
							{"new_status", prop::integer(new_s)},
							{"prev_status", prop::integer(prev_s)}
						});
				}
			);
			bsout() << "*-* node: subscribed to LinkStatusChanged event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkInserted)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_ack, a_node_insert, const lid_type& lid, std::size_t pos, InsertPolicy pol
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->f(std::move(N), Event::LinkInserted, {
							{"link_id", to_string(lid)},
							{"pos", (prop::integer)pos}
						});
				},

				[self, wN = std::weak_ptr{N}](
					a_ack, a_node_insert, const lid_type& lid, std::size_t to_idx, std::size_t from_idx
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
					a_ack, a_node_erase, const lids_v& lids, std::vector<std::string>& oids
				) {
					bsout() << "*-* node: fired LinkErased event" << bs_end;
					auto N = wN.lock();
					if(!N) return;

					auto killed = prop::propdict{};
					killed["oids"] = std::move(oids);
					// convert link IDs to strings
					std::vector<std::string> slids(lids.size());
					std::transform(
						lids.begin(), lids.end(), slids.begin(),
						[](const lid_type& lid) { return to_string(lid); }
					);

					self->f(std::move(N), Event::LinkErased, killed);
				}
			);
			bsout() << "*-* node: subscribed to LinkErased event" << bs_end;
		}

		return res;
	};

	// make shiny new subscriber actor and place into parent's room
	auto baby = kernel::radio::system().spawn_in_group<baby_t>(
		pimpl_->self_grp, pimpl_->self_grp, std::move(f), std::move(make_ev_character)
	);
	// and return ID
	return baby.id();
}

auto node::unsubscribe(std::uint64_t event_cb_id) -> void {
	auto& AS = kernel::radio::system();
	const auto ev_actor = AS.registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	caf::anon_send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
BS_TYPE_IMPL(node, objbase, "node", "BS tree node", true, true);
BS_TYPE_ADD_CONSTRUCTOR(node, (std::string))
BS_REGISTER_TYPE("kernel", node)

NAMESPACE_END(tree)

namespace detail {

BS_API void adjust_cloned_node(const sp_obj& pnode) {
	// fix owner in node's clone
	if(pnode->is_node())
		std::static_pointer_cast<tree::node>(pnode)->propagate_owner();
}

} /* namespace detail */

NAMESPACE_END(blue_sky)

