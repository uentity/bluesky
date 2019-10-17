/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include <bs/kernel/types_factory.h>
#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/tree/tree.h>

#include <boost/uuid/string_generator.hpp>

OMIT_OBJ_SERIALIZATION

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

NAMESPACE_BEGIN() // hidden namespace

static boost::uuids::string_generator uuid_from_str;

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
node::node(std::string custom_id)
	: objbase(true, custom_id),
	aimpl_(kernel::radio::system().spawn<node_actor>(id_))
{
	pimpl_ = caf::actor_cast<node_actor*>(aimpl_);
	if(!pimpl_) throw error{ "Trying to construct tree::node with invalid actor" };
	fimpl_ = caf::make_function_view(aimpl_, caf::duration{pimpl_->timeout_});
}

node::node(const node& src)
	: objbase(src),
	aimpl_(kernel::radio::system().spawn<node_actor>(id_, src.aimpl_))
{
	pimpl_ = caf::actor_cast<node_actor*>(aimpl_);
	if(!pimpl_) throw error{ "Trying to copy construct tree::node with invalid actor" };
	fimpl_ = caf::make_function_view(aimpl_, caf::duration{pimpl_->timeout_});
}

node::~node() {
	pimpl_->goodbye();
}

auto node::disconnect(bool deep) -> void {
	if(deep) {
		// don't follow symlinks & lazy links
		walk(bs_shared_this<node>(), [](const sp_node&, std::list<sp_node>& subnodes, std::vector<sp_link>&) {
			for(const auto& N : subnodes)
				N->disconnect(false);
		}, true, false, false);
	}

	// disconnect self
	pimpl_->disconnect();
}

void node::propagate_owner(bool deep) {
	auto solo = std::lock_guard{ pimpl_->links_guard_ };
	// properly setup owner in node's leafs
	const auto self = bs_shared_this<node>();
	sp_node child_node;
	for(auto& plink : pimpl_->links_) {
		child_node = node_actor::adjust_inserted_link(plink, self);
		if(deep && child_node)
			child_node->propagate_owner(true);
	}
}

sp_link node::handle() const {
	return pimpl_->handle_.lock();
}

bool node::accepts(const sp_link& what) const {
	return pimpl_->accepts(what);
}

void node::accept_object_types(std::vector<std::string> allowed_types) {
	pimpl_->allowed_otypes_ = std::move(allowed_types);
}

std::vector<std::string> node::allowed_object_types() const {
	return pimpl_->allowed_otypes_;
}

void node::set_handle(const sp_link& handle) {
	pimpl_->set_handle(handle);
}

//sp_link node::create_self_link(std::string name, bool force) {
//	if(!force && !pimpl_->handle_.expired()) {
//		return pimpl_->handle_.lock();
//	}
//	// create new self link
//	auto root_lnk = std::make_shared<hard_link>(std::move(name), bs_shared_this<node>());
//	pimpl_->set_handle(root_lnk);
//	return root_lnk;
//}

///////////////////////////////////////////////////////////////////////////////
//  leafs container
//
std::size_t node::size() const {
	return pimpl_->links_.size();
}

bool node::empty() const {
	return pimpl_->links_.empty();
}

iterator<Key::AnyOrder> node::begin(Key_const<Key::AnyOrder>) const {
	return pimpl_->begin<>();
}

iterator<Key::AnyOrder> node::end(Key_const<Key::AnyOrder>) const {
	return pimpl_->end<>();
}

iterator<Key::ID> node::begin(Key_const<Key::ID>) const {
	return pimpl_->begin<Key::ID>();
}

iterator<Key::ID> node::end(Key_const<Key::ID>) const {
	return pimpl_->end<Key::ID>();
}

iterator<Key::Name> node::begin(Key_const<Key::Name>) const {
	return pimpl_->begin<Key::Name>();
}

iterator<Key::Name> node::end(Key_const<Key::Name>) const {
	return pimpl_->end<Key::Name>();
}

iterator<Key::OID> node::begin(Key_const<Key::OID>) const {
	return pimpl_->begin<Key::OID>();
}

iterator<Key::OID> node::end(Key_const<Key::OID>) const {
	return pimpl_->end<Key::OID>();
}

iterator<Key::Type> node::begin(Key_const<Key::Type>) const {
	return pimpl_->begin<Key::Type>();
}

iterator<Key::Type> node::end(Key_const<Key::Type>) const {
	return pimpl_->end<Key::Type>();
}

///////////////////////////////////////////////////////////////////////////////
//  find
//
iterator<Key::AnyOrder> node::find(const std::size_t idx) const {
	auto i = begin();
	std::advance(i, idx);
	return i;
}

iterator<Key::AnyOrder> node::find(const id_type& id) const {
	return pimpl_->find<Key::ID>(id);
}

iterator<Key::AnyOrder> node::find(const std::string& key, Key key_meaning) const {
	switch(key_meaning) {
	case Key::ID:
		return pimpl_->find<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return pimpl_->find<Key::OID>(key);
	case Key::Type:
		return pimpl_->find<Key::Type>(key);
	default:
	case Key::Name:
		return pimpl_->find<Key::Name>(key);
	}
}

// ---- index
std::size_t node::index(const id_type& lid) const {
	auto i = pimpl_->find<Key::ID, Key::AnyOrder>(lid);
	return std::distance(begin<Key::AnyOrder>(), i);
}

std::size_t node::index(const iterator<Key::AnyOrder>& pos) const {
	return std::distance(begin<Key::AnyOrder>(), pos);
}

std::size_t node::index(const std::string& key, Key key_meaning) const {
	auto i = find(key, key_meaning);
	return std::distance(begin<Key::AnyOrder>(), i);
}

// ---- equal_range
range<Key::Name> node::equal_range(const std::string& link_name) const {
	return pimpl_->equal_range<Key::Name>(link_name);
}

range<Key::OID> node::equal_range_oid(const std::string& oid) const {
	return pimpl_->equal_range<Key::OID>(oid);
}

range<Key::Type> node::equal_type(const std::string& type_id) const {
	return pimpl_->equal_range<Key::Type>(type_id);
}

///////////////////////////////////////////////////////////////////////////////
//  insert
//
insert_status<Key::ID> node::insert(sp_link l, InsertPolicy pol) {
	auto self = bs_shared_this<node>();
	auto res = pimpl_->insert(l, pol, self);
	if(res.second) {
		// inserted link postprocessing
		node_actor::adjust_inserted_link(*res.first, self);
	}
	else if(enumval(pol & InsertPolicy::Merge) && res.first != end<Key::ID>()) {
		// check if we need to deep merge given links
		// go one step down the hierarchy
		auto src_node = l->data_node();
		auto dst_node = (*res.first)->data_node();
		if(src_node && dst_node) {
			// insert all links from source node into destination
			dst_node->insert(
				std::vector<sp_link>(src_node->begin(), src_node->end()), pol
			);
		}
	}
	return res;
}

insert_status<Key::AnyOrder> node::insert(sp_link l, iterator<> pos, InsertPolicy pol) {
	// 1. insert an element using ID index
	auto res = insert(std::move(l), pol);
	if(res.first != end<Key::ID>()) {
		// 2. reposition an element in AnyOrder index
		auto solo = std::lock_guard{ pimpl_->links_guard_ };
		auto src = pimpl_->project<Key::ID>(res.first);
		if(pos != src) {
			auto& ord_idx = pimpl_->links_.get<Key_tag<Key::AnyOrder>>();
			ord_idx.relocate(pos, src);
			return {src, true};
		}
	}
	return {pimpl_->project<Key::ID>(res.first), res.second};
}

insert_status<Key::AnyOrder> node::insert(sp_link l, std::size_t idx, InsertPolicy pol) {
	return insert(std::move(l), std::next(begin<Key::AnyOrder>(), std::min(idx, size())), pol);
}

insert_status<Key::ID> node::insert(std::string name, sp_obj obj, InsertPolicy pol) {
	return insert(
		std::make_shared<hard_link>(std::move(name), std::move(obj)), pol
	);
}

///////////////////////////////////////////////////////////////////////////////
//  erase
//
void node::erase(const std::size_t idx) {
	auto solo = std::lock_guard{ pimpl_->links_guard_ };
	pimpl_->links_.get<Key_tag<Key::AnyOrder>>().erase(find(idx));
}

void node::erase(const id_type& lid) {
	pimpl_->erase<>(lid);
}

void node::erase(const std::string& key, Key key_meaning) {
	switch(key_meaning) {
	case Key::ID:
		pimpl_->erase<Key::ID>(uuid_from_str(key));
		break;
	case Key::OID:
		pimpl_->erase<Key::OID>(key);
		break;
	case Key::Type:
		pimpl_->erase<Key::Type>(key);
		break;
	default:
	case Key::Name:
		return pimpl_->erase<Key::Name>(key);
	}
}

void node::erase(const range<Key::ID>& r) {
	pimpl_->erase<>(r);
}

void node::erase(const range<Key::Name>& r) {
	pimpl_->erase<Key::Name>(r);
}

void node::erase(const range<Key::OID>& r) {
	pimpl_->erase<Key::OID>(r);
}

void node::clear() {
	auto solo = std::lock_guard{ pimpl_->links_guard_ };
	pimpl_->links_.clear();
}

///////////////////////////////////////////////////////////////////////////////
//  deep_search
//
sp_link node::deep_search(const id_type& id) const {
	return pimpl_->deep_search<>(id);
}

sp_link node::deep_search(const std::string& key, Key key_meaning) const {
	switch(key_meaning) {
	case Key::ID:
		return pimpl_->deep_search<Key::ID>(uuid_from_str(key));
	case Key::OID:
		return pimpl_->deep_search<Key::OID>(key);
	case Key::Type:
		return pimpl_->deep_search<Key::Type>(key);
	default:
	case Key::Name:
		return pimpl_->deep_search<Key::Name>(key);
	}
}

///////////////////////////////////////////////////////////////////////////////
//  keys
//
std::vector<Key_type<Key::ID>> node::keys(Key_const<Key::ID>) const {
	return pimpl_->keys<Key::ID>();
}

std::vector<Key_type<Key::Name>> node::keys(Key_const<Key::Name>) const {
	return pimpl_->keys<Key::Name>();
}

std::vector<Key_type<Key::OID>> node::keys(Key_const<Key::OID>) const {
	return pimpl_->keys<Key::OID>();
}

std::vector<Key_type<Key::Type>> node::keys(Key_const<Key::Type>) const {
	return pimpl_->keys<Key::Type>();
}

///////////////////////////////////////////////////////////////////////////////
//  rename
//
bool node::rename(iterator<Key::AnyOrder> pos, std::string new_name) {
	return pimpl_->rename<Key::AnyOrder>(std::move(pos), std::move(new_name));
}

bool node::rename(const std::size_t idx, std::string new_name) {
	return rename(find(idx), std::move(new_name));
}

bool node::rename(const id_type& lid, std::string new_name) {
	return pimpl_->rename<Key::ID>(lid, std::move(new_name)) > 0;
}

std::size_t node::rename(const std::string& key, std::string new_name, Key key_meaning, bool all) {
	switch(key_meaning) {
	default:
	case Key::ID:
		return pimpl_->rename<Key::ID>(uuid_from_str(key), std::move(new_name), all);
	case Key::OID:
		return pimpl_->rename<Key::OID>(key, std::move(new_name), all);
	case Key::Type:
		return pimpl_->rename<Key::Type>(key, std::move(new_name), all);
	case Key::Name:
		return pimpl_->rename<Key::Name>(key, std::move(new_name), all);
	}
}

auto node::on_rename(const id_type& renamed_id) const -> void {
	pimpl_->on_rename(renamed_id);
}

///////////////////////////////////////////////////////////////////////////////
//  project
//
iterator<Key::AnyOrder> node::project(iterator<Key::ID> src) const {
	return pimpl_->project<Key::ID>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::Name> src) const {
	return pimpl_->project<Key::Name>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::OID> src) const {
	return pimpl_->project<Key::OID>(std::move(src));
}

iterator<Key::AnyOrder> node::project(iterator<Key::Type> src) const {
	return pimpl_->project<Key::Type>(std::move(src));
}

///////////////////////////////////////////////////////////////////////////////
//  events handling
//
auto node::subscribe(handle_event_cb f, Event listen_to) -> std::uint64_t {
	struct ev_state { handle_event_cb f; };

	auto make_ev_character = [N = bs_shared_this<node>(), listen_to, f = std::move(f)](
		caf::stateful_actor<ev_state>* self
	) {
		auto res = caf::message_handler{};
		self->state.f = std::move(f);

		if(enumval(listen_to & Event::LinkRenamed)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}] (
					a_lnk_rename, a_ack, const link::id_type& lid, std::string new_name, std::string old_name
				) {
					bsout() << "*-* node: fired LinkRenamed event" << bs_end;
					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkRenamed, {
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
					a_lnk_status, a_ack, link::id_type lid, Req req, ReqStatus new_s, ReqStatus prev_s
				) {
					bsout() << "*-* node: fired LinkStatusChanged event" << bs_end;
					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkStatusChanged, {
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
					a_lnk_insert, a_ack, link::id_type lid
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkInserted, {
							{"link_id", to_string(lid)}
						});
				}
			);
			bsout() << "*-* node: subscribed to LinkInserted event" << bs_end;
		}

		if(enumval(listen_to & Event::LinkInserted)) {
			res = res.or_else(
				[self, wN = std::weak_ptr{N}](
					a_lnk_erase, a_ack, const std::vector<std::pair<link::id_type, std::string>>& zombies
				) {
					bsout() << "*-* node: fired LinkErased event" << bs_end;
					auto killed = prop::propdict{};
					for(const auto& [z_id, z_oid] : zombies)
						killed[to_string(z_id)] = z_oid;

					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkErased, killed);
				}
			);
			bsout() << "*-* node: subscribed to LinkErased event" << bs_end;
		}

		return res;
	};

	// make shiny new subscriber actor and place into parent's room
	auto baby = kernel::radio::system().spawn(
		ev_listener_actor<ev_state>, pimpl_->self_grp, std::move(make_ev_character)
	);
	// and return ID
	return baby.id();
}

auto node::unsubscribe(std::uint64_t event_cb_id) -> void {
	auto& AS = kernel::radio::system();
	const auto ev_actor = AS.registry().get(event_cb_id);
	// [NOTE] need to do `actor_cast` to resolve `send()` resolution ambiguity
	pimpl_->send(caf::actor_cast<caf::actor>(ev_actor), a_bye());
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

