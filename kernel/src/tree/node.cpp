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
#include <bs/serialize/cafbind.h>

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/random_generator.hpp>

OMIT_OBJ_SERIALIZATION
OMIT_ITERATORS_SERIALIZATION
CAF_ALLOW_UNSAFE_MESSAGE_TYPE(std::vector<blue_sky::tree::node::id_type>)

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

const auto uuid_from_str = boost::uuids::string_generator{};
using EraseOpts = node_actor::EraseOpts;

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
		factor_ = caf::function_view{ actor_, def_timeout(true) };
		return true;
	}
	return false;
}

auto node::actor() const -> const caf::actor& { return actor_; }

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

auto node::accepts(const sp_link& what) const -> bool {
	return pimpl_->accepts(what);
}

auto node::accept_object_types(std::vector<std::string> allowed_types) -> void {
	pimpl_->accept_object_types(std::move(allowed_types));
}

auto node::allowed_object_types() const -> std::vector<std::string> {
	auto guard = pimpl_->lock<node_impl::Metadata>(shared);
	return pimpl_->allowed_otypes_;
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

auto node::begin(Key_const<Key::AnyOrder>) const -> iterator<Key::AnyOrder> {
	return pimpl_->begin<>();
}

auto node::end(Key_const<Key::AnyOrder>) const -> iterator<Key::AnyOrder> {
	return pimpl_->end<>();
}

auto node::begin(Key_const<Key::ID>) const -> iterator<Key::ID> {
	return pimpl_->begin<Key::ID>();
}

auto node::end(Key_const<Key::ID>) const -> iterator<Key::ID> {
	return pimpl_->end<Key::ID>();
}

auto node::begin(Key_const<Key::Name>) const -> iterator<Key::Name> {
	return pimpl_->begin<Key::Name>();
}

auto node::end(Key_const<Key::Name>) const -> iterator<Key::Name> {
	return pimpl_->end<Key::Name>();
}

auto node::begin(Key_const<Key::OID>) const -> iterator<Key::OID> {
	return pimpl_->begin<Key::OID>();
}

auto node::end(Key_const<Key::OID>) const -> iterator<Key::OID> {
	return pimpl_->end<Key::OID>();
}

auto node::begin(Key_const<Key::Type>) const -> iterator<Key::Type> {
	return pimpl_->begin<Key::Type>();
}

auto node::end(Key_const<Key::Type>) const -> iterator<Key::Type> {
	return pimpl_->end<Key::Type>();
}

///////////////////////////////////////////////////////////////////////////////
//  find
//
auto node::find(const std::size_t idx) const -> iterator<Key::AnyOrder> {
	return std::next(begin(), idx);
}

auto node::find(const id_type& id) const -> iterator<Key::AnyOrder> {
	return pimpl_->find<Key::ID>(id);
}

auto node::find(const std::string& key, Key key_meaning) const -> iterator<Key::AnyOrder> {
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
auto node::index(const id_type& lid) const -> std::size_t {
	auto i = pimpl_->find<Key::ID, Key::AnyOrder>(lid);
	return std::distance(begin<Key::AnyOrder>(), i);
}

auto node::index(const iterator<Key::AnyOrder>& pos) const -> std::size_t {
	return std::distance(begin<Key::AnyOrder>(), pos);
}

auto node::index(const std::string& key, Key key_meaning) const -> std::size_t {
	auto i = find(key, key_meaning);
	return std::distance(begin<Key::AnyOrder>(), i);
}

// ---- equal_range
auto node::equal_range(const std::string& link_name) const -> range<Key::Name> {
	return pimpl_->equal_range<Key::Name>(link_name);
}

auto node::equal_range_oid(const std::string& oid) const -> range<Key::OID> {
	return pimpl_->equal_range<Key::OID>(oid);
}

auto node::equal_type(const std::string& type_id) const -> range<Key::Type> {
	return pimpl_->equal_range<Key::Type>(type_id);
}

///////////////////////////////////////////////////////////////////////////////
//  insert
//
auto node::insert(sp_link l, InsertPolicy pol) -> insert_status<Key::ID> {
	using R = insert_status<Key::ID>;
	using AR = std::pair<std::optional<link::id_type>, bool>;

	return actorf<AR>(factor_, a_lnk_insert(), std::move(l), pol)
		.and_then([=](const AR& r) -> result_or_err<R> {
			auto pos = r.first ? pimpl_->find<Key::ID, Key::ID>(*r.first) : pimpl_->end<Key::ID>();
			return R{ std::move(pos), r.second };
		})
		.value_or(R{ pimpl_->end<Key::ID>(), false });
}

auto node::insert(sp_link l, std::size_t idx, InsertPolicy pol) -> insert_status<Key::AnyOrder> {
	using R = insert_status<Key::AnyOrder>;
	using AR = std::pair<std::size_t, bool>;

	return actorf<AR>(factor_, a_lnk_insert(), std::move(l), idx, pol)
		.and_then([=](const AR& r) -> result_or_err<R> {
			return R{ std::next(begin<Key::AnyOrder>(), r.first), r.second };
		})
		.value_or(R{ end<Key::AnyOrder>(), false });
}

auto node::insert(sp_link l, iterator<> pos, InsertPolicy pol) -> insert_status<Key::AnyOrder> {
	return insert(std::move(l), (size_t)std::distance(begin<Key::AnyOrder>(), pos), pol);
}

auto node::insert(std::string name, sp_obj obj, InsertPolicy pol) -> insert_status<Key::ID> {
	return insert(
		std::make_shared<hard_link>(std::move(name), std::move(obj)), pol
	);
}

///////////////////////////////////////////////////////////////////////////////
//  erase
//
auto node::erase(const std::size_t idx) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), idx).value_or(0);
}

auto node::erase(const id_type& lid) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), lid, EraseOpts::Normal).value_or(0);
}

auto node::erase(const std::string& key, Key key_meaning) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), key, key_meaning).value_or(0);
}

auto node::erase(const range<Key::ID>& r) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), r.export_lids()).value_or(0);
}

auto node::erase(const range<Key::Name>& r) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), r.export_lids()).value_or(0);
}

auto node::erase(const range<Key::OID>& r) -> size_t {
	return actorf<size_t>(factor_, a_lnk_erase(), r.export_lids()).value_or(0);
}

auto node::clear() -> void {
	caf::anon_send(
		actor_, a_lnk_erase(),
		range<Key::ID>{ begin<Key::ID>(), end<Key::ID>() }.export_lids()
	);
}

///////////////////////////////////////////////////////////////////////////////
//  deep_search
//
auto node::deep_search(const id_type& id) const -> sp_link {
	return pimpl_->deep_search<>(id);
}

auto node::deep_search(const std::string& key, Key key_meaning) const -> sp_link {
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
auto node::keys(Key_const<Key::ID>) const -> std::vector<Key_type<Key::ID>> {
	return pimpl_->keys<Key::ID>();
}

auto node::keys(Key_const<Key::Name>) const -> std::vector<Key_type<Key::Name>> {
	return pimpl_->keys<Key::Name>();
}

auto node::keys(Key_const<Key::OID>) const -> std::vector<Key_type<Key::OID>> {
	return pimpl_->keys<Key::OID>();
}

auto node::keys(Key_const<Key::Type>) const -> std::vector<Key_type<Key::Type>> {
	return pimpl_->keys<Key::Type>();
}

///////////////////////////////////////////////////////////////////////////////
//  rename
//
auto node::rename(iterator<Key::AnyOrder> pos, std::string new_name) -> bool {
	return pimpl_->rename<Key::AnyOrder>(std::move(pos), std::move(new_name));
}

auto node::rename(const std::size_t idx, std::string new_name) -> bool {
	return rename(find(idx), std::move(new_name));
}

auto node::rename(const id_type& lid, std::string new_name) -> bool {
	return pimpl_->rename<Key::ID>(lid, std::move(new_name)) > 0;
}

auto node::rename(const std::string& key, std::string new_name, Key key_meaning, bool all) -> std::size_t {
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

///////////////////////////////////////////////////////////////////////////////
//  project
//
auto node::project(iterator<Key::ID> src) const -> iterator<Key::AnyOrder> {
	return pimpl_->project<Key::ID>(std::move(src));
}

auto node::project(iterator<Key::Name> src) const -> iterator<Key::AnyOrder> {
	return pimpl_->project<Key::Name>(std::move(src));
}

auto node::project(iterator<Key::OID> src) const -> iterator<Key::AnyOrder> {
	return pimpl_->project<Key::OID>(std::move(src));
}

auto node::project(iterator<Key::Type> src) const -> iterator<Key::AnyOrder> {
	return pimpl_->project<Key::Type>(std::move(src));
}

///////////////////////////////////////////////////////////////////////////////
//  rearrrange
//
auto node::rearrange(std::vector<id_type> new_order) -> void {
	caf::anon_send(actor(), a_node_rearrange(), std::move(new_order));
}

auto node::rearrange(std::vector<std::size_t> new_order) -> void {
	caf::anon_send(actor(), a_node_rearrange(), std::move(new_order));
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
					a_ack, a_lnk_rename, const link::id_type& lid, std::string new_name, std::string old_name
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
					a_ack, a_lnk_status, link::id_type lid, Req req, ReqStatus new_s, ReqStatus prev_s
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
					a_ack, a_lnk_insert, link::id_type lid, std::size_t pos, InsertPolicy pol
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkInserted, {
							{"link_id", to_string(lid)},
							{"pos", (prop::integer)pos}
						});
				},

				[self, wN = std::weak_ptr{N}](
					a_ack, a_lnk_insert, link::id_type lid, std::size_t to_idx, std::size_t from_idx
				) {
					bsout() << "*-* node: fired LinkInserted event" << bs_end;
					if(auto N = wN.lock())
						self->state.f(std::move(N), Event::LinkInserted, {
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
					a_ack, a_lnk_erase, const std::vector<link::id_type>& lids, std::vector<std::string>& oids
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
						[](const link::id_type& lid) { return to_string(lid); }
					);

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

