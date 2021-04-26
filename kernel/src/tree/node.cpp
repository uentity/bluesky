/// @file
/// @author uentity
/// @date 15.09.2016
/// @brief BlueSKy tree node implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_actor.h"
#include "nil_engine.h"

#include <bs/kernel/types_factory.h>
#include <bs/kernel/radio.h>
#include <bs/log.h>
#include <bs/tree/tree.h>

#include <memory_resource>

NAMESPACE_BEGIN(blue_sky::tree)
// setup synchronized pool allocator for node impls
static auto impl_pool = std::pmr::synchronized_pool_resource{};
static auto node_impl_alloc = std::pmr::polymorphic_allocator<node_impl>(&impl_pool);

node::node(engine&& e) :
	engine(std::move(e))
{
	if(!has_engine())
		*this = nil_node::nil_engine();
}

node::node(sp_engine_impl impl) :
	engine(nil_node::actor(), std::move(impl))
{
	start_engine();
}

node::node(links_v leafs) :
	node(std::allocate_shared<node_impl>(node_impl_alloc))
{
	// insert each link with proper locking
	insert(std::move(leafs));
}

node::node(const bare_node& rhs) : node(rhs.armed()) {}


auto node::bare() const -> bare_node {
	return bare_node(std::static_pointer_cast<node_impl>(pimpl_));
}

auto node::operator=(const bare_node& rhs) -> node& {
	return (*this = rhs.armed());
}

auto node::nil() -> node {
	return engine(nil_node::nil_engine());
}

auto node::is_nil() const -> bool {
	return *this == nil_node::nil_engine();
}

auto node::reset() -> void {
	if(!is_nil())
		*this = nil_node::nil_engine();
}

auto node::start_engine() -> bool {
	if(nil_node::nil_engine() == raw_actor()) {
		install_raw_actor(node_impl::spawn_actor(std::static_pointer_cast<node_impl>(pimpl_)));
		// set myself as owner of my leafs
		// [NOTE] can only be executed AFTER engine is installed & started
		pimpl()->propagate_owner(*this, false);
		return true;
	}
	return false;
}

auto node::pimpl() const -> node_impl* {
	return static_cast<node_impl*>(pimpl_.get());
}

auto node::handle() const -> link {
	return pimpl()->handle();
}

auto node::clone(bool deep) const -> node {
	return pimpl()->actorf<sp_nimpl>(
		*this, kernel::radio::timeout(true), a_clone{}, a_impl{}, deep
	)
	.map([](auto&& clone_impl) {
		return node{std::move(clone_impl)};
	}).value_or(node::nil());
}

///////////////////////////////////////////////////////////////////////////////
//  leafs container
//
auto node::size() const -> std::size_t {
	return pimpl()->actorf<std::size_t>(*this, a_node_size()).value_or(0);
}

auto node::empty() const -> bool {
	return size() == 0;
}

auto node::leafs(Key order) const -> links_v {
	return pimpl()->actorf<links_v>(
		*this, a_node_leafs(), order
	).value_or(links_v{});
}

///////////////////////////////////////////////////////////////////////////////
//  keys
//
auto node::keys(Key ordering) const -> lids_v {
	return pimpl()->actorf<lids_v>(
		*this, a_node_keys(), ordering
	).value_or(lids_v{});
}

auto node::ikeys(Key ordering) const -> std::vector<std::size_t> {
	using R = std::vector<std::size_t>;
	return pimpl()->actorf<std::vector<std::size_t>>(
		*this, a_node_ikeys(), ordering
	).value_or(R{});
}

auto node::skeys(Key key_meaning, Key ordering) const -> std::vector<std::string> {
	using R = std::vector<std::string>;
	return pimpl()->actorf<R>(
		*this, a_node_keys(), key_meaning, ordering
	).value_or(R{});
}

///////////////////////////////////////////////////////////////////////////////
//  find
//
auto node::find(std::size_t idx) const -> link {
	return pimpl()->actorf<link>(
		*this, a_node_find(), idx
	).value_or(link{});
}

auto node::find(lid_type id) const -> link {
	return pimpl()->actorf<link>(
		*this, a_node_find(), std::move(id)
	).value_or(link{});
}

auto node::find(std::string key, Key key_meaning) const -> link {
	return pimpl()->actorf<link>(
		*this, a_node_find(), std::move(key), key_meaning
	).value_or(link{});
}

// ---- deep_search
auto node::deep_search(lid_type id) const -> link {
	auto res = pimpl()->actorf<links_v>(
		*this, a_node_deep_search(), std::move(id)
	).value_or(links_v{});
	return res.empty() ? link{} : res.front();
}

auto node::deep_search(std::string key, Key key_meaning) const -> link {
	auto res = pimpl()->actorf<links_v>(
		*this, a_node_deep_search(), std::move(key), key_meaning, true
	).value_or(links_v{});
	return res.empty() ? link{} : res.front();
}

auto node::deep_equal_range(std::string key, Key key_meaning) const -> links_v {
	return pimpl()->actorf<links_v>(
		*this, a_node_deep_search(), std::move(key), key_meaning, false
	).value_or(links_v{});
}

// ---- index
auto node::index(lid_type lid) const -> existing_index {
	return pimpl()->actorf<existing_index>(
		*this, a_node_index(), std::move(lid)
	).value_or(existing_index{});
}

auto node::index(std::string key, Key key_meaning) const -> existing_index {
	return pimpl()->actorf<existing_index>(
		*this, a_node_index(), std::move(key), key_meaning
	).value_or(existing_index{});
}

// ---- equal_range
auto node::equal_range(std::string key, Key key_meaning) const -> links_v {
	return pimpl()->actorf<links_v>(
		*this, a_node_equal_range(), std::move(key), key_meaning
	).value_or(links_v{});
}

///////////////////////////////////////////////////////////////////////////////
//  insert
//
auto node::insert(link l, InsertPolicy pol) const -> insert_status {
	return pimpl()->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(link l, std::size_t idx, InsertPolicy pol) const -> insert_status {
	return pimpl()->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), idx, pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(links_v ls, InsertPolicy pol) const -> std::size_t {
	return pimpl()->actorf<std::size_t>(
		*this, a_node_insert(), std::move(ls), pol
	).value_or(0);
}

auto node::insert(std::string name, sp_obj obj, InsertPolicy pol) const -> insert_status {
	return insert( hard_link(std::move(name), std::move(obj)), pol );
}

auto node::insert(std::string name, node N, InsertPolicy pol) const -> insert_status {
	return insert( hard_link(std::move(name), std::move(N)), pol );
}

///////////////////////////////////////////////////////////////////////////////
//  erase
//
auto node::erase(std::size_t idx) const -> size_t {
	return pimpl()->actorf<size_t>(
		*this, a_node_erase(), idx
	).value_or(0);
}

auto node::erase(lid_type lid) const -> size_t {
	return pimpl()->actorf<size_t>(
		*this, a_node_erase(), std::move(lid), EraseOpts::Normal
	).value_or(0);
}

auto node::erase(std::string key, Key key_meaning) const -> size_t {
	return pimpl()->actorf<size_t>(
		*this, a_node_erase(), std::move(key), key_meaning
	).value_or(0);
}

auto node::clear() const -> std::size_t {
	return pimpl()->actorf<std::size_t>(*this, a_node_clear()).value_or(0);
}

auto node::clear(launch_async_t) const -> void {
	caf::anon_send(actor(), a_node_clear());
}

///////////////////////////////////////////////////////////////////////////////
//  rename
//
auto node::rename(std::size_t idx, std::string new_name) const -> bool {
	return pimpl()->actorf<std::size_t>(
		*this, a_lnk_rename(), idx, std::move(new_name)
	).value_or(false);
}

auto node::rename(lid_type lid, std::string new_name) const -> bool {
	return pimpl()->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(lid), std::move(new_name)
	).value_or(false);
}

auto node::rename(std::string old_name, std::string new_name) const -> std::size_t {
	return pimpl()->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(old_name), std::move(new_name)
	).value_or(false);
}

///////////////////////////////////////////////////////////////////////////////
//  rearrrange
//
auto node::rearrange(lids_v new_order) const -> error {
	return pimpl()->actorf<error>(
		*this, a_node_rearrange(), std::move(new_order)
	);
}

auto node::rearrange(std::vector<std::size_t> new_order) const -> error {
	return pimpl()->actorf<error>(
		*this, a_node_rearrange(), std::move(new_order)
	);
}

///////////////////////////////////////////////////////////////////////////////
//  apply
//
auto node::apply(node_transaction tr) const -> error {
	return pimpl()->actorf<tr_result>(*this, a_apply(), std::move(tr));
}

auto node::apply(launch_async_t, node_transaction tr) const -> void {
	caf::anon_send(pimpl()->actor(*this), a_apply(), std::move(tr));
}

NAMESPACE_END(blue_sky::tree)
