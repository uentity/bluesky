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
#include <bs/serialize/tree.h>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  node
 *-----------------------------------------------------------------------------*/
node::node(std::string custom_id)
	: objbase(true, std::move(custom_id)), pimpl_(std::make_shared<node_impl>(this))
{
	start_engine();
}

node::node(const node& src)
	: objbase(src), pimpl_(std::make_shared<node_impl>(*src.pimpl_, this))
{
	start_engine();
}

auto node::operator=(const node& rhs) -> node& {
	*pimpl_ = *rhs.pimpl_;
	propagate_owner();
	return *this;
}

node::~node() {
	// mark self as deleted for impl & actor
	pimpl_->super_ = nullptr;
	// then kill self actor
	caf::anon_send_exit(actor(), caf::exit_reason::user_shutdown);
}

auto node::start_engine(std::string gid) -> void {
	pimpl_->start_engine(pimpl_, std::move(gid));
}

auto node::raw_actor() const -> const caf::actor& { return pimpl_->actor_; }

auto node::home() const -> const caf::group& { return pimpl_->home_; }

auto node::gid() const -> std::string {
	return pimpl_->actorf<std::string>(
		*this, a_node_gid()
	).value_or("");
}

auto node::propagate_owner(bool deep) -> void {
	// properly setup owner in node's leafs
	auto self = bs_shared_this<node>();
	for(auto& L : leafs()) {
		auto child_node = node_impl::adjust_inserted_link(L, self);
		if(deep && child_node)
			child_node->propagate_owner(true);
	}
}

auto node::handle() const -> link {
	return pimpl_->handle();
}

auto node::set_handle(const link& handle) -> void {
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
	return pimpl_->actorf<lids_v>(
		*this, a_node_keys(), ordering
	).value_or(lids_v{});
}

auto node::ikeys(Key ordering) const -> std::vector<std::size_t> {
	using R = std::vector<std::size_t>;
	return pimpl_->actorf<std::vector<std::size_t>>(
		*this, a_node_ikeys(), ordering
	).value_or(R{});
}

auto node::skeys(Key key_meaning, Key ordering) const -> std::vector<std::string> {
	using R = std::vector<std::string>;
	return pimpl_->actorf<R>(
		*this, a_node_keys(), key_meaning, ordering
	).value_or(R{});
}

///////////////////////////////////////////////////////////////////////////////
//  find
//
auto node::find(std::size_t idx) const -> link {
	return pimpl_->actorf<link>(
		*this, a_node_find(), idx
	).value_or(link{});
}

auto node::find(lid_type id) const -> link {
	return pimpl_->actorf<link>(
		*this, a_node_find(), std::move(id)
	).value_or(link{});
}

auto node::find(std::string key, Key key_meaning) const -> link {
	return pimpl_->actorf<link>(
		*this, a_node_find(), std::move(key), key_meaning
	).value_or(link{});
}

// ---- deep_search
auto node::deep_search(lid_type id) const -> link {
	return pimpl_->actorf<link>(
		*this, a_node_deep_search(), std::move(id)
	).value_or(link{});
}

auto node::deep_search(std::string key, Key key_meaning) const -> link {
	auto res = pimpl_->actorf<links_v>(
		*this, a_node_deep_search(), std::move(key), key_meaning, false
	).value_or(links_v{});
	return res.empty() ? link{} : res[0];
}

auto node::deep_equal_range(std::string key, Key key_meaning) const -> links_v {
	return pimpl_->actorf<links_v>(
		*this, a_node_deep_search(), std::move(key), key_meaning, true
	).value_or(links_v{});
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
auto node::insert(link l, InsertPolicy pol) const -> insert_status {
	return pimpl_->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(link l, std::size_t idx, InsertPolicy pol) const -> insert_status {
	return pimpl_->actorf<insert_status>(
		*this, a_node_insert(), std::move(l), idx, pol
	).value_or(insert_status{ {}, false });
}

auto node::insert(links_v ls, InsertPolicy pol) const -> std::size_t {
	return pimpl_->actorf<std::size_t>(
		*this, a_node_insert(), std::move(ls), pol
	).value_or(0);
}

auto node::insert(std::string name, sp_obj obj, InsertPolicy pol) const -> insert_status {
	return insert( hard_link(std::move(name), std::move(obj)), pol );
}

///////////////////////////////////////////////////////////////////////////////
//  erase
//
auto node::erase(std::size_t idx) const -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), idx
	).value_or(0);
}

auto node::erase(lid_type lid) const -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), std::move(lid), EraseOpts::Normal
	).value_or(0);
}

auto node::erase(std::string key, Key key_meaning) const -> size_t {
	return pimpl_->actorf<size_t>(
		*this, a_node_erase(), std::move(key), key_meaning
	).value_or(0);
}

auto node::clear() const -> void {
	caf::anon_send(actor(), a_node_clear());
}

///////////////////////////////////////////////////////////////////////////////
//  rename
//
auto node::rename(std::size_t idx, std::string new_name) const -> bool {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), idx, std::move(new_name)
	).value_or(false);
}

auto node::rename(lid_type lid, std::string new_name) const -> bool {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(lid), std::move(new_name)
	).value_or(false);
}

auto node::rename(std::string old_name, std::string new_name) const -> std::size_t {
	return pimpl_->actorf<std::size_t>(
		*this, a_lnk_rename(), std::move(old_name), std::move(new_name)
	).value_or(false);
}

///////////////////////////////////////////////////////////////////////////////
//  rearrrange
//
auto node::rearrange(lids_v new_order) const -> error {
	return pimpl_->actorf<error>(
		*this, a_node_rearrange(), std::move(new_order)
	);
}

auto node::rearrange(std::vector<std::size_t> new_order) const -> error {
	return pimpl_->actorf<error>(
		*this, a_node_rearrange(), std::move(new_order)
	);
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
BS_TYPE_IMPL(node, objbase, "node", "BS tree node");
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

