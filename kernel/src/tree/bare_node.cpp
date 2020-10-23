/// @date 22.09.2020
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "node_impl.h"
#include "nil_engine.h"

NAMESPACE_BEGIN(blue_sky::tree)

bare_node::bare_node(const node& rhs) : bare_node(rhs.bare()) {
	if(is_nil()) throw error("Bare access to nil node");
}

bare_node::bare_node(std::shared_ptr<node_impl> impl) : pimpl_(std::move(impl)) {
	if(is_nil()) throw error("Bare access to nil node");
}

auto bare_node::pimpl() const -> node_impl* { return pimpl_.get(); }

auto bare_node::operator=(const node& rhs) -> bare_node& {
	return (*this = rhs.bare());
}

auto bare_node::armed() const -> node {
	return pimpl_->super_engine();
}

auto bare_node::is_nil() const -> bool {
	return pimpl_ == nil_node::pimpl();
}

auto bare_node::handle() const -> link {
	return pimpl()->handle();
}

auto bare_node::hash() const noexcept -> std::size_t {
	return std::hash<sp_nimpl>{}(pimpl_);
}

auto bare_node::size() const -> std::size_t {
	return pimpl_->size();
}

auto bare_node::empty() const -> bool {
	return pimpl_->links_.empty();
}

auto bare_node::leafs(Key order) const -> links_v {
	return pimpl_->leafs(order);
}

auto bare_node::keys(Key order) const -> lids_v {
	return pimpl_->keys(order);
}

auto bare_node::ikeys(Key order) const -> std::vector<std::size_t> {
	return pimpl_->ikeys(order);
}

auto bare_node::find(std::size_t idx) const -> link {
	return pimpl_->search<Key::AnyOrder>(idx);
}

auto bare_node::find(lid_type lid) const -> link {
	return pimpl_->search<Key::ID>(lid);
}

auto bare_node::index(lid_type id) const -> existing_index {
	return pimpl_->index<Key::ID>(id);
}

auto bare_node::insert(unsafe_t, link l, InsertPolicy pol) -> insert_status {
	auto [pos, is_inserted] = pimpl_->insert(std::move(l), pol);
	return { pimpl_->index<Key::ID>(pos), is_inserted };
}

auto bare_node::insert(link l, InsertPolicy pol) -> insert_status {
	insert_status res;
	l.apply([&] {
		res = insert(unsafe, l, pol);
		return perfect;
	});
	return res;
}

auto bare_node::insert(links_v ls, InsertPolicy pol) -> std::size_t {
	auto res = std::size_t{0};
	std::for_each(ls.begin(), ls.end(), [&](auto& L) {
		auto [pos, is_inserted] = insert(L, pol);
		if(is_inserted) ++res;
	});
	return res;
}

auto bare_node::insert(unsafe_t, links_v ls, InsertPolicy pol) -> std::size_t {
	auto res = std::size_t{0};
	std::for_each(ls.begin(), ls.end(), [&](auto& L) {
		auto [pos, is_inserted] = insert(unsafe, L, pol);
		if(is_inserted) ++res;
	});
	return res;
}

auto bare_node::insert(std::string name, sp_obj obj, InsertPolicy pol) -> insert_status {
	return insert(unsafe, link(std::move(name), std::move(obj)), pol);
}

auto bare_node::insert(std::string name, node N, InsertPolicy pol) -> insert_status {
	return insert(unsafe, link(std::move(name), std::move(N)), pol);
}

auto bare_node::erase(std::size_t idx) -> std::size_t {
	return pimpl_->erase<Key::AnyOrder>(idx);
}

auto bare_node::erase(lid_type link_id) -> std::size_t {
	return pimpl_->erase<Key::ID>(link_id);
}

auto bare_node::clear() -> std::size_t {
	auto res = pimpl_->size();
	pimpl_->links_.clear();
	return res;
}

auto bare_node::rearrange(std::vector<lid_type> new_order) -> error {
	return pimpl_->rearrange<Key::ID>(std::move(new_order));
}

auto bare_node::rearrange(std::vector<std::size_t> new_order) -> error {
	return pimpl_->rearrange<Key::AnyOrder>(std::move(new_order));
}

NAMESPACE_END(blue_sky::tree)
