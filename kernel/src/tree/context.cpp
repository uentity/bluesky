/// @file
/// @author uentity
/// @date 26.02.2020
/// @brief Qt model helper impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/tree/context.h>
#include "tree_impl.h"

#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_hash.hpp>
#include <boost/container_hash/hash.hpp>

#include <unordered_map>

NAMESPACE_BEGIN(blue_sky::tree)

auto to_string(const lids_v& path, bool as_absolute) -> std::string {
	auto res = std::vector<std::string>(as_absolute ? path.size() + 1 : path.size());
	std::transform(
		path.begin(), path.end(),
		as_absolute ? std::next(res.begin()) : res.begin(),
		[](const auto& Lid) { return to_string(Lid); }
	);
	return boost::join(res, "/");
}

// convert string path to vector of lids
static auto to_lids_v(const std::string& path) -> lids_v {
	[[maybe_unused]] static const auto uuid_from_str = boost::uuids::string_generator{};

	auto path_split = std::vector< std::pair<std::string::const_iterator, std::string::const_iterator> >{};
	boost::split(path_split, path, boost::is_any_of("/"));
	if(path_split.empty()) return {};

	auto from = path_split.begin();
	const auto skip_first = from->first == from->second;
	auto res = skip_first ? lids_v(path_split.size() - 1) : lids_v(path_split.size());
	std::transform(
		skip_first ? ++from : from, path_split.end(), res.begin(),
		[](const auto& Lid) { return uuid_from_str(Lid.first, Lid.second); }
	);
	return res;
}

// enters data node only if allowed to (don't auto-expand lazy links)
static auto data_node(const link& L) -> sp_node {
	return L.data_node(unsafe);
}

// simple `deref_path` impl for vector of lids
template<typename PathIterator>
static auto deref_path(PathIterator from, PathIterator to, const link& root) -> link {
	auto res = link{};
	auto level = root;
	for(; from != to; ++from) {
		if(auto N = data_node(level)) {
			if(( res = N->find(*from) ))
				level = res;
			else
				break;
		}
	}
	return res;
}

/*-----------------------------------------------------------------------------
 *  impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API context::impl {
	using path_t = lids_v;
	using path_hash_t = boost::hash<path_t>;
	using idata_t = std::unordered_map<path_t, link::weak_ptr, path_hash_t>;
	idata_t idata_;

	sp_node root_;
	link root_lnk_;

	///////////////////////////////////////////////////////////////////////////////
	//  ctors
	//
	impl(link root) :
		root_(data_node(root)), root_lnk_(root)
	{
		verify();
	}

	impl(sp_node root) :
		root_(std::move(root)), root_lnk_(link::make_root<hard_link>("/", root_))
	{
		verify();
	}

	auto verify() -> void {
		if(!root_) {
			if(root_lnk_) root_ = data_node(root_lnk_);
			if(!root_) {
				root_ = std::make_shared<node>();
				root_lnk_.reset();
			}
		}
		if(!root_lnk_)
			root_lnk_ = link::make_root<hard_link>("/", root_);
	}

	auto reset(sp_node root, link root_handle) {
		idata_.clear();
		root_ = std::move(root);
		root_lnk_ = root_handle;
		verify();
	}

	auto push(path_t path, const link& item = {}) -> item_tag& {
		auto [pos, is_inserted] = idata_.try_emplace(std::move(path), item);
		// update existing tag if valid link passed in
		if(!is_inserted && item)
			pos->second = item;
		// DEBUG
		//dump();
		return *pos;
	}

	decltype(auto) push(const path_t& base, lid_type leaf, const link& item = {}) {
		auto leaf_path = path_t(base.size() + 1);
		std::copy(base.begin(), base.end(), leaf_path.begin());
		leaf_path.back() = std::move(leaf);
		return push(std::move(leaf_path), item);
	}

	auto pop(const path_t& path) -> bool {
		if(auto pos = idata_.find(path); pos != idata_.end()) {
			idata_.erase(pos);
			return true;
		}
		return false;
	}

	template<typename Item>
	auto pop_item(const Item& item) -> std::size_t {
		std::size_t cnt = 0;
		for(auto pos = idata_.begin(); pos != idata_.end();) {
			if(pos->second == item) {
				idata_.erase(pos++);
				++cnt;
			}
			else
				++pos;
		}
		return cnt;
	}

	// searh idata for given link (no midifications are made)
	auto find(const link& what) const -> existing_tag {
		for(auto pos = idata_.begin(), end = idata_.end(); pos != end; ++pos) {
			if(pos->second == what)
				return &*pos;
		}
		return {};
	}

	auto make(const std::string& path, bool nonexact_match = false) -> item_index {
		//bsout() << "*** index_from_path()" << bs_end;
		auto level = root_;
		auto res = item_index{{}, -1};

		auto cur_subpath = std::string{};
		auto push_subpath = [&](const std::string& next_lid, const sp_node& cur_level) {
			if(auto item = level->find(next_lid, Key::ID)) {
				cur_subpath += '/';
				cur_subpath += next_lid;
				res = { &push(to_lids_v(cur_subpath), item), level->index(item.id()).value_or(-1) };
				return item;
			}
			return link{};
		};

		// walk down tree
		detail::deref_path_impl(path, root_lnk_, root_, false, std::move(push_subpath));
		return nonexact_match || path == cur_subpath ? res : item_index{{}, -1};
	}

	auto make(const link& L, std::string path_hint = "/") -> item_index {
		//bsout() << "*** index_from_link()" << bs_end;
		// make path hint start relative to current model root
		// [NOTE] including leading slash!
		auto rootp = abspath(root_lnk_, Key::ID);
		// if path_hint.startswith(rootp)
		if(path_hint.size() >= rootp.size() && std::equal(rootp.begin(), rootp.end(), path_hint.begin()))
			path_hint = path_hint.substr(rootp.size());

		// get search starting point & ensure it's cached in idata
		auto start_tag = make(path_hint, true).first;
		auto start_link = start_tag ? (**start_tag).second.lock() : root_lnk_;

		// walk down from start node and do step-by-step search for link
		// all intermediate paths are added to `idata`
		auto res = item_index{{}, -1};
		auto develop_link = [&, Lid = L.id()](link R, std::list<link>& nodes, std::vector<link>& objs)
		mutable {
			// get current root path
			auto R_path = path_t{};
			if(auto R_tag = find(R))
				R_path = (**R_tag).first;
			else if(R != root_lnk_) {
				nodes.clear();
				return;
			}

			// leaf checker
			bool found = false;
			const auto check_link = [&](auto& item) {
				if(item.id() == Lid) {
					if(auto Rnode = data_node(R)) {
						if(auto row = Rnode->index(Lid)) {
							res = { &push(R_path, std::move(Lid), item), *row };
							found = true;
						}
					}
				}
				return found;
			};

			// check each leaf (both nodes & links)
			for(const auto& N : nodes) { if(check_link(N)) break; }
			if(!found) {
				for(const auto& O : objs) { if(check_link(O)) break; }
			}
			// if we found a link, stop further iterations
			if(found) nodes.clear();
		};

		walk(start_link, develop_link);
		return res;
	}

	auto make(std::int64_t row, existing_tag parent) -> existing_tag {
		//bsout() << "*** index()" << bs_end;
		// extract parent node & path
		// [NOTE] empty parent means root
		auto parent_node = sp_node{};
		if(auto parent_link = (parent ? (**parent).second.lock() : root_lnk_))
			parent_node = data_node(parent_link);
		if(!parent_node) return {};

		// obtain child link
		auto child_link = [&] {
			if(auto par_sz = (std::int64_t)parent_node->size(); row < par_sz) {
				// sometimes we can get negative row, it just means last element
				if(row < 0) row = par_sz - 1;
				if(row >= 0) return parent_node->find((std::size_t)row);
			}
			return link{};
		}();

		if(child_link)
			return &push(parent_node != root_ ? (**parent).first : lids_v{}, child_link.id(), child_link);
		return {};
	}

	auto make(const item_tag& child) -> item_index {
		// obtain parent
		auto parent_node = child.second.lock().owner();
		if(!parent_node || parent_node == root_) return {{}, -1};

		// child path have to consist at least of two parts
		// if it contains one part, then parent is hidden root
		const auto& child_path = child.first;
		if(child_path.size() < 2) return {};

		// extract grandparent, it's ID is child_path[-3]
		auto grandpa_node = [&] {
			if(child_path.size() < 3)
				return root_;
			else {
				auto grandpa_link = deref_path(child_path.begin(), child_path.end() - 2, root_lnk_);
				return grandpa_link ? data_node(grandpa_link) : sp_node{};
			}
		}();
		if(!grandpa_node) return {};

		// parent's ID is child_path[-2]
		if(auto parent_link = grandpa_node->find(*(child_path.end() - 2)))
			return {
				&push(lids_v(child_path.begin(), child_path.end() - 1), parent_link),
				grandpa_node->index(parent_link.id()).value_or(-1)
			};
		return {{}, -1};
	}

	auto make(existing_tag child) -> item_index {
		//bsout() << "*** parent()" << bs_end;
		return child ? make(**child) : item_index{{}, -1};
	}

	auto dump() const -> void {
		for(const auto& [p, l] : idata_) {
			auto L = l.lock();
			if(!L) continue;
			bsout() << "{} -> [{}, {}]" << to_string(p) << to_string(L.id()) << L.name(unsafe) << bs_end;
		}
		bsout() << "====" << bs_end;
	}
};

/*-----------------------------------------------------------------------------
 *  context
 *-----------------------------------------------------------------------------*/
context::context(sp_node root) :
	pimpl_{std::make_unique<impl>(std::move(root))}
{}

context::context(link root) :
	pimpl_{std::make_unique<impl>(std::move(root))}
{}

context::~context() = default;

auto context::reset(link root) -> void {
	pimpl_->reset(nullptr, root);
}

auto context::reset(sp_node root, link root_handle) -> void {
	pimpl_->reset(std::move(root), root_handle);
}

auto context::root() const -> sp_node {
	return pimpl_->root_;
}

auto context::root_link() const -> link {
	return pimpl_->root_lnk_;
}

auto context::root_path(Key path_unit) const -> std::string {
	return abspath(pimpl_->root_lnk_, path_unit);
}

/// make tag for given path
auto context::operator()(const std::string& path, bool nonexact_match) -> item_index {
	return pimpl_->make(path, nonexact_match);
}
/// for given link + possible hint
auto context::operator()(const link& L, std::string path_hint) -> item_index {
	return pimpl_->make(L, std::move(path_hint));
}

/// helper for abstrct model's `index()`
auto context::operator()(std::int64_t row, existing_tag parent) -> existing_tag {
	return pimpl_->make(row, parent);
}
/// for `parent()`
auto context::operator()(existing_tag child) -> item_index {
	return pimpl_->make(child);
}

auto context::dump() const -> void {
	pimpl_->dump();
}

NAMESPACE_END(blue_sky::tree)
