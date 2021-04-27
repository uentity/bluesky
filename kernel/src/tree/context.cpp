/// @file
/// @author uentity
/// @date 26.02.2020
/// @brief Qt model helper impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/objbase.h>
#include <bs/tree/context.h>
#include <bs/kernel/radio.h>
#include "tree_impl.h"

#include <boost/container_hash/hash.hpp>

#include <unordered_map>
#include <unordered_set>
#include <algorithm>

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
static auto to_lids_v(unsafe_t, const std::string& path) -> lids_v {
	auto path_split = std::vector< std::pair<std::string::const_iterator, std::string::const_iterator> >{};
	if(path_split.empty()) return {};

	auto from = path_split.begin();
	const auto skip_first = from->first == from->second;
	auto res = skip_first ? lids_v(path_split.size() - 1) : lids_v(path_split.size());
	std::transform(
		skip_first ? ++from : from, path_split.end(), res.begin(),
		[](const auto& Lid) {
			return to_uuid(
				unsafe, {&*Lid.first, static_cast<std::size_t>(Lid.second - Lid.first)}
			);
		}
	);
	return res;
}

static auto to_lids_v(const std::string& path) -> result_or_err<lids_v> {
	auto res = lids_v{};
	if(auto er = error::eval_safe_quiet([&] { res = to_lids_v(unsafe, path); }))
		return unexpected_err_quiet(er);
	return res;
}

template<typename Path>
static auto concat(Path&& lhs, lid_type rhs) {
	// detect that we can move from lhs
	if constexpr(!std::is_reference_v<Path> && !std::is_const_v<Path>) {
		lhs.push_back(std::move(rhs));
		return lhs;
	}
	else {
		using path_t = std::decay_t<Path>;
		auto leaf_path = path_t(lhs.size() + 1);
		std::copy(lhs.begin(), lhs.end(), leaf_path.begin());
		leaf_path.back() = std::move(rhs);
		return leaf_path;
	}
}

// enters data node only if allowed to (don't auto-expand lazy links)
inline static auto data_node(const link& L) -> node {
	return L.bare().data_node_if_ok();
}

inline static auto cached_data_node(const link& L) -> node {
	auto val = node::nil();
	L.apply([&](bare_link bl) {
		val = bl.data_node();
		return perfect;
	});
	return val;
}

// simple `deref_path` impl for vector of lids
template<typename PathIterator>
static auto deref_path(PathIterator from, PathIterator to, const link& root) -> link {
	auto res = link{};
	for(auto level = root; from != to; ++from) {
		if(auto N = data_node(level)) {
			if(( res = N.find(*from) ))
				level = res;
			else
				break;
		}
	}
	return res;
}

// sentinel
inline constexpr auto none = context::item_index{{}, -1};

inline static auto is_valid(const context::item_index& i) -> bool {
	return !(i == none);
}

/*-----------------------------------------------------------------------------
 *  impl
 *-----------------------------------------------------------------------------*/
struct BS_HIDDEN_API context::impl {
	using path_t = lids_v;
	using path_hash_t = boost::hash<path_t>;
	using idata_t = std::unordered_map<path_t, link::weak_ptr, path_hash_t>;
	idata_t idata_;

	node root_;
	link root_lnk_;

	///////////////////////////////////////////////////////////////////////////////
	//  ctors
	//
	impl(link root) :
		root_(root ? cached_data_node(root) : node::nil()), root_lnk_(root)
	{
		verify();
	}

	impl(sp_obj root) :
		root_(root ? root->data_node() : node::nil()), root_lnk_(root_ ? root_.handle() : link{})
	{
		verify();
	}

	impl(node root) :
		root_(root), root_lnk_(root_ ? root_.handle() : link{})
	{
		verify();
	}

	auto verify() -> void {
		if(!root_) {
			if(root_lnk_) root_ = cached_data_node(root_lnk_);
			if(!root_) {
				root_ = node();
				root_lnk_.reset();
			}
		}
		if(!root_lnk_)
			root_lnk_ = link::make_root<hard_link>("", root_);
	}

	auto reset(node root, link root_handle) {
		// cleanup
		idata_.clear();
		// assign new root
		root_ = std::move(root);
		root_lnk_ = root_handle;
		verify();
	}

	// [NOTE] always returns valid pointer
	auto push(path_t path, const link& item = {}) -> item_tag const* {
		auto [pos, is_inserted] = idata_.try_emplace(std::move(path), item);
		// update existing tag if valid link passed in
		if(!is_inserted && item)
			pos->second = item;
		// DEBUG
		//dump();
		return &*pos;
	}

	template<typename Path>
	auto push(Path&& base, lid_type leaf, const link& item = {}) {
		return push( concat(std::forward<Path>(base), std::move(leaf)), item );
	}

	// erases all subpath under given path
	// because if parent is invalid -> all child indexes are also invalid
	auto pop(const path_t& path, std::optional<idata_t::iterator> hint = {}) -> std::size_t {
		const auto n = path.size();
		std::size_t cnt = 0;
		for(auto pos = hint ? *hint : idata_.begin(); pos != idata_.end();) {
			const auto& pos_path = pos->first;
			if(pos_path.size() >= n && std::equal(path.begin(), path.end(), pos_path.begin())) {
				idata_.erase(pos++);
				++cnt;
			}
			else
				++pos;
		}
		return cnt;
	}

	template<typename Item>
	auto pop_item(const Item& item) -> std::size_t {
		std::size_t cnt = 0;
		while(true) {
			if(auto pos = std::find_if(
				idata_.begin(), idata_.end(), [&](auto& i) { return i.second == item; }
			); pos != idata_.end())
				cnt += pop(pos->first, pos);
			else
				break;
		}
		return cnt;
	}

	// find tag by path - returns unique element
	auto find(const std::string& path) const -> existing_tag {
		if(path.empty() || path == "/") return std::nullopt;
		return to_lids_v(path).map([&](auto&& upath) -> existing_tag {
			if(auto r = idata_.find(upath); r != idata_.end())
				return &*r;
			return std::nullopt;
		})
		.value_or(std::nullopt);
	}

	// find tag by link - can return multiple tags for same link, including invalid ones
	auto find(const link& what) const -> std::vector<existing_tag> {
		auto res = std::vector<existing_tag>{};
		for(auto pos = idata_.begin(), end = idata_.end(); pos != end; ++pos) {
			if(pos->second == what)
				res.push_back(&*pos);
		}
		return res;
	}

	// verify tag by comparing with given link
	// on success return valid item index
	auto verify_item_index(existing_tag L_tag, const link& L) -> item_index {
		const auto parent_node = L.owner();
		if(!parent_node) return none;

		const auto& [L_path, L_ptr] = **L_tag;
		if(L_ptr == L) {
			// check that cached parent matches link's parent
			auto L_row = node::existing_index{};
			if(L_path.size() == 1 && parent_node == root_)
				L_row = parent_node.index(L.id());
			else {
				auto [parent, _] = make(L_tag);
				if(parent) {
					const auto& [_, parent_ptr] = **parent;
					if(parent_ptr == parent_node.handle())
						L_row = parent_node.index(L.id());
				}
			}
			// if valid row is found - return
			if(L_row) return { std::move(L_tag), *L_row };
			// [NOTE] disable remove invalid rows, because clients may rely on 'em
			//else
			//	pop(L_path);
		}
		return none;
	}

	// simpler and quicker version of above when item row inside parent isn't needed
	auto verify_tag(existing_tag L_tag, const link& L) -> existing_tag {
		const auto parent_node = L.owner();
		if(!parent_node) return std::nullopt;

		const auto& [L_path, L_ptr] = **L_tag;
		if(L_ptr == L) {
			// check that cached parent matches link's parent
			if(L_path.size() == 1 && parent_node == root_)
				return L_tag;
			else {
				auto [parent, _] = make(L_tag);
				if(parent) {
					const auto& [_, parent_ptr] = **parent;
					if(parent_ptr == parent_node.handle())
						return L_tag;
				}
			}
		}
		return std::nullopt;
	}

	auto make(const std::string& path, bool nonexact_match = false) -> item_index {
		//bsout() << "*** index_from_path()" << bs_end;
		auto level = root_;
		auto res = none;

		auto cur_subpath = lids_v{};
		auto push_subpath = [&](const std::string& next_lid, const node& cur_level) {
			return to_uuid(next_lid).map([&](auto cur_uuid) {
				if(auto item = level.find(cur_uuid)) {
					if(auto item_row = level.index(item.id())) {
						cur_subpath.push_back(std::move(cur_uuid));
						res = { push(cur_subpath, item), *item_row };
						return item;
					}
				}
				return link{};
			})
			.value_or(link{});
		};

		// walk down tree
		detail::deref_path_impl(path, root_lnk_, {}, TreeOpts::Normal, std::move(push_subpath));
		return nonexact_match || path == to_string(cur_subpath) ? res : none;
	}

	auto make(const link& L, std::string path_hint = "/") -> item_index {
		//bsout() << "*** index_from_link()" << bs_end;
		// link is expected to be inside root's subtree
		const auto parent_node = L.owner();
		if(!parent_node) return none;

		// make path hint start relative to current model root
		auto rootp = abspath(root_lnk_, Key::ID);
		// if path_hint.startswith(rootp)
		if(path_hint.size() >= rootp.size() && std::equal(rootp.begin(), rootp.end(), path_hint.begin()))
			path_hint = path_hint.substr(rootp.size());

		// get search starting point & ensure it's cached in idata
		auto start_index = make(path_hint, true);
		auto start_tag = start_index.first;
		auto start_link = start_tag ? (**start_tag).second.lock() : root_lnk_;
		if(start_link == L) return start_index;

		// walk down from start node and do step-by-step search for link
		// all intermediate paths are added to `idata`
		auto res = none;
		auto develop_link = [&](link R, std::list<link>& nodes, std::vector<link>& objs)
		mutable {
			// early exit if we already found valid result
			if(is_valid(res)) {
				nodes.clear();
				return;
			}
			// get current root path
			auto R_path = path_t{};
			if(R == start_link && R != root_lnk_)
				R_path = (**start_tag).first;
			else {
				for(auto r : find(R))
					if(auto rtag = verify_tag(std::move(r), R)) {
						R_path = (**rtag).first;
						break;
					}
			}
			// if root path wasn't found - return
			if(R != root_lnk_ && R_path.empty()) {
				nodes.clear();
				return;
			}

			// leaf checker
			const auto check_link = [&, Lid = L.id()](auto& item) {
				// remember each encountered link
				auto ltag = push(R_path, item.id(), item);
				// match check
				if(item.id() == Lid) {
					if(auto Rnode = data_node(R)) {
						if(auto row = Rnode.index(Lid))
							res = { std::move(ltag), *row };
					}
				}
				return is_valid(res);
			};

			// first check object leafs
			for(const auto& O : objs) { if(check_link(O)) break; }
			// next check each nodes leafs
			if(!is_valid(res))
				for(const auto& N : nodes) { if(check_link(N)) break; }
			// if we found a link, stop further iterations
			if(is_valid(res)) nodes.clear();
		};

		walk(start_link, std::move(develop_link));
		return res;
	}

	auto make(std::int64_t row, existing_tag parent) -> existing_tag {
		//bsout() << "*** index()" << bs_end;
		// extract parent node & path
		// [NOTE] empty parent means root
		auto parent_node = node::nil();
		if(auto parent_link = (parent ? (**parent).second.lock() : root_lnk_))
			parent_node = data_node(parent_link);
		if(!parent_node) return {};

		// obtain child link
		auto child_link = [&] {
			if(auto par_sz = (std::int64_t)parent_node.size(); row < par_sz) {
				// support negative indexing from node end
				if(row < 0) row += par_sz;
				if(row >= 0) return parent_node.find((std::size_t)row);
			}
			return link{};
		}();

		if(child_link)
			return push(parent_node != root_ ? (**parent).first : lids_v{}, child_link.id(), child_link);
		return {};
	}

	// returns parent index
	auto make(const item_tag& child) -> item_index {
		// [NOTE] do we need this extra check?
		auto parent_node = child.second.lock().owner();
		if(!parent_node || parent_node == root_) return none;

		// child path have to consist at least of two parts
		// if it contains one part, then parent is hidden root
		const auto& child_path = child.first;
		if(child_path.size() < 2) return none;

		// extract grandparent, it's ID is child_path[-3]
		auto grandpa_node = node::nil();
		if(child_path.size() < 3)
			grandpa_node = root_;
		else if(auto grandpa_link = deref_path(child_path.begin(), child_path.end() - 2, root_lnk_))
			grandpa_node = data_node(grandpa_link);
		if(!grandpa_node) return none;

		// parent's ID is child_path[-2]
		auto parent_link = grandpa_node.find(*(child_path.end() - 2));
		auto parent_path = lids_v(child_path.begin(), child_path.end() - 1);
		if(auto parent_row = grandpa_node.index(parent_link.id()))
			return { push(std::move(parent_path), parent_link), *parent_row };
		else {
			// remove incorrect path from tags cache
			// [NOTE] disabled, because clients may rely on invalid entries
			//pop(parent_path);
			return none;
		}
	}

	auto make(existing_tag child) -> item_index {
		//auto t = log::I("*** parent({}, {}) = [{}, {}]") <<
		//	(child ? to_string((*child)->first) : "") <<
		//	(child ? (*child)->second.lock().name(unsafe) : "");
		return child ? make(**child) : none;
		//if(res != none)
		//	bsout() << t <<
		//		(res == none ? "none" : to_string((*res.first)->first)) << res.second << bs_end;
		//return res;
	}

	auto dump() const -> void {
		//static auto state = idata_t{};
		//if(idata_ == state) return;
		//state = idata_;

		for(const auto& [p, l] : idata_) {
			auto L = l.lock();
			if(!L) continue;
			bsout() << "{} -> [{}]" << to_string(p) << L.name(unsafe) << bs_end;
		}
		bsout() << "====" << bs_end;
	}
};

/*-----------------------------------------------------------------------------
 *  context
 *-----------------------------------------------------------------------------*/
context::context(node root) :
	pimpl_{std::make_unique<impl>(std::move(root))}
{}

context::context(sp_obj root) :
	pimpl_{std::make_unique<impl>(std::move(root))}
{}

context::context(link root) :
	pimpl_{std::make_unique<impl>(std::move(root))}
{}

context::~context() = default;

auto context::reset(link root) -> void {
	pimpl_->reset(node::nil(), root);
}

auto context::reset(node root, link root_handle) -> void {
	pimpl_->reset(std::move(root), root_handle);
}

auto context::root() const -> node {
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
