/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief Implementation details of tree-related functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/tree.h>
#include <bs/detail/enumops.h>

#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky::tree::detail)

auto walk_down_tree(
	const std::string& next_lid, const node& cur_level, Key path_unit = Key::ID
) -> link;

NAMESPACE_BEGIN()
// put into hidden namespace to prevent equal multiple instantiations
auto gen_walk_down_tree(Key path_unit = Key::ID) {
	return [path_unit](const std::string& next_lid, const node& cur_level) {
		return walk_down_tree(next_lid, cur_level, path_unit);
	};
}

NAMESPACE_END()

// find out if we can call `data_node()` honoring LazyLoad flag
inline auto can_call_dnode(const link& L, TreeOpts opts) -> bool {
	using namespace allow_enumops;
	return enumval(opts & TreeOpts::FollowLazyLinks)
		|| L.req_status(Req::DataNode) == ReqStatus::OK
		|| !(L.flags() & LazyLoad);
}

// If `DerefControlElements` == true, processing function will be invoked for all path parts
// including ".", ".." and empty part (root handle)
template<
	bool DerefControlElements = false,
	typename level_deref_f = decltype(gen_walk_down_tree())
>
auto deref_path_impl(
	const std::string& path, link L, node root = node::nil(), TreeOpts opts = TreeOpts::Normal,
	level_deref_f deref_f = gen_walk_down_tree()
) -> link {
	using namespace allow_enumops;

	// split path into elements
	if(path.empty()) return {};
	std::vector<std::string> path_parts;
	boost::split(path_parts, path, boost::is_any_of("/"));

	// setup search root
	if(path_parts[0].empty()) {
		// absolute path case
		root = root ? find_root(root) : find_root(L);
	}
	if(root) L = root.handle();

	// deref each element
	for(const auto& part : path_parts) {
		bool is_control_elem = false;
		if(part.empty() || part == ".")
			is_control_elem = true;
		else if(part == "..") {
			root = L ? L.owner() : node::nil();
			is_control_elem = true;
		}
		else if(!root)
			root = L && can_call_dnode(L, opts) ? L.data_node() : node::nil();

		if constexpr(DerefControlElements) {
			// intentional ignore of deref return value
			if(is_control_elem) deref_f(part, root);
		}
		if(!is_control_elem) {
			L = root ? deref_f(part, root) : link{};
			if(!L) break;
			// force root recalc on next spin
			root.reset();
		}
	}
	return L;
}

NAMESPACE_END(blue_sky::tree::detail)
