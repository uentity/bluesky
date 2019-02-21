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
#include <boost/algorithm/string.hpp>

#define CAN_CALL_DNODE(L) \
( !((L).flags() & link::LazyLoad) || (L).req_status(link::Req::DataNode) == link::ReqStatus::OK )

NAMESPACE_BEGIN(blue_sky::tree::detail)

sp_link walk_down_tree(
	const std::string& next_lid, const sp_node& cur_level, node::Key path_unit = node::Key::ID
);

NAMESPACE_BEGIN()
// put into hidden namespace to prevent equal multiple instantiations
auto gen_walk_down_tree(node::Key path_unit = node::Key::ID) {
	return [path_unit](const std::string& next_lid, const sp_node& cur_level) {
		return walk_down_tree(next_lid, cur_level, path_unit);
	};
}

NAMESPACE_END()

// find out if we can call `data_node()` honoring LazyLoad flag
inline auto can_call_dnode(const link& L) -> bool {
	return !(L.flags() & link::LazyLoad) || L.req_status(link::Req::DataNode) == link::ReqStatus::OK;
}

// If `DerefControlElements` == true, processing function will be invoked for all path parts
// including ".", ".." and empty part (root handle)
template<
	bool DerefControlElements = false,
	typename level_deref_f = decltype(gen_walk_down_tree())
>
auto deref_path_impl(
	const std::string& path, sp_link L, sp_node root = nullptr, bool follow_lazy_links = true,
	level_deref_f deref_f = gen_walk_down_tree()
) -> sp_link {
	// split path into elements
	if(path.empty()) return nullptr;
	std::vector<std::string> path_parts;
	boost::split(path_parts, path, boost::is_any_of("/"));

	// setup search root
	if(path_parts[0].empty()) {
		// absolute path case
		root = root ? find_root(root) : find_root(L);
	}
	if(root) L = root->handle();

	// deref each element
	for(const auto& part : path_parts) {
		bool is_control_elem = false;
		if(part.empty() || part == ".")
			is_control_elem = true;
		else if(part == "..") {
			root = L ? L->owner() : nullptr;
			is_control_elem = true;
		}
		else if(!root)
			root = L && (follow_lazy_links || can_call_dnode(*L)) ?
				L->data_node() : nullptr;

		if constexpr(DerefControlElements) {
			// intentional ignore of deref return value
			if(is_control_elem) deref_f(part, root);
		}
		if(!is_control_elem) {
			L = root ? deref_f(part, root) : nullptr;
			if(!L) break;
			// force root recalc on next spin
			root.reset();
		}
	}
	return L;
}

NAMESPACE_END(blue_sky::tree::detail)
