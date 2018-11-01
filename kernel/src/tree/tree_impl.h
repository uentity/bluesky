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

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree) NAMESPACE_BEGIN(detail)

sp_link walk_down_tree(
	const std::string& cur_lid, const sp_node& level, node::Key path_unit = node::Key::ID
);

NAMESPACE_BEGIN()
// put into hidden namespace to prevent equal multiple instantiations
auto gen_walk_down_tree(node::Key path_unit = node::Key::ID) {
	return [path_unit](const std::string& cur_lid, const sp_node& level) {
		return walk_down_tree(cur_lid, level, path_unit);
	};
}

//inline auto can_call_dnode(const link& L) -> bool {
//	return ( !(L.flags() & link::LazyLoad) || L.req_status(link::Req::DataNode) == link::ReqStatus::OK );
//}

NAMESPACE_END()

template< typename level_process_f = decltype(gen_walk_down_tree()) >
sp_link deref_path(
	const std::string& path, const link& l, level_process_f&& proc_f = gen_walk_down_tree()
) {
	// split 
	std::vector<std::string> path_parts;
	boost::split(path_parts, path, boost::is_any_of("/"));
	// setup search root
	sp_node root = l.owner();
	if(!root) {
		// given link points to tree root?
		root = l.data_node();
	}
	else if(!path_parts[0].size()) {
		// link is inside the tree and we have absolute path
		// walk up the tree to find root node
		sp_link h_root;
		while(root && (h_root = root->handle())) {
			root = h_root->owner();
		}
	}

	// follow the root and find target link
	sp_link res;
	for(const auto& part : path_parts) {
		if(!root) return nullptr;
		if(!part.size() || part == ".") continue;

		// invoke level processing function that should return link to next level
		res = proc_f(part, (const sp_node&)root);

		// calc new root from next level link
		if(part == ".." && (res = root->handle())) {
			// go up one level
			root = res->owner();
		}
		else if(res) {
			root = res->data_node();
		}
		else root.reset();
	}
	return res;
}

NAMESPACE_END(detail) NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

