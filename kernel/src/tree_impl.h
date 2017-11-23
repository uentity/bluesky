/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief Implementation details of tree-related functions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <bs/tree.h>
#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree) NAMESPACE_BEGIN(detail)

sp_link walk_down_tree(const std::string& cur_lid, const sp_node& level);

template<typename level_process_f = decltype(walk_down_tree)>
sp_link deref_path(
	const std::string& path, const link& l, const level_process_f& proc_f = walk_down_tree
) {
	// split 
	std::vector<std::string> path_parts;
	boost::split(path_parts, path, boost::is_any_of("/"));
	// setup search root
	sp_node root = l.owner();
	if(root && !path_parts[0].size()) {
		// we have absolute path
		// walk up the tree
		// [NOTE] consider a case when path points to different tree?
		auto O = root->self_link();
		while(O) {
			root = O->data_node();
			if(!root) {
				// we should never come here if tree is in correct state
				//BSERROR << log::E("sym_link: Tree is incorrect, can't traverse to root!") << log::end;
				root = l.owner();
				break;
			}
			O = root->self_link();
		}
	}

	// follow the root and find target link
	sp_link res;
	for(const auto& part : path_parts) {
		if(!part.size() || part == ".") continue;
		if(!root) return nullptr;

		// invoke level processing function that should return link to next level
		res = proc_f(part, (const sp_node&)root);

		// calc new root from next level link
		if(part == ".." && (res = root->self_link())) {
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

