/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief BS tree-related functions impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/tree.h>
#include "tree_impl.h"

#include <set>
#include <boost/uuid/uuid_io.hpp>
#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

using Key = node::Key;
template<Key K> using Key_type = typename node::Key_type<K>;

/*-----------------------------------------------------------------------------
 *  impl details
 *-----------------------------------------------------------------------------*/
// hidden
namespace {

template<class Callback>
void walk_impl(
	const std::vector<sp_link>& nodes, const Callback& step_f,
	const bool topdown, const bool follow_symlinks,
	std::set<Key_type<Key::ID>> active_symlinks = {}
) {
	sp_node cur_node;
	std::vector<sp_link> next_nodes;
	std::vector<sp_link> next_leafs;
	// for each node
	for(const auto& N : nodes) {
		if(!N || !(cur_node = N->data_node())) continue;
		// remember symlink
		const auto is_symlink = N->type_id() == "sym_link";
		if(is_symlink){
			if(follow_symlinks && active_symlinks.find(N->id()) == active_symlinks.end())
				active_symlinks.insert(N->id());
			else continue;
		}

		next_nodes.clear();
		next_leafs.clear();
		// for each link in node
		for(const auto& l : *cur_node) {
			// collect nodes
			if(l->data_node()) {
				next_nodes.push_back(l);
			}
			else
				next_leafs.push_back(l);
		}

		// if `topdown` == true, process this node BEFORE leafs processing
		if(topdown)
			step_f(N, next_nodes, next_leafs);
		// process list of next nodes
		walk_impl(next_nodes, step_f, topdown, follow_symlinks, active_symlinks);
		// if walking from most deep subdir, process current node after all subtree
		if(!topdown)
			step_f(N, next_nodes, next_leafs);

		// forget symlink
		if(is_symlink)
			active_symlinks.erase(N->id());
	}
}

inline std::string link2path_unit(const sp_clink& l, Key path_unit) {
	switch(path_unit) {
	default:
	case Key::ID : return boost::uuids::to_string(l->id());
	case Key::OID : return l->oid();
	case Key::Name : return l->name();
	case Key::Type : return l->type_id();
	}
}

} // hidden ns

// public
NAMESPACE_BEGIN(detail)

sp_link walk_down_tree(const std::string& cur_lid, const sp_node& level, node::Key path_unit) {
	if(cur_lid != "..") {
		const auto next = level->find(cur_lid, path_unit);
		if(next != level->end()) {
			return *next;
		}
	}
	return nullptr;
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  public API
 *-----------------------------------------------------------------------------*/
std::string abspath(sp_clink l, Key path_unit) {
	std::deque<std::string> res;
	while(l) {
		res.push_front(link2path_unit(l, path_unit));
		if(const auto parent = l->owner()) {
			l = parent->handle();
		}
		else return boost::join(res, "/");
	}
	// leadind slash is appended only if we have 'real' root node without self link
	return std::string("/") + boost::join(res, "/");

	// another possible implementation without multiple returns
	// just leave it here -)
	//sp_node parent;
	//do {
	//	if(l)
	//		res.push_front(human_readable ? l->name() : boost::uuids::to_string(l->id()));
	//	else {
	//		// for root node
	//		res.emplace_front("");
	//		break;
	//	}
	//	if((parent = l->owner())) {
	//		l = parent->handle();
	//	}
	//} while(parent);
	//return boost::join(res, "/");
}

std::string convert_path(
	std::string src_path, const sp_clink& start,
	Key src_path_unit, Key dst_path_unit
) {
	std::string res_path;
	// convert from ID-based path to human-readable
	const auto converter = [&res_path, src_path_unit, dst_path_unit](std::string part, const sp_node& level) {
		sp_link res;
		if(part != "..") {
			const auto next = level->find(part, src_path_unit);
			if(next != level->end()) {
				res = *next;
				part = link2path_unit(res, dst_path_unit);
			}
		}
		// append link ID to link's path
		if(res_path.size()) res_path += '/';
		res_path += std::move(part);
		return res;
	};

	// do conversion
	boost::trim(src_path);
	detail::deref_path(src_path, *start, converter);
	// if abs path given, return abs path
	if(src_path[0] == '/')
		res_path.insert(res_path.begin(), '/');
	return res_path;
}

sp_link deref_path(const std::string& path, const sp_clink& start, node::Key path_unit) {
	if(!start) return nullptr;
	return detail::deref_path(path, *start, detail::gen_walk_down_tree(path_unit));
}

void walk(const sp_link& root, step_process_fp step_f, bool topdown, bool follow_symlinks) {
	walk_impl({root}, step_f, topdown, follow_symlinks);
}

void walk(const sp_link& root, const step_process_f& step_f,bool topdown, bool follow_symlinks) {
	walk_impl({root}, step_f, topdown, follow_symlinks);
}


NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

