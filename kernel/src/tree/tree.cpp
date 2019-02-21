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
NAMESPACE_BEGIN()

template<class Callback>
void walk_impl(
	const std::list<sp_link>& nodes, const Callback& step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links,
	std::set<Key_type<Key::ID>> active_symlinks = {}
) {
	using detail::can_call_dnode;
	sp_node cur_node;
	std::list<sp_link> next_nodes;
	std::vector<sp_link> next_leafs;
	// for each node
	for(const auto& N : nodes) {
		if(!N) continue;
		// remember symlink
		const auto is_symlink = N->type_id() == "sym_link";
		if(is_symlink) {
			if(follow_symlinks && active_symlinks.find(N->id()) == active_symlinks.end())
				active_symlinks.insert(N->id());
			else continue;
		}

		// obtain node from link honoring LazyLoad flag
		cur_node = (follow_lazy_links || can_call_dnode(*N)) ? N->data_node() : nullptr;

		next_nodes.clear();
		next_leafs.clear();

		if(cur_node) {
			// for each link in node
			for(const auto& l : *cur_node) {
				// collect nodes
				if((follow_lazy_links || can_call_dnode(*l)) && l->data_node()) {
					next_nodes.push_back(l);
				}
				else
					next_leafs.push_back(l);
			}
		}

		// if `topdown` == true, process this node BEFORE leafs processing
		if(topdown)
			step_f(N, next_nodes, next_leafs);
		// process list of next nodes
		if(!next_nodes.empty())
			walk_impl(next_nodes, step_f, topdown, follow_symlinks, follow_lazy_links, active_symlinks);
		// if walking from most deep subdir, process current node after all subtree
		if(!topdown)
			step_f(N, next_nodes, next_leafs);

		// forget symlink
		if(is_symlink)
			active_symlinks.erase(N->id());
	}
}

inline std::string link2path_unit(const link& l, Key path_unit) {
	switch(path_unit) {
	default:
	case Key::ID : return boost::uuids::to_string(l.id());
	case Key::OID : return l.oid();
	case Key::Name : return l.name();
	case Key::Type : return l.type_id();
	}
}

NAMESPACE_END()

NAMESPACE_BEGIN(detail)

sp_link walk_down_tree(const std::string& cur_lid, const sp_node& level, node::Key path_unit) {
	if(!level) return nullptr;
	const auto next = level->find(cur_lid, path_unit);
	return next != level->end() ? *next : nullptr;
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  public API
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  abspath
//
auto abspath(const link& L, Key path_unit) -> std::string {
	std::deque<std::string> res{link2path_unit(L, path_unit)};
	auto parent = L.owner();
	while(parent) {
		if(const auto parent_h = parent->handle()) {
			res.push_front(link2path_unit(*parent_h, path_unit));
			parent = parent_h->owner();
		}
		else {
			res.emplace_front("");
			break;
		}
	}
	// [NOTE] root ID is irrelevant => abs path always starts with '/'
	res[0].clear();
	return boost::join(res, "/");
}

auto abspath(const sp_clink& L, Key path_unit) -> std::string {
	return L ? abspath(*L) : "";
}

///////////////////////////////////////////////////////////////////////////////
//  find_root & find_root_handle
//
auto find_root(link& L) -> sp_node {
	if(auto root = L.owner()) {
		while(const auto root_h = root->handle()) {
			if(auto parent = root_h->owner())
				root = std::move(parent);
			else return root;
		}
	}
	// root handle is passed
	return L.data_node();
}

auto find_root(const sp_link& L) -> sp_node {
	return L ? find_root(*L) : nullptr;
}

// [NOTE] avoid `data_node()` call
auto find_root(sp_node N) -> sp_node {
	if(N) {
		while(const auto root_h = N->handle()) {
			if(auto parent = root_h->owner())
				N = std::move(parent);
			else break;
		}
	}
	return N;
}

NAMESPACE_BEGIN()

const auto find_root_handle_impl = [](auto L) {
	if(L) {
		while(const auto root = L->owner()) {
			if(auto root_h = root->handle())
				L = std::move(root_h);
			else break;
		}
	}
	return L;
};

NAMESPACE_END()

auto find_root_handle(sp_link L) -> sp_link {
	return find_root_handle_impl(std::move(L));
}
auto find_root_handle(sp_clink L) -> sp_clink {
	return find_root_handle_impl(std::move(L));
}
auto find_root_handle(const sp_node& N) -> sp_link {
	return find_root_handle_impl(N->handle());
}

///////////////////////////////////////////////////////////////////////////////
//  convert_path
//
std::string convert_path(
	std::string src_path, sp_link start,
	Key src_path_unit, Key dst_path_unit,
	bool follow_lazy_links
) {
	std::vector<std::string> res_path;
	const auto converter = [&res_path, src_path_unit, dst_path_unit](
		std::string next_level, const sp_node& cur_level
	) {
		sp_link res;
		if(next_level.empty() || next_level == "." || next_level == "..")
			res_path.emplace_back(std::move(next_level));
		else {
			const auto next = cur_level->find(next_level, src_path_unit);
			if(next != cur_level->end()) {
				res = *next;
				res_path.emplace_back(link2path_unit(*res, dst_path_unit));
			}
		}
		return res;
	};

	// do conversion
	boost::trim(src_path);
	if(detail::deref_path_impl<true>(src_path, std::move(start), nullptr, follow_lazy_links, converter)) {
		return boost::join(res_path, "/");
	}
	return "";
}

///////////////////////////////////////////////////////////////////////////////
//  deref_path
//
sp_link deref_path(
	const std::string& path, sp_link start, node::Key path_unit, bool follow_lazy_links
) {
	return detail::deref_path_impl(
		path, std::move(start), nullptr, follow_lazy_links, detail::gen_walk_down_tree(path_unit)
	);
}
sp_link deref_path(
	const std::string& path, sp_node start, node::Key path_unit, bool follow_lazy_links
) {
	return detail::deref_path_impl(
		path, nullptr, std::move(start), follow_lazy_links, detail::gen_walk_down_tree(path_unit)
	);
}

///////////////////////////////////////////////////////////////////////////////
//  walk
//
auto walk(
	const sp_link& root, step_process_fp step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links
) -> void {
	walk_impl({root}, step_f, topdown, follow_symlinks, follow_lazy_links);
}

auto walk(
	const sp_link& root, const step_process_f& step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links
) -> void {
	walk_impl({root}, step_f, topdown, follow_symlinks, follow_lazy_links);
}


auto make_root_link(
	const std::string& link_type, std::string name, sp_node root_node
) -> sp_link {
	if(!root_node) root_node = std::make_shared<node>();
	// create link depending on type
	if(link_type == "fusion_link")
		return link::make_root<fusion_link>(std::move(name), std::move(root_node));
	else
		return link::make_root<hard_link>(std::move(name), std::move(root_node));
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

