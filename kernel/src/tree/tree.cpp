/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief BS tree-related functions impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/tree.h>
#include <bs/objbase.h>
#include <bs/uuid.h>
#include <bs/kernel/types_factory.h>
#include "tree_impl.h"

#include <set>
#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  impl details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()

void walk_impl(
	const std::list<link>& nodes, walk_links_fv step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links,
	std::set<lid_type> active_symlinks = {}
) {
	using detail::can_call_dnode;
	node cur_node;
	std::list<link> next_nodes;
	links_v next_leafs;
	// for each node
	for(const auto& N : nodes) {
		if(!N) continue;
		// remember symlink
		const auto is_symlink = N.type_id() == sym_link::type_id_();
		if(is_symlink && (!follow_symlinks || !active_symlinks.insert(N.id()).second))
			continue;

		// obtain node from link honoring LazyLoad flag
		cur_node = (follow_lazy_links || can_call_dnode(N)) ? N.data_node() : node::nil();

		next_nodes.clear();
		next_leafs.clear();

		if(cur_node) {
			// for each link in node
			for(const auto& l : cur_node.leafs()) {
				// collect nodes
				if((follow_lazy_links || can_call_dnode(l)) && l.is_node()) {
					next_nodes.push_back(l);
				}
				else
					next_leafs.push_back(l);
			}
		}

		// if `topdown` == true, process this node BEFORE leafs
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
			active_symlinks.erase(N.id());
	}
}

void walk_impl(
	const std::list<node>& nodes, walk_nodes_fv step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links,
	std::set<lid_type> active_symlinks = {}
) {
	using detail::can_call_dnode;
	std::list<node> next_nodes;
	links_v next_leafs;

	// for each node
	for(const auto& cur_node : nodes) {
		if(!cur_node) continue;

		next_nodes.clear();
		next_leafs.clear();

		// symlinks among cur_node
		lids_v cur_symlinks;

		// for each link in node
		for(const auto& l : cur_node.leafs()) {
			// remember symlinks
			if(follow_symlinks && l.type_id() == sym_link::type_id_()) {
				if(active_symlinks.insert(l.id()).second)
					cur_symlinks.push_back(l.id());
				else continue;
			}

			// collect nodes
			auto sub_node = follow_lazy_links || can_call_dnode(l) ? l.data_node() : node::nil();
			if(sub_node)
				next_nodes.push_back(sub_node);
			else
				next_leafs.push_back(l);
		}

		// if `topdown` == true, process this node BEFORE leafs processing
		if(topdown)
			step_f(cur_node, next_nodes, next_leafs);
		// process list of next nodes
		if(!next_nodes.empty())
			walk_impl(next_nodes, step_f, topdown, follow_symlinks, follow_lazy_links, active_symlinks);
		// if walking from most deep subdir, process current node after all subtree
		if(!topdown)
			step_f(cur_node, next_nodes, next_leafs);

		// forget symlinks
		for(const auto& sym_id : cur_symlinks)
			active_symlinks.erase(sym_id);
	}
}

inline std::string link2path_unit(const link& l, Key path_unit) {
	switch(path_unit) {
	default:
	case Key::ID : return to_string(l.id());
	case Key::OID : return l.oid();
	case Key::Name : return l.name();
	case Key::Type : return std::string{ l.type_id() };
	}
}

NAMESPACE_END()

NAMESPACE_BEGIN(detail)

auto walk_down_tree(const std::string& cur_lid, const node& level, Key path_unit) -> link {
	if(!level) return {};
	return level.find(cur_lid, path_unit);
}

NAMESPACE_END(detail)

/*-----------------------------------------------------------------------------
 *  public API
 *-----------------------------------------------------------------------------*/
///////////////////////////////////////////////////////////////////////////////
//  abspath
//
auto abspath(link L, Key path_unit) -> std::string {
	// [NOTE] root ID is irrelevant => abs path always starts with '/'
	auto parent = L.owner();
	std::deque<std::string> res;
	while(true) {
		if(parent)
			res.push_front(link2path_unit(L, path_unit));
		else {
			res.emplace_front("");
			break;
		}

		if(( L = parent.handle() ))
			parent = L.owner();
		else
			parent.reset();
	}
	if(res.size() == 1) res.emplace_front("");
	return boost::join(res, "/");
}

///////////////////////////////////////////////////////////////////////////////
//  find_root & find_root_handle
//
auto find_root(link L) -> node {
	if(auto root = L.owner()) {
		while(const auto root_h = root.handle()) {
			if(auto parent = root_h.owner())
				root = std::move(parent);
			else return root;
		}
	}
	// root handle is passed
	return L.data_node();
}

// [NOTE] avoid `data_node()` call
auto find_root(node N) -> node {
	if(N) {
		while(const auto root_h = N.handle()) {
			if(auto parent = root_h.owner())
				N = std::move(parent);
			else break;
		}
	}
	return N;
}

NAMESPACE_BEGIN()

const auto find_root_handle_impl = [](auto L) {
	if(L) {
		while(const auto root = L.owner()) {
			if(auto root_h = root.handle())
				L = std::move(root_h);
			else break;
		}
	}
	return L;
};

NAMESPACE_END()

auto find_root_handle(link L) -> link {
	return find_root_handle_impl(std::move(L));
}

auto find_root_handle(node N) -> link {
	return find_root_handle_impl(N.handle());
}

///////////////////////////////////////////////////////////////////////////////
//  convert_path
//
std::string convert_path(
	std::string src_path, link start,
	Key src_path_unit, Key dst_path_unit,
	bool follow_lazy_links
) {
	std::vector<std::string> res_path;
	const auto converter = [&res_path, src_path_unit, dst_path_unit](
		std::string next_level, const node& cur_level
	) {
		link res;
		if(next_level.empty() || next_level == "." || next_level == "..")
			res_path.emplace_back(std::move(next_level));
		else if(auto next = cur_level.find(next_level, src_path_unit)) {
			res = std::move(next);
			res_path.emplace_back(link2path_unit(res, dst_path_unit));
		}
		return res;
	};

	// do conversion
	boost::trim(src_path);
	if(detail::deref_path_impl<true>(src_path, std::move(start), node::nil(), follow_lazy_links, converter)) {
		return boost::join(res_path, "/");
	}
	return "";
}

///////////////////////////////////////////////////////////////////////////////
//  deref_path
//
link deref_path(
	const std::string& path, link start, Key path_unit, bool follow_lazy_links
) {
	return detail::deref_path_impl(
		path, std::move(start), node::nil(), follow_lazy_links, detail::gen_walk_down_tree(path_unit)
	);
}
link deref_path(
	const std::string& path, node start, Key path_unit, bool follow_lazy_links
) {
	return detail::deref_path_impl(
		path, {}, std::move(start), follow_lazy_links, detail::gen_walk_down_tree(path_unit)
	);
}

///////////////////////////////////////////////////////////////////////////////
//  walk
//
auto walk(
	link root, walk_links_fv step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links
) -> void {
	walk_impl({root}, step_f, topdown, follow_symlinks, follow_lazy_links);
}

auto walk(
	const node& root, walk_links_fv step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links
) -> void {
	if(!root) return;
	auto hr = root.handle();
	if(!hr) hr = hard_link("/", std::make_shared<objnode>(root));
	walk_impl({std::move(hr)}, step_f, topdown, follow_symlinks, follow_lazy_links);
}

auto walk(
	node root, walk_nodes_fv step_f,
	bool topdown, bool follow_symlinks, bool follow_lazy_links
) -> void {
	walk_impl({std::move(root)}, step_f, topdown, follow_symlinks, follow_lazy_links);
}

///////////////////////////////////////////////////////////////////////////////
//  misc
//
auto make_root_link(
	std::string_view link_type, std::string name, sp_obj root_obj
) -> link {
	if(!root_obj) root_obj = std::make_shared<objnode>();
	// create link depending on type
	if(link_type == fusion_link::type_id_())
		return link::make_root<fusion_link>(std::move(name), std::move(root_obj));
	else if(link_type == hard_link::type_id_())
		return link::make_root<hard_link>(std::move(name), std::move(root_obj));
	return link{};
}

NAMESPACE_END(blue_sky::tree)
