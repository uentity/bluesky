/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief BS tree-related functions impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree/tree.h>
#include <bs/defaults.h>
#include <bs/objbase.h>
#include <bs/uuid.h>
#include <bs/kernel/types_factory.h>
#include <bs/kernel/radio.h>
#include <bs/detail/enumops.h>

#include "tree_impl.h"

#include <boost/algorithm/string.hpp>
#include <caf/event_based_actor.hpp>

#include <algorithm>
#include <set>

NAMESPACE_BEGIN(blue_sky::tree)
/*-----------------------------------------------------------------------------
 *  impl details
 *-----------------------------------------------------------------------------*/
NAMESPACE_BEGIN()
using namespace allow_enumops;

void walk_impl(
	const std::list<link>& nodes, walk_links_fv step_f, TreeOpts opts,
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
		if(is_symlink && (!enumval(opts & TreeOpts::FollowSymLinks) || !active_symlinks.insert(N.id()).second))
			continue;

		// obtain node from link honoring LazyLoad flag
		cur_node = can_call_dnode(N, opts) ? N.data_node() : node::nil();

		next_nodes.clear();
		next_leafs.clear();

		if(cur_node) {
			// for each link in node
			for(const auto& l : cur_node.leafs()) {
				// collect nodes
				if(can_call_dnode(l, opts) && l.is_node())
					next_nodes.push_back(l);
				else
					next_leafs.push_back(l);
			}
		}

		// if walking down the tree, process this node BEFORE leafs
		if(!enumval(opts & TreeOpts::WalkUp))
			step_f(N, next_nodes, next_leafs);
		// process list of next nodes
		if(!next_nodes.empty())
			walk_impl(next_nodes, step_f, opts, active_symlinks);
		// if walking up, process current node after all subtree
		if(enumval(opts & TreeOpts::WalkUp))
			step_f(N, next_nodes, next_leafs);

		// forget symlink
		if(is_symlink)
			active_symlinks.erase(N.id());
	}
}

void walk_impl(
	const std::list<node>& nodes, walk_nodes_fv step_f, TreeOpts opts,
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
			if(enumval(opts & TreeOpts::FollowSymLinks) && l.type_id() == sym_link::type_id_()) {
				if(active_symlinks.insert(l.id()).second)
					cur_symlinks.push_back(l.id());
				else continue;
			}

			// collect nodes
			auto sub_node = enumval(opts & TreeOpts::FollowSymLinks) || can_call_dnode(l, opts) ?
				l.data_node() : node::nil();
			if(sub_node)
				next_nodes.push_back(sub_node);
			else
				next_leafs.push_back(l);
		}

		// if `topdown` == true, process this node BEFORE leafs processing
		if(!enumval(opts & TreeOpts::WalkUp))
			step_f(cur_node, next_nodes, next_leafs);
		// process list of next nodes
		if(!next_nodes.empty())
			walk_impl(next_nodes, step_f, opts, active_symlinks);
		// if walking from most deep subdir, process current node after all subtree
		if(!!enumval(opts & TreeOpts::WalkUp))
			step_f(cur_node, next_nodes, next_leafs);

		// forget symlinks
		for(const auto& sym_id : cur_symlinks)
			active_symlinks.erase(sym_id);
	}
}

std::string link2path_unit(const link& l, Key path_unit) {
	switch(path_unit) {
	default:
	case Key::ID : return to_string(l.id());
	case Key::OID : return l.oid();
	case Key::Name : return l.name();
	case Key::Type : return std::string{ l.type_id() };
	}
}

NAMESPACE_END()

/*-----------------------------------------------------------------------------
 *  utilities
 *-----------------------------------------------------------------------------*/
auto to_string(const lids_v& path, bool as_absolute) -> std::string {
	auto res = std::vector<std::string>(as_absolute ? path.size() + 1 : path.size());
	std::transform(
		path.begin(), path.end(),
		as_absolute ? std::next(res.begin()) : res.begin(),
		[](const auto& Lid) { return to_string(Lid); }
	);
	return boost::join(res, "/");
}

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
			res.emplace_front();
			break;
		}

		if(( L = parent.handle() ))
			parent = L.owner();
		else
			parent.reset();
	}
	if(res.size() == 1) res.emplace_front();
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
	Key src_path_unit, Key dst_path_unit, TreeOpts opts
) {
	std::vector<std::string> res_path;
	const auto converter = [&res_path, src_path_unit, dst_path_unit](
		std::string_view next_level, const node& cur_level
	) {
		link res;
		if(next_level.empty() || next_level == "." || next_level == "..")
			res_path.emplace_back(std::move(next_level));
		else if(auto next = cur_level.find(std::string{next_level}, src_path_unit)) {
			res = std::move(next);
			res_path.emplace_back(link2path_unit(res, dst_path_unit));
		}
		return res;
	};

	// do conversion
	boost::trim(src_path);
	if(detail::deref_path_impl<true>(src_path, std::move(start), node::nil(), opts, converter)) {
		return boost::join(res_path, "/");
	}
	return "";
}

///////////////////////////////////////////////////////////////////////////////
//  deref_path
//
auto deref_path(
	const std::string& path, link start, Key path_unit, TreeOpts opts
) -> link {
	return detail::deref_path_impl(
		path, std::move(start), node::nil(), opts, detail::gen_walk_down_tree(path_unit)
	);
}

auto deref_path(
	const std::string& path, node start, Key path_unit, TreeOpts opts
) -> link {
	return detail::deref_path_impl(
		path, {}, std::move(start), opts, detail::gen_walk_down_tree(path_unit)
	);
}

auto deref_path(
	deref_process_f f, std::string path, link start, Key path_unit, TreeOpts opts
) -> void {

	auto work = [=, f = std::move(f), path = std::move(path)]() {
		f(detail::deref_path_impl(path, start, node::nil(), opts, detail::gen_walk_down_tree(path_unit)));
	};

	kernel::radio::system().spawn(std::move(work));
}

///////////////////////////////////////////////////////////////////////////////
//  walk
//
auto walk(link root, walk_links_fv step_f, TreeOpts opts) -> void {
	walk_impl({root}, step_f, opts);
}

auto walk(const node& root, walk_links_fv step_f, TreeOpts opts) -> void {
	if(!root) return;
	auto hr = root.handle();
	if(!hr) hr = hard_link("/", std::make_shared<objnode>(root));
	walk_impl({std::move(hr)}, step_f, opts);
}

auto walk(node root, walk_nodes_fv step_f, TreeOpts opts) -> void {
	walk_impl({std::move(root)}, step_f, opts);
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

auto owner(const objbase& obj) -> node {
	return owner(obj.data_node());
}

NAMESPACE_END(blue_sky::tree)
