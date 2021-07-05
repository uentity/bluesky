/// @author uentity
/// @date 22.11.2017
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include <bs/tree/tree.h>
#include <bs/detail/enumops.h>

#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky::tree::detail)
using namespace std::string_view_literals;

inline auto walk_down_tree(Key path_unit, std::string_view next_lid, const node& cur_level) -> link {
	if(!cur_level) return {};
	return cur_level.find(std::string{next_lid}, path_unit);
}

// put into hidden namespace to prevent equal multiple instantiations
inline auto gen_walk_down_tree(Key path_unit = Key::ID) {
	return [path_unit](std::string_view next_lid, const node& cur_level) {
		return walk_down_tree(path_unit, next_lid, cur_level);
	};
}

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
	std::string_view path, link L, node root = node::nil(), TreeOpts opts = TreeOpts::Normal,
	level_deref_f deref_f = gen_walk_down_tree()
) -> link {
	using namespace allow_enumops;
	using namespace std::string_view_literals;

	// split path into elements
	if(path.empty()) return {};
	auto path_parts = std::vector< std::pair<std::string::const_iterator, std::string::const_iterator> >{};
	boost::split(path_parts, path, boost::is_any_of("/"));

	// setup search root
	if(path_parts[0].first == path_parts[0].second) {
		// absolute path case
		root = root ? find_root(root) : find_root(L);
	}
	if(root) L = root.handle();

	// deref each element
	for(const auto& part_tok : path_parts) {
		const auto part = std::string_view{
			&*part_tok.first, static_cast<std::size_t>(part_tok.second - part_tok.first)
		};
		bool is_control_elem = false;
		if(part.empty() || part == "."sv)
			is_control_elem = true;
		else if(part == ".."sv) {
			root = L ? L.owner() : node::nil();
			is_control_elem = true;
		}
		else if(!root)
			root = L && can_call_dnode(L, opts) ? L.data_node() : node::nil();

		if(is_control_elem) {
			if constexpr(DerefControlElements) {
				// intentional ignore of deref return value
				deref_f(part, root);
			}
			else continue;
		}

		L = root ? deref_f(part, root) : link{};
		if(!L) break;
		// force root recalc on next spin
		root.reset();
	}
	return L;
}

NAMESPACE_END(blue_sky::tree::detail)
