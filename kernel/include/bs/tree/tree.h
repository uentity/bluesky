/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief blue-sky tree summary header
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "errors.h"
#include "link.h"
#include "fusion.h"
#include "node.h"
#include "type_caf_id.h"
#include "../detail/function_view.h"
#include "../detail/enumops.h"

#include <list>

NAMESPACE_BEGIN(blue_sky::tree)

inline constexpr auto def_deref_opts = TreeOpts::FollowLazyLinks;
inline constexpr auto def_walk_opts = TreeOpts::FollowSymLinks;

/*-----------------------------------------------------------------------------
 *  Sync API
 *-----------------------------------------------------------------------------*/
/// returns absolute link's path consisting from `path_unit` separated by '/'
/// root ID isn't included, so absolute path always starts with '/'
BS_API auto abspath(link L, Key path_unit = Key::ID) -> std::string;

/// find root node from given link (can call `data_node()` if and only if root handle is passed)
BS_API auto find_root(link L) -> node;
/// ... and from given node (avoids call to `data_node()` at all)
BS_API auto find_root(node N) -> node;

/// same, but returns link to root node -- root's handle
BS_API auto find_root_handle(link L) -> link;
BS_API auto find_root_handle(node N) -> link;

/// convert path one path representaion to another
BS_API auto convert_path(
	std::string src_path, link start,
	Key src_path_unit = Key::ID, Key dst_path_unit = Key::Name,
	TreeOpts opts = TreeOpts::Normal
) -> std::string;

/// quick link search by given absolute or relative path
/// may be faster that full `node::deep_search()`
/// also can lookup starting from any tree node given absolute path
BS_API link deref_path(
	const std::string& path, link start, Key path_unit = Key::ID,
	TreeOpts opts = def_deref_opts
);
/// sometimes it may be more convinient
BS_API link deref_path(
	const std::string& path, node start, Key path_unit = Key::ID,
	TreeOpts opts = def_deref_opts
);

/// walk the tree just like the Python's `os.walk` is implemented
using walk_links_fv = function_view<void (link, std::list<link>&, std::vector<link>&)>;
BS_API void walk(
	link root, walk_links_fv step_f, TreeOpts opts = def_walk_opts
);

/// alt walk implementation
using walk_nodes_fv = function_view<void (node, std::list<node>&, std::vector<link>&)>;
BS_API void walk(
	node root, walk_nodes_fv step_f, TreeOpts opts = def_walk_opts
);

/*-----------------------------------------------------------------------------
 *  Async API
 *-----------------------------------------------------------------------------*/
/// deferred `deref_path` that accepts callback
using deref_process_f = std::function<void(link)>;

BS_API auto deref_path(
	deref_process_f f,
	std::string path, link start, Key path_unit = Key::ID, TreeOpts opts = def_deref_opts
) -> void;

/*-----------------------------------------------------------------------------
 *  Walk up the tree by jumping over owners
 *-----------------------------------------------------------------------------*/
/// unify access to owner of link or node
inline auto owner(const link& lnk) {
	return lnk ? lnk.owner() : node::nil();
}
inline auto owner(const node& N) {
	if(auto h = N.handle())
		return h.owner();
	return node::nil();
}

BS_API auto owner(const objbase& obj) -> node;
inline auto owner(const objbase* obj) {
	return obj ? owner(*obj) : node::nil();
}

template<typename T>
auto owner(const std::shared_ptr<T>& obj) -> std::enable_if_t<std::is_base_of_v<objnode, T>> {
	return owner(obj.get());
}

/// obtain link to node that contains `self` (link or node)
template<typename T>
auto owner_handle(const T& self) {
	if(auto super = owner(self))
		return super.handle();
	return link{};
}

/// obtain parent object (inhertied from `objnode`) that contains onwer node
template<typename R = objnode, typename T>
auto owner_obj(const T& self) -> std::shared_ptr<R> {
	if(auto H = owner_handle(self))
		return std::static_pointer_cast<R>(H.data());
	return nullptr;
}

/// produce link to given object, if object contains node it's handle will point to returned link
/// if `root_obj` is omitted, empty `objnode` will be substituted
BS_API auto make_root_link(
	std::string_view link_type, std::string name = "/", sp_obj root_obj = nullptr
) -> link;

/// same as above, but with `root_obj` as first arg and hard link by default
inline auto make_root_link(sp_obj root_obj, std::string name = "/", std::string_view link_type = {}) {
	return make_root_link(
		link_type.empty() ? hard_link::type_id_() : link_type, std::move(name), std::move(root_obj)
	);
}

/*-----------------------------------------------------------------------------
 *  Tree save/load to different archives
 *-----------------------------------------------------------------------------*/
enum class TreeArchive { Text, Binary, FS };
using on_serialized_f = std::function<void(link, error)>;

BS_API auto save_tree(
	link root, std::string filename,
	TreeArchive ar = TreeArchive::FS, timespan wait_for = infinite
) -> error;
// async save
BS_API auto save_tree(
	on_serialized_f callback, link root, std::string filename,
	TreeArchive ar = TreeArchive::FS
) -> void;

BS_API auto load_tree(
	std::string filename, TreeArchive ar = TreeArchive::FS
) -> link_or_err;
// async load
BS_API auto load_tree(
	on_serialized_f callback, std::string filename,
	TreeArchive ar = TreeArchive::FS
) -> void;

NAMESPACE_END(blue_sky::tree)
