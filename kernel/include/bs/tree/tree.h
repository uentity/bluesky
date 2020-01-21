/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief blue-sky tree summary header
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "link.h"
#include "fusion.h"
#include "node.h"
#include "errors.h"
#include "../detail/function_view.h"

#include <list>

NAMESPACE_BEGIN(blue_sky::tree)

/*-----------------------------------------------------------------------------
 *  Sync API
 *-----------------------------------------------------------------------------*/
/// returns absolute link's path consisting from `path_unit` separated by '/'
/// root ID isn't included, so absolute path always starts with '/'
BS_API auto abspath(const link& L, Key path_unit = Key::ID) -> std::string;
BS_API auto abspath(const sp_clink& L, Key path_unit = Key::ID) -> std::string;

/// find root node from given link (can call `data_node()` if and only if root handle is passed)
BS_API auto find_root(link& L) -> sp_node;
BS_API auto find_root(const sp_link& L) -> sp_node;
/// ... and from given node (avoids call to `data_node()` at all)
BS_API auto find_root(sp_node N) -> sp_node;

/// same, but returns link to root node -- root's handle
BS_API auto find_root_handle(sp_link L) -> sp_link;
BS_API auto find_root_handle(sp_clink L) -> sp_clink;
BS_API auto find_root_handle(const sp_node& N) -> sp_link;

/// convert path one path representaion to another
BS_API auto convert_path(
	std::string src_path, sp_link start,
	Key src_path_unit = Key::ID, Key dst_path_unit = Key::Name,
	bool follow_lazy_links = false
) -> std::string;

/// quick link search by given absolute or relative path
/// may be faster that full `node::deep_search()`
/// also can lookup starting from any tree node given absolute path
BS_API sp_link deref_path(
	const std::string& path, sp_link start, Key path_unit = Key::ID,
	bool follow_lazy_links = true
);
/// sometimes it may be more convinient
BS_API sp_link deref_path(
	const std::string& path, sp_node start, Key path_unit = Key::ID,
	bool follow_lazy_links = true
);

/// walk the tree just like the Python's `os.walk` is implemented
using walk_links_fv = function_view<void (const sp_link&, std::list<sp_link>&, std::vector<sp_link>&)>;
BS_API void walk(
	const sp_link& root, walk_links_fv step_f,
	bool topdown = true, bool follow_symlinks = true, bool follow_lazy_links = false
);

/// alt walk implementation
using walk_nodes_fv = function_view<void (const sp_node&, std::list<sp_node>&, std::vector<sp_link>&)>;
BS_API void walk(
	const sp_node& root, walk_nodes_fv step_f,
	bool topdown = true, bool follow_symlinks = true, bool follow_lazy_links = false
);

/*-----------------------------------------------------------------------------
 *  Async API
 *-----------------------------------------------------------------------------*/
/// deferred `deref_path` that accepts callback
using deref_process_f = std::function<void(const sp_link&)>;

BS_API auto deref_path(
	deref_process_f f,
	std::string path, sp_link start, Key path_unit = Key::ID,
	bool follow_lazy_links = true, bool high_priority = false
) -> void;

/*-----------------------------------------------------------------------------
 *  Misc utility functions
 *-----------------------------------------------------------------------------*/
// unify access to owner of link or node
inline auto owner(const link* lnk) {
	return lnk->owner();
}
inline auto owner(const node* N) {
	const auto& h = N->handle();
	return h ? h->owner() : nullptr;
}
// for shared ptrs
template<typename T>
auto owner(const std::shared_ptr<T>& obj) {
	return owner(obj.get());
}
// with conversion to target type
template<typename TargetOwner, typename T>
auto owner_t(T&& obj) {
	return std::static_pointer_cast<TargetOwner>( owner(std::forward<T>(obj)) );
}

/// make root link to node which handle is preset to returned link
BS_API auto make_root_link(
	const std::string& link_type, std::string name = "/",
	sp_node root_node = nullptr
) -> sp_link;

///////////////////////////////////////////////////////////////////////////////
//  Tree save/load to JSON or binary archive
//
enum class TreeArchive { Text, Binary, FS };
using on_serialized_f = std::function<void(sp_link, error)>;

BS_API auto save_tree(
	sp_link root, std::string filename,
	TreeArchive ar = TreeArchive::FS, timespan wait_for = infinite
) -> error;
// async save
BS_API auto save_tree(
	on_serialized_f callback, sp_link root, std::string filename,
	TreeArchive ar = TreeArchive::FS
) -> void;

BS_API auto load_tree(
	std::string filename, TreeArchive ar = TreeArchive::FS
) -> result_or_err<sp_link>;
// async load
BS_API auto load_tree(
	on_serialized_f callback, std::string filename,
	TreeArchive ar = TreeArchive::FS
) -> void;

NAMESPACE_END(blue_sky::tree)

