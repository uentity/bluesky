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

#include <list>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/*-----------------------------------------------------------------------------
 *  Sync API
 *-----------------------------------------------------------------------------*/
/// returns absolute link's path consisting from `path_unit` separated by '/'
/// root ID isn't included, so absolute path always starts with '/'
BS_API auto abspath(const link& L, node::Key path_unit = node::Key::ID) -> std::string;
BS_API auto abspath(const sp_clink& L, node::Key path_unit = node::Key::ID) -> std::string;

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
	node::Key src_path_unit = node::Key::ID, node::Key dst_path_unit = node::Key::Name,
	bool follow_lazy_links = false
) -> std::string;

/// quick link search by given absolute or relative path
/// may be faster that full `node::deep_search()`
/// also can lookup starting from any tree node given absolute path
BS_API sp_link deref_path(
	const std::string& path, sp_link start, node::Key path_unit = node::Key::ID,
	bool follow_lazy_links = true
);
/// sometimes it may be more convinient
BS_API sp_link deref_path(
	const std::string& path, sp_node start, node::Key path_unit = node::Key::ID,
	bool follow_lazy_links = true
);

/// walk the tree just like the Python's `os.walk` is implemented
using step_process_fp = void (*)(const sp_link&, std::list<sp_link>&, std::vector<sp_link>&);
BS_API void walk(
	const sp_link& root, step_process_fp step_f,
	bool topdown = true, bool follow_symlinks = true, bool follow_lazy_links = false
);
// overload for std::function instead of function pointer
using step_process_f = std::function<void(const sp_link&, std::list<sp_link>&, std::vector<sp_link>&)>;
BS_API void walk(
	const sp_link& root, const step_process_f& step_f,
	bool topdown = true, bool follow_symlinks = true, bool follow_lazy_links = false
);

/*-----------------------------------------------------------------------------
 *  Async API
 *-----------------------------------------------------------------------------*/
/// deferred `deref_path` that accepts callback
using deref_process_f = std::function<void(const sp_link&)>;
// accept std::function callback
BS_API auto deref_path(
	deref_process_f f,
	std::string path, sp_link start, node::Key path_unit = node::Key::ID,
	bool follow_lazy_links = true, bool high_priority = false
) -> void;
// [TODO] can't be compiled for some reason, enable after problem is solved
// same as above but accept function pointer
//using deref_process_fp = void (*)(const sp_link&);
//BS_API auto deref_path(
//	deref_process_fp f,
//	std::string path, sp_link start, node::Key path_unit = node::Key::ID
//) -> void;

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
enum class TreeArchive { Text, Binary };
BS_API auto
	save_tree(const sp_link& root, const std::string& filename, TreeArchive ar = TreeArchive::Text)
-> error;
BS_API auto
	load_tree(const std::string& filename, TreeArchive ar = TreeArchive::Text)
-> result_or_err<sp_link>;

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

