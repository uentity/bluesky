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

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/// returns absolute link's path consisting from link IDs separated by '/'
/// if `human_readable` is set, replace IDs with link names
BS_API std::string abspath(sp_clink l, node::Key path_unit = node::Key::ID);

/// convert path from ID-based to human readable (name-bases) or vice versa
BS_API std::string convert_path(
	std::string src_path, const sp_clink& start,
	node::Key src_path_unit = node::Key::ID, node::Key dst_path_unit = node::Key::Name
);

/// quick link search by given absolute or relative path (ID-based!)
/// may be faster that full `node::deep_search()`
/// also can lookup starting from any tree node given absolute path
BS_API sp_link deref_path(const std::string& path, const sp_clink& start, node::Key path_unit = node::Key::ID);

/// walk the tree just like the Python's `os.walk` is implemented
using step_process_fp = void (*)(const sp_link&, std::vector<sp_link>&, std::vector<sp_link>&);
BS_API void walk(
	const sp_link& root, step_process_fp step_f,
	bool topdown = true, bool follow_symlinks = true
);
// overload for std::function instead of function pointer
using step_process_f = std::function<void(const sp_link&, std::vector<sp_link>&, std::vector<sp_link>&)>;
BS_API void walk(
	const sp_link& root, const step_process_f& step_f,
	bool topdown = true, bool follow_symlinks = true
);

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

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

