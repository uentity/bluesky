/// @file
/// @author uentity
/// @date 22.11.2017
/// @brief BS tree-related functions impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/tree.h>
#include "tree_impl.h"

#include <boost/uuid/string_generator.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/algorithm/string.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)
/*-----------------------------------------------------------------------------
 *  impl details
 *-----------------------------------------------------------------------------*/
// hidden
namespace {

static boost::uuids::string_generator uuid_from_str;

} // hidden ns

// public
NAMESPACE_BEGIN(detail)

sp_link walk_down_tree(const std::string& cur_lid, const sp_node& level) {
	if(cur_lid != "..") {
		const auto next = level->find(uuid_from_str(cur_lid));
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
std::string abspath(sp_clink l, bool human_readable) {
	std::deque<std::string> res;
	while(l) {
		res.push_front(human_readable ? l->name() : boost::uuids::to_string(l->id()));
		if(const auto parent = l->owner()) {
			l = parent->self_link();
		}
	}
	return std::string("/") + boost::join(res, "/");
}

std::string convert_path(std::string src_path, const sp_clink& start, bool from_human_readable) {
	std::string res_path;
	// convert from ID-based path to human-readable
	const auto id2name = [&res_path](std::string part, const sp_node& level) {
		sp_link res;
		if(part != "..") {
			const auto next = level->find(part);
			if(next != level->end()) {
				res = *next;
				part = boost::uuids::to_string(res->id());
			}
		}
		// append link ID to link's path
		if(res_path.size()) res_path += '/';
		res_path += std::move(part);
		return res;
	};
	// convert from name-based to ID-based path
	const auto name2id = [&res_path](std::string part, const sp_node& level) {
		sp_link res;
		if(part != "..") {
			auto next = level->find(uuid_from_str(part));
			if(next != level->end()) {
				res = *next;
				part = res->name();
			}
		}
		// append link ID to link's path
		if(res_path.size()) res_path += '/';
		res_path += std::move(part);
		return res;
	};

	// do conversion
	boost::trim(src_path);
	from_human_readable ?
		detail::deref_path(src_path, *start, name2id) :
		detail::deref_path(src_path, *start, id2name);
	// if abs path given, return abs path
	if(src_path[0] == '/')
		res_path.insert(res_path.begin(), '/');
	return res_path;
}

sp_link search_by_path(const std::string& path, const sp_clink& start) {
	if(!start) return nullptr;
	return detail::deref_path<>(path, *start);
}


NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

