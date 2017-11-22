/// @file
/// @author uentity
/// @date 20.11.2017
/// @brief Symbolic link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/link.h>
#include <bs/node.h>
#include <bs/log.h>

#include <boost/algorithm/string.hpp>
#include <boost/uuid/uuid_io.hpp>
#include <boost/uuid/string_generator.hpp>

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)
// hidden impl details
namespace {

static boost::uuids::string_generator uuid_from_str;

sp_link walk_down_tree(const std::string& cur_lid, const sp_node& level) {
	if(cur_lid != "..") {
		const auto next = level->find(uuid_from_str(cur_lid));
		if(next != level->end()) {
			return *next;
		}
	}
	return nullptr;
}

template<typename level_process_f = decltype(walk_down_tree)>
sp_link deref_symlink(
	const std::string& path, const sym_link& l, const level_process_f& proc_f = walk_down_tree
) {
	// split 
	std::vector<std::string> path_parts;
	boost::split(path_parts, path, boost::is_any_of("/"));
	// setup search root
	sp_node root = l.owner();
	if(root && !path_parts[0].size()) {
		// we have absolute path
		// walk up the tree
		// [NOTE] consider a case when path points to different tree?
		auto O = root->self_link();
		while(O) {
			root = O->data_node();
			if(!root) {
				// we should never come here if tree is in correct state
				//BSERROR << log::E("sym_link: Tree is incorrect, can't traverse to root!") << log::end;
				root = l.owner();
				break;
			}
			O = root->self_link();
		}
	}

	// follow the root and find target link
	sp_link res;
	for(const auto& part : path_parts) {
		if(!part.size() || part == ".") continue;
		if(!root) return nullptr;

		// invoke level processing function that should return link to next level
		res = proc_f(part, (const sp_node&)root);

		// calc new root from next level link
		if(part == ".." && (res = root->self_link())) {
			// go up one level
			root = res->owner();
		}
		else if(res) {
			root = res->data_node();
		}
		else root.reset();
	}
	return res;
}

/// convert from human-friendly name path to ID-based path
std::string convert2idpath(std::string namepath, const sym_link& l) {
	std::string id_path;
	const auto concat_f = [&id_path](std::string part, const sp_node& level) {
		sp_link res;
		if(part != "..") {
			const auto next = level->find(part);
			if(next != level->end()) {
				res = *next;
				part = boost::uuids::to_string(res->id());
			}
		}
		// append link ID to link's path
		if(id_path.size()) id_path += '/';
		id_path += std::move(part);
		return res;
	};

	boost::trim(namepath);
	deref_symlink(namepath, l, concat_f);
	if(namepath[0] == '/')
		id_path.insert(id_path.begin(), '/');
	return id_path;
}

/// convert from ID-based path to human readable format
std::string convert2namepath(std::string id_path, const sym_link& l) {
	std::string name_path;
	const auto concat_f = [&name_path](std::string part, const sp_node& level) {
		sp_link res;
		if(part != "..") {
			auto next = level->find(uuid_from_str(part));
			if(next != level->end()) {
				res = *next;
				part = res->name();
			}
		}
		// append link ID to link's path
		if(name_path.size()) name_path += '/';
		name_path += std::move(part);
		return res;
	};

	boost::trim(id_path);
	deref_symlink(id_path, l, concat_f);
	if(id_path[0] == '/')
		name_path.insert(name_path.begin(), '/');
	return name_path;
}

std::string abspath(sp_link l, bool name_bases = false) {
	std::string res;
	while(l) {
		res += '/';
		res += name_bases ? l->name() : boost::uuids::to_string(l->id());
		if(const auto parent = l->owner()) {
			l = parent->self_link();
		}
	}
	return res;
}

} // eof hidden namespace

/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::move(name), f), path_(std::move(path))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const sp_link& src, Flags f)
	: link(std::move(name), f), path_(abspath(src))
{}

/// implement link's API
sp_link sym_link::clone(bool deep) const {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link>(name_, path_, flags());
}

sp_obj sym_link::data() const {
	const auto src_link = deref_symlink<>(path_, *this);
	return src_link ? src_link->data() : nullptr;
}

std::string sym_link::type_id() const {
	return "sym_link";
}

std::string sym_link::oid() const {
	if(const auto D = data())
		return D->id();
	return boost::uuids::to_string(boost::uuids::uuid());
}

std::string sym_link::obj_type_id() const {
	if(const auto D = data())
		return D->type_id();
	return type_descriptor::nil().name;
}

sp_node sym_link::data_node() const {
	const auto obj = data();
	return obj && obj->is_node() ? std::static_pointer_cast<tree::node>(obj) : nullptr;
}

inode sym_link::info() const {
	const auto obj = data();
	return obj ? obj->info() : inode();
}
void sym_link::set_info(inodeptr i) {
	auto obj = data();
	if(obj) obj->set_info(std::move(i));
}

bool sym_link::is_alive() const {
	return bool(deref_symlink<>(path_, *this));
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	return human_readable ? convert2namepath(path_, *this) : path_;
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

