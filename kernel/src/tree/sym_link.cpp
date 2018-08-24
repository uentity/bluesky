/// @file
/// @author uentity
/// @date 20.11.2017
/// @brief Symbolic link implementation
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/log.h>
#include <bs/tree/tree.h>
#include "tree_impl.h"

NAMESPACE_BEGIN(blue_sky) NAMESPACE_BEGIN(tree)

/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::move(name), f), path_(std::move(path))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const sp_clink& src, Flags f)
	: link(std::move(name), f), path_(abspath(src))
{}

/// implement link's API
sp_link sym_link::clone(bool deep) const {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link>(name(), path_, flags());
}

std::string sym_link::type_id() const {
	return "sym_link";
}

result_or_err<sp_obj> sym_link::data_ex() const {
	// cannot dereference dangling sym link
	if(!owner()) return nullptr;
	const auto src_link = detail::deref_path(path_, *this);
	return src_link ? src_link->data_ex() : nullptr;
}

result_or_err<sp_node> sym_link::data_node_ex() const {
	const auto obj = data();
	return obj && obj->is_node() ? std::static_pointer_cast<tree::node>(obj) : nullptr;
}

bool sym_link::is_alive() const {
	return bool(detail::deref_path(path_, *this));
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	return human_readable ? convert_path(path_, bs_shared_this<sym_link>()) : path_;
}

NAMESPACE_END(tree) NAMESPACE_END(blue_sky)

