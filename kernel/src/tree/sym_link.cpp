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
#include <bs/tree/errors.h>
#include "tree_impl.h"

NAMESPACE_BEGIN(blue_sky::tree)

/// ctor -- pointee is specified by string path
sym_link::sym_link(std::string name, std::string path, Flags f)
	: link(std::move(name), f), path_(std::move(path))
{}
/// ctor -- pointee is specified directly - absolute path will be stored
sym_link::sym_link(std::string name, const sp_link& src, Flags f)
	: sym_link(std::move(name), abspath(src), f)
{}

/// implement link's API
sp_link sym_link::clone(bool deep) const {
	// no deep copy support for symbolic link
	return std::make_shared<sym_link>(name(), path_, flags());
}

std::string sym_link::type_id() const {
	return "sym_link";
}

void sym_link::reset_owner(const sp_node& new_owner) {
	link::reset_owner(new_owner);
	// update link's status
	check_alive();
}

result_or_err<sp_obj> sym_link::data_impl() const {
	// cannot dereference dangling sym link
	const auto parent = owner();
	if(!parent) return tl::make_unexpected(error::quiet(Error::UnboundSymLink));
	const auto src_link = deref_path(path_, parent);
	return src_link ?
		result_or_err<sp_obj>(src_link->data_ex()) :
		tl::make_unexpected(error::quiet(Error::LinkExpired));
}

bool sym_link::check_alive() {
	auto res = bool(deref_path(path_, owner()));
	rs_reset(Req::Data, res ? ReqStatus::OK : ReqStatus::Error);
	return res;
}

/// return stored pointee path
std::string sym_link::src_path(bool human_readable) const {
	if(!human_readable) return path_;
	else if(const auto parent = owner())
		return convert_path(path_, parent->handle());
	return {};
}

result_or_err<sp_node> sym_link::propagate_handle() {
	// sym link cannot be a node's handle
	return data_node_ex();
}

NAMESPACE_END(blue_sky::tree)

