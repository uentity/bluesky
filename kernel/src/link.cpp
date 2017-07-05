/// @file
/// @author uentity
/// @date 14.09.2016
/// @brief Implementation os link
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/link.h>
#include <boost/uuid/uuid_generators.hpp>

NAMESPACE_BEGIN(blue_sky)
NAMESPACE_BEGIN(tree)

namespace {

// global random UUID generator for BS links
static auto gen = boost::uuids::random_generator();

} // eof hidden namespace

link::link(std::string name)
	: name_(std::move(name)),
	id_(gen())
{}

// copy ctor does not copy uuid from lhs
// instead it creates a new one
link::link(const link& lhs)
	: name_(lhs.name_), id_(gen())
{}

link::~link() {}

NAMESPACE_END(tree)
NAMESPACE_END(blue_sky)

