/// @file
/// @author uentity
/// @date 11.03.2020
/// @brief Global defaults impl
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <bs/defaults.h>

#include <boost/uuid/nil_generator.hpp>
#include <boost/uuid/uuid_io.hpp>

NAMESPACE_BEGIN(blue_sky::defaults)

const char* nil_type_name = "__bs_nil_type__";

NAMESPACE_BEGIN(kernel)

const char* version = "2.devel";
const char* plugin_name = "kernel";
const char* rt_plugin_name = "runtime";
const char* py_namespace = "bs";
const char* nil_plugin_tag = "__bs_nil_plugin__";

NAMESPACE_END(kernel)

NAMESPACE_BEGIN(tree)

const boost::uuids::uuid nil_uid = boost::uuids::nil_uuid();
const std::string nil_oid = to_string(nil_uid);
const char* nil_grp_id = "__bs_nil_group__";
const char* nil_link_name = "__bs_nil_link__";

NAMESPACE_END(tree)

NAMESPACE_BEGIN(radio)

const timespan timeout = std::chrono::seconds(5);
const timespan long_timeout = std::chrono::seconds(30);

NAMESPACE_END(radio)

NAMESPACE_END(blue_sky::defaults)
