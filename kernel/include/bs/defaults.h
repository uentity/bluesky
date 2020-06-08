/// @file
/// @author uentity
/// @date 11.03.2020
/// @brief Global BS defaults
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include "common.h"
#include "timetypes.h"
#include "uuid.h"

NAMESPACE_BEGIN(blue_sky::defaults)

BS_API extern const char* nil_type_name;

NAMESPACE_BEGIN(kernel)

BS_API extern const char* version;
BS_API extern const char* plugin_name;
BS_API extern const char* rt_plugin_name;
BS_API extern const char* py_namespace;
BS_API extern const char* nil_plugin_tag;

NAMESPACE_END(kernel)

NAMESPACE_BEGIN(tree)

BS_API extern const uuid nil_uid;
BS_API extern const std::string nil_oid;
BS_API extern const char* nil_grp_id;
BS_API extern const char* nil_link_name;

NAMESPACE_END(tree)

NAMESPACE_BEGIN(radio)

BS_API extern const timespan timeout;
BS_API extern const timespan long_timeout;

NAMESPACE_END(radio)

NAMESPACE_END(blue_sky::defaults)
