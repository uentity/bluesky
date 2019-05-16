/// @file
/// @author uentity
/// @date 10.10.2018
/// @brief Header to be included by seralization impl code
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/
#pragma once

#include "atomizer.h"
#include "macro.h"

#include <cereal/cereal.hpp>
// polymorphic types & strings support
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/string.hpp>
// Archives
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/json.hpp>

