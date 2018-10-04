/// @file
/// @author uentity
/// @date 06.10.2016
/// @brief C++17 std::invoke implementation (from cppreference.com)
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <type_traits>
#include <functional>

namespace blue_sky {

template<typename Fn, typename... Args, 
        std::enable_if_t<std::is_member_pointer<std::decay_t<Fn>>{}, int> = 0>
constexpr decltype(auto) invoke(Fn&& f, Args&&... args)
    noexcept(noexcept(std::mem_fn(f)(std::forward<Args>(args)...)))
{
    return std::mem_fn(f)(std::forward<Args>(args)...);
}

template<typename Fn, typename... Args, 
         std::enable_if_t<!std::is_member_pointer<std::decay_t<Fn>>{}, int> = 0>
constexpr decltype(auto) invoke(Fn&& f, Args&&... args)
    noexcept(noexcept(std::forward<Fn>(f)(std::forward<Args>(args)...)))
{
    return std::forward<Fn>(f)(std::forward<Args>(args)...);
}

} // namespace blue_sky::detail

