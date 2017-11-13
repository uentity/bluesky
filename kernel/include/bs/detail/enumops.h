/// @file
/// @author uentity
/// @date 13.11.2017
/// @brief Operator overloading for enums that makes it possible to convert to integral types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#include <type_traits>

namespace blue_sky {

template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T operator~(T a) {
	return T(~std::underlying_type_t<T>(a));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T operator|(T a, T b) {
	return T(std::underlying_type_t<T>(a) | std::underlying_type_t<T>(b));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T operator&(T a, T b) {
	return T(std::underlying_type_t<T>(a) & std::underlying_type_t<T>(b));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T operator^(T a, T b) {
	return T(std::underlying_type_t<T>(a) ^ std::underlying_type_t<T>(b));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T& operator|=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a |= std::underlying_type_t<T>(b));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T& operator&=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a &= std::underlying_type_t<T>(b));
}
template<class T, typename = std::enable_if_t<std::is_enum<T>::value>> inline T& operator^=(T& a, T b) {
	return (T&)((std::underlying_type_t<T>&)a ^= std::underlying_type_t<T>(b));
}
	
} /* namespace blue_sky */

