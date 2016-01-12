/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief contains some blue-sky singleton macro-definitions
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef BS_MACRO_PRIVATE_H_
#define BS_MACRO_PRIVATE_H_

/*! 
	Macro for creating singleton<T> of class with private constructor/
	\param T = class, which you want to make single, name
*/
#define SINGLETON_CLASS_DEF_BEGIN(T) namespace bs_private { struct wrapper_##T; } \
namespace blue_sky { class BS_API T { friend bs_private::wrapper_##T;

/*!
	This is the end of previous ( SINGLETON_CLASS_DEF_BEGIN() ) macro.
	\param T = class, which you want to make single, name
	\param ston_name = name of class, that will be singleton
 */
#define SINGLETON_CLASS_DEF_END(T, ston_name) }; \
class ston_name : public Loki::Singleton< T > {}; }

/*! 
	Implimentation of Loki::Singleton for concrete class.
	\param T = class, which you want to make single, name
 */
#define SINGLETON_CLASS_IMPL(T) namespace bs_private { struct wrapper_##T { T impl_; };\
typedef Loki::SingletonHolder<wrapper_##T> holder_##T; } \
namespace Loki { template<> T& Singleton< T >::Instance()\
{ return bs_private::holder_##T::Instance().impl_; } } \
template Loki::Singleton< blue_sky::T >;

#endif
