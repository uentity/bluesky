// This file is part of BlueSky
// 
// BlueSky is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 3
// of the License, or (at your option) any later version.
// 
// BlueSky is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
// 
// You should have received a copy of the GNU General Public License
// along with BlueSky; if not, see <http://www.gnu.org/licenses/>.

/*!
 * \file bs_macro_private.h
 * \brief contains some blue-sky singleton macro-definitions
 * \author uentity
 */
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
