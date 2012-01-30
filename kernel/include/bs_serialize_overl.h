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

#ifndef BS_SERIALIZE_OVERL_NDALCHMR
#define BS_SERIALIZE_OVERL_NDALCHMR

#include <boost/serialization/void_cast.hpp>
#include <boost/serialization/base_object.hpp>

// Std void_cast_register< Derived, Base >() implementation from boost::serialization
// generates compilation error when deducing whether Base is virtual
// base of Derived or not. This happens probably because objbase is inherited virtually
// from bs_refcounter ans somewhere in deep deep Boost magic a structure appears
// that inherit virtually from objbase and from Derived and compiler complains that there is
// no unique definition of bs_refcounter::dispose (though it should be objbase::dispose).
// To solve that I include here own void_cast_register implementation that omits
// compile-time checking and assumes that Base IS SIMPLE BASE of Derived!
namespace boost { namespace serialization {

template< class Derived, class Base >
BOOST_DLLEXPORT
inline const void_cast_detail::void_caster & bs_void_cast_register(
	Derived const * /* dnull = NULL */, 
	Base const * /* bnull = NULL */
){
	typedef void_cast_detail::void_caster_primitive<Derived, Base> typex;
	return singleton< typex >::get_const_instance();
}

namespace detail {
	// only register void casts if the types are polymorphic
	template<class Base, class Derived>
	struct bs_base_register {
		struct polymorphic {
			static void const * invoke(){
				Base const * const b = 0;
				Derived const * const d = 0;
				return & bs_void_cast_register(d, b);
			}
		};
		struct non_polymorphic {
			static void const * invoke(){
				return 0;
			}
		};
		static void const * invoke(){
			typedef BOOST_DEDUCED_TYPENAME mpl::eval_if<
				is_polymorphic<Base>,
				mpl::identity<polymorphic>,
				mpl::identity<non_polymorphic>
					>::type type;
			return type::invoke();
		}
	};
}

template<class Base, class Derived>
BOOST_DEDUCED_TYPENAME detail::base_cast<Base, Derived>::type & 
bs_base_object(Derived &d)
{
	BOOST_STATIC_ASSERT(( is_base_and_derived<Base,Derived>::value));
	BOOST_STATIC_ASSERT(! is_pointer<Derived>::value);
	typedef BOOST_DEDUCED_TYPENAME detail::base_cast<Base, Derived>::type type;
	detail::bs_base_register<type, Derived>::invoke();
	return access::cast_reference<type, Derived>(d);
}

}}

#endif /* end of include guard: BS_SERIALIZE_OVERL_NDALCHMR */

