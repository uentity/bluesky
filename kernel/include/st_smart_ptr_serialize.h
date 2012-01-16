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

#ifndef ST_SMART_PTR_SERIALIZE_RINMRD5M
#define ST_SMART_PTR_SERIALIZE_RINMRD5M

#include "smart_ptr.h"

#include <boost/mpl/integral_c.hpp>
#include <boost/mpl/integral_c_tag.hpp>

// code to serialize boost_132::detail::shared_count
//#include <boost/serialization/shared_ptr_132.hpp>
#include <boost/serialization/split_free.hpp>
#include <boost/serialization/nvp.hpp>
#include <boost/serialization/tracking.hpp>
#include <boost/serialization/void_cast.hpp>
#include <boost/serialization/tracking.hpp>

namespace boost { namespace serialization {

// turn of tracking of st_smart_ptr instancies
template< class T >
struct tracking_level< ::blue_sky::st_smart_ptr< T > > {
	typedef mpl::integral_c_tag tag;
	typedef mpl::int_< ::boost::serialization::track_never> type;
	BOOST_STATIC_CONSTANT(int, value = type::value);
};

/*-----------------------------------------------------------------
 * Serialization of st_smart_ptr to any type
 *----------------------------------------------------------------*/
/////////////////////////////////////////////////////////////
// bs_refcounter_sp serialization
//
template< class Archive, class T >
inline void serialize(
	Archive & /* ar */,
	blue_sky::bs_private::bs_refcounter_sp< T > & /* t */,
	const unsigned int /*file_version*/
){
	// register the relationship between each derived class
	// and its polymorphic base
	boost::serialization::void_cast_register<
		blue_sky::bs_private::bs_refcounter_sp< T >,
		blue_sky::bs_refcounter
	>(
		static_cast< blue_sky::bs_private::bs_refcounter_sp< T >* >(NULL),
		static_cast< blue_sky::bs_refcounter* >(NULL)
	);
}

/////////////////////////////////////////////////////////////
// bs_refcounter_spd serialization
//
template< class Archive, class T, class D >
inline void serialize(
	Archive & /* ar */,
	blue_sky::bs_private::bs_refcounter_spd< T, D > & /* t */,
	const unsigned int /*file_version*/
){
	// register the relationship between each derived class
	// and its polymorphic base
	boost::serialization::void_cast_register<
		blue_sky::bs_private::bs_refcounter_spd< T, D >,
		blue_sky::bs_refcounter
	>(
		static_cast< blue_sky::bs_private::bs_refcounter_spd< T, D >* >(NULL),
		static_cast< blue_sky::bs_refcounter* >(NULL)
	);
}

/////////////////////////////////////////////////////////////
// bs_refcounter_sp counstruction
//
template< class Archive, class T >
inline void save_construct_data(
	Archive & ar,
	const blue_sky::bs_private::bs_refcounter_sp< T > *t,
	const unsigned int /* file_version */
){
	// variables used for construction
	ar << boost::serialization::make_nvp("ptr", t->p_);
}

template< class Archive, class T >
inline void load_construct_data(
	Archive & ar,
	blue_sky::bs_private::bs_refcounter_sp< T >* t,
	const unsigned int /* file_version */
){
	typename blue_sky::bs_private::bs_refcounter_sp< T >::pointer_t ptr;
	ar >> boost::serialization::make_nvp("ptr", ptr);
	// invoke ctor
	::new(t) blue_sky::bs_private::bs_refcounter_sp< T >*(ptr, 0);

	// uentity: already compensated in ctor
	// compensate for that fact that a new shared count always is 
	// initialized with one. the add_ref_copy below will increment it
	// every time its serialized so without this adjustment
	// the use and weak counts will be off by one.
	//t->use_count_ = 0;
}

/////////////////////////////////////////////////////////////
// bs_refcounter_spd counstruction
//
template< class Archive, class T, class D >
inline void save_construct_data(
	Archive & ar,
	const blue_sky::bs_private::bs_refcounter_spd< T, D > *t,
	const unsigned int /* file_version */
){
	// variables used for construction
	ar << boost::serialization::make_nvp("ptr", t->p_);
}

template< class Archive, class T, class D >
inline void load_construct_data(
	Archive & ar,
	blue_sky::bs_private::bs_refcounter_spd< T, D >* t,
	const unsigned int /* file_version */
){
	typename blue_sky::bs_private::bs_refcounter_spd< T, D >::pointer_t ptr;
	ar >> boost::serialization::make_nvp("ptr", ptr);
	// invoke ctor
	::new(t) blue_sky::bs_private::bs_refcounter_spd< T, D >*(ptr, 0);
}

/////////////////////////////////////////////////////////////
// bs_refcounter_ptr serialization
template< class Archive, class T >
inline void save(
	Archive & ar,
	const blue_sky::bs_private::bs_refcounter_ptr< T > &t,
	const unsigned int /* file_version */
){
	ar << boost::serialization::make_nvp("rc", t.rc_);
}

template< class Archive, class T >
inline void load(
	Archive & ar,
	blue_sky::bs_private::bs_refcounter_ptr< T > &t,
	const unsigned int /* file_version */
){
	ar >> boost::serialization::make_nvp("rc", t.rc_);
	if(NULL != t.pi_)
		t.rc_->add_ref();
}

template< class Archive, class T >
inline void serialize(
	Archive & ar,
	blue_sky::bs_private::bs_refcounter_ptr< T > &t,
	const unsigned int file_version
){
	boost::serialization::split_free(ar, t, file_version);
}

/////////////////////////////////////////////////////////////
// implement serialization for st_smart_ptr< T >
//
template< class Archive, class T >
inline void save(
	Archive & ar,
	const blue_sky::st_smart_ptr< T > &t,
	const unsigned int /* file_version */
){
	// only the raw pointer has to be saved
	// the ref count is maintained automatically as shared pointers are loaded
	ar.register_type(static_cast<
		blue_sky::bs_private::bs_refcounter_sp< T >*
	>(NULL));
	ar << boost::serialization::make_nvp("px", t.p_);
	ar << boost::serialization::make_nvp("pn", t.count_);
}

template< class Archive, class T >
inline void load(
	Archive & ar,
	blue_sky::smart_ptr< T > &t,
	const unsigned int /* file_version */
){
	// only the raw pointer has to be saved
	// the ref count is maintained automatically as shared pointers are loaded
	ar.register_type(static_cast<
		blue_sky::bs_private::bs_refcounter_sp< T >*
	>(NULL));
	ar >> boost::serialization::make_nvp("px", t.p_);
	ar >> boost::serialization::make_nvp("pn", t.count_);
}

template< class Archive, class T >
inline void serialize(
	Archive & ar,
	blue_sky::st_smart_ptr< T > &t,
	const unsigned int file_version
){
	// correct shared_ptr serialization depends upon object tracking
	// being used.
	BOOST_STATIC_ASSERT(
		boost::serialization::tracking_level< T >::value
		!= boost::serialization::track_never
	);
	boost::serialization::split_free(ar, t, file_version);
}

}} /* boost::serialization */

// note: change below uses null_deleter 
// This macro is used to export GUIDS for shared pointers to allow
// the serialization system to export them properly. David Tonge
#define BS_ST_SMARTPTR_EXPORT_GUID(T, K)                               \
    typedef boost::detail::sp_counted_base_impl<                       \
        T *,                                                           \
        boost::checked_deleter< T >                                    \
    > __st_smart_ptr_ ## T;                                            \
    BOOST_CLASS_EXPORT_GUID(__st_smart_ptr_ ## T, "__st_smart_ptr_" K) \
    BOOST_CLASS_EXPORT_GUID(T, K)                                      \
    /**/

#define BS_ST_SMARTPTR_EXPORT(T) \
    BS_ST_SMART_PTR_EXPORT_GUID( \
        T,                       \
        BOOST_PP_STRINGIZE(T)    \
    )                            \
    /**/

#endif /* end of include guard: ST_SMART_PTR_SERIALIZE_RINMRD5M */

