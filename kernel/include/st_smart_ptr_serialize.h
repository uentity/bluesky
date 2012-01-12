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

namespace boost { namespace serialize {

// turn of tracking of st_smart_ptr instancies
template< class T >
struct tracking_level< ::blue_sky::st_smart_ptr< T > > {
	typedef mpl::integral_c_tag tag;
	typedef mpl::int_< ::boost::serialization::track_never> type;
};

/*-----------------------------------------------------------------
 * Serialization of st_smart_ptr to any type
 *----------------------------------------------------------------*/
/////////////////////////////////////////////////////////////
// sp_counted_base_impl serialization

template< class Archive, class P, class D >
inline void serialize(
	Archive & /* ar */,
	boost_132::detail::sp_counted_base_impl<P, D> & /* t */,
	const unsigned int /*file_version*/
){
	// register the relationship between each derived class
	// its polymorphic base
	boost::serialization::void_cast_register<
		boost_132::detail::sp_counted_base_impl<P, D>,
		boost_132::detail::sp_counted_base
	>(
		static_cast<boost_132::detail::sp_counted_base_impl<P, D> *>(NULL),
		static_cast<boost_132::detail::sp_counted_base *>(NULL)
	);
}

template< class Archive, class P, class D >
inline void save_construct_data(
	Archive & ar,
	const boost_132::detail::sp_counted_base_impl<P, D> *t,
	const BOOST_PFTO unsigned int /* file_version */
){
	// variables used for construction
	ar << boost::serialization::make_nvp("ptr", t->ptr);
}

template< class Archive, class P, class D >
inline void load_construct_data(
	Archive & ar,
	boost_132::detail::sp_counted_base_impl<P, D> * t,
	const unsigned int /* file_version */
){
	P ptr_;
	ar >> boost::serialization::make_nvp("ptr", ptr_);
	::new(t)boost_132::detail::sp_counted_base_impl<P, D>(ptr_,  D());
	// uentity - delete underlying object as usual
	// new shared_ptr sserialize system is not involved
	// placement
	// note: the original ::new... above is replaced by the one here.  This one
	// creates all new objects with a null_deleter so that after the archive
	// is finished loading and the shared_ptrs are destroyed - the underlying
	// raw pointers are NOT deleted.  This is necessary as they are used by the 
	// new system as well.
	//::new(t)boost_132::detail::sp_counted_base_impl<
	//	P,
	//	boost_132::serialization::detail::null_deleter
	//>(
	//	ptr_,  boost_132::serialization::detail::null_deleter()
	//); // placement new

	// compensate for that fact that a new shared count always is 
	// initialized with one. the add_ref_copy below will increment it
	// every time its serialized so without this adjustment
	// the use and weak counts will be off by one.
	t->use_count_ = 0;
}

/////////////////////////////////////////////////////////////
// shared_count serialization
template<class Archive>
inline void save(
	Archive & ar,
	const boost_132::detail::shared_count &t,
	const unsigned int /* file_version */
){
	ar << boost::serialization::make_nvp("pi", t.pi_);
}

template<class Archive>
inline void load(
	Archive & ar,
	boost_132::detail::shared_count &t,
	const unsigned int /* file_version */
){
	ar >> boost::serialization::make_nvp("pi", t.pi_);
	if(NULL != t.pi_)
		t.pi_->add_ref_copy();
}

/////////////////////////////////////////////////////////////
// implement serialization for smart_ptr< T >
//
template<class Archive, class T>
inline void save(
	Archive & ar,
	const blue_sky::st_smart_ptr< T > &t,
	const unsigned int /* file_version */
){
	// only the raw pointer has to be saved
	// the ref count is maintained automatically as shared pointers are loaded
	ar.register_type(static_cast<
		boost_132::detail::sp_counted_base_impl<T *, boost::checked_deleter< T > > *
	>(NULL));
	ar << boost::serialization::make_nvp("px", t.p_);
	ar << boost::serialization::make_nvp("pn", t.count_);
}

template<class Archive, class T>
inline void load(
	Archive & ar,
	blue_sky::smart_ptr< T > &t,
	const unsigned int /* file_version */
){
	// only the raw pointer has to be saved
	// the ref count is maintained automatically as shared pointers are loaded
	ar.register_type(static_cast<
		boost_132::detail::sp_counted_base_impl<T *, boost::checked_deleter< T > > *
	>(NULL));
	ar >> boost::serialization::make_nvp("px", t.p_);
	ar >> boost::serialization::make_nvp("pn", t.count_);
}

template<class Archive, class T>
inline void serialize(
	Archive & ar,
	boost::shared_ptr< T > &t,
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

}} /* boost::serialize */

BOOST_SERIALIZATION_SPLIT_FREE(boost::detail::shared_count)

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

