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

#ifndef BS_SERIALIZE_MACRO_OS0F6LJB
#define BS_SERIALIZE_MACRO_OS0F6LJB

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/preprocessor/facilities/empty.hpp>
#include <boost/preprocessor/repetition/enum.hpp>
#include <boost/preprocessor/tuple/to_seq.hpp>
#include <boost/preprocessor/tuple/rem.hpp>
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/seq/size.hpp>
#include <boost/preprocessor/facilities/expand.hpp>

/*-----------------------------------------------------------------
 * helper macro
 *----------------------------------------------------------------*/
#define BS_TPL_ARG_(r, param, i, prefix_elem) \
BOOST_PP_COMMA_IF(i) prefix_elem BOOST_PP_CAT(param, i)

#define BS_TYPE_TPL_ARG_(z, i, param) \
BOOST_PP_CAT(param, i)

#define BS_TEXT_(z, i, text) text

// expands to < A0, A1, ..., An-1 >
#define BS_MAKE_TPL_ARGS_1(tpl_args_num) \
< BOOST_PP_ENUM_PARAMS(tpl_args_num, A) >

#define BS_MAKE_TPL_ARGS_0(...)

// expands to T< A0, A1, ..., An-1 > if n > 0
// otherwise to T
#define BS_MAKE_FULL_TYPE(T, tpl_args_num) \
T BOOST_PP_CAT(BS_MAKE_TPL_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_num)

// expands to < tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] >
#define BS_MAKE_FULL_TYPE_ARGS_1(tpl_args_num, tpl_args) \
< BOOST_PP_TUPLE_REM_CTOR(tpl_args_num, tpl_args) >
//< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args)) >

#define BS_MAKE_FULL_TYPE_ARGS_0(...)

// expands to T< tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] > if n > 0
// otherwise to T
#define BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)                \
T BOOST_PP_CAT(BS_MAKE_FULL_TYPE_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_num, tpl_args)

// expands to A0, A1, ..., An-1
#define BS_ENUM_TPL_ARGS_1(tpl_args_prefix) \
BOOST_PP_SEQ_FOR_EACH_I(BS_TPL_ARG_, A, tpl_args_prefix)

#define BS_ENUM_TPL_ARGS_0(...)

// expands to A0, A1, ..., An-1 if n > 0
// otherwise to nothing
#define BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_ENUM_TPL_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_prefix)

#define BS_ENUM_TPL_ARGS_IMPL_1(tpl_args_prefix) \
template < BS_ENUM_TPL_ARGS_1(tpl_args_prefix) >

#define BS_ENUM_TPL_ARGS_IMPL_0(...)

// expands to template< A0, A1, ..., An-1 > if n > 0
// otherwise to nothing
#define BS_ENUM_TPL_ARGS_IMPL(tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_ENUM_TPL_ARGS_IMPL_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_prefix)

// expands to tpl_args[0], tpl_args[0], ..., tpl_args[n - 1]
#define BS_ENUM_TYPE_TPL_ARGS_1(tpl_args_num, tpl_args) \
BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args))

#define BS_ENUM_TYPE_TPL_ARGS_0(...)

// expands to tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] if n > 0
// otherwise to nothing
#define BS_ENUM_TYPE_TPL_ARGS(tpl_args_num, tpl_args) \
BOOST_PP_CAT(BS_ENUM_TYPE_TPL_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_num, tpl_args)

/*-----------------------------------------------------------------
 * Sugar for generating boost::serialization functions for any class
 *----------------------------------------------------------------*/
////////////////////////////////////////////////////////////////////
// helper macro to generate overloads of boost::serialization free functions
// third param passed as sequence!
//
#define BS_CLASS_FREE_FCN_BODY_0(fcn, T, tpl_args_num) ;

#define BS_CLASS_FREE_FCN_BODY_1(fcn, T, tpl_args_num) \
{ ::blue_sky::bs_serialize:: fcn < Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) >::go(ar, t, version); }

#define BS_CLASS_FREE_FCN_serialize(T, tpl_args_num, tpl_args_prefix, has_body) \
namespace boost { namespace serialization {             \
template< class Archive                                 \
BOOST_PP_COMMA_IF(tpl_args_num)                         \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >       \
BS_API_PLUGIN void serialize(                           \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version)                         \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_BODY_, has_body)(serialize, T, tpl_args_num) \
}}

#define BS_CLASS_FREE_FCN_save_construct_data(T, tpl_args_num, tpl_args_prefix, has_body) \
namespace boost { namespace serialization {                   \
template< class Archive                                       \
BOOST_PP_COMMA_IF(tpl_args_num)                               \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >             \
BS_API_PLUGIN void save_construct_data(                       \
    Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)* t, \
    const unsigned int version)                               \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_BODY_, has_body)(save_construct_data, T, tpl_args_num) \
}}

#define BS_CLASS_FREE_FCN_load_construct_data(T, tpl_args_num, tpl_args_prefix, has_body) \
namespace boost { namespace serialization {             \
template< class Archive                                 \
BOOST_PP_COMMA_IF(tpl_args_num)                         \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >       \
BS_API_PLUGIN void load_construct_data(                 \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)* t, \
    const unsigned int version)                         \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_BODY_, has_body)(load_construct_data, T, tpl_args_num) \
}}

#define BS_CLASS_FREE_FCN_save(T, tpl_args_num, tpl_args_prefix, has_body) \
namespace boost { namespace serialization {                    \
template< class Archive                                        \
BOOST_PP_COMMA_IF(tpl_args_num)                                \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >              \
BS_API_PLUGIN void save(                                       \
    Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)& t,  \
    const unsigned int version)                                \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_BODY_, has_body)(save, T, tpl_args_num) \
}}

#define BS_CLASS_FREE_FCN_load(T, tpl_args_num, tpl_args_prefix, has_body) \
namespace boost { namespace serialization {              \
template< class Archive                                  \
BOOST_PP_COMMA_IF(tpl_args_num)                          \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >        \
BS_API_PLUGIN void load(                                 \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t,  \
    const unsigned int version)                          \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_BODY_, has_body)(load, T, tpl_args_num) \
}}

////////////////////////////////////////////////////////////////////
// declarations of boost::serialization free functions
//
#define BLUE_SKY_CLASS_SRZ_FCN_DECL_EXT(fcn, T, tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_, fcn) \
(T, tpl_args_num, BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix), 0)

#define BLUE_SKY_CLASS_SRZ_FCN_DECL_T(fcn, T, tpl_args_num) \
BLUE_SKY_CLASS_SRZ_FCN_DECL_EXT(fcn, T, tpl_args_num, \
(BOOST_PP_ENUM(tpl_args_num, BS_TEXT_, class)))

#define BLUE_SKY_CLASS_SRZ_FCN_DECL(fcn, T) \
BLUE_SKY_CLASS_SRZ_FCN_DECL_EXT(fcn, T, 0, ())

////////////////////////////////////////////////////////////////////
// generate common overloads to omit duplicating a lot of code in
// implementation macro
//
#define BS_CLASS_OVERL_IMPL_(fcn, T, tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_CLASS_FREE_FCN_, fcn)(T, tpl_args_num, tpl_args_prefix, 1)

////////////////////////////////////////////////////////////////////
// implementation
//
#define BS_CLASS_FCN_BEGIN_save(T, tpl_args_num, tpl_args_prefix)            \
BS_CLASS_OVERL_IMPL_(save, T, tpl_args_num, tpl_args_prefix)                 \
namespace blue_sky {                                                         \
template< class Archive BOOST_PP_COMMA_IF(tpl_args_num)                      \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                            \
struct bs_serialize::save< Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) > {   \
    static void go(Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version                                               \
    ){ typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                      \
/* */

#define BS_CLASS_FCN_BEGIN_load(T, tpl_args_num, tpl_args_prefix)      \
BS_CLASS_OVERL_IMPL_(load, T, tpl_args_num, tpl_args_prefix)           \
namespace blue_sky {                                                   \
template< class Archive BOOST_PP_COMMA_IF(tpl_args_num)                \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                      \
struct bs_serialize::load< Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) > {    \
    static void go(Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version                                         \
    ){ typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                \
/* */

#define BS_CLASS_FCN_BEGIN_serialize(T, tpl_args_num, tpl_args_prefix)   \
BS_CLASS_OVERL_IMPL_(serialize, T, tpl_args_num, tpl_args_prefix)        \
namespace blue_sky {                                                     \
template< class Archive BOOST_PP_COMMA_IF(tpl_args_num)                  \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                        \
struct bs_serialize::serialize< Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) > { \
    static void go(Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t,   \
    const unsigned int version                                           \
    ){ typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                  \
/* */

#define BS_CLASS_FCN_BEGIN_save_construct_data(T, tpl_args_num, tpl_args_prefix) \
BS_CLASS_OVERL_IMPL_(save_construct_data, T, tpl_args_num, tpl_args_prefix)      \
namespace blue_sky {                                                             \
template< class Archive BOOST_PP_COMMA_IF(tpl_args_num)                          \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                                \
struct bs_serialize::save_construct_data< Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) > { \
    static void go(Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)* t,     \
    const unsigned int version                                                   \
    ){ typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                          \
/* */

#define BS_CLASS_FCN_BEGIN_load_construct_data(T, tpl_args_num, tpl_args_prefix) \
BS_CLASS_OVERL_IMPL_(load_construct_data, T, tpl_args_num, tpl_args_prefix)      \
namespace blue_sky {                                                             \
template< class Archive BOOST_PP_COMMA_IF(tpl_args_num)                          \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                                \
struct bs_serialize::load_construct_data< Archive, BS_MAKE_FULL_TYPE(T, tpl_args_num) > { \
    static void go(Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)* t,           \
    const unsigned int version                                                   \
    ){ typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                          \
/* */

////////////////////////////////////////////////////////////////////
// use this macro as function terminator
//
#define BLUE_SKY_CLASS_SRZ_FCN_END \
(void)version; (void)ar; (void)t; } }; }

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(fcn, T, tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_CLASS_FCN_BEGIN_, fcn) \
(T, tpl_args_num, BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix))

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(fcn, T, tpl_args_num) \
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(fcn, T, tpl_args_num, \
(BOOST_PP_ENUM(tpl_args_num, BS_TEXT_, class)))

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN(fcn, T) \
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(fcn, T, 0, ())

////////////////////////////////////////////////////////////////////
// generate body of serialize() function that splits to save & load
//
#define BS_CLASS_SERIALIZE_SPLIT_BODY_ \
    boost::serialization::split_free(ar, t, version); \
BLUE_SKY_CLASS_SRZ_FCN_END

#define BLUE_SKY_CLASS_SERIALIZE_SPLIT_EXT(T, tpl_args_num, tpl_args_prefix) \
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(serialize, T, tpl_args_num, tpl_args_prefix) \
BS_CLASS_SERIALIZE_SPLIT_BODY_

#define BLUE_SKY_CLASS_SERIALIZE_SPLIT_T(T, tpl_args_num) \
BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(serialize, T, tpl_args_num) \
BS_CLASS_SERIALIZE_SPLIT_BODY_

#define BLUE_SKY_CLASS_SERIALIZE_SPLIT(T) \
BLUE_SKY_CLASS_SRZ_FCN_BEGIN(serialize, T) \
BS_CLASS_SERIALIZE_SPLIT_BODY_

/*-----------------------------------------------------------------
 * Generate specific overloads for BlueSky types
 *----------------------------------------------------------------*/
// insert declaraion of nessessary boost::serialization overrides
// needed for correct BS objects creation when they are serialized via pointers
// (pointers contained in smart_ptr)
// third param passed as a sequence
#define BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, tpl_args_prefix)        \
BS_CLASS_FCN_BEGIN_load_construct_data(T, tpl_args_num, tpl_args_prefix) \
BLUE_SKY_CLASS_SRZ_FCN_END                                               \
namespace boost { namespace archive { namespace detail {                 \
template< BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >              \
struct heap_allocator< BS_MAKE_FULL_TYPE(T, tpl_args_num) > {            \
    typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                     \
    typedef blue_sky::smart_ptr< type, true > sp_type;                   \
    static BS_MAKE_FULL_TYPE(T, tpl_args_num)* invoke() {                \
        sp_type t = BS_KERNEL.create_object(type::bs_type(), false);     \
        return t.lock();                                                 \
} }; }}}

// *_EXT macro accept third argument in enum form, i.e.
// BLUE_SKY_TYPE_SERIALIZE_DECL(bs_array, 2, (class, template< class > class))
#define BLUE_SKY_TYPE_SERIALIZE_DECL_EXT(T, tpl_args_num, tpl_args_prefix) \
BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix))

// simplified versions of above macroses for simple template types
// assume that prefix for every template parameter is 'class'
#define BLUE_SKY_TYPE_SERIALIZE_DECL_T(T, tpl_args_num) \
BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, (BOOST_PP_ENUM(tpl_args_num, BS_TEXT_, class)))

// most simple versions for non-template types
// here we can automatically include *_GUID macro in DECL
// cause it anyway should be there
#define BLUE_SKY_TYPE_SERIALIZE_DECL(T) \
BS_TYPE_SERIALIZE_DECL_(T, 0, ()) \
BLUE_SKY_TYPE_SERIALIZE_GUID(T)

// DECL without GUID can be used for interfaces
#define BLUE_SKY_TYPE_SERIALIZE_DECL_NOGUID(T) \
BS_TYPE_SERIALIZE_DECL_(T, 0, ())

////////////////////////////////////////////////////////////////////
// *_DECL_BYNAME* macro provided to create BlueSky object by string
// type name instead of typeinfo provided by static bs_type() fcn
// can be useful for serializing interfaces
//
#define BS_TYPE_SERIALIZE_DECL_BYNAME_(T, tpl_args_num, tpl_args_prefix, stype) \
BS_CLASS_FCN_BEGIN_load_construct_data(T, tpl_args_num, tpl_args_prefix) \
BLUE_SKY_CLASS_SRZ_FCN_END                                               \
namespace boost { namespace archive { namespace detail {                 \
template< BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >              \
struct heap_allocator< BS_MAKE_FULL_TYPE(T, tpl_args_num) > {            \
    typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                     \
    typedef blue_sky::smart_ptr< type, true > sp_type;                   \
    static BS_MAKE_FULL_TYPE(T, tpl_args_num)* invoke() {                \
        sp_type t = BS_KERNEL.create_object(stype, false);               \
        return t.lock();                                                 \
} }; }}}

#define BLUE_SKY_TYPE_SERIALIZE_DECL_BYNAME_EXT(T, tpl_args_num, tpl_args_prefix, stype) \
BS_TYPE_SERIALIZE_DECL_BYNAME_(T, tpl_args_num, \
    BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix), stype)

#define BLUE_SKY_TYPE_SERIALIZE_DECL_BYNAME_T(T, tpl_args_num, stype) \
BS_TYPE_SERIALIZE_DECL_BYNAME_(T, tpl_args_num, \
    (BOOST_PP_ENUM(tpl_args_num, BS_TEXT_, class)), stype)

#define BLUE_SKY_TYPE_SERIALIZE_DECL_BYNAME(T, stype) \
BS_TYPE_SERIALIZE_DECL_BYNAME_(T, 0, (), stype) \
BLUE_SKY_TYPE_SERIALIZE_GUID(T)

#define BLUE_SKY_TYPE_SERIALIZE_DECL_BYNAME_NOGUID(T, stype) \
BS_TYPE_SERIALIZE_DECL_BYNAME_(T, 0, (), stype)

/*-----------------------------------------------------------------
 * serialization GUID provider for BlueSky types
 *----------------------------------------------------------------*/

#define BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, tpl_args_num, tpl_args)                     \
namespace boost { namespace serialization {                                             \
template< >                                                                             \
struct guid_defined<                                                                    \
    BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)                                   \
> : boost::mpl::true_ {};                                                               \
template< >                                                                             \
BS_API_PLUGIN const char* guid< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >();  \
}}

// simplified versions of above macroses for one template parameter
#define BLUE_SKY_TYPE_SERIALIZE_GUID_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, 1, (tpl_arg))

// for non-template types
#define BLUE_SKY_TYPE_SERIALIZE_GUID(T) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, 0, ())

/*-----------------------------------------------------------------
 * macro for generating blue_sky::serialize_register_eti< T >() body
 *----------------------------------------------------------------*/
#define BLUE_SKY_CLASS_REGISTER_ETI_EXT(T, tpl_args_num, tpl_args)                   \
namespace blue_sky { namespace detail {                                              \
template< >                                                                          \
struct bs_init_eti< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) > {            \
    static ::boost::serialization::extended_type_info const& eti;                    \
};                                                                                   \
::boost::serialization::extended_type_info const&                                    \
bs_init_eti< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >::eti =              \
    ::boost::serialization::type_info_implementation<                                \
        BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)                            \
    >::type::get_const_instance();                                                   \
}                                                                                    \
template< >                                                                          \
void serialize_register_eti< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >() { \
    detail::bs_init_eti< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >();      \
} }

// simplified versions of above macroses for one template parameter
#define BLUE_SKY_CLASS_REGISTER_ETI_T(T, tpl_arg) \
BLUE_SKY_CLASS_REGISTER_ETI_EXT(T, 1, (tpl_arg))

// for non-template types
#define BLUE_SKY_CLASS_REGISTER_ETI(T) \
BLUE_SKY_CLASS_REGISTER_ETI_EXT(T, 0, ())

/*-----------------------------------------------------------------
 * explicitly instantiate serialize() templates for given class
 *----------------------------------------------------------------*/
#define BLUE_SKY_CLASS_SERIALIZE_INST_AR(Ar_t, T, tpl_args_num, tpl_args)      \
namespace boost { namespace serialization {                                    \
template void serialize<                                                       \
    boost::archive::Ar_t BOOST_PP_COMMA_IF(tpl_args_num)                       \
    BOOST_PP_TUPLE_REM_CTOR(tpl_args_num, tpl_args)                            \
>(  boost::archive::Ar_t&, BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)&, \
    const unsigned int                                                         \
); }}

#ifdef _WIN32
// explicit instantiation of serialize() not needed in VC++
#define BLUE_SKY_CLASS_SERIALIZE_INST_EXT(T, tpl_args_num, tpl_args)
#define BLUE_SKY_CLASS_SERIALIZE_INST_T(T, tpl_arg)
#define BLUE_SKY_CLASS_SERIALIZE_INST(T)
#else
#define BLUE_SKY_CLASS_SERIALIZE_INST_EXT(T, tpl_args_num, tpl_args)              \
BLUE_SKY_CLASS_SERIALIZE_INST_AR(polymorphic_iarchive, T, tpl_args_num, tpl_args) \
BLUE_SKY_CLASS_SERIALIZE_INST_AR(polymorphic_oarchive, T, tpl_args_num, tpl_args) \
BLUE_SKY_CLASS_SERIALIZE_INST_AR(text_iarchive, T, tpl_args_num, tpl_args)        \
BLUE_SKY_CLASS_SERIALIZE_INST_AR(text_oarchive, T, tpl_args_num, tpl_args)

#define BLUE_SKY_CLASS_SERIALIZE_INST_T(T, tpl_arg) \
BLUE_SKY_CLASS_SERIALIZE_INST_EXT(T, 1, (tpl_arg))

#define BLUE_SKY_CLASS_SERIALIZE_INST(T) \
BLUE_SKY_CLASS_SERIALIZE_INST_EXT(T, 0, ())
#endif

/*-----------------------------------------------------------------
 * instantiate serialization code for BS types
 *----------------------------------------------------------------*/

#define BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(T, tpl_args_num, tpl_args)                         \
BLUE_SKY_CLASS_REGISTER_ETI_EXT(T, tpl_args_num, tpl_args)                                  \
BLUE_SKY_CLASS_SERIALIZE_INST_EXT(T, tpl_args_num, tpl_args)                                \
namespace boost { namespace serialization {                                                 \
template< > BS_API_PLUGIN                                                                   \
const char* guid< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >() {                   \
    static std::string stype =                                                              \
        BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)::bs_type().stype_.c_str();        \
    return stype.c_str();                                                                   \
} }}                                                                                        \
namespace boost { namespace archive { namespace detail { namespace extra_detail {           \
template< >                                                                                 \
struct init_guid< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) > {                     \
    static guid_initializer< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) > const & g; \
};                                                                                          \
guid_initializer< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) > const &               \
init_guid< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >::g =                         \
    ::boost::serialization::singleton<                                                      \
        guid_initializer< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >               \
    >::get_mutable_instance().export_guid();                                                \
}}}}

// simplified versions of above macroses for one template parameter
#define BLUE_SKY_TYPE_SERIALIZE_IMPL_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(T, 1, (tpl_arg))

// for non-template types
#define BLUE_SKY_TYPE_SERIALIZE_IMPL(T) \
BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(T, 0, ())

/*-----------------------------------------------------------------
 * EXPORT = GUID + IMPL
 *----------------------------------------------------------------*/
#define BLUE_SKY_TYPE_SERIALIZE_EXPORT_EXT(T, tpl_args_num, tpl_args) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, tpl_args_num, tpl_args) \
BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(T, tpl_args_num, tpl_args)

#define BLUE_SKY_TYPE_SERIALIZE_EXPORT_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_GUID_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_IMPL_T(T, tpl_arg)

#define BLUE_SKY_TYPE_SERIALIZE_EXPORT(T) \
BLUE_SKY_TYPE_SERIALIZE_GUID(T) \
BLUE_SKY_TYPE_SERIALIZE_IMPL(T)

/*-----------------------------------------------------------------
 * override of boost::detail::base_register to call
 * bs_void_cast_register() instead of void_cast_register()
 * skiping compile-time check if Base is virtual base of Derived
 * and resolving compilation error that pops up when using
 * boost::serialization::base_object
 *----------------------------------------------------------------*/
// generator, prefixes passed as sequences
#define BS_CLASS_HAS_NONVIRTUAL_BASE_(T, tpl_args_num, tpl_args_prefix, B, btpl_args_num, btpl_args_prefix) \
namespace boost { namespace detail {                                                 \
template < BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix)                           \
BOOST_PP_COMMA_IF(btpl_args_num) BS_ENUM_TPL_ARGS(btpl_args_num, btpl_args_prefix) > \
struct base_register< BS_MAKE_FULL_TYPE(T, tpl_args_num), \
BS_MAKE_FULL_TYPE(B, btpl_args_num) > {                   \
    typedef BS_MAKE_FULL_TYPE(B, btpl_args_num) Base;     \
    typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) Derived;   \
    struct polymorphic {                                  \
        static void const * invoke(){                     \
            Base const * const b = 0;                     \
            Derived const * const d = 0;                  \
            return & bs_void_cast_register(d, b);         \
        }                                                 \
    };                                                    \
    struct non_polymorphic {                              \
        static void const * invoke(){                     \
            return 0;                                     \
        }                                                 \
    };                                                    \
    static void const * invoke(){                         \
        typedef BOOST_DEDUCED_TYPENAME mpl::eval_if<      \
            is_polymorphic<Base>,                         \
            mpl::identity<polymorphic>,                   \
            mpl::identity<non_polymorphic>                \
        >::type type;                                     \
        return type::invoke();                            \
    }                                                     \
}; }}


#endif /* end of include guard: BS_SERIALIZE_MACRO_OS0F6LJB */

