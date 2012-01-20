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
#include <boost/preprocessor/seq/enum.hpp>
#include <boost/preprocessor/seq/for_each_i.hpp>
#include <boost/preprocessor/repetition/enum_params.hpp>
#include <boost/preprocessor/seq/size.hpp>

#define BS_TPL_ARG_(r, param, i, prefix_elem) \
BOOST_PP_COMMA_IF(i) prefix_elem BOOST_PP_CAT(param, i)

#define BS_TYPE_TPL_ARG_(z, i, param) \
BOOST_PP_CAT(param, i)

#define BS_TEXT(z, i, text) text

// expands to < A0, A1, ..., An-1 >
#define BS_MAKE_TPL_ARGS_1(tpl_args_num) \
< BOOST_PP_ENUM_PARAMS(tpl_args_num, A) >

#define BS_MAKE_TPL_ARGS_0(tpl_args_num)

// expands to T< A0, A1, ..., An-1 > if n > 0
// otherwise to T
#define BS_MAKE_FULL_TYPE(T, tpl_args_num) \
T BOOST_PP_CAT(BS_MAKE_TPL_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_num)

// expands to < tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] >
#define BS_MAKE_FULL_TYPE_ARGS_1(tpl_args_num, tpl_args) \
< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args)) >

#define BS_MAKE_FULL_TYPE_ARGS_0(tpl_args_num, tpl_args)

// expands to T< tpl_args[0], tpl_args[0], ..., tpl_args[n - 1] > if n > 0
// otherwise to T
#define BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)                \
T BOOST_PP_CAT(BS_MAKE_FULL_TYPE_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_num, tpl_args)

// expands to < A0, A1, ..., An-1 >
#define BS_ENUM_TPL_ARGS_1(tpl_args_prefix) \
BOOST_PP_SEQ_FOR_EACH_I(BS_TPL_ARG_, A, tpl_args_prefix)

#define BS_ENUM_TPL_ARGS_0(tpl_args_prefix)

#define BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BS_ENUM_TPL_ARGS_, BOOST_PP_BOOL(tpl_args_num))(tpl_args_prefix)

// third param passed as a sequence
#define BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, tpl_args_prefix)    \
namespace boost { namespace serialization {                          \
template< class Archive                                              \
BOOST_PP_COMMA_IF(tpl_args_num)                                      \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                    \
BS_API void load_construct_data(                                     \
    Archive&, BS_MAKE_FULL_TYPE(T, tpl_args_num)*,                   \
    const unsigned int);                                             \
} namespace archive { namespace detail {                             \
template< BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >          \
struct BS_API heap_allocator< BS_MAKE_FULL_TYPE(T, tpl_args_num) > { \
    typedef BS_MAKE_FULL_TYPE(T, tpl_args_num) type;                 \
    typedef blue_sky::smart_ptr< type, true > sp_type;               \
    static type* invoke();                                           \
}; }}}

#define BS_TYPE_SERIALIZE_IMPL_(T, tpl_args_num, tpl_args_prefix)    \
namespace boost { namespace serialization {                          \
template< class Archive                                              \
BOOST_PP_COMMA_IF(tpl_args_num)                                      \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >                    \
BS_API void load_construct_data(                                     \
    Archive&, BS_MAKE_FULL_TYPE(T, tpl_args_num)*,                   \
    const unsigned int)                                              \
{}                                                                   \
} namespace archive { namespace detail {                             \
template< BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >          \
typename heap_allocator< BS_MAKE_FULL_TYPE(T, tpl_args_num) >::type* \
heap_allocator< BS_MAKE_FULL_TYPE(T, tpl_args_num) >::invoke() {     \
    sp_type t = BS_KERNEL.create_object(type::bs_type(), false);     \
    return t.lock();                                                 \
} }}}

// insert declaraion of nessessary boost::serialization overrides
// needed for correct BS objects creation when they are serialized via pointers
// (pointers contained in smart_ptr)
// macro accept third argument in enum form, i.e.
// BLUE_SKY_TYPE_SERIALIZE_DECL(bs_array, 2, (class, template< class > class))
#define BLUE_SKY_TYPE_SERIALIZE_DECL_EXT(T, tpl_args_num, tpl_args_prefix) \
BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix))

// implementation of declarations made above
#define BLUE_SKY_TYPE_SERIALIZE_IMPL_EXT(T, tpl_args_num, tpl_args_prefix) \
BS_TYPE_SERIALIZE_IMPL_(T, tpl_args_num, BOOST_PP_TUPLE_TO_SEQ(tpl_args_num, tpl_args_prefix))

// simplified versions of above macroses for simple template types
// assume that prefix for every template parameter is 'class'
#define BLUE_SKY_TYPE_SERIALIZE_DECL_T(T, tpl_args_num) \
BS_TYPE_SERIALIZE_DECL_(T, tpl_args_num, (BOOST_PP_ENUM(tpl_args_num, BS_TEXT, class)))

#define BLUE_SKY_TYPE_SERIALIZE_IMPL_T(T, tpl_args_num) \
BS_TYPE_SERIALIZE_IMPL_(T, tpl_args_num, (BOOST_PP_ENUM(tpl_args_num, BS_TEXT, class)))

// most simple versions for non-template types
#define BLUE_SKY_TYPE_SERIALIZE_DECL(T) \
BS_TYPE_SERIALIZE_DECL_(T, 0, ())

#define BLUE_SKY_TYPE_SERIALIZE_IMPL(T) \
BS_TYPE_SERIALIZE_IMPL_(T, 0, ())

/*-----------------------------------------------------------------
 * GUID and EXPORT macro declaration
 *----------------------------------------------------------------*/

#define BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, tpl_args_num, tpl_args)                     \
namespace boost { namespace serialization {                                             \
template< >                                                                             \
struct guid_defined<                                                                    \
    BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)                                   \
> : boost::mpl::true_ {};                                                               \
template< >                                                                             \
const char* guid< BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args) >() {               \
    return BS_MAKE_FULL_TYPE_IMPL(T, tpl_args_num, tpl_args)::bs_type().stype_.c_str(); \
} }}

#define BLUE_SKY_TYPE_SERIALIZE_EXPORT_EXT(T, tpl_args_num, tpl_args)                       \
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
#define BLUE_SKY_TYPE_SERIALIZE_GUID_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, 1, (tpl_arg))

#define BLUE_SKY_TYPE_SERIALIZE_EXPORT_T(T, tpl_arg) \
BLUE_SKY_TYPE_SERIALIZE_EXPORT_EXT(T, 1, (tpl_arg))

// for non-template types
#define BLUE_SKY_TYPE_SERIALIZE_GUID(T) \
BLUE_SKY_TYPE_SERIALIZE_GUID_EXT(T, 0, ())

#define BLUE_SKY_TYPE_SERIALIZE_EXPORT(T) \
BLUE_SKY_TYPE_SERIALIZE_EXPORT_EXT(T, 0, ())

/*-----------------------------------------------------------------
 * Sugar for generating boost::serialization functions for any class
 *----------------------------------------------------------------*/
#define BLUE_SKY_CLASS_FCN_DECL_EXT_serialize(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {        \
template< class Archive                            \
BOOST_PP_COMMA_IF(tpl_args_num)                    \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >  \
BS_API_PLUGIN void serialize(                      \
    Archive&, BS_MAKE_FULL_TYPE(T, tpl_args_num)&, \
    const unsigned int);                           \
}}

#define BLUE_SKY_CLASS_FCN_DECL_EXT_save_construct_data(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {              \
template< class Archive                                  \
BOOST_PP_COMMA_IF(tpl_args_num)                          \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >        \
BS_API_PLUGIN void save_construct_data(                  \
    Archive&, const BS_MAKE_FULL_TYPE(T, tpl_args_num)*, \
    const unsigned int);                                 \
}}

#define BLUE_SKY_CLASS_FCN_DECL_EXT_load_construct_data(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {        \
template< class Archive                            \
BOOST_PP_COMMA_IF(tpl_args_num)                    \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >  \
BS_API_PLUGIN void load_construct_data(            \
    Archive&, BS_MAKE_FULL_TYPE(T, tpl_args_num)*, \
    const unsigned int);                           \
}}

#define BLUE_SKY_CLASS_FCN_BEGIN_EXT_save(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {                   \
template< class Archive                                       \
BOOST_PP_COMMA_IF(tpl_args_num)                               \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >             \
void save(                                                    \
    Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version                                \
){ (void)version;                                             \
/* */

#define BLUE_SKY_CLASS_FCN_BEGIN_EXT_load(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {             \
template< class Archive                                 \
BOOST_PP_COMMA_IF(tpl_args_num)                         \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >       \
void load(                                              \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version                          \
){ (void)version;                                       \
/* */

#define BLUE_SKY_CLASS_FCN_BEGIN_EXT_serialize(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {             \
template< class Archive                                 \
BOOST_PP_COMMA_IF(tpl_args_num)                         \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >       \
void serialize(                                         \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)& t, \
    const unsigned int version                          \
){ (void)version;                                       \
/* */

#define BLUE_SKY_CLASS_FCN_BEGIN_EXT_save_construct_data(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {                   \
template< class Archive                                       \
BOOST_PP_COMMA_IF(tpl_args_num)                               \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >             \
void save_construct_data(                                     \
    Archive& ar, const BS_MAKE_FULL_TYPE(T, tpl_args_num)* t, \
    const unsigned int version                                \
){ (void)version;                                             \
/* */

#define BLUE_SKY_CLASS_FCN_BEGIN_EXT_load_construct_data(T, tpl_args_num, tpl_args_prefix) \
namespace boost { namespace serialization {             \
template< class Archive                                 \
BOOST_PP_COMMA_IF(tpl_args_num)                         \
BS_ENUM_TPL_ARGS(tpl_args_num, tpl_args_prefix) >       \
void load_construct_data(                               \
    Archive& ar, BS_MAKE_FULL_TYPE(T, tpl_args_num)* t, \
    const unsigned int version                          \
){ (void)version;                                       \
/* */

#define BLUE_SKY_CLASS_SRZ_FCN_END \
} }}

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN_EXT(fcn, T, tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_BEGIN_EXT_, fcn)(T, tpl_args_num, tpl_args_prefix)

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN_T(fcn, T, tpl_args_num) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_BEGIN_EXT_, fcn)(T, tpl_args_num, \
(BOOST_PP_ENUM(tpl_args_num, BS_TEXT, class)))

#define BLUE_SKY_CLASS_SRZ_FCN_BEGIN(fcn, T) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_BEGIN_EXT_, fcn)(T, 0, ())

#define BLUE_SKY_CLASS_SRZ_FCN_DECL_EXT(fcn, T, tpl_args_num, tpl_args_prefix) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_DECL_EXT_, fcn)(T, tpl_args_num, tpl_args_prefix)

#define BLUE_SKY_CLASS_SRZ_FCN_DECL_T(fcn, T, tpl_args_num) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_DECL_EXT_, fcn)(T, tpl_args_num, \
(BOOST_PP_ENUM(tpl_args_num, BS_TEXT, class)))

#define BLUE_SKY_CLASS_SRZ_FCN_DECL(fcn, T) \
BOOST_PP_CAT(BLUE_SKY_CLASS_FCN_DECL_EXT_, fcn)(T, 0, ())

#endif /* end of include guard: BS_SERIALIZE_MACRO_OS0F6LJB */

