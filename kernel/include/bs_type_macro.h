/// @file
/// @author uentity
/// @date 26.09.2009
/// @brief Macro definitions for automatic type_descriptor maintance in BlueSky objects
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_TYPE_MACRO_H
#define _BS_TYPE_MACRO_H

#include "boost/preprocessor/cat.hpp"
#include "boost/preprocessor/punctuation/comma_if.hpp"
#include "boost/preprocessor/control/iif.hpp"
#include <boost/preprocessor/control/expr_iif.hpp>
#include "boost/preprocessor/facilities/empty.hpp"
#include "boost/preprocessor/tuple/to_seq.hpp"
#include <boost/preprocessor/seq/seq.hpp>
#include "boost/preprocessor/seq/enum.hpp"
#include "boost/preprocessor/seq/for_each_i.hpp"
#include "boost/preprocessor/seq/size.hpp"
#include "boost/preprocessor/seq/cat.hpp"
#include "boost/preprocessor/seq/push_back.hpp"
#include "boost/preprocessor/seq/to_array.hpp"
#include <boost/preprocessor/logical/compl.hpp>
#include <boost/preprocessor/identity.hpp>

// trick to overcome M$VC c4003 warnings
#include <boost/preprocessor/array/data.hpp>
#include <boost/preprocessor/tuple/rem.hpp> 

/*-----------------------------------------------------------------
 * helpers
 *----------------------------------------------------------------*/
#define BS_SEQ_NIL() (;)
#define BS_ARRAY_NIL() (0, ())

#define BS_FMT_TYPE_SPEC(T, is_decl) \
BOOST_PP_SEQ_ENUM(BOOST_PP_IIF(is_decl, BOOST_PP_SEQ_NIL(), T))BOOST_PP_EXPR_IIF(BOOST_PP_COMPL(is_decl), ::)

//================================ macro definitions ===================================================================
// IFACE version declare only static functions
#define BS_TYPE_DECL_IFACE                                                               \
public: static blue_sky::type_descriptor bs_type();                                      \
private: friend class blue_sky::type_descriptor;                                         \
static blue_sky::objbase* bs_create_instance(blue_sky::bs_type_ctor_param param = NULL); \
static blue_sky::objbase* bs_create_copy(blue_sky::bs_type_cpy_ctor_param param = NULL);

// normal decl = IFACE + bs_resolve_type()
#define BS_TYPE_DECL      \
BS_RESOLVE_TYPE_IMPL_MEM  \
BS_TYPE_DECL_IFACE

#define BS_TYPE_DECL_MEM(T, base, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_MEM                                                                \
BS_RESOLVE_TYPE_IMPL_MEM                                                        \
private: friend class blue_sky::type_descriptor;

// for templates we can normally include bs_resolve_type() definition in class body
// so there is no specific *_IFACE macro
#define BS_TYPE_DECL_T_MEM(T, base, stype_prefix, short_descr, long_descr, nocopy) \
BS_RESOLVE_TYPE_IMPL_MEM                                                           \
public: static blue_sky::type_descriptor bs_type();                                \
private: friend class type_descriptor;                                             \
static const type_descriptor& td_maker(const std::string& stype_postfix) {         \
    static blue_sky::type_descriptor td(                                           \
    Loki::Type2Type< T >(), Loki::Type2Type< base >(), Loki::Int2Type< nocopy >(), \
        std::string(stype_prefix) + stype_postfix, short_descr, long_descr);       \
    return td;                                                                     \
}

//----------------- common implementation ------------------------------------------------------------------------------
#define BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy)                                          \
blue_sky::type_descriptor(Loki::Type2Type< BOOST_PP_SEQ_ENUM(T) >(), Loki::Type2Type< BOOST_PP_SEQ_ENUM(base) >(), \
Loki::Int2Type< nocopy >(), type_string, short_descr, long_descr)

#define BS_TYPE_IMPL_EXT_(prefix, T, base, type_string, short_descr, long_descr, nocopy, is_decl) \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BS_FMT_TYPE_SPEC(T, is_decl)                  \
bs_type() { return BS_TD_IMPL(T, base, type_string, short_descr, long_descr, nocopy); }

#define BS_TYPE_IMPL_MEM(T, base, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_EXT_((public: static), T, base, type_string, short_descr, long_descr, nocopy, 1)

// auto-generate bs_resolve_type() body which calls bs_type()
#define BS_RESOLVE_TYPE_IMPL_EXT_(prefix, T, is_decl)                            \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::type_descriptor BS_FMT_TYPE_SPEC(T, is_decl) \
bs_resolve_type() const { return bs_type(); }

#define BS_RESOLVE_TYPE_IMPL_MEM \
BS_RESOLVE_TYPE_IMPL_EXT_((public:), BS_SEQ_NIL(), 1)

//----------------- implementation for non-templated classes -----------------------------------------------------------
#define BS_TYPE_IMPL(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_(BS_SEQ_NIL(), (T), (base), type_string, short_descr, long_descr, false, 0)

#define BS_TYPE_IMPL_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_(BS_SEQ_NIL(), (T), (base), type_string, short_descr, long_descr, true, 0)

#define BS_TYPE_IMPL_SHORT(T, short_descr) \
BS_TYPE_IMPL(T, #T, short_descr, "")

#define BS_TYPE_IMPL_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_NOCOPY(T, #T, short_descr, "")

#define BS_RESOLVE_TYPE_IMPL(T) \
BS_RESOLVE_TYPE_IMPL_EXT_(BS_SEQ_NIL(), (T), 0)

//------------------- templated implementation I - creates specializations ---------------------------------------------
#define BS_TYPE_IMPL_T(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, false, 0)

#define BS_TYPE_IMPL_T_NOCOPY(T, base, type_string, short_descr, long_descr) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_string, short_descr, long_descr, true, 0)

// for templates with > 1 template parameters
//! put your class specification as well as base's specification in round braces!
#define BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_string, short_descr, long_descr, nocopy) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup),                             \
BOOST_PP_TUPLE_TO_SEQ(base_tup_size, base_tup), type_string, short_descr, long_descr, nocopy, 0)

#define BS_TYPE_IMPL_T_SHORT(T, short_descr) \
BS_TYPE_IMPL_T(T, #T, short_descr, "")

#define BS_TYPE_IMPL_T_NOCOPY_SHORT(T, short_descr) \
BS_TYPE_IMPL_T_NOCOPY(T, #T, short_descr, "")

#define BS_RESOLVE_TYPE_IMPL_T(T) \
BS_RESOLVE_TYPE_IMPL_EXT_((template< >), (T), 0)

// for templates with > 1 template parameters
#define BS_RESOLVE_TYPE_IMPL_T_EXT(T_tup_size, T_tup) \
BS_RESOLVE_TYPE_IMPL_EXT_((template< >), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup), 0)

//------------------- templated implementation II - creates definition of bs_type --------------------------------------
#define _OP(r, data, x) x

#if defined(_MSC_VER)

// NOTE: it seems that VS compiler doesn't instantiate static template class members
// even when class is explicitly instantiated.
// So we need to add explicit instantiation for static member functions, defined in
// class body.
#define BS_TYPE_IMPL_T_EXT_MEM(T, spec_tup_size, spec_tup)                                                                                 \
template< > BS_API_PLUGIN blue_sky::type_descriptor                                                                                        \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_type()                                                          \
{ return td_maker(std::string("_") + BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)))); } \
template BS_API_PLUGIN const blue_sky::type_descriptor&                                                                                    \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::td_maker(const std::string&);                                      \
template BS_API_PLUGIN blue_sky::objbase*                                                                                                  \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_create_instance(blue_sky::bs_type_ctor_param);                  \
template BS_API_PLUGIN blue_sky::objbase*                                                                                                  \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_create_copy(blue_sky::bs_type_cpy_ctor_param);

#else

#define BS_TYPE_IMPL_T_EXT_MEM(T, spec_tup_size, spec_tup)                                                                                 \
template< > BS_API_PLUGIN blue_sky::type_descriptor                                                                                        \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_type()                                                          \
{ return td_maker(std::string("_") + BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)))); }

#endif

#define BS_TYPE_IMPL_T_MEM(T, spec_type)  \
BS_TYPE_IMPL_T_EXT_MEM(T, 1, (spec_type))

//------------------- implementation of bs_resolve_type() functions for interface realizations--------------------------

//------------------- common extended create & copy instance macroses --------------------------------------------------
#define BS_TYPE_STD_CREATE_EXT_(prefix, T, is_decl)                             \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BS_FMT_TYPE_SPEC(T, is_decl)       \
bs_create_instance(bs_type_ctor_param param BOOST_PP_EXPR_IIF(is_decl, = NULL)) \
{ return new BOOST_PP_SEQ_ENUM(T)(param); }

#define BS_TYPE_STD_COPY_EXT_(prefix, T, is_decl)                                            \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BS_FMT_TYPE_SPEC(T, is_decl)                    \
bs_create_copy(bs_type_cpy_ctor_param src) {                                                 \
    return new BOOST_PP_SEQ_ENUM(T)(*static_cast< const BOOST_PP_SEQ_ENUM(T)* >(src.get())); \
}

// IFACE macro create derived type T_obj as interface realization
#define BS_TYPE_STD_CREATE_EXT_IFACE_(prefix, T, T_obj, is_decl)                \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BS_FMT_TYPE_SPEC(T, is_decl)       \
bs_create_instance(bs_type_ctor_param param BOOST_PP_EXPR_IIF(is_decl, = NULL)) \
{ return new BOOST_PP_SEQ_ENUM(T_obj)(param); }

#define BS_TYPE_STD_COPY_EXT_IFACE_(prefix, T, T_obj, is_decl)            \
BOOST_PP_SEQ_ENUM(prefix) blue_sky::objbase* BS_FMT_TYPE_SPEC(T, is_decl) \
bs_create_copy(bs_type_cpy_ctor_param src) {                              \
    return new BOOST_PP_SEQ_ENUM(T_obj)(                                  \
        *static_cast< const BOOST_PP_SEQ_ENUM(T_obj)* >(src.get())        \
); }

//------------------- bs_create_instance macro -------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_CREATE(T) \
BS_TYPE_STD_CREATE_EXT_(BS_SEQ_NIL(), (T), 0)

#define BLUE_SKY_TYPE_STD_CREATE_IFACE(T, T_obj) \
BS_TYPE_STD_CREATE_EXT_IFACE_(BS_SEQ_NIL(), (T), (T_obj), 0)

#define BLUE_SKY_TYPE_STD_CREATE_MEM(T) \
BS_TYPE_STD_CREATE_EXT_((public: static), (T), 1)

#define BLUE_SKY_TYPE_STD_CREATE_T(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), (T), 0)

// there hardly exist any templated interface,
// but provide this macro for generality
#define BLUE_SKY_TYPE_STD_CREATE_T_IFACE(T, T_obj) \
BS_TYPE_STD_CREATE_EXT_IFACE_((template< > BS_API_PLUGIN), (T), (T_obj), 0)

//! put full specialization of T in round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_EXT(T) \
BS_TYPE_STD_CREATE_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), 0)

#define BLUE_SKY_TYPE_STD_CREATE_T_EXT_IFACE(T, T_obj)                                     \
BS_TYPE_STD_CREATE_EXT_IFACE_(                                                             \
    (template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), BOOST_PP_TUPLE_TO_SEQ(T_obj), 0 \
)

// generates bs_create_instance() as template member function
#define BLUE_SKY_TYPE_STD_CREATE_T_MEM(T) \
BS_TYPE_STD_CREATE_EXT_((public: static), (T), 1)

//--------- extended create instance generator for templated classes with any template parameters number ---------------
//some helper macro
#define BS_CLIST_CHOKER(r, data, i, elem) BOOST_PP_COMMA_IF(i) BOOST_PP_CAT(A, i)
#define BS_CLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_CLIST_CHOKER, _, tp_seq)

#define BS_TLIST_NAMER(r, data, i, elem) BOOST_PP_COMMA_IF(i) elem BOOST_PP_CAT(A, i)
#define BS_TLIST_FORMER(tp_seq) BOOST_PP_SEQ_FOR_EACH_I(BS_TLIST_NAMER, _, tp_seq)

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_CREATE_T_DEF(T, t_params) BS_TYPE_STD_CREATE_EXT_(                 \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

// for super-rare cases of templated interface implemented by templated derived object
// different sets of template params can be specified for interface T and derived T_obj
#define BLUE_SKY_TYPE_STD_CREATE_T_DEF_IFACE(T, t_params, T_obj, obj_params)                     \
BS_TYPE_STD_CREATE_EXT_IFACE_(                                                                   \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)),     \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)),            \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(obj_params), (T_obj< BS_CLIST_FORMER(obj_params) >)), 0)

//----------------- bs_create_copy macro -------------------------------------------------------------------------------
#define BLUE_SKY_TYPE_STD_COPY(T) \
BS_TYPE_STD_COPY_EXT_(BS_SEQ_NIL(), (T), 0)

#define BLUE_SKY_TYPE_STD_COPY_IFACE(T, T_obj) \
BS_TYPE_STD_COPY_EXT_IFACE_(BS_SEQ_NIL(), (T), (T_obj), 0)

#define BLUE_SKY_TYPE_STD_COPY_MEM(T) \
BS_TYPE_STD_COPY_EXT_((public: static), (T), 1)

#define BLUE_SKY_TYPE_STD_COPY_T(T) \
BS_TYPE_STD_COPY_EXT_((template< > BS_API_PLUGIN), (T), 0)

// for templated interfaces with 1 template param
#define BLUE_SKY_TYPE_STD_COPY_T_IFACE(T, T_obj) \
BS_TYPE_STD_COPY_EXT_IFACE_((template< > BS_API_PLUGIN), (T), (T_obj), 0)

//! put full specialization of T in round braces
#define BLUE_SKY_TYPE_STD_COPY_T_EXT(T) \
BS_TYPE_STD_COPY_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), 0)

#define BLUE_SKY_TYPE_STD_COPY_T_EXT_IFACE(T, T_obj)                                     \
BS_TYPE_STD_COPY_EXT_IFACE_(                                                               \
    (template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T), BOOST_PP_TUPLE_TO_SEQ(T_obj), 0 \
)

#define BLUE_SKY_TYPE_STD_COPY_T_MEM(T) \
BS_TYPE_STD_COPY_EXT_((public: static), (T), 1)

//! surround template params list with round braces
#define BLUE_SKY_TYPE_STD_COPY_T_DEF(T, t_params) BS_TYPE_STD_COPY_EXT_(                     \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

// for super-rare cases of templated interface implemented by templated derived object
// different sets of template params can be specified for interface T and derived T_obj
#define BLUE_SKY_TYPE_STD_COPY_T_DEF_IFACE(T, t_params, T_obj, obj_params)                       \
BS_TYPE_STD_COPY_EXT_IFACE_(                                                                     \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)),     \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)),            \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(obj_params), (T_obj< BS_CLIST_FORMER(obj_params) >)), 0)

// generic definition for templated classes
// surround template params list with round braces
#define BLUE_SKY_RESOLVE_TYPE_T_DEF(T, t_params) BS_RESOLVE_TYPE_IMPL_EXT_(                  \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (template< BS_TLIST_FORMER(t_params) >)), \
BOOST_PP_TUPLE_TO_SEQ(BOOST_PP_SEQ_SIZE(t_params), (T< BS_CLIST_FORMER(t_params) >)), 0)

#endif	// guard

