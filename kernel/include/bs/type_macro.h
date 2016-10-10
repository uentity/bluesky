/// @file
/// @author uentity
/// @date 07.10.2016
/// @brief Macro definitions for automatic type_descriptor maintance in BlueSky objects
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include "boost/preprocessor/cat.hpp"
#include <boost/preprocessor/slot/counter.hpp>
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

/*-----------------------------------------------------------------
 * declaration
 *----------------------------------------------------------------*/
// IFACE version declare only static functions
#define BS_TYPE_DECL_IFACE                                 \
public: static const blue_sky::type_descriptor& bs_type(); \
private: friend class blue_sky::type_descriptor;

// normal decl = IFACE + bs_resolve_type()
#define BS_TYPE_DECL      \
BS_RESOLVE_TYPE_IMPL_MEM  \
BS_TYPE_DECL_IFACE

#define BS_TYPE_DECL_MEM \
BS_TYPE_IMPL_MEM         \
BS_RESOLVE_TYPE_IMPL_MEM \
private: friend class blue_sky::type_descriptor;

// for templates we include bs_resolve_type() definition in class body
// so there is no specific *_IFACE macro
#define BS_TYPE_DECL_T_MEM(T, base, type_name_prefix, descr, add_std_create, add_std_copy) \
BS_RESOLVE_TYPE_IMPL_MEM                                                                   \
public: static const blue_sky::type_descriptor& bs_type();                                 \
private: friend class type_descriptor;                                                     \
static const type_descriptor& td_maker(const std::string& tname_postfix) {                 \
    static blue_sky::type_descriptor td< T, base >(                                        \
        std::string(type_name_refix) + tname_postfix, descr, add_std_create, add_std_copy  \
    );                                                                                     \
    return td;                                                                             \
}

/*-----------------------------------------------------------------
 * bs_type() implementation
 *----------------------------------------------------------------*/
#define BS_TD_IMPL(T, base, type_name, descr, add_std_create, add_std_copy)          \
blue_sky::type_descriptor< T, base >(type_name, descr, add_std_copy, add_std_create)

#define BS_TYPE_IMPL_EXT_(prefix, T, base, type_name, descr, add_std_create, add_std_copy, is_decl)          \
BOOST_PP_SEQ_ENUM(prefix) const blue_sky::type_descriptor& BS_FMT_TYPE_SPEC(T, is_decl)                      \
bs_type() { static  blue_sky::type_descriptor< T, base > td(type_name, descr, add_std_copy, add_std_create); \
    return td; }

#define BS_TYPE_IMPL(T, base, type_name, descr, add_std_create, add_std_copy) \
BS_TYPE_IMPL_EXT_(BS_SEQ_NIL(), (T), (base), type_name, descr, add_std_create, add_std_copy, false, 0)

#define BS_TYPE_IMPL_MEM(T, base, type_name, descr, add_std_create, add_std_copy) \
BS_TYPE_IMPL_EXT_((public: static), T, base, type_name, descr, add_std_create, add_std_copy, 1)

/*-----------------------------------------------------------------
 * bs_resolve_type() implementation - calls bs_type()
 *----------------------------------------------------------------*/
#define BS_RESOLVE_TYPE_IMPL_EXT_(prefix, T, is_decl)                                   \
BOOST_PP_SEQ_ENUM(prefix) const blue_sky::type_descriptor& BS_FMT_TYPE_SPEC(T, is_decl) \
bs_resolve_type() const { return bs_type(); }

#define BS_RESOLVE_TYPE_IMPL(T) \
BS_RESOLVE_TYPE_IMPL_EXT_(BS_SEQ_NIL(), (T), 0)

#define BS_RESOLVE_TYPE_IMPL_MEM \
BS_RESOLVE_TYPE_IMPL_EXT_((public:), BS_SEQ_NIL(), 1)

/*-----------------------------------------------------------------
 * implementation for templated types I
 *----------------------------------------------------------------*/
// templated implementation with one template parameter
#define BS_TYPE_IMPL_T(T, base, type_name, descr, add_std_create, add_std_copy) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), (T), (base), type_name, descr, add_std_create, add_std_copy, 0)

#define BS_RESOLVE_TYPE_IMPL_T(T) \
BS_RESOLVE_TYPE_IMPL_EXT_((template< >), (T), 0)

// for templates with > 1 template parameters
//! put your class specification as well as base's specification in round braces!
#define BS_TYPE_IMPL_T_EXT(T_tup_size, T_tup, base_tup_size, base_tup, type_name, descr, add_std_create, add_std_copy) \
BS_TYPE_IMPL_EXT_((template< > BS_API_PLUGIN), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup),                               \
BOOST_PP_TUPLE_TO_SEQ(base_tup_size, base_tup), type_name, descr, add_std_create, add_std_copy, 0)

#define BS_RESOLVE_TYPE_IMPL_T_EXT(T_tup_size, T_tup) \
BS_RESOLVE_TYPE_IMPL_EXT_((template< >), BOOST_PP_TUPLE_TO_SEQ(T_tup_size, T_tup), 0)

/*-----------------------------------------------------------------
 * implementation for templated types II
 *----------------------------------------------------------------*/
#define _OP(r, data, x) x

#if defined(_MSC_VER)

// NOTE: it seems that VS compiler doesn't instantiate static template class members
// even when class is explicitly instantiated.
// So we need to add explicit instantiation for static member functions, declared in
// class body.
#define BS_TYPE_IMPL_T_EXT_MEM(T, spec_tup_size, spec_tup)                                                                                 \
template< > BS_API_PLUGIN const blue_sky::type_descriptor&                                                                                 \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_type()                                                          \
{ return td_maker(std::string("_") + BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)))); } \
template BS_API_PLUGIN const blue_sky::type_descriptor&                                                                                    \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::td_maker(const std::string&);

#else

#define BS_TYPE_IMPL_T_EXT_MEM(T, spec_tup_size, spec_tup)                                                                                 \
template< > BS_API_PLUGIN const blue_sky::type_descriptor&                                                                                 \
T< BOOST_PP_SEQ_ENUM(BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)) >::bs_type()                                                          \
{ return td_maker(std::string("_") + BOOST_PP_STRINGIZE(BOOST_PP_SEQ_FOR_EACH(_OP, _, BOOST_PP_TUPLE_TO_SEQ(spec_tup_size, spec_tup)))); }

#endif

#define BS_TYPE_IMPL_T_MEM(T, spec_type)  \
BS_TYPE_IMPL_T_EXT_MEM(T, 1, (spec_type))


/*-----------------------------------------------------------------
 * auto-register type constructors
 *----------------------------------------------------------------*/
// create unqiue static int variables
// registration code is executed during variables initialization
#define BLUE_SKY_TYPE_STD_CREATE_EXT(T, ctor_args_num, ctor_args_tuple)                                                    \
namespace { #include BOOST_PP_UPDATE_COUNTER()                                                                             \
static int BOOST_PP_CAT(reg_create, BOOST_PP_COUNTER) = [](){                                                              \
T::bs_type()::add_constructor< T BOOST_PP_COMMA_IF(ctor_args_num) BOOST_PP_TUPLE_ENUM(ctor_args_num, ctor_args_tuple) >(); \
    return 0; }(); }

#define BLUE_SKY_TYPE_STD_CREATE(T)    \
BLUE_SKY_TYPE_STD_CREATE_EXT(T, 0, ())

#define BLUE_SKY_TYPE_STD_COPY(T)                           \
namespace { #include BOOST_PP_UPDATE_COUNTER()              \
static int BOOST_PP_CAT(reg_copy, BOOST_PP_COUNTER) = [](){ \
T::bs_type()::add_copy_constructor< T >();                  \
    return 0; }(); }

