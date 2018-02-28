/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Forward declarations of base BlueSky types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <memory>

namespace blue_sky {

// singleton
template< class > class singleton;

// type_info && type_descriptor
class nil;
class type_descriptor;
struct plugin_descriptor;

// objbase & command
class objbase;
typedef std::shared_ptr< objbase > sp_obj;
typedef std::shared_ptr< const objbase > sp_cobj;

// signals & slots
class bs_signal;
class bs_slot;
class bs_imessaging;
class bs_messaging;
typedef std::shared_ptr< bs_signal > sp_signal;
typedef std::shared_ptr< bs_slot > sp_slot;
typedef std::shared_ptr< const bs_imessaging > sp_mobj;

// kernel
class kernel;

// log
namespace log {
	class bs_log;
}

// exception
class error;

template< template< class > class > class any_array;

template< class > class bs_arrbase;
template< class > class bs_vecbase;
template< class, template< class > class > class bs_array;
// traits
template< class > struct vector_traits;
template< class > struct str_val_traits;

namespace tree {

// data storage
class link;
class node;
typedef std::shared_ptr< link > sp_link;
typedef std::shared_ptr< node > sp_node;

}

// common typedefs
typedef std::size_t  ulong; //!< unsigned long
typedef unsigned int uint; //!< unsigned int

} // eof blue_sky namespace

