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
class type_descriptor;
struct plugin_descriptor;

// serialization support
class atomizer;

// objbase & command
class objbase;
using sp_obj = std::shared_ptr<objbase>;
using sp_cobj = std::shared_ptr<const objbase>;

class objnode;
using sp_objnode = std::shared_ptr<objnode>;
using sp_cobjnode = std::shared_ptr<const objnode>;

// signals & slots
class bs_signal;
class bs_slot;
class bs_imessaging;
class bs_messaging;
typedef std::shared_ptr< bs_signal > sp_signal;
typedef std::shared_ptr< bs_slot > sp_slot;
typedef std::shared_ptr< const bs_imessaging > sp_mobj;

// log
namespace log {
	class bs_log;
}

class error;

// array
template< template< class > class > class any_array;

template< class > class bs_arrbase;
template< class > class bs_vecbase;
template< class, template< class > class > class bs_array;
// traits
template< class > struct vector_traits;
template< class > struct str_val_traits;

// tree-like data storage model
namespace tree {

struct inode;
class link;
class link_actor;
class link_impl;
class node;
class node_actor;
class node_impl;

}

// common typedefs
using ulong = std::size_t;
using uint = unsigned int;

} // eof blue_sky namespace

