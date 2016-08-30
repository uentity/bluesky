/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Forward declarations of base BlueSky types
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#pragma once

#include <cstddef>
#include <memory>

//this file contains forward declarations of BlueSky kernel types

namespace blue_sky {
	//reference counter
	class bs_refcounter;

	//conversion
	template< class, class > class conversion;

	//smart pointers
	template< class > class bs_locker;
	template< class > class mt_ptr;
	template< class > class st_smart_ptr;
	template< class, bool > class smart_ptr;
	template< class > class lsmart_ptr;

	//singleton
	template< class > class singleton;

	//type_info && type_descriptor
	class nil;
	class type_descriptor;

	//objbase & command
	class objbase;
	class combase;
	//smart pointers to them
	typedef std::shared_ptr< objbase > sp_obj;
	typedef std::shared_ptr< combase > sp_com;

	//signals & slots
	class bs_signal;
	class bs_slot;
	class bs_imessaging;
	class bs_messaging;
	//common used smart pointers to them
	typedef std::shared_ptr< bs_signal > sp_signal;
	typedef std::shared_ptr< bs_slot > sp_slot;
	typedef std::shared_ptr< bs_imessaging > sp_mobj;

	//kernel
	class kernel;

	//log
	namespace log {
		class bs_log;
	}

	//exception
	class bs_exception;

	//data_table
	template< class > class bs_arrbase;
	template< class > class bs_vecbase;
	template< class > class bs_array_shared;
	template< class > class bs_vector_shared;
	template< class, template< class > class > class bs_array;
	template< class, template< class > class > class bs_map;
	// traits
	template< class > struct vector_traits;
	template< class > struct str_val_traits;
	//template< class > class str_val_table;
	//template< class > class bs_array;
	template< template< class, template< class > class > class, template< class > class > class data_table;

	//data storage
	class bs_inode;
	class bs_link;
	class bs_node;
	class bs_shell;
	class deep_iterator;
	//common used smart pointers to them
	typedef std::shared_ptr< bs_inode > sp_inode;
	typedef std::shared_ptr< bs_link > sp_link;
	typedef std::shared_ptr< bs_node > sp_node;
	typedef std::shared_ptr< bs_shell > sp_shell;

	namespace bs_private {
		struct log_wrapper;
		struct thread_log_wrapper;
	}

//common typedefs
typedef std::size_t  ulong; //!< unsigned long
typedef unsigned int uint; //!< unsigned int

} // eof blue_sky namespace

