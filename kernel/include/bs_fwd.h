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

#ifndef _BS_FWD_H
#define _BS_FWD_H

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
	class bs_type_info;
	class type_descriptor;

	//objbase & command
	class objbase;
	class combase;
	//smart pointers to them
	typedef smart_ptr< objbase, true > sp_obj;
	typedef smart_ptr< combase, true > sp_com;

	//signals & slots
	class bs_signal;
	class bs_slot;
	class bs_imessaging;
	class bs_messaging;
	//common used smart pointers to them
	typedef smart_ptr< bs_signal, true > sp_signal;
	typedef smart_ptr< bs_slot, true > sp_slot;
	typedef smart_ptr< bs_imessaging, true > sp_mobj;

	//kernel
	class kernel;

	//log
	class bs_log;
	class bs_channel;
	class bs_stream;
	//class thread_log;

	//exception
	class bs_exception;

	//data_table
	template< class> class bs_arrbase;
	template< class, template< class > class > class bs_array;
	template< class, template< class > class > class bs_map;
	// traits
	template< class > class vector_traits;
	template< class > class str_val_traits;
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
	typedef smart_ptr< bs_inode, true > sp_inode;
	typedef smart_ptr< bs_link, true > sp_link;
	typedef smart_ptr< bs_node, true > sp_node;
	typedef smart_ptr< bs_shell, true > sp_shell;

	namespace bs_private {
		struct log_wrapper;
		struct thread_log_wrapper;
	}
}

//common typedefs
typedef unsigned long ulong; //!< unsigned long
typedef unsigned int uint; //!< unsigned int

#endif
