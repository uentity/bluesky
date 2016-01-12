/// @file
/// @author uentity
/// @date 12.01.2016
/// @brief Contains base class of all commands
/// @copyright
/// This Source Code Form is subject to the terms of the Mozilla Public License,
/// v. 2.0. If a copy of the MPL was not distributed with this file,
/// You can obtain one at https://mozilla.org/MPL/2.0/

#ifndef _BS_COMMAND_H
#define _BS_COMMAND_H

#include "setup_common_api.h"
#include "bs_fwd.h"
#include "bs_refcounter.h"

namespace blue_sky
{
	/*!
		\class combase
		\brief This is a base interface of all commands.
	 */
	class BS_API combase : virtual public bs_refcounter
	{
		friend class kernel;
		friend class objbase;

	public:
		
		//! type of blue-sky smart pointer of command
		//typedef smart_ptr< combase, true > sp_com;

		//! Virtual destructor.
		virtual ~combase() {}

		/*!
			\brief Execute command method.
			\return Next command to execute in current thread (overrides task manager load balancing)
		 */
		virtual sp_com execute() = 0;

		/*!
			\brief Undo command method.
		 */
		virtual void unexecute() = 0;

		/*!
			\brief Tests if undo is supported by this command.
		 */
		virtual bool can_unexecute() const;

		//destruction method
		//virtual void dispose() const = 0;
	};

	//! type of combase::sp_com
	//typedef combase::sp_com sp_com;

}	//namespace blue_sky

#endif // __COMMANDS_CLASS_H
