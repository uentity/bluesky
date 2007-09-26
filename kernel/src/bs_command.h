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

/*!
 * \file bs_command.h
 * \brief Contains base class of all commands and some macroses
 * \author uentity
 */
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
