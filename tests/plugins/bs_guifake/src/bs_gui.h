#ifndef BLUESKY_GUI_H
#define BLUESKY_GUI_H

#include "bs_object_base.h"
#include "bs_command.h"
#include "type_descriptor.h"

#include "py_bs_object_base.h"

#include <string>

namespace blue_sky {
	class BS_API_PLUGIN bs_gui : public objbase {
	public:
		// typedefs
		typedef std::map <type_descriptor, type_descriptor> map_t;

		// dtor
		~bs_gui ();

		// methods
		void add_pair (const type_descriptor&, const type_descriptor&);

	private:
		map_t gui_holder;

		// blue-sky definitions
		BLUE_SKY_TYPE_DECL(bs_gui);
	};

	namespace python {
		class BS_API_PLUGIN py_bs_gui : public py_objbase	{
		public:
			// typedefs
			typedef bs_gui wrapped_t;

			py_bs_gui();
			py_bs_gui(const py_objbase&);
			~py_bs_gui();

			void add_pair (const type_descriptor&, const type_descriptor&);
		};
	}
}

#endif // BLUESKY_GUI_H
