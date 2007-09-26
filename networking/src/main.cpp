#include "pch.h"
#include <networking/folder_listing.h>



namespace blue_sky {
	BLUE_SKY_PLUGIN_DESCRIPTOR("blue_sky::networking", "0.0.1", "Networking support for blue_sky.", "")

	BLUE_SKY_REGISTER_PLUGIN_FUN
	{
		bool res = BLUE_SKY_REGISTER_TYPE(*bs_init.pd_, blue_sky::networking::folder_listing);
		
			//return res;
			return true;
	}
}