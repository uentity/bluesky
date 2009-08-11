#ifndef __CLIENTSETTINGS_H__C8DCA0CE_D7F1_433C_B34A_831208D69D65_
#define __CLIENTSETTINGS_H__C8DCA0CE_D7F1_433C_B34A_831208D69D65_

namespace blue_sky
{
	namespace networking
	{
		class ClientSettings
		{
		public:
			int buffer_size;
			ClientSettings()
				: buffer_size(0x400)
			{

			}
		};

		typedef boost::shared_ptr<ClientSettings> ClientSettingsPtr;
	}
}

#endif //__CLIENTSETTINGS_H__C8DCA0CE_D7F1_433C_B34A_831208D69D65_