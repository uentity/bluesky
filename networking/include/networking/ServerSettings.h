#ifndef __SERVERSETTINGS_H__C3159ACE_4405_4315_B39F_FFEC95746B33_
#define __SERVERSETTINGS_H__C3159ACE_4405_4315_B39F_FFEC95746B33_

namespace blue_sky 
{
	namespace networking {
		class ServerSettings
		{
		public:
			int port;
			int thread_count;
			int buffer_size;
			int queue_size;

			ServerSettings()
				:port(80),
				thread_count(10),
				buffer_size(0x400),
				queue_size(20)
			{}
		};

		typedef boost::shared_ptr<ServerSettings> ServerSettingsPtr;
	}
}

#endif //__SERVERSETTINGS_H__C3159ACE_4405_4315_B39F_FFEC95746B33_