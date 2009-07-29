#ifndef __HTTPHEADER_H__06246DD8_FAC9_4670_B557_E21D4D188951_
#define __HTTPHEADER_H__06246DD8_FAC9_4670_B557_E21D4D188951_

#include <string>
#include <map>

namespace blue_sky
{
	namespace http_library
	{		
		using namespace std;
		struct HttpHeader {
			string name;
			string value;
		};

		class HttpHeaderCollection
		{
			typedef map<string, string> internal_collection_;
			internal_collection_ map_;
		public:			
			void add(const HttpHeader & header);
			void add(const string & name, const string & value);
			void remove(const string & name);
			bool contains(const string & name)const;
			const HttpHeader & lookup(const string & name);
			const string & get_value(const string & name)const;
			void set_value(const string & name, const string & value);
			void clear();

			size_t size() {	return map_.size();	}

			template<typename Func> void iterate(Func f)const
			{
				typedef internal_collection_::const_iterator CIT;
				for (CIT it = map_.begin(), end_it = map_.end();
					it != end_it; ++it)
				{
					f(it->first, it->second);
				}
			}
		};
	}
}

#endif //__HTTPHEADER_H__06246DD8_FAC9_4670_B557_E21D4D188951_