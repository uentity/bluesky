#ifndef __PATH_H__ED05E52B_6AC3_4517_AD30_1A2FF9CAA935_
#define __PATH_H__ED05E52B_6AC3_4517_AD30_1A2FF9CAA935_

#include <deque>
#include <string>

#include <boost/smart_ptr.hpp>
#include <boost/enable_shared_from_this.hpp>

#include <networking/lib.h>

namespace blue_sky
{
	namespace networking
	{	
		class BSN_API Path 
		{			
			class Impl;
			boost::shared_ptr<Impl> pimpl;
			typedef std::deque<std::string> PathQueue;
			void split(const std::string & name);
			void join();
		public:
			typedef PathQueue::const_iterator iterator;
			typedef PathQueue::const_iterator const_iterator;
			typedef PathQueue::reverse_iterator reverse_iterator;
			typedef PathQueue::const_reverse_iterator const_reverse_iterator;

			iterator begin();
			const_iterator begin()const;
			iterator end();
			const_iterator end()const;

			reverse_iterator rbegin();
			reverse_iterator rend();

			Path ();
			Path (const char * path);
			Path (const std::string & path);	
			Path (Path::const_iterator & begin, Path::const_iterator & end);
			
			const char * c_str()const;
			const std::string & str()const;

			// "/some/path/".up() => "/some/"
			Path up()const;

			// "/some/path/".down("name") => "/some/path/name"
			Path down(const char * name)const;
			Path down(const std::string & name)const;
			Path down(const Path & path)const;	

			bool subfolderof(const Path& p)const;
			bool imediatesubfolderof(const Path & p)const;

			bool equalto(const Path&p)const;

			// "/some/long/path".subtract("/some/") => "/long/path/"
			Path subtract(const Path &p)const;			

			bool  operator==(const Path& p)const
			{
				return equalto(p);
			}
		};
		
	}
}


#endif //__PATH_H__ED05E52B_6AC3_4517_AD30_1A2FF9CAA935_