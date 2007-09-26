#ifndef __FOLDER_LISTING_H__25DF9B7E_0CD7_4AB6_B6CD_9FB2A6C01C0C__
#define __FOLDER_LISTING_H__25DF9B7E_0CD7_4AB6_B6CD_9FB2A6C01C0C__

#include <networking/lib.h>
#include <string>
#include <boost/smart_ptr.hpp>
#include <bs_object_base.h>
#include <networking/ISerializable.h>

namespace blue_sky
{
	namespace networking
	{
		struct listing_entry_t
		{
			std::string name;
			bool isFolder;
			time_t mtime;
		};

		BSN_API std::ostream & operator <<(std::ostream & stream, const listing_entry_t & entry);		

		//typedef blue_sky::smart_ptr<listing_entry_t, false> sp_listing_entry;
		
		class BSN_API folder_listing : 
			public blue_sky::objbase, 
			public blue_sky::networking::ISerializable
		{			
			class Impl;
			boost::scoped_ptr<Impl> pimpl;
		public:
			void fill_from(const std::vector<listing_entry_t> & data);
			void add(const listing_entry_t & entry);
			size_t size()const;
			const listing_entry_t & at(size_t)const;
			const listing_entry_t & operator[](size_t idx)const
			{
				return at(idx);
			}	
			virtual void serialize(std::ostream &) const;
			virtual void deserialize(std::istream &);
			~folder_listing();
		private:
			BLUE_SKY_TYPE_DECL(folder_listing)
		};

		BSN_API typedef blue_sky::smart_ptr<folder_listing> sp_folder_listing;
		BSN_API typedef blue_sky::lsmart_ptr<sp_folder_listing> lsp_folder_listing;

		std::ostream & operator<<(std::ostream&, const folder_listing &);
	}
}

#endif //__FOLDER_LISTING_H__25DF9B7E_0CD7_4AB6_B6CD_9FB2A6C01C0C__