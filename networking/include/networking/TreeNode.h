#ifndef __TREENODE_H__F20F8BEA_403A_4602_A808_54C2CCE18958_
#define __TREENODE_H__F20F8BEA_403A_4602_A808_54C2CCE18958_

#include <boost/enable_shared_from_this.hpp>

#include <bs_object_base.h>
#include <bs_kernel.h>

#include <networking/lib.h>
#include <networking/Path.h>
#include <networking/folder_listing.h>

namespace blue_sky
{
	namespace networking
	{
		
		class BSN_API obj_ro
		{			
			friend class Tree;
			class Impl;
			boost::shared_ptr<Impl> pimpl;
			obj_ro(Tree tree, const Path & path);
		public:			
			const sp_obj get();			
			void reload();
			~obj_ro();
		};

		class BSN_API obj_rw
		{
			friend class Tree;
			class Impl;
			boost::shared_ptr<Impl> pimpl;
			obj_rw(Tree tree, const Path & path);
		public:
			sp_obj get();
			void reload();
			void close();
			~obj_rw();
		};		
				
		class BSN_API TreeNode
		{
			friend class Tree;			
			friend class LockHelper;			
						
			class TreeNodeImpl;
			boost::shared_ptr<TreeNodeImpl> pimpl;
			
			TreeNode(
				Tree tree,
				const Path & path,
				const Uri & uri);
			void core_lock();
			void unlock();
		public:
			Uri uri();

			TreeNode get(const Path & path)const;
			TreeNode operator[](const Path & path)
			{
				return get(path);
			}	
			
			void create(sp_obj obj);
			void erase();
			obj_ro open_ro();
			obj_rw open_rw();

			sp_folder_listing list();
		private:			
		};
		
						
	}
}

#endif //__TREENODE_H__F20F8BEA_403A_4602_A808_54C2CCE18958_