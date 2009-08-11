#ifndef __TREE_H__B93F1BC8_7274_4041_905C_D9B364F04EAF_
#define __TREE_H__B93F1BC8_7274_4041_905C_D9B364F04EAF_

#include <boost/smart_ptr.hpp>
#include <networking/lib.h>
#include <networking/Context.h>
#include <networking/Uri.h>
#include <networking/Path.h>
#include <networking/TreeNode.h>
#include <networking/folder_listing.h>

namespace blue_sky
{
	namespace networking
	{
		class Path;	
		
		struct MountPoint
		{
			Path path;
			Uri uri;

			MountPoint()
			{}

			MountPoint(Path p, Uri u)
				: path(p), uri(u)
			{}
		};

		typedef std::list<MountPoint> MountPointList;
		typedef boost::shared_ptr<MountPointList> MountPointListPtr;

		
		
		class BSN_API Tree
		{
			friend class TreeNode;
			friend class obj_ro;
			friend class obj_rw;
			class TreeImpl;
			boost::shared_ptr<TreeImpl> pimpl;	

			blue_sky::sp_obj get_object(const Path& path);
			blue_sky::sp_obj lock_and_get(const Path& path);			
			void put_and_unlock(const Path& path, blue_sky::sp_obj);			
		public:			
			Tree(ContextPtr context);
			TreeNode get(const Path& path);
			TreeNode operator[](const Path&path)
			{
				return get(path);
			}			

			void mount(const Uri& uri, const Path& path);
			void umount(const Path& path);
			MountPointListPtr get_mount_points();
			
			void create(sp_obj obj, const Path& path);
			void erase(const Path& path);
			obj_ro open_ro(const Path& path);
			obj_rw open_rw(const Path& path);

			sp_folder_listing list(const Path& path);

			void close_connections();
			void wait_for_changes_in(const Path & path);			
		};	
	}
}

#endif //__TREE_H__B93F1BC8_7274_4041_905C_D9B364F04EAF_