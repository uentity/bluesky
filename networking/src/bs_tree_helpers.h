#ifndef __BS_TREE_HELPERS_H__5B34AD50_D40A_44F8_95E7_40DF3ECE718D__
#define __BS_TREE_HELPERS_H__5B34AD50_D40A_44F8_95E7_40DF3ECE718D__

#include <bs_fwd.h>
#include <networking/Path.h>
#include <bs_tree.h>
#include <bs_link.h>

namespace blue_sky
{
	namespace networking
	{
		sp_node get_node(const Path & path);
		sp_link get_link(const Path & path);
		bool exists(const Path & path);
		void delete_link(const Path & path);
		sp_link create_inode(const Path & path, sp_obj obj);
	}
}
#endif //__BS_TREE_HELPERS_H__5B34AD50_D40A_44F8_95E7_40DF3ECE718D__