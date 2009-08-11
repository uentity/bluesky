#ifndef __HTTP_H__F46D4990_FE5E_452E_A3FD_84F460529D40_
#define __HTTP_H__F46D4990_FE5E_452E_A3FD_84F460529D40_

namespace blue_sky
{
	namespace http_library
	{
		namespace http
		{
			const int STATUS_200_OK = 200;
			const int STATUS_204_No_Content = 204;
			const int STATUS_400_Bad_Request = 400;
			const int STATUS_404_Not_Found = 404;
			const int STATUS_500_Internal_Server_Error = 500;
			const char CONTENT_TYPE[] = "Content-type";
			const char TRANSFER_ENCODING[] = "Transfer-Encoding";
			const char ETAG[] = "ETag";
			const char LAST_MODIFIED[] = "Last-Modified";
		}
	}
}



#endif //__HTTP_H__F46D4990_FE5E_452E_A3FD_84F460529D40_