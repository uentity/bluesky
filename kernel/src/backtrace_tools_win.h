/**
 * \file backtrace_tools_win.h
 * \brief
 * \author Sergey Miryanov
 * \date 16.06.2009
 * */
#ifndef BS_BOS_CORE_BASE_BACKTRACE_TOOLS_WIN_H_
#define BS_BOS_CORE_BASE_BACKTRACE_TOOLS_WIN_H_

#include <dbghelp.h>
#include <Psapi.h>
#include <boost/format.hpp>

#pragma comment (lib, "dbghelp.lib")
#pragma comment (lib, "psapi.lib")

namespace blue_sky {
namespace tools {

  static int
  get_backtrace (void **backtrace_, int size_)
  {
    STACKFRAME64 stack_frame = {0};
    CONTEXT context = {0};
    HANDLE thread = GetCurrentThread ();
    HANDLE process = GetCurrentProcess ();
    if (!thread)
      {
        throw bs_exception ("get_backtrace", "Can't get thread handler");
      }
    if (!process)
      {
        throw bs_exception ("get_backtrace", "Can't get process handler");
      }

    static bool sym_init = true;
    if (sym_init)
      {
        if (!SymInitialize (process, "d:\\blue-sky\\exe\\release\\plugins", TRUE))
          {
            throw bs_exception ("get_backtrace", "Can't initialize symbol handler");
          }
        sym_init = false;
      }

    context.ContextFlags = CONTEXT_CONTROL;
    __asm 
      {
      label_:
        mov [context.Ebp], ebp;  
        mov [context.Esp], esp;  
        mov eax, [label_];
        mov [context.Eip], eax;  
      };

    stack_frame.AddrPC.Offset     = context.Eip;
    stack_frame.AddrPC.Mode       = AddrModeFlat;
    stack_frame.AddrFrame.Offset  = context.Ebp;
    stack_frame.AddrFrame.Mode    = AddrModeFlat;
    stack_frame.AddrStack.Offset  = context.Esp;
    stack_frame.AddrStack.Mode    = AddrModeFlat;

    BOOL res = FALSE;
    int i = 0;
    for (; i < size_; ++i)
      {
        res = StackWalk64 (
          IMAGE_FILE_MACHINE_I386,
          process,
          thread,
          &stack_frame,
          NULL,
          NULL,
          SymFunctionTableAccess64,
          SymGetModuleBase64,
          NULL);

        if (!res || stack_frame.AddrPC.Offset == 0)
          break;

        backtrace_[i] = (void *)stack_frame.AddrPC.Offset;
      }

    if (!res && !i)
      {
        throw bs_exception ("get_backtrace", "Can't obtain call-stack info");
      }

    return i;
  }

  static char **
  get_backtrace_names (void *const *backtrace_, int size_)
  {
    char symbol_ [sizeof (IMAGEHLP_SYMBOL64) + sizeof (TCHAR) * (MAX_PATH + 1)] = {0};
    IMAGEHLP_SYMBOL64 *symbol = (IMAGEHLP_SYMBOL64 *)symbol_;

    symbol->SizeOfStruct = sizeof (IMAGEHLP_SYMBOL64);
    symbol->MaxNameLength = MAX_PATH;

    HANDLE process = GetCurrentProcess ();
    char **names = (char **)malloc ((MAX_PATH + 1 + sizeof (char *)) * size_);
    memset (names, 0, (MAX_PATH + 1 + sizeof (char *)) * size_);

    for (int i = 0; i < size_; ++i)
      {
        names[i] = (char *)names + sizeof (char *) * size_ + (MAX_PATH + 1) * i;

        BOOL res = SymGetSymFromAddr64 (process, (DWORD64)backtrace_[i], 0, symbol);
        if (!res)
          {
            LPVOID lpMsgBuf;
            LPVOID lpDisplayBuf;
            DWORD dw = GetLastError(); 

            FormatMessage(
              FORMAT_MESSAGE_ALLOCATE_BUFFER | 
              FORMAT_MESSAGE_FROM_SYSTEM |
              FORMAT_MESSAGE_IGNORE_INSERTS,
              NULL,
              dw,
              MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT),
              (LPTSTR) &lpMsgBuf,
              0, NULL );

            BSERROR << (char*)lpMsgBuf << bs_end;
            LocalFree (lpMsgBuf);

            break;
          }

        memcpy (names[i], symbol->Name, (std::min <size_t>) (MAX_PATH, strlen (symbol->Name)));
      }

    return names;
  }

} // namespace tools
} // namespace blue_sky

#endif // #ifdef BS_BOS_CORE_BASE_BACKTRACE_TOOLS_WIN_H_

