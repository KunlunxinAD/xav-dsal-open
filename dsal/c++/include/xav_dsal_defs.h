#pragma once

#if defined XAV_NO_EXPORTS
#define XAV_EXPORTS
#else
#if defined(_WIN32) || defined(WIN32)
#define XAV_EXPORTS __declspec(dllexport)
#define XAV_IMPORTS __declspec(dllimport)
#else
#define XAV_EXPORTS __attribute__((visibility("default")))
#define XAV_IMPORTS __attribute__((visibility("default")))
#endif
#endif /* XAV_NO_EXPORTS */

#ifndef XAV_API
#define XAV_API XAV_EXPORTS
#endif

#define M_ESC(...) __VA_ARGS__
#define M_RM_PAREN(x) M_ESC x

#define F_ARGS(...) __VA_ARGS__

#define TMPL_ARGS(...) (__VA_ARGS__)

#define XAV_FUNC(ns, name, ...)    \
    namespace ns {                 \
    XAV_API int name(__VA_ARGS__); \
    }

#define XAV_FUNC_TMPL(ns, name, tp, ...) \
    namespace ns {                       \
    template <M_RM_PAREN(tp)>            \
    XAV_API int name(__VA_ARGS__);       \
    }

#define XAV_FUNCTOR(ns, name, ...)    \
    namespace ns {                    \
    struct XAV_API name {             \
        static int eval(__VA_ARGS__); \
    };                                \
    }

#define XAV_FUNCTOR_TMPL(ns, name, tp, ...) \
    namespace ns {                          \
    template <M_RM_PAREN(tp)>               \
    struct XAV_API name {                   \
        static int eval(__VA_ARGS__);       \
    };                                      \
    }

#define XAV_FUNC_CPU(name, ...) XAV_FUNC(cpu, name, __VA_ARGS__)
#define XAV_FUNC_TMPL_CPU(name, tp, ...) XAV_FUNC_TMPL(cpu, name, tp, __VA_ARGS__)

#define XAV_FUNC_XPU(name, ...) XAV_FUNC(xpu, name, __VA_ARGS__)
#define XAV_FUNC_TMPL_XPU(name, tp, ...) XAV_FUNC_TMPL(xpu, name, tp, __VA_ARGS__)

#define XAV_FUNCTOR_CPU(name, ...) XAV_FUNCTOR(cpu, name, __VA_ARGS__)
#define XAV_FUNCTOR_TMPL_CPU(name, tp, ...) XAV_FUNCTOR_TMPL(cpu, name, tp, __VA_ARGS__)

#define XAV_FUNCTOR_XPU(name, ...) XAV_FUNCTOR(xpu, name, __VA_ARGS__)
#define XAV_FUNCTOR_TMPL_XPU(name, tp, ...) XAV_FUNCTOR_TMPL(xpu, name, tp, __VA_ARGS__)

#define XAV_FUNC_XPU_AND_CPU(name, ...) \
    XAV_FUNC_XPU(name, __VA_ARGS__)     \
    XAV_FUNC_CPU(name, __VA_ARGS__)

#define XAV_FUNC_TMPL_XPU_AND_CPU(name, tp, ...) \
    XAV_FUNC_TMPL_XPU(name, tp, __VA_ARGS__)     \
    XAV_FUNC_TMPL_CPU(name, tp, __VA_ARGS__)

#define XAV_FUNCTOR_XPU_AND_CPU(name, ...) \
    XAV_FUNCTOR_XPU(name, __VA_ARGS__)     \
    XAV_FUNCTOR_CPU(name, __VA_ARGS__)

#define XAV_FUNCTOR_TMPL_XPU_AND_CPU(name, tp, ...) \
    XAV_FUNCTOR_TMPL_XPU(name, tp, __VA_ARGS__)     \
    XAV_FUNCTOR_TMPL_CPU(name, tp, __VA_ARGS__)

//
#define XAV_HOST

//
#define XAV_DEVICE

