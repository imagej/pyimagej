import ctypes
import ctypes.util
from threading import Event, Thread
import sys
import select
import scyjava as sj
from jpype._jproxy import JProxy
from jpype._jclass import JClass


stop_loop = False

def _utf8(s):
    """ensure utf8 bytes"""
    if not isinstance(s, bytes):
        s = s.encode("utf8")
    return s


def n(name):
    """create a selector name (for ObjC methods)"""
    return objc.sel_registerName(_utf8(name))


def C(classname):
    """get an ObjC Class by name"""
    return objc.objc_getClass(_utf8(classname))


def _NSApp():
    """Return the global NSApplication instance (NSApp)"""
    objc.objc_msgSend.argtypes = [void_p, void_p]
    return msg(C("NSApplication"), n("sharedApplication"))


def _wake(NSApp):
    """Wake the Application"""
    objc.objc_msgSend.argtypes = [
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
        void_p,
    ]
    event = msg(
        C("NSEvent"),
        n(
            "otherEventWithType:location:modifierFlags:"
            "timestamp:windowNumber:context:subtype:data1:data2:"
        ),
        15,  # Type
        0,  # location
        0,  # flags
        0,  # timestamp
        0,  # window
        None,  # context
        0,  # subtype
        0,  # data1
        0,  # data2
    )
    objc.objc_msgSend.argtypes = [void_p, void_p, void_p, void_p]
    msg(NSApp, n("postEvent:atStart:"), void_p(event), True)


def stop(timer=None, loop=None):
    """Callback to fire when there's input to be read"""
    _triggered.set()
    NSApp = _NSApp()
    # if NSApp is not running, stop CFRunLoop directly,
    # otherwise stop and wake NSApp
    objc.objc_msgSend.argtypes = [void_p, void_p]
    if msg(NSApp, n("isRunning")):
        objc.objc_msgSend.argtypes = [void_p, void_p, void_p]
        msg(NSApp, n("stop:"), NSApp)
        _wake(NSApp)
    else:
        CFRunLoopStop(CFRunLoopGetCurrent())


def _stop_after(delay):
    """Register callback to stop eventloop after a delay"""
    timer = CFRunLoopTimerCreate(
        None,  # allocator
        CFAbsoluteTimeGetCurrent() + delay,  # fireDate
        0,  # interval
        0,  # flags
        0,  # order
        _c_stop_callback,
        None,
    )
    CFRunLoopAddTimer(
        CFRunLoopGetMain(),
        timer,
        kCFRunLoopCommonModes,
    )


def cocoa_mainloop(duration=1):
    """run the Cocoa eventloop for the specified duration (seconds)"""
    _triggered.clear()
    NSApp = _NSApp()
    _stop_after(duration)
    objc.objc_msgSend.argtypes = [void_p, void_p]
    msg(NSApp, n("run"))
    if not _triggered.is_set():
        # app closed without firing callback,
        # probably due to last window being closed.
        # Run the loop manually in this case,
        # since there may be events still to process (ipython/ipython#9734)
        CoreFoundation.CFRunLoopRun()

# setup Apple "objc" resources
objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))  # type:ignore[arg-type]
void_p = ctypes.c_void_p
objc.objc_getClass.restype = void_p
objc.sel_registerName.restype = void_p
objc.objc_msgSend.restype = void_p
msg = objc.objc_msgSend

# CoreFoundation C-API calls we will use:
CoreFoundation = ctypes.cdll.LoadLibrary(
    ctypes.util.find_library("CoreFoundation")  # type:ignore[arg-type]
)
CFAbsoluteTimeGetCurrent = CoreFoundation.CFAbsoluteTimeGetCurrent
CFAbsoluteTimeGetCurrent.restype = ctypes.c_double
CFRunLoopGetCurrent = CoreFoundation.CFRunLoopGetCurrent
CFRunLoopGetCurrent.restype = void_p
CFRunLoopGetMain = CoreFoundation.CFRunLoopGetMain
CFRunLoopGetMain.restype = void_p
CFRunLoopStop = CoreFoundation.CFRunLoopStop
CFRunLoopStop.restype = None
CFRunLoopStop.argtypes = [void_p]
CFRunLoopTimerCreate = CoreFoundation.CFRunLoopTimerCreate
CFRunLoopTimerCreate.restype = void_p
CFRunLoopTimerCreate.argtypes = [
    void_p,  # allocator (NULL)
    ctypes.c_double,  # fireDate
    ctypes.c_double,  # interval
    ctypes.c_int,  # flags (0)
    ctypes.c_int,  # order (0)
    void_p,  # callout
    void_p,  # context
]
CFRunLoopAddTimer = CoreFoundation.CFRunLoopAddTimer
CFRunLoopAddTimer.restype = None
CFRunLoopAddTimer.argtypes = [void_p, void_p, void_p]
kCFRunLoopCommonModes = void_p.in_dll(CoreFoundation, "kCFRunLoopCommonModes")

# setup trigger and callback
_triggered = Event()
_c_callback_func_type = ctypes.CFUNCTYPE(None, void_p, void_p)
_c_stop_callback = _c_callback_func_type(stop)
