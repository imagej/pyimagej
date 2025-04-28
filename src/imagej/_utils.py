from Foundation import (
    NSDate,
    NSDefaultRunLoopMode,
    NSRunLoop,
)
from PyObjCTools import AppHelper
from jpype._jproxy import JProxy
from jpype._jclass import JClass


def run_non_blocking_event_loop(callback):
    """ """
    # setup Java gui thread
    m = {"run": callback}
    proxy = JProxy("java.lang.Runnable", m)
    cb_thread = JClass("java.lang.Thread")(proxy)
    cb_thread.start()
    # AppHelper.runConsoleEventLoop()
    # create special event loop that yields
    max_timeout = 0.01
    mode = NSDefaultRunLoopMode
    run_loop = NSRunLoop.currentRunLoop()
    stopper = AppHelper.PyObjcAppHelperRunLoopStopper.alloc().init()
    stopper.isConsole = True
    AppHelper.PyObjCAppHelperRunLoopStopper.addRunLoopStopper_toRunLoop_(
        stopper, run_loop
    )
    try:
        while stopper.shouldRun():
            nextfire = run_loop.limitDateForMode_(mode)
            if not stopper.shouldRun():
                break

            soon = NSDate.dateWithTimeIntervalSinceNow_(max_timeout)
            if nextfire is not None:
                nextfire = soon.earlierDate_(nextfire)
            if not run_loop.runMode_beforeDate_(mode, nextfire):
                stopper.stop()

            # Yield control back to the caller
            yield

    finally:
        AppHelper.PyObjCAppHelperRunLoopStopper.removeRunLoopStopperFromRunLoop_(
            run_loop
        )
