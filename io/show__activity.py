import contextlib, itertools, sys, threading, time


# ========================================================= #
# ===  show__activity                                    === #
# ========================================================= #
@contextlib.contextmanager
def show__activity( label="Processing", interval=0.1, statusInterval=10.0 ):
    """Show a spinner and elapsed time while a blocking operation is running."""

    stopEvent = threading.Event()
    startTime = time.perf_counter()
    stream    = sys.stdout
    isTTY     = stream.isatty()
    frames    = itertools.cycle( "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏" )

    def _show__spinner():
        waitTime = interval if ( isTTY ) else statusInterval
        while ( not stopEvent.wait( waitTime ) ):
            elapsed = time.perf_counter() - startTime
            prefix  = "\r" if ( isTTY ) else ""
            ending  = "" if ( isTTY ) else "\n"
            print( "{}{}... {}  elapsed {:6.1f} s".format(
                prefix, label, next( frames ), elapsed ),
                end=ending, file=stream, flush=True )

    print( "{}... {}  elapsed    0.0 s".format( label, next( frames ) ),
           end="" if ( isTTY ) else "\n", file=stream, flush=True )
    worker = threading.Thread( target=_show__spinner, daemon=True )
    worker.start()
    failed = False
    try:
        yield
    except BaseException:
        failed = True
        raise
    finally:
        stopEvent.set()
        worker.join()
        elapsed = time.perf_counter() - startTime
        if ( isTTY ):
            print( "\r\033[2K", end="", file=stream, flush=True )
        result = "failed" if ( failed ) else "done"
        print( "{}... {} ({:.1f} s)".format( label, result, elapsed ),
               file=stream, flush=True )
