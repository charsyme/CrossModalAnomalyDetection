"""Optional Linux/X11 hardware key-state sampling for reliable hold semantics.

X11 keyboard auto-repeat can emit KeyRelease followed by KeyPress even while
one physical key is continuously held. Tk KeyRelease events alone cannot
reliably distinguish those events from a real release. XQueryKeymap exposes the
actual key-down bitmask, which is sampled by the GUI event loop.

Returns None when X11 is unavailable; the controller then uses an event-based
fallback. No third-party Python packages are required.
"""

import ctypes
import ctypes.util


class X11KeyState:
    def __init__(self):
        self._lib = None
        self._display = None
        try:
            library = ctypes.util.find_library('X11')
            if not library:
                return
            lib = ctypes.CDLL(library)
            lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
            lib.XOpenDisplay.restype = ctypes.c_void_p
            lib.XCloseDisplay.argtypes = [ctypes.c_void_p]
            lib.XCloseDisplay.restype = ctypes.c_int
            lib.XStringToKeysym.argtypes = [ctypes.c_char_p]
            lib.XStringToKeysym.restype = ctypes.c_ulong
            lib.XKeysymToKeycode.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
            lib.XKeysymToKeycode.restype = ctypes.c_ubyte
            lib.XQueryKeymap.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_char)]
            lib.XQueryKeymap.restype = ctypes.c_int
            display = lib.XOpenDisplay(None)
            if not display:
                return
            self._lib = lib
            self._display = display
        except (AttributeError, OSError):
            self._lib = None
            self._display = None

    @property
    def available(self):
        return self._display is not None

    def is_down(self, key):
        """True/False when X11 is available, None otherwise."""
        if not self.available:
            return None
        sym = self._lib.XStringToKeysym(key.encode('ascii'))
        if sym == 0:
            return None
        code = self._lib.XKeysymToKeycode(self._display, sym)
        if code == 0:
            return None
        keymap = (ctypes.c_char * 32)()
        self._lib.XQueryKeymap(self._display, keymap)
        return bool(keymap[code // 8][0] & (1 << (code % 8)))

    def close(self):
        if self._display is not None:
            self._lib.XCloseDisplay(self._display)
            self._display = None
