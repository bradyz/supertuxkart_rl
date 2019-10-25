from ctypes import *


class RENDERDOC_API_1_1_0(Structure):
    _fields_ = [
        ('GetAPIVersion', CFUNCTYPE(None, POINTER(c_int), POINTER(c_int), POINTER(c_int))),
        ('SetCaptureOptionU32', CFUNCTYPE(c_int, c_int, c_uint32)),
        ('SetCaptureOptionF32', CFUNCTYPE(c_int, c_int, c_float)),
        ('GetCaptureOptionU32', CFUNCTYPE(c_uint32, c_int)),
        ('GetCaptureOptionF32', CFUNCTYPE(c_float, c_int)),
        ('SetFocusToggleKeys', CFUNCTYPE(None, POINTER(c_int), c_int)),
        ('SetCaptureKeys', CFUNCTYPE(None, POINTER(c_int), c_int)),
        ('GetOverlayBits', CFUNCTYPE(c_uint32)),
        ('MaskOverlayBits', CFUNCTYPE(None, c_uint32, c_uint32)),

        ('Shutdown', CFUNCTYPE(None)),
        ('UnloadCrashHandler', CFUNCTYPE(None)),
        ('SetLogFilePathTemplate', CFUNCTYPE(None, c_char_p)),
        ('GetLogFilePathTemplate', CFUNCTYPE(c_char_p)),
        ('GetNumCaptures', CFUNCTYPE(c_uint32)),
        ('GetCapture', CFUNCTYPE(c_uint32, c_uint32, c_char_p, POINTER(c_uint32), POINTER(c_uint64))),

        ('TriggerCapture', CFUNCTYPE(None)),
        ('IsRemoteAccessConnected', CFUNCTYPE(c_uint32)),
        ('LaunchReplayUI', CFUNCTYPE(c_uint32, c_uint32, c_char_p)),
        ('SetActiveWindow', CFUNCTYPE(None, c_void_p, c_void_p)),
        ('StartFrameCapture', CFUNCTYPE(None, c_void_p, c_void_p)),
        ('IsFrameCapturing', CFUNCTYPE(c_uint32)),
        ('EndFrameCapture', CFUNCTYPE(c_uint32, c_void_p, c_void_p)),
        ('TriggerMultiFrameCapture', CFUNCTYPE(None, c_uint32)),
    ]


try:
    _librenderdoc = cdll.LoadLibrary('librenderdoc.so')
    p_api = POINTER(RENDERDOC_API_1_1_0)()
    _librenderdoc.RENDERDOC_GetAPI(10000, byref(p_api))
    api = p_api.contents

    def start_frame_capture():
        api.StartFrameCapture(c_void_p(0), c_void_p(0))

    def is_frame_capturing():
        return api.IsFrameCapture()

    def end_frame_capture():
        return api.EndFrameCapture(c_void_p(0), c_void_p(0))

except OSError:
    def start_frame_capture():
        pass

    def is_frame_capturing():
        pass

    def end_frame_capture():
        pass

