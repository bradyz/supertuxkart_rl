from typing import Dict, List, Tuple
import moderngl
from .renderer import Renderer, MapInfo


def nptype(t: str):
    mp = {'f1': 'u1'}
    if t in mp:
        return mp[t]
    return t


class Mapper:
    class Surface:
        fbo: moderngl.Framebuffer = None
        dbo: moderngl.Texture = None
        w: int
        h: int
        type: Tuple[int, str]

        def __init__(self, ctx: moderngl.Context, w: int, h: int, type: Tuple[int, str] = (1, 'f1'),
                     use_depth: bool = False):
            self.w, self.h, self.type = w, h, type
            if use_depth:
                self.dbo = ctx.depth_texture((w,h))
            self.fbo = ctx.framebuffer([ctx.renderbuffer((w, h), type[0], dtype=type[1])], depth_attachment=self.dbo)

        def fetch_depth(self):
            import numpy as np
            if self.dbo is not None:
                return np.frombuffer(self.dbo.read(), dtype=nptype(self.dbo.dtype))
            return None

        def fetch(self):
            import numpy as np
            b = np.frombuffer(self.fbo.read(components=self.type[0], dtype=self.type[1]), dtype=nptype(self.type[1]))
            if self.type[0] != 1:
                return b.reshape(self.h, self.w, self.type[0])
            return b.reshape(self.h, self.w)

    map_info: MapInfo
    ctx: moderngl.Context
    surfaces: Dict[str, Surface]
    renderers: List[Renderer]

    def __init__(self, map_info: MapInfo = MapInfo(), ctx: moderngl.Context = None):
        self.map_info = map_info
        if ctx is None:
            ctx = moderngl.create_standalone_context()
        self.ctx = ctx
        self.surfaces = {}
        self.renderers = []

    def add_surface(self, name: str):
        assert name not in self.surfaces, "Surface '%s' already exists!" % name
        self.surfaces[name] = Mapper.Surface(self.ctx, self.map_info.width, self.map_info.height)

    def add(self, renderer: Renderer):
        if renderer.output_name in self.surfaces:
            assert self.surfaces[renderer.output_name].type == renderer.output_type, "Output type mismatch"
        else:
            self.surfaces[renderer.output_name] = Mapper.Surface(self.ctx, self.map_info.width, self.map_info.height,
                                                                 renderer.output_type, renderer.use_depth)
        self.renderers.append(renderer)

    def draw(self, renderdoc_debug: bool = False):
        if renderdoc_debug:
            from . import renderdoc
            renderdoc.start_frame_capture()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.ctx.disable(moderngl.BLEND)
        self.ctx.depth_func = '<'
        for s in self.surfaces.values():
            s.fbo.depth_mask = True
            s.fbo.clear()

        self.ctx.viewport = (0, 0, self.map_info.width, self.map_info.height)
        for r in self.renderers:
            self.surfaces[r.output_name].fbo.use()
            r.draw(self.ctx, self.map_info)

        r = {}
        for k, s in self.surfaces.items():
            r[k] = s.fetch()
            if s.dbo is not None:
                r[k+'_d'] = s.fetch_depth()
        if renderdoc_debug:
            renderdoc.end_frame_capture()
        return r


if __name__ == "__main__":
    m = Mapper()
    m.add_surface("test")

