import mapper
import numpy as np
import moderngl
from typing import Tuple, List


class TrackRenderer(mapper.Renderer):
    primitive_type = moderngl.LINES
    use_depth = True
    vertex_shader = """#version 410 core
    in vec2 position;
    in float track_width;
    in float track_len;
    void main() {
        gl_Position = vec4(position.xy, track_width, track_len);
    }
    """
    geometry_shader = """#version 410 core
    uniform vec4 map_info; // x0 x1 y0 y1
    layout (lines) in;
    layout (triangle_strip, max_vertices=10) out;
    out vec2 tex_coord;
    out float track_len;
    void setPos(vec2 p, float z) {
        gl_Position = vec4(2 * (p-map_info.xy) / (map_info.zw-map_info.xy) - 1, z, 1);
    }
    void main() {
        vec2 direction = normalize(gl_in[1].gl_Position.xy - gl_in[0].gl_Position.xy);
        vec2 tangent = 0.5*normalize(vec2(direction.y, -direction.x));

        for(int i=0; i<5; i++) {
            float w = 0.5*i - 0.5;
            float wc = clamp(w, 0, 1);
            vec4 p = gl_in[0].gl_Position * (1-wc) + gl_in[1].gl_Position * wc;
            p.xy += 0.2*p.z*(w-wc)*direction;
            track_len = p.w;
            tex_coord = vec2(0, wc);
            setPos(p.xy + p.z*tangent, abs(0.5-w));
            EmitVertex();
            tex_coord = vec2(1, wc);
            setPos(p.xy - p.z*tangent, abs(0.5-w));
            EmitVertex();
        }
        EndPrimitive();
    }
    """
    fragment_shader = """#version 410 core
    uniform float track_pos = 0;
    uniform float total_track_len = 0;
    uniform float max_offset = 50;
    in vec2 tex_coord;
    in float track_len;
    out vec4 result;
    void main() {
        float p = track_len-track_pos;
        if (p < -total_track_len/2) p += total_track_len;
        if (p >  total_track_len/2) p -= total_track_len;
        if (abs(p) > max_offset) discard;
        result = vec4(1, 1.5-3*max(0.1666,abs(0.5-tex_coord.x)), p, gl_FragCoord.z);
    }
    """

    def __init__(self, *args, total_track_len: float = 0, max_offset: float = 50, **kwargs):
        super().__init__(*args, output_type=(4, 'f4'), **kwargs)
        self.uniform_data['track_pos'] = (0,)
        self.uniform_data['total_track_len'] = (total_track_len,)
        self.uniform_data['max_offset'] = (max_offset,)


class Map:
    world_offset: np.array
    drawer: mapper.Mapper = None
    track_renderer: TrackRenderer = None

    def __init__(self, track, world_size: Tuple[float, float] = (50, 50), output_size: Tuple[int, int] = (64, 64),
                 max_offset: float = 50):
        self.world_offset = np.asarray(world_size)/2
        map_info = mapper.MapInfo(*output_size, *(-self.world_offset), *self.world_offset)
        self.drawer = mapper.Mapper(map_info, ctx=moderngl.create_context())
        self.track_renderer = TrackRenderer("track", total_track_len=track.length, max_offset=max_offset)
        self.track_renderer.set_data(position=track.path_nodes[:, :, ::2].reshape(-1, 2),
                                     track_width=np.tile(track.path_width, [1, 2]).flatten(),
                                     track_len=track.path_distance.flatten())
        self.drawer.add(self.track_renderer)

    def draw_track(self, kart, renderdoc_debug: bool = False):
        x = np.asarray(kart.location[::2])
        self.drawer.map_info.x0, self.drawer.map_info.y0 = x - self.world_offset
        self.drawer.map_info.x1, self.drawer.map_info.y1 = x + self.world_offset
        self.track_renderer.uniform_data['track_pos'] = (kart.distance_down_track,)
        return self.drawer.draw(renderdoc_debug)

    def to_map(self, x: float, y: float):
        return self.drawer.map_info.to_map(x, y)

    def to_world(self, x: float, y: float):
        return self.drawer.map_info.to_world(x, y)


def colored_map(m: np.array, colors: List[int] = [0xeeeeec, 0xef2929, 0x729fcf, 0x8ae234]):
    r = np.zeros(m.shape[:2] + (3,), dtype=np.uint8)
    for i, c in enumerate(colors):
        if i < m.shape[2]:
            r[m[:, :, i] > 0] = (c >> 16, (c >> 8) & 0xff, c & 0xff)
    return r
