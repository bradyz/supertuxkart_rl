import moderngl
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import OrderedDict


class MapInfo:
    width: int = 128
    height: int = 128
    x0: float = -1
    y0: float = -1
    x1: float = 1
    y1: float = 1

    def __init__(self, width: int = 128, height: int = 128, x0: float = -1,
                 y0: float = -1, x1: float = 1, y1: float = 1):
        self.width, self.height = width, height
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    def to_map(self, x: float, y: float):
        return (x-self.x0)/(self.x1-self.x0)*(self.width-1), (y-self.y0)/(self.y1-self.y0)*(self.height-1)

    def to_world(self, x: float, y: float):
        return x/(self.width-1)*(self.x1-self.x0)+self.x0, y/(self.height-1)*(self.y1-self.y0)+self.y0,


class Renderer:
    program: moderngl.Program
    vertex_shader = """#version 410 core
    uniform vec4 map_info; // x0 y0 x1 y1
    in vec2 position;
    void main() {
        gl_Position = vec4(2 * (position.xy-map_info.xy) / (map_info.zw-map_info.xy) - 1, 0, 1);
    }
    """
    geometry_shader = None
    fragment_shader = """#version 410 core
    out float output;
    void main() {
        output = 1;
    }
    """

    inputs: Dict[str, moderngl.Attribute]
    uniforms: Dict[str, moderngl.Uniform]

    data: Dict[str, np.array]
    data_buf: Dict[str, moderngl.Buffer]

    vao: moderngl.VertexArray = None
    _vao_buffers: Set[str] = None
    uniform_data: Dict[str, Tuple]

    output_name: str
    output_type: Tuple[int, str]
    program: moderngl.Program = None
    primitive_type: int = moderngl.POINTS
    use_depth: bool = False

    def __init__(self, output_name: str, output_type: Tuple[int, str] = (1, 'f1')):
        self.output_name = output_name
        self.output_type = output_type
        self.data = {}
        self.data_buf = {}
        self.uniforms = {}
        self.uniform_data = {}
        self.inputs = {}

    def _compile(self, ctx: moderngl.Context):
        if self.program is None:
            self.program = ctx.program(vertex_shader=self.vertex_shader, fragment_shader=self.fragment_shader,
                                       geometry_shader=self.geometry_shader)
            for n in self.program:
                v = self.program[n]
                if isinstance(v, moderngl.Attribute):
                    self.inputs[n] = v
                if isinstance(v, moderngl.Uniform):
                    self.uniforms[n] = v

    def _update_data(self, ctx: moderngl.Context):
        sizes = [v.shape[0] for v in self.data.values()]
        assert np.std(sizes) == 0, "All data needs to be of the same size! Got %s" % str(sizes)

        for k in self.data:
            assert k in self.inputs, 'Data \'%s\' not a shader input. Possible values : %s' % (k, str(self.inputs))

        buffers = []
        for k, i in self.inputs.items():
            if k not in self.data:
                if k[:3] != 'gl_':
                    print('Input \'%s\' not specified! Ignoring it.' % k)
                continue

            data = self.data[k].astype('f4').tobytes()

            if k not in self.data_buf or self.data_buf[k].size < len(data):
                self.data_buf[k] = ctx.buffer(data)
            else:
                self.data_buf[k].write(data)
            if len(self.data[k].shape) <= 1 or self.data[k].shape[1] == 1:
                buffers.append((self.data_buf[k], 'f4', k))
            else:
                buffers.append((self.data_buf[k], '%df4' % self.data[k].shape[1], k))

        buffer_names = set([k for _, _, k in buffers])
        if self.vao is None or self._vao_buffers != buffer_names:
            self.vao = ctx.vertex_array(self.program, buffers)
            self._vao_buffers = buffer_names

        return sizes[0]

    def draw(self, ctx: moderngl.Context, map_info: MapInfo):
        self._compile(ctx)

        # Update the uniform
        self.uniform_data['map_info'] = (map_info.x0, map_info.y0, map_info.x1, map_info.y1)
        for k in self.uniform_data:
            if k in self.uniforms:
                if len(self.uniform_data[k]) == 1:
                    self.uniforms[k].value, = self.uniform_data[k]
                else:
                    self.uniforms[k].value = self.uniform_data[k]

        # Update the inputs
        n_elements = self._update_data(ctx)
        self.vao.render(self.primitive_type, n_elements)

    def set_data(self, **kwargs):
        self.data = kwargs


class PointRenderer(Renderer):
    primitive_type = moderngl.POINTS
    geometry_shader = """#version 410 core
    uniform vec4 map_info; // x0 x1 y0 y1
    uniform float point_size;
    layout (points) in;
    layout (triangle_strip, max_vertices=4) out;
    out vec2 tex_coord;
    void main() {
        vec2 o = point_size / (map_info.zw-map_info.xy);
        tex_coord = vec2(0, 0);
        gl_Position = gl_in[0].gl_Position + vec4(-o.x,-o.y, 0, 0);
        EmitVertex();
        tex_coord = vec2(1, 0);
        gl_Position = gl_in[0].gl_Position + vec4( o.x,-o.y, 0, 0);
        EmitVertex();
        tex_coord = vec2(0, 1);
        gl_Position = gl_in[0].gl_Position + vec4(-o.x, o.y, 0, 0);
        EmitVertex();
        tex_coord = vec2(1, 1);
        gl_Position = gl_in[0].gl_Position + vec4( o.x, o.y, 0, 0);
        EmitVertex();
        EndPrimitive();
    }
    """
    fragment_shader = """#version 410 core
    uniform float falloff_point;
    in vec2 tex_coord;
    out float result;
    void main() {
        float l = 2*length(tex_coord-vec2(0.5,0.5)); 
        if (l>=1)
            discard;
        if (falloff_point < 1)
            result = min(1, (1-l)/(1-falloff_point));
        else
            result = 1;
    }
    """

    def __init__(self, *args, point_size: float = 1, falloff_point: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        self.uniform_data['point_size'] = (point_size,)
        self.uniform_data['falloff_point'] = (falloff_point,)


class LineRenderer(Renderer):
    primitive_type = moderngl.LINES
    vertex_shader = """#version 410 core
    uniform float line_width;
    in vec2 position;
    out float lw;
    void main() {
        gl_Position = vec4(position.xy, 0, 1);
        lw = line_width;
    }
    """
    geometry_shader = """#version 410 core
    uniform vec4 map_info; // x0 x1 y0 y1
    layout (lines) in;
    layout (triangle_strip, max_vertices=4) out;
    in float lw[];
    out vec2 tex_coord;
    void setPos(vec2 p) {
        gl_Position = vec4(2 * (p-map_info.xy) / (map_info.zw-map_info.xy) - 1, 0, 1);
    }
    void main() {
        vec2 direction = 0.5*normalize(gl_in[1].gl_Position.xy - gl_in[0].gl_Position.xy);
        vec2 tangent = 0.5*normalize(vec2(direction.y, -direction.x));
        tex_coord = vec2(0, 0);
        setPos(gl_in[0].gl_Position.xy + lw[0]*(- direction + tangent));
        EmitVertex();
        tex_coord = vec2(1, 0);
        setPos(gl_in[0].gl_Position.xy + lw[0]*(- direction - tangent));
        EmitVertex();
        tex_coord = vec2(0, 1);
        setPos(gl_in[1].gl_Position.xy + lw[1]*(+ direction + tangent));
        EmitVertex();
        tex_coord = vec2(1, 1);
        setPos(gl_in[1].gl_Position.xy + lw[1]*(+ direction - tangent));
        EmitVertex();
        EndPrimitive();
    }
    """
    fragment_shader = """#version 410 core
    uniform float falloff_point;
    in vec2 tex_coord;
    out float result;
    void main() {
        if (falloff_point < 1)
            result = min(1, (1-2*abs(0.5-tex_coord.x))/(1-falloff_point));
        else
            result = 1;
    }
    """

    def __init__(self, *args, line_width: float = 1, falloff_point: float = 1, **kwargs):
        super().__init__(*args, **kwargs)
        if line_width is not None:
            self.uniform_data['line_width'] = (line_width,)
        self.uniform_data['falloff_point'] = (falloff_point,)


class VariableLineRenderer(LineRenderer):
    vertex_shader = """#version 410 core
    in vec2 position;
    in float line_width;
    out float lw;
    void main() {
        gl_Position = vec4(position.xy, 0, 1);
        lw = line_width;
    }
    """

    def __init__(self, *args, falloff_point: float = 1, **kwargs):
        super().__init__(*args, line_width=None, falloff_point=falloff_point, **kwargs)


if __name__ == "__main__":
    from .mapper import Mapper
    m = Mapper()
    # m = Mapper()
    line_r = LineRenderer("lines", line_width=0.5, falloff_point=0.5)
    line_r.set_data(position=np.array([[-0.5,-0.5], [1,0.5]], dtype=np.float32))
    m.add(line_r)
    point_r = PointRenderer("points", point_size=0.5, falloff_point=0.5)
    point_r.set_data(position=np.array([[-0.5+(i&1),-0.5+0.5*(i&2)] for i in range(4)], dtype=np.float32))
    m.add(point_r)

    from time import time
    R = m.draw()
    t0 = time()
    for it in range(1000):
        line_r.set_data(position=np.random.randn(1000, 2).astype(np.float32))
        point_r.set_data(position=np.random.randn(1000, 2).astype(np.float32))
        R = m.draw()
    print(time()-t0)
    print(R)
    from pylab import *
    subplot(2, 2, 1)
    imshow(R['lines'])
    subplot(2, 2, 2)
    imshow(R['points'])
    show()

