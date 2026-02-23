(function() {
  const files = window.MANIM_DATA.files;

  files["_2024/holograms/diffraction.py"] = {
    description: "Diffraction and interference pattern scenes for the hologram video. Simulates wave propagation, single/double slit experiments, and how interference creates holographic images.",
    code: `from __future__ import annotations
from manim_imports_ext import *

def hsl_to_rgb(hsl):
    """
    Convert an array of HSL values to RGB.

    Args:
    hsl (np.ndarray): A numpy array of shape (n, 3), where each row represents an HSL value
                      (Hue [0, 1), Saturation [0, 1], Lightness [0, 1]).

    Returns:
    np.ndarray: An array of shape (n, 3), containing RGB values in the range [0, 1].
    """
    h = hsl[:, 0]
    s = hsl[:, 1]
    l = hsl[:, 2]

    def hue_to_rgb(p, q, t):
        t = np.where(t < 0, t + 1, np.where(t > 1, t - 1, t))
        return np.where(t < 1/6, p + (q - p) * 6 * t,
               np.where(t < 1/2, q,
               np.where(t < 2/3, p + (q - p) * (2/3 - t) * 6, p
        )))

    q = np.where(l < 0.5, l * (1 + s), l + s - l * s)
    p = 2 * l - q

    r = hue_to_rgb(p, q, h + 1 / 3)
    g = hue_to_rgb(p, q, h)
    b = hue_to_rgb(p, q, h - 1 / 3)

    rgb = np.stack([r, g, b], axis=1)
    return rgb


class LightWaveSlice(Mobject):
    shader_folder: str = str(Path(Path(__file__).parent, "diffraction_shader"))
    data_dtype: Sequence[Tuple[str, type, Tuple[int]]] = [
        ('point', np.float32, (3,)),
    ]
    render_primitive: int = moderngl.TRIANGLE_STRIP

    def __init__(
        self,
        point_sources: DotCloud,
        shape: tuple[float, float] = (8.0, 8.0),
        color: ManimColor = BLUE_D,
        opacity: float = 1.0,
        frequency: float = 1.0,
        wave_number: float = 1.0,
        max_amp: Optional[float] = None,
        decay_factor: float = 0.5,
        show_intensity: bool = False,
        **kwargs
    ):
        self.shape = shape
        self.point_sources = point_sources
        self._is_paused = False
        super().__init__(**kwargs)

        if max_amp is None:
            max_amp = point_sources.get_num_points()
        self.set_uniforms(dict(
            frequency=frequency,
            wave_number=wave_number,
            max_amp=max_amp,
            time=0,
            decay_factor=decay_factor,
            show_intensity=float(show_intensity),
            time_rate=1.0,
        ))
        self.set_color(color, opacity)

        self.add_updater(lambda m, dt: m.increment_time(dt))
        self.always.sync_points()
        self.apply_depth_test()

    def init_data(self) -> None:
        super().init_data(length=4)
        self.data["point"][:] = [UL, DL, UR, DR]

    def init_points(self) -> None:
        self.set_shape(*self.shape)

    def set_color(
        self,
        color: ManimColor | Iterable[ManimColor] | None,
        opacity: float | Iterable[float] | None = None,
        recurse=False,
    ) -> Self:
        if color is not None:
            self.set_uniform(color=color_to_rgb(color))
        if opacity is not None:
            self.set_uniform(opacity=opacity)
        return self

    def set_opacity(self, opacity: float, recurse=False):
        self.set_uniform(opacity=opacity)
        return self

    def set_wave_number(self, wave_number: float):
        self.set_uniform(wave_number=wave_number)
        return self

    def set_frequency(self, frequency: float):
        self.set_uniform(frequency=frequency)
        return self

    def set_max_amp(self, max_amp: float):
        self.set_uniform(max_amp=max_amp)
        return self

    def set_decay_factor(self, decay_factor: float):
        self.set_uniform(decay_factor=decay_factor)
        return self

    def set_time_rate(self, time_rate: float):
        self.set_uniform(time_rate=time_rate)
        return self

    def set_sources(self, point_sources: DotCloud):
        self.point_sources = point_sources
        return self

    def sync_points(self):
        sources: DotCloud = self.point_sources
        for n, point in enumerate(sources.get_points()):
            self.set_uniform(**{f"point_source{n}": point})
        self.set_uniform(n_sources=sources.get_num_points())
        return self

    def increment_time(self, dt):
        self.uniforms["time"] += self.uniforms["time_rate"] * dt
        return self

    def show_intensity(self, show: bool = True):
        self.set_uniform(show_intensity=float(show))

    def pause(self):
        self.set_uniform(time_rate=0)
        return self

    def unpause(self):
        self.set_uniform(time_rate=1)
        return self

    def interpolate(
        self,
        wave1: LightWaveSlice,
        wave2: LightWaveSlice,
        alpha: float,
        path_func: Callable[[np.ndarray, np.ndarray, float], np.ndarray] = straight_path
    ) -> Self:
        self.locked_uniform_keys.add("time")
        super().interpolate(wave1, wave2, alpha, path_func)

    def wave_func(self, points):
        time = self.uniforms["time"]
        wave_number = self.uniforms["wave_number"]
        frequency = self.uniforms["frequency"]
        decay_factor = self.uniforms["decay_factor"]

        values = np.zeros(len(points))
        for source_point in self.point_sources.get_points():
            dists = np.linalg.norm(points - source_point, axis=1)
            values += np.cos(TAU * (wave_number * dists - frequency * time)) * (dists + 1)**(-decay_factor)
        return values


class LightIntensity(LightWaveSlice):
    def __init__(
        self,
        *args,
        color: ManimColor = BLUE,
        show_intensity: bool = True,
        **kwargs
    ):
        super().__init__(*args, color=color, show_intensity=show_intensity, **kwargs)


# Scenes


class LightFieldAroundScene(InteractiveScene):
    def construct(self):
        # Add scene
        frame = self.frame
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/holograms/Paul Animations/LightFieldDraft"
        scene, scene_top, lamp = group = Group(
            ImageMobject(os.path.join(folder, "LightFieldScene")),
            ImageMobject(os.path.join(folder, "TopHalfCutoff")),
            ImageMobject(os.path.join(folder, "Lamp")),
        )
        group.set_height(7)
        group.to_edge(RIGHT)

        light_point = scene.get_corner(UL) + 1.6 * RIGHT + 0.8 * DOWN
        scene_point = scene.get_center() + 0.8 * RIGHT + 0.4 * DOWN

        frame.reorient(0, 0, 0, (3.44, -0.2, 0.0), 4.78)
        self.add(scene)
        self.play(FadeIn(scene, 0.25 * UP))
        self.wait()

        # Add light
        line = Line(light_point, scene_point)
        n_dots = 50
        light = Group(
            GlowDot(line.pfp(a), color=WHITE, radius=interpolate(0.5, 4, a), opacity=3 / n_dots)
            for a in np.linspace(0, 1, n_dots)
        )
        self.play(
            FadeIn(lamp, time_span=(1, 3)),
            FadeIn(light, lag_ratio=0.2, time_span=(1, 3)),
            frame.animate.reorient(0, 0, 0, (3.64, 0.68, 0.0), 5.30),
            run_time=2,
        )
        self.wait()

        # Set up wave
        sources = DotCloud([
            (2.42, -0.27, 0.0),
            (1.96, -0.14, 0.0),
            (2.11, 0.05, 0.0),
            (2.51, 0.16, 0.0),
            (2.75, 0.07, 0.0),
            (3.19, 0.41, 0.0),
            (3.48, 0.39, 0.0),
            (4.81, 0.28, 0.0),
            (5.1, 0.22, 0.0),
            (5.53, -0.04, 0.0),
            (5.8, -0.18, 0.0),
            (4.87, -0.58, 0.0),
            (4.71, -0.68, 0.0),
            (4.28, -0.61, 0.0),
            (3.92, -0.51, 0.0),
            (4.18, -0.27, 0.0),
            (3.8, 0.09, 0.0),
            (3.55, 0.23, 0.0),
            (2.43, 0.7, 0.0),
            (2.43, 0.7, 0.0),
            (3.71, 0.51, 0.0),
            (3.94, -0.49, 0.0),
            (1.9, -0.3, 0.0),
            (2.61, 0.06, 0.0),
            (2.92, 0.23, 0.0),
            (3.25, 0.38, 0.0),
        ])
        wave = LightWaveSlice(sources)
        wave.set_max_amp(math.sqrt(30))
        wave.set_shape(50, 100)
        wave.rotate(70 * DEGREES, LEFT)
        wave.set_wave_number(4)
        wave.set_frequency(1)

        n_perp_waves = 10
        perp_waves = Group(wave.copy().rotate(PI / 2, RIGHT).shift(z * OUT) for z in np.linspace(-1, 1, n_perp_waves))
        perp_waves.set_opacity(0.1)

        self.add(scene, wave, perp_waves, scene_top, lamp, light)
        self.play(
            FadeIn(wave),
            FadeIn(perp_waves),
            light.animate.set_opacity(2 / n_dots),
            scene.animate.set_opacity(0.5),
            run_time=2
        )

        # Slow zoom out
        self.play(
            frame.animate.to_default_state(),
            run_time=12
        )

        # Linger
        self.wait(12)

        # Reflection
        if False:
            self.remove(light)
            self.remove(lamp)
            self.frame.set_theta(-10 * DEGREES).set_y(0.5)
            scene_top.set_opacity(0.5)
            self.wait(12)

        # Show an observer
        eye_dot = GlowDot(1.0 * LEFT, radius=0.5, color=WHITE)

        self.play(FadeIn(eye_dot))
        self.wait(3)
        self.play(
            eye_dot.animate.move_to(1.5 * LEFT + 2 * DOWN).set_anim_args(path_arc=45 * DEGREES),
            run_time=12,
            rate_func=there_and_back,
        )
        self.play(FadeOut(eye_dot))

        # Show recreation
        right_rect = FullScreenRectangle().set_fill(BLACK, 1)
        right_rect.stretch(0.35, 0, about_edge=RIGHT)
        film_rect = Rectangle()
        film_rect.set_fill(interpolate_color(RED_E, BLACK, 0.8), 1).set_stroke(WHITE, 1)
        film_rect.rotate(80 * DEGREES, UP)
        film_rect.rotate(20 * DEGREES, RIGHT)
        film_rect.move_to(right_rect.get_left())

        laser_point = RIGHT_SIDE + RIGHT
        laser_light = VGroup(
            Line(laser_point, film_rect.pfp(a + random.random() / 250))
            for a in np.linspace(0, 1, 250)
        )
        laser_light.set_stroke(GREEN_SCREEN, (0, 1), 0.2)
        laser_light.shuffle()

        self.play(
            FadeOut(lamp),
            FadeOut(light),
            FadeOut(scene),
            FadeOut(scene_top),
            FadeIn(right_rect),
            FadeIn(film_rect),
        )
        self.play(ShowCreation(laser_light, lag_ratio=0.001))
        self.wait(12)
        self.wait(2)  # Just in case


class DiffractionGratingScene(InteractiveScene):
    light_position = 10 * DOWN + 5 * OUT + 3 * LEFT

    def setup(self):
        super().setup()
        self.camera.light_source.move_to(self.light_position)

    def get_wall_with_slits(self, n_slits, spacing=1.0, slit_width=0.1, height=0.25, depth=3.0, total_width=40, color=GREY_D, shading=(0.5, 0.5, 0.5)):
        width = spacing - slit_width
        cube = Cube().set_shape(width, height, depth)
        parts = cube.replicate(n_slits + 1)
        parts.arrange(RIGHT, buff=slit_width)
        edge_piece_width = 0.5 * (total_width - parts.get_width()) + parts[0].get_width()
        parts[0].set_width(edge_piece_width, stretch=True, about_edge=RIGHT)
        parts[-1].set_width(edge_piece_width, stretch=True, about_edge=LEFT)

        parts.set_color(color)
        parts.set_shading(*shading)
        return parts

    def get_point_sources_from_wall(self, wall, z=0):
        sources = GlowDots(np.array([
            midpoint(p1.get_right(), p2.get_left())
            for p1, p2 in zip(wall, wall[1:])
        ]))
        sources.set_color(WHITE)
        sources.set_z(z)
        return sources

    def get_plane_wave(self, direction=UP):
        return LightWaveSlice(DotCloud([-1000 * direction]), decay_factor=0)

    def get_graph_over_wave(self, line, light_wave, color=WHITE, stroke_width=2, direction=OUT, scale_factor=0.5, n_curves=500):
        line.insert_n_curves(n_curves - line.get_num_curves())
        graph = line.copy()
        graph.line = line

        def update_graph(graph):
            points = graph.line.get_anchors()
            values = scale_factor * light_wave.wave_func(points)
            graph.set_points_smoothly(points + values[:, np.newaxis] * direction)

        graph.add_updater(update_graph)
        graph.apply_depth_test()
        graph.set_stroke(color, stroke_width)
        return graph


class LightExposingFilm(DiffractionGratingScene):
    def construct(self):
        # Set up wave
        frame = self.frame
        self.set_floor_plane("xz")

        source_dist = 16.5
        source = GlowDot(source_dist * OUT).set_opacity(0)
        wave = LightWaveSlice(source, decay_factor=0, wave_number=0.5)
        wave.set_opacity(0)
        wave_line = Line(source.get_center(), ORIGIN)
        wave_line.set_stroke(width=0)
        initial_wave_amp = 0.75
        wave_amp_tracker = ValueTracker(initial_wave_amp)
        graph = self.get_graph_over_wave(wave_line, wave, direction=UP, scale_factor=wave_amp_tracker.get_value(), n_curves=200)
        graph.add_updater(lambda m: m.stretch(wave_amp_tracker.get_value() / initial_wave_amp, dim=1))

        # Set up linear vector field
        def field_func(points):
            result = np.zeros_like(points)
            result[:, 1] = wave_amp_tracker.get_value() * wave.wave_func(points)
            return result

        linear_field = VectorField(field_func, sample_points=wave_line.get_points()[::4], max_vect_len=2.0)
        linear_field.always.update_vectors()
        linear_field.set_stroke(WHITE, width=1.5, opacity=0.75)

        # Add film
        film = Rectangle(16, 9)
        film.set_fill(GREY_E, 0.75)
        film.set_height(8)
        film.center()

        exp_source = GlowDot(OUT).set_opacity(0)
        exposure = LightIntensity(exp_source)
        exposure.set_color(GREEN)
        exposure.set_decay_factor(3)
        exposure.set_max_amp(0.15)
        exposure.set_opacity(0.7)
        exposure.replace(film, stretch=True)

        film_label = Text("Film", font_size=96)
        film_label.next_to(film, UP, MED_SMALL_BUFF)

        frame.reorient(-18, -7, 0, (0.46, -0.4, -2.46), 17.86)
        self.add(film, exposure, film_label)
        self.add(wave, linear_field, graph)

        # Fade in
        self.play(
            frame.animate.reorient(-88, -4, 0, (3.56, -0.71, 4.59), 14.60),
            FadeIn(exposure, time_span=(0, 3)),
            run_time=7.5
        )

        # Label amplitude
        low_line = DashedLine(8 * OUT, ORIGIN)
        high_line = low_line.copy().shift(wave_amp_tracker.get_value() * UP)
        amp_lines = VGroup(low_line, high_line)
        amp_lines.set_stroke(YELLOW, 3)
        brace = Brace(amp_lines, RIGHT)
        brace.rotate(PI / 2, DOWN)
        brace.next_to(amp_lines, OUT, SMALL_BUFF)
        amp_label = Tex(R"\\text{Amplitude}", font_size=72)
        amp_label.set_backstroke(BLACK, 5)
        amp_label.rotate(PI / 2, DOWN)
        amp_label.next_to(brace, OUT, SMALL_BUFF)
        fade_rect = Rectangle(8, 3)
        fade_rect.rotate(PI / 2, DOWN)
        fade_rect.next_to(amp_lines, OUT, buff=0)
        fade_rect.set_stroke(BLACK, 0)
        fade_rect.set_fill(BLACK, 0.7)

        self.play(
            FadeIn(fade_rect),
            GrowFromCenter(brace),
            FadeIn(amp_lines[0]),
            ReplacementTransform(amp_lines[0].copy().fade(1), amp_lines[1]),
        )
        self.play(Write(amp_label, stroke_color=WHITE, lag_ratio=0.1, run_time=2.0))
        self.wait_until(lambda: 8 / 30 < wave.uniforms["time"] % 1 < 9 / 30)

        # Label phase
        phase_text = Text("Phase", font_size=72)
        phase_circle = Circle(radius=0.75)
        phase_circle.set_stroke(BLUE, 2)
        phase_circle.next_to(phase_text, DOWN)
        phase_vect = Arrow(phase_circle.get_center(), phase_circle.get_right(), buff=0, thickness=2)
        phase_vect.set_fill(BLUE)

        phase_label = VGroup(phase_text, phase_circle, phase_vect)
        phase_label.rotate(PI / 2, DOWN)
        phase_label.next_to(amp_lines, DOWN, buff=2)
        phase_label.set_z(2)
        og_phase_label = phase_label.copy()

        wavelength = 1.0 / wave.uniforms["wave_number"]
        start_z = 0
        phase_line = Line(start_z * OUT, (start_z + wavelength) * OUT)
        phase_line.set_stroke(BLUE, 3)

        phase_arrow = Arrow(phase_text.get_corner(IN + UP) + 0.2 * (OUT + UP), phase_line.get_start(), buff=0)
        phase_arrow.always.set_perpendicular_to_camera(self.frame)

        self.play(
            wave.animate.pause(),
            FadeIn(phase_label),
            FadeIn(phase_arrow),
            frame.animate.reorient(-99, -2, 0, (3.56, -0.71, 4.59), 14.60),
        )
        self.play(
            Rotate(phase_vect, -TAU, axis=LEFT, about_point=phase_vect.get_start()),
            ShowCreation(phase_line),
            phase_arrow.animate.put_start_and_end_on(phase_arrow.get_start(), phase_line.get_end()),
            run_time=3
        )
        self.play(
            Rotate(phase_vect, TAU, axis=LEFT, about_point=phase_vect.get_start()),
            phase_arrow.animate.put_start_and_end_on(phase_arrow.get_start(), phase_line.get_start()),
            run_time=3
        )
        self.wait(2)

        # Decrease and increase amplitude
        amp_group = VGroup(amp_lines, brace)
        amp_group.f_always.set_height(wave_amp_tracker.get_value, stretch=lambda: True)
        amp_group.always.move_to(ORIGIN, IN + DOWN)
        amp_label.always.next_to(brace, OUT, SMALL_BUFF)
        self.add(amp_group, amp_label)

        phase_vect.add_updater(lambda m: m.put_start_and_end_on(
            phase_circle.get_center(),
            phase_circle.pfp((wave.uniforms["frequency"] * wave.uniforms["time"]) % 1),
        ))
        phase_vect.always.set_perpendicular_to_camera(self.frame)

        self.play(
            wave.animate.set_time_rate(0.5),
            frame.animate.reorient(-55, -13, 0, (2.59, -0.61, 2.18), 17.00),
            phase_arrow.animate.fade(0.8),
            phase_label.animate.fade(0.8),
            FadeOut(phase_line),
            run_time=3
        )
        self.play(
            wave_amp_tracker.animate.set_value(0.25),
            exposure.animate.set_opacity(0.25),
            run_time=3,
        )
        self.wait(2)
        self.play(
            wave_amp_tracker.animate.set_value(1.0),
            exposure.animate.set_opacity(1.0),
            run_time=4
        )
        self.wait()

        # Write exposure expression
        exp_expr = Tex(R"\\text{Exposure} = c \\cdot |\\text{Amplitude}|^2", font_size=72)
        exp_expr.move_to(2 * UP)

        self.play(
            LaggedStart(
                Write(exp_expr[R"\\text{Exposure} = c \\cdot |"][0]),
                TransformFromCopy(amp_label.copy().clear_updaters(), exp_expr[R"\\text{Amplitude}"][0]),
                Write(exp_expr[R"|^2"][0]),
                lag_ratio=0.1,
            ),
            frame.animate.reorient(-34, -15, 0, (1.48, -0.23, 1.29), 16.08),
            run_time=3,
        )
        self.wait(8)

        # Focus on phase again
        to_fade = VGroup(fade_rect, amp_group, amp_label)
        wave.pause()
        self.play(
            FadeOut(to_fade, lag_ratio=0.01, time_span=(0, 1.5)),
            phase_label.animate.match_style(og_phase_label),
            phase_arrow.animate.set_fill(opacity=1),
            frame.animate.reorient(-86, -4, 0, (2.43, -0.97, 1.94), 12.91),
            run_time=2
        )

        # Shift back the phase
        shift_label = TexText(R"Shift back phase $\\rightarrow$")
        shift_label.rotate(PI / 2, DOWN)
        shift_label.move_to(7 * OUT + 1.5 * DOWN)

        self.play(
            wave.animate.set_uniform(time_rate=-0.5).set_anim_args(rate_func=there_and_back),
            FadeIn(shift_label, shift=OUT),
            run_time=2.0
        )
        self.play(FadeOut(shift_label))
        self.wait()

        # Play for a while
        self.play(wave.animate.set_time_rate(0.5))
        self.wait(16)

        # Shine a second beam in
        source2 = GlowDot(source_dist * (OUT + RIGHT))
        wave2 = LightWaveSlice(source2)
        wave2.set_uniforms(dict(wave.uniforms))
        wave2.set_opacity(0)
        line2 = Line(ORIGIN, source2.get_center())
        ref_amp = 0.75
        graph2 = self.get_graph_over_wave(line2, wave2, direction=UP, scale_factor=ref_amp, n_curves=int(200 * math.sqrt(2)))
        graph2.set_color(YELLOW)
        graph2.update()

        def field_func2(points):
            result = np.zeros_like(points)
            result[:, 1] = ref_amp * wave2.wave_func(points)
            return result

        linear_field2 = VectorField(field_func2, sample_points=line2.get_points()[::4], max_vect_len=2.0)
        linear_field2.always.update_vectors()
        linear_field2.set_stroke(YELLOW, width=1.5, opacity=0.75)

        ref_wave_label = Text("Reference Wave", font_size=72)
        ref_wave_label.set_color(YELLOW)
        ref_wave_label.set_backstroke(BLACK, 3)
        ref_wave_label.rotate(PI / 4, DOWN)
        ref_wave_label.move_to([5, 1.25, 5])

        self.play(
            FadeOut(phase_label),
            FadeOut(phase_arrow),
            wave_amp_tracker.animate.set_value(0.75),
            exposure.animate.set_opacity(0.75),
            frame.animate.reorient(-32, -20, 0, (1.48, -0.73, 0.84), 14.72),
            run_time=2,
        )
        self.wait_until(lambda: 8 / 30 < wave.uniforms["time"] % 1 < 9 / 30)
        self.add(wave2)
        self.play(
            FadeIn(graph2),
            FadeIn(linear_field2),
        )
        self.wait()
        self.play(FadeIn(ref_wave_label, shift=UP))
        self.wait(5)

        # Zoom in
        self.play(
            frame.animate.reorient(-58, -18, 0, (3.29, 0.39, 0.55), 10.68),
            run_time=4
        )
        self.play(
            exposure.animate.set_opacity(1).set_max_amp(0.1),
            run_time=2
        )
        self.wait(3)

        # Shift back
        shift_label = TexText(R"Shift back phase $\\rightarrow$")
        shift_label.rotate(PI / 2, DOWN)
        shift_label.set_backstroke(BLACK, 5)
        shift_label.move_to(5 * OUT + 1.25 * UP)

        self.play(
            wave.animate.pause(),
            wave2.animate.pause(),
        )
        self.play(
            wave.animate.set_uniform(time_rate=-0.5).set_anim_args(rate_func=there_and_back),
            exposure.animate.set_max_amp(0.2).set_opacity(0.15),
            FadeIn(shift_label, OUT),
            run_time=2.0
        )
        self.play(FadeOut(shift_label))
        self.play(
            wave.animate.set_time_rate(0.5),
            wave2.animate.set_time_rate(0.5),
        )
        self.play(
            frame.animate.reorient(-54, -16, 0, (3.01, -0.09, 0.55), 14.47),
            run_time=10,
        )

        wave.pause()
        wave2.pause()
        self.play(
            wave.animate.set_uniform(time_rate=-0.5).set_anim_args(rate_func=there_and_back),
            exposure.animate.set_max_amp(0.1).set_opacity(1),
            FadeIn(shift_label, OUT),
            run_time=2.0
        )
        self.play(FadeOut(shift_label))
        wave.set_time_rate(0.5)
        wave2.set_time_rate(0.5)
        self.play(
            frame.animate.reorient(15, -22, 0, (2.14, -0.48, -0.07), 15.51),
            run_time=12,
        )

        wave.pause()
        wave2.pause()
        self.play(
            wave.animate.set_uniform(time_rate=-0.5).set_anim_args(rate_func=there_and_back),
            exposure.animate.set_max_amp(0.2).set_opacity(0.15),
            run_time=2.0
        )
        wave.set_time_rate(0.5)
        wave2.set_time_rate(0.5)
        self.play(
            frame.animate.reorient(15, -14, 0, (1.47, 0.09, -0.04), 13.26),
            run_time=12,
        )


class TwoInterferingWaves(DiffractionGratingScene):
    def construct(self):
        # Setup reference and object waves
        frame = self.frame
        self.set_floor_plane("xz")

        film_border = ScreenRectangle()
        film_border.set_height(6)
        film_border.set_fill(BLACK, 0)
        film_border.set_stroke(WHITE, 1)
        film_image = ImageMobject("HologramFilm.jpg")
        film_image.replace(film_border)

        frame.set_height(6)
        self.add(film_image, film_border)

        # Add waves
        obj_vect = 5 * OUT + 2 * LEFT
        ref_vect = 5 * OUT + 2 * RIGHT
        lp = film_border.get_left() + 3 * DOWN
        rp = film_border.get_right() + 3 * DOWN

        obj_points = DotCloud(np.random.random((31, 3)))
        obj_points.set_height(10)
        obj_points.move_to(obj_vect)
        obj_wave = LightWaveSlice(obj_points)
        obj_wave.set_wave_number(8)
        obj_wave.set_frequency(1)
        obj_wave.set_max_amp(3)

        ref_wave = LightWaveSlice(TrueDot(10 * ref_vect))
        ref_wave.match_points(obj_wave)
        ref_wave.set_uniforms(dict(obj_wave.uniforms))
        ref_wave.set_max_amp(1.25)
        ref_wave.set_decay_factor(0)
        ref_wave.set_opacity(0.85)

        rect = film_border
        obj_wave.set_points([
            2 * obj_vect, rect.get_corner(UL), rect.get_corner(DL),
            2 * obj_vect, rect.get_corner(UL), rect.get_corner(UR),
            2 * obj_vect, rect.get_corner(UR), rect.get_corner(DR),
            2 * obj_vect, rect.get_corner(DR), rect.get_corner(DL),
        ])
        ref_wave.set_points([
            2 * ref_vect, rect.get_corner(UL), rect.get_corner(DL),
            2 * ref_vect, rect.get_corner(UL), rect.get_corner(UR),
            2 * ref_vect, rect.get_corner(UR), rect.get_corner(DR),
            2 * ref_vect, rect.get_corner(DR), rect.get_corner(DL),
        ])
        obj_wave.deactivate_depth_test()
        ref_wave.deactivate_depth_test()

        waves = Group(ref_wave, obj_wave)
        waves.set_opacity(0.25)

        self.play(
            FadeIn(waves),
            frame.animate.reorient(0, -21, 0, (-0.09, -0.86, 0.04), 9.40),
            run_time=3
        )
        self.wait(3)

        # Diversion into backdrop for complex wave scene
        if False:
            # Isolate object wave
            new_obj_wave = LightWaveSlice(obj_points)
            new_obj_wave.set_points([2 * obj_vect, rect.get_left(), rect.get_right()])
            new_obj_wave.set_uniforms(dict(obj_wave.uniforms))
            new_obj_wave.set_opacity(1)
            new_obj_wave.set_frequency(0.25)

            frame.reorient(0, -19, 0, (0.01, -0.03, -0.28), 12.53)
            waves.set_opacity(0.4)
            for wave in waves:
                wave.set_frequency(0.25)
            self.wait(4)

            self.play(
                FadeOut(ref_wave),
                FadeOut(film_border),
                FadeOut(film_image),
                FadeOut(obj_wave),
                FadeIn(new_obj_wave)
            )

            # Go to 2d slice
            self.play(
                new_obj_wave.animate.scale(10),
                frame.animate.reorient(-3, -5, 0, (-0.96, 0.03, -0.15), 5.32),
                run_time=5
            )
            self.wait(4)

            # Circle a point
            circle = Circle(radius=0.1).set_stroke(WHITE, 2)
            circle.move_to(LEFT + 2 * DOWN)
            circle.reverse_points()
            full_rect = FullScreenRectangle()
            full_rect.set_fill(BLACK, 0.75)
            full_rect.append_points([full_rect.get_end(), *circle.get_points()])

            circle.fix_in_frame()
            full_rect.fix_in_frame()

            self.play(
                FadeIn(full_rect),
                ShowCreation(circle),
            )
            self.wait(8)

        # Add exposure
        exposure = LightIntensity(obj_points)
        exposure.set_color(WHITE)
        exposure.replace(film_border, stretch=True)
        exposure.set_wave_number(256)
        exposure.set_max_amp(4)

        # One more insertion
        exposure.set_opacity(0.5)
        self.add(exposure, waves)
        frame.reorient(-18, -30, 0, (-5.85, -0.18, -0.89), 15.16)
        self.play(
            frame.animate.reorient(17, -24, 0, (-5.5, -0.82, -0.67), 15.86),
            run_time=30
        )

        self.add(exposure, waves)
        self.play(
            FadeIn(exposure, run_time=5),
            FadeOut(waves, time_span=(3, 5)),
        )

        # Zoom in on the film
        self.play(
            frame.animate.reorient(0, 0, 0, (0.5, 0.07, 0.0), 0.15),
            film_image.animate.set_opacity(0.1).set_anim_args(time_span=(0, 5)),
            run_time=8,
        )
        self.play(frame.animate.reorient(0, 0, 0, (0.41, 0.03, 0.0), 0.05), run_time=8)


class SinglePointOnFilm(DiffractionGratingScene):
    def construct(self):
        # Reference image
        img = ImageMobject("SinglePointHologram")
        img.set_height(FRAME_HEIGHT)
        img.fix_in_frame()
        img.set_opacity(0.75)
        # self.add(img)

        # Create object and reference wave
        # Much of this copied from CreateZonePlate below
        frame = self.frame
        frame.set_field_of_view(42 * DEGREES)
        axes = ThreeDAxes()
        self.set_floor_plane("xz")

        wave_width = 100
        wave_number = 2
        frequency = 1

        source_point = GlowDot(color=WHITE, radius=0.5)
        source_point.move_to([0., -0.75, 4.62])
        obj_wave = LightWaveSlice(source_point)
        obj_wave.set_decay_factor(0.7)

        obj_wave.set_width(wave_width)
        obj_wave.rotate(PI / 2, RIGHT, about_point=ORIGIN)
        obj_wave.move_to(source_point)
        obj_wave.set_wave_number(wave_number)
        obj_wave.set_frequency(frequency)

        # Add film
        plate_border = Rectangle(4, 4)
        plate_border.set_fill(BLACK, 0)
        plate_border.set_stroke(WHITE, 2)
        plate_body = Square3D()
        plate_body.set_color(BLACK, 0.9)
        plate_body.replace(plate_border, stretch=True)
        plate_body.set_shading(0.1, 0.1, 0)

        plate = Group(plate_body, plate_border)
        plate.center()

        film_label = Text("Film")
        film_label.next_to(plate, UP)
        film_label.set_backstroke(BLACK, 3)
        film_label.set_z_index(1)

        # Spherical waves
        wave_stack = Group(
            obj_wave.copy().rotate(PI / 2, OUT).set_opacity(0.1)
            for x in range(30)
        )
        wave_stack.set_opacity(0.07)
        wave_stack.arrange_to_fit_width(4)
        wave_stack.sort(lambda p: -p[0])
        wave_stack.move_to(obj_wave)

        # Label object wave
        source_label = Text("Object (idealized point)", font_size=36)
        source_label.next_to(source_point, UR, buff=MED_LARGE_BUFF)
        source_label.shift(0.25 * UL)
        source_label.set_backstroke(BLACK, 2)
        source_arrow = Arrow(source_label["Object"].get_bottom(), source_point.get_center(), buff=0.1)
        source_arrow.set_perpendicular_to_camera(self.frame)
        source_label.add(source_arrow)
        source_label.rotate(PI / 2, DOWN, about_point=source_point.get_center())
        obj_point = TrueDot()
        obj_point.move_to(source_point)

        obj_wave_label = Text("Object wave")
        obj_wave_label.rotate(PI / 2, LEFT)
        obj_wave_label.next_to(source_point, IN, buff=0.1)
        obj_wave_label.set_backstroke(BLACK, 2)

        frame.reorient(-88, -12, 0, (0.22, -0.81, 4.62), 13.76)
        self.add(plate, film_label)
        self.add(obj_wave, wave_stack, obj_point, source_point)
        self.add(source_label)
        self.wait()

        # Slowly reorient
        self.play(
            frame.animate.reorient(-14, -11, 0, (0.27, 0.14, 3.59), 6.17),
            Rotate(source_label, PI / 2, UP, about_point=source_point.get_center()),
            Rotate(wave_stack, PI / 2, UP, about_point=source_point.get_center()),
            run_time=10
        )

        # Look from above
        self.play(
            FadeOut(wave_stack),
            FadeOut(source_label),
            FadeOut(plate),
            FadeOut(film_label),
            obj_wave.animate.set_decay_factor(0.5),
            frame.animate.reorient(0, -90, 0, source_point.get_center(), 4).set_anim_args(run_time=4)
        )
        self.wait(2)


class ExplainWaveVisualization(DiffractionGratingScene):
    def construct(self):
        # Set up the wave
        full_width = 20
        frame = self.frame

        source = GlowDot(ORIGIN, color=WHITE)
        source.set_radius(0.5)
        wave = LightWaveSlice(source)
        wave.set_width(full_width)
        wave.move_to(ORIGIN)

        frame.reorient
        self.add(wave, source)

        # Talk through what's displayed
        def field_func(points):
            result = np.zeros_like(points)
            result[:, 2] = 0.5 * wave.wave_func(points)
            return result

        linear_field = VectorField(
            field_func,
            sample_points=np.linspace(ORIGIN, 10 * UP, 100),
            max_vect_len=1.0,
        )
        linear_field.always.update_vectors()
        linear_field.set_stroke(WHITE, width=1.5, opacity=0.75)
        full_field = VectorField(
            field_func,
            width=full_width,
            height=full_width,
            x_density=5,
            y_density=5,
            max_vect_len=0.5,
        )
        full_field.sample_points
        full_field.set_stroke(WHITE, width=1.5, opacity=0.25)
        full_field.always.update_vectors()

        self.play(
            frame.animate.reorient(71, 77, 0, (-0.62, 0.7, 0.19), 3.11),
            wave.animate.set_uniform(time_rate=0.5).set_anim_args(suspend_mobject_updating=False),
            VFadeIn(full_field, time_span=(0, 2)),
            run_time=6
        )
        self.wait(2)
        self.play(
            VFadeIn(linear_field),
            VFadeOut(full_field),
        )
        self.wait(5)
        self.wait_until(lambda: 15 / 30 < wave.uniforms["time"] % 1 < 16 / 30)
        self.play(
            wave.animate.pause().set_anim_args(suspend_mobject_updating=False),
        )
        self.wait()

        # Show example vectors
        sample_vect = Vector(OUT, thickness=1.0)
        sample_vect.set_fill(WHITE, 1, border_width=0.5)
        sample_vect.base_point = Dot(0.705 * UP, fill_color=BLUE, radius=0.02)

        def update_sample_vect(vect):
            point = vect.base_point.get_center()
            vect.put_start_and_end_on(point, point + 0.95 * field_func([point])[0])
            vect.set_perpendicular_to_camera(self.frame)

        update_sample_vect(sample_vect)

        self.play(
            linear_field.animate.set_stroke(opacity=0.2),
            GrowArrow(sample_vect, run_time=2),
            frame.animate.reorient(81, 84, 0, (-0.5, 0.69, 0.23), 1.65).set_anim_args(run_time=3)
        )
        self.wait()
        self.play(FadeIn(sample_vect.base_point, 0.1 * IN))
        self.wait()
        sample_vect.add_updater(update_sample_vect)
        self.play(sample_vect.base_point.animate.move_to(1.315 * UP).set_color(RED), run_time=3)
        self.wait()
        self.play(sample_vect.base_point.animate.move_to(1.01 * UP).set_color(GREY_E), run_time=2)
        self.wait()
        self.play(
            VFadeOut(sample_vect, time_span=(0, 2)),
            FadeOut(sample_vect.base_point, time_span=(0, 2)),
            linear_field.animate.set_stroke(opacity=0.75).set_anim_args(time_span=(0, 2)),
            wave.animate.set_uniform(time_rate=0.5).set_anim_args(suspend_mobject_updating=False),
            frame.animate.reorient(49, 79, 0, (-0.14, 1.34, 0.23), 2.78),
            run_time=5,
        )
        self.wait(2)

        # Show full 3d field
        full_field = VectorField(
            lambda p: 0.5 * field_func(p),
            x_density=4,
            y_density=4,
            z_density=2,
            depth=4,
            max_vect_len=0.5,
        )
        full_field.set_stroke(WHITE, width=1, opacity=0.25)
        full_field.always.update_vectors()

        added_waves = Group(
            wave.copy().rotate(PI / 2, UP).shift(z * RIGHT).set_opacity(0.1)
            for z in np.arange(-2, 2, 0.1)
        )

        self.play(
            # VFadeIn(full_field),
            VFadeOut(linear_field),
            FadeIn(added_waves, suspend_mobject_updating=False),
            frame.animate.reorient(48, 79, 0, (-0.42, 1.07, 0.35), 4.18).set_anim_args(run_time=8)
        )
        wave.save_state()
        wave.scale(0)
        self.play(
            Restore(wave, suspend_mobject_updating=False),
            FadeOut(added_waves, time_span=(0, 1), suspend_mobject_updating=False),
            run_time=3,
        )
        self.wait(2)

        # Show a sine wave
        line = Line(ORIGIN, 10 * UR)
        line.shift(source.get_center())
        line.set_stroke(TEAL, 2)
        graph = self.get_graph_over_wave(line, wave)
        graph.set_stroke(WHITE, 2)
        wave.set_z_index(1)
        source.set_z_index(1)

        linear_field = VectorField(
            field_func,
            sample_points=line.get_anchors()[::2],
            max_vect_len=1.0,
        )
        linear_field.set_stroke(WHITE, 0.75)
        linear_field.always.update_vectors()

        self.add(line, graph)
        self.play(
            ShowCreation(line, time_span=(0, 3)),
            ShowCreation(graph, time_span=(0, 3)),
            VFadeIn(linear_field, time_span=(0, 3), suspend_mobject_updating=False),
            frame.animate.reorient(0, 73, 0, (-0.03, 1.7, 0.2), 4.65),
            run_time=10
        )
        self.wait(8)


class CreateZonePlate(DiffractionGratingScene):
    samples = 4

    def construct(self):
        # Create object and reference wave
        frame = self.frame
        axes = ThreeDAxes()
        self.set_floor_plane("xz")

        wave_width = 100
        wave_number = 4
        frequency = 1

        ref_wave = self.get_plane_wave(direction=IN)
        ref_wave.set_opacity(0.75)
        ref_source = ref_wave.point_sources
        source_point = GlowDot(OUT, color=WHITE, radius=0.5)
        obj_wave = LightWaveSlice(source_point)
        obj_wave.set_decay_factor(0.7)

        for wave in [obj_wave, ref_wave]:
            wave.set_width(wave_width)
            wave.rotate(PI / 2, RIGHT, about_point=ORIGIN)
            wave.center()
            wave.set_wave_number(wave_number)
            wave.set_frequency(frequency)

        frame.reorient(-32, -21, 0, (-0.74, 0.32, -0.49), 7.08)

        def get_all_sources():
            return np.vstack([
                obj_wave.point_sources.get_points(),
                ref_wave.point_sources.get_points()
            ])

        # Add film
        plate = Rectangle(16, 9)
        plate.set_height(4)
        plate.set_stroke(WHITE, 1, 0.5).set_fill(BLACK, 0.0)
        plate.set_shading(0.1, 0.1, 0)
        plate.apply_depth_test()
        plate_body = Square3D()
        plate_group = Group(plate_body, plate)

        plate_body.set_color(BLACK, 0.9)
        plate_body.set_shape(plate.get_width(), plate.get_height())
        plate_body.move_to(plate.get_center() + 1e-2 * IN)

        exposure = LightIntensity(DotCloud(get_all_sources()))
        exposure.set_decay_factor(0)
        exposure.set_wave_number(wave_number)
        exposure.replace(plate, stretch=True).shift(1e-2 * OUT)
        exposure.set_color(WHITE, 0.85)

        film = Group(plate, exposure)
        film.set_height(4)
        film.set_z(-2)

        film_label = Text("Film")
        film_label.next_to(plate, UP)
        film_label.set_backstroke(BLACK, 3)
        film_label.set_z_index(1)

        # Label object wave
        source_label = Text("Object (idealized point)", font_size=24)
        source_label.next_to(source_point, UR, buff=0)
        source_label.shift(0.25 * UL)
        source_label.set_backstroke(BLACK, 2)
        source_arrow = Arrow(source_label["Object"].get_bottom(), source_point.get_center(), buff=0.1)
        source_arrow.always.set_perpendicular_to_camera(self.frame)
        obj_point = TrueDot()
        obj_point.move_to(source_point)

        obj_wave_label = Text("Object wave")
        obj_wave_label.rotate(PI / 2, LEFT)
        obj_wave_label.next_to(source_point, IN, buff=0.1)
        obj_wave_label.set_backstroke(BLACK, 2)

        frame.reorient(41, -9, 0, (-0.72, 0.27, -0.49), 6.75)
        self.add(plate_group, obj_wave, film_label)
        plate_body.move_to(plate.get_center() + 1e-2 * IN)
        self.play(
            FadeIn(source_point),
            FadeIn(source_label),
            GrowArrow(source_arrow),
            frame.animate.reorient(-4, -10, 0, (-0.74, 0.32, -0.49), 7.08).set_anim_args(run_time=5)
        )
        self.play(
            TransformMatchingStrings(source_label, obj_wave_label),
            FadeOut(source_arrow),
            frame.animate.reorient(0, -47, 0, (-0.74, 0.32, -0.49), 7.08).set_anim_args(run_time=3),
        )
        self.wait(3)

        # Label reference wave
        ref_wave.match_width(film, stretch=True)
        ref_wave.move_to(film, IN)
        ref_wave_label = Text("Reference wave")
        ref_wave_label.rotate(PI / 2, LEFT)
        ref_wave_label.next_to(obj_wave_label, OUT, buff=1.5)
        ref_wave_label.set_backstroke(BLACK, 2)

        wave_fronts = Group(
            plate.copy().set_color([BLUE, RED][z % 2], 0.15).shift(0.5 * z * OUT)
            for z in range(4, 16)
        )
        for front in wave_fronts:
            front.add_updater(lambda m, dt: m.shift(dt * (frequency / wave_number) * IN))

        self.play(
            FadeOut(obj_wave, 0.1 * DOWN),
            FadeOut(obj_wave_label),
            FadeOut(source_point),
            FadeIn(ref_wave, 0.1 * DOWN),
            FadeIn(ref_wave_label),
            frame.animate.reorient(0, -40, 0, (-0.16, 0.01, -0.24), 7.08).set_anim_args(run_time=4),
        )
        self.play(
            FadeIn(wave_fronts, time_span=(0, 2), lag_ratio=0.05),
            frame.animate.reorient(0, -37, 0, (-0.16, 0.01, -0.24), 7.08),
            run_time=8
        )
        self.play(FadeOut(wave_fronts, run_time=2, lag_ratio=0.1))
        self.add(ref_source)

        # Put reference at an angle
        angle = 60 * DEGREES
        direction = rotate_vector(OUT, angle, axis=UP)
        dist = ref_wave.get_depth()
        ref_wave.save_state()
        ref_wave.point_sources.save_state()
        ref_wave.target = ref_wave.generate_target()
        p0 = film.get_left()
        p1 = film.get_right()
        p2 = p0 + dist * direction
        p3 = p1 + dist * direction
        ref_wave.target.set_points([p2, p0, p3, p1])

        self.play(
            MoveToTarget(ref_wave, run_time=2),
            Rotate(ref_wave.point_sources, angle, axis=UP, about_point=ORIGIN),
            Rotate(ref_wave_label, angle, axis=UP, about_point=film.get_center()),
            run_time=10,
            rate_func=lambda t: there_and_back_with_pause(t, 0.5)
        )
        self.wait(8)

        # Also show the object wave from this perspective
        self.remove(ref_wave, ref_wave_label)
        self.add(obj_wave, obj_wave_label, source_point)
        obj_wave_label.shift(0.25 * OUT)
        self.wait(8)

        # Show combined wave
        comb_label = Text("Combined wave")
        comb_label.rotate(PI / 2, LEFT)
        comb_label.next_to(ref_wave_label, UP)
        comb_label.set_backstroke(BLACK, 3)
        comb_wave = obj_wave.copy()
        comb_wave.point_sources = DotCloud([
            *(source_point.get_center() for x in range(2)),
            *np.linspace(20 * OUT + 4 * LEFT, 20 * OUT + 4 * RIGHT, 25)
        ])
        comb_wave.set_decay_factor(0.8)
        comb_wave.set_max_amp(1.5)

        self.remove(obj_wave, obj_wave_label)
        self.add(comb_wave, comb_label, source_point)
        self.wait(8)

        # Preview exposure
        self.add(exposure, comb_wave, comb_label)
        self.play(FadeIn(exposure, run_time=3))
        self.wait(8)

        # Change to side view
        self.play(
            FadeOut(comb_label),
            comb_wave.animate.set_opacity(0.1),
            FadeOut(film_label),
            FadeOut(exposure),
            frame.animate.reorient(80, -2, 0, (-0.64, 0.23, -0.93), 4.34),
            run_time=5
        )

        # Add graphs to middle
        ref_source.get_center()
        ref_source.move_to(source_point.get_center() + ((1000 // wave_number) * wave_number) * OUT)
        ref_wave.set_uniform(time=obj_wave.uniforms["time"])

        obj_color, ref_color = colors = [TEAL, YELLOW]
        obj_line, ref_line = lines = VGroup(
            Line(source_point.get_center(), film.get_center()).set_stroke(color, 1, 0.5)
            for color in colors
        )
        ref_line.scale(2, about_point=ref_line.get_end())
        ref_line.shift(0.02 * RIGHT)

        obj_graph, ref_graph = graphs = VGroup(
            self.get_graph_over_wave(line, wave, scale_factor=sf, direction=UP, color=color)
            for line, color, wave, sf in zip(lines, colors, [obj_wave, ref_wave], [0.15, 0.1])
        )
        graphs.set_stroke(width=2, opacity=1)

        obj_label = Text("Object wave", font_size=24).rotate(PI / 2, UP)
        ref_label = Text("Reference wave", font_size=24).rotate(PI / 2, UP)
        obj_label.set_color(obj_color).next_to(obj_graph, UP, aligned_edge=OUT)
        obj_label.shift(0.1 * IN)
        ref_label.set_color(ref_color).next_to(ref_graph, DOWN)
        ref_label.match_z(obj_label).shift(OUT)

        obj_wave.set_opacity(0)
        obj_wave.set_decay_factor(0.5)
        ref_wave.set_opacity(0)

        comb_wave.set_z_index(1)

        self.add(obj_wave, ref_wave)
        self.play(
            ShowCreation(obj_line),
            ShowCreation(obj_graph),
            FadeIn(obj_label, lag_ratio=0.1),
            run_time=2
        )
        self.wait(3)
        self.play(
            ShowCreation(ref_line),
            ShowCreation(ref_graph),
            FadeIn(ref_label, lag_ratio=0.1),
        )
        self.wait(4)

        # Show middle exposure
        round_exposure = self.get_round_exposure(exposure, radius=0.25)

        self.play(
            GrowFromCenter(round_exposure),
            frame.animate.reorient(53, -15, 0, (-0.83, 0.12, -0.62), 5.25).set_anim_args(run_time=3),
        )

        # Look off center
        exposure.replace(plate, stretch=True)
        exposure.set_shape(0.5 * plate.get_width(), 0.25)
        exposure.move_to(plate.get_center(), LEFT).shift(1e-2 * OUT)
        full_exposure = exposure.copy()
        full_exposure.replace(plate, stretch=True)
        full_exposure.shift(2e-2 * OUT)
        exposure.save_state()
        exposure.stretch(0, 0, about_edge=LEFT)

        O_point = obj_label[0].get_center()
        obj_label.add_updater(
            lambda m: m.rotate(
                angle_of_vector((m[-1].get_center() - m[0].get_center())[0::2]) - angle_of_vector(obj_line.get_vector()[0::2]),
                axis=UP,
            ).shift(O_point - m[0].get_center())
        )

        trg_point = VectorizedPoint(plate.get_center())
        obj_line.add_updater(lambda m: m.put_start_and_end_on(source_point.get_center(), trg_point.get_center()))
        ref_line.add_updater(lambda m: m.move_to(trg_point.get_center() + 0.02 * RIGHT, IN))

        self.play(
            obj_wave.animate.pause(),
            ref_wave.animate.pause(),
            comb_wave.animate.pause(),
        )
        self.play(
            trg_point.animate.move_to(film.get_right()),
            round_exposure.animate.shift(0.01 * OUT),
            MaintainPositionRelativeTo(ref_label, ref_line),
            Restore(exposure),
            frame.animate.reorient(35, -14, 0, (1.01, 0.23, -2.8), 8.40),
            run_time=12
        )
        self.wait(2)

        # Look to halfwavelength point
        trg_x = 0.9
        mid_line = Line(source_point.get_center(), plate.get_center())
        mid_line.set_stroke(TEAL, 2)
        mid_line_label = Tex("D", font_size=30).rotate(PI / 2, LEFT)
        mid_line_label.next_to(mid_line, LEFT)
        d_line_label = Tex(R"D + \\frac{\\lambda}{2}", font_size=30).rotate(PI / 2, LEFT)
        VGroup(mid_line_label, d_line_label).set_fill(TEAL, 1)

        self.play(
            FadeOut(obj_label),
            FadeOut(ref_label),
            FadeOut(ref_line),
            FadeOut(ref_graph),
            frame.animate.reorient(0, -62, 0, (-0.27, -1.26, -1.17), 7.38),
            trg_point.animate.move_to(plate.get_center() + trg_x * RIGHT),
            run_time=3
        )
        d_line_label.next_to(obj_line.get_center(), RIGHT, buff=SMALL_BUFF)
        self.play(
            FadeIn(mid_line),
            FadeIn(mid_line_label),
            FadeIn(d_line_label),
        )
        self.wait(2)
        self.play(
            frame.animate.reorient(84, -9, 0, (-0.26, -0.27, -1.86), 3.23),
            FadeOut(VGroup(mid_line, mid_line_label, d_line_label)),
            FadeIn(ref_line),
            FadeIn(ref_graph),
            obj_wave.animate.unpause(),
            ref_wave.animate.unpause(),
            comb_wave.animate.unpause(),
            run_time=5,
        )
        self.wait(5)

        # Show the circle
        circle = Circle(radius=trg_x)
        circle.move_to(film)
        circle.set_stroke(GREY_D, 1)

        tail = TracingTail(circle.get_end, stroke_color=BLUE_D, stroke_width=(0, 3))

        self.add(tail)
        self.wait()
        self.add(circle, tail)
        round_exposure.set_width(0.25)
        self.play(
            frame.animate.reorient(41, -15, 0, (-0.54, -0.17, -1.78), 4.65),
            ShowCreation(circle),
            UpdateFromFunc(trg_point, lambda m, c=circle: m.move_to(c.get_end())),
            round_exposure.animate.set_width(3),
            FadeOut(exposure, time_span=(0, 1)),
            run_time=4
        )
        self.play(FadeOut(circle, run_time=2))
        self.remove(tail)
        self.wait(2)

        # Grow rings fully
        exposure.replace(plate, stretch=True).shift(0.01 * OUT)

        self.add(exposure, round_exposure)
        self.play(
            FadeOut(round_exposure, run_time=2),
            FadeIn(exposure, run_time=2),
            comb_wave.animate.set_opacity(0.5).set_anim_args(time_span=(6, 8)),
            frame.animate.reorient(0, -23, 0, (-0.14, 0.01, -2.23), 6.52).set_anim_args(run_time=8)
        )

        # Just kinda hang for a bit
        time0 = self.time
        frame.add_updater(lambda m, t0=time0, sc=self: m.set_theta(math.sin(0.1 * (sc.time - t0)) * 30 * DEGREES))
        self.play(trg_point.animate.move_to(film.get_left()), run_time=10)
        self.play(trg_point.animate.move_to(film.get_right()), run_time=20)
        self.wait(5)
        frame.clear_updaters()

        # Change wavelength
        trg_wave_number = 16

        for wave in [obj_wave, ref_wave, comb_wave, exposure]:
            wave.set_wave_number(trg_wave_number)
            wave.set_frequency(0.25 * trg_wave_number)

        ref_line.insert_n_curves(1000)
        obj_line.insert_n_curves(1000)
        ref_graph.set_stroke(width=1)
        obj_graph.set_stroke(width=1)

        self.wait(4)
        self.play(
            FadeOut(VGroup(ref_line, ref_graph, obj_line, obj_graph)),
            FadeOut(comb_wave)
        )

        # Bring objet closer in
        self.play(
            frame.animate.reorient(-85, -9, 0, (0.65, -0.24, -0.14), 6.52),
            run_time=4
        )
        exposure.point_sources.set_points([OUT, 1001 * OUT])
        exposure.point_sources.set_radius(0)

        mid_line = Line()
        mid_line.set_stroke(TEAL, 2)

        def get_film_point():
            x, y, _ = source_point.get_center()
            z = film.get_z()
            return np.array([x, y, z])

        mid_line.add_updater(lambda m: m.set_points_as_corners([source_point.get_center(), get_film_point()]))

        def get_dist_label():
            label = DecimalNumber(mid_line.get_length(), font_size=24)
            label.set_backstroke(BLACK, 3)
            label.rotate(PI / 2, DOWN)
            label.next_to(mid_line, UP, SMALL_BUFF)
            return label

        dist_label = always_redraw(get_dist_label)

        self.play(
            ShowCreation(mid_line, suspend_mobject_updating=True),
            VFadeIn(dist_label),
        )

        for vect in [2 * IN, 4 * OUT, 2 * IN]:
            self.play(
                source_point.animate.shift(vect),
                exposure.point_sources.animate.shift(vect),
                run_time=3
            )
            self.wait()

        # Move in 3d
        axes = ThreeDAxes()
        axes.match_width(film)
        axes.move_to(film)
        mid_line.set_z_index(1)
        dist_label.clear_updaters()

        self.play(
            Write(axes, lag_ratio=0.01),
            frame.animate.reorient(-44, -21, 0, (0.09, -0.06, -1.4), 7.22),
            exposure.animate.set_opacity(0.5),
            FadeOut(dist_label),
            run_time=3
        )

        frame.add_ambient_rotation(2 * DEGREES)
        exposure.point_sources.add_updater(lambda m: m.move_to(source_point, IN))
        points = [RIGHT, RIGHT + IN, 3 * LEFT + IN, 3 * LEFT + 2 * OUT, 2 * OUT + 3 * RIGHT + 2 * UP, UR, OUT]
        for point in points:
            self.play(source_point.animate.move_to(point), run_time=2)
            self.wait()
        frame.clear_updaters()
        self.play(
            FadeOut(axes),
            FadeOut(mid_line),
            FadeOut(source_point),
        )

        # Shine the reference through it
        ref_wave.set_opacity(0.75)
        ref_wave.set_wave_number(4.0)
        ref_wave.set_frequency(1.0)
        ref_wave.unpause()

        self.add(exposure, ref_wave)
        self.play(
            FadeIn(ref_wave, time_span=(0, 2)),
            frame.animate.reorient(0, -24, 0, (0.08, -0.07, -1.39), 7.22),
            run_time=5
        )
        self.wait(4)

        # Go to other side
        frame.reorient(-171, -18, 0, (0.12, -0.21, -1.33), 9.31)
        self.remove(film)
        self.add(exposure, plate)
        self.play(
            frame.animate.reorient(0, -16, 0, (-3.06, -0.18, -3.19), 1.49),
            run_time=10,
        )
        self.wait(4)

    def get_round_exposure(self, exposure, radius=1.0, n_pieces=128):
        d_theta = TAU / n_pieces
        vects = [rotate_vector(RIGHT, theta) for theta in np.linspace(0, TAU, n_pieces + 1)]
        result = Group(
            exposure.copy().set_points([ORIGIN, v1, v2])
            for v1, v2 in zip(vects, vects[1:])
        )
        result.set_width(radius)
        result.move_to(exposure)
        return result

        self.add(round_exposure)

    def get_3d_waves(self, wave, x_range=(-4, 4, 0.5), opacity=0.25):
        waves = Group(
            wave.copy().rotate(PI / 2, OUT).move_to(x * RIGHT)
            for x in np.arange(*x_range)
        )
        waves.set_opacity(opacity)
        cam_point = self.frame.get_implied_camera_location()
        waves.sort(lambda p: -get_norm(p - cam_point))
        return waves


class ShowEffectOfChangedReferenceAngle(InteractiveScene):
    def construct(self):
        # Create object and reference wave
        frame = self.frame
        axes = ThreeDAxes()
        self.set_floor_plane("xz")

        wave_width = 100
        wave_number = 10
        frequency = 1

        source_points = GlowDots([OUT, 10 * OUT], color=WHITE, radius=0.0)
        wave = LightWaveSlice(source_points)

        wave.set_width(wave_width)
        wave.rotate(PI / 2, RIGHT, about_point=ORIGIN)
        wave.center()
        wave.set_wave_number(wave_number)
        wave.set_frequency(frequency)
        wave.set_max_amp(1.5)
        wave.set_decay_factor(0.5)

        # Add film
        plate_border = Rectangle(16, 9)
        plate_border.set_fill(BLACK, 0)
        plate_border.set_stroke(WHITE, 2)
        plate = Square3D()
        plate.set_color(BLACK, 0.9)
        plate.replace(plate_border, stretch=True)
        plate.set_shading(0.1, 0.1, 0)
        exposure = LightIntensity(source_points)
        exposure.set_decay_factor(0)
        exposure.set_wave_number(wave_number)
        exposure.replace(plate_border, stretch=True).shift(1e-2 * OUT)
        exposure.set_color(WHITE, 0.85)

        plate_group = Group(plate_border, plate)
        film = Group(plate_group, exposure)
        film.set_height(4)
        film.set_z(-2)

        film_label = Text("Film")
        film_label.next_to(plate_group, UP)
        film_label.set_backstroke(BLACK, 3)
        film_label.set_z_index(1)

        source_dot = GlowDot(source_points.get_points()[0], color=WHITE, radius=0.5)

        self.add(film, exposure, wave, source_points, source_dot)

        # Add reference wave lines
        plate_border.insert_n_curves(max(25 - plate_border.get_num_curves(), 0))
        film_points = np.array([plate_border.pfp(a) for a in np.linspace(0, 1, 500)])
        ref_lines = Line().replicate(len(film_points))
        ref_lines.set_stroke(GREEN_SCREEN, width=1, opacity=0.25)

        def update_ref_lines(lines):
            for line, point in zip(lines, film_points):
                line.set_points_as_corners([source_points.get_points()[1], point])
            return lines

        ref_lines.add_updater(update_ref_lines)

        self.add(ref_lines)

        # Move reference wave
        frame.reorient(-52, -29, 0, (1.21, -0.31, 1.52), 9.28)
        self.wait()
        self.play(
            Rotate(source_points, 60 * DEGREES, axis=UP, about_point=source_points.get_points()[0], run_time=5),
            frame.animate.reorient(-48, -46, 0, (1.17, -0.41, 1.55), 9.28),
            run_time=5
        )
        self.wait(3)


class DoubleSlit(DiffractionGratingScene):
    def construct(self):
        # Show a diffraction grating
        frame = self.frame
        full_width = 40

        n_slit_wall = self.get_wall_with_slits(16, spacing=1.0, total_width=full_width)
        n_slit_wall.move_to(0.5 * IN, IN)
        n_slit_wall.save_state()
        n_slit_wall.arrange(RIGHT, buff=0)
        n_slit_wall.move_to(n_slit_wall.saved_state)

        in_wave = self.get_plane_wave()
        in_wave.set_opacity(0.85)
        in_wave.set_width(full_width)
        in_wave.move_to(ORIGIN, UP)

        line = Line(0.5 * IN + 16 * DOWN + 0.5 * OUT, 0.5 * IN + 0.5 * OUT)
        graph = self.get_graph_over_wave(line, in_wave)

        self.add(graph)
        self.add(n_slit_wall)
        self.add(in_wave)

        frame.reorient(-31, 67, 0, (-3.1, 1.32, -1.12), 15.89)
        self.play(
            frame.animate.reorient(33, 65, 0, (1.24, 1.09, -0.39), 10.01),
            UpdateFromAlphaFunc(
                graph,
                lambda m, a: m.set_stroke(width=3 * clip(there_and_back_with_pause(2 * a, 0.7), 0, 1)),
            ),
            Restore(n_slit_wall, time_span=(9, 12)),
            run_time=15
        )

        # Preview the other side
        sources = self.get_point_sources_from_wall(n_slit_wall)
        sources.set_opacity(0)
        out_wave = LightWaveSlice(sources)
        out_wave.set_max_amp(1)
        out_wave.set_opacity(0.85)
        out_wave.set_decay_factor(0.5)
        out_wave.set_width(full_width * 2.5)
        out_wave.move_to(ORIGIN, DOWN)

        self.add(sources)
        self.play(
            frame.animate.reorient(1, 49, 0, (-0.02, 3.62, 0.61), 11.96),
            FadeIn(out_wave, time_span=(0, 2), suspend_mobject_updating=False),
            run_time=10
        )
        self.wait(3)

        # Change spacing
        wall = n_slit_wall
        wall.target = self.get_wall_with_slits(16, spacing=2 + 0.2 * PI, total_width=2 * full_width)
        wall.target.move_to(wall)
        in_wave.match_width(out_wave)
        in_wave.move_to(ORIGIN, UP)

        start_arrows, end_arrows = [
            VGroup(
                Tex(R"\\leftrightarrow").set_width(0.7 * block.get_width(), stretch=True).next_to(block, OUT)
                for block in group[1:-1]
            ).rotate(30 * DEGREES, RIGHT).set_backstroke(BLACK, 5)
            for group in [wall, wall.target]
        ]

        self.play(FadeIn(start_arrows))
        self.play(
            Transform(start_arrows, end_arrows),
            MoveToTarget(wall),
            sources.animate.match_points(self.get_point_sources_from_wall(wall.target)),
            # frame.animate.reorient(0, 40, 0, (-0.61, 5.11, 1.62), 24.87),
            run_time=5
        )
        self.play(FadeOut(start_arrows))

        out_wave.set_width(500, about_edge=DOWN)
        self.play(
            frame.animate.reorient(0, 52, 0, (0.91, 35.42, 33.91), 102.49),
            run_time=16
        )

        # Reduce to one slit
        single_slit_wall = self.get_wall_with_slits(1)
        single_slit_wall.move_to(wall)
        source = self.get_point_sources_from_wall(single_slit_wall)
        source.set_radius(0.5)
        radial_wave = LightWaveSlice(source)
        radial_wave.set_width(full_width)
        radial_wave.move_to(ORIGIN, DOWN)

        self.play(
            frame.animate.reorient(11, 65, 0, (-0.03, 0.06, 0.14), 8.66),
            run_time=3
        )
        self.add(single_slit_wall, in_wave)
        self.play(
            FadeOut(wall, scale=0.9),
            FadeOut(out_wave, suspend_mobject_updating=False),
            FadeIn(single_slit_wall),
        )
        self.wait(3)

        self.play(
            frame.animate.reorient(0, 9, 0, (-0.04, 1.61, 0.15), 8.66),
            FadeIn(radial_wave, suspend_mobject_updating=False),
            run_time=3
        )
        self.wait(2)
        single_slit_wall.set_z_index(1)
        self.play(
            FadeIn(source),
            single_slit_wall.animate.set_opacity(0.1),
            in_wave.animate.set_opacity(0.1),
        )
        self.wait(6)

        # Setup for sine waves
        def field_func(points):
            result = np.zeros_like(points)
            result[:, 2] = 0.5 * radial_wave.wave_func(points)
            return result

        # Expose some film
        film_shape = (12, 6)
        film = Rectangle(*film_shape)
        film.set_fill(GREY_E, 1)
        film.set_stroke(WHITE, 1)
        film.rotate(PI / 2, RIGHT)
        film.move_to(source).set_y(5)

        exposure = LightIntensity(source, shape=film_shape)
        exposure.rotate(PI / 2, RIGHT)
        exposure.move_to(film)
        exposure.set_color(GREEN_SCREEN)
        exposure.set_decay_factor(3.5)
        exposure.set_max_amp(0.005)
        exposure.set_opacity(1e-3)

        radial_wave.set_z_index(1)

        single_slit_wall.set_opacity(1)
        single_slit_wall.set_depth(1.5, about_edge=IN)

        self.play(
            FadeOut(source),
            FadeIn(film, shift=5 * IN),
            FadeIn(exposure, shift=5 * IN),
            FadeIn(single_slit_wall),
            in_wave.animate.set_opacity(0.85).set_time_rate(1.0).set_anim_args(suspend_mobject_updating=False),
            radial_wave.animate.set_time_rate(1.0).set_anim_args(suspend_mobject_updating=False),
            frame.animate.reorient(-19, 68, 0, (-2.38, 4.22, 0.53), 10.86),
            run_time=3
        )
        self.play(exposure.animate.set_opacity(1))
        self.wait(3)

        # Wave to various spots
        exposure_glow = GlowDot(color=GREEN_SCREEN)
        exposure_glow.move_to(film.get_center())
        line = Line(stroke_color=TEAL)
        line.f_always.put_start_and_end_on(
            source.get_center, exposure_glow.get_center
        )
        graph = self.get_graph_over_wave(line, radial_wave, scale_factor=0.2)
        graph.set_stroke(WHITE, 2, 1)
        line.set_stroke(opacity=0)

        graph.set_z_index(0)
        line.set_stroke(TEAL, 2, 1)
        self.add(line, graph, radial_wave)
        self.play(
            VFadeIn(graph),
            VFadeIn(line),
            FadeIn(exposure_glow),
            frame.animate.reorient(-46, 65, 0, (-1.12, 3.76, 0.49), 6.83),
            run_time=3
        )
        self.wait(3)
        self.play(exposure_glow.animate.shift(5 * RIGHT).set_opacity(0.5), run_time=3)
        self.play(
            radial_wave.animate.set_decay_factor(1.0).set_max_amp(0.75).set_anim_args(suspend_mobject_updating=False),
            run_time=2,
        )
        self.play(
            frame.animate.reorient(9, 59, 0, (0.54, 4.02, 0.04), 8.32),
            run_time=12,
        )
        self.wait(4)
        source.match_points(self.get_point_sources_from_wall(single_slit_wall))

        # Change to double slit
        two_slit_wall = self.get_wall_with_slits(2, spacing=3.0, depth=single_slit_wall.get_depth())
        two_slit_wall.move_to(single_slit_wall)

        source.match_points(self.get_point_sources_from_wall(two_slit_wall))

        self.remove(single_slit_wall, graph, line, exposure_glow)
        self.add(two_slit_wall)
        self.wait(4)
        self.play(
            frame.animate.reorient(-25, 48, 0, (0.23, 4.15, -0.03), 9.80),
            run_time=8
        )
        self.play(
            frame.animate.reorient(28, 50, 0, (0.23, 4.15, -0.03), 9.80),
            run_time=8
        )

        out_wave = radial_wave  # Just rename

        # Down to two point sources
        source_pair = source
        source1 = source.copy().set_points(source.get_points()[0:1])
        source2 = source.copy().set_points(source.get_points()[1:2])

        self.play(
            FadeOut(two_slit_wall, shift=2 * IN),
            FadeOut(in_wave, suspend_mobject_updating=False),
            FadeIn(source1),
            FadeIn(source2),
        )
        self.play(
            frame.animate.reorient(0, 68, 0, (0.23, 4.15, -0.03), 9.80),
            run_time=4
        )

        # Show each individual wave
        wave1 = out_wave.copy().set_sources(source1).shift(1e-3 * OUT)
        wave2 = out_wave.copy().set_sources(source2).shift(1e-3 * IN)
        exp1 = exposure.copy().set_sources(source1).shift(1e-3 * DOWN)
        exp2 = exposure.copy().set_sources(source2).shift(2e-3 * DOWN)

        self.play(
            FadeOut(source2),
            FadeOut(out_wave, suspend_mobject_updating=False),
            FadeIn(wave1, suspend_mobject_updating=False),
            exposure.animate.set_opacity(0),
            FadeIn(exp1),
        )
        self.wait(3)
        self.add(wave2, wave1)
        self.play(
            FadeOut(source1),
            FadeIn(source2),
            FadeOut(wave1, suspend_mobject_updating=False),
            FadeIn(wave2, suspend_mobject_updating=False),
            FadeOut(exp1),
            FadeIn(exp2),
        )
        self.wait(3)
        self.play(
            FadeIn(source1),
            FadeOut(wave2, suspend_mobject_updating=False),
            FadeIn(out_wave, suspend_mobject_updating=False),
            FadeOut(exp2),
            exposure.animate.set_opacity(1),
        )
        self.wait(3)

        # Focus on center point
        exposure_point = GlowDot(color=GREEN_SCREEN)
        exposure_point.move_to(film.get_center())

        lines = Line().replicate(2)
        lines.set_stroke(TEAL, 2)
        lines[0].f_always.put_start_and_end_on(source1.get_center, exposure_point.get_center)
        lines[1].f_always.put_start_and_end_on(source2.get_center, exposure_point.get_center)

        graphs = VGroup(
            self.get_graph_over_wave(lines[0], wave1),
            self.get_graph_over_wave(lines[1], wave2),
        )

        self.play(
            out_wave.animate.pause().set_opacity(0.5),
            exposure.animate.set_opacity(0),
            frame.animate.reorient(0, 69, 0, (-0.09, 4.1, -0.16), 7.52),
            FadeIn(exposure_point),
            run_time=3,
        )
        wave1.set_uniform(time=out_wave.uniforms["time"])
        wave2.set_uniform(time=out_wave.uniforms["time"])
        self.play(ShowCreation(lines, lag_ratio=0, suspend_mobject_updating=True))
        self.wait()
        self.play(ShowCreation(graphs, lag_ratio=0, run_time=3, suspend_mobject_updating=True))

        # Show combination from a side angle
        self.play(frame.animate.reorient(-80, 83, 0, (-0.09, 4.1, -0.16), 7.52), run_time=3)
        wave1.pause().set_opacity(0)
        wave2.pause().set_opacity(0)
        self.add(wave1, wave2)
        self.play(*(
            wave.animate.unpause().set_anim_args(suspend_mobject_updating=False)
            for wave in [out_wave, wave1, wave2]
        ))
        self.wait(4)
        self.play(
            frame.animate.reorient(0, 66, 0, (0.12, 4.03, -0.38), 7.52),
            *(
                wave.animate.pause().set_anim_args(suspend_mobject_updating=False)
                for wave in [out_wave, wave1, wave2]
            ),
            run_time=3
        )
        self.wait()

        # Shift to a destructive point
        self.play(
            exposure_point.animate.shift(0.9 * RIGHT).set_opacity(0.25),
            run_time=2
        )
        self.wait()
        self.play(
            frame.animate.reorient(-98, 82, 0, (0.27, 4.19, -0.01), 2.93),
            *(
                wave.animate.unpause().set_anim_args(suspend_mobject_updating=False)
                for wave in [out_wave, wave1, wave2]
            ),
            run_time=3,
        )
        self.wait(8)
        self.play(
            frame.animate.reorient(-2, 69, 0, (-0.23, 2.91, 0.1), 6.38),
            exposure.animate.set_opacity(1),
            run_time=4
        )
        self.wait(3)

        # And now over to another constructive point
        self.play(
            exposure_point.animate.shift(0.9 * RIGHT).set_opacity(1),
            run_time=2
        )
        self.wait(2)
        self.play(
            frame.animate.reorient(81, 88, 0, (-0.23, 2.91, 0.1), 6.38),
            run_time=5,
        )
        self.wait(2)
        self.play(
            frame.animate.reorient(0, 67, 0, (-0.07, 1.75, 0.9), 6.42),
            *(
                wave.animate.pause().set_anim_args(suspend_mobject_updating=False)
                for wave in [out_wave, wave1, wave2]
            ),
            run_time=4,
        )
        self.play(
            exposure_point.animate.shift(5 * LEFT),
            run_time=6
        )
        self.play(
            exposure_point.animate.move_to(film.get_center()),
            run_time=5
        )

        # Shorten wave length
        trg_color = Color(hsl=(0.7, 0.7, 0.5))
        self.play(
            *(
                wave.animate.set_wave_number(2).set_anim_args(suspend_mobject_updating=False)
                for wave in [out_wave, wave1, wave2, exposure]
            ),
            UpdateFromAlphaFunc(
                Point(),
                lambda m, a: exposure.set_color(interpolate_color_by_hsl(GREEN_SCREEN, trg_color, a)),
                remover=True
            ),
            exposure_point.animate.set_color(trg_color),
            run_time=6
        )

        new_out_wave = LightWaveSlice(source_pair)
        new_out_wave.replace(out_wave, stretch=True)
        new_out_wave.set_uniforms(dict(out_wave.uniforms))
        new_out_wave.pause()
        self.remove(out_wave)
        self.add(new_out_wave)
        out_wave = new_out_wave

        # Pan over various spots
        for wave in [out_wave, wave1, wave2]:
            wave.set_frequency(2)

        self.play(
            frame.animate.reorient(-80, 68, 0, (-0.22, 2.44, 0.8), 6.42),
            run_time=3
        )
        self.play(
            exposure_point.animate.shift(3 * LEFT),
            rate_func=there_and_back,
            run_time=16,
        )
        self.play(
            FadeOut(lines),
            FadeOut(graphs),
        )
        self.remove(wave1, wave2)

        # Bring back slits, look over it all
        self.play(out_wave.animate.unpause().set_opacity(1).set_anim_args(suspend_mobject_updating=False))
        new_in_wave = self.get_plane_wave()
        new_in_wave.replace(in_wave)
        new_in_wave.set_wave_number(2)
        new_in_wave.set_frequency(2)
        new_in_wave.set_opacity(0.75)

        self.play(
            FadeIn(new_in_wave, time_span=(0, 2), suspend_mobject_updating=False),
            FadeIn(two_slit_wall, time_span=(0, 1)),
            FadeOut(source1, time_span=(0, 2)),
            FadeOut(source2, time_span=(0, 2)),
            frame.animate.reorient(-25, 62, 0, (-0.82, 2.58, 0.5), 9.13),
            run_time=8
        )
        self.play(
            frame.animate.reorient(30, 60, 0, (-0.48, 2.77, 0.52), 9.13),
            rate_func=there_and_back,
            run_time=24
        )


class FullDiffractionGrating(DiffractionGratingScene):
    def construct(self):
        # Set up the grating
        full_width = 100
        wave_number = 2 + 0.1 * PI  # Make it irrational
        frequency = 1
        slit_dist = 1.0

        frame = self.frame
        frame.reorient(-28, 76, 0, (0, 3.29, -0.61), 8.17)

        wall = self.get_wall_with_slits(32, spacing=slit_dist, depth=2.0, total_width=full_width)
        wall.move_to(0.5 * IN, IN)

        in_wave = self.get_plane_wave()
        in_wave.set_wave_number(wave_number)
        in_wave.set_frequency(frequency)
        in_wave.set_shape(full_width, full_width)
        in_wave.set_opacity(0.85)
        in_wave.move_to(ORIGIN, UP)

        sources = self.get_point_sources_from_wall(wall)
        out_wave = LightWaveSlice(sources, wave_number=wave_number, frequency=frequency)
        out_wave.set_shape(full_width, full_width)
        out_wave.set_opacity(0.25)
        out_wave.set_max_amp(2)
        out_wave.move_to(ORIGIN, DOWN)

        self.add(wall)
        self.add(in_wave)
        self.add(out_wave)

        # Label the distance apart
        piece = wall[16]
        brace = Brace(piece, UP)
        brace.rotate(PI / 2, RIGHT)
        brace.next_to(piece, OUT, SMALL_BUFF)
        dist_label = Tex(R"d", font_size=60)
        dist_label.rotate(PI / 2, RIGHT)
        dist_label.next_to(brace, OUT, SMALL_BUFF)
        VGroup(brace, dist_label).set_backstroke(BLACK, 5)

        self.play(
            GrowFromCenter(brace),
            FadeIn(dist_label, 0.25 * OUT),
            frame.animate.reorient(11, 72, 0, (-1.12, 5.34, -0.64), 11.13).set_anim_args(run_time=8),
        )
        dist_label.add(brace)

        # Show model as an array of point sources
        sources.set_radius(0.5)
        wall.save_state()
        wall.target = wall.generate_target()
        wall.target.stretch(0.05, dim=2, about_point=ORIGIN)
        wall.target.stretch(0.5, dim=1, about_point=ORIGIN)

        dist_label.set_z_index(1)
        self.play(
            frame.animate.reorient(0, 12, 0, (0, 4.45, -0.62), 11.20),
            dist_label.animate.rotate(PI / 2, LEFT).next_to(piece, UP, SMALL_BUFF),
            MoveToTarget(wall, time_span=(3, 5)),
            FadeIn(sources, time_span=(1, 3)),
            out_wave.animate.set_opacity(1).set_anim_args(suspend_mobject_updating=False),
            in_wave.animate.set_opacity(0.25).set_anim_args(suspend_mobject_updating=False),
            run_time=8,
        )
        self.wait(4)

        # Show the N graphs
        point_tracker = GlowDot(color=YELLOW, radius=1)
        point_tracker.move_to(8 * UP)

        def update_lines(lines):
            for line, source_point in zip(lines, sources.get_points()):
                line.put_start_and_end_on(source_point, point_tracker.get_center())

        lines = Line().replicate(sources.get_num_points())
        lines.set_stroke(YELLOW, 2)
        lines.add_updater(update_lines)

        individual_sources = Group(
            sources.copy().set_points(sources.get_points()[i:i + 1])
            for i in range(sources.get_num_points())
        )

        waves = Group(
            out_wave.copy().set_sources(src).set_opacity(0)
            for src in individual_sources
        )
        waves.scale(0)

        graphs = VGroup(
            self.get_graph_over_wave(line, wave, scale_factor=0.25)
            for line, wave in zip(lines, waves)
        )

        self.play(
            out_wave.animate.set_opacity(0.2).set_anim_args(suspend_mobject_updating=False),
            FadeOut(dist_label),
            FadeIn(point_tracker),
        )
        self.wait()
        self.add(lines, graphs, out_wave)
        self.add(waves)
        self.play(
            ShowCreation(lines, lag_ratio=0.01, time_span=(0, 2)),
            ShowCreation(graphs, lag_ratio=0.01, suspend_mobject_updating=False, time_span=(0, 2)),
            frame.animate.reorient(1, 57, 0, (-0.17, 5.75, 0.32), 14.54),
            run_time=5
        )
        self.wait(3)
        self.play(
            FadeOut(lines),
            FadeOut(graphs),
            FadeOut(point_tracker),
            out_wave.animate.set_opacity(0.85).set_anim_args(suspend_mobject_updating=False),
            frame.animate.reorient(0, 42, 0, (0, 5.4, 0.45), 13.00),
            run_time=3,
        )
        self.remove(waves)

        # Zoom out to large
        out_wave.set_width(500, about_edge=DOWN)
        self.play(
            frame.animate.reorient(0, 0, 0, (0, 95, 0), 200),
            out_wave.animate.set_max_amp(1).set_anim_args(suspend_mobject_updating=False),
            run_time=20
        )

        # Let it run for a few cycles, we'll use this as an underlay for parts that follow
        self.wait(4)

        # Highlight the higher order beams
        in_wave.scale(0)
        out_wave.scale(0)

        beam_point = GlowDot(color=WHITE, radius=3)
        beam_point.move_to(1000 * UP)
        beam_outlines = Line().replicate(2)
        center_beam_line = Line()
        VGroup(beam_outlines, center_beam_line).set_stroke(WHITE, 50)
        beam_outlines[0].f_always.put_start_and_end_on(sources.get_left, beam_point.get_center)
        beam_outlines[1].f_always.put_start_and_end_on(sources.get_right, beam_point.get_center)
        center_beam_line.f_always.put_start_and_end_on(sources.get_center, beam_point.get_center)

        theta = math.asin(1.0 / wave_number / slit_dist)  # Diffraction equation!

        self.play(ShowCreation(beam_outlines, lag_ratio=0))
        self.wait(3)
        self.play(
            Rotate(beam_point, -theta, about_point=ORIGIN),
            run_time=1
        )
        self.wait()
        self.play(
            Rotate(beam_point, 2 * theta, about_point=ORIGIN),
            run_time=1
        )
        self.wait(8)

        # Ask about the angle
        v_line = Line(ORIGIN, get_norm(beam_point.get_center()) * UP)
        d_line = Line(ORIGIN, beam_point.get_center())
        VGroup(v_line, d_line).set_stroke(WHITE, 50)

        arc = og_big_arc = Arc(PI / 2 + theta, -theta, radius=30)
        arc.set_stroke(WHITE, 50)
        theta_sym = Tex(R"\\theta")
        theta_sym.set_width(arc.get_width() / 2)
        theta_sym.next_to(arc, UP, buff=2).shift(LEFT)

        self.remove(beam_outlines)
        self.play(
            TransformFromCopy(beam_outlines[0], d_line),
            TransformFromCopy(beam_outlines[1], d_line),
        )
        self.play(
            TransformFromCopy(d_line, v_line),
            ShowCreation(arc),
            Write(theta_sym),
        )
        self.wait(3)

        # Analyze central beam
        beam_point.rotate(-theta, about_point=ORIGIN)
        point_tracker.move_to(180 * UP)
        point_tracker.set_radius(8)

        L_line = Line(ORIGIN, point_tracker.get_center())
        x_line = Line(sources.get_center(), sources.get_left())
        hyp = Line(sources.get_left(), point_tracker.get_center())
        VGroup(L_line, hyp).set_stroke(YELLOW, width=50)
        x_line.set_stroke(WHITE, 50)

        L_label = Tex("L", font_size=800)
        x_label = Tex("x", font_size=800)
        hyp_label = Tex(R"\\sqrt{L^2 + x^2}", font_size=800)

        L_label.next_to(L_line.pfp(0.4), RIGHT, buff=2)
        L_label.match_color(L_line)
        x_label.next_to(x_line, UP, buff=3)
        hyp_label.next_to(hyp.pfp(0.4), LEFT, buff=2)
        hyp_label.match_color(hyp)

        self.play(
            FadeIn(beam_outlines),
            FadeOut(d_line),
            FadeOut(v_line),
            FadeOut(arc),
            FadeOut(theta_sym),
        )
        self.wait(2)
        self.play(
            FadeIn(point_tracker),
            out_wave.animate.set_opacity(0.5).set_anim_args(suspend_mobject_updating=False)
        )
        self.play(
            ShowCreation(L_line),
            VFadeIn(L_label),
            FadeOut(beam_outlines),
        )
        self.wait()
        self.play(
            TransformFromCopy(L_line, hyp),
            ShowCreation(x_line),
            TransformMatchingStrings(L_label.copy(), hyp_label),
            FadeIn(x_label, shift=3 * LEFT),
        )

        # Show the approximation (In another scene)
        self.wait(4)

        # Show all the different lines
        self.play(FadeOut(VGroup(L_label, x_label, hyp_label, x_line, L_line, hyp)))
        lines.set_stroke(YELLOW, 30)
        lines.update()

        self.play(LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=0.25, run_time=8))

        # Analyze a point off the center
        new_angle = theta
        lines.set_stroke(width=10)
        arc = Arc(PI / 2 - new_angle, new_angle, radius=30)
        arc.set_stroke(WHITE, 50)
        theta_sym = Tex(R"\\theta")
        theta_sym.set_width(0.45 * arc.get_width())
        theta_sym.next_to(arc.get_center(), UP, buff=2).shift(RIGHT)
        d_line = v_line.copy().rotate(-new_angle, about_edge=DOWN)
        question = Text("What about\\nover here?")
        question.set_height(15)
        question.rotate(frame.get_phi(), RIGHT)
        question.always.next_to(point_tracker, DR, buff=-2)

        self.play(
            Rotate(point_tracker, -new_angle, about_point=ORIGIN),
            VFadeIn(question),
            run_time=3
        )
        self.play(ShowCreation(d_line))
        self.play(
            TransformFromCopy(d_line, v_line),
            ShowCreation(arc),
            Write(theta_sym)
        )
        self.wait(2)
        self.play(
            ShowCreation(lines, lag_ratio=0.01, run_time=2, suspend_mobject_updating=True),
            FadeOut(VGroup(v_line, d_line, arc, theta_sym))
        )
        self.wait(4)
        self.play(
            point_tracker.animate.move_to(1.15 * point_tracker.get_center()),
            run_time=2
        )

        # Zoom in near the slits again
        self.play(
            frame.animate.reorient(0, 0, 0, (0.0, 2.75, 0.0), 8),
            lines.animate.set_stroke(width=5),
            out_wave.animate.set_opacity(0.1).set_anim_args(suspend_mobject_updating=False),
            in_wave.animate.set_opacity(0.1).set_anim_args(suspend_mobject_updating=False),
            sources.animate.set_radius(0.35),
            run_time=6,
        )
        self.wait(4)

        # Show individual lines
        lines.suspend_updating()
        line1 = lines[15].copy()
        line2 = lines[16].copy()
        line2.set_stroke(WHITE)
        for line in [line1, line2]:
            line.set_length(8, about_point=line.get_start())
            line.set_stroke(opacity=1)
            line.save_state()

        lines.target = lines.generate_target()
        lines.target.set_stroke(width=1, opacity=0.5)

        long_label = Text("Is this longer...")
        short_label = Text("...than this?")
        long_label.match_color(line1)
        long_label.next_to(line1.get_center(), LEFT)
        short_label.next_to(line2.get_center(), RIGHT)

        self.play(
            MoveToTarget(lines),
            ShowCreation(line1, time_span=(0.5, 1.5)),
            FadeIn(long_label, time_span=(0.5, 1.5))
        )
        self.wait()
        self.play(
            ShowCreation(line2),
            TransformMatchingStrings(long_label.copy(), short_label),
        )
        self.wait()

        # Zoom out and pivot
        for line in [line1, line2]:
            line.put_start_and_end_on(line.get_start(), point_tracker.get_center())

        tail = TracingTail(line.get_start, time_traced=3.0, stroke_width=(0, 10))
        point_label = TexText(R"Point we're\\\\analyzing")
        point_label.set_height(1.5 * question.get_height())
        point_label.move_to(question, UL)
        self.remove(question)
        self.add(point_label)

        self.play(
            frame.animate.reorient(0, 0, 0, (14.74, 92.42, 0.0), 209.77),
            line2.animate.set_stroke(width=50),
            run_time=3
        )
        self.add(tail)
        self.wait(2)
        self.play(
            Rotate(
                line2, -30 * DEGREES,
                about_point=point_tracker.get_center(),
                rate_func=lambda t: wiggle(t, 2),
                run_time=8,
            )
        )
        self.wait(2)
        self.play(
            frame.animate.reorient(0, 0, 0, (0, 1.5, 0.0), 6.0),
            line2.animate.set_stroke(width=5),
            VFadeOut(tail),
            FadeOut(point_label),
            FadeOut(in_wave, suspend_mobject_updating=False),
            FadeOut(out_wave, suspend_mobject_updating=False),
            run_time=3,
        )

        # Rotate again, as a perp
        tail.add_updater(lambda m: m.set_stroke(width=(0, 5)))
        self.add(tail)
        self.play(
            Rotate(
                line2, -1 * DEGREES,
                about_point=point_tracker.get_center(),
                # rate_func=lambda t: wiggle(t, 2),
                rate_func=there_and_back,
                run_time=5,
            )
        )
        self.wait(3)
        self.remove(tail)

        # Drop perpendicular
        p1 = line1.get_start()
        p2 = line2.get_start()
        to_point = rotate_vector(UP, -theta)
        foot = p1 + math.sin(theta) * to_point

        diff_label_group = always_redraw(lambda: self.get_diff_label_group(
            p1=sources.get_points()[15],
            p2=sources.get_points()[16],
            theta=PI / 2 - line1.get_angle()
        ))
        diff_label_group.suspend_updating()
        triangle, elbow, altitude, arc, small_theta_sym, diff_segment, brace, d_label = diff_label_group

        self.play(
            ShowCreation(altitude),
            FadeOut(VGroup(long_label, short_label)),
            frame.animate.reorient(0, 0, 0, (0.5, 0.88, 0.0), 3.8),
            sources.animate.set_radius(0.2),
        )
        self.play(ShowCreation(elbow))
        self.wait()

        # Compare lengths
        for line in line1, line2:
            line.set_length(5, about_point=line.get_start())
        matched_segment = line2.copy().shift(altitude.get_vector())
        matched_segment.set_color(TEAL)
        label1 = TexText(R"Length of shorter line $\\rightarrow$", font_size=24)
        label1.next_to(p2, UR, buff=SMALL_BUFF)
        label1.rotate(PI / 2 - theta, about_point=p2)
        label1.shift(0.1 * to_point)
        label2 = label1.copy()
        label2.match_color(matched_segment)
        label2.shift(matched_segment.get_start() - line2.get_start())

        diff_label = Text("Difference", font_size=24)
        diff_label.next_to(brace.get_center(), LEFT, buff=0.2).shift(0.15 * UP)
        diff_label.set_color(RED)
        diff_segment.set_stroke(RED, 5)

        self.play(
            Write(label1, stroke_width=1),
            ShowCreation(line2),
            run_time=1,
        )
        self.wait()
        self.play(
            TransformFromCopy(line2, matched_segment),
            line1.animate.set_stroke(width=1)
        )
        self.wait()
        self.play(LaggedStart(
            GrowFromCenter(brace),
            GrowFromCenter(diff_segment),
            Write(diff_label, stroke_width=1),
            lag_ratio=0.2
        ))
        self.wait()

        # Draw the appropriate right triangle
        d_sine_theta = Tex(R"d \\cdot \\sin(\\theta)", font_size=24)
        d_sine_theta.move_to(diff_label, RIGHT)

        self.add(triangle, elbow, altitude, diff_segment)
        self.play(
            wall.animate.set_height(0.01, stretch=True),
            FadeIn(triangle),
            FadeOut(label1),
        )
        self.wait()
        self.play(FadeIn(d_label, 0.25 * DOWN))
        self.wait()
        self.play(
            TransformMatchingStrings(d_label.copy(), d_sine_theta, run_time=1),
            FadeOut(diff_label),
        )
        self.wait()
        self.play(
            TransformFromCopy(d_sine_theta[R"\\theta"][0], small_theta_sym),
            ShowCreation(arc),
        )
        self.wait()

        # Lock the leg to match wavelength
        self.remove(in_wave, out_wave)
        self.checkpoint("d*sin(theta)")

        lambda_label = Tex(R"= \\lambda")
        lambda_label[1].set_color(TEAL)
        lambda_label.set_height(0.75 * d_sine_theta.get_height())
        lambda_label.add_updater(lambda m: m.next_to(brace.pfp(0.5), UL, buff=0.025))

        n_cycles = 8
        sine = FunctionGraph(lambda x: -math.sin(x), x_range=(0, n_cycles * TAU, 0.1))
        sine.set_stroke(TEAL, 1)
        sine.set_width(n_cycles * diff_segment.get_length())
        sine.add_updater(lambda m: m.put_start_and_end_on(
            diff_segment.get_start(), diff_segment.get_end()
        ).scale(n_cycles, about_point=diff_segment.get_start()))

        lock_arrow = Vector(0.5 * DOWN, thickness=2).next_to(brace, UP, buff=0.05)
        lock_label = Text("Consider this\\nlocked", font_size=16)
        lock_label.next_to(lock_arrow, UP, SMALL_BUFF)

        self.play(
            d_sine_theta.animate.scale(0.75).next_to(lambda_label, LEFT, buff=0.05).shift(0.025 * DOWN),
            FadeIn(lambda_label, 0.25 * RIGHT),
            FadeOut(line2),
            FadeOut(matched_segment),
            diff_segment.animate.set_stroke(width=2),
            frame.animate.reorient(0, 0, 0, (0.14, 0.48, 0.0), 3.17).set_anim_args(run_time=2),
        )
        self.play(ShowCreation(sine, rate_func=linear))
        self.wait()
        self.play(
            FadeIn(lock_label),
            GrowArrow(lock_arrow)
        )
        self.wait()
        self.play(FadeOut(VGroup(lock_arrow, lock_label)))

        # Show the other sine waves
        shift_value = p2 - p1
        other_sines = VGroup(sine.copy().shift(x * shift_value) for x in range(-2, 4) if x != 0)
        other_sines.clear_updaters()

        self.play(ShowCreation(other_sines, lag_ratio=0.25, run_time=4))
        self.wait()
        self.play(FadeOut(other_sines, lag_ratio=0.25, run_time=2))

        # Change the distance between points
        self.add(diff_label_group)
        diff_label_group.resume_updating()

        lines.resume_updating()
        line1.clear_updaters()
        line1.add_updater(lambda m: m.match_points(lines[15]))
        line1.resume_updating()

        def get_dist_point(wavelength):
            dist_to_point = get_norm(point_tracker.get_center())
            d = get_norm(sources.get_points()[1] - sources.get_points()[0])
            angle = math.asin(wavelength / d)
            return rotate_vector(UP, -angle) * dist_to_point

        self.add(lines, line1, diff_label_group, sine)
        wall_center = sources.get_points()[15]
        scale_factors = [0.5, 2.0, 1.5, 1.0 / 1.5][:-1]
        for scale_factor in scale_factors:
            arrows = VGroup(Vector(0.3 * RIGHT, thickness=1), Vector(0.3 * LEFT, thickness=1))
            arrows.arrange(RIGHT if scale_factor < 1 else LEFT, buff=0.25)
            arrows.always.move_to(d_label)
            self.play(
                UpdateFromFunc(point_tracker, lambda m: m.move_to(get_dist_point(1.0 / wave_number))),
                MaintainPositionRelativeTo(d_sine_theta, lambda_label),
                sources.animate.scale(scale_factor, about_point=wall_center),
                wall.animate.scale(scale_factor, about_point=wall_center),
                FadeIn(arrows, scale=scale_factor, suspend_mobject_updating=False, time_span=(0, 2)),
                run_time=5,
            )
            self.play(FadeOut(arrows))

        # Show double
        if False:  # This was just a temporary insert, not to be run in general
            d_angle2, d_angle3 = [
                angle_of_vector(get_dist_point(n / wave_number)) - angle_of_vector(get_dist_point((n + 1) / wave_number))
                for n in (1, 2)
            ]
            sine.clear_updaters()
            new_rhs = Tex(Rf"= 1.00 \\lambda", t2c={R"\\lambda": TEAL})
            new_rhs.set_height(0.8 * lambda_label.get_height())
            factor = new_rhs.make_number_changeable("1.00")
            factor_tracker = ValueTracker(1.0)
            new_rhs.f_always.set_value(factor_tracker.get_value)
            new_rhs.always.move_to(lambda_label, RIGHT)

            self.play(
                d_sine_theta.animate.next_to(new_rhs, LEFT, buff=0.05).shift(0.02 * DOWN),
                lambda_label.animate.set_opacity(0),
                FadeIn(new_rhs)
            )
            self.play(
                Rotate(point_tracker, -d_angle2, about_point=ORIGIN),
                Rotate(sine, -d_angle2, about_point=sine.get_start()),
                factor_tracker.animate.set_value(2.0),
                MaintainPositionRelativeTo(d_sine_theta, lambda_label),
                run_time=5,
            )
            self.wait()
            self.play(
                Rotate(point_tracker, -d_angle2, about_point=ORIGIN),
                Rotate(sine, -d_angle2, about_point=sine.get_start()),
                factor_tracker.animate.set_value(3.0),
                MaintainPositionRelativeTo(d_sine_theta, lambda_label),
                run_time=5,
            )
            self.wait()

        # Show the angle match
        self.revert_to_checkpoint("d*sin(theta)")

        p4 = p2 + 2 * (p2 - p1)
        h_line = Line(p2, p1).scale(3, about_edge=RIGHT)
        h_line.set_stroke(WHITE, 0)
        angle_group = VGroup(triangle, arc, small_theta_sym, h_line).copy()
        angle_group[0].set_opacity(0)
        angle_group.target = angle_group.generate_target()
        angle_group.target.rotate(-PI / 2)
        angle_group.target.move_to(p4, DL)
        angle_group.target[2].rotate(PI / 2).shift(0.01 * UR)
        angle_group.target[3].set_stroke(WHITE, 3)
        angle_group.target.scale(2, about_edge=DL)

        self.play(
            MoveToTarget(angle_group),
            run_time=2
        )
        self.wait()

        # Write conclusion
        conclusion = VGroup(
            Text("Difference in distance:", font_size=36),
            Tex(R"d \\cdot \\sin(\\theta)")
        )
        conclusion.arrange(DOWN)
        conclusion_box = SurroundingRectangle(conclusion, buff=MED_SMALL_BUFF)
        conclusion_box.set_stroke(WHITE, 1)
        conclusion_box.set_fill(BLACK, 1)
        conclusion_group = VGroup(conclusion_box, conclusion)
        conclusion_group.to_corner(UL, buff=SMALL_BUFF)
        conclusion_group.fix_in_frame()
        conclusion_group.set_fill(border_width=0)

        self.play(
            FadeIn(conclusion_box),
            FadeIn(conclusion[0]),
            TransformFromCopy(d_sine_theta, conclusion[1])
        )
        self.wait()
        self.play(FadeOut(conclusion_group))

        # Zoom out
        out_wave.set_sources(sources.copy().set_points(sources.get_points()[10:-10]))
        out_wave.set_sources(sources)
        out_wave.set_width(800).move_to(ORIGIN, DOWN)
        out_wave.set_opacity(0)
        lines.resume_updating()

        self.add(out_wave)
        self.play(
            FadeOut(diff_label_group, lag_ratio=0.1, time_span=(0, 1.5)),
            FadeOut(VGroup(d_sine_theta, matched_segment, line1, line2, angle_group), lag_ratio=0.1, time_span=(0, 1.5)),
            out_wave.animate.set_opacity(0.85).set_anim_args(suspend_mobject_updating=False, time_span=(0, 2)),
            frame.animate.reorient(0, 0, 0, (0, 200, 0.0), 400),
            sources.animate.set_radius(0.75),
            lines.animate.set_stroke(width=20).set_anim_args(suspend_mobject_updating=False),
            point_tracker.animate.scale(2.0, about_point=ORIGIN).set_anim_args(time_span=(0, 8)),
            run_time=20,
            rate_func=lambda t: t**6,
        )
        self.wait()

        # Add back arc label
        arc = always_redraw(lambda: Arc(
            PI / 2, angle_of_vector(point_tracker.get_center()) - PI / 2,
            radius=50,
            stroke_color=WHITE,
            stroke_width=100
        ))
        theta_sym.set_height(16)
        theta_sym.add_updater(lambda m, arc=arc: m.next_to(arc.pfp(0.7), UP, buff=6))
        theta_sym.suspend_updating()
        VGroup(v_line, d_line).set_stroke(WHITE, 150)
        d_line.add_updater(lambda m, pt=point_tracker: m.put_start_and_end_on(ORIGIN, 5 * pt.get_center()))
        self.play(
            ShowCreation(v_line),
            ShowCreation(d_line),
            ShowCreation(arc),
            Write(theta_sym, stroke_width=20),
            run_time=1
        )
        theta_sym.resume_updating()
        self.wait(4)

        # Change the slit distance zoomed out
        for scale_factor in scale_factors:
            self.play(
                UpdateFromFunc(point_tracker, lambda m: m.move_to(get_dist_point(1.0 / wave_number))),
                sources.animate.scale(scale_factor, about_point=wall_center),
                wall.animate.scale(scale_factor, about_point=wall_center),
                run_time=5,
            )
            self.wait()

        # Double and triple the angle
        for n in [1, 2]:
            d_angle = angle_of_vector(get_dist_point(n / wave_number)) - angle_of_vector(get_dist_point((n + 1) / wave_number))
            self.play(
                Rotate(point_tracker, -d_angle, about_point=ORIGIN),
                frame.animate.set_height(450),
                run_time=5
            )
            self.wait()

    def get_diff_label_group(self, p1, p2, theta):
        # Altitude
        to_point = rotate_vector(UP, -theta)
        dist = get_norm(p2 - p1)
        foot = p1 + dist * math.sin(theta) * to_point

        altitude = DashedLine(p2, foot, dash_length=get_norm(foot - p2) / 39.5)
        elbow = Elbow(width=0.1 * dist, angle=-theta - PI / 2).shift(foot)

        altitude.set_stroke(WHITE, 2)
        elbow.set_stroke(WHITE, 2)

        # Triangle
        triangle = Polygon(p1, foot, p2)
        triangle.set_stroke(width=0)
        triangle.set_fill(YELLOW, 0.5)
        d_label = Tex(R"d", font_size=24)
        d_label.next_to(triangle, DOWN, SMALL_BUFF)

        # Leg
        diff_segment = Line(p1, foot)
        diff_segment.set_stroke(RED, 2)
        brace = VMobject().set_points_as_corners([LEFT, UL, UR, RIGHT])
        brace.set_shape(diff_segment.get_length(), 0.1)
        brace.set_stroke(WHITE, 1)
        # brace = Brace(Line(ORIGIN, 0.75 * RIGHT), UP)
        # brace.set_shape(diff_segment.get_length(), 0.15)
        brace.rotate(PI / 2 - theta)
        brace.move_to(diff_segment).shift(0.1 * rotate_vector(to_point, PI / 2))

        # Angle label
        arc_rad = min(0.35 * dist, 0.35 * get_norm(foot - p2))
        arc = Arc(PI, -theta, radius=arc_rad).shift(p2)
        arc.set_stroke(WHITE, 2)
        small_theta_sym = Tex(R"\\theta")
        small_theta_sym.set_height(0.8 * arc.get_height())
        small_theta_sym.next_to(arc.pfp(0.5), LEFT, buff=0.05)

        return VGroup(triangle, elbow, altitude, arc, small_theta_sym, diff_segment, brace, d_label)

    def old(self):
        # Old
        dist_label = DecimalNumber(num_decimal_places=1)
        dist_label.set_height(4)
        dist_label.add_updater(lambda m: m.next_to(L_line.get_center(), RIGHT, buff=2))
        dist_label.add_updater(lambda m: m.set_value(L_line.get_length()))


class PlaneWaveThroughZonePlate(DiffractionGratingScene):
    def construct(self):
        # Set up the zone plate and object
        frame = self.frame
        wave_number = 4
        frequency = 2.0

        obj_dot = Group(GlowDot(), TrueDot())
        obj_dot.move_to(4 * RIGHT)
        obj_dot.set_color(WHITE)

        zone_sources = DotCloud([obj_dot.get_center(), 1002 * RIGHT])
        plate = LightIntensity(zone_sources)
        plate.set_shape(9, 16)
        plate.rotate(PI / 2, UP)
        plate.set_height(8)
        plate.set_color(WHITE, 0.7)
        plate.set_wave_number(24)
        plate.set_decay_factor(0)
        plate_top = plate.copy()
        plate_top.rotate(PI / 2, DOWN)
        plate_top.set_width(0.075, stretch=True)
        plate_top.move_to(plate)

        ref_wave = self.get_plane_wave(LEFT)
        ref_wave.set_shape(10, plate.get_height())
        ref_wave.set_frequency(frequency)
        ref_wave.set_color(BLUE_C, 0.5)
        ref_wave.set_wave_number(wave_number)
        ref_wave.move_to(plate, LEFT)

        frame.reorient(19, 77, 0, ORIGIN, 8.00)
        self.add(plate)
        self.add(obj_dot)

        # Add Number plane
        plane = NumberPlane(x_range=(-10, 10, 1), y_range=(-8, 8, 1.0))
        plane.become(NumberPlane(x_range=(-10, 10, 1), y_range=(-8, 8, 1.0)))
        plane.fade(0.5)
        plane.apply_depth_test()
        self.add(plate, plane)

        self.play(
            frame.animate.reorient(58, 73, 0, ORIGIN, 8.00),
            Write(plane, stroke_width=3, lag_ratio=0.01, time_span=(2, 6)),
            run_time=6
        )

        # Draw a line
        film_point = 2 * UP
        line = Line(film_point, obj_dot.get_center())
        line.set_stroke(TEAL, 3)

        self.play(
            ShowCreation(line),
            frame.animate.reorient(0, 0, 0),
            FadeIn(plate_top),
            run_time=3,
        )
        self.wait()

        # Where the object had been
        dash_circle = DashedVMobject(Arc(angle=(23 / 24) * TAU), num_dashes=12)
        dash_circle.set_stroke(YELLOW, 3)
        dash_circle.replace(obj_dot).set_width(0.2)
        for part in dash_circle:
            dash_circle.set_joint_type("no_joint")
        had_been_words = Text("Where the object\\nhad been", font_size=36)
        had_been_words.next_to(dash_circle, UP, buff=0, aligned_edge=LEFT)

        self.play(
            FadeOut(obj_dot),
            Write(dash_circle, stroke_width=3, run_time=1),
            Write(had_been_words, run_time=1),
        )
        self.wait()
        self.play(FadeOut(had_been_words))

        # Show angle
        theta = -line.get_angle()
        arc = Arc(PI - theta, theta, radius=1)
        arc.shift(obj_dot.get_center())
        h_line = Line(ORIGIN, obj_dot.get_center())
        h_line.set_stroke(WHITE, 2)
        theta_prime_sym = Tex(R"\\theta'")
        theta_prime_sym.set_max_height(0.8 * arc.get_height())
        theta_prime_sym.next_to(arc.pfp(0.4), LEFT, SMALL_BUFF)

        self.play(
            TransformFromCopy(line, h_line),
            ShowCreation(arc),
            Write(theta_prime_sym),
        )
        self.wait()

        # Set up terms for the calculations for the spacing
        # TODO, consider adding many little lines for all the fringes
        self.remove(plate)
        v_line = Line(ORIGIN, film_point)

        d_lines = Line(LEFT, RIGHT).replicate(2).set_width(0.3)
        d_lines.set_stroke(WHITE, 2)
        d_lines.arrange(DOWN, buff=0.1)
        d_lines.move_to(film_point, DOWN)
        lil_brace = Brace(Line(ORIGIN, 0.25 * UP), LEFT)
        lil_brace.match_height(d_lines)
        lil_brace.next_to(d_lines, LEFT, buff=0.05)
        big_brace = Brace(Group(d_lines[1], Point(ORIGIN)), LEFT, buff=0)
        big_brace.match_width(lil_brace, about_edge=RIGHT, stretch=True)

        kw = dict(font_size=42)
        L_label = Tex("L", **kw).next_to(h_line, DOWN, 2 * SMALL_BUFF)
        x_label = Tex("x", **kw).next_to(big_brace, LEFT, SMALL_BUFF)
        d_label = Tex("d", **kw).next_to(lil_brace, LEFT, SMALL_BUFF, aligned_edge=DOWN)
        L_label.set_color(BLUE)
        x_label.set_color(RED)
        VGroup(L_label, x_label, d_label).set_backstroke(BLACK, 5)

        terms = VGroup(
            d_lines, lil_brace, big_brace,
            L_label, x_label, d_label
        )

        # Limit to reference beam at just one point
        equations_tex = [
            R"\\lambda = \\sqrt{L^2 + (x + d)^2} - \\sqrt{L^2 + x^2}",
            R"= \\sqrt{L^2 + x^2 + 2xd + d^2} - \\sqrt{L^2 + x^2}",
            R"\\approx \\sqrt{L^2 + x^2 + 2xd} - \\sqrt{L^2 + x^2}",
            R"\\approx \\frac{1}{2\\sqrt{L^2 + x^2}} 2xd",
            R"= d \\cdot \\frac{x}{\\sqrt{L^2 + x^2}}",
            R"= d \\cdot \\sin(\\theta')",
        ]
        equations = VGroup(
            Tex(eq, t2c={R"\\lambda": YELLOW, "L": BLUE, "x": RED}, font_size=36)
            for eq in equations_tex
        )
        equations.arrange(DOWN, buff=0.65, aligned_edge=LEFT)
        equations.move_to(9.5 * LEFT + 5.65 * UP, UL)
        equations.set_backstroke(BLACK, 10)

        annotations = VGroup(
            Text("The distances between adjacent fringes and\\nthe object should differ by one wavelength"),
            TexText(R"$d^2$ is small compared to $xd$"),
            TexText(R"Linear approximation:\\\\ \\quad \\\\$\\sqrt{X + \\epsilon} \\approx \\sqrt{X} + \\frac{1}{2\\sqrt{X}} \\epsilon$"),
        )
        annotations.scale(0.75)
        for annotation, i in zip(annotations, [0, 1, 3]):
            eq = equations[i]
            annotation.next_to(eq, RIGHT, buff=1.5)
            if i == 2:
                annotation.next_to(eq, DR)
            arrow = Arrow(annotation.get_left(), eq.get_right())
            annotation.add(arrow)
            annotation.set_color(GREY_A)
        annotations[2][:-1].align_to(annotations[2][-1], UP)

        braces = VGroup(
            Brace(equations[0][R"\\sqrt{L^2 + (x + d)^2}"], UP, SMALL_BUFF),
            Brace(equations[0][R"\\sqrt{L^2 + x^2}"], UP, SMALL_BUFF),
        )
        brace_texts = VGroup(
            TexText(R"Dist. to fringe\\\\at height $(x + d)$", font_size=24).next_to(braces[0], UP, SMALL_BUFF),
            TexText(R"Dist. to fringe\\\\at height $x$", font_size=24).next_to(braces[1], UP, SMALL_BUFF),
        )

        self.play(
            FadeIn(terms, lag_ratio=0.1, time_span=(0, 2)),
            Write(equations, time_span=(2, 5)),
            frame.animate.reorient(0, 0, 0, (-2, 2.5, 0.0), 9).set_anim_args(run_time=3),
        )
        self.wait()
        self.play(LaggedStart(
            FadeIn(annotations[0]),
            FadeIn(braces),
            FadeIn(brace_texts),
        ))
        self.wait()
        self.play(FadeIn(annotations[1]))
        self.play(FadeIn(annotations[2]))
        self.wait()

        # Reduce down to the key conclusion
        key_equation = Tex(R"d \\cdot \\sin(\\theta') = \\lambda", **kw)
        key_equation.next_to(line, UP, MED_LARGE_BUFF)
        key_equation.scale(1.25)
        key_equation.shift(RIGHT + 0.5 * UP)

        box = SurroundingRectangle(key_equation, buff=MED_SMALL_BUFF)
        box.set_fill(BLACK, 1)
        box.set_stroke(YELLOW, 1)

        terms.remove(d_label, lil_brace, d_lines)

        self.add(d_label, lil_brace, d_lines)
        self.play(
            ReplacementTransform(equations[-1][0], key_equation[9], time_span=(0, 2)),
            ReplacementTransform(equations[-1][1:], key_equation[:9], time_span=(0, 2)),
            ReplacementTransform(equations[0][0], key_equation[10], time_span=(0, 2)),
            FadeOut(equations[0][1:], time_span=(1.0, 1.5)),
            FadeOut(equations[1:-1], lag_ratio=0.01, time_span=(1.0, 2.5)),
            FadeOut(annotations, lag_ratio=0.01),
            FadeOut(terms, lag_ratio=0.1, time_span=(1.0, 3.0)),
            FadeOut(h_line),
            FadeOut(braces),
            FadeOut(brace_texts),
            frame.animate.reorient(0, 0, 0, (0, 1, 0), 6).set_anim_args(time_span=(1, 3.5)),
        )
        self.add(box, key_equation)
        self.play(
            Write(box),
            FlashAround(key_equation, buff=MED_SMALL_BUFF, time_width=1.5, run_time=2),
        )
        self.wait()

        # Smaller slit width
        lil_brace.generate_target()
        lil_brace.target.flip().next_to(d_lines, RIGHT, buff=0.025)

        arrow = Vector(0.3 * DL, thickness=2)
        arrow.next_to(lil_brace.target, UR, buff=0)

        new_plate_top = plate_top.copy()
        new_plate_top.set_wave_number(50)
        new_plate_top.save_state()
        new_plate_top.stretch(0, 0)

        self.play(
            d_label.animate.next_to(arrow.get_start(), UR, 0.5 * SMALL_BUFF),
            GrowArrow(arrow),
            MoveToTarget(lil_brace)
        )
        self.play(
            d_lines.animate.stretch(0.25, 1, about_edge=DOWN),
            lil_brace.animate.scale(0.25, about_edge=DL).set_stroke(WHITE, 1),
            arrow.animate.put_start_and_end_on(arrow.get_start(), arrow.get_end() + 0.05 * LEFT + 0.05 * DOWN),
            plate_top.animate.stretch(0, 0),
            Restore(new_plate_top),
            run_time=2
        )
        self.wait()
        self.remove(new_plate_top)
        plate_top.become(new_plate_top)

        # Shine reference beam in
        ref_wave = self.get_beam()
        ref_wave.move_to(film_point, LEFT)

        out_beams = self.get_triple_beam(film_point, obj_dot.get_center())

        self.play(GrowFromPoint(ref_wave, film_point + 8 * RIGHT, run_time=2, rate_func=linear))
        self.play(*(
            GrowFromPoint(beam[1], film_point, run_time=2, rate_func=linear)
            for beam in out_beams
        ))
        self.wait(4)

        # Note the matching angle
        upper_arc = arc.copy()
        upper_arc.shift(film_point - obj_dot.get_center())
        theta_sym = Tex(R"\\theta", font_size=42)
        theta_sym.next_to(upper_arc.pfp(0.4), LEFT, SMALL_BUFF)

        self.play(
            ShowCreation(upper_arc),
            Write(theta_sym),
        )
        self.wait(4)

        # Write the diffraction equation
        key_equation.set_backstroke(BLACK, 8)
        key_equation.generate_target()
        box.generate_target()
        diff_eq = Tex(R"d \\cdot \\sin(\\theta) = \\lambda")
        key_equation.target.match_height(diff_eq)
        key_equation.target.next_to([6.5, 5.5, 0], DL, SMALL_BUFF)
        diff_eq.next_to(key_equation.target, DOWN, MED_LARGE_BUFF)

        box.target.surround(VGroup(key_equation.target, diff_eq))
        box.target.set_opacity(0)

        diff_eq_label = VGroup(
            Text("Diffraction\\nequation", font_size=36),
            Vector(RIGHT),
        )
        diff_eq_label.arrange(RIGHT)
        diff_eq_label.next_to(diff_eq, LEFT)

        VGroup(diff_eq, diff_eq_label).set_backstroke(BLACK, 8)

        theta_sym_copy = theta_sym.copy()
        theta_sym_copy.set_backstroke()

        self.play(
            frame.animate.reorient(0, 0, 0, (0.0, 2, 0.0), 8.00),
            MoveToTarget(box),
            MoveToTarget(key_equation),
            Transform(theta_sym_copy, diff_eq[R"\\theta"][0]),
            run_time=2
        )
        self.play(
            Write(diff_eq),
            FadeIn(diff_eq_label[0], lag_ratio=0.1),
            GrowArrow(diff_eq_label[1]),
        )
        self.remove(theta_sym_copy)
        self.wait(2)

        # Write implication
        implication = VGroup(Tex(R"\\Downarrow"), Tex(R"\\theta = \\theta'"))
        implication.arrange(DOWN)
        implication.next_to(diff_eq, DOWN)
        implication.set_backstroke(width=5)

        self.play(Write(implication))
        self.wait()
        self.play(
            Transform(theta_sym.copy(), theta_prime_sym, remover=True),
            Transform(upper_arc.copy(), arc, remover=True),
            run_time=2
        )
        self.wait(4)

        # Move film point around
        film_dot = Point(film_point)

        line.f_always.put_start_and_end_on(film_dot.get_center, obj_dot.get_center)

        ref_wave.always.match_y(film_dot)

        def update_out_beams(beams):
            beams.become(self.get_triple_beam(
                film_dot.get_center(),
                obj_dot.get_center(),
            ))
            for beam in beams:
                beam[1].set_uniform(time=self.time)

        out_beams.clear_updaters()
        out_beams.add_updater(update_out_beams)

        self.add(out_beams)
        self.play(film_dot.animate.move_to(film_point))

        arc.add_updater(lambda m: m.become(
            Arc(PI, line.get_angle()).shift(obj_dot.get_center())
        ))
        upper_arc.add_updater(lambda m: m.match_points(arc).shift(
            film_dot.get_center() - obj_dot.get_center()
        ))
        theta_prime_sym.add_updater(
            lambda m: m.set_height(min(0.8 * arc.get_height(), 0.35)).next_to(arc.pfp(0.6), LEFT, SMALL_BUFF)
        )
        theta_sym.add_updater(lambda m: m.replace(theta_prime_sym[0]).shift(
            film_dot.get_center() - obj_dot.get_center()
        ))

        d_group = VGroup(d_label, arrow, d_lines, lil_brace)

        self.play(
            FadeOut(d_group, time_span=(0, 1)),
            film_dot.animate.set_y(1),
            run_time=3
        )
        self.play(film_dot.animate.set_y(3.5), run_time=5)
        self.play(film_dot.animate.set_y(0.5), run_time=6)
        self.play(film_dot.animate.set_y(3.5), run_time=6)

        # Show zone plate and observer
        equaiton_group = VGroup(box, key_equation, diff_eq, diff_eq_label, implication)

        randy = Randolph(height=2)
        randy.move_to(4 * LEFT, DOWN)

        plate.set_opacity(0.5)
        plate.set_wave_number(plate_top.uniforms["wave_number"])

        self.add(plate, plane)
        self.play(
            FadeOut(equaiton_group, lag_ratio=0.1, time_span=(0, 2)),
            film_dot.animate.set_y(1.0),
            FadeOut(plate_top, time_span=(0, 1)),
            FadeIn(randy, time_span=(1, 3)),
            frame.animate.reorient(-33, 43, 0, (-2.43, 0.5, -0.12), 9.60),
            run_time=5
        )
        self.play(randy.change("pondering", obj_dot))
        self.play(Blink(randy))
        self.wait(3)

        # Show full reference wave
        big_ref_wave = self.get_3d_ref_wave(plate)

        self.add(big_ref_wave, plate)
        self.play(
            FadeIn(big_ref_wave),
            FadeOut(ref_wave),
            frame.animate.reorient(-40, 67, 0, (-2.43, 0.5, -0.12), 9.60).set_anim_args(run_time=3)
        )
        self.wait(2)

        # Show many beams off the plate
        mid_line_points = DotCloud().to_grid(25, 1)
        mid_line_points.replace(plate, dim_to_match=1)
        mid_line_points.rotate(PI)
        plate_points = DotCloud().to_grid(15, 11)
        dense_plate_points = DotCloud().to_grid(60, 40)
        for dot_cloud in [plate_points, dense_plate_points]:
            dot_cloud.rotate(PI / 2, UP)
            dot_cloud.replace(plate, stretch=True)

        mid_lines_out = self.get_radiating_lines(mid_line_points, obj_dot)
        lines_out = self.get_radiating_lines(plate_points, obj_dot)
        dense_lines_out = self.get_radiating_lines(dense_plate_points, obj_dot)
        ghost_lines = self.get_ghost_lines(mid_line_points, obj_dot)
        dense_ghost_lines = self.get_ghost_lines(dense_plate_points, obj_dot)

        out_beams.clear_updaters()
        self.play(
            FadeOut(out_beams),
            FadeOut(VGroup(theta_sym, theta_prime_sym, upper_arc, arc, line)),
            ShowCreation(lines_out, lag_ratio=0.01, run_time=4),
            frame.animate.reorient(-93, 62, 0, (-2.43, 0.5, -0.12), 9.60).set_anim_args(run_time=5)
        )
        self.play(
            FadeOut(lines_out, time_span=(1, 2)),
            FadeOut(big_ref_wave),
            FadeIn(mid_lines_out, time_span=(1, 2)),
            frame.animate.to_default_state(),
            run_time=3,
        )
        self.play(LaggedStartMap(ShowCreation, ghost_lines, lag_ratio=0.01))
        self.wait()
        self.play(Blink(randy))
        self.wait()

        # Move character around
        def get_view_point():
            eye_point = randy.eyes[1].get_center()
            obj_point = obj_dot.get_center()
            vect = obj_point - eye_point
            alpha = 1.0 - (obj_point[0] / vect[0])
            return eye_point + alpha * vect

        def update_lines(lines):
            view_point = get_view_point()
            min_dist = 2.5 * get_norm(lines[0].get_start() - lines[1].get_start())
            for line in lines:
                dist = get_norm(line.get_start() - view_point)
                alpha = clip(inverse_interpolate(min_dist, 0, dist), 0, 1)
                line.set_stroke(opacity=interpolate(0, 1, alpha))

        screen_dot = GlowDot(radius=0.5)
        screen_dot.f_always.move_to(get_view_point)

        mid_lines_out.add_updater(update_lines)
        ghost_lines.add_updater(update_lines)

        randy.always.look_at(obj_dot)

        self.play(FadeIn(screen_dot))
        self.add(mid_lines_out, ghost_lines)
        for y in [-2.8, 2.2]:
            self.play(randy.animate.set_y(y), run_time=4)

        # Movement in 3d
        dense_lines_out.clear_updaters()
        dense_ghost_lines.clear_updaters()
        dense_lines_out.add_updater(update_lines)
        dense_ghost_lines.add_updater(update_lines)

        glass = Rectangle()
        glass.rotate(PI / 2, UP)
        glass.replace(plate, stretch=True)
        glass.set_stroke(WHITE, 1)
        glass.set_fill(BLACK, 0.25)

        self.remove(plate)
        self.add(glass, randy)
        self.play(
            FadeIn(glass, time_span=(0, 1)),
            FadeOut(mid_lines_out, time_span=(1, 2)),
            FadeOut(ghost_lines, time_span=(1, 2)),
            FadeIn(dense_lines_out, time_span=(1, 2)),
            FadeIn(dense_ghost_lines, time_span=(1, 2)),
            randy.animate.rotate(PI / 2, RIGHT).shift(0.2 * (IN + DOWN)),
            frame.animate.reorient(-48, 71, 0, (0.53, -0.56, 0.06), 10.86),
            run_time=3
        )

        frame.add_ambient_rotation(1 * DEGREES)
        for (y, z) in [(-3, 2), (-2, -1.5), (-1, 1), (2, 2), (1.1, 1.1)]:
            self.play(randy.animate.set_y(y).set_z(z), run_time=3)

        # Reintroduce the beams
        frame.clear_updaters()
        dense_lines_out.clear_updaters()
        dense_ghost_lines.clear_updaters()
        self.play(
            FadeOut(dense_lines_out, time_span=(0, 1)),
            FadeOut(dense_ghost_lines, time_span=(0, 1)),
            FadeOut(screen_dot, time_span=(0, 1)),
            FadeOut(glass, time_span=(2, 3)),
            FadeIn(plate_top, time_span=(2.0, 3)),
            randy.animate.rotate(PI / 2, LEFT).move_to(4 * LEFT, DOWN),
            frame.animate.reorient(0, 0, 0, ORIGIN, FRAME_HEIGHT),
            run_time=3
        )
        self.play(GrowFromPoint(ref_wave, ref_wave.get_right(), rate_func=linear))
        out_beams.clear_updaters()
        self.play(GrowFromPoint(out_beams, film_dot.get_center(), rate_func=linear))
        out_beams.add_updater(update_out_beams)
        self.wait(3)

        # Move film point
        self.play(
            film_dot.animate.match_y(randy.eyes),
            run_time=2
        )
        self.wait(6)
        self.play(
            randy.animate.move_to(2.4 * LEFT),
            run_time=2
        )

        # Highlight other first order beam
        self.remove(out_beams)
        out_beams = self.get_triple_beam(film_dot.get_center(), obj_dot.get_center())
        self.add(out_beams)
        self.play(
            out_beams[0][1].animate.set_opacity(0.25),
            out_beams[1][1].animate.set_opacity(0.25),
        )
        randy.clear_updaters()
        self.play(randy.change("confused", film_point))
        self.play(Blink(randy))
        self.wait(4)
        self.play(FadeOut(randy))

        # Add all other first order beams
        out_beams.add_updater(update_out_beams)
        out_beams.add_updater(lambda m: m[0][1].set_opacity(0.25))
        out_beams.add_updater(lambda m: m[1][1].set_opacity(0.25))

        conj_lines = VGroup(
            Line(point, obj_dot.get_center())
            for point in mid_line_points.get_points()
        )
        conj_lines.flip(UP, about_point=ORIGIN)
        conj_lines.set_stroke(YELLOW, 1)
        conj_lines.sort(lambda p: -p[1])

        self.add(out_beams)
        self.play(film_dot.animate.set_y(4), run_time=4)
        self.play(
            film_dot.animate.set_y(-4),
            FadeIn(conj_lines, lag_ratio=0.25, run_time=4),
            run_time=5,
        )
        self.play(film_dot.animate.set_y(3.5), run_time=8)
        self.play(film_dot.animate.set_y(2), run_time=8)
        self.wait()
        self.play(FadeOut(conj_lines))

        # Show higher order beams
        theta = -angle_of_vector(obj_dot.get_center() - film_dot.get_center())
        wave_number = 250
        wavelength = 1.0 / wave_number
        spacing = wavelength / math.sin(theta)
        n_sources = 32
        sources = DotCloud().to_grid(n_sources, 1)
        sources.set_height(spacing * (n_sources - 1))
        sources.move_to(film_dot)
        out_wave = LightWaveSlice(sources)
        out_wave.set_color(BLUE_A)
        out_wave.set_shape(20, 20)
        out_wave.move_to(ORIGIN, RIGHT)
        out_wave.set_wave_number(wave_number)
        out_wave.set_frequency(1.0)
        out_wave.set_decay_factor(0)
        out_wave.set_max_amp(10)

        self.play(
            FadeOut(out_beams),
            GrowFromPoint(out_wave, film_dot.get_center()),
            frame.animate.reorient(0, 0, 0, (-0.1, 1.85, 0.0), 11.08).set_anim_args(run_time=4),
        )
        self.wait(4)

    def get_radiating_lines(self, point_cloud, obj_dot, length=20, stroke_color=YELLOW, stroke_width=1):
        lines = VGroup()
        for point in point_cloud.get_points():
            line = Line(obj_dot.get_center(), point)
            line.set_length(length)
            line.shift(point - line.get_start())
            line.set_stroke(stroke_color, stroke_width)
            lines.add(line)
        return lines

    def get_ghost_lines(self, point_cloud, obj_dot, dash_length=0.15, stroke_color=WHITE, stroke_width=1):
        lines = VGroup(
            DashedLine(point, obj_dot.get_center(), dash_length=dash_length)
            for point in point_cloud.get_points()
        )
        lines.set_stroke(stroke_color, stroke_width)
        return lines

    def get_3d_ref_wave(self, plate, n_planes=50, spacing=1.0, speed=1.0, opacity=0.25):
        planes = Square3D().replicate(n_planes)
        planes.rotate(PI / 2, UP)
        planes.replace(plate, stretch=True)
        planes.arrange(RIGHT, buff=spacing)
        planes.move_to(RIGHT, LEFT)
        for n, plane in enumerate(planes):
            plane.set_color([BLUE, RED][n % 2])
            plane.set_opacity(opacity)

        def update_planes(planes, dt):
            for plane in planes:
                plane.shift(dt * LEFT * speed)
                x = plane.get_x()
                if x < 1:
                    plane.set_opacity(opacity * x)
                if x < 0.1:
                    plane.next_to(planes, RIGHT, buff=spacing)
                    plane.set_opacity(opacity)
            planes.sort(lambda p: -p[0])
            return planes

        planes.add_updater(update_planes)

        return planes

    def get_triple_beam(self, film_point, obj_point, **kwargs):
        theta = PI - angle_of_vector(film_point - obj_point)
        beams = Group(
            self.get_beam(angle=angle, **kwargs)
            for angle in [-theta, 0, theta]
        )
        beams.shift(film_point)
        beams.deactivate_depth_test()
        return beams

    def get_beam(self, height=0.1, width=15, n_sources=8, source_height=0.15, wave_number=20, frequency=2.3, color=BLUE_A, opacity=0.75, angle=0):
        mini_sources = DotCloud().to_grid(n_sources, 1)
        mini_sources.set_height(source_height)
        mini_sources.set_radius(0).set_opacity(0)
        mini_sources.move_to(0.75 * RIGHT)
        wave = LightWaveSlice(
            mini_sources,
            wave_number=wave_number,
            frequency=frequency,
            color=color,
            opacity=opacity,
            decay_factor=0.25,
            max_amp=0.4 * n_sources,
        )
        wave.set_shape(width, height)
        wave.move_to(ORIGIN, RIGHT)
        beam = Group(mini_sources, wave)
        beam.rotate(angle, about_point=ORIGIN)
        return beam


class SuperpositionOfPoints(InteractiveScene):
    def construct(self):
        # Set up pi creature dot cloud
        frame = self.frame
        self.set_floor_plane("xz")

        output_dir = Path(self.file_writer.output_directory)
        data_file = output_dir.parent.joinpath("data", "PiCreaturePointCloud.csv")
        all_points = np.loadtxt(data_file, delimiter=',', skiprows=1)
        all_points = all_points[:int(0.8 * len(all_points))]  # Limit to first 400k
        dot_cloud = DotCloud(all_points)
        dot_cloud.set_height(4).center()
        dot_cloud.rotate(50 * DEGREES, DOWN)
        points = dot_cloud.get_points().copy()
        max_z_index = np.argmax(points[:, 2])
        min_z_index = np.argmin(points[:, 2])
        all_points = np.array([points[max_z_index], points[min_z_index], *points])

        dot_cloud.set_points(all_points[:100_000])
        dot_cloud.set_radius(0.02)

        # Add axes, points and plate
        plate_center = 5 * IN
        axes = ThreeDAxes(x_range=(-6, 6), y_range=(-4, 4), z_range=(-4, 8))
        axes.shift(plate_center - axes.get_origin())

        dist_point = 1000 * OUT
        dot_cloud.set_points(np.array([2 * LEFT, 2 * LEFT, dist_point]))
        dot_cloud.set_color(BLUE_B)
        dot_cloud.set_radius(0.5)
        dot_cloud.set_glow_factor(2)

        plate = LightIntensity(dot_cloud)
        plate.set_color(WHITE)
        plate.set_shape(16, 9)
        plate.set_height(6)
        plate.move_to(plate_center)
        plate.set_wave_number(16)
        plate.set_max_amp(4)
        plate.set_decay_factor(0)

        frame.reorient(-66, -21, 0, (-0.95, 0.41, -1.11), 11.73)
        frame.clear_updaters()
        frame.add_ambient_rotation(1 * DEGREES)
        self.add(axes)
        self.add(plate)
        self.add(dot_cloud)

        # Separate out pair of points
        point_sets = [
            (2 * LEFT, RIGHT + OUT),
            (2 * LEFT + IN, RIGHT + OUT),
            (2 * LEFT + IN, 3 * RIGHT + 2 * IN),
            (LEFT + 2 * OUT, RIGHT + OUT),
        ]

        for point_set in point_sets:
            self.play(
                dot_cloud.animate.set_points([*point_set, dist_point]),
                run_time=3
            )
            self.wait(2)

        # Zoom in on the plate
        frame.clear_updaters()
        self.play(
            frame.animate.reorient(-18, -11, 0, (-1.52, 1.18, -0.67), 0.92),
            run_time=6,
        )
        self.wait()

        dot_cloud.set_points([point_sets[-1][0], dist_point])
        plate.set_max_amp(3)
        self.wait(2)
        dot_cloud.set_points([point_sets[-1][1], dist_point])
        self.wait(2)
        dot_cloud.set_points([*point_sets[-1], dist_point])
        plate.set_max_amp(4)
        self.play(
            frame.animate.reorient(61, -7, 0, (0.61, -0.11, -2.44), 8.66),
            run_time=5
        )

        # Add on up to 32 points
        self.play(
            dot_cloud.animate.set_points([*all_points[:2], dist_point]).set_radius(0.2),
            run_time=8
        )
        frame.clear_updaters()
        frame.add_ambient_rotation(0.5 * DEGREES)
        self.play(
            UpdateFromAlphaFunc(
                dot_cloud,
                lambda m, a: m.set_points(
                    [*all_points[:int(2 + a * 29)], dist_point]
                )
            ),
            UpdateFromFunc(plate, lambda m: m.set_max_amp(2 * np.sqrt(dot_cloud.get_num_points()))),
            run_time=10
        )
        self.wait(2)

        # Describe as a combination of zone plates
        zone_plates = Group()
        for point in all_points[:30]:
            zone_plate = LightIntensity(DotCloud([point, dist_point]))
            zone_plate.set_uniforms(dict(plate.uniforms))
            zone_plate.match_points(plate)
            zone_plate.set_max_amp(10)
            zone_plate.set_opacity(0.25)
            zone_plates.add(zone_plate)

        zone_plates.deactivate_depth_test()
        self.remove(plate)
        self.add(zone_plates)

        for n, zone_plate, point in zip(it.count(1), zone_plates, all_points[:30]):
            zone_plate.shift(1e-2 * IN)
            zone_plate.save_state()
            zone_plate.scale(0).move_to(point).set_max_amp(2).set_opacity(1)

        self.play(
            UpdateFromAlphaFunc(plate, lambda m, a: m.set_opacity(1 - there_and_back_with_pause(a, 0.6))),
            LaggedStartMap(Restore, zone_plates, lag_ratio=0.05),
            frame.animate.reorient(66, -18, 0, (1.44, 0.59, -6.18), 16.05),
            run_time=5
        )
        self.play(FadeOut(zone_plates))

        # Move around points
        frame.clear_updaters()
        frame.add_ambient_rotation(-2 * DEGREES)

        dot_cloud.save_state()
        self.play(dot_cloud.animate.shift(2 * IN), run_time=3)
        self.play(Rotate(dot_cloud, PI / 2 , axis=UP, about_point=ORIGIN), run_time=3)
        self.play(Restore(dot_cloud), run_time=3)
        self.wait(3)

        # Show reference wave through it
        rect = Rectangle().replace(plate, stretch=True)
        rect.insert_n_curves(20)
        beam = VGroup(
            Line(25 * OUT, rect.pfp(a))
            for a in np.linspace(0, 1, 500)
        )
        beam.set_stroke(GREEN_SCREEN, (1, 0), 0.5)
        beam.shuffle()

        self.play(
            ShowCreation(beam, lag_ratio=1 / len(beam)),
            FadeOut(dot_cloud),
            frame.animate.reorient(83, -27, 0, (-0.63, -0.01, -0.79), 14.36),
            run_time=2
        )
        frame.clear_updaters()
        dot_cloud.set_color(GREEN_SCREEN)

        # Test
        frame.reorient(-26, -9, 0, (0.15, -0.46, -0.42), 15.13)
        self.play(frame.animate.reorient(0, 0, 0, (0.53, 0.16, -0.0), 0.22), run_time=8)
        self.play(frame.animate.reorient(3, 0, 0, (0.14, -0.02, 0.03), 0.22), run_time=8)

        # Build it up again from the other side
        self.play(
            plate.animate.set_opacity(0.2).set_anim_args(time_span=(1.6, 1.7)),
            frame.animate.reorient(162, -3, 0, (-0.89, 0.06, 0.03), 12.77),
            run_time=4,
        )
        dot_cloud.set_points([all_points[0], dist_point])
        plate.set_max_amp(2 * np.sqrt(dot_cloud.get_num_points()))
        self.add(dot_cloud)
        self.wait()
        self.play(
            UpdateFromAlphaFunc(
                dot_cloud,
                lambda m, a: m.set_points(
                    [*all_points[:int(1 + a * 29)], dist_point]
                )
            ),
            UpdateFromFunc(plate, lambda m: m.set_max_amp(2 * np.sqrt(dot_cloud.get_num_points()))),
            frame.animate.reorient(207, -8, 0, (-0.89, 0.06, 0.03), 12.77),
            run_time=12
        )
        self.wait(2)

        # Close up on cloud
        self.play(
            FadeOut(beam),
            FadeOut(dot_cloud),
            frame.animate.reorient(185, -39, 0, (-0.89, 0.06, 0.03), 12.77).set_anim_args(run_time=3),
        )
        dot_cloud.set_color(BLUE).set_glow_factor(1).set_radius(0.1)
        self.play(
            FadeOut(plate),
            FadeIn(dot_cloud),
            frame.animate.reorient(115, -16, 0, (0.33, 0.28, -0.52), 4.38),
            run_time=3
        )

        # Denser cloud
        self.play(
            UpdateFromAlphaFunc(
                dot_cloud,
                lambda m, a: m.set_points(
                    [*all_points[:int(interpolate(31, 500, a))], dist_point]
                ).set_glow_factor(interpolate(1, 0, a**0.25)).set_radius(interpolate(0.1, 0.02, a**0.25)).set_opacity(interpolate(1, 0.5, a**0.25)),
            ),
            run_time=4
        )
        self.play(
            UpdateFromAlphaFunc(
                dot_cloud,
                lambda m, a: m.set_points(
                    [*all_points[:int(interpolate(500, len(all_points), a**3))]]
                ).set_radius(interpolate(0.02, 0.01, a)).set_opacity(interpolate(0.5, 0.2, a)),
            ),
            run_time=5
        )

        # Add better updating film
        sheet_dots = self.create_dot_sheet(plate.get_width(), plate.get_height(), radius=0.025, z=plate.get_z())
        self.color_sheet_by_exposure(sheet_dots, dot_cloud.get_points()[:1000], wave_number=32)
        self.add(sheet_dots)

        self.play(
            frame.animate.reorient(-16, -45, 0, (-0.97, 1.52, -1.18), 8.67),
            run_time=2,
        )
        frame.clear_updaters()
        frame.add_ambient_rotation(3 * DEGREES)

        # Move dot cloud around
        self.play(
            dot_cloud.animate.shift(2 * IN),
            UpdateFromFunc(sheet_dots, lambda m: self.color_sheet_by_exposure(m, dot_cloud.get_points()[:1000], wave_number=32)),
            run_time=3,
        )
        self.wait(3)
        self.play(
            Rotate(dot_cloud, 120 * DEGREES, axis=UP),
            UpdateFromFunc(sheet_dots, lambda m: self.color_sheet_by_exposure(m, dot_cloud.get_points()[:1000], wave_number=32)),
            run_time=3,
        )
        self.wait(3)
        self.play(
            dot_cloud.animate.shift(3 * OUT),
            UpdateFromFunc(sheet_dots, lambda m: self.color_sheet_by_exposure(m, dot_cloud.get_points()[:1000], wave_number=32)),
            run_time=3,
        )
        self.wait(3)

        # Transform point into film
        frame.clear_updaters()
        frame.reorient(111, -13, 0, (-0.54, 0.04, -1.71), 5.72)
        pre_dots = dot_cloud.copy()
        pre_dots.set_points(dot_cloud.get_points()[:len(sheet_dots.get_points())])

        self.play(
            TransformFromCopy(pre_dots, sheet_dots, time_span=(2, 8)),
            frame.animate.reorient(17, -19, 0, (0.82, 0.57, -3.07), 7.99),
            run_time=12
        )
        self.play(
            dot_cloud.animate.shift(2 * IN),
            UpdateFromFunc(sheet_dots, lambda m: self.color_sheet_by_exposure(m, dot_cloud.get_points()[:1000], wave_number=32)),
            run_time=3,
        )
        self.wait()

    def color_sheet_by_exposure(self, sheet_dots, point_sources, wave_number=16, opacity=0.5):
        centers = sheet_dots.get_points()
        diffs = centers[:, np.newaxis, :] - point_sources[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        amplitudes = np.exp(distances * TAU * 1j * wave_number).sum(1)
        mags = abs(amplitudes)
        max_amp = 2 * np.sqrt(len(point_sources))
        opacities = opacity * np.clip(mags / max_amp, 0, 1)
        sheet_dots.set_opacity(opacities)
        return sheet_dots

    def create_dot_sheet(self, width=4, height=4, radius=0.05, z=0, make_3d=False):
        # Add dots
        dots = DotCloud()
        dots.set_color(WHITE)
        dots.to_grid(int(height / radius), int(width / radius))
        dots.set_shape(width, height)
        dots.set_radius(radius)
        dots.set_z(z)

        if make_3d:
            dots.make_3d()

        return dots


class ComplexWavesBase(InteractiveScene):
    def construct(self):
        # Transition from TwoInterferingWaves to just show the object wave
        # Maybe it makes more sense to do that from TwoInterferingWaves itself?
        pass


class ComplexWaves(InteractiveScene):
    def construct(self):
        # Add Amplitude(R + O)^2
        amp_expr = Tex(R"\\text{Amplitude}(R + O)^2", font_size=60)
        amp_expr.to_edge(UP)
        RO = amp_expr[R"R + O"][0]
        RO.save_state()
        RO.set_x(0)

        self.play(FadeIn(RO, UP))
        self.wait()
        self.play(
            Write(amp_expr[R"\\text{Amplitude}("][0]),
            Write(amp_expr[R")"][0], time_span=(1.5, 2)),
            Restore(RO, time_span=(0.5, 1.5)),
            run_time=2
        )
        self.wait()
        self.play(FadeIn(amp_expr[R"^2"], 0.25 * UP, scale=0.8))
        self.wait()

        # Expand as functions of (x, y, z, t)
        amp_expr.save_state()

        O_func = Tex("O(x, y, z, t)", font_size=60)
        O_func.move_to(UP + LEFT)

        xyz_rect = SurroundingRectangle(O_func["x, y, z"], buff=0.05)
        xyz_rect.set_stroke(YELLOW)
        xyz_rect.stretch(1.3, 1, about_edge=DOWN)
        xyz_rect.round_corners()
        xyz_arrow = Vector(2.2 * UP, thickness=5).next_to(xyz_rect, DOWN)
        xyz_arrow.set_backstroke(BLACK, 4)
        space_words = Text("Point\\nin space", font_size=36)
        space_words.next_to(xyz_rect, UP)

        time_rect = SurroundingRectangle(O_func["t"], buff=0.05)
        time_rect.match_height(xyz_rect, stretch=True, about_edge=DOWN)
        time_rect.align_to(xyz_rect, DOWN)
        time_rect.round_corners()
        time_rect.set_stroke(TEAL)
        time_word = Text("Time", font_size=36)  # Make a clock instead?
        time_word.next_to(time_rect, UP)

        self.play(
            amp_expr.animate.scale(0.5).to_corner(UL).set_opacity(0.5),
            TransformFromCopy(amp_expr["O"][0], O_func["O"][0]),
        )
        self.play(Write(O_func[1:], run_time=1, stroke_color=WHITE))
        O_func.set_backstroke(BLACK, 5)
        self.play(
            ShowCreation(xyz_rect),
            GrowArrow(xyz_arrow),
            FadeIn(space_words, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            FadeTransformPieces(space_words, time_word),
            ShowCreation(time_rect),
        )
        self.wait()

        # Show O(x, y, z, t) outputting to a real line
        frequency = 0.25
        amplitude = 1.5
        out_arrow = Vector(RIGHT, thickness=4)
        out_arrow.next_to(O_func, RIGHT)

        real_line = NumberLine((-2, 2, 0.25), width=4, tick_size=0.025, big_tick_spacing=1.0, longer_tick_multiple=3)
        real_line.next_to(out_arrow, RIGHT)
        plane = ComplexPlane(
            (-2, 2), (-2, 2),
            width=4,
            background_line_style=dict(
                stroke_color=GREY_C,
                stroke_width=1,
            ),
            faded_line_style=dict(
                stroke_color=GREY_D,
                stroke_width=0.5,
                stroke_opacity=1,
            )
        )
        plane.move_to(real_line)

        real_line.add_numbers(list(range(-2, 3)), font_size=16)
        plane.add_coordinate_labels(font_size=16)
        plane.set_stroke(behind=True)

        time_tracker = ValueTracker()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        def get_z():
            return amplitude * np.exp(complex(0, TAU * frequency * time_tracker.get_value()))

        def get_z_point():
            return plane.n2p(get_z())

        real_indicator = Group(GlowDot(radius=0.3), TrueDot().make_3d())

        def update_real_indicator(indicator):
            x = get_z().real
            indicator.move_to(plane.n2p(x))
            if x > 0:
                indicator.set_color(interpolate_color(GREY_D, BLUE, x / amplitude))
            else:
                indicator.set_color(interpolate_color(GREY_D, RED, -x / amplitude))

        real_indicator.add_updater(update_real_indicator)

        self.add(time_tracker)
        self.play(
            GrowArrow(out_arrow),
            FadeIn(real_line),
            xyz_rect.animate.set_stroke(width=1),
            time_rect.animate.set_stroke(width=1),
            FadeOut(time_word),
            FadeIn(real_indicator)
        )
        self.wait(12)

        # Extend to complex plane
        complex_label = Text("Complex Plane")
        complex_label.next_to(plane, UP)
        complex_dot = real_indicator.copy()
        complex_dot.clear_updaters()
        complex_dot.set_color(YELLOW)
        complex_dot.f_always.move_to(get_z_point)

        complex_arrow = Vector(RIGHT)
        complex_arrow.set_color(YELLOW)
        complex_arrow.f_always.put_start_and_end_on(plane.get_origin, get_z_point)

        v_line = Line(UP, DOWN)
        v_line.set_stroke(GREY, 1)
        v_line.f_always.put_start_and_end_on(get_z_point, real_indicator.get_center)

        self.add(plane, real_indicator)
        self.play(
            FadeIn(plane),
            FadeOut(real_line),
            FadeIn(complex_arrow),
            FadeIn(v_line),
        )
        self.play(Write(complex_label))
        self.wait(12)

        # Get into a good position
        time_tracker.resume_updating()
        self.wait_until(lambda: 0.4 < time_tracker.get_value() % 4 < 0.5)
        time_tracker.suspend_updating()

        # Mention amplitude and phase
        angle = complex_arrow.get_angle()
        rot_arrow = complex_arrow.copy()
        rot_arrow.clear_updaters()
        rot_arrow.rotate(-angle, about_point=rot_arrow.get_start())
        rot_arrow.set_opacity(0)
        brace = Brace(rot_arrow, UP, buff=0)
        amp_label = brace.get_text("Amplitude", font_size=30)
        amp_label.set_backstroke(BLACK, 5)
        VGroup(brace, amp_label).rotate(angle, about_point=complex_arrow.get_start())

        arc = Arc(0, angle, radius=0.5, arc_center=plane.get_origin())
        phase_label = Text("Phase", font_size=30)
        phase_label.next_to(arc, RIGHT, SMALL_BUFF)
        phase_label.shift(SMALL_BUFF * UR)

        self.play(
            GrowFromCenter(brace),
            Write(amp_label),
        )
        self.wait()
        self.play(
            ShowCreation(arc),
            TransformFromCopy(rot_arrow, complex_arrow, path_arc=angle),
            Write(phase_label)
        )
        self.wait()

        # Re-emphasize the real component
        self.play(FadeOut(VGroup(brace, amp_label, arc, phase_label)))
        time_tracker.resume_updating()

        plane.save_state()
        self.play(
            plane.animate.fade(0.75),
            FadeIn(real_line),
            complex_arrow.animate.set_fill(opacity=0.25)
        )
        self.wait(8)
        self.play(
            Restore(plane),
            FadeOut(real_line),
            complex_arrow.animate.set_fill(opacity=1.0)
        )
        self.wait(8)
        self.play(
            FadeOut(xyz_rect),
            FadeOut(time_rect),
            FadeOut(xyz_arrow),
            FadeOut(out_arrow),
            FadeOut(real_indicator),
            FadeOut(v_line),
        )

        # Package back into R + O expression
        time_tracker.suspend_updating()

        self.remove(complex_label)
        plane.add(complex_label)
        self.add(plane, complex_arrow)
        self.play(
            Transform(O_func, amp_expr.saved_state[-3], remover=True),
            Restore(amp_expr),
            plane.animate.move_to(DOWN),
            run_time=2
        )

        # Add R arrow
        O_arrow = complex_arrow
        O_arrow.clear_updaters()
        R_arrow = Vector().set_color(TEAL)
        comb_arrow = Vector().set_color(GREY_B)

        R_phase_tracker = ValueTracker(30 * DEGREES)
        O_phase_tracker = ValueTracker(complex_arrow.get_angle())
        R_amp = math.sqrt(2)
        O_amp = 1.5

        def get_R():
            return R_amp * np.exp(complex(0, R_phase_tracker.get_value()))

        def get_O():
            return O_amp * np.exp(complex(0, O_phase_tracker.get_value()))

        R_arrow.put_start_and_end_on(plane.get_origin(), plane.n2p(get_R()))
        comb_arrow.put_start_and_end_on(plane.get_origin(), plane.n2p(get_R() + get_O()))

        R_label = self.get_arrow_label(R_arrow, "R")
        O_label = self.get_arrow_label(O_arrow, "O")
        comb_label = self.get_arrow_label(comb_arrow, "R + O", buff=-0.5)

        self.play(
            GrowArrow(R_arrow),
            O_arrow.animate.shift(R_arrow.get_vector()),
            FadeIn(R_label),
            FadeIn(O_label),
        )
        self.play(
            FadeIn(comb_arrow),
            FadeIn(comb_label),
        )
        self.wait()

        R_arrow.add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(), plane.n2p(get_R())
        ))
        O_arrow.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(get_R()), plane.n2p(get_R() + get_O()),
        ))
        comb_arrow.add_updater(lambda m: m.put_start_and_end_on(
            plane.get_origin(), plane.n2p(get_R() + get_O()),
        ))

        # Write |R + O|^2
        lhs = Text("Film opacity = ")
        lhs.move_to(amp_expr, LEFT)
        lhs.set_color(GREY_B)
        new_amp_expr = Tex(R"c|R + O|^2")
        new_amp_expr.next_to(lhs, RIGHT)

        self.play(
            ReplacementTransform(amp_expr["(R + O)^2"][0], new_amp_expr),
            FadeOut(amp_expr["Amplitude"][0]),
        )
        self.play(FadeIn(lhs, lag_ratio=0.1))
        self.wait()

        # Change R and O values
        self.play(
            R_phase_tracker.animate.set_value(-45 * DEGREES),
            O_phase_tracker.animate.set_value(-45 * DEGREES), run_time=2
        )
        self.wait()
        self.play(
            O_phase_tracker.animate.set_value(-125 * DEGREES),
            R_phase_tracker.animate.set_value(45 * DEGREES),
            run_time=3
        )
        self.wait()

    def get_arrow_label(self, arrow, symbol, font_size=24, buff=0.25):
        result = Tex(symbol, font_size=font_size)
        result.match_color(arrow)
        result.add_updater(lambda m: m.move_to(arrow.get_center() + buff * normalize(rotate_vector(arrow.get_vector(), PI / 2))))
        return result


class StateOnA2DScreen(InteractiveScene):
    def construct(self):
        # Add screen
        frame = self.frame
        self.set_floor_plane("xz")

        screen = ScreenRectangle()
        screen.set_height(6)
        screen.set_stroke(WHITE, 1)

        source_points = DotCloud(np.random.random((10, 3)))
        source_points.set_width(8)
        source_points.move_to(screen, OUT)
        wave2d = LightWaveSlice(source_points)
        wave2d.replace(screen, stretch=True)
        wave2d.set_wave_number(4)
        wave2d.set_frequency(1)
        wave2d.set_max_amp(4)
        wave2d.set_decay_factor(0)

        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))

        self.add(axes)
        self.add(screen)
        self.add(wave2d)
        self.wait(4)

        # Zoom out to 3d waves
        wave3d = Group(
            # wave2d.copy().rotate(PI / 2, RIGHT).stretch(10, 2).move_to(ORIGIN, IN)
        )
        n_slices = 3
        for x in np.linspace(-5, 5, n_slices):
            wave_slice = wave2d.copy()
            wave_slice.scale(20)
            wave_slice.rotate(PI / 2, UP)
            wave_slice.move_to(ORIGIN, IN)
            wave_slice.set_x(x)
            wave_slice.set_opacity(1.0 / n_slices)
            wave3d.add(wave_slice)

        for wave in wave3d:
            wave.set_opacity(1)
            wave.set_max_amp(10)

        # wave3d[n_slices // 2].set_opacity(0.75)
        # wave3d.set_opacity(0.1)
        # wave3d.save_state()
        # wave3d.stretch(0, dim=2, about_point=ORIGIN)

        self.play(
            frame.animate.reorient(101, -1, 0, (-0.53, 0.13, 7.82), 13.40),
            FadeIn(wave3d, time_span=(3, 8)),
            run_time=8,
        )

        # Linger and collapse
        self.wait(4)
        self.play(
            LaggedStart(*(
                wave.animate.match_points(wave2d).set_opacity(1).shift(1e-2 * IN)
                for wave in wave3d
            ), lag_ratio=0),
            frame.animate.reorient(25, -6, 0, (0.18, 0.29, 0.15), 9.15).set_anim_args(time_span=(1, 4)),
            run_time=4
        )
        self.wait(8)


## Old ##

class PointSourceDiffractionPattern(InteractiveScene):
    # default_frame_orientation = (-35, -10)
    include_axes = True
    axes_config = dict(
        x_range=(-6, 6),
        y_range=(-6, 6),
        z_range=(-6, 6),
    )
    light_frequency = 2.0
    wave_length = 1.0
    use_hue = False
    max_mag = 3.0

    def setup(self):
        super().setup()

        self.set_floor_plane("xz")
        self.frame.reorient(-35, -10)

        if self.include_axes:
            axes = self.axes = ThreeDAxes(**self.axes_config)
            axes.set_stroke(opacity=0.5)
            self.add(axes)

        # Set up light sources
        points = self.point_sources = DotCloud(self.get_point_source_locations())
        points.set_glow_factor(2)
        points.set_radius(0.2)
        points.set_color(WHITE)
        self.add(points)

        # Add frequency trackerg
        self.frequency_tracker = ValueTracker(self.light_frequency)
        self.wave_length_tracker = ValueTracker(self.wave_length)
        self.light_time_tracker = ValueTracker(0)
        self.max_mag_tracker = ValueTracker(self.max_mag)

    def get_point_source_locations(self):
        radius = 2.0
        n_sources = 5
        ring = Circle(radius=radius)
        ring.set_stroke(WHITE, 3)
        return np.array([ring.pfp(a) for a in np.arange(0, 1, 1 / n_sources)])

    def get_light_time(self):
        return self.light_time_tracker.get_value()

    def get_frequency(self):
        return self.frequency_tracker.get_value()

    def color_dot_cloud_by_diffraction(self, dot_cloud):
        frequency = self.get_frequency()
        point_sources = self.point_sources.get_points()
        max_mag = self.max_mag_tracker.get_value()

        centers = dot_cloud.get_points()
        diffs = centers[:, np.newaxis, :] - point_sources[np.newaxis, :, :]
        distances = np.linalg.norm(diffs, axis=2)
        amplitudes = np.exp(distances * TAU * 1j * frequency).sum(1)
        mags = abs(amplitudes)
        opacities = 0.5 * np.clip(mags / max_mag, 0, 1)

        n = len(centers)
        rgbas = dot_cloud.data["rgba"]
        rgbas[:, 3] = opacities

        if self.use_hue:
            hues = (np.log(amplitudes).imag / TAU) % 1
            hsl = 0.5 * np.ones((n, 3))
            hsl[:, 0] = hues
            rgbas[:, :3] = hsl_to_rgb(hsl)

        dot_cloud.set_rgba_array(rgbas)
        return dot_cloud

    def create_dot_sheet(self, width=4, height=4, radius=0.05, z=0, make_3d=False):
        # Add dots
        dots = DotCloud()
        dots.set_color(WHITE)
        dots.to_grid(int(height / radius / 2), int(width / radius / 2))
        dots.set_shape(width, height)
        dots.set_radius(radius)
        dots.add_updater(self.color_dot_cloud_by_diffraction)
        dots.suspend_updating = lambda: None  # Never suspend!
        dots.set_z(z)

        if make_3d:
            dots.make_3d()

        return dots

    # TODO, have a picture-in-picture graph showing the sine waves for a given source
    # TODO, have a picture in picture phasor`,
    annotations: {
      1: "Enables PEP 604 union types (X | Y) and postponed evaluation of annotations for cleaner type hints.",
      2: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      37: "LightWaveSlice extends Mobject. Base Mobject class — everything visible on screen inherits from this. Provides position, color, and transformation methods.",
      75: "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion.",
      77: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      148: "interpolate(alpha) is called each frame with alpha from 0→1. Defines the custom animation's behavior over its duration.",
      156: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      166: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      167: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      171: "Class LightIntensity inherits from LightWaveSlice.",
      185: "LightFieldAroundScene extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      186: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      201: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      203: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      204: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      210: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      211: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      213: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      214: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      215: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      216: "Smoothly animates the camera to a new orientation over the animation duration.",
      219: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      222: "DotCloud efficiently renders many small dots using a point-based shader — much faster than individual Dot mobjects.",
      258: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      262: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      263: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      264: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      265: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      266: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      271: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      272: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      277: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      285: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      288: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      290: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      291: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      292: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      293: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      297: "FadeOut transitions a mobject from opaque to transparent.",
      303: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      311: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      316: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      317: "FadeOut transitions a mobject from opaque to transparent.",
      318: "FadeOut transitions a mobject from opaque to transparent.",
      319: "FadeOut transitions a mobject from opaque to transparent.",
      320: "FadeOut transitions a mobject from opaque to transparent.",
      321: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      322: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      324: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      325: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      326: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      329: "DiffractionGratingScene extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      332: "setup() runs before construct(). Used to initialize shared state and add persistent mobjects.",
      377: "Class LightExposingFilm inherits from DiffractionGratingScene.",
      378: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      690: "Class TwoInterferingWaves inherits from DiffractionGratingScene.",
      691: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      832: "Class SinglePointOnFilm inherits from DiffractionGratingScene.",
      833: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      933: "Class ExplainWaveVisualization inherits from DiffractionGratingScene.",
      934: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1086: "Class CreateZonePlate inherits from DiffractionGratingScene.",
      1089: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1576: "ShowEffectOfChangedReferenceAngle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1577: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1652: "Class DoubleSlit inherits from DiffractionGratingScene.",
      1653: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2077: "Class FullDiffractionGrating inherits from DiffractionGratingScene.",
      2078: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2799: "Class PlaneWaveThroughZonePlate inherits from DiffractionGratingScene.",
      2800: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3462: "SuperpositionOfPoints extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3463: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3758: "ComplexWavesBase extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3759: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3765: "ComplexWaves extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3766: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      4064: "StateOnA2DScreen extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      4065: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      4135: "PointSourceDiffractionPattern extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
    }
  };

  files["_2024/holograms/model.py"] = {
    description: "Core hologram model: simulates the recording and playback of holograms by modeling reference beams, object waves, and their interference patterns on a photographic plate.",
    code: `import cv2
from pathlib import Path

from manim_imports_ext import *


class ExtractFramesFromFootage(InteractiveScene):
    video_file = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/holograms/SceneModel/MultiplePOVs.2.mp4"
    image_dir = "/tmp/"
    frequency = 0.25
    start_time = 0
    end_time = 14
    small_image_size = (1080, 1080)
    n_cols = 8

    def setup(self):
        super().setup()
        # Open the video file
        self.video = cv2.VideoCapture(self.video_file)
        self.video_fps = self.video.get(cv2.CAP_PROP_FPS)
        self.video_duration = self.video.get(cv2.CAP_PROP_FRAME_COUNT) / self.video_fps

    def tear_down(self):
        super().tear_down()
        self.video.release()

    def construct(self):
        video_box = Square()
        video_box.set_height(4)
        video_box.to_edge(LEFT)
        video_box.set_stroke(WHITE, 0.0)
        self.add(video_box)

        # Collect images
        start_time = self.start_time
        end_time = min(self.end_time, self.video_duration)
        times = np.arange(start_time, end_time, self.frequency)
        images = Group()
        for time in ProgressDisplay(times, desc="Loading images"):
            image = self.image_from_timestamp(time)
            if image is not None:
                border = SurroundingRectangle(image, buff=0)
                border.set_stroke(WHITE, 1)
                images.add(Group(border, image))

        # Show clip extraction
        images.arrange_in_grid(n_cols=self.n_cols, buff=0.1 * images[0].get_width())
        images.set_width(FRAME_WIDTH - video_box.get_width() - 1.5)
        images.to_corner(UR, buff=0.25)
        flash = video_box.copy()
        flash.set_fill(WHITE, 0.25)

        for image in images:
            self.add(image)
            self.play(FadeOut(flash, run_time=self.frequency / 2))
            self.wait(self.frequency / 2)

        # Add still
        still_image = images[-1].copy()
        still_image.replace(video_box)
        self.add(still_image)
        self.wait()

        # Do something to zoom in on example frames

    def image_from_timestamp(self, time):
        video_name = Path(self.video_file).stem
        file_name = Path(self.image_dir, f"{video_name}_{time}.png")

        frame_number = int(time * self.video_fps)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = self.video.read()
        if not ret:
            print(f"Failed to capture at {time}")
            return
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        small_image = pil_image.resize(self.small_image_size, Image.LANCZOS)
        small_image.save(file_name)

        return ImageMobject(file_name)`,
    annotations: {
      4: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      7: "ExtractFramesFromFootage extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      16: "setup() runs before construct(). Used to initialize shared state and add persistent mobjects.",
      27: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      55: "FadeOut transitions a mobject from opaque to transparent.",
      62: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
    }
  };

  files["_2024/holograms/supplements.py"] = {
    description: "Supplementary scenes for the hologram video: additional demonstrations of wave optics, Fourier analysis of diffraction, and bonus visualizations.",
    code: `from manim_imports_ext import *


class PhotographVsHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Single piece of film")
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        photo = VGroup(
            Text("Photograph"),
            Text("One point of view"),
        )
        holo = VGroup(
            Text("Our Recording"),
            Text("Many points of view"),
        )
        for part, x in zip([photo, holo], [-1, 1]):
            part[1].scale(0.7)
            part[1].set_color(GREY_A)
            part.arrange(DOWN)
            part.set_x(x * FRAME_WIDTH / 4)
            part.set_y(2)
            part.add_to_back(Arrow(title.get_edge_center(x * RIGHT), part.get_top(), path_arc=-x * 60 * DEGREES))

        self.add(title)

        for part in [photo, holo]:
            self.play(
                GrowArrow(part[0]),
                FadeInFromPoint(part[1], part[0].get_start()),
            )
            self.wait()
            self.play(Write(part[2]))
            self.wait()


class GlintArrow(InteractiveScene):
    def construct(self):
        # Test
        circle = Circle(radius=0.25)
        circle.set_stroke(RED, 4)
        circle.insert_n_curves(20)
        arrow = Vector(DR, thickness=5, fill_color=RED)
        arrow.next_to(circle, UL, buff=0)
        self.play(
            GrowArrow(arrow),
            VShowPassingFlash(circle, time_width=1.5, time_span=(0.5, 1.5)),
        )
        self.wait()


class WriteHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Hologram", font_size=96)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=GREEN_SCREEN, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class WriteTransmissionHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Transmission Hologram", font_size=96)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=GREEN_SCREEN, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class WriteWhiteLightReflectionHologram(InteractiveScene):
    def construct(self):
        # Test
        title = Text("White Light Reflection Hologram", font_size=72)
        title.to_edge(UP)
        self.play(Write(title, stroke_color=WHITE, stroke_width=3, lag_ratio=0.1, run_time=3))
        self.wait()


class IndicationOnPhotograph(InteractiveScene):
    def construct(self):
        # Image
        image = ImageMobject("ContrastWithPhotography")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        # Dots
        p0 = (-3.82, 2.11, 0.0)
        p1 = (-3.14, 2.17, 0.0)
        p2 = (2.61, 2.82, 0.0)
        p3 = (3.53, 2.24, 0.0)

        dot1 = GlowDot(p0, color=TEAL)
        dot2 = GlowDot(p2, color=TEAL)

        arrow = StrokeArrow(p0, p2, buff=SMALL_BUFF, path_arc=-30 * DEGREES)
        arrow.set_stroke(TEAL)

        self.play(FadeIn(dot1), FlashAround(dot1, stroke_width=5, time_width=2, color=TEAL))
        self.play(
            TransformFromCopy(dot1, dot2, path_arc=-30 * DEGREES),
            ShowCreation(arrow, run_time=2),
        )
        self.play(
            dot1.animate.move_to(p1),
            dot2.animate.move_to(p3),
            arrow.animate.match_points(StrokeArrow(p1, p3, buff=0.1, path_arc=-30 * DEGREES)),
            rate_func=there_and_back,
            run_time=3
        )
        self.play(
            FadeOut(dot1),
            FadeOut(dot2),
            FadeOut(arrow),
        )


class ContrastPhotographyAndHolography(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Information lost in ordinary photography", font_size=60)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.05)

        points = BulletedList(
            "All but one viewing angle",
            "Phase",
            buff=LARGE_BUFF
        )

        points.scale(1.25)
        points.next_to(underline, DOWN, buff=1.5)
        points.shift(2 * LEFT)
        self.add(*(p[0] for p in points))

        self.play(
            FadeIn(title),
            ShowCreation(underline)
        )

        for point in points:
            self.wait()
            self.play(Write(point[1:]))

        self.play(
            FlashAround(points[1], color=TEAL, time_width=1.5, run_time=2),
            points[1].animate.set_color(TEAL)
        )
        self.wait()


class Outline(InteractiveScene):
    def construct(self):
        # Add top part
        frame = self.frame
        frame.set_y(2)

        background = FullScreenRectangle()
        background.fix_in_frame()
        self.add(background)

        top_outlines = ScreenRectangle().replicate(3)
        top_outlines.arrange(RIGHT, buff=1.5)
        top_outlines.set_width(13)
        top_outlines.set_stroke(WHITE, 2)
        top_outlines.set_fill(BLACK, 1)
        top_outlines.to_edge(UP, buff=LARGE_BUFF)

        top_images = Group(
            ImageMobject("HologramProcess.jpg"),
            ImageMobject("SimplestHologram.jpg"),
            ImageMobject("HologramEquation.png"),
        )
        top_rects = Group(
            Group(outline, image.replace(outline))
            for outline, image in zip(top_outlines, top_images)
        )

        top_titles = VGroup(
            Text("The process"),
            Text("The simplest hologram"),
            Text("The general derivation"),
        )
        top_titles.scale(0.75)

        for rect, title in zip(top_rects, top_titles):
            title.next_to(rect, UP, SMALL_BUFF)
            self.play(
                FadeIn(rect),
                FadeIn(title, 0.25 * UP)
            )
            self.wait()

        # Highlight process
        self.play(
            FadeOut(top_rects[1:]),
            FadeOut(top_titles[1:]),
            frame.animate.set_height(4).move_to(top_rects[0]),
            run_time=2
        )
        self.wait()
        self.play(
            FadeIn(top_rects[1:]),
            FadeIn(top_titles[1:]),
            frame.animate.to_default_state().set_y(2),
            run_time=2
        )

        # Isolation animation
        def isolate(rects, titles, index, faded_opacity=0.15):
            return AnimationGroup(
                *(
                    title.animate.set_opacity(1 if i == index else faded_opacity)
                    for i, title in enumerate(titles)
                ),
                *(
                    rect.animate.set_opacity(1 if i == index else faded_opacity)
                    for i, rect in enumerate(rects)
                ),
            )

        self.play(isolate(top_rects, top_titles, 0))
        self.wait()
        self.play(isolate(top_rects, top_titles, 1))
        self.wait()

        # Break down middle step
        mid_outlines = top_outlines.copy()
        mid_outlines.set_opacity(1)
        mid_outlines.next_to(top_rects, DOWN, buff=1.5)
        outer_lines = VGroup(
            CubicBezier(
                top_rects[1].get_corner(DL),
                top_rects[1].get_corner(DL) + DOWN,
                mid_outlines[0].get_corner(UL) + 2 * UP,
                mid_outlines[0].get_corner(UL),
            ),
            CubicBezier(
                top_rects[1].get_corner(DR),
                top_rects[1].get_corner(DR) + DOWN,
                mid_outlines[2].get_corner(UR) + 2 * UP,
                mid_outlines[2].get_corner(UR),
            ),
        )
        mid_images = Group(
            ImageMobject("ZonePlateExposure"),
            ImageMobject("DotReconstruction"),
            ImageMobject("MultipleDotHologram"),
        )

        mid_rects = Group(
            Group(outline, image.replace(outline))
            for outline, image in zip(mid_outlines, mid_images)
        )

        mid_titles = VGroup(
            Text("Exposure pattern"),
            Text("Reconstruction"),
            Text("Added complexity"),
        )
        mid_titles.scale(0.75)
        for rect, title in zip(mid_rects, mid_titles):
            title.next_to(rect, UP, SMALL_BUFF)
            rect.save_state()
            rect.replace(top_rects[1])
            rect.set_opacity(0)

        self.play(
            LaggedStartMap(Restore, mid_rects),
            ShowCreation(outer_lines, lag_ratio=0),
            frame.animate.set_y(0),
            FadeIn(mid_titles, time_span=(1.5, 2.0), lag_ratio=0.025),
            run_time=2
        )
        self.wait()
        for index in range(3):
            self.play(isolate(mid_rects, mid_titles, index))
            self.wait()
        self.play(isolate(mid_rects, mid_titles, 0))
        self.wait()

        # Minilesson
        low_outline = mid_outlines[1].copy()
        low_outline.next_to(mid_rects[1], DOWN, buff=1.5)

        low_image = ImageMobject("DiffractionGrating")
        low_image.replace(low_outline)
        low_rect = Group(low_outline, low_image)
        low_rect.shift(1.5 * LEFT)

        in_arrow = Arrow(mid_rects[0].get_bottom() + LEFT, low_rect.get_left(), path_arc=PI / 2, thickness=5, buff=0.15)
        up_arrow = Arrow(low_rect, mid_rects[1], thickness=5, buff=0.15)

        low_title = Text("Mini-lesson on\\nDiffraction Gratings")
        low_title.next_to(low_rect, DOWN)

        self.play(
            frame.animate.set_y(-4),
            FadeIn(low_rect, DOWN),
            FadeIn(low_title, DOWN),
            GrowArrow(in_arrow),
            run_time=2
        )
        self.play(GrowArrow(up_arrow))
        self.wait()

        # Zoom in on mini-lesson
        frame.save_state()
        self.play(
            frame.animate.set_height(4).move_to(Group(low_rect, low_title)),
            FadeOut(VGroup(in_arrow, up_arrow)),
            run_time=2,
        )
        self.wait()
        self.play(
            Restore(frame),
            FadeIn(VGroup(in_arrow, up_arrow)),
            run_time=2
        )
        self.wait()

        # Back to the middle
        self.play(
            LaggedStartMap(FadeOut, Group(low_title, low_rect, arrow), lag_ratio=0.1, run_time=1),
            frame.animate.set_y(0).set_anim_args(run_time=2),
        )
        self.wait()
        self.play(isolate(mid_rects, mid_titles, 2))
        self.wait()

        # Back to the top
        self.play(
            LaggedStart(
                (mid_rect.animate.replace(top_rects[1]).set_opacity(0)
                for mid_rect in mid_rects),
                lag_ratio=0.05,
                group_type=Group
            ),
            FadeOut(mid_titles),
            Uncreate(outer_lines, lag_ratio=0),
            frame.animate.set_y(2),
            run_time=2
        )
        self.wait()
        self.play(isolate(top_rects, top_titles, 2))
        self.wait()


class GoalOfRediscovery(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            self.change_students("pondering", "confused", "maybe", look_at=self.screen),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            morty.says("I want this to\\nfeel rediscovered"),
            self.change_students("tease", "erm", "pondering", look_at=morty.eyes)
        )
        self.wait(2)
        self.look_at(self.screen)
        self.play(self.change_students("thinking", "pondering", "thinking"))
        self.wait(4)


class NameCraigAndSally(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("CraigSallyPaulGrant")
        image.set_height(FRAME_HEIGHT)

        craig_point = (1.2, 2.0, 0)
        sally_point = (3.9, 1.3, 0)

        craig_arrow = Vector(DR, thickness=5)
        craig_arrow.next_to(craig_point, UL)
        craig_name = Text("Craig Newswanger")
        craig_name.next_to(craig_arrow.get_start(), UP, buff=SMALL_BUFF)

        sally_arrow = Vector(DL, thickness=5)
        sally_arrow.next_to(sally_point, UR)
        sally_name = Text("Sally Weber")
        sally_name.next_to(sally_arrow.get_start(), UP, buff=SMALL_BUFF)

        VGroup(craig_arrow, craig_name, sally_arrow, sally_name).set_fill(WHITE).set_backstroke(BLACK, 5)

        self.play(
            GrowArrow(craig_arrow),
            FadeIn(craig_name),
        )
        self.play(
            GrowArrow(sally_arrow),
            FadeIn(sally_name),
        )


class ShowALens(InteractiveScene):
    def construct(self):
        # Add lens
        arc = Arc(-30 * DEGREES, 60 * DEGREES)
        flipped = arc.copy().rotate(PI).next_to(arc, LEFT, buff=0)
        arc.append_vectorized_mobject(flipped)
        lens = arc
        lens.set_stroke(WHITE, 2)
        lens.set_fill(GREY_E, 1)
        lens.set_shading(0.1, 0.5, 0)
        lens.set_height(3)

        # Add lines
        n_lines = 20
        points = np.linspace(lens.get_top(), lens.get_bottom(), n_lines)
        focal_point = lens.get_left() + 2 * LEFT
        in_lines = VGroup(Line(point + 8 * RIGHT, point) for point in points)
        out_lines = VGroup(Line(point, focal_point) for point in points)
        for in_line, out_line in zip(in_lines, out_lines):
            out_line.scale(1.25, about_point=out_line.get_start())
            in_line.append_vectorized_mobject(out_line)

        in_lines.set_stroke(WHITE, 1)
        in_lines.insert_n_curves(100)

        self.add(in_lines, lens)
        self.play(ShowCreation(in_lines, lag_ratio=0.01, run_time=2, rate_func=linear))
        self.wait()

        # Label viewing angle
        text = Text("Single viewing angle")
        text.next_to(lens.get_top(), UR)
        arrow = Vector(RIGHT).next_to(text, RIGHT)

        self.play(LaggedStart(Write(text, run_time=1), GrowArrow(arrow), lag_ratio=0.5))
        self.wait()


class AskAboutRecordingPhase(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        randy = Randolph(height=2.5)
        randy.next_to(morty, LEFT, buff=1.5, aligned_edge=DOWN)
        for pi in [morty, randy]:
            pi.body.insert_n_curves(100)

        self.add(morty, randy)
        self.play(
            morty.says(
                Text("""
                    How could you
                    detect this phase
                    difference?
                """),
                look_at=randy.eyes,
            ),
            randy.change("pondering", morty.eyes)
        )
        self.play(Blink(randy))
        self.play(
            randy.animate.look_at(3 * UL),
            morty.change("tease", 3 * UL),
        )
        self.play(randy.change("erm", 3 * UL))
        self.play(Blink(morty))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", 3 * UL))
        self.play(morty.change("pondering", 3 * UL))
        self.play(Blink(randy))
        self.wait(3)
        self.play(randy.change("thinking", 3 * LEFT))
        self.play(Blink(randy))
        self.play(Blink(morty))
        self.wait(2)


class HowIsThisHelpful(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher
        self.play(LaggedStart(
            stds[1].says("What does this have\\nto do with recording 3d?", mode="maybe", look_at=self.screen),
            stds[0].change("confused", self.screen),
            stds[2].change("erm", self.screen),
            lag_ratio=0.25,
            run_time=2
        ))
        self.play(morty.change('well'))
        self.wait(2)
        self.play(self.change_students("maybe", "confused", "pondering", look_at=self.screen))
        self.wait(5)


class NoWhiteLight(InteractiveScene):
    def construct(self):
        # Test
        white_label = VGroup(
            Text("White light", font_size=60),
            Text("(many frequencies)").set_color(GREY_B)
        )
        white_label.arrange(DOWN)
        white_label.to_edge(UP).set_x(-3)
        laser_label = VGroup(
            Text("Laser", font_size=60).set_color(GREEN_SCREEN),
            Text("(single frequency)").set_color(GREY_B),
        )
        laser_label.arrange(DOWN)
        laser_label.move_to(white_label, UP)

        cross = Cross(white_label, stroke_width=(0, 7, 7, 7, 0))

        self.add(white_label)
        self.wait()
        self.play(ShowCreation(cross))
        self.wait()
        self.play(
            FadeOut(VGroup(white_label, cross), DR),
            FadeIn(laser_label, DR),
        )
        self.wait()


class AnnotateSetup(InteractiveScene):
    def construct(self):
        # Add iamge
        image = ImageMobject("CraigSallyHologramSetup")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        # Reference polarization
        pol_words = Text("Correct the polarization")
        pol_words.set_backstroke(BLACK, 5)
        point = (-0.5, 1.2, 0)
        pol_words.next_to(point, UR, buff=1.5).shift(LEFT)
        pol_arrow = Arrow(pol_words["Correct"].get_bottom(), point)

        self.play(
            Write(pol_words),
            GrowArrow(pol_arrow)
        )
        self.wait()
        self.play(
            FadeOut(pol_words),
            FadeOut(pol_arrow),
        )

        # Show object beam
        obj_beam_points = [
            (-1.625, 2.167, 0),
            (-1.625, 1.264, 0),
            (0.375, 1.264, 0),
            (-0.625, -3.625, 0),
            (-0.111, -3.167, 0),
        ]

        obj_beam = VMobject().set_points_as_corners(obj_beam_points)
        obj_beam.set_stroke(GREEN_SCREEN, 4)
        obj_beam_name = Text("Object beam")
        obj_beam_name.set_color(GREEN_SCREEN)
        obj_beam_name.next_to(obj_beam.pfp(0.5), RIGHT)

        scene_point = (1.278, -1.972, 0)
        scene_circle = Circle(radius=0.75).shift(scene_point)
        obj_beam_spread = VGroup(
            Line(obj_beam_points[-1], scene_circle.pfp(a) + np.random.uniform(-0.025, 0.025, 3))
            for a in np.arange(0, 1, 0.001)
        )
        obj_beam_spread.set_stroke(GREEN_SCREEN, 1, 0.25)

        self.play(
            ShowCreation(obj_beam, rate_func=linear),
            FadeIn(obj_beam_name, lag_ratio=0.1),
        )
        self.play(ShowCreation(obj_beam_spread, lag_ratio=0, rate_func=rush_from))
        self.wait()

        # Show reference beam
        ref_beam_points = [
            obj_beam_points[0],
            (-1.61, -1.39, 0),
            (-1.056, -1.809, 0),
        ]
        ref_beam = obj_beam.copy()
        ref_beam.set_points_as_corners(ref_beam_points)
        ref_beam_name = Text("Reference beam")
        ref_beam_name.match_style(obj_beam_name)
        ref_beam_name.next_to(ref_beam.pfp(0.5), LEFT)

        ref_circle = Circle(radius=0.5)
        ref_circle.rotate(PI / 2, RIGHT)
        ref_circle.move_to((1.23, -3.39, 0))
        ref_beam_spread = VGroup(
            Line(ref_beam_points[-1], ref_circle.pfp(a) + np.random.uniform(-0.025, 0.025, 3))
            for a in np.arange(0, 1, 0.001)
        )
        ref_beam_spread.match_style(obj_beam_spread)

        self.play(
            FadeTransform(obj_beam_name, ref_beam_name),
            ReplacementTransform(obj_beam, ref_beam),
            ReplacementTransform(obj_beam_spread, ref_beam_spread),
        )
        self.wait()


class ArrowWithQMark(InteractiveScene):
    def construct(self):
        arrow = Arrow(3 * LEFT, 3 * RIGHT, path_arc=-PI / 2)
        q_marks = Tex(R"???", font_size=72)
        q_marks.next_to(arrow, UP)

        self.play(
            Write(arrow),
            Write(q_marks),
            run_time=2
        )
        self.wait()


class ProblemSolvingTipNumber1(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Universal problem-solving tip #1", font_size=72)
        title.to_edge(UP, buff=LARGE_BUFF)
        underline = Underline(title, buff=-SMALL_BUFF)
        parts = VGroup(title[text][0] for text in re.split(" |-", title.text))

        words = Text("Start with the simplest\\nversion of your problem", font_size=72)
        words.set_color(YELLOW)

        self.play(LaggedStartMap(FadeIn, parts, shift=0.25 * UP, lag_ratio=0.25, run_time=2))
        self.play(ShowCreation(underline))
        self.wait()
        self.play(Write(words), run_time=3)
        self.wait()


class NameElectromagneticField(InteractiveScene):
    def construct(self):
        # Test
        name1 = Text("Electromagetic Field")
        name2 = Text("Electric Field")
        for name in [name1, name2]:
            name.set_backstroke(BLACK, 5)
            name.scale(1.5)
            name.to_edge(UP)

        expl = Text("3d Vector\\nat each point")
        expl.set_x(FRAME_WIDTH / 4)
        expl.to_edge(UP)
        arrow = Vector(RIGHT)
        arrow = Vector(2 * RIGHT)
        arrow.next_to(expl, LEFT)

        self.play(Write(name1), run_time=1)
        self.wait()
        self.play(TransformMatchingStrings(name1, name2), run_time=1)
        self.wait()
        self.play(
            name2.animate.next_to(arrow, LEFT),
            GrowArrow(arrow),
            FadeIn(expl, RIGHT),
        )
        self.wait()


class HoldUpDiffractionEquationInAnticipation(TeacherStudentsScene):
    def construct(self):
        # Test
        equation = Tex(R"d \\cdot \\sin(\\theta) = n \\lambda", font_size=60)
        equation.move_to(self.hold_up_spot, DOWN)
        name = TexText("Diffraction equation")
        name.next_to(equation, UP, buff=1.5)
        arrow = Arrow(name, equation)

        morty = self.teacher
        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", equation),
            FadeIn(equation, UP),
            self.change_students("confused", "confused", "erm", equation)
        )
        self.play(
            Write(name),
            GrowArrow(arrow),
        )
        self.wait()
        self.play(
            self.change_students("maybe", "confused", "sassy", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            morty.says("You can discover\\nthis yourself", mode="hooray"),
            LaggedStartMap(FadeOut, VGroup(equation, arrow, name), shift=UR, lag_ratio=0.2, run_time=2),
            self.change_students("pondering", "pondering", "thinking", look_at=morty.eyes)
        )
        self.wait(3)


class HowLongIsThatSnippet(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("How long is\\nthat snippet?"),
            self.change_students("happy", "tease", "thinking", look_at=self.screen)
        )
        self.wait(3)


class HoldUpDiffractionEquation(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        equation = Tex(R"d \\cdot \\sin(\\theta) = \\lambda", t2c={R"\\lambda": TEAL}, font_size=60)
        equation.move_to(self.hold_up_spot, DOWN)

        words = Text("Remember this")
        words.next_to(equation, UP, 1.5)
        equation.to_edge(RIGHT)
        arrow = Arrow(words, equation)

        self.play(
            morty.change("raise_left_hand", look_at=equation),
            FadeIn(equation, UP),
            self.change_students("pondering", "pondering", "pondering", equation)
        )
        self.play(
            FadeIn(words, lag_ratio=0.1),
            GrowArrow(arrow),
        )
        self.play(morty.change("tease"))
        self.wait(3)

        morty.body.insert_n_curves(100)
        self.play(
            self.change_students("thinking", "erm", "pondering", look_at=self.screen),
            morty.change("raise_right_hand", look_at=self.screen),
        )
        self.wait(4)


class AskAboutCenterBeam(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(DOWN)
        self.play(
            randy.thinks("Why is there a\\ncentral beam?", mode="confused")
        )
        self.play(Blink(randy))
        self.play(randy.animate.look_at(3 * UR))
        self.play(randy.animate.look_at(3 * DR))


class DistanceApproximation(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(R"\\sqrt{L^2 + x^2} \\approx L + \\frac{x^2}{2L}")
        expr.to_corner(UL)

        circle = SurroundingRectangle(expr[R"\\frac{x^2}{2L}"])
        circle.round_corners()
        circle.set_stroke(RED)
        words = Text("First order\\nTaylor approx.")
        words.next_to(circle, DOWN, LARGE_BUFF)
        words.set_color(RED)
        words.set_backstroke(BLACK, 5)
        arrow = Arrow(words.get_top(), circle.get_bottom())
        arrow.set_color(RED)

        self.play(Write(expr))
        self.wait()
        self.play(
            ShowCreation(circle),
            GrowArrow(arrow),
            FadeIn(words)
        )
        self.wait()


class CircleDiffractionEquation(InteractiveScene):
    def construct(self):
        # Add equation
        self.add(FullScreenRectangle())
        equation = Tex(
            R"{d} \\cdot \\sin(\\theta) = n \\lambda",
            font_size=72,
            t2c={R"{d}": BLUE, R"\\theta": YELLOW, R"\\lambda": TEAL}
        )
        equation.set_backstroke(BLACK)
        title = Text("The Diffraction Equation", font_size=60)
        title.set_color(GREY_A)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        equation.next_to(title, DOWN, MED_LARGE_BUFF)

        self.add(title)
        self.add(equation)
        self.play(
            FlashAround(equation, run_time=4, time_width=1.5, buff=MED_SMALL_BUFF),
        )
        self.wait()


class DiffractionGratingGreenLaserExampleNumbers(InteractiveScene):
    def construct(self):
        # Test
        lines = VGroup(
            VGroup(
                Text("Grating: 500 lines / mm"),
                Tex(R"\\Rightarrow d = 2 \\times 10^{-6} \\text{m}"),
            ).arrange(RIGHT),
            VGroup(
                Text("Green light wavelength: 500 nm "),
                Tex(R"\\Rightarrow \\lambda = 5 \\times 10^{-7} \\text{m}"),
            ).arrange(RIGHT),
            Tex(R"\\theta = \\sin^{-1}(\\lambda / d) = \\sin^{-1}\\left(5 \\times 10^{-7} / 2 \\times 10^{-6} \\right) \\approx 14.5^\\circ")
        )
        lines.arrange(DOWN, aligned_edge=LEFT, buff=LARGE_BUFF)
        lines.to_edge(LEFT)
        lines[0][1].align_to(lines[1][1], LEFT)
        self.add(lines)


class AnnotateZerothOrderBeam2(InteractiveScene):
    def construct(self):
        image = ImageMobject("DiffractionGratingGreenLaser.jpg")
        # self.add(image.set_height(FRAME_HEIGHT))

        # Names
        points = [
            (-0.67, 1.64, 0.0),
            (3.64, 1.9, 0.0),
            (-4.9, 1.85, 0.0),
            (-5.4, 1.01),
            (2.42, 0.028),
        ]

        names = VGroup(
            Text("Zeroth-order beam"),
            Text("First-order beams"),
            Text("Second-order beams"),
        )
        names.next_to(points[0], UP, buff=2.5)
        names.shift_onto_screen()
        arrow0 = Arrow(names[0], points[0])
        arrows1 = VGroup(
            Arrow(names[1].get_bottom(), points[1]),
            Arrow(names[1].get_bottom(), points[2]),
        )
        arrows2 = VGroup(
            Arrow(names[2].get_bottom(), points[3]),
            Arrow(names[2].get_bottom(), points[4]),
        )
        VGroup(names, arrow0, arrows1, arrows2).set_fill(GREY_E)

        self.play(
            Write(names[0]),
            GrowArrow(arrow0),
            run_time=1
        )
        self.wait()
        self.remove(arrow0)
        self.play(
            TransformMatchingStrings(names[0], names[1], key_map={"Zeroth": "First", "m": "ms"}),
            TransformFromCopy(arrow0.replicate(2), arrows1),
            run_time=1
        )
        self.wait()
        self.play(
            TransformMatchingStrings(names[1], names[2], key_map={"First": "Second"}),
            ReplacementTransform(arrows1, arrows2),
            run_time=1
        )
        self.wait()


class ExactSpacingQuestion(InteractiveScene):
    def construct(self):
        # Test
        question = Text("What is the exact spacing?", font_size=72)
        question.set_backstroke()
        question.to_edge(UP)
        subq = TexText(R"As a function of $\\theta'$")
        subq.set_color(GREY_A)

        self.play(Write(question, stroke_color=WHITE))
        self.wait()
        self.play(question.animate.scale(48 / 72).set_x(-0.25 * FRAME_WIDTH))
        self.wait()
        subq.next_to(question, DOWN)
        self.play(FadeIn(subq, 0.5 * DOWN))
        self.wait()


class DejaVu(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("pondering", self.screen),
            stds[2].says("That looks\\nfamiliar", bubble_config=dict(direction=LEFT), look_at=self.screen),
            self.teacher.change("tease"),
        )
        self.wait(2)

        diff_eq = Tex(R"d \\cdot \\sin(\\theta) = \\lambda")
        diff_eq.next_to(stds[0], UR)
        self.play(
            stds[1].change("raise_left_hand", look_at=diff_eq),
            FadeIn(diff_eq, UP)
        )
        self.look_at(diff_eq)
        self.play(stds[2].change("pondering", diff_eq))
        self.wait(3)


class WhatAHologramReconstructs(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        title = Text("What a Hologram reproduces", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)

        items = VGroup(
            Text("1) Copy of the reference wave"),
            Text("2) Copy of the object wave"),
            TexText(R"3) Reflection$^*$ of the object wave"),
        )
        items.scale(0.8)
        items.arrange(DOWN, buff=1.75, aligned_edge=LEFT)
        items.next_to(underline, DOWN, LARGE_BUFF)
        items.to_edge(LEFT)

        self.play(
            FadeIn(title),
            ShowCreation(underline)
        )

        self.play(
            LaggedStart(
                (FadeIn(item[:2])
                for item in items),
                lag_ratio=0.05,
            )
        )
        self.wait()
        for n, item in enumerate(items):
            self.play(
                items[:n].animate.set_opacity(0.25),
                FadeIn(item[2:], lag_ratio=0.1)
            )
            self.wait()


class AskAboutHigherOrderBeams(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        morty = self.teacher
        self.play(LaggedStart(
            stds[0].change('pondering', self.screen),
            stds[1].says("What about higher\\norder beams?", look_at=self.screen),
            stds[2].change("confused", self.screen),
            morty.change("tease")
        ))
        self.wait()
        self.play(stds[1].change("sassy", morty.eyes))
        self.wait(6)


class BinaryVsSinusoidalDiffraction(InteractiveScene):
    def construct(self):
        # Add axes
        top_axes, low_axes = two_axes = VGroup(
            Axes((0, 4, 0.5), (0, 1, 0.5), width=6, height=1.5)
            for _ in range(2)
        )
        two_axes.arrange(DOWN, buff=2.0)
        two_axes.to_edge(LEFT, buff=1.5)

        for axes in two_axes:
            y_label = Text("Opacity", font_size=30)
            y_label.next_to(axes.y_axis, UP, buff=MED_SMALL_BUFF)
            axes.add(y_label)
            axes.y_axis.add_numbers(font_size=16, num_decimal_places=1)

        self.add(two_axes)

        # Add graphs
        discontinuities = [*np.arange(5), *(np.arange(5) + 0.2)]
        discontinuities.sort()
        top_graph = top_axes.get_graph(self.func1, x_range=(0, 3.99), discontinuities=discontinuities)
        top_graph.set_stroke(TEAL, 2)

        low_axes.num_sampled_graph_points_per_tick = 20
        low_graph = low_axes.get_graph(self.func2)
        low_graph.set_stroke(BLUE, 2)

        graphs = VGroup(top_graph, low_graph)

        # Add gratings
        top_grating = self.get_grating(top_axes, self.func1, 1000)
        low_grating = self.get_grating(low_axes, self.func2, 1000)
        gratings = VGroup(top_grating, low_grating)

        # Animate in
        graphs.set_stroke(width=1)
        self.add(graphs)
        for graph, grating in [(top_graph, top_grating), (low_graph, low_grating)]:
            self.add(grating, graph)
            self.play(
                graph.animate.set_stroke(width=2),
                VShowPassingFlash(graph.copy().set_stroke(width=5), time_width=3, rate_func=linear),
                # ShowCreation(graph, rate_func=linear),
                FadeIn(grating, lag_ratio=1e-3),
                run_time=2
            )
            self.play(
                grating.animate.next_to(grating, RIGHT, MED_LARGE_BUFF),
                run_time=2
            )
            self.wait()

        # Labels
        high_beam_labels = VGroup(
            Text("Higher order beams", font_size=36).next_to(grating, UP)
            for grating in gratings
        )
        check = Checkmark().set_color(GREEN).scale(1.5)
        check.next_to(high_beam_labels[0])
        ex = Exmark().set_color(RED).scale(1.5)
        ex.next_to(high_beam_labels[1])

        self.play(LaggedStart(
            FadeIn(high_beam_labels),
            Write(check),
            Write(ex),
            lag_ratio=0.5
        ))
        self.wait()

    def get_grating(self, axes, func, n_slits):
        grating = Rectangle().replicate(n_slits)
        grating.set_stroke(width=0)
        grating.set_fill(GREY_D)
        grating.set_shading(0.25, 0.25, 0)
        grating.arrange(RIGHT, buff=0)
        grating.set_shape(axes.x_axis.get_width(), axes.y_axis.get_height())
        grating.move_to(axes.c2p(0, 0), DL)

        for mob in grating:
            mob.set_opacity(func(axes.x_axis.p2n(mob.get_center())))

        # grating.next_to(grating, RIGHT, MED_LARGE_BUFF)

        return grating

    def func1(self, x):
        return 0 if x % 1 < 0.2 else 1

    def func2(self, x):
        return math.sin(TAU * x)**2


class AskWhyForSinusoidalGratingFact(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(LaggedStart(
            stds[1].says("Wait, why?", mode="confused", look_at=self.screen),
            stds[0].change("pondering", self.screen),
            stds[2].change("maybe", self.screen),
        ))
        self.wait()

        implication = VGroup(
            Tex(R"\\Leftarrow").scale(2),
            Text("More formal\\nholography explanation")
        )
        implication.arrange(RIGHT, buff=0.25)
        implication.move_to(2.5 * UP + 2.5 * RIGHT)

        self.play(
            morty.change("raise_right_hand"),
            GrowFromPoint(implication, morty.get_corner(UL))
        )
        self.play(self.change_students("thinking", "sassy", "erm"))
        self.look_at(self.screen)
        self.play(stds[1].change("pondering", self.screen))
        self.wait(4)


class NeedsMoreRigor(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        arrow = Vector(LEFT, thickness=4)
        arrow.move_to(2.5 * UP)
        words = Text("Sketchy and non-rigorous")
        words.next_to(arrow)

        self.play(
            morty.change("guilty"),
            self.change_students("erm", "hesitant", "sassy", look_at=self.screen)
        )
        self.play(
            GrowArrow(arrow),
            Write(words)
        )
        self.wait(4)


class CompareFilmTypes(InteractiveScene):
    def construct(self):
        # Test
        categories = VGroup(
            Text("Polaroid instant film"),
            Text("Microfilm"),
            Text("What we used (Bayfol HX200)"),
        )
        categories[2]["(Bayfol HX200)"].set_color(GREY_B)

        resolutions = VGroup(
            Text("10-20 lines per mm"),
            Text("100-200 lines per mm"),
            Text("> 5,000 lines per mm"),
        )

        arrows = Vector(RIGHT, thickness=4).replicate(3)

        last_group = VGroup()
        for text, res, arrow in zip(categories, resolutions, arrows):
            arrow = Vector(RIGHT)
            group = VGroup(text, arrow, res)
            group.arrange(RIGHT)
            res.align_to(text, UP)
            group.to_edge(UP)

            text.save_state()
            text.set_x(0)

            self.play(
                FadeOut(last_group, 0.5 * UP),
                FadeIn(text, 0.5 * UP),
            )
            self.wait()
            self.play(
                Restore(text),
                GrowArrow(arrow),
                FadeIn(res, RIGHT)
            )
            self.wait()

            last_group = group


class TheresADeeperIssue(TeacherStudentsScene):
    def construct(self):
        for std in self.students:
            std.change_mode("pondering")
            std.look_at(self.screen)
        self.play(
            self.teacher.says("There's a\\ndeeper issue"),
            self.change_students("erm", "guilty", "hesitant", look_at=self.teacher.eyes)
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "pondering", 'pondering', look_at=self.screen)
        )
        self.wait(2)


class SharpMindedViewer(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        randy = Randolph()
        randy.to_edge(LEFT)
        randy.shift(DOWN)

        self.play(randy.change("sassy", RIGHT))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused", RIGHT))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class WhatThisExplanationLacks(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Gaps in the single-point explanation", font_size=60)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        underline = Underline(title, buff=-0.05)

        self.play(
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline),
        )
        self.wait()

        # Add points
        points = BulletedList(
            "How do errors in the various approximations accumulate?",
            "Why don't higher order beams exist?",
            "What happens when you change the reference beam angle?",
            buff=0.5,
            font_size=48,
        )
        points.next_to(underline, DOWN, buff=0.5)
        points.to_edge(LEFT, buff=LARGE_BUFF)

        self.play(
            LaggedStartMap(FadeIn, points, shift=UP, lag_ratio=0.5)
        )
        self.wait()
        self.play(points[:2].animate.set_opacity(0.5))
        self.wait()


class SimplicityToGenerality(InteractiveScene):
    def construct(self):
        # Initialize little triangles
        frame = self.frame
        surface = Torus(resolution=(49, 49))
        surface.set_width(4)
        verts = surface.get_shader_data()['point']
        triangles = VGroup(
            Polygon(*tri)
            for tri in zip(verts[0::3], verts[1::3], verts[2::3])
        )
        triangles.set_stroke(WHITE, 0.5, 0.5)

        tri1 = triangles[3500]
        dists = [get_norm(tri.get_center() - tri1.get_center()) for tri in triangles]
        neighbors = VGroup(triangles[idx] for idx in np.argsort(dists))
        ng1 = neighbors[1:10]
        ng2 = neighbors[10:200]
        ng3 = neighbors[200:]
        for group in [ng1, ng2, ng3]:
            group.save_state()
            for tri in group:
                tri.become(tri1)
                tri.set_stroke(width=0.1, opacity=0.25)
        ng3.set_stroke(opacity=0.05)

        frame.reorient(4, 83, 0, (-0.11, -0.05, -0.34), 1.91)

        self.add(tri1)
        self.play(
            LaggedStart(
                Restore(ng1, lag_ratio=0.3),
                Restore(ng2, lag_ratio=0.02),
                Restore(ng3, lag_ratio=3e-4),
                lag_ratio=0.4
            ),
            frame.animate.reorient(29, 64, 0, (-0.15, 0.02, -0.43), 3.21),
            run_time=9,
        )
        self.add(triangles)

        # True surface
        equation = Tex(R"\\left(x^2+y^2+z^2+R^2-r^2\\right)^2=4 R^2\\left(x^2+y^2\\right)")
        equation.fix_in_frame()
        equation.to_edge(UP, buff=0.35)

        new_surface = ParametricSurface(
            lambda u, v: surface.uv_func(v, u),
            u_range=(0, TAU),
            v_range=(0, TAU),
            resolution=(250, 250)
        )
        new_surface.set_width(4)
        new_surface.set_opacity(0.5)
        new_surface.always_sort_to_camera(self.camera)

        triangles.shuffle()
        self.play(
            FadeOut(triangles, scale=0.8, lag_ratio=3e-4),
            frame.animate.reorient(77, 62, 0, (-0.27, -0.05, -0.23), 4.29),
            run_time=2
        )
        self.play(
            ShowCreation(new_surface, run_time=3),
            frame.animate.reorient(58, 66, 0, (-0.27, -0.05, -0.23), 4.29).set_anim_args(run_time=4),
        )
        self.play(frame.animate.reorient(50, 79, 0, (-0.27, -0.05, -0.23), 4.29), run_time=3)


class ModeledAs2D(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Modeled as 2D")
        words.to_edge(UP)
        point = 1.5 * RIGHT
        arrow = Arrow(words.get_right(), point, path_arc=-180 * DEGREES, thickness=4)

        self.play(
            FadeIn(words, 0.25 * UP),
            DrawBorderThenFill(arrow, run_time=2)
        )
        self.wait()


class ThinkingAboutRediscovery(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.body.insert_n_curves(100)
        randy.to_corner(DL)
        bubble = ThoughtBubble(filler_shape=(5, 2))
        bubble.pin_to(randy)
        bubble = bubble[0]
        bubble.set_fill(opacity=0)

        rect = FullScreenRectangle().set_fill(BLACK, 1)
        rect.append_points([rect.get_points()[-1], *bubble[-1].get_points()])
        self.add(rect, randy)

        self.play(
            randy.change('pondering'),
            Write(bubble, stroke_color=WHITE, lag_ratio=0.2)
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change('thinking', bubble.get_top()))
        self.play(Blink(randy))
        self.play(randy.change('tease', bubble.get_top()))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class IntroduceFormalSection(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(LaggedStart(
            stds[0].change("sassy", morty.eyes),
            stds[1].says("Okay but really, why\\ndoes it work?", look_at=morty.eyes),
            stds[2].change("angry", look_at=morty.eyes),
            morty.change("guilty"),
        ))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("tease"),
            stds[0].change("well"),
            stds[1].debubble(),
            stds[2].change("hesitant"),
            lag_ratio=0.5
        ))
        self.wait(2)

        # Show complex plane
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.faded_lines.set_stroke(BLUE_E, 1, 0.5)
        plane.axes.set_stroke(width=1)
        plane.background_lines.set_stroke(width=1)
        plane.set_width(3)
        plane.next_to(self.hold_up_spot, UP)

        a, b = (2, 1.5)
        z = complex(a, b)
        z_dot = GlowDot(plane.n2p(0), color=WHITE)
        z_dot.set_z_index(1)
        h_line = Line(plane.n2p(0), plane.n2p(a))
        v_line = Line(plane.n2p(a), plane.n2p(z))
        h_line.set_stroke(YELLOW, 3)
        v_line.set_stroke(RED, 3)
        a_label = Tex(R"a", font_size=24)
        a_label.match_color(h_line)
        a_label.next_to(h_line, DOWN, buff=0.1)
        b_label = Tex(R"bi", font_size=24)
        b_label.match_color(v_line)
        b_label.next_to(v_line, RIGHT, buff=0.1)
        z_label = Tex(R"a + bi", font_size=24)
        z_label.next_to(v_line.get_end(), UR, SMALL_BUFF)
        VGroup(a_label, b_label, z_label).set_backstroke(BLACK, 3)

        self.play(
            morty.change("raise_right_hand", plane),
            self.change_students(*3 * ["pondering"], look_at=plane),
            FadeIn(plane, UP),
        )
        self.play(
            ShowCreation(h_line),
            FadeIn(a_label, 0.5 * RIGHT),
            z_dot.animate.move_to(plane.n2p(a)),
        )
        self.play(
            ShowCreation(v_line),
            FadeIn(b_label, 0.5 * UP),
            z_dot.animate.move_to(plane.n2p(z)),
        )
        self.play(
            TransformMatchingShapes(VGroup(a_label, b_label).copy(), z_label)
        )
        self.play(self.change_students("erm", "thinking", "well", look_at=plane))
        self.wait(2)


class HoldUpEquation(InteractiveScene):
    def construct(self):
        pass


class GaborQuote(InteractiveScene):
    def construct(self):
        # Test
        quote = TexText(R"""
            \`\`This was an exercise\\\\
            in serendipity, the art of\\\\
            looking for something and\\\\
            finding something else''
        """, font_size=60, alignment="")
        quote.to_edge(RIGHT)

        self.play(Write(quote), run_time=2, lag_ratio=0.01)
        self.play(quote["looking for something"].animate.set_color(BLUE), lag_ratio=0.1)
        self.play(quote["finding something else"].animate.set_color(TEAL), lag_ratio=0.1)
        self.wait()


class GaborQuote2(InteractiveScene):
    def construct(self):
        # Test
        quote = TexText(R"""
            \`\`In holography, nature\\\\
            is on the inventor's side''
        """, font_size=60, alignment="")
        quote.to_edge(RIGHT)

        self.play(Write(quote), run_time=2, lag_ratio=0.1)
        self.wait()


class ComplexConjugateFact(InteractiveScene):
    def construct(self):
        # Show conjugate
        z = complex(2, 1)
        plane = ComplexPlane(
            (-6, 6), (-4, 4),
            faded_line_ratio=4,
            background_line_style=dict(
                stroke_color=BLUE_D,
                stroke_width=1,
            ),
            faded_line_style=dict(
                stroke_color=BLUE_E,
                stroke_width=1,
                stroke_opacity=0.5,
            ),
        )
        plane.set_height(7)
        plane.to_edge(DOWN, buff=0.25)
        plane.add_coordinate_labels(font_size=16)
        title = Text("Complex plane")
        title.next_to(plane, UP, buff=SMALL_BUFF)

        z_arrow = Arrow(plane.get_origin(), plane.n2p(z), buff=0)
        z_arrow.set_color(YELLOW)
        z_label = Tex(R"z = a + bi")
        z_label.next_to(z_arrow.get_end(), UP)

        z_conj = complex(z.real, -z.imag)
        z_conj_arrow = Arrow(plane.get_origin(), plane.n2p(z_conj), buff=0)
        z_conj_arrow.set_color(TEAL)
        z_conj_label = Tex("z^* = a - bi")
        z_conj_label.next_to(z_conj_arrow.get_end(), DOWN)

        conj_label = Text("Complex conjugate")
        conj_label.next_to(z_conj_label, DOWN, aligned_edge=LEFT)
        conj_label.set_backstroke(BLACK, 5)

        self.add(plane, title)
        self.add(z_arrow, z_label)
        self.wait()
        self.play(Write(conj_label, stroke_color=WHITE))
        self.play(
            TransformFromCopy(z_arrow, z_conj_arrow, path_arc=-PI / 2),
            TransformMatchingStrings(z_label.copy(), z_conj_label, path_arc=-PI / 2),
            run_time=2
        )
        self.wait()

        # Show product
        product_arrow = Arrow(plane.get_origin(), plane.n2p(z * z_conj), buff=0)
        product_arrow.set_color(WHITE)
        product_label = Tex(R"z \\cdot z^* = |z|^2")
        product_label.next_to(product_arrow.get_end(), UP)
        product_label.shift(0.75 * RIGHT)

        self.play(
            TransformFromCopy(z_label[0], product_label[R"z \\cdot"][0]),
            TransformFromCopy(z_conj_label[:2], product_label["z^*"][0]),
            TransformFromCopy(z_arrow, product_arrow),
            TransformFromCopy(z_conj_arrow, product_arrow),
        )
        self.play(Write(product_label["= |z|^2"][0]))
        self.wait()


class PrepareComplexAlgebra(InteractiveScene):
    def construct(self):
        # Test
        lines = VGroup(
            Tex(R"R \\cdot (1 - \\text{Opacity})"),
            Tex(R"R \\cdot (1 - c|R + O|^2)"),
            Tex(R"R - c R \\cdot |R + O|^2"),
        )
        lines.arrange(DOWN, buff=0.75)
        lines.to_edge(UP)

        label = Text("Wave beyond\\nthe film")
        arrow = Vector(LEFT)
        arrow.next_to(lines[0], RIGHT)
        label.next_to(arrow, RIGHT)

        rect = SurroundingRectangle(lines[2][R"R \\cdot |R + O|^2"], buff=0.05)
        rect.set_stroke(TEAL, 2)

        self.play(
            FadeIn(label, lag_ratio=0.1),
            GrowArrow(arrow),
            FadeIn(lines[0], LEFT),
        )
        self.wait()
        self.play(
            TransformMatchingStrings(lines[0].copy(), lines[1]),
            run_time=2
        )
        self.wait()
        self.play(
            TransformMatchingStrings(lines[1].copy(), lines[2]),
            run_time=2
        )
        self.wait()
        self.play(
            ShowCreation(rect),
            lines[:2].animate.set_opacity(0.5),
            lines[2][R"R - c"].animate.set_opacity(0.5),
        )
        self.wait()


class ComplexAlgebra(InteractiveScene):
    def construct(self):
        # Add first lines
        lines = VGroup(
            Tex(R"R \\cdot |R + O|^2"),
            Tex(R"R \\cdot (R + O)(R^* + O^*)"),
            Tex(R"R \\cdot R \\cdot R^* + R \\cdot R^* \\cdot O  + R \\cdot R \\cdot O^*+ R \\cdot O \\cdot O^*"),
            Tex(R"\\left(|R|^2 + |O|^2 \\right) \\cdot R + |R|^2 \\cdot O + R^2 \\cdot O^*"),
        )
        lines[2].scale(0.75)
        lines.arrange(DOWN, buff=LARGE_BUFF)
        lines.to_edge(UP)

        label = Text("Part of the wave\\nbeyond the film", font_size=36)
        arrow = Vector(LEFT)
        arrow.next_to(lines[0], RIGHT)
        label.next_to(arrow, RIGHT)
        label.shift_onto_screen()

        self.add(lines[0])
        self.play(
            Write(label),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(lines[0][:-1], lines[1][R"R \\cdot (R + O)"][0]),
            TransformFromCopy(lines[0]["|R + O|"][0].copy(), lines[1]["(R^* + O^*)"][0]),
            lag_ratio=0.1,
            run_time=2
        ))
        self.wait()

        # FOIL expansion
        R0 = lines[1]["R"][0]
        R = lines[1]["R"][1]
        O = lines[1]["O"][0]
        Rc = lines[1]["R^*"][0]
        Oc = lines[1]["O^*"][0]
        l1_groups = [
            VGroup(R0, R, Rc),
            VGroup(R0, O, Rc),
            VGroup(R0, R, Oc),
            VGroup(R0, O, Oc),
        ]
        l2_groups = [
            lines[2][substr]
            for substr in [
                R"R \\cdot R \\cdot R^*",
                R"R \\cdot R^* \\cdot O",
                R"R \\cdot R \\cdot O^*",
                R"R \\cdot O \\cdot O^*",
            ]
        ]
        plusses = lines[2]["+"]
        plusses.add_to_back(VectorizedPoint())

        pre_rects = VGroup(map(self.get_term_rect, l1_groups[0]))
        post_rect = self.get_term_rect(l2_groups[0])
        VGroup(pre_rects, post_rect).set_stroke(width=0, opacity=0)
        self.add(pre_rects, post_rect)

        for l1_group, l2_group, plus in zip(l1_groups, l2_groups, plusses):
            self.play(
                Transform(pre_rects, VGroup(map(self.get_term_rect, l1_group))),
                Transform(post_rect, self.get_term_rect(l2_group)),
                FadeIn(l2_group),
                FadeIn(plus),
            )
            self.wait(0.5)

        self.add(lines[2])
        self.play(
            FadeOut(pre_rects),
            FadeOut(post_rect),
        )
        self.wait()

        # Highlight conjugate pairs
        pair_rects = VGroup(
            self.get_term_rect(lines[2][R"R \\cdot R^*"][0]),
            self.get_term_rect(lines[2][R"R \\cdot R^*"][1]),
            self.get_term_rect(lines[2][R"O \\cdot O^*"]),
        )
        pair_rects.set_stroke(RED, 2)

        real_label = Text("Real numbers")
        real_label.next_to(lines[2], DOWN, buff=1.5)
        real_label.set_color(RED)
        arrows = VGroup(
            Arrow(real_label, rect.get_bottom())
            for rect in pair_rects
        )
        arrows.set_color(RED)

        self.play(ShowCreation(pair_rects, lag_ratio=0.1))
        self.play(
            FadeIn(real_label, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows)
        )
        self.wait()
        self.play(
            FadeOut(real_label),
            FadeOut(arrows),
            FadeOut(pair_rects),
        )
        self.wait()

        # Organize into the last line
        lines[3].next_to(lines[2], DOWN, buff=1.5)
        lines[3].set_opacity(1)
        l3_groups = [
            lines[3][R"\\left(|R|^2 + |O|^2 \\right) \\cdot R"],
            lines[3][R"|R|^2 \\cdot O"],
            lines[3][R"R^2 \\cdot O^*"],
        ]
        l2_plusses = lines[2]["+"]
        l3_plusses = lines[3]["+"]

        l2_rects = VGroup(map(self.get_term_rect, l2_groups))
        l3_rects = VGroup(map(self.get_term_rect, l3_groups))
        l3_rects[2].match_height(l3_rects, stretch=True, about_edge=UP)

        box1_lines = VGroup(
            self.connecting_line(l2_rects[0], l3_rects[0]),
            self.connecting_line(l2_rects[3], l3_rects[0]),
        )
        box2_line = self.connecting_line(l2_rects[1], l3_rects[1])
        box3_line = self.connecting_line(l2_rects[2], l3_rects[2])

        real_brace1 = Brace(lines[3][R"\\left(|R|^2 + |O|^2 \\right)"], DOWN)
        real_brace2 = Brace(lines[3][R"|R|^2"][1], DOWN)
        real_label = real_brace1.get_text("Some real number", font_size=36)

        fade_opacity = 0.5

        self.play(
            ShowCreation(box1_lines, lag_ratio=0.5),
            FadeIn(l2_rects[0]),
            FadeIn(l2_rects[3]),
            FadeTransform(l2_groups[0].copy(), l3_groups[0], time_span=(0.5, 2)),
            FadeTransform(l2_groups[3].copy(), l3_groups[0], time_span=(1, 2)),
            FadeIn(l3_rects[0], time_span=(1, 2)),
            l2_groups[1].animate.set_opacity(fade_opacity),
            l2_groups[2].animate.set_opacity(fade_opacity),
            l2_plusses.animate.set_opacity(fade_opacity),
            run_time=2
        )
        self.wait()
        self.play(
            GrowFromCenter(real_brace1),
            FadeIn(real_label, shift=0.25 * DOWN)
        )
        self.wait()
        self.play(
            box1_lines.animate.set_stroke(WHITE, 1, 0.5),
            VGroup(l2_rects[0], l2_rects[3], l3_rects[0]).animate.set_stroke(GREY, 1, 0.5),
            l2_groups[1].animate.set_opacity(1),
            l2_groups[0].animate.set_opacity(fade_opacity),
            l2_groups[3].animate.set_opacity(fade_opacity),
            l3_groups[0].animate.set_opacity(fade_opacity),
            FadeOut(real_brace1),
            real_label.animate.set_opacity(fade_opacity),
        )
        self.play(
            FadeIn(l2_rects[1]),
            ShowCreation(box2_line),
            FadeIn(l3_rects[1], time_span=(0.5, 1.5)),
            TransformMatchingShapes(l2_groups[1].copy(), l3_groups[1], run_time=1),
            FadeIn(l3_plusses[1]),
        )
        self.wait()
        self.play(
            GrowFromCenter(real_brace2),
            real_label.animate.next_to(real_brace2, DOWN).set_opacity(1)
        )
        self.wait()
        self.play(
            box2_line.animate.set_stroke(WHITE, 1, 0.5),
            VGroup(l2_rects[1], l3_rects[1]).animate.set_stroke(GREY, 1, 0.5),
            l2_groups[2].animate.set_opacity(1),
            l2_groups[1].animate.set_opacity(fade_opacity),
            l3_groups[1].animate.set_opacity(fade_opacity),
            l3_plusses[1].animate.set_opacity(fade_opacity),
            FadeOut(real_brace2),
            FadeOut(real_label),
        )
        self.play(
            FadeTransform(l2_groups[2].copy(), l3_groups[2], run_time=1.5),
            ShowCreation(box3_line, run_time=1.5),
            FadeIn(l2_rects[1]),
            FadeIn(l3_rects[2], time_span=(0.5, 1.5)),
            FadeIn(l3_plusses[2])
        )
        self.wait()

        # Bring back
        self.play(
            FadeOut(VGroup(box1_lines, box2_line, box3_line)),
            FadeOut(l2_rects),
            l3_rects.animate.set_stroke(TEAL, 1, 1),
            lines[2].animate.set_opacity(0.5),
            lines[3].animate.set_opacity(1),
        )
        self.wait()

    def get_term_rect(self, term):
        rect = SurroundingRectangle(term)
        rect.round_corners()
        rect.set_stroke(TEAL, 2)
        return rect

    def connecting_line(self, high_box, low_box):
        return CubicBezier(
            high_box.get_bottom(),
            high_box.get_bottom() + 1.0 * DOWN,
            low_box.get_top() + 1.0 * UP,
            low_box.get_top(),
        ).set_stroke(WHITE, 2)


class PlaneAfterFilm(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("FilmFromBehind")
        image.set_height(FRAME_HEIGHT)
        image.set_height(FRAME_HEIGHT)
        image.fix_in_frame()
        # self.add(image)

        rect = Polygon(
            (2.56, -0.12, 0.0),
            (5.69, -2.51, 0.0),
            (6.25, 1.31, 0.0),
            (2.72, 2.62, 0.0),
        )
        rect.set_fill(BLACK, 0.75)
        rect.set_stroke(BLUE, 10)
        rect.fix_in_frame()
        words = Text("True on this\\n2D Plane")

        self.set_floor_plane("xz")
        self.frame.reorient(73, -24, 0, (-1.53, -0.68, 4.14), 9.19)

        self.play(
            DrawBorderThenFill(rect),
            FadeIn(words),
        )
        self.wait()


class LookingAtScreen(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.change_students("confused", "pondering", "maybe", look_at=self.screen),
            self.teacher.change("well")
        )
        self.wait(5)


class EndScreen(PatreonEndScreen):
    pass


# Old stubs

class DoubleSlitSupplementaryGraphs(InteractiveScene):
    def construct(self):
        # Setup all three axes, with labels

        # Show constructive interference

        # Show destructive interference
        ...


class DistApproximations(InteractiveScene):
    def construct(self):
        # Show sqrt(L^2 + x^2) approx L + x/(2L) approx L
        pass


class DiffractionEquation(InteractiveScene):
    def construct(self):
        # Add equation
        equation = Tex(R"{d} \\cdot \\sin(\\theta) = \\lambda", font_size=60)
        equation.set_backstroke(BLACK)
        arrow = Vector(DOWN, thickness=4)
        arrow.set_color(BLUE)

        d, theta, lam = syms = [equation[s][0] for s in [R"{d}", R"\\theta", R"\\lambda"]]
        colors = [BLUE, YELLOW, TEAL]

        arrow.next_to(d, UP, LARGE_BUFF)
        arrow.set_fill(opacity=0)

        self.add(equation)

        for sym, color in zip(syms, colors):
            self.play(
                FlashAround(sym, color=color, time_span=(0.25, 1.25)),
                sym.animate.set_fill(color),
                arrow.animate.next_to(sym, UP).set_fill(color, 1),
            )
            self.wait()
        self.play(FadeOut(arrow))`,
    annotations: {
      4: "PhotographVsHologram extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      37: "GlintArrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      38: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      52: "WriteHologram extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      53: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      61: "WriteTransmissionHologram extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      62: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      70: "WriteWhiteLightReflectionHologram extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      71: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      79: "IndicationOnPhotograph extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      80: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      117: "ContrastPhotographyAndHolography extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      118: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      151: "Outline extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      152: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      348: "Class GoalOfRediscovery inherits from TeacherStudentsScene.",
      349: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      367: "NameCraigAndSally extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      368: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      398: "ShowALens extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      399: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      436: "AskAboutRecordingPhase extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      437: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      477: "Class HowIsThisHelpful inherits from TeacherStudentsScene.",
      478: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      495: "NoWhiteLight extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      496: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      524: "AnnotateSetup extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      525: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      607: "ArrowWithQMark extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      608: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      621: "ProblemSolvingTipNumber1 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      622: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      639: "NameElectromagneticField extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      640: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      668: "Class HoldUpDiffractionEquationInAnticipation inherits from TeacherStudentsScene.",
      669: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      702: "Class HowLongIsThatSnippet inherits from TeacherStudentsScene.",
      703: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      711: "Class HoldUpDiffractionEquation inherits from TeacherStudentsScene.",
      712: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      744: "AskAboutCenterBeam extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      745: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      757: "DistanceApproximation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      758: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      783: "CircleDiffractionEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      784: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      806: "DiffractionGratingGreenLaserExampleNumbers extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      807: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      826: "AnnotateZerothOrderBeam2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      827: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      879: "ExactSpacingQuestion extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      880: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      897: "Class DejaVu inherits from TeacherStudentsScene.",
      898: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      920: "WhatAHologramReconstructs extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      921: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      959: "Class AskAboutHigherOrderBeams inherits from TeacherStudentsScene.",
      960: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      974: "BinaryVsSinusoidalDiffraction extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      975: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1068: "Class AskWhyForSinusoidalGratingFact inherits from TeacherStudentsScene.",
      1069: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1101: "Class NeedsMoreRigor inherits from TeacherStudentsScene.",
      1102: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1121: "CompareFilmTypes extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1122: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1165: "Class TheresADeeperIssue inherits from TeacherStudentsScene.",
      1166: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1181: "SharpMindedViewer extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1182: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1198: "WhatThisExplanationLacks extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1199: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1230: "SimplicityToGenerality extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1231: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1299: "ModeledAs2D extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1300: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1314: "ThinkingAboutRediscovery extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1315: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1343: "Class IntroduceFormalSection inherits from TeacherStudentsScene.",
      1344: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1413: "HoldUpEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1414: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1418: "GaborQuote extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1419: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1435: "GaborQuote2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1436: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1448: "ComplexConjugateFact extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1449: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1514: "PrepareComplexAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1515: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1557: "ComplexAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1558: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1778: "PlaneAfterFilm extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1779: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1808: "Class LookingAtScreen inherits from TeacherStudentsScene.",
      1809: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1817: "Class EndScreen inherits from PatreonEndScreen.",
      1823: "DoubleSlitSupplementaryGraphs extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1824: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1833: "DistApproximations extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1834: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1839: "DiffractionEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1840: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();