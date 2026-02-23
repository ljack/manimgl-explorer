(function() {
  const files = window.MANIM_DATA.files;

  files["_2025/colliding_blocks_v2/blocks.py"] = {
    description: "Main colliding blocks simulation: elastic collisions between blocks of mass ratio 100^n compute digits of pi. Includes physics engine, phase space visualization, and the connection to billiards in a circle.",
    code: `from manim_imports_ext import *


LITTLE_BLOCK_COLOR = "#51463E"


class StateTracker(ValueTracker):
    """
    Tracks the state of the block collision process as a 4d vector
    [
        x1 * sqrt(m1),
        x2 * sqrt(m2),
        v1 * sqrt(m1),
        v2 * sqrt(m2),
    ]
    """

    def __init__(self, blocks, initial_positions=[8, 5], initial_velocities=[-1, 0]):
        sqrt_m1, sqrt_m2 = self.sqrt_mass_vect = np.sqrt([b.mass for b in blocks])
        self.theta = math.atan2(sqrt_m2, sqrt_m1)

        self.state0 = np.array([
            *np.array(initial_positions) * self.sqrt_mass_vect,
            *np.array(initial_velocities) * self.sqrt_mass_vect,
        ])

        super().__init__(self.state0.copy())

    def set_time(self, t):
        pos0 = self.state0[0:2]
        vel0 = self.state0[2:4]

        self.set_value([*(pos0 + t * vel0), *vel0])

    def reflect_vect(self, vect):
        n_reflections = self.get_n_collisions()
        rot_angle = -2 * self.theta * ((n_reflections + 1) // 2)
        result = rotate_vector_2d(vect, rot_angle)
        result[1] *= (-1)**(n_reflections % 2)
        return result

    def get_block_positions(self):
        scaled_pos = self.get_value()[0:2]
        norm = get_norm(scaled_pos)

        rot_scaled_pos = self.reflect_vect(scaled_pos)
        return rot_scaled_pos / self.sqrt_mass_vect

    def get_scaled_block_velocities(self):
        return self.reflect_vect(self.get_value()[2:4])

    def get_block_velocities(self):
        return self.get_scaled_block_velocities() / self.sqrt_mass_vect

    def get_kinetic_energy(self):
        v1, v2 = self.get_value()[2:4]
        return v1**2 + v2**2

    def get_momentum(self):
        v1, v2 = self.get_block_velocities()
        m1, m2 = self.sqrt_mass_vect**2
        return m1 * v1 + m2 * v2

    def get_n_collisions(self):
        state = self.get_value()  # First two coords are [sqrt(m1) * d1, sqrt(m2) * d2]
        angle = math.atan2(state[1], state[0])
        return int(angle / self.theta)

    def time_from_last_collision(self):
        state = self.get_value()
        angle = math.atan2(state[1], state[0])
        n_collisions = int(angle / self.theta)
        if n_collisions == 0:
            return 0
        collisions_angle = n_collisions * self.theta
        rot_state = rotate_vector_2d(state[:2], collisions_angle - angle)
        collision_state = rot_state * (state[1] / rot_state[1])  # Rescale to match y coordinate
        return (collision_state[0] - state[0]) / state[2]


class ClackAnimation(Restore):
    def __init__(self, point, run_time=0.25, **kwargs):
        circles = Circle(radius=0.25).replicate(3)
        circles.move_to(point)
        circles.set_stroke(YELLOW, 0, 0)
        circles.save_state()
        circles.scale(0.1)
        circles.set_stroke(YELLOW, 2, 1)

        super().__init__(circles, lag_ratio=0.1, run_time=run_time, **kwargs)


class Blocks(InteractiveScene):
    initial_positions = [10.5, 8]
    initial_velocities = [-0.975, 0]
    masses = [10, 1]
    widths = [1.0, 0.5]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    three_d = True
    clack_sound = "clack.wav"
    reverse_clack_sound = "reverse_clack.wav"
    floor_width = 13
    floor_depth = 1.5
    wall_height = 2
    block_shading = [0.5, 0.5, 0]
    field_of_view = 45 * DEG

    def setup(self):
        super().setup()

        self.set_floor_plane("xz")
        self.frame.set_field_of_view(self.field_of_view)

        self.floor_wall = self.get_floor_and_wall_3d() if self.three_d else self.get_floor_and_wall_2d()
        self.floor, self.wall = self.floor_wall
        self.blocks = self.get_block_pair(self.floor, three_d=self.three_d)

        self.state_tracker = StateTracker(self.blocks, self.initial_positions, self.initial_velocities)
        self.time_tracker = ValueTracker(0)
        self.state_tracker.f_always.set_time(self.time_tracker.get_value)

        self.bind_blocks_to_state(self.blocks, self.floor, self.state_tracker)

        self.listen_for_clacks = True
        self.last_collision_count = self.state_tracker.get_n_collisions()
        self.last_clack_time = 0
        self.flashes = []

        self.add(self.state_tracker)
        self.add(self.time_tracker)
        self.add(self.floor_wall)
        self.add(self.blocks)

    def construct(self):
        # Setup the experiment
        light_source = self.camera.light_source
        light_source.set_z(3)
        frame = self.frame
        self.set_floor_plane("xz")

        blocks = self.blocks
        floor = self.floor
        state_tracker = self.state_tracker
        time_tracker = self.time_tracker

        for block in blocks:
            block.mass_label.set_opacity(0)

        # Set up equations
        kw = dict(t2c={
            "m_1": BLUE,
            "m_2": BLUE,
            "v_1": RED,
            "v_2": RED,
        })
        ke_equation = Tex(R"\\frac{1}{2} m_1 (v_1)^2 + \\frac{1}{2}m_2 (v_2)^2 = \\text{E}", **kw)
        p_equation = Tex(R"m_1 v_1 + m_2 v_2 = \\text{P}", **kw)
        equations = VGroup(ke_equation, p_equation)
        equations.arrange(DOWN, buff=0.75)
        equations.to_edge(UP, buff=0.5)
        equations.fix_in_frame()

        # Introductory shot
        frame.reorient(-42, -3, 0, (-2.08, -2.25, -1.87), 6.95)
        stop_time = 12
        self.play(
            frame.animate.to_default_state(),
            FadeIn(ke_equation, shift=0.5 * UP, time_span=(7, 8)),
            FadeIn(p_equation, shift=0.5 * UP, time_span=(10, 11)),
            time_tracker.animate.set_value(stop_time).set_anim_args(rate_func=linear),
            run_time=stop_time
        )
        self.wait()

        self.add(equations)

        # Add velocity vectors
        vel = state_tracker.get_block_velocities()
        marked_velocity = ValueTracker(vel)
        marked_velocity.f_always.set_value(state_tracker.get_block_velocities)
        self.add(marked_velocity)
        velocity_vectors = VGroup(
            self.get_velocity_vector(blocks[0], lambda: marked_velocity.get_value()[0]),
            self.get_velocity_vector(blocks[1], lambda: marked_velocity.get_value()[1]),
        )
        velocity_vectors.update()
        velocity_vectors.set_z_index(1)

        # Highlight kinetic energy of each one
        self.play(
            frame.animate.reorient(-12, 3, 0, (-3.41, -1.48, 0.01), 4.63),
            blocks[0].mass_label.animate.set_opacity(1),
            blocks[1].mass_label.animate.set_opacity(1),
            VFadeIn(velocity_vectors),
            FadeOut(p_equation, time_span=(1, 2)),
            run_time=3
        )
        self.wait()

        # Highlight corresponding terms
        tex = ["m_1", "v_1", "m_2", "v_2"]
        labels = [
            blocks[0][2],
            velocity_vectors[0][1],
            blocks[1][2],
            velocity_vectors[1][1],
        ]

        last_rects = VGroup()
        for tex, label in zip(tex, labels):
            rects = VGroup(
                SurroundingRectangle(ke_equation[tex], buff=0.05),
                SurroundingRectangle(label, buff=0.05),
            )
            rects[0].set_stroke(YELLOW, 5)
            rects[1].set_stroke(YELLOW, 3)
            self.play(
                FadeOut(last_rects),
                ShowCreation(rects, lag_ratio=0),
            )
            self.wait()
            last_rects = rects
        self.play(FadeOut(last_rects))
        self.wait()

        # Show constant energy
        dec_equation = Tex(R"\\frac{1}{2}(10)(+0.00)^2 + \\frac{1}{2}(1)(+0.00)^2 = +0.00", font_size=42)
        dec_equation.next_to(ke_equation, DOWN, LARGE_BUFF)

        terms = dec_equation.make_number_changeable("+0.00", replace_all=True, include_sign=True)
        dec_equation.fix_in_frame()
        dec_equation["(1)"].set_color(BLUE)
        dec_equation["(10)"].set_color(BLUE)
        terms[:2].set_color(RED)
        terms[0].f_always.set_value(lambda: state_tracker.get_block_velocities()[0])
        terms[1].f_always.set_value(lambda: state_tracker.get_block_velocities()[1])
        terms[2].set_value(state_tracker.get_kinetic_energy())
        terms.update()

        label_targets = VGroup(dec_equation["(10)"], terms[0], dec_equation["(1)"], terms[1]).copy()
        label_targets.clear_updaters()

        self.play(
            VFadeIn(dec_equation, time_span=(1.5, 2)),
            Transform(VGroup(*labels).copy().clear_updaters(), label_targets, remover=True, run_time=2, lag_ratio=1e-3),
        )
        self.wait()
        self.play(time_tracker.animate.increment_value(1), rate_func=there_and_back, run_time=6)
        self.wait()
        self.play(
            time_tracker.animate.increment_value(10).set_anim_args(rate_func=linear),
            frame.animate.reorient(-1, 3, 0, (-1.59, -0.79, 0.17), 6.46),
            run_time=10,
            rate_func=linear
        )
        dec_equation.clear_updaters()
        self.play(
            time_tracker.animate.set_value(12),
            frame.animate.reorient(-12, 3, 0, (-3.41, -1.48, 0.01), 4.63),
            FadeOut(dec_equation),
            run_time=2,
        )
        self.wait()

        # Highlight momentum
        self.play(
            ke_equation.animate.set_opacity(0.2),
            FadeIn(p_equation),
            time_tracker.animate.increment_value(-1),
        )
        self.wait()

        # Show the net momentum calculation (much copying here)
        p_dec_equation = Tex(R"(10)(+0.00) + (1)(+0.00) = +0.00", font_size=42)
        p_terms = p_dec_equation.make_number_changeable("+0.00", replace_all=True, include_sign=True)
        p_terms[:2].set_color(RED)
        p_dec_equation["(1)"].set_color(BLUE)
        p_dec_equation["(10)"].set_color(BLUE)
        p_dec_equation.fix_in_frame()
        p_dec_equation.next_to(p_equation, DOWN, buff=0.75)

        p_terms[0].f_always.set_value(lambda: state_tracker.get_block_velocities()[0])
        p_terms[1].f_always.set_value(lambda: state_tracker.get_block_velocities()[1])
        p_terms[2].f_always.set_value(state_tracker.get_momentum)

        label_targets = VGroup(p_dec_equation["(10)"], p_terms[0], p_dec_equation["(1)"], p_terms[1]).copy()
        label_targets.clear_updaters()

        self.play(
            frame.animate.reorient(-2, 2, 0, (-2.82, -0.66, 0.11), 5.96),
            VFadeIn(p_dec_equation),
            Transform(VGroup(*labels).copy().clear_updaters(), label_targets, remover=True),
            run_time=2
        )
        self.play(
            time_tracker.animate.increment_value(8).set_anim_args(rate_func=linear),
            frame.animate.reorient(6, -1, 0, (-0.7, -0.25, -0.17), 7.75),
            run_time=16,
        )
        self.wait()

        # Reset
        marked_velocity.clear_updaters()
        blocks.clear_updaters()
        self.remove(state_tracker)

        state_tracker = StateTracker(blocks, [9.5, 8], [-1, 0])
        time_tracker.set_value(0)
        state_tracker.add_updater(lambda m: m.set_time(time_tracker.get_value()))
        self.state_tracker = state_tracker
        self.bind_blocks_to_state(blocks, floor, state_tracker)
        self.add(state_tracker)

        p_dec_equation.clear_updaters()
        self.play(
            FadeOut(p_dec_equation, 2 * LEFT),
            frame.animate.to_default_state(),
            equations.animate.set_width(4.5).arrange(DOWN, buff=0.5).set_opacity(1).to_corner(UL),
            run_time=2
        )
        self.wait()

        # Highlight velocity terms
        v_terms = VGroup(
            eq[f"v_{n}"]
            for eq in equations
            for n in [1, 2]
        )
        v_point = Point([*marked_velocity.get_value(), 0])
        v_point.target = v_point.generate_target()
        v_point.target.move_to([*state_tracker.get_block_velocities(), 0])

        self.play(
            LaggedStartMap(FlashAround, v_terms, lag_ratio=0.1, time_width=1.0, time_span=(0, 2)),
            MoveToTarget(v_point, path_arc=PI),
            UpdateFromFunc(marked_velocity, lambda m: m.set_value(v_point.get_center()[:2])),
            run_time=3,
        )
        self.wait()

        # Introduce coordinate plane
        plane = NumberPlane((-4, 4, 1), (-4, 4, 1), faded_line_ratio=1)
        plane.set_height(5)
        plane.to_corner(UR, buff=0.25)
        plane.shift(2.0 * LEFT)
        plane.axes.set_stroke(WHITE, 1)
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        state_point = Group(
            TrueDot(radius=0.05).make_3d(),
            GlowDot(radius=0.2),
        )
        state_point.set_color(RED)
        state_point.add_updater(lambda m: m.move_to(plane.c2p(*marked_velocity.get_value())))

        kw["font_size"] = 36
        x_label, y_label = axis_labels = VGroup(
            Tex("x = v_1", **kw),
            Tex("y = v_2", **kw),
        )
        x_label.next_to(plane.x_axis.get_right(), UR, SMALL_BUFF)
        x_label.shift_onto_screen(buff=SMALL_BUFF)
        y_label.next_to(plane.y_axis.get_top(), DR, SMALL_BUFF)

        self.play(ShowCreation(plane, lag_ratio=2e-2))
        self.play(
            Write(x_label[:2]),
            TransformFromCopy(ke_equation["v_1"][0], x_label["v_1"][0]),
            FadeIn(state_point),
        )
        self.add(x_label)
        self.play(
            marked_velocity.animate.set_value([3, 0]),
            run_time=3,
            rate_func=there_and_back,
        )
        self.wait()
        self.play(
            Write(y_label[:2]),
            TransformFromCopy(ke_equation["v_2"][0], y_label["v_2"][0]),
        )
        self.add(y_label)
        self.play(
            marked_velocity.animate.set_value([-1, 3]),
            run_time=5,
            rate_func=wiggle,
        )
        self.wait()

        # Let it play forward
        marked_velocity.add_updater(lambda m: m.set_value(state_tracker.get_block_velocities()))
        self.add(state_tracker, marked_velocity)

        self.play(
            time_tracker.animate.set_value(10),
            run_time=10,
            rate_func=linear,
        )

        # Show coordinates
        coord_label = Tex(R"(+0.00, +0.00)", font_size=36)
        coord_label.set_color(RED)
        terms = coord_label.make_number_changeable("+0.00", replace_all=True, include_sign=True)
        for term, v in zip(terms, state_tracker.get_block_velocities()):
            term.set_value(v)
        coord_label.next_to(state_point, UR, buff=0)

        self.play(LaggedStart(
            TransformFromCopy(velocity_vectors[0][1].copy().clear_updaters(), terms[0]),
            TransformFromCopy(velocity_vectors[1][1].copy().clear_updaters(), terms[1]),
            FadeIn(coord_label[0::2]),
            lag_ratio=0.5,
        ))
        self.wait()

        # Insertion
        if False:
            # This is just for a certain scene we're plugging in,
            # not meant to be part of the main scene
            self.remove(equations)
            self.frame.set_field_of_view(1 * DEG)
            self.camera.light_source.set_z(3)

            blocks.clear_updaters()
            blocks.scale(1.2)
            velocity_vectors.scale(0.8)
            blocks.arrange(LEFT, buff=LARGE_BUFF, aligned_edge=DOWN)
            blocks[0][0].set_opacity(0.5)
            blocks[1][0].set_opacity(0.5)
            blocks.next_to(plane, LEFT, buff=LARGE_BUFF)

            marked_velocity.clear_updaters()

            # Many random points
            for _ in range(15):
                coords = np.random.uniform(-4, 4, 2)
                self.play(marked_velocity.animate.set_value(coords))
                self.wait()

        # Back up and add tail (Maybe just fade to here)
        self.play(
            FadeOut(coord_label),
            time_tracker.animate.set_value(0).set_anim_args(run_time=2)
        )

        traced_path = self.get_traced_path(state_point)
        self.add(traced_path)

        self.play(
            time_tracker.animate.set_value(45),
            run_time=25,
            rate_func=linear,
        )
        traced_path.clear_updaters()

        # Highlight the conservation of energy
        ke_rect = SurroundingRectangle(ke_equation).set_stroke(YELLOW, 2)
        question = Text("How would you plot this?", font_size=36)
        question.next_to(ke_rect, DOWN)
        question.set_color(YELLOW)

        self.play(
            FadeOut(p_equation),
            ShowCreation(ke_rect),
            Write(question, run_time=1),
        )
        self.wait()
        time_tracker.set_value(0)
        blocks.update()

        # Show xy equation
        xy_equation = Tex(R"5.00 {x}^2 + 0.50 {y}^2 = \\text{E}", font_size=36)
        xy_equation["{x}"].set_color(RED)
        xy_equation["{y}"].set_color(RED)
        xy_equation.next_to(question, DOWN, buff=MED_LARGE_BUFF)
        x_coef = xy_equation.make_number_changeable("5.00")
        y_coef = xy_equation.make_number_changeable("0.50")

        ellipse = Circle(radius=plane.x_axis.get_unit_size())
        ellipse.set_stroke(YELLOW, 2)
        ellipse.stretch(math.sqrt(10), 1)
        ellipse.move_to(plane.c2p(0, 0))

        state_point.set_z_index(1)

        self.play(FadeIn(xy_equation, DOWN))
        self.wait()
        self.play(
            traced_path.animate.set_stroke(WHITE, 1, 0.25),
            ShowCreation(ellipse),
            run_time=3
        )
        self.wait()

        # Emphasize squish
        self.play(
            FadeOut(state_point),
            FadeOut(traced_path),
            FadeOut(ke_rect),
            FadeOut(question),
            VGroup(
                ke_equation, xy_equation
            ).animate.arrange(DOWN, buff=0.5).next_to(plane, LEFT, buff=LARGE_BUFF),
        )
        underlines = VGroup(
            Underline(x_coef, buff=SMALL_BUFF),
            Underline(ke_equation["m_1"], buff=SMALL_BUFF),
        )
        underlines.set_color(YELLOW)

        l_squish_arrows = Vector(0.75 * RIGHT).get_grid(4, 1, v_buff=0.75)
        l_squish_arrows.next_to(plane.get_origin(), LEFT, buff=0.75)
        r_squish_arrows = l_squish_arrows.copy().flip(about_point=plane.get_origin())
        squish_arrows = VGroup(*l_squish_arrows, *r_squish_arrows)

        self.play(LaggedStartMap(GrowArrow, squish_arrows, lag_ratio=1e-2, run_time=1))
        self.play(
            ShowCreation(underlines, lag_ratio=0),
        )
        self.wait()
        self.play(FadeOut(squish_arrows), FadeOut(underlines))

        # Show the size of the ellipse
        marked_velocity.suspend_updating()
        E_rects = VGroup(
            SurroundingRectangle(ke_equation[R"\\text{E}"], buff=SMALL_BUFF),
            SurroundingRectangle(xy_equation[R"\\text{E}"], buff=SMALL_BUFF),
        )
        E_rects.set_stroke(YELLOW, 1)
        self.play(ShowCreation(E_rects, lag_ratio=0))
        self.play(
            marked_velocity.animate.set_value([-0.5, 0]),
            ellipse.animate.scale(0.25),
            rate_func=there_and_back,
            run_time=8,
        )
        self.play(FadeOut(E_rects))
        marked_velocity.resume_updating()

        # Show the experiment play out one more time
        traced_path = self.get_traced_path(state_point)
        self.add(traced_path, state_point)

        self.play(
            time_tracker.animate.set_value(20),
            run_time=20,
            rate_func=linear,
        )
        traced_path.clear_updaters()

        # Stretch into a circle
        state_point.clear_updaters()
        new_ke_eq = Tex(
            R"\\left(\\sqrt{m_1} \\cdot v_1\\right)^2 + \\left(\\sqrt{m_2} \\cdot v_2\\right)^2 = 2 \\text{E}",
            **kw
        )
        new_xy_eq = Tex(R"x^2 + y^2 = 2\\text{E}", **kw)
        new_xy_eq.scale(48 / 36)
        ke_equation.target = ke_equation.generate_target()
        stack = VGroup(
            ke_equation.target,
            new_ke_eq,
            new_xy_eq,
        )
        stack.arrange(DOWN, buff=0.5)
        stack.next_to(plane, LEFT, buff=1.0)

        self.play(
            Group(ellipse, traced_path, state_point).animate.stretch(math.sqrt(10), 0, about_point=plane.get_origin()),
            FadeOut(axis_labels, time_span=(0, 1)),
            LaggedStart(
                MoveToTarget(ke_equation),
                FadeOut(xy_equation, DOWN),
                FadeIn(new_ke_eq),
                FadeIn(new_xy_eq),
                lag_ratio=0.5,
            ),
            run_time=4
        )
        self.wait()

        # Highlight circle equation
        rect = SurroundingRectangle(new_xy_eq)
        rect.set_stroke(YELLOW, 2)
        words = Text("Equation for a circle", font_size=36)
        words.next_to(rect, DOWN)
        words.set_color(YELLOW)
        self.play(
            ShowCreation(rect),
            Write(words),
            run_time=1
        )
        self.wait()

        # Show new axis labels
        x_axis_label, y_axis_label = axis_labels = VGroup(
            Tex(R"x = \\sqrt{m_1} \\cdot v_1", **kw),
            Tex(R"y = \\sqrt{m_2} \\cdot v_2", **kw),
        )
        axis_labels.scale(30 / 36)
        x_axis_label.next_to(plane.x_axis.get_right(), DR, SMALL_BUFF)
        y_axis_label.next_to(plane.y_axis.get_top(), DR, SMALL_BUFF)

        x_rhs_tex = R"\\sqrt{m_1} \\cdot v_1"
        y_rhs_tex = R"\\sqrt{m_2} \\cdot v_2"

        rect2 = SurroundingRectangle(new_ke_eq[R"\\left(\\sqrt{m_1} \\cdot v_1\\right)^2"])
        rect2.match_style(rect)

        self.play(
            rect.animate.surround(new_xy_eq["x^2"]),
            FadeIn(rect2),
            FadeOut(words),
        )
        self.wait()
        self.play(
            TransformFromCopy(new_xy_eq["x"][0], x_axis_label["x"][0]),
            TransformFromCopy(new_ke_eq[x_rhs_tex][0], x_axis_label[x_rhs_tex][0]),
            Write(x_axis_label["="][0]),
            run_time=2
        )
        self.wait()

        self.play(
            rect.animate.surround(new_xy_eq["y^2"]),
            rect2.animate.surround(new_ke_eq[R"\\left(\\sqrt{m_2} \\cdot v_2\\right)^2"]),
        )
        self.wait()
        self.play(
            TransformFromCopy(new_xy_eq["y"][0], y_axis_label["y"][0]),
            TransformFromCopy(new_ke_eq[y_rhs_tex][0], y_axis_label[y_rhs_tex][0]),
            Write(y_axis_label["="][0]),
            run_time=2
        )
        self.wait()
        self.play(
            rect.animate.surround(new_xy_eq),
            rect2.animate.surround(VGroup(ke_equation, new_ke_eq)),
        )
        self.wait()
        self.play(FadeOut(rect), FadeOut(rect2))
        self.wait()

        # Roll back to the beginning once more
        state_point.add_updater(lambda m: m.move_to(
            plane.c2p(*state_tracker.get_scaled_block_velocities())
        ))

        self.play(
            time_tracker.animate.set_value(0),
            FadeOut(new_ke_eq),
            FadeOut(new_xy_eq),
            FadeOut(traced_path),
            run_time=4
        )

        # Shrinking circle
        if False:
            # For an insertion
            time_tracker.clear_updaters()
            time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
            last_n_collisions = state_tracker.get_n_collisions()

            ghost_circles = VGroup()
            self.add(ghost_circles)
            self.add(self.get_traced_path(state_point))
            for _ in range(10):
                self.wait_until(lambda: state_tracker.get_n_collisions() > last_n_collisions)
                ghost_circles.add(ellipse.copy().set_stroke(opacity=0.25))
                ellipse.scale(0.9)
                time_tracker.set_value(time_tracker.get_value() / 0.9)
                time_tracker.increment_value(1 / 30)
                state_tracker.state0[2:] *= 0.9
                last_n_collisions = state_tracker.get_n_collisions()

        # Show the regions of the state space
        region = Square()
        region.set_fill(RED, 0.25)
        region.set_stroke(RED, 0)
        region.replace(plane, stretch=True)
        region.stretch(0.5, 0, about_edge=LEFT)
        region.save_state()
        region.stretch(1e-5, 0, about_edge=RIGHT)

        self.play(Restore(region))
        self.wait()
        self.play(region.animate.stretch(0, 1).set_stroke(RED, 2))
        self.wait()
        self.play(region.animate.scale(0, about_point=state_point.get_center()))
        self.remove(region)
        self.wait()

        # Think through first collision
        self.play(time_tracker.animate.set_value(2), run_time=2, rate_func=linear)

        big_block_rect, lil_block_rect = rects = VGroup(
            SurroundingRectangle(Group(blocks[i], velocity_vectors[i]), buff=buff)
            for i, buff in zip([0, 1], [0.25, 0.1])
        )
        rects.set_stroke(RED, 2)
        x_rect, y_rect = xy_rects = Square().replicate(2)
        xy_rects.set_stroke(RED, width=0).set_fill(RED, 0.2)
        xy_rects.replace(plane, stretch=True)
        y_rect.stretch(0.5, 1, about_edge=DOWN)
        x_rect.stretch(0.05, 0, about_point=plane.c2p(-2.7, 0))

        self.wait()
        self.play(ShowCreation(lil_block_rect))
        self.wait()
        self.play(ReplacementTransform(lil_block_rect, y_rect))
        self.wait()
        self.play(ShowCreation(big_block_rect))
        self.wait()
        self.play(ReplacementTransform(big_block_rect, x_rect))
        self.play(x_rect.animate.shift(0.2 * RIGHT), run_time=4, rate_func=lambda t: wiggle(t, 5))
        self.wait()

        # Constrain to circle
        state_point.clear_updaters()
        marked_velocity.clear_updaters()
        marked_velocity.add_updater(
            lambda m: m.set_value(
                np.array(plane.p2c(state_point.get_center())) / state_tracker.sqrt_mass_vect
            )
        )
        arc = Arc(190 * DEG, 50 * DEG, radius=0.5 * ellipse.get_width(), arc_center=plane.get_center())
        arc.set_stroke(RED, 5)

        self.play(Rotate(state_point, TAU, about_point=plane.get_origin(), run_time=5))
        self.play(
            ReplacementTransform(x_rect, arc),
            ReplacementTransform(y_rect, arc),
        )
        self.play(
            Rotate(
                state_point,
                25 * DEG,
                about_point=plane.get_center(),
                rate_func=lambda t: wiggle(t, 5),
                run_time=8,
            )
        )
        self.wait()

        # Bring in conservation of momentum
        p_equation.next_to(ke_equation, DOWN)
        new_p_eq = Tex(
            R"\\sqrt{m_1}\\left(\\sqrt{m_1} \\cdot v_1\\right) + \\sqrt{m_2}\\left(\\sqrt{m_2} \\cdot v_2\\right) = P",
            **kw
        )
        new_p_eq.next_to(p_equation, DOWN, buff=0.75)
        p_xy_eq = Tex(R"\\sqrt{m_1} x + \\sqrt{m_2} y = P", **kw)
        p_xy_eq.next_to(new_p_eq, DOWN, buff=0.75)

        p1_rects, p2_rects = p_rects = VGroup(
            VGroup(
                SurroundingRectangle(p_equation[f"m_{i} v_{i}"]),
                SurroundingRectangle(p_xy_eq[Rf"\\sqrt{{m_{i}}} {x}"]),
            )
            for i, x in [(1, "x"), (2, "y")]
        )
        p_rects.set_stroke(GREEN, 2)

        self.play(
            ke_equation.animate.to_edge(UP).set_opacity(0.33),
            FadeIn(p_equation),
        )
        self.wait()
        self.play(TransformMatchingTex(p_equation.copy(), new_p_eq))
        self.play(
            TransformMatchingTex(
                new_p_eq.copy(),
                p_xy_eq,
                key_map={
                    R"\\left(\\sqrt{m_1} \\cdot v_1\\right)": "x",
                    R"\\left(\\sqrt{m_2} \\cdot v_2\\right)": "y",
                }
            )
        )
        self.wait()
        self.play(ShowCreation(p1_rects, lag_ratio=0))
        self.wait()
        self.play(ReplacementTransform(p1_rects, p2_rects))
        self.wait()
        self.play(FadeOut(p2_rects))
        self.wait()

        # Show the momentum line
        p_line = Line(LEFT, RIGHT)
        p_line.set_width(6)
        p_line.set_stroke(GREEN, 3)
        p_line.rotate(-90 * DEG + state_tracker.theta)
        p_line.move_to(ellipse.get_left())

        slope_eq = Tex(R"\\text{Slope} = -\\sqrt{m_1 \\over m_2}", font_size=36)
        slope_eq.next_to(p_line.pfp(0.6), RIGHT)

        self.play(
            ShowCreation(p_line),
            FadeOut(arc),
        )
        self.wait()
        self.play(Write(slope_eq))
        self.wait()
        self.play(FadeOut(slope_eq))
        self.play(
            Group(p_line, state_point).animate.shift(4 * RIGHT),
            run_time=8,
            rate_func=there_and_back,
        )
        self.wait()

        # Note the two intersection points
        point2 = state_point.get_center().copy()
        state_point.add_updater(lambda m: m.move_to(plane.c2p(*state_tracker.get_scaled_block_velocities())))
        self.play(time_tracker.animate.set_value(0), run_time=2)
        point1 = state_point.get_center().copy()
        self.wait()
        state_point.suspend_updating()

        intersection_arrows = VGroup(
            Vector(0.5 * DL).next_to(point, UR, buff=SMALL_BUFF)
            for point in [point1, point2]
        )

        self.play(
            state_point.animate.move_to(p_line.get_start()),
            rate_func=lambda t: wiggle(t, 5),
            run_time=8
        )
        self.wait()
        self.play(LaggedStartMap(GrowArrow, intersection_arrows, lag_ratio=0.5))
        self.wait()

        state_point.resume_updating()
        traced_path = self.get_traced_path(state_point)
        self.add(traced_path)
        self.play(time_tracker.animate.set_value(2), run_time=2, rate_func=linear)
        traced_path.suspend_updating()
        self.wait()
        self.play(
            FadeOut(intersection_arrows),
            FadeOut(new_p_eq),
            FadeOut(p_xy_eq),
            ke_equation.animate.set_opacity(1),
        )

        # Little block off wall
        traced_path.resume_updating()
        self.play(time_tracker.animate.set_value(7), run_time=5, rate_func=linear)
        traced_path.suspend_updating()
        self.wait()

        state_point.suspend_updating()
        self.play(
            state_point.animate.move_to(point2),
            rate_func=there_and_back,
            run_time=4,
        )
        self.wait()
        self.play(
            p_line.animate.move_to(plane.c2p(state_tracker.get_momentum() / math.sqrt(self.masses[0]))),
            run_time=2
        )
        self.wait()

        # Let play out for a while
        p_line.add_updater(
            lambda m: m.move_to(plane.c2p(state_tracker.get_momentum() / math.sqrt(self.masses[0])))
        )

        traced_path.resume_updating()
        state_point.resume_updating()

        self.play(
            time_tracker.animate.set_value(25),
            run_time=(25 - time_tracker.get_value()),
            rate_func=linear,
        )

        # Show the end zone
        endzone_line = Line(
            plane.get_origin(),
            plane.c2p(*(state_tracker.sqrt_mass_vect * 4)),
        )
        endzone_line.set_stroke(WHITE, 2)
        endzone = Polygon(
            endzone_line.get_end(),
            plane.get_origin(),
            plane.c2p(10, 0),
        )
        endzone.set_fill(GREEN, 0.25)
        endzone.set_stroke(width=0)

        ur_quadrant = Square(4.0)
        ur_quadrant.set_fill(GREEN, 0.25).set_stroke(width=0)
        ur_quadrant.move_to(plane.get_origin(), DL)

        endzone_line_label = Tex(R"v_1 = v_2", **kw)
        endzone_line_label.next_to(ORIGIN, UP, buff=SMALL_BUFF)
        slope_label = Tex(R"y = x \\cdot \\sqrt{m_2 / m_1}", **kw)
        slope_label.scale(20 / kw["font_size"])
        slope_label.next_to(ORIGIN, DOWN, SMALL_BUFF)

        VGroup(
            endzone_line_label, slope_label
        ).rotate(state_tracker.theta, about_point=ORIGIN).shift(endzone_line.pfp(0.45))

        time_tracker.clear_updaters()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.wait(5)
        self.play(FadeOut(p_line), FadeIn(ur_quadrant))
        self.wait(5)
        self.play(
            FadeOut(ur_quadrant),
            FadeIn(endzone)
        )
        self.wait(4)
        self.play(ShowCreation(endzone_line), Write(endzone_line_label))
        self.wait(2)
        self.play(Write(slope_label))
        self.wait(20)

    def get_floor_and_wall_2d(
        self,
        width=13,
        stroke_width=2,
        height=2,
        tick_spacing=0.5,
        tick_vect=0.25 * DL,
        buff_to_bottom=0.75,
    ):
        floor = Line(LEFT, RIGHT)
        floor.set_width(width)
        floor.center()
        floor.to_edge(DOWN, buff=buff_to_bottom)
        dl_point = floor.get_left()

        wall = Line(ORIGIN, UP)
        wall.set_height(height)
        wall.move_to(dl_point, DOWN)

        ticks = VGroup()
        for y in np.arange(tick_spacing, height + tick_spacing, tick_spacing):
            start = dl_point + y * UP
            ticks.add(Line(start, start + tick_vect))

        result = VGroup(floor, VGroup(wall, ticks))
        result.set_stroke(WHITE, stroke_width)

        return result

    def get_floor_and_wall_3d(
        self,
        buff_to_bottom=0.75,
        color=GREY_D,
        shading=(0.2, 0.2, 0.2)
    ):
        floor = Square3D(resolution=(20, 20))
        floor.rotate(90 * DEG, LEFT)
        floor.set_shape(self.floor_width, 0, self.floor_depth)
        floor.to_edge(DOWN, buff=buff_to_bottom)

        wall = Square3D()
        wall.rotate(90 * DEG, UP)
        wall.set_shape(0, self.wall_height, self.floor_depth)
        wall.move_to(floor.get_left(), DOWN)

        result = Group(floor, wall)
        result.set_color(color)
        result.set_shading(*shading)
        result.to_corner(DL)

        for part, vect in zip(result, [OUT, RIGHT]):
            part.data['d_normal_point'] = part.data['point'] + 1e-3 * vect

        return result

    def get_block_pair(self, floor, three_d=False):
        return Group(
            self.get_block(mass, color, width, floor, three_d=three_d)
            for mass, color, width in zip(self.masses, self.colors, self.widths)
        )

    def get_block(
        self,
        mass,
        color,
        width,
        floor,
        font_size=24,
        three_d=False,
        stroke_width=2,
        shading=None,
        floor_buff=0.01,
    ):
        if shading is None:
            shading = self.block_shading
        if three_d:
            body = Cube()
            body.set_color(color)
            body.set_shading(*shading)

            shell = VCube()
            shell.set_fill(opacity=0)
            shell.set_stroke(WHITE, width=1)
            shell.set_anti_alias_width(3)
            shell.replace(body)
            shell.apply_depth_test()
            shell.set_anti_alias_width(1)

            block = Group(body, shell)
        else:
            block = Square()
            block.set_stroke(WHITE, stroke_width)
            block.set_fill(color, 1)
            block.set_shading(*shading)

        block.set_width(width)
        block.next_to(floor, UP, buff=floor_buff)
        block.mass = mass

        mass_label = Tex(R"10 \\, \\text{kg}", font_size=font_size)
        mass_label.make_number_changeable("10", edge_to_fix=RIGHT).set_value(mass)
        mass_label.next_to(block, UP, buff=SMALL_BUFF)
        if three_d:
            mass_label.set_backstroke(BLACK, 1)

        block.add(mass_label)
        block.mass_label = mass_label

        return block

    def bind_blocks_to_state(self, blocks, floor, state_tracker):
        min_x = floor.get_x(LEFT) + blocks[1].get_width()

        def update_blocks(blocks):
            pos = state_tracker.get_block_positions()
            blocks[0].set_x(min_x + pos[0], LEFT)
            blocks[1].set_x(min_x + pos[1], RIGHT)

        blocks.add_updater(update_blocks)

    def get_velocity_vector(
        self,
        block,
        vel_function,
        thickness=2,
        scale_factor=0.5,
        color=RED,
        backstroke_width=1,
        label_backstroke_width=1,
        num_decimal_places=2,
        max_width=1.0,
        font_size=18
    ):
        vector = Vector(RIGHT, thickness=thickness)
        vector.set_fill(RED)
        vector.set_backstroke(BLACK, backstroke_width)

        def update_vector(vector):
            start = block[0].get_zenith() + 0.1 * DOWN
            vel = vel_function()
            width = max_width * math.tanh(scale_factor * vel)
            vector.put_start_and_end_on(start, start + width * RIGHT)
            return vector

        vector.add_updater(update_vector)

        label = DecimalNumber(
            0,
            num_decimal_places=num_decimal_places,
            font_size=font_size
        )
        label.set_fill(RED)
        label.set_backstroke(BLACK, label_backstroke_width)
        label.add_updater(lambda m: m.set_value(vel_function()).next_to(
            vector.get_start(), UP, buff=0.1
        ))
        # label.always.next_to(vector, UP, buff=0.05)
        # label.f_always.set_value(vel_function)

        return VGroup(vector, label)

    def get_traced_path(self, state_point):
        return TracedPath(state_point.get_center, stroke_color=RED, stroke_width=1)

    def check_for_clacks(self):
        n_collisions = self.state_tracker.get_n_collisions()

        if n_collisions == self.last_collision_count:
            return
        if n_collisions % 2 == 0:
            point = self.blocks[1][0].get_left()
        else:
            point = self.blocks[1][0].get_right()

        time_since = self.state_tracker.time_from_last_collision()

        if n_collisions > self.last_collision_count:
            flash = turn_animation_into_updater(ClackAnimation(point))
            flash.update(dt=time_since)
            flash.time = self.time - time_since
            self.flashes.append(flash)
            self.add(flash)

        sound_file = self.clack_sound if n_collisions > self.last_collision_count else self.reverse_clack_sound
        self.add_sound(sound_file, time_offset=-time_since)

        for flash in self.flashes:
            if flash.time < self.time - 0.25:
                self.remove(flash)
                self.flashes.remove(flash)

        self.last_collision_count = n_collisions

    def update_frame(self, dt=0, force_draw=False):
        if dt > 0 and self.listen_for_clacks:
            self.check_for_clacks()
        super().update_frame(dt, force_draw)


class BasicBlockCount(Blocks):
    initial_positions = [10, 7]
    initial_velocities = [-2, 0]
    masses = [10, 1]
    widths = [1.0, 0.5]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    total_time = 30
    field_of_view = 10 * DEG
    floor_width = 15
    floor_depth = 6
    wall_height = 5
    initial_orientation = (0, 0, 0)
    samples = 4

    def construct(self):
        self.add_count_label()
        self.frame.reorient(*self.initial_orientation)
        # Test

        self.play(
            self.time_tracker.animate.set_value(self.total_time),
            run_time=self.total_time,
            rate_func=linear,
        )

    def add_count_label(self):
        count_label = Tex(R"\\# \\text{Collisions} = 0")
        count = count_label.make_number_changeable("0")
        count.f_always.set_value(self.state_tracker.get_n_collisions)
        count_label.to_corner(UL)
        count_label.fix_in_frame()
        self.count_label = count_label
        self.add(count_label)


class PreviewClip(BasicBlockCount):
    initial_velocities = [-0.75, 0]
    masses = [100, 1]
    widths = [2.0, 0.5]
    initial_positions = [10, 7]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    floor_depth = 2
    wall_height = 2

    def construct(self):
        # Basic shot
        frame = self.frame
        frame.reorient(-46, -6, 0, (0.41, -2.47, 1.07), 3.59)
        self.add(self.wall)
        self.add_count_label()
        self.time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.play(
            frame.animate.reorient(-46, -4, 0, (-0.78, -2.2, -0.17), 5.41),
            run_time=8
        )
        self.play(
            frame.animate.reorient(-4, -4, 0, (-2.38, -1.95, -0.99), 6.58),
            run_time=12,
        )
        self.play(frame.animate.reorient(-3, -2, 0, (1.68, -0.59, -0.79), 9.86), run_time=20)


class BasicBlockCount1e1(BasicBlockCount):
    initial_positions = [6, 3]
    initial_velocities = [-2, 0]
    masses = [1, 1]
    widths = [0.5, 0.5]
    colors = [LITTLE_BLOCK_COLOR, LITTLE_BLOCK_COLOR]
    total_time = 15


class BasicBlockCount4(BasicBlockCount):
    initial_positions = [8, 5]
    initial_velocities = [-2, 0]
    masses = [4, 1]
    widths = [1.0, 0.75]
    colors = [BLUE_D, LITTLE_BLOCK_COLOR]
    total_time = 15


class BasicBlockCount16Faster(BasicBlockCount):
    initial_positions = [8, 5]
    initial_velocities = [-2, 0]
    masses = [16, 1]
    widths = [1.5, 0.75]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    total_time = 15


class BasicBlockCount64Slower(BasicBlockCount):
    initial_positions = [5, 3]
    initial_velocities = [-1.0, 0]
    masses = [64, 1]
    widths = [2.0, 0.75]
    colors = [interpolate_color(BLUE_E, BLACK, 0.5), LITTLE_BLOCK_COLOR]
    total_time = 30


class BasicBlockCount256(BasicBlockCount):
    initial_positions = [5, 3]
    initial_velocities = [-1.5, 0]
    masses = [256, 1]
    widths = [2.5, 0.75]
    colors = [interpolate_color(BLUE_E, BLACK, 0.8), LITTLE_BLOCK_COLOR]
    total_time = 20


class BasicBlockCount1e2(BasicBlockCount):
    initial_positions = [8, 5]
    initial_velocities = [-1, 0]
    masses = [100, 1]
    widths = [1.0, 0.5]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    total_time = 30


class BasicBlockCount1e4(BasicBlockCount1e2):
    masses = [int(1e4), 1]
    widths = [1.5, 0.5]
    colors = [interpolate_color(BLUE_E, BLACK, 0.5), LITTLE_BLOCK_COLOR]
    total_time = 40


class BasicBlockCount1e6(BasicBlockCount1e2):
    masses = [int(1e6), 1]
    widths = [2.0, 0.5]
    colors = [interpolate_color(BLUE_E, BLACK, 0.8), LITTLE_BLOCK_COLOR]


class BasicBlockCount1e8(BasicBlockCount1e2):
    masses = [int(1e8), 1]
    widths = [3.0, 0.5]
    colors = [interpolate_color(BLUE_E, BLACK, 0.95), LITTLE_BLOCK_COLOR]
    block_shading = [0.25, 0.25, 0]
    initial_orientation = (-7, 1, 0)


class BasicBlockCount1e10(BasicBlockCount1e2):
    masses = [int(1e10), 1]
    widths = [3.0, 0.5]
    initial_positions = [5, 3]
    colors = [BLACK, LITTLE_BLOCK_COLOR]
    block_shading = [0.1, 0.1, 0]
    initial_orientation = (-7, 1, 0)
    total_time = 20


class SlowMoBlockCount(BasicBlockCount1e4):
    time_to_slomo = 7.95
    wall_height = 3

    def setup(self):
        super().setup()
        self.add_count_label()

        time_tracker = self.time_tracker
        rate_tracker = ValueTracker(1)
        time_tracker.add_updater(lambda m, dt: m.increment_value(rate_tracker.get_value() * dt))
        self.add(time_tracker)

        slowdown_label = Tex(R"100 \\text{x } \\text{Slower}")
        slowdown_label.next_to(self.count_label, DOWN)
        slowdown_label.set_color(RED)
        factor_label = slowdown_label.make_number_changeable("100", edge_to_fix=UR)
        factor_label.add_updater(lambda m: m.set_value(int(1 / rate_tracker.get_value())).fix_in_frame())
        slowdown_label.fix_in_frame()

        self.slowdown_label = slowdown_label
        self.rate_tracker = rate_tracker

    def construct(self):
        # Test
        self.play(
            self.frame.animate.reorient(-1, 0, 0, (-5.94, -2.36, -1.0), 3.80).set_field_of_view(25 * DEG),
            run_time=self.time_to_slomo
        )
        self.clack_sound = "slow_clack"

        self.rate_tracker.set_value(0.01)
        self.add(self.slowdown_label)
        self.wait(7)

        self.rate_tracker.set_value(1.0)
        self.remove(self.slowdown_label)
        self.clack_sound = "clack"
        self.play(
            self.frame.animate.to_default_state().set_field_of_view(15 * DEG),
            run_time=8
        )
        self.wait(30)


class SlowMoBlockCount1e6(SlowMoBlockCount):
    masses = [int(1e6), 1]
    widths = [2.0, 0.5]
    colors = [interpolate_color(BLUE_E, BLACK, 0.8), LITTLE_BLOCK_COLOR]
    time_to_slomo = 7.97
    slow_factor = 2000
    time_in_slo_mo = 10 / 2000
    time_after = 10

    def construct(self):
        frame = self.frame
        rate_tracker = self.rate_tracker

        # Test
        self.play(
            frame.animate.reorient(-9, 3, 0, (-4.25, -1.33, -0.98), 5.70).set_field_of_view(25 * DEG),
            run_time=7.90
        )

        self.clack_sound = "slow_clack"
        self.add(self.slowdown_label)
        rate_tracker.set_value(1e-2)
        self.wait(8)

        rate_tracker.set_value(1e-3)
        self.clack_sound = "super_slow_clack"
        self.wait(7)

        rate_tracker.set_value(1e-4)
        self.clack_sound = "super_super_slow_clack"
        self.wait(10)

        rate_tracker.set_value(1e-3)
        self.clack_sound = "super_slow_clack"
        self.wait()

        self.clack_sound = "slow_clack"
        self.add(self.slowdown_label)
        rate_tracker.set_value(1e-2)
        self.wait()

        rate_tracker.set_value(1)
        self.remove(self.slowdown_label)
        self.clack_sound = "clack"
        self.play(frame.animate.to_default_state().set_field_of_view(10 * DEG), run_time=5)
        self.wait(15)


class SlowMoBlockCount1e8(SlowMoBlockCount):
    masses = [int(1e8), 1]
    widths = [3.0, 0.5]
    colors = [interpolate_color(BLUE_E, BLACK, 0.95), LITTLE_BLOCK_COLOR]
    block_shading = [0.25, 0.25, 0]

    def construct(self):
        frame = self.frame
        rate_tracker = self.rate_tracker

        # Test
        self.play(
            frame.animate.reorient(-8, -6, 0, (-4.15, -1.22, -0.96), 6.18).set_field_of_view(25 * DEG),
            run_time=7.98
        )

        self.clack_sound = "slow_clack"
        self.add(self.slowdown_label)
        rate_tracker.set_value(1e-2)
        self.wait()

        rate_tracker.set_value(1e-3)
        self.clack_sound = "super_slow_clack"
        self.wait()

        rate_tracker.set_value(1e-4)
        self.clack_sound = "super_super_slow_clack"
        self.wait(10)

        rate_tracker.set_value(1e-3)
        self.clack_sound = "super_slow_clack"
        self.wait()

        self.clack_sound = "slow_clack"
        self.add(self.slowdown_label)
        rate_tracker.set_value(1e-2)
        self.wait()

        rate_tracker.set_value(1)
        self.remove(self.slowdown_label)
        self.clack_sound = "clack"
        self.play(frame.animate.to_default_state().set_field_of_view(10 * DEG), run_time=5)
        self.wait(10)


class IntroduceSetup(BasicBlockCount):
    masses = [16, 1]

    def construct(self):
        frame = self.frame
        time_tracker = self.time_tracker
        self.add_count_label()
        count_label = self.count_label
        self.remove(count_label)

        # Describe the setup
        floor_label = Text("Frictionless plane")
        floor_label.next_to(self.floor, UP, SMALL_BUFF)
        floor_label.shift(2 * LEFT)
        floor_label.set_backstroke(BLACK, 3)
        alt_floor = self.floor.copy()
        alt_floor.set_color(YELLOW, opacity=0.5)
        alt_floor.shift(1e-3 * UP)
        alt_floor.save_state()
        alt_floor.stretch(0, 0, about_edge=LEFT)

        time_tracker.clear_updaters()
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.wait(6)
        self.add(floor_label, self.blocks)
        self.play(
            Write(floor_label),
            Restore(alt_floor),
            run_time=2
        )
        self.wait()
        self.play(FadeOut(alt_floor), FadeOut(floor_label))
        self.wait(2)
        time_tracker.clear_updaters()

        # Back up
        self.play(time_tracker.animate.set_value(0), run_time=2)

        # Label
        blocks = self.blocks
        lil_block_rect = SurroundingRectangle(blocks[1])
        lil_block_label = Text("Initially stationary")
        lil_block_label.set_color(YELLOW)
        lil_block_label.next_to(lil_block_rect, UP, buff=1.5)
        lil_block_arrow = Arrow(lil_block_label.get_bottom(), lil_block_rect.get_top())
        lil_block_arrow.set_fill(YELLOW)

        big_block_vect = Vector(1.5 * LEFT, thickness=4)
        big_block_vect.set_fill(RED)
        big_block_vect.shift(blocks[0][0].get_center())

        self.play(
            FadeIn(lil_block_label),
            ShowCreation(lil_block_rect),
            GrowArrow(lil_block_arrow),
        )
        self.wait()
        self.play(
            lil_block_rect.animate.surround(blocks[0]).set_stroke(opacity=0),
            FadeOut(lil_block_label, shift=2 * RIGHT),
            ReplacementTransform(lil_block_arrow, big_block_vect),
        )
        self.wait()

        # Let it play out
        big_block_vect.add_updater(lambda m: m.shift(blocks[0][0].get_center() - m.get_start()))
        time_tracker.clear_updaters()
        time_tracker.add_updater(lambda m, dt: m.increment_value(1.5 * dt))

        self.play(
            VFadeOut(big_block_vect),
            VFadeIn(self.count_label),
        )
        self.wait(10)


class ThumbnailShot(BasicBlockCount):
    initial_velocities = [-0.75, 0]
    masses = [int(1e6), 1]
    widths = [1.0, 0.5]
    initial_positions = [3, 2]
    colors = [BLUE_E, LITTLE_BLOCK_COLOR]
    floor_depth = 2
    wall_height = 2

    def construct(self):
        # Orient
        frame = self.frame
        frame.reorient(-40, -5, 0, (-3.45, -2.54, -0.34), 3.00)

        blocks = self.blocks
        blocks.deactivate_depth_test()
        for block in blocks:
            block[2].set_opacity(0).scale(0, about_point=block[0].get_top())
            for cube in block[:2]:
                cube[1].set_opacity(0)
                cube[2].set_opacity(0)
                cube[5].set_opacity(0)

        # Add speed lines
        speed_lines = Line(ORIGIN, 1.5 * RIGHT).get_grid(7, 1)
        speed_lines.set_stroke(YELLOW, width=(5, 0))
        speed_lines.arrange_to_fit_height(0.8 * blocks[0][0].get_height())
        speed_lines.next_to(blocks[0][0], RIGHT, buff=0.05, aligned_edge=OUT)

        self.add(speed_lines)

        lil_speed_lines = speed_lines[1:-1].copy()
        lil_speed_lines.flip()
        lil_speed_lines.set_shape(0.8, 0.3)
        lil_speed_lines.next_to(blocks[1], LEFT, buff=0.05)
        self.add(lil_speed_lines)

        # Test

    def old(self):
        # Add ghosts
        block_template = blocks[0].copy()
        block_template.clear_updaters()
        for a in np.linspace(0, 1, 4):
            ghost_block = block_template.copy()
            ghost_block.set_opacity(interpolate(0.9, 0.1, a))
            ghost_block.shift(a * 2 * RIGHT)
            self.add(ghost_block)
        self.add(blocks)

class MovementOfWall(Blocks):
    initial_positions = [6, 5]
    initial_velocities = [-2, 0]
    masses = [1, 1]
    widths = [0.5, 0.5]
    colors = [LITTLE_BLOCK_COLOR, LITTLE_BLOCK_COLOR]

    def construct(self):
        # Nudge the wall
        frame = self.frame
        frame.reorient(-2, -4, 0, (-6.01, -2.36, -2.04), 6.24)

        time_tracker = self.time_tracker
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        wall = self.wall
        ghost_wall = wall.copy().set_opacity(0.5)
        nudge = 0.25 * LEFT
        brace = Brace(Line(ORIGIN, nudge), UP, tex_string=R"\\underbrace{\\quad\\qquad}")
        brace.next_to(wall, UP, SMALL_BUFF, aligned_edge=RIGHT)
        brace.save_state()
        brace.stretch(0, 0, about_edge=RIGHT).set_fill(opacity=0)

        self.wait(3)
        self.add(ghost_wall)
        self.play(
            wall.animate.shift(nudge),
            Restore(brace),
            rate_func=rush_from
        )
        self.wait(2)

        # Make it massive
        time_tracker.clear_updaters()
        block = Cube()
        block.set_shape(0.5, self.wall_height, self.floor_depth)
        block.move_to(ghost_wall, DR)
        block.set_color(GREY_D)

        self.play(
            FadeOut(Group(wall, ghost_wall, brace)),
            FadeIn(block),
        )
        self.play(block.animate.set_height(10, about_edge=UP, stretch=True), run_time=3)

        time_tracker.set_value(0)
        self.play(time_tracker.animate.set_value(6), run_time=6, rate_func=linear)
        self.wait()


class CirclePuzzle(InteractiveScene):
    def construct(self):
        # Add axes
        x_axis, y_axis = axes = VGroup(Line(1.5 * LEFT, 1.5 * RIGHT), Line(UP, DOWN))
        axes.set_stroke(WHITE, 2, 0.33)
        axes.set_height(7)
        kw = dict(
            t2c={
                f"v_1": RED,
                f"v_2": RED,
                f"m_1": BLUE,
                f"m_2": BLUE,
            },
            font_size=24
        )
        axis_labels = VGroup(
            Tex(R"x = \\sqrt{m_1} \\cdot v_1", **kw).next_to(x_axis.get_right(), DOWN),
            Tex(R"y = \\sqrt{m_2} \\cdot v_2", **kw).next_to(y_axis.get_top(), RIGHT),
        )

        self.add(axes, axis_labels)

        # Add circle
        circle = Circle(radius=3)
        circle.set_stroke(YELLOW, 2)

        self.play(ShowCreation(circle))
        self.wait()

        # Set up lines
        slope_tracker = ValueTracker(-4)
        get_slope = slope_tracker.get_value
        lines = self.get_lines(circle, get_slope())

        state_point = Group(
            TrueDot(radius=0.05).make_3d(),
            GlowDot(radius=0.2),
        )
        state_point.set_color(RED)
        state_point.move_to(circle.get_left())

        self.play(FadeIn(state_point, shift=2 * DR, scale=0.25))
        self.wait()

        # Note the slope
        slope_label = Tex(R"\\text{Slope} = -4.0", font_size=36)
        slope_label.next_to(lines[0].get_center(), RIGHT)
        slope_dec = slope_label.make_number_changeable("-4.0")
        count_label = Tex(R"\\# \\text{Lines} = 1", font_size=36)
        count = count_label.make_number_changeable("1")
        count_label.to_corner(UL)
        right_shift = (slope_dec.get_width() - count.get_width()) * RIGHT

        self.play(
            ShowCreation(lines[0]),
            state_point.animate.move_to(lines[0].get_end()),
            FadeIn(slope_label),
        )
        self.wait()
        self.play(
            slope_label.animate.next_to(count_label, DOWN, aligned_edge=RIGHT).shift(right_shift),
            VFadeIn(count_label),
            ChangeDecimalToValue(count, 2),
            ShowCreation(lines[1]),
            state_point.animate.move_to(lines[1].get_end()),
        )
        self.wait()

        # Show remaining lines
        for line in lines[2:]:
            self.play(
                ShowCreation(line),
                state_point.animate.move_to(line.get_end()),
                ChangeDecimalToValue(count, count.get_value() + 1)
            )
        self.wait()
        self.add(lines)

        # Show the end zone
        endzone = self.get_end_zone(-1.0 / get_slope())
        self.play(FadeIn(endzone))
        self.wait()

        # Ask the question
        count_rect = SurroundingRectangle(count_label)
        count_rect.set_stroke(PINK, 3)

        self.play(
            LaggedStart(
                (line.animate.set_stroke(PINK, 5).set_anim_args(rate_func=there_and_back)
                for line in lines),
                lag_ratio=0.5,
                run_time=6
            ),
            ShowCreation(count_rect)
        )

        # Vary the slope
        self.add(lines)
        lines.f_always.set_submobjects(lambda: self.get_lines(circle, get_slope()))
        count.f_always.set_value(lambda: len(lines))
        slope_dec.f_always.set_value(get_slope)
        endzone.f_always.become(lambda: self.get_end_zone(-1.0 / get_slope()))
        updated_pieces = [lines, count, slope_dec, endzone]

        slope_rect = SurroundingRectangle(slope_label)
        slope_rect.add_updater(lambda m: m.set_width(
            slope_label.get_width() + 2 * SMALL_BUFF,
            stretch=True,
            about_edge=LEFT
        ))

        self.play(
            ReplacementTransform(count_rect, slope_rect, suspend_mobject_updating=True),
            FadeOut(state_point),
        )
        self.play(
            slope_tracker.animate.set_value(-10),
            rate_func=there_and_back,
            run_time=10
        )
        for piece in updated_pieces:
            piece.suspend_updating()

        # Show slope equation
        rhs = Tex(R"= -\\sqrt{m_1 / m_2}", **kw)
        rhs.scale(1.5)
        rhs.next_to(slope_label, RIGHT)

        mass_equations = VGroup(
            Tex(R"m_1 = 16", **kw),
            Tex(R"m_2 = 1", **kw),
        )
        mass_equations.scale(1.5)
        mass_equations.arrange(DOWN, aligned_edge=LEFT)
        mass_equations.next_to(slope_label, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(
            Write(rhs),
            FadeOut(slope_rect),
        )
        self.wait()
        self.play(
            TransformFromCopy(rhs["m_1"][0], mass_equations[0]["m_1"][0]),
            TransformFromCopy(rhs["m_2"][0], mass_equations[1]["m_2"][0]),
            FadeIn(mass_equations[0][2:]),
            FadeIn(mass_equations[1][2:]),
        )
        self.wait()
        self.add(mass_equations)

        # Increase slope
        mass_tracker = ValueTracker(16)
        get_mass_ratio = mass_tracker.get_value
        m1_dec = mass_equations[0].make_number_changeable("16")
        m1_dec.add_updater(lambda m: m.set_value(get_mass_ratio()))

        for piece in updated_pieces:
            piece.resume_updating()

        slope_tracker.add_updater(lambda m: m.set_value(-math.sqrt(int(get_mass_ratio()))))

        self.add(slope_tracker)
        self.play(
            mass_tracker.animate.set_value(100),
            rate_func=lambda t: there_and_back_with_pause(t, 0.1),
            run_time=12
        )

        for piece in updated_pieces:
            piece.suspend_updating()

        # Clear labels
        self.play(FadeOut(mass_equations), FadeOut(rhs))

        # Show arcs
        points = np.array([line.get_start() for line in lines] + [lines[-1].get_end()])
        dots = Group(state_point.copy().move_to(point) for point in points)
        arcs = self.get_arcs(points, slope=get_slope())
        for arc in arcs:
            arc.save_state()
            arc.scale(1.25, about_point=ORIGIN)
            arc.set_stroke(width=10, opacity=0)

        state_point.move_to(lines[0].get_start())
        self.add(dots[0])
        for line, dot in zip(lines, dots[1:]):
            self.play(state_point.animate.move_to(dot), run_time=0.5)
            self.add(dot)
        self.add(dots)
        self.play(FadeOut(state_point))

        self.play(LaggedStartMap(Restore, arcs, lag_ratio=0.25, run_time=3))
        self.wait()

        # Stack the arcs
        theta = math.atan(-1.0 / get_slope())
        stacked_arcs = VGroup(
            arc.copy().rotate(PI - angle_of_vector(arc.pfp(0.5)))
            for arc in arcs
        )
        stacked_arcs.arrange(RIGHT, buff=0.1)
        stacked_arcs.to_corner(DL)

        self.remove(circle)
        self.play(
            arcs.animate.set_stroke(opacity=0.25),
            TransformFromCopy(arcs, stacked_arcs, lag_ratio=0.35, run_time=4),
        )
        self.wait()
        self.play(
            Transform(
                stacked_arcs,
                arcs.copy().set_stroke(opacity=1),
                remover=True,
                lag_ratio=0.1,
            ),
            arcs.animate.set_stroke(opacity=1),
            run_time=3,
        )
        self.wait()

        # Show just two lines
        self.play(
            lines[:3].animate.set_stroke(opacity=0.15),
            lines[5:].animate.set_stroke(opacity=0.15),
            arcs[:4].animate.set_stroke(opacity=0.15),
            arcs[5:].animate.set_stroke(opacity=0.15),
            FadeOut(dots),
        )
        self.wait()

        # Mark the 2*theta arc
        lil_arc = Arc(-90 * DEG, theta, radius=1.5, arc_center=points[4])
        theta_label = Tex(R"\\theta")
        theta_label.next_to(lil_arc.pfp(0.5), DOWN, SMALL_BUFF)
        new_lines = VGroup(
            Line(points[3], ORIGIN),
            Line(ORIGIN, points[5]),
        )
        new_lines.set_stroke(WHITE, 2)

        lil_two_arc = Arc(-PI + 4 * theta, 2 * theta, radius=1.0, arc_center=ORIGIN)
        two_theta_label = Tex(R"2\\theta")
        two_theta_label.next_to(lil_two_arc.pfp(0.9), DL, SMALL_BUFF)

        self.play(
            ShowCreation(lil_arc),
            Write(theta_label),
        )
        self.wait()
        self.play(
            TransformFromCopy(lines[3:5], new_lines),
            TransformFromCopy(lil_arc, lil_two_arc),
            TransformFromCopy(theta_label, two_theta_label),
        )
        self.wait()
        self.play(
            ReplacementTransform(lil_two_arc, arcs[4]),
            two_theta_label.animate.next_to(arcs[4].pfp(0.5), DOWN, SMALL_BUFF),
            new_lines.animate.set_stroke(WHITE, 1),
        )
        self.wait()

        # Show the other lower arcs
        theta_group = VGroup(lil_arc, theta_label)
        for n in range(6, len(lines), 2):
            self.play(
                lines[n - 3:n - 1].animate.set_stroke(opacity=0.15),
                lines[n - 1:n + 1].animate.set_stroke(opacity=1),
                theta_group.animate.shift(points[n] - points[n - 2]),
                Rotate(new_lines, 2 * theta, about_point=ORIGIN),
                two_theta_label.animate.next_to(arcs[n].pfp(0.5), arcs[n].pfp(0.5), SMALL_BUFF),
                arcs[n - 2].animate.set_stroke(opacity=0.25),
                arcs[n].animate.set_stroke(opacity=1)
            )
            self.wait()

        # Show the upper arcs as well
        self.play(
            theta_group.animate.rotate(PI, about_point=points[10]).shift(points[1] - points[10]).scale(0.75, about_point=points[1]),
            Rotate(new_lines, -12 * theta, about_point=ORIGIN),
            two_theta_label.animate.next_to(arcs[1].pfp(0.5), arcs[1].pfp(0.5), buff=0.05),
            lines[9:11].animate.set_stroke(opacity=0.15),
            lines[0:2].animate.set_stroke(opacity=1),
            arcs[10].animate.set_stroke(opacity=0.15),
            arcs[1].animate.set_stroke(opacity=1),
            FadeOut(axis_labels),
            run_time=1,
        )
        self.wait()

        for n in range(3, len(lines), 2):
            self.play(
                lines[n - 3:n - 1].animate.set_stroke(opacity=0.15),
                lines[n - 1:n + 1].animate.set_stroke(opacity=1),
                theta_group.animate.shift(points[n] - points[n - 2]),
                Rotate(new_lines, -2 * theta, about_point=ORIGIN),
                two_theta_label.animate.next_to(arcs[n].pfp(0.5), arcs[n].pfp(0.5), SMALL_BUFF),
                arcs[n - 2].animate.set_stroke(opacity=0.25),
                arcs[n].animate.set_stroke(opacity=1)
            )
            self.wait()
        state_point.move_to(lines[0].get_start())
        self.play(
            FadeOut(two_theta_label),
            FadeOut(new_lines),
            FadeOut(theta_group),
            arcs[-1].animate.set_stroke(opacity=0.15),
            lines[-2:].animate.set_stroke(opacity=0.15),
            FadeIn(state_point),
        )

        # Show the process of dropping arcs while bouncing along diagram
        two_theta_labels = VGroup()
        past_state_points = Group()
        self.add(past_state_points)
        for line, arc in zip(lines, arcs):
            ghost_line = line.copy()
            ghost_arc = arc.copy()
            line.set_stroke(opacity=1)
            arc.set_stroke(opacity=1)

            two_theta_label = Tex(R"2\\theta", font_size=36)
            two_theta_label.next_to(arc.pfp(0.5), arc.pfp(0.5), buff=0.05)
            two_theta_labels.add(two_theta_label)

            arc.save_state()
            arc.scale(1.25, about_point=ORIGIN)
            arc.set_stroke(width=12, opacity=0)

            past_state_points.add(state_point.copy())
            self.play(
                Animation(ghost_line, remover=True),
                Animation(ghost_arc, remover=True),
                ShowCreation(line),
                state_point.animate.move_to(line.get_end()),
                Restore(arc),
                FadeIn(two_theta_label, arc.saved_state.pfp(0.5) - arc.pfp(0.5)),
            )
        self.wait()

        # Emphasize endzone line
        max_x = 8
        endzone_line = Line(ORIGIN, max_x * RIGHT + max_x * -1.0 / get_slope() * UP)
        full_screen_rect = FullScreenFadeRectangle()
        full_screen_rect.set_fill(BLACK, 0.8)
        bold_lines = lines[0::2].copy()

        self.add(full_screen_rect, bold_lines)
        self.play(
            FadeIn(full_screen_rect),
            ShowCreation(endzone_line),
        )
        self.wait()
        self.play(
            FadeOut(full_screen_rect),
            FadeOut(bold_lines),
            FadeOut(endzone_line),
        )

        # Show end zone condition
        wedge = arcs[-1].copy()
        wedge.add_line_to(ORIGIN)
        wedge.add_line_to(arcs[-1].get_start())
        wedge.set_fill(PINK, 0.5)
        wedge.set_stroke(width=0)
        wedge.add(arcs[-1].copy().set_stroke(PINK))

        self.play(FadeIn(wedge))
        for angle in [-2 * theta, 10 * DEG, -10 * DEG, 5 * DEG, -5 * DEG]:
            self.play(Rotate(wedge, angle, about_point=ORIGIN))
            self.wait()

        self.play(FadeOut(wedge))

        # Bring back the mass ratio labels
        slope_label_rect = SurroundingRectangle(slope_label)
        slope_label_rect.set_stroke(YELLOW, 2)
        self.play(
            FadeIn(circle),
            FadeOut(two_theta_labels, lag_ratio=0.1),
            FadeOut(arcs, lag_ratio=0.1),
            FadeOut(past_state_points, lag_ratio=0.1),
            FadeOut(state_point),
        )
        self.play(ShowCreation(slope_label_rect))

        self.play(
            FadeIn(rhs),
            FadeIn(mass_equations),
        )
        self.wait()
        self.play(
            slope_label_rect.animate.surround(mass_equations).stretch(1.2, 0, about_edge=LEFT),
        )

        # Scale back up to 100
        self.add(lines)
        for piece in updated_pieces:
            piece.resume_updating()
        self.play(
            mass_tracker.animate.set_value(100),
            rhs.animate.shift(0.1 * RIGHT),
            run_time=5,
        )
        for piece in updated_pieces:
            piece.suspend_updating()

        # Make room
        frame = self.frame
        shift_value = 2.25 * LEFT
        line_index = 13

        movers = VGroup(count_label, slope_label, rhs, mass_equations)

        self.play(
            FadeOut(slope_label_rect, shift=shift_value),
            lines[:line_index + 1].animate.set_stroke(opacity=0.1),
            lines[line_index + 2:].animate.set_stroke(opacity=0.1),
            frame.animate.shift(shift_value),
            movers.animate.shift(shift_value),
        )

        # Show the slope
        dy_line = lines[line_index].copy().rotate(PI)
        dy_line.set_stroke(PINK, 3, 1)
        dx_line = Line(
            dy_line.get_end(),
            find_intersection(
                dy_line.get_end(),
                RIGHT,
                lines[line_index + 1].get_start(),
                lines[line_index + 1].get_vector(),
            )
        )
        dx_line.set_stroke(RED, 5)

        dy_label = Tex(R"\\Delta y", font_size=36).match_color(dy_line)
        dx_label = Tex(R"\\Delta x", font_size=36).match_color(dx_line)
        dy_label.next_to(dy_line, LEFT, SMALL_BUFF).shift(0.25 * DOWN)
        dx_label.next_to(dx_line, UP, SMALL_BUFF)

        slope_tex_kw = dict(
            t2c={
                R"\\Delta y": PINK,
                R"\\Delta x": RED,
            },
            font_size=36
        )
        new_slope_eq = Tex(R"\\text{Slope} = {\\Delta y \\over \\Delta x}", **slope_tex_kw)
        new_slope_eq.next_to(slope_label, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(
            FadeIn(new_slope_eq, DOWN),
            mass_equations.animate.to_edge(DOWN),
        )
        self.wait()
        self.play(
            ShowCreation(dy_line),
            TransformFromCopy(new_slope_eq[R"\\Delta y"][0], dy_label)
        )
        self.wait()
        self.play(
            ShowCreation(dx_line),
            TransformFromCopy(new_slope_eq[R"\\Delta x"][0], dx_label)
        )
        self.wait()

        # Show the angle
        theta = math.atan(-1 / get_slope())
        point = dy_line.get_start()
        arc = Arc(-90 * DEG, theta, radius=2.25, arc_center=point)
        theta_label = Tex(R"\\theta", font_size=36)
        theta_label.next_to(arc, DOWN, SMALL_BUFF)

        tan_eq = Tex(R"\\tan(\\theta) = {\\Delta x \\over -\\Delta y}", **slope_tex_kw)
        tan_eq.next_to(new_slope_eq, DOWN, buff=0.5, aligned_edge=LEFT)

        self.play(
            ShowCreation(arc),
            Write(theta_label),
        )
        self.play(
            TransformFromCopy(theta_label, tan_eq[R"\\theta"]),
            Write(tan_eq[R"\\tan("]),
            Write(tan_eq[R") = "]),
        )
        self.wait()
        self.play(
            TransformFromCopy(dx_label, tan_eq[R"\\Delta x"][0]),
        )
        self.play(
            TransformFromCopy(dy_label, tan_eq[R"\\Delta y"][0]),
            Write(tan_eq[R"\\over"]),
            Write(tan_eq[R"-"]),
        )
        self.wait()

        # Relate tangent to slope
        kw["font_size"] = 36
        new_rhs = Tex(R"= \\sqrt{m_2 / m_1}", **kw)
        new_rhs.next_to(tan_eq, RIGHT, SMALL_BUFF)

        new_rhs_brace = Brace(new_rhs[1:], UP, SMALL_BUFF)
        ratio_example = new_rhs_brace.get_text("0.1", buff=SMALL_BUFF, font_size=36)

        arctan_eq = Tex(R"\\theta = \\arctan\\left(\\sqrt{m_2 / m_1}\\right)", **kw)
        arctan_eq.next_to(tan_eq, DOWN, buff=0.5, aligned_edge=LEFT)

        example_arctan = Tex(R"\\theta = \\arctan(0.1)", font_size=36)
        example_arctan.next_to(arctan_eq, DOWN, buff=0.5, aligned_edge=LEFT)

        example_arctan_rect = SurroundingRectangle(example_arctan)

        self.play(
            LaggedStart(
                TransformFromCopy(new_slope_eq[R"\\Delta y"], tan_eq[R"\\Delta y"], path_arc=-90 * DEG),
                TransformFromCopy(new_slope_eq[R"\\Delta x"], tan_eq[R"\\Delta x"], path_arc=-90 * DEG),
                run_time=2
            )
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                rhs.copy(),
                new_rhs,
                path_arc=-90 * DEG,
                run_time=2,
                lag_ratio=0.01,
                matched_keys=["m_1", "m_2", R"\\sqrt"],
            )
        )
        self.wait()
        self.play(
            GrowFromCenter(new_rhs_brace),
            TransformFromCopy(mass_equations.copy().clear_updaters(), ratio_example),
        )
        self.wait()

        self.play(LaggedStart(
            TransformFromCopy(tan_eq[R"\\tan"], arctan_eq[R"\\arctan"]),
            TransformFromCopy(tan_eq[R"("], arctan_eq[R"\\left("]),
            TransformFromCopy(tan_eq[R")"], arctan_eq[R"\\right)"]),
            TransformFromCopy(tan_eq[R"\\theta"], arctan_eq[R"\\theta"]),
            TransformFromCopy(tan_eq[R"="], arctan_eq[R"="]),
            TransformFromCopy(new_rhs[R"\\sqrt{m_2 / m_1}"], arctan_eq[R"\\sqrt{m_2 / m_1}"]),
            lag_ratio=0.1,
            run_time=2
        ))
        self.wait()
        self.play(FadeIn(example_arctan, DOWN))
        self.play(ShowCreation(example_arctan_rect))
        self.wait()

    def get_lines(self, circle, slope, stroke_color=WHITE, stroke_width=2):
        theta = math.atan(-1.0 / slope)
        radius = 0.5 * circle.get_width()

        top_point = circle.get_left()
        low_point = rotate_vector(top_point, 2 * theta)

        result = VGroup()

        while True:
            if math.atan2(top_point[1], top_point[0]) < theta:
                # In the end zone
                break
            result.add(Line(top_point, low_point))
            top_point = rotate_vector(top_point, -2 * theta)
            if top_point[1] < 0:
                break
            result.add(Line(low_point, top_point))
            low_point = rotate_vector(low_point, 2 * theta)

        result.set_stroke(stroke_color, stroke_width)
        return result

    def get_end_zone(self, slope, max_x=8, color=GREEN, opacity=0.25):
        zone = Polygon(
            max_x * RIGHT + slope * max_x * UP,
            ORIGIN, 
            max_x * RIGHT
        )
        zone.set_stroke(color, 0)
        zone.set_fill(color, opacity)
        label = Text("End Zone", font_size=36)
        label.set_color(color)
        label.next_to(zone.get_bottom(), UP, SMALL_BUFF)
        label.to_edge(RIGHT, buff=0.5)

        return VGroup(zone, label)

    def get_arcs(self, points, slope):
        # Test
        theta = math.atan(-1.0 / slope)
        arcs = VGroup()
        for i in range(-1, len(points) - 2):
            arc = Line(
                points[max(i, 0)],
                points[i + 2],
                path_arc=-2 * theta * (-1)**(i % 2)
            )
            color = [BLUE, BLUE, RED, RED][i % 4]
            arc.set_stroke(color, width=4)
            arcs.add(arc)

        return arcs`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      7: "Class StateTracker inherits from ValueTracker.",
      44: "Returns the Euclidean length of a vector. ManimGL utility wrapping np.linalg.norm.",
      81: "Class ClackAnimation inherits from Restore.",
      86: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      93: "Blocks extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      108: "setup() runs before construct(). Used to initialize shared state and add persistent mobjects.",
      119: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      134: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      156: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      157: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      161: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      164: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      166: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      167: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      168: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      169: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      170: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      173: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      179: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      190: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      191: "Smoothly animates the camera to a new orientation over the animation duration.",
      192: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      193: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      194: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      195: "FadeOut transitions a mobject from opaque to transparent.",
      198: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      217: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      218: "FadeOut transitions a mobject from opaque to transparent.",
      219: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      221: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      223: "FadeOut transitions a mobject from opaque to transparent.",
      224: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      227: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      231: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      243: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      244: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      245: "Transform smoothly morphs one mobject into another by interpolating their points.",
      247: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      248: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      249: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      250: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      251: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      252: "Smoothly animates the camera to a new orientation over the animation duration.",
      257: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      258: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      259: "Smoothly animates the camera to a new orientation over the animation duration.",
      260: "FadeOut transitions a mobject from opaque to transparent.",
      263: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      266: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      267: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      268: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      1124: "Class BasicBlockCount inherits from Blocks.",
      1138: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1159: "Class PreviewClip inherits from BasicBlockCount.",
      1168: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1186: "Class BasicBlockCount1e1 inherits from BasicBlockCount.",
      1195: "Class BasicBlockCount4 inherits from BasicBlockCount.",
      1204: "Class BasicBlockCount16Faster inherits from BasicBlockCount.",
      1213: "Class BasicBlockCount64Slower inherits from BasicBlockCount.",
      1222: "Class BasicBlockCount256 inherits from BasicBlockCount.",
      1231: "Class BasicBlockCount1e2 inherits from BasicBlockCount.",
      1240: "Class BasicBlockCount1e4 inherits from BasicBlockCount1e2.",
      1247: "Class BasicBlockCount1e6 inherits from BasicBlockCount1e2.",
      1253: "Class BasicBlockCount1e8 inherits from BasicBlockCount1e2.",
      1261: "Class BasicBlockCount1e10 inherits from BasicBlockCount1e2.",
      1271: "Class SlowMoBlockCount inherits from BasicBlockCount1e4.",
      1294: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1316: "Class SlowMoBlockCount1e6 inherits from SlowMoBlockCount.",
      1325: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1364: "Class SlowMoBlockCount1e8 inherits from SlowMoBlockCount.",
      1370: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1409: "Class IntroduceSetup inherits from BasicBlockCount.",
      1412: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1485: "Class ThumbnailShot inherits from BasicBlockCount.",
      1494: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1535: "Class MovementOfWall inherits from Blocks.",
      1542: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1585: "CirclePuzzle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1586: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/colliding_blocks_v2/grover.py"] = {
    description: "Bridge scenes connecting colliding blocks to Grover's quantum search algorithm: both involve counting collisions/reflections that approximate pi.",
    code: `from manim_imports_ext import *


class GroverPreview(InteractiveScene):
    def construct(self):
        # Setup blocks
        block_spacing = 1.0
        blocks = Rectangle(12, 1).get_grid(12, 1, v_buff=block_spacing)
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(BLUE, 1)
        blocks.move_to(2.5 * UP, UP)

        for block, tex, color in zip(blocks, it.cycle(["U_k", "U_s"]), it.cycle([BLUE, GREEN])):
            label = Tex(tex)
            label.set_color(color)
            label.move_to(block)
            block.set_stroke(color, 1)
            block.add(label)

        self.add(blocks)

        # Wires
        d = 100
        wires = VGroup(
            *Line(UP, DOWN).set_height(1).replicate(3),
            Tex(R"\\dots"),
            *Line(UP, DOWN).set_height(1).replicate(3),
            Tex(R"\\dots"),
            *Line(UP, DOWN).set_height(1).replicate(3),
        )
        wires.arrange(RIGHT)
        wires.arrange_to_fit_width(11)
        wires.next_to(blocks, UP, buff=0)
        true_wires = VGroup(w for w in wires if isinstance(w, Line))
        syms = ["1", "2", "3", "k-1", "k", "k+1", "98", "99", "100"]
        kets = VGroup(Tex(Rf"|{sym}\\rangle") for sym in syms)
        for ket, line in zip(kets, true_wires):
            ket.set_height(0.35)
            ket.next_to(line, UP)

        key_index = 4
        kets[key_index].set_color(YELLOW)

        for block in blocks:
            block.wires = wires.copy()
            for wire in block.wires:
                if isinstance(wire, Line):
                    wire.set_height(block_spacing)
            block.wires.next_to(block, DOWN, buff=0)
            block.add(block.wires)

        self.add(wires)
        self.add(kets)

        # Add numbers
        amplitudes = VGroup(
            DecimalNumber(0.1, font_size=24, include_sign=True, num_decimal_places=3).next_to(wire, LEFT, buff=0.1)
            for wire in true_wires
        )
        amplitudes[key_index].set_color(YELLOW)

        diag_axis = np.array([math.sqrt(0.99), 0.1, 0])
        state_point = Point(diag_axis)

        def get_new_amplitudes():
            result = amplitudes.copy()
            value = math.sqrt(state_point.get_x()**2 / (d - 1))
            for dec in result:
                dec.set_value(value)
            result[key_index].set_value(state_point.get_y())
            return result

        self.add(amplitudes, state_point)

        # Repeatedly cycle
        curr_amplitudes = amplitudes
        frame = self.frame
        frame.set_y(1)

        for block, axis in zip(blocks, it.cycle([RIGHT, diag_axis])):
            state_point.flip(axis=axis, about_point=ORIGIN)
            new_amplitudes = get_new_amplitudes()
            new_amplitudes.match_y(block.wires)

            self.play(
                FadeOutToPoint(curr_amplitudes.copy(), block[0].get_center(), lag_ratio=0.0025, time_span=(0, 2)),
                FadeInFromPoint(new_amplitudes, block[0].get_center(), lag_ratio=0.0025, time_span=(1, 3.0)),
                frame.animate.set_y(min(block.wires.get_y() + 1, 1)),
                run_time=3,
            )

            curr_amplitudes = new_amplitudes


class ClassicalSearch(InteractiveScene):
    def construct(self):
        # Test
        in_out = VGroup(ArrowTip(), ArrowTip(angle=PI))
        in_out.arrange(RIGHT, buff=0.5)
        in_out.set_shape(1.75, 0.75)
        box = Square().set_shape(1.5, 1)
        machine = Union(in_out, box)
        machine.set_fill(GREY_E, 1).set_stroke(GREY_B, 2)
        machine.set_z_index(1)
        self.add(machine)

        items = VGroup(Integer(n) for n in range(25))
        items.arrange(DOWN, buff=0.5)
        items.next_to(box, LEFT, LARGE_BUFF)
        items.shift((box.get_y() - items[0].get_y()) * UP)

        self.add(items)

        # Loop through
        key = 12
        last_sym = VMobject()
        for n in range(key + 1):
            item = items[n]
            self.play(
                items.animate.shift((box.get_y() - item.get_y()) * UP),
                FadeOut(last_sym)
            )
            sym = Checkmark().set_color(GREEN) if n == key else Exmark().set_color(RED)
            sym.set_height(0.75)
            sym.next_to(box, RIGHT, LARGE_BUFF)
            self.play(
                FadeOutToPoint(item.copy(), box.get_right(), time_span=(0, 0.7)),
                FadeInFromPoint(sym, box.get_left(), time_span=(0.3, 1)),
            )
            last_sym = sym

        rect = SurroundingRectangle(items[key])
        rect.set_stroke(GREEN, 3)
        self.play(
            items[:n].animate.set_opacity(0.5),
            ShowCreation(rect),
            items[n + 1:].animate.set_opacity(0.5),
        )`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "GroverPreview extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      14: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      26: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      28: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      36: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      57: "DecimalNumber displays a formatted decimal that can be animated. Tracks a value and auto-updates display.",
      85: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      88: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      95: "ClassicalSearch extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      96: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      107: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      119: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      120: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      121: "FadeOut transitions a mobject from opaque to transparent.",
      126: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      134: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      135: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      136: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      137: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
    }
  };

  files["_2025/colliding_blocks_v2/supplements.py"] = {
    description: "Supplementary scenes for the colliding blocks video: additional phase space diagrams, conservation of energy/momentum, and limiting behavior.",
    code: `from manim_imports_ext import *


class ShowPastVideos(InteractiveScene):
    def construct(self):
        # Show the video
        title = Text("2019 Video:", font_size=72)
        title[-1].set_opacity(0)
        title.to_edge(UP)
        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 1)
        screen = ScreenRectangle().set_height(6)
        screen.next_to(title, DOWN)
        screen.set_fill(BLACK, 1).set_stroke(WHITE, 2)

        vertical_frame = screen.copy().set_shape(7 * 9 / 16, 7)
        vertical_frame.to_corner(UR)
        vertical_frame.match_style(screen)

        self.add(background, screen)
        self.play(
            Write(title),
            VShowPassingFlash(screen.copy().set_fill(opacity=0).set_stroke(BLUE, 4).insert_n_curves(20), time_width=1.5, run_time=3),
        )
        self.wait()

        # Three versions
        versions = VGroup(
            Text("2019 video: "),
            Text("Adapted as a short:\\n(separate channel) ", alignment="LEFT"),
            Text("Re-posted to this channel: "),
        )
        versions[1]["(separate channel)"].set_fill(GREY_B)
        versions.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        versions.to_corner(UL)

        counts = VGroup(
            Text("13.8M Views").set_color(YELLOW),
            Text("42.6M Views").set_color(ORANGE),
            Text("62.0M Views").set_color(RED),
        )
        for version, count in zip(versions, counts):
            count.next_to(version, RIGHT, aligned_edge=UP)

        explanation = VGroup(
            Text("My only reason for making shorts at all"),
            Text("is to pique the curiosity of people in the shorts feed and"),
            Text("encourage them to pop out for a full explanation. Originally I didn't"),
            Text("want to trash up this channel with shorts, but then the only way to link to"),
            Text("link to a long-form videos was if the short lived on the same channel.  ¯\\\\_(ツ)_/¯"),
        )
        for part in explanation:
            part.match_width(versions[2])
        explanation.arrange(DOWN, buff=0.15)
        explanation.set_fill(GREY_B)
        explanation.next_to(versions[2], DOWN, buff=0.5)

        self.play(
            Transform(title, versions[0]),
            FadeIn(counts[0]),
            Transform(screen, vertical_frame)
        )
        for i in [1, 2]:
            self.play(
                FadeIn(versions[i]),
                FadeIn(counts[i], lag_ratio=0.1,)
            )
        self.play(FadeIn(explanation, lag_ratio=0.01, run_time=1))
        self.wait()


class ConfettiSpiril(Animation):
    x_start = 0
    spiril_radius = 0.5
    num_spirals = 4
    run_time = 10
    rate_func = None

    def __init__(self, mobject, **kwargs):
        x_start = kwargs.pop("x_start", self.x_start)
        self.num_spirals = kwargs.pop("num_spirals", self.num_spirals)
        mobject.next_to(x_start * RIGHT + FRAME_Y_RADIUS * UP, UP)
        self.total_vert_shift = FRAME_HEIGHT + mobject.get_height() + 2 * MED_SMALL_BUFF

        super().__init__(mobject, **kwargs)

    def interpolate_submobject(self, submobject, starting_submobject, alpha):
        submobject.set_points(starting_submobject.get_points())

    def interpolate_mobject(self, alpha):
        Animation.interpolate_mobject(self, alpha)
        angle = alpha * self.num_spirals * TAU
        vert_shift = alpha * self.total_vert_shift

        start_center = self.mobject.get_center()
        self.mobject.shift(self.spiril_radius * OUT)
        self.mobject.rotate(angle, axis=UP, about_point=start_center + 0.5 * RIGHT)
        self.mobject.shift(vert_shift * DOWN)


class Confetti(InteractiveScene):
    def construct(self):
        # Test
        num_confetti_squares = 300
        colors = [RED, YELLOW, GREEN, BLUE, PURPLE, RED]
        confetti_squares = [
            Square(
                side_length=0.2,
                stroke_width=0,
                fill_opacity=0.75,
                fill_color=random.choice(colors),
            )
            for x in range(num_confetti_squares)
        ]
        confetti_spirils = [
            ConfettiSpiril(
                square,
                x_start=2 * random.random() * FRAME_X_RADIUS - FRAME_X_RADIUS,
                num_spirals=np.random.uniform(-5, 5),
            )
            for square in confetti_squares
        ]

        self.play(LaggedStart(*confetti_spirils, lag_ratio=1e-2, run_time=10))


class HappyPiDay(TeacherStudentsScene):
    def construct(self):
        # Test
        title = Text("Happy Pi Day!", font_size=72)
        title.to_edge(UP)
        morty = self.teacher
        morty.change_mode("surprised")

        self.play(
            Write(title),
            morty.change("hooray", look_at=title),
            self.change_students("hooray", "surprised", "jamming", look_at=title)
        )
        self.wait()
        self.play(
            morty.change("tease", 2 * UP),
            self.change_students("tease", "surprised", "well", look_at=3 * UR)
        )
        self.wait(4)


class Leftrightarrow(InteractiveScene):
    def construct(self):
        arrow = Tex(R"\\longleftrightarrow", font_size=90)
        self.play(GrowFromCenter(arrow, run_time=2))
        self.wait(2)


class GroversAlgorithmLabel(InteractiveScene):
    def construct(self):
        labels = VGroup(
            TexText("Quantum Computing"),
            TexText("Grover's Algorithm"),
        )
        self.play(FadeIn(labels[0], 0.5 * UP))
        self.wait()
        self.play(
            labels[0].animate.shift(UP),
            FadeIn(labels[1], 0.5 * UP)
        )
        self.wait()


class UnsolvedReference(InteractiveScene):
    def construct(self):
        rect = Rectangle(8, 1.25)
        rect.set_stroke(RED, 4)
        label = Text("Unsolved", font_size=60)
        label.set_color(RED)
        label.next_to(rect, UP, buff=MED_SMALL_BUFF)

        self.play(ShowCreation(rect), FadeIn(label, 0.25 * UP))
        self.wait()


class Recap(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        self.play(morty.says(TexText("Let's Recap"), look_at=DL))
        self.wait()


class RewindArrows(InteractiveScene):
    def construct(self):
        # Test
        arrows = ArrowTip(angle=PI).get_grid(1, 3)
        arrows.scale(2)
        self.play(LaggedStartMap(FadeIn, arrows, lag_ratio=0.1, shift=0.5 * LEFT, run_time=1))
        self.wait()


class CommentOnElastic(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.says("Assume perfectly\\nelastic collisions"),
            self.change_students("pondering", "erm", "tease", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            stds[2].says(
                "Then there should\\nbe no sound",
                mode="sassy",
                bubble_direction=LEFT,
                look_at=morty.eyes
            ),
            morty.debubble("guilty"),
        )
        self.play(self.change_students("erm", "angry", "sassy", look_at=morty.eyes))
        self.wait(3)


class WritePiDigits(InteractiveScene):
    def construct(self):
        eq = Tex(R"\\pi = 3.14159265358 \\dots")
        self.play(FadeIn(eq, lag_ratio=0.25, run_time=4))
        self.wait()


class ReactToQuantumComparisson(InteractiveScene):
    def construct(self):
        randy = Randolph(height=3)
        randy.to_edge(LEFT).shift(DOWN)
        randy.body.insert_n_curves(100)

        self.play(randy.change("sassy"))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("awe"))
        self.play(Blink(randy))
        self.wait(2)
        self.play(randy.change("confused"))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class LoadSolutionIntoHead(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenFadeRectangle().set_fill(interpolate_color(GREY_E, BLACK, 0.5), 1))
        randy = Randolph()
        randy.to_edge(DOWN, buff=0.25)
        bubble = ThoughtBubble(direction=RIGHT, filler_shape=(4, 2.5)).pin_to(randy)
        bubble[0][-1].set_fill(GREEN_SCREEN, 1)

        self.play(
            ShowCreation(bubble),
            randy.change("concentrating", 3 * UR)
        )
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class ShowMassRatioToCountChart(InteractiveScene):
    def construct(self):
        # Chart
        n_terms = 6
        v_line = Line(UP, DOWN).set_height(6)
        points = [v_line.pfp(a) for a in np.linspace(0, 1, n_terms + 2)]
        h_lines = VGroup(Line(LEFT, RIGHT).set_width(7).move_to(p) for p in points)

        VGroup(v_line, h_lines).set_stroke(WHITE, 2)
        self.add(v_line, h_lines[1:-1])

        # Content
        titles = VGroup(
            Text("Mass ratio"),
            Text("#Collisions"),
        )
        mass_ratios = VGroup(
            Integer(int(10**n), unit=":1", font_size=42)
            for n in range(0, 2 * n_terms, 2)
        )
        counts = VGroup(
            Integer(int(PI * 10**n), font_size=42)
            for n in range(0, n_terms)
        )
        for point, ratio, count in zip(points[2:], mass_ratios, counts):
            ratio.next_to(point, UL, buff=0.2).shift(0.1 * LEFT)
            count.next_to(point, UR, buff=0.2).shift(0.1 * RIGHT)

        titles[0].next_to(points[1], UL).shift(0.25 * LEFT)
        titles[1].next_to(points[1], UR).shift(0.25 * RIGHT)
        titles[0].align_to(titles[1], UP)

        self.add(titles)
        for ratio, count in zip(mass_ratios, counts):
            ratio.align_to(count, UP)
            self.play(FadeIn(ratio))
            self.wait()
            self.play(TransformFromCopy(ratio, count))
            self.wait()


class StateThePuzzle(InteractiveScene):
    def construct(self):
        # Test
        question = TexText(
            R"Given $m_1$ and $m_2$, how \\\\ many collisions take place?",
            t2c={R"$m_1$": BLUE, R"$m_2$": BLUE}
        )
        question.to_edge(UP)
        self.play(Write(question))
        self.wait()


class EnergyAndMomentumLaws(InteractiveScene):
    def construct(self):
        # Test
        laws = VGroup(
            Text("Conservation of Energy"),
            Text("Conservation of Momentum"),
        )
        laws.arrange(DOWN, buff=0.75)
        laws.move_to(3 * UP, UP)

        self.play(FadeIn(laws[0], UP))
        self.wait()
        self.play(
            TransformMatchingStrings(
                *laws,
                key_map={"Energy": "Momentum"},
                mismatch_animation=FadeTransform,
                run_time=1
            )
        )
        self.wait()


class ProblemSolvingPrinciplesWithPis(TeacherStudentsScene):
    def construct(self):
        # Prep
        morty = self.teacher
        stds = self.students
        for pi in [morty, *stds]:
            pi.body.insert_n_curves(100)

        # Title
        title = Text("Problem-solving principles", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.1)
        VGroup(title, underline).set_color(BLUE)
        title.set_backstroke(width=3)

        # Items
        items = BulletedList(
            "Try a simpler version of the problem",
            "Use the defining features of the problem",
            "List any equations that might be relevant",
            "Seek symmetry",
            "Compute something (anything) to build intuition",
            "Run simulations (if possible) to build intuition",
            "Draw pictures!",
        )
        items.next_to(title, DOWN, buff=0.35)
        items.to_edge(LEFT, buff=0.75)
        items.save_state()

        # Have students toss up examples
        np.random.seed(0)
        for y, item in enumerate(items):
            item.set_height(0.25)
            item.set_opacity(0.8)
            item[0].set_opacity(0)
            item.next_to(underline, DOWN)
            item.shift(np.random.uniform(-5, 5) * RIGHT)
            item.shift(0.5 * y * DOWN)

        self.play(
            morty.change("raise_right_hand", underline),
            self.change_students("pondering", "pondering", "pondering", look_at=underline),
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline),
        )
        self.play(morty.change("tease", stds))
        self.wait()
        index_mode_pairs = [
            (0, "raise_right_hand"),
            (2, "raise_right_hand"),
            (1, "raise_right_hand"),
            (2, "raise_left_hand"),
            (0, "raise_left_hand"),
            (1, "raise_left_hand"),
            (2, "raise_right_hand"),
        ]
        for item, pair in zip(items, index_mode_pairs):
            index, mode = pair
            if mode == "raise_left_hand":
                item.shift(2 * LEFT).shift_onto_screen()
            self.play(
                stds[index].change(mode, item),
                FadeIn(item, scale=3, shift=item.get_center() - stds[index].get_center())
            )
        self.wait(3)

        # Return to position
        self.play(
            Restore(items, run_time=2, lag_ratio=1e-3),
            FadeOut(self.pi_creatures),
            FadeOut(self.background),
        )
        self.wait()

        # Isolate a few items
        for index in [2, 6, 3]:
            self.play(
                items.animate.fade_all_but(index),
            )
            self.wait()


class StaysConstant(InteractiveScene):
    word = "Unchanged!"
    color = YELLOW

    def construct(self):
        # Note the change
        unchanged_label = Text(self.word)
        unchanged_label.set_color(self.color)
        unchanged_label.next_to(ORIGIN, DOWN, buff=1.5)
        unchanged_label.shift(0.5 * RIGHT)
        unchanged_arrow = Arrow(unchanged_label, ORIGIN, buff=0.25)
        unchanged_arrow.set_fill(self.color)

        self.play(
            FadeIn(unchanged_label),
            GrowArrow(unchanged_arrow),
        )
        self.wait()
        self.play(FadeOut(VGroup(unchanged_label, unchanged_arrow)))


class NoteChange(StaysConstant):
    word = "Changed!"
    color = RED


class SimpleArrow(InteractiveScene):
    def construct(self):
        arrow = Vector(0.5 * DL)
        self.play(GrowArrow(arrow))
        self.wait()


class KeyStep(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("This is the\\nkey step"),
            self.change_students("pondering", "thinking", "erm", look_at=self.screen)
        )
        self.wait()
        self.play(self.change_students("thinking", "well", "pondering", look_at=self.screen))
        self.wait(3)


class StateSpaceLabel(InteractiveScene):
    def construct(self):
        # Test
        title = TexText("\`\`State Space''", font_size=60)
        title.to_corner(UL)
        arrow = Arrow(title.get_bottom() + 0.1 * DOWN + 1.0 * RIGHT, 1.0 * UP, thickness=5)
        arrow.set_color(YELLOW)
        title.set_color(YELLOW)

        self.play(
            Write(title),
            GrowArrow(arrow)
        )
        self.wait()


class HoldUpEllipseVsCircle(InteractiveScene):
    def construct(self):
        # Show pi
        radius = 3
        circle = Circle(radius=radius)
        circle.set_stroke(WHITE, 3)

        radial_line = Line()
        radial_line.set_stroke(BLUE, 3)
        radial_line.f_always.put_start_and_end_on(lambda: ORIGIN, circle.get_end)
        radial_label = Tex(R"1", font_size=48)
        radial_label.add_updater(lambda m: m.move_to(
            radial_line.get_center() + 0.1 * rotate_vector(radial_line.get_vector(), -90 * DEG),
        ))

        theta_tracker = ValueTracker(0)
        get_theta = theta_tracker.get_value
        arc = always_redraw(lambda: Arc(0, get_theta(), radius=radius).set_stroke(RED, 5))
        arc_len_label = DecimalNumber(0, show_ellipsis=True, num_decimal_places=5)
        arc_len_label.set_color(RED)
        arc_len_label.f_always.set_value(get_theta)
        arc_len_label.f_always.move_to(lambda: 1.4 * arc.get_end())

        self.add(radial_line, radial_label)
        self.play(ShowCreation(circle, run_time=3))
        self.add(arc, arc_len_label)
        self.play(theta_tracker.animate.set_value(PI), run_time=3)
        self.wait()
        circle_group = VGroup(circle, radial_line, radial_label, arc, arc_len_label)
        circle_group.clear_updaters()

        # Make the comparison
        morty = Mortimer(mode="raise_right_hand").flip()
        morty.to_edge(DOWN)

        circle_group.target = circle_group.generate_target()
        circle_group.target.set_height(3)
        circle_group.target.next_to(morty.get_corner(UR), UP, buff=0.5)
        circle_group.target[1:].set_opacity(0)

        self.play(
            MoveToTarget(circle_group),
            VFadeIn(morty),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("raise_left_hand", look_at=2 * UL))
        for _ in range(2):
            self.play(Blink(morty))
            self.wait(2)


class AskWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[2].says("...why?", mode="confused", bubble_direction=LEFT, look_at=self.screen),
            self.teacher.change("tease"),
            self.change_students("pondering", "thinking", look_at=self.screen)
        )
        self.wait()
        self.play(
            self.teacher.says("Symmetry breeds\\ninsight!", mode="hooray"),
            self.students[2].debubble(look_at=self.teacher.eyes)
        )
        self.play(self.change_students("thinking", "tease", "erm", self.teacher.eyes))
        self.wait()


class PiTime1e5(InteractiveScene):
    def construct(self):
        # Test
        value = Integer(PI * 1e5, font_size=60)
        self.play(Write(value, lag_ratio=0.2))
        self.wait()


class HighlightTheSlope(TeacherStudentsScene):
    def construct(self):
        # Test
        slope = Tex(R"-\\sqrt{m_1 \\over m_2}", t2c={"m_1": BLUE, "m_2": BLUE}, font_size=60)
        slope.next_to(self.hold_up_spot, UP)
        rect = SurroundingRectangle(slope)

        sqrt_highlight = slope[R"\\sqrt"][0].copy()
        sqrt_highlight.set_stroke(YELLOW, 4).set_fill(opacity=0)
        sqrt_highlight.insert_n_curves(100)

        self.play(
            self.teacher.change("raise_right_hand", slope),
            self.change_students("erm", "sassy", "well", look_at=slope),
            FadeIn(slope, UP),
        )
        self.wait()
        self.play(
            FlashAround(slope),
            ShowCreation(rect),
            self.change_students("pondering", "pondering", "thinking", look_at=slope)
        )
        self.wait()

        self.remove(rect)
        rect_copy = rect.replicate(2)
        self.play(Transform(rect_copy, sqrt_highlight))
        self.wait()
        self.play(FadeOut(rect_copy))
        self.wait(4)


class MostOfTheReasoning(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(height=2.5).flip()
        morty.to_corner(DL)
        self.play(morty.says("This is most of\\nthe physics"))
        self.play(Blink(morty))
        self.wait(2)


class StareAtDiagram(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            self.change_students("well", "happy", look_at=self.screen),
            stds[2].says("I can see\\nmyself there", mode="tease", look_at=self.screen, bubble_direction=LEFT),
            morty.change("tease")
        )
        self.wait(4)
        self.play(
            stds[2].debubble(mode="pondering", look_at=3 * UP),
            self.change_students("pondering", "pondering", look_at=3 * UP),
            morty.change("raise_right_hand"),
            FadeOut(self.background),
        )
        self.wait()
        self.play(stds[1].change("concentrating", look_at=3 * UP))
        self.wait()
        self.play(LaggedStart(
            stds[1].change("hooray"),
            stds[0].change("erm", look_at=stds[1].eyes),
            stds[2].change("erm", look_at=stds[1].eyes),
            morty.change("tease", look_at=stds[1].eyes),
            lag_ratio=0.25,
        ))
        self.wait()


class AskHowThisIsHelpful(InteractiveScene):
    def construct(self):
        randy = Randolph()
        randy.to_corner(DL)
        self.play(randy.says("Why is this\\nhelpful?"))
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.wait()


class ISeeWhereThisIsGoing(TeacherStudentsScene):
    def construct(self):
        # Setup
        morty = self.teacher
        stds = self.students
        std = stds[1]
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(std.says("I see!", mode="hooray"))
        self.play(morty.change("hesitant"))

        # Show the progression
        mass_terms = VGroup(
            Tex(R"m_1 = 100^n", t2c={"m_1": BLUE}),
            Tex(R"m_2 = 1", t2c={"m_2": BLUE}),
        )
        mass_terms.arrange(DOWN, aligned_edge=LEFT)
        mass_terms.next_to(std.get_corner(UL), UP, buff=0.5)

        implies = Tex(R"\\Longrightarrow", font_size=72)
        implies.next_to(mass_terms, RIGHT)
        q_mark = Tex(R"?").next_to(implies, UP, SMALL_BUFF)
        theta_eq = Tex(R"\\theta=(0.1)^n")
        theta_eq.next_to(implies, RIGHT)

        group = VGroup(mass_terms, implies, q_mark, theta_eq)

        self.play(
            std.debubble(mode="raise_left_hand", look_at=mass_terms),
            FadeIn(mass_terms, UL, scale=2),
            stds[0].change("hesitant", mass_terms),
            stds[2].change("erm", mass_terms),
        )
        self.wait()
        self.play(
            std.change("raise_right_hand", look_at=theta_eq),
            Write(implies),
            Write(q_mark),
            FadeTransform(mass_terms[0].copy(), theta_eq),
            stds[0].animate.look_at(theta_eq),
            stds[2].animate.look_at(theta_eq),
        )
        self.wait()
        self.play(
            std.change("hooray", group),
            group.animate.set_height(0.75).to_edge(UP),
        )
        self.wait(3)
        self.play(
            morty.says("Almost", mode="well"),
            self.change_students("pondering", "angry", "confused", look_at=morty.eyes)
        )
        self.wait(3)

        # Final statement
        self.play(
            FadeOut(morty.bubble),
            morty.says("One final bit\\nof reasoning", mode="speaking"),
            self.change_students("thinking", "hesitant", "happy"),
        )
        self.wait(3)


class ReferenceSmallAngleApproximations(TeacherStudentsScene):
    def construct(self):
        self.add_title()
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Test
        equation1 = Tex(R"\\arctan(x) \\approx x")
        equation1[R"\\arctan"].set_color(GREEN)
        equation2 = Tex(R"x \\approx \\tan(x)")
        equation2[R"\\tan"].set_color(YELLOW)
        for mob in equation1, equation2:
            mob.move_to(self.hold_up_spot, DOWN)

        self.play(
            FadeIn(equation1, UP),
            self.teacher.change("raise_right_hand"),
            self.change_students(
                "erm", "sassy", "confused"
            )
        )
        self.look_at(3 * UL)
        self.play(equation1.animate.shift(UP))
        self.play(
            TransformMatchingTex(
                equation1.copy(),
                equation2,
                lag_ratio=0.01,
                key_map={R"\\arctan": R"\\tan"}
            )
        )
        self.play(self.change_students("confused", "erm", "sassy"))
        self.look_at(3 * UL)
        self.wait(3)

    def add_title(self):
        title = TexText("For small $x$")
        subtitle = TexText("(e.g. $x = 0.001$)")
        subtitle.scale(0.75)
        subtitle.next_to(title, DOWN)
        title.add(subtitle)
        title.move_to(self.hold_up_spot)
        title.to_edge(UP)
        self.add(title)


class ExplainSmallAngleApprox(InteractiveScene):
    def construct(self):
        # Setup
        frame = self.frame
        height = 7.5
        axes = Axes(
            (-2, 2),
            (-2, 2),
            height=height,
            width=height,
            axis_config=dict(include_tip=True)
        )
        axes.shift(-axes.get_origin())
        unit_size = axes.x_axis.get_unit_size()

        circle = Circle(radius=unit_size)
        circle.set_stroke(WHITE, 3)
        point = GlowDot(color=RED)
        point.move_to(circle.get_right())

        radial_line = Line()
        radial_line.set_stroke(BLUE, 3)
        radial_line.f_always.put_start_and_end_on(
            axes.get_origin,
            point.get_center,
        )
        radial_label = Tex(R"1", font_size=36)
        radial_label.add_updater(lambda m: m.move_to(
            radial_line.get_center() + 0.15 * rotate_vector(radial_line.get_vector(), -90 * DEG),
        ))

        self.add(axes)
        self.add(radial_line)
        self.add(radial_label)
        self.play(
            ShowCreation(circle),
            UpdateFromFunc(point, lambda m: m.move_to(circle.get_end())),
            frame.animate.set_height(5),
            run_time=3
        )
        self.wait()

        # Show angle
        theta_color = YELLOW
        x_color = RED
        y_color = GREEN

        h_radial_line = radial_line.copy().clear_updaters()
        radial_label.clear_updaters()
        theta_tracker = ValueTracker(1e-2)
        get_theta = theta_tracker.get_value
        point.add_updater(lambda m: m.move_to(unit_size * rotate_vector(RIGHT, get_theta())))

        arc = always_redraw(lambda: Arc(0, get_theta(), radius=0.5))
        theta_label = Tex(R"\\theta", font_size=36)
        theta_label.set_color(theta_color)
        theta_label.f_always.set_height(lambda: clip(arc.get_height(), 1e-2, 0.30))
        theta_label.add_updater(lambda m: m.next_to(arc.pfp(0.65), RIGHT, buff=0.075))

        tan_eq = Tex(
            R"\\tan(\\theta) = {{y} \\over {x}} \\approx {\\theta \\over 1}",
            t2c={R"\\theta": theta_color, "{x}": x_color, "{y}": y_color},
        )
        tan_eq.fix_in_frame()
        tan_eq.to_corner(UR).shift(LEFT)

        self.add(h_radial_line)
        self.add(arc)
        self.add(theta_label)
        self.play(
            theta_tracker.animate.set_value(30 * DEG),
            radial_line.animate.set_stroke(WHITE, 3),
            run_time=1
        )
        self.play(
            TransformFromCopy(theta_label.copy().clear_updaters(), tan_eq[R"\\theta"][0]),
            Write(tan_eq[R"\\tan("]),
            Write(tan_eq[R") ="]),
        )
        self.wait()

        # Show lines
        x_line = Line().set_stroke(x_color, 4)
        y_line = Line().set_stroke(y_color, 4)
        x_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.get_origin(), axes.c2p(math.cos(get_theta()), 0)
        ))
        y_line.add_updater(lambda m: m.put_start_and_end_on(
            axes.c2p(math.cos(get_theta()), 0),
            axes.c2p(math.cos(get_theta()), math.sin(get_theta()))
        ))

        x_label = Tex(R"x", font_size=30).set_color(x_color)
        y_label = Tex(R"y", font_size=30).set_color(y_color)
        VGroup(x_label, y_label).set_backstroke(BLACK, 3)
        x_label.always.next_to(x_line, DOWN, SMALL_BUFF)
        y_label.always.next_to(y_line, RIGHT, buff=0.05)

        self.play(LaggedStart(
            ShowCreation(y_line, suspend_mobject_updating=True),
            VFadeIn(y_label),
            FadeOut(h_radial_line),
            FadeOut(radial_label),
            ShowCreation(x_line, suspend_mobject_updating=True),
            VFadeIn(x_label),
            lag_ratio=0.15
        ))
        self.play(LaggedStart(
            TransformFromCopy(y_label.copy().clear_updaters(), tan_eq["{y}"][0]),
            Write(tan_eq[R"\\over"][0]),
            TransformFromCopy(x_label.copy().clear_updaters(), tan_eq["{x}"][0]),
            lag_ratio=0.5
        ))
        self.wait()

        # Shrink down
        self.play(theta_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

        # Analogy
        lil_rect = SurroundingRectangle(tan_eq["{x}"])
        self.play(
            Write(tan_eq[R"\\approx"]),
            TransformFromCopy(tan_eq[R"\\over"][0], tan_eq[R"\\over"][1]),
            ShowCreation(lil_rect),
        )
        self.play(
            TransformFromCopy(tan_eq["{x}"][0], tan_eq["1"][0]),
            lil_rect.animate.surround(tan_eq["1"]),
        )
        self.wait()
        self.play(
            lil_rect.animate.surround(tan_eq["{y}"]),
            y_label.animate.scale(0.5),
            x_label.animate.scale(0.5),
            point.animate.set_radius(0.25 * point.get_radius()),
            frame.animate.reorient(0, 0, 0, (1.46, -0.0, 0.0), 1.87).set_anim_args(run_time=3),
        )
        self.play(
            TransformFromCopy(tan_eq["{y}"][0], tan_eq[R"\\theta"][1]),
            lil_rect.animate.surround(tan_eq[R"\\theta"][1]),
        )
        self.play(FadeOut(lil_rect))

        # More shrinking!
        self.play(theta_tracker.animate.set_value(1e-2), run_time=8)


class AngryStudents(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("angry", "confused", "sassy"),
            self.teacher.change("guilty"),
        )
        self.wait(3)


class DigitsOfPi(InteractiveScene):
    def construct(self):
        # Test
        pi_text = Path(Path(__file__).parent, "digits_of_pi.txt").read_text()
        pi_text_lines = pi_text.split("\\n")
        one_line_pi = Tex(R"\\pi = " + pi_text_lines[0])
        one_line_pi[0].scale(1.5)
        self.play(FadeIn(one_line_pi, lag_ratio=0.25, run_time=5))
        self.wait()

        # Second half
        index = 25
        nines = Tex(R"\\dots999999999999999999999 \\dots")
        nines.next_to(one_line_pi[:index], RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)

        prefix_brace = Brace(VGroup(one_line_pi[2:index], nines[:3]), UP, MED_SMALL_BUFF)
        prefix_label = prefix_brace.get_text("First n digits")
        nines_brace = Brace(nines[3:], UP, MED_SMALL_BUFF)
        nines_label = nines_brace.get_text("Next n digits")

        self.play(
            FadeOut(one_line_pi[index:]),
            FadeIn(nines, lag_ratio=0.25, run_time=3),
        )
        self.wait()
        self.play(LaggedStart(
            GrowFromCenter(prefix_brace),
            Write(prefix_label),
            GrowFromCenter(nines_brace),
            Write(nines_label),
            lag_ratio=0.25
        ))
        self.wait()

        # Show full pi
        frame = self.frame
        pi_tex = pi_text[2:].replace("\\n", R"\\\\&")
        full_pi = Tex(R"\\pi = 3.&" + pi_tex)
        full_pi.to_edge(UP)
        dots = Tex(R"\\vdots", font_size=72)
        dots.next_to(full_pi, DOWN)
        digits_per_line = len(pi_text_lines[-1].strip())
        big_nines = Text("\\n".join([
            "9" * digits_per_line
            for line in pi_text_lines
        ]))
        big_nines.match_width(full_pi[-digits_per_line:])
        big_nines.next_to(dots, DOWN)
        big_nines.align_to(full_pi, RIGHT)
        big_nines.set_color(RED)

        self.play(
            ReplacementTransform(one_line_pi[:index], full_pi[:index]),
            FadeIn(full_pi[index:], lag_ratio=1e-3, time_span=(0.5, 3)),
            FadeOut(VGroup(prefix_brace, prefix_label, nines_brace, nines_label, nines)),
            FadeIn(dots, time_span=(2, 3))
        )
        self.wait()
        self.play(
            frame.animate.scale(1.8, about_edge=UP),
            FadeIn(big_nines, lag_ratio=1e-2),
            run_time=6
        )
        self.wait()


class WriteExactSolution(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Answer", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, stretch_factor=1.5)
        VGroup(title, underline).set_color(YELLOW)
        self.play(
            FadeIn(title, lag_ratio=0.1),
            ShowCreation(underline)
        )

        # Answer
        kw = dict(
            t2c={
                R"\\theta": YELLOW,
                R"m_1": BLUE,
                R"m_2": BLUE,
            }
        )
        solution = VGroup(
            Tex(R"\\#\\text{Collisions} = \\lceil \\pi / \\theta - 1 \\rceil", **kw),
            Tex(R"\\text{Where } \\; \\theta = \\text{arctan}\\left(\\sqrt{m_2 / m_1}\\right)", **kw),
        )
        solution.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        solution.next_to(underline, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStartMap(FadeIn, solution, shift=0.5 * DOWN, lag_ratio=0.5))
        self.wait()


class ExactSolutionForMatt(InteractiveScene):
    def construct(self):
        # Answer
        kw = dict(
            t2c={
                R"\\theta": YELLOW,
                R"m_1": BLUE,
                R"m_2": BLUE,
                R"\\pi": WHITE
            }
        )
        equations = VGroup(
            Tex(R"\\#\\text{Collisions} = {\\pi \\over \\text{arctan}\\left(\\sqrt{m_2 / m_1}\\right)}", **kw),
            Tex(R"\\#\\text{Collisions} \\approx {\\pi \\over \\sqrt{m_2 / m_1}}", **kw),
            Tex(R"\\#\\text{Collisions} \\approx \\pi \\cdot \\sqrt{m_1 / m_2}", **kw),
        )
        for eq in equations[:2]:
            eq[R"\\pi"].scale(2, about_edge=DOWN)
        equations[2][R"\\pi"].scale(1.5, about_edge=DOWN).shift(0.05 * RIGHT)

        rect = SurroundingRectangle(equations[0][len("#Collisions="):])
        rect.set_stroke(YELLOW, 2)
        words = VGroup(
            Text("If fractional, round down").set_color(YELLOW),
            TexText("If whole...").set_color(GREEN),
        )
        for word in words:
            word.next_to(rect, UP)

        minus_1 = Tex(R"-1", font_size=60)
        minus_1.next_to(equations[0][R"\\over"], RIGHT)

        self.play(FadeIn(equations[0], lag_ratio=0.25))
        self.wait()
        self.play(ShowCreation(rect), FadeIn(words[0], 0.25 * UP))
        self.wait()
        self.play(
            rect.animate.set_color(GREEN),
            FadeTransformPieces(words[0], words[1], run_time=1),
        )
        self.play(Write(minus_1))
        self.wait()
        self.play(LaggedStart(FadeOut(rect), FadeOut(words[1]), FadeOut(minus_1)))

        # Pi creature
        randy = Randolph()
        pi = equations[0][R"\\pi"][0][0]
        randy.scale(pi.get_height() / randy.body.get_height())
        randy.shift(pi.get_center() - randy.body.get_center())

        self.play(
            FadeOut(pi),
            FadeIn(randy)
        )
        self.play(randy.change("hooray", DOWN))
        self.play(Blink(randy))
        self.play(FadeOut(randy), FadeIn(pi))

        # Cross out arctan
        exmark = Cross(equations[0][R"\\text{arctan}"])
        self.play(ShowCreation(exmark))
        self.wait()
        self.play(
            FadeOut(exmark),
            TransformMatchingTex(equations[0], equations[1]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                equations[1],
                equations[2],
                matched_keys=["m_1", "m_2", R"\\sqrt"],
                path_arc=45 * DEG
            )
        )
        self.wait()


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[2].says("Question...", mode="dance_3"),
            self.change_students("pondering", "pondering", look_at=stds[2].eyes),
            morty.change('tease'),
        )
        self.wait()
        old_bubble = stds[2].bubble
        bubble = stds[2].get_bubble("Who cares?", bubble_type=SpeechBubble)
        self.play(LaggedStart(
            FadeTransformPieces(old_bubble, bubble),
            stds[2].change("angry"),
            self.change_students("guilty", "hesitant"),
            morty.change("hesitant"),
            lag_ratio=0.25
        ))
        self.play(LaggedStart(
            stds[1].animate.look_at(stds[0].eyes),
            stds[0].change("maybe")
        ))
        self.play(Blink(stds[0]))
        self.wait(5)


class SimplifyingMessiness(InteractiveScene):
    samples = 4

    def construct(self):
        # Test
        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 3)

        rough_points = np.array([np.random.uniform(0.95, 1.05) * circle.pfp(a) for a in np.linspace(0, 1, 200)])
        rough_points[-1] = rough_points[0]
        rough_circle = VMobject().set_points_as_corners(rough_points)

        circles = VGroup(rough_circle, circle)
        circles.arrange(RIGHT, buff=3.0)

        arrows = VGroup(
            Arrow(rough_circle.get_right(), circle.get_left(), path_arc=-sgn * 90 * DEG, thickness=5).shift(sgn * UP)
            for sgn in [+1, -1]
        )
        mid_arrow = Arrow(rough_circle, circle, buff=0.25, thickness=5, path_arc=1 * DEG)

        self.play(ShowCreation(rough_circle))
        self.wait()
        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.7),
            TransformFromCopy(rough_circle, circle, run_time=2),
        )
        self.wait()
        self.play(ReplacementTransform(arrows[0], mid_arrow))
        self.wait()
        self.play(ReplacementTransform(mid_arrow, arrows[1]))
        self.play(
            circle.animate.center(),
            FadeOut(rough_circle, 2 * LEFT),
            FadeOut(arrows[1], 2 * LEFT),
        )
        self.wait()


class HiddenConnections(InteractiveScene):
    def construct(self):
        # Transition from circle
        circle = Circle(radius=2)
        circle.set_stroke(WHITE, 3)
        self.add(circle)
        self.wait()

        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 0.5)
        background.set_z_index(-1)

        # Screens
        screens = ScreenRectangle(height=3.0).get_grid(2, 2, h_buff=2.0, v_buff=1.5)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 1)
        screens.set_height(7.5)

        q_marks = VGroup(
            Tex(R"???", font_size=72).move_to(screen)
            for screen in screens
        )

        connections = VGroup()
        for s1, s2 in it.combinations(screens, 2):
            vect = s2.get_center() - s1.get_center()
            line = Line(s1.get_corner(vect), s2.get_corner(-vect))
            connections.add(line)
        connections.set_stroke(BLUE, 3)
        screens[1:].set_fill(GREY_E, 0.5)

        self.remove(circle)
        self.play(
            FadeIn(background),
            ReplacementTransform(circle.replicate(len(connections)), connections, lag_ratio=0.1),
            LaggedStartMap(FadeIn, screens, scale=1.25),
            Write(q_marks[1:], time_span=(1, 3)),
            run_time=3
        )
        self.wait()

        # Expose individual screens
        self.play(
            FadeOut(q_marks[1]),
            screens[1].animate.set_fill(BLACK, 1),
            connections[1:].animate.set_stroke(width=1, opacity=0.5),
        )
        self.wait()
        self.play(
            FadeOut(q_marks[2]),
            screens[2].animate.set_fill(BLACK, 1),
            connections[1:4:2].animate.set_stroke(width=3, opacity=1),
        )
        self.wait()
        self.play(
            FadeOut(q_marks[3]),
            screens[3].animate.set_fill(BLACK, 1),
            connections.animate.set_stroke(width=3, opacity=1),
        )
        self.wait()


class WebOfConnections(InteractiveScene):
    n_points = 1000

    def setup(self):
        super().setup()

        # Points
        points = np.random.uniform(-1, 1, (self.n_points, 3))

        curr_norms = np.linalg.norm(points, axis=1)
        target_norms = curr_norms**1.5
        points *= (target_norms / curr_norms)[:, np.newaxis]

        points = points[curr_norms < 1]
        points = np.vstack([[ORIGIN], points])  # Ensure origin is in there
        points *= 25

        dots = DotCloud(points)
        dots.stretch(0.2, 2, about_point=ORIGIN)
        dots.set_radius(0.02)
        self.dots = dots

        # Web of connections
        web = VGroup()
        for p1, p2 in it.combinations(dots.get_points(), 2):
            if random.random() < np.exp(-0.8 * get_norm(p1 - p2)):
                line = Line(p1, p2, stroke_color=WHITE, stroke_width=1, stroke_opacity=random.random())
                web.add(line)
        self.web = web

        # Zoom out over connections
        self.frame.add_updater(lambda m: m.set_height(2 + 0.5 * self.time))
        self.frame.add_updater(lambda m, dt: m.increment_theta(0.5 * dt * DEG))
        self.frame.add_updater(lambda m, dt: m.increment_phi(dt * DEG))


class CentralWebConnections(WebOfConnections):
    n_iterations = 13
    n_neighbors = 10
    n_examples = 5

    def construct(self):
        # Test
        point_list = list(self.dots.get_points())
        dots = GlowDot(ORIGIN, radius=0.15).replicate(self.n_examples)

        for dot in dots:
            path = TracedPath(dot.get_center, stroke_width=1, stroke_color=WHITE)
            path.set_stroke(opacity=0.5)
            tail = TracingTail(dot, time_traced=2.0)
            self.add(dot, path, tail)

        dots.set_opacity(0)
        dots[0].set_opacity(1)
        self.wait()

        for n in range(self.n_iterations):
            for dot in dots:
                dot_center = dot.get_center()
                indices = np.argsort([get_norm(p - dot_center) for p in point_list])
                choice = np.random.randint(0, self.n_neighbors - 1)
                new_center = point_list[indices[choice]]
                point_list.pop(indices[choice])

                dot.target = dot.generate_target()
                dot.target.move_to(new_center)
                dot.target.set_opacity(1)
                if n > 0:
                    self.add(dot.copy().set_opacity(0.5))

            self.play(LaggedStartMap(MoveToTarget, dots), lag_ratio=0.5, run_time=3)


class ShowSimpleWeb(WebOfConnections):
    def construct(self):
        # Test
        self.web.sort(lambda p: get_norm(p))

        self.add(self.dots, self.web)
        self.wait(30)

        # Dense web
        dense_web = VGroup()
        for p1, p2 in it.combinations(self.dots.get_points(), 2):
            if random.random() < np.exp(-0.2 * get_norm(p1 - p2)):
                line = Line(
                    p1,
                    p2,
                    stroke_color=WHITE,
                    stroke_width=0.5,
                    stroke_opacity=random.random()**10
                )
                dense_web.add(line)
        self.play(ShowCreation(dense_web, lag_ratio=1 / len(dense_web), run_time=5))
        self.wait(5)


class GrowingWhiteDot(InteractiveScene):
    def construct(self):
        # Test
        dot = GlowDot(radius=1)
        dot.set_color(WHITE)
        self.add(dot)
        self.play(
            dot.animate.set_radius(FRAME_WIDTH).set_glow_factor(0.5),
            run_time=35
        )


class EndScreen(PatreonEndScreen):
    pass`,
    annotations: {
      4: "ShowPastVideos extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      72: "ConfettiSpiril extends Animation. Custom Animation subclass. Override interpolate(alpha) to define how the animation progresses from 0 to 1.",
      101: "Confetti extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      102: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      127: "Class HappyPiDay inherits from TeacherStudentsScene.",
      128: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      148: "Leftrightarrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      149: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      155: "GroversAlgorithmLabel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      156: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      170: "UnsolvedReference extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      171: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      182: "Recap extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      183: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      190: "RewindArrows extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      191: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      199: "Class CommentOnElastic inherits from TeacherStudentsScene.",
      200: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      222: "WritePiDigits extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      223: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      229: "ReactToQuantumComparisson extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      230: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      248: "LoadSolutionIntoHead extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      249: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      266: "ShowMassRatioToCountChart extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      267: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      307: "StateThePuzzle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      308: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      319: "EnergyAndMomentumLaws extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      320: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      342: "Class ProblemSolvingPrinciplesWithPis inherits from TeacherStudentsScene.",
      343: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      424: "StaysConstant extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      428: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      445: "Class NoteChange inherits from StaysConstant.",
      450: "SimpleArrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      451: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      457: "Class KeyStep inherits from TeacherStudentsScene.",
      458: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      469: "StateSpaceLabel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      470: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      485: "HoldUpEllipseVsCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      486: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      537: "Class AskWhy inherits from TeacherStudentsScene.",
      538: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      554: "PiTime1e5 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      555: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      562: "Class HighlightTheSlope inherits from TeacherStudentsScene.",
      563: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      594: "MostOfTheReasoning extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      595: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      604: "Class StareAtDiagram inherits from TeacherStudentsScene.",
      605: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      635: "AskHowThisIsHelpful extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      636: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      646: "Class ISeeWhereThisIsGoing inherits from TeacherStudentsScene.",
      647: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      715: "Class ReferenceSmallAngleApproximations inherits from TeacherStudentsScene.",
      716: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      761: "ExplainSmallAngleApprox extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      762: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      909: "Class AngryStudents inherits from TeacherStudentsScene.",
      910: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      919: "DigitsOfPi extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      920: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      985: "WriteExactSolution extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      986: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1016: "ExactSolutionForMatt extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1017: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1094: "Class WhoCares inherits from TeacherStudentsScene.",
      1095: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1122: "SimplifyingMessiness extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1125: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1161: "HiddenConnections extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1162: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1223: "WebOfConnections extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1259: "Class CentralWebConnections inherits from WebOfConnections.",
      1264: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1296: "Class ShowSimpleWeb inherits from WebOfConnections.",
      1297: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1320: "GrowingWhiteDot extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1321: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1332: "Class EndScreen inherits from PatreonEndScreen.",
    }
  };

})();