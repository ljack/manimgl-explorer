(function() {
  const files = window.MANIM_DATA.files;

  files["_2025/spheres/random_puzzles.py"] = {
    description: "Random geometric puzzles on spheres: probability of random points forming specific configurations, great circle distributions, and spherical geometry.",
    code: `from manim_imports_ext import *


class IntervalWithSample(Group):
    def __init__(
        self,
        width=4,
        numbers_font_size=36,
        x_range=(-1, 1, 0.25),
        label_step=0.5,
        marker_length=0.25,
        marker_width=(0, 15),
        marker_color=BLUE_B,
        dec_label_font_size=30,
    ):
        self.number_line = NumberLine(x_range, width=width)
        self.number_line.add_numbers(
            (-1, 0, 1),
            font_size=numbers_font_size,
            num_decimal_places=0
        )
        self.x_tracker = ValueTracker()
        get_x = self.x_tracker.get_value

        self.marker = Line(ORIGIN, marker_length * UP)
        self.marker.set_stroke(marker_color, width=marker_width, flat=True)
        self.marker.add_updater(lambda m: m.put_start_on(self.number_line.n2p(get_x())))

        self.dec_label = DecimalNumber(font_size=dec_label_font_size)
        self.dec_label.add_updater(lambda m: m.set_value(get_x()))
        self.dec_label.add_updater(
            lambda m: m.next_to(self.marker.get_end(), normalize(self.marker.get_vector()), SMALL_BUFF)
        )

        super().__init__(
            self.x_tracker,
            self.number_line,
            self.marker,
            self.dec_label
        )


class DotHistory(GlowDots):
    def __init__(self, pos_func, color=GREEN, fade_rate=0.95):
        super().__init__(points=[pos_func()], color=color)
        self.pos_func = pos_func
        self.fade_rate = fade_rate
        self.add_updater(lambda m: m.update_dots())

    def update_dots(self):
        curr_points = self.get_points()
        point = self.pos_func()
        opacities = self.get_opacities().copy()
        if not np.isclose(point, curr_points[-1]).all():
            self.append_points([point])
            opacities = np.append(opacities, [1])
        opacities *= self.fade_rate
        self.set_opacity(opacities)

    def reset(self):
        self.set_points([self.pos_func()])


class RandomSumsOfSquares(InteractiveScene):
    tex_to_color = {"x": BLUE, "y": YELLOW, "z": RED, "w": PINK}

    def get_intervals(self, n_intervals, buff=MED_LARGE_BUFF, **kwargs):
        result = Group(IntervalWithSample(**kwargs) for n in range(n_intervals))
        result.arrange(DOWN, buff=buff)
        return result

    def get_labeled_intervals(self, label_texs, font_size=48, buff=MED_LARGE_BUFF):
        intervals = self.get_intervals(len(label_texs), buff=buff)
        labels = VGroup(
            Tex(tex, font_size=font_size, t2c=self.tex_to_color)
            for tex in label_texs
        )
        for label, interval in zip(labels, intervals):
            label.next_to(interval.number_line.get_start(), LEFT, MED_LARGE_BUFF)
            interval.label = label
            interval.add(label)
        return intervals

    def get_probability_question(self, sum_tex="x^2 + y^2"):
        question = Tex(Rf"P\\left({sum_tex} \\le 1 \\right)", t2c=self.tex_to_color)
        question.to_corner(UR, buff=LARGE_BUFF)
        return question

    def get_evaluation_object(self, intervals):
        equation = Tex(" + ".join(["(+0.00)^2"] * len(intervals)) + "= 1.00")
        x_terms = equation.make_number_changeable("+0.00", replace_all=True, include_sign=True)
        for x_term, interval in zip(x_terms, intervals):
            if hasattr(interval, "label"):
                x_term.match_color(interval.label[0])
            x_term.f_always.set_value(interval.x_tracker.get_value)
        rhs = equation.make_number_changeable("1.00")
        rhs.add_updater(lambda m: m.set_value(sum([
            interval.x_tracker.get_value()**2
            for interval in intervals
        ])))

        marks = VGroup(
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        )
        marks.set_width(0.75 * rhs.get_width())
        marks[0].add_updater(lambda m: m.set_opacity(float(rhs.get_value() <= 1)))
        marks[1].add_updater(lambda m: m.set_opacity(float(rhs.get_value() > 1)))
        marks.next_to(rhs, DOWN)
        equation.add(marks)

        return equation

    def set_interval_values_randomly(self, intervals):
        for interval in intervals:
            value = random.uniform(*interval.number_line.x_range[:2])
            interval.x_tracker.set_value(value)

    def randomize_intervals(self, intervals, run_time=3, frequency=0.1):
        intervals.value_at_last_change = -1

        def update_intervals(intervals, alpha):
            curr_value = int(alpha * run_time / frequency)
            if curr_value != intervals.value_at_last_change:
                self.set_interval_values_randomly(intervals)
                intervals.value_at_last_change = curr_value

        return UpdateFromAlphaFunc(intervals, update_intervals, rate_func=linear, run_time=run_time)


class SumOfTwoSquares(RandomSumsOfSquares):
    def construct(self):
        # Set up
        intervals = self.get_labeled_intervals("xy", buff=LARGE_BUFF)
        intervals.to_corner(UL)
        question = self.get_probability_question()
        question.set_width(5)
        question.set_x(3.5)
        evaluation = self.get_evaluation_object(intervals)
        evaluation.next_to(question, DOWN, LARGE_BUFF)

        # Show initial randomization
        frame = self.frame
        frame.set_height(6).move_to(intervals)

        self.add(intervals[0])
        self.play(self.randomize_intervals(intervals[:1], frequency=0.25, run_time=10))
        self.add(intervals)
        self.play(self.randomize_intervals(intervals, frequency=0.25, run_time=10))
        self.play(
            frame.animate.to_default_state(),
            FadeIn(question),
            run_time=2,
        )
        self.wait()
        evaluation.update()
        evaluation.suspend_updating()
        self.play(Write(evaluation))
        self.wait()
        evaluation.resume_updating()
        intervals[0].x_tracker.set_value(0.25)
        intervals[1].x_tracker.set_value(-0.15)
        self.wait()

        self.play(self.randomize_intervals(intervals, frequency=0.25, run_time=10))

        # Change to axes
        axes = Axes((-1, 1, 0.25), (-1, 1, 0.25), width=6, height=6)
        axes.set_height(6)
        axes.move_to(3 * LEFT)
        for axis in axes:
            axis.add_numbers([-1, 0, 1])
            axis.numbers[1].set_opacity(0)

        get_x = intervals[0].x_tracker.get_value
        get_y = intervals[1].x_tracker.get_value
        x_dot = GlowDot(color=BLUE)
        x_dot.add_updater(lambda m: m.move_to(intervals[0].number_line.n2p(get_x())))
        y_dot = GlowDot(color=YELLOW)
        y_dot.add_updater(lambda m: m.move_to(intervals[1].number_line.n2p(get_y())))

        xy_dot = GlowDot(color=WHITE)
        xy_dot.add_updater(lambda m: m.move_to(axes.c2p(get_x(), get_y())))
        h_line, v_line = Line().set_stroke(WHITE, 1).replicate(2)
        h_line.f_always.put_start_and_end_on(y_dot.get_center, xy_dot.get_center)
        v_line.f_always.put_start_and_end_on(x_dot.get_center, xy_dot.get_center)

        xy_coord_label = Tex(R"(x, y)", t2c={"x": BLUE, "y": YELLOW}, font_size=24)
        xy_coord_label.add_updater(lambda m: m.next_to(xy_dot.get_center(), UR, buff=SMALL_BUFF))

        self.play(
            question.animate.set_height(0.7).to_corner(UR),
            FadeOut(evaluation.clear_updaters()),
            Transform(intervals[0].number_line, axes.x_axis),
            Transform(intervals[1].number_line, axes.y_axis),
            intervals[0].dec_label.animate.set_opacity(0),
            intervals[1].dec_label.animate.set_opacity(0),
            intervals[0].marker.animate.set_opacity(0),
            intervals[1].marker.animate.set_opacity(0),
            FadeIn(x_dot),
            FadeIn(y_dot),
            intervals[0].label.animate.next_to(axes.x_axis.get_end(), RIGHT, SMALL_BUFF),
            intervals[1].label.animate.next_to(axes.y_axis.get_end(), UP, SMALL_BUFF),
            run_time=2
        )
        VGroup(h_line, v_line).update()
        self.play(
            ShowCreation(h_line, suspend_mobject_updating=True),
            ShowCreation(v_line, suspend_mobject_updating=True),
            TransformFromCopy(x_dot, xy_dot, suspend_mobject_updating=True),
            TransformFromCopy(y_dot, xy_dot, suspend_mobject_updating=True),
            FadeIn(xy_coord_label[0::2]),
            TransformFromCopy(intervals[0].label, xy_coord_label[1]),
            TransformFromCopy(intervals[1].label, xy_coord_label[3]),
        )
        self.add(xy_coord_label, xy_dot, h_line, v_line)
        self.wait()

        # Show the random points within a square
        dot_history = DotHistory(xy_dot.get_center)
        dot_history.set_z_index(-1)

        square = Square(side_length=axes.x_axis.get_length())
        square.move_to(axes.get_origin())
        square.set_fill(GREEN, 0.1)
        square.set_stroke(GREEN, 1)

        self.add(dot_history)
        self.play(
            self.randomize_intervals(intervals, frequency=0.2, run_time=10),
            FadeIn(square),
        )
        self.wait(3)
        self.remove(dot_history)

        # Show circle
        circle = Circle()
        circle.replace(square)
        circle.set_stroke(TEAL, 3)
        circle.set_fill(TEAL, 0.1)
        underline = Underline(question[R"x^2 + y^2 \\le 1"], buff=0, stretch_factor=1)
        underline.match_style(circle)

        self.play(ShowCreation(underline))
        self.wait()
        self.play(ReplacementTransform(underline, circle, run_time=1.5))
        self.wait()

        # Pythagorean Theorem
        self.play(
            intervals[0].x_tracker.animate.set_value(0.6),
            intervals[1].x_tracker.animate.set_value(0.8),
        )

        r_line = Line(axes.get_origin(), xy_dot.get_center())
        r_line.set_stroke(RED, 4)
        r_label = Tex(R"r")
        r_label.set_color(RED)
        r_label.next_to(r_line.get_center(), UL, SMALL_BUFF)

        h_line.save_state()
        v_line.save_state()
        h_line.suspend_updating()
        v_line.suspend_updating()
        h_line.match_y(axes.get_origin())
        h_line.set_stroke(BLUE, 4)
        v_line.set_stroke(YELLOW, 4)

        self.tex_to_color["{r}"] = RED
        pythag = Tex(R"x^2 + y^2 = {r}^2", t2c=self.tex_to_color, font_size=72)
        pythag.to_edge(RIGHT, buff=LARGE_BUFF)

        self.play(
            ShowCreation(r_line),
            FadeIn(r_label, 0.25 * r_line.get_vector())
        )
        self.play(Write(pythag))
        self.wait()
        self.play(
            FadeOut(r_line),
            FadeOut(r_label),
            FadeOut(pythag),
            Restore(h_line),
            Restore(v_line),
        )

        h_line.resume_updating()
        v_line.resume_updating()

        # More randomness
        dot_history.reset()
        self.add(dot_history)
        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=30))
        self.wait(2)
        self.remove(dot_history)

        # Go three d
        q3d_tex = "x^2 + y^2 + z^2"
        question_3d = self.get_probability_question(q3d_tex)
        question_3d.match_height(question)
        question_3d.move_to(question, RIGHT)
        question_3d.fix_in_frame()
        frame = self.frame

        length = axes.x_axis.get_length()
        axes3d = ThreeDAxes((-1, 1), (-1, 1), (-1, 1), width=length, height=length, depth=length)
        axes3d.move_to(axes.get_origin())
        z_interval = self.get_intervals(1, width=length)[0]
        z_interval.number_line.rotate(90 * DEG, OUT).rotate(90 * DEG, RIGHT)
        z_interval.number_line.remove(z_interval.number_line.numbers)
        z_interval.number_line.ticks.stretch(0.5, 0)
        z_interval.shift(axes.get_origin() - z_interval.number_line.n2p(0))
        z_interval.marker.set_opacity(0)
        z_interval.dec_label.set_opacity(0)
        z_axis_label = Tex(R"z")
        z_axis_label.set_color(RED)
        z_axis_label.rotate(90 * DEG, RIGHT)
        z_axis_label.next_to(axes3d, OUT, MED_SMALL_BUFF)

        get_z = z_interval.x_tracker.get_value
        z_interval.x_tracker.set_value(random.random())
        xyz_dot = GlowDot(color=WHITE)
        xyz_dot.add_updater(lambda m: m.move_to(axes3d.c2p(get_x(), get_y(), get_z())))
        z_line = Line().set_stroke(WHITE, 1)
        z_line.f_always.put_start_and_end_on(xy_dot.get_center, xyz_dot.get_center)
        xyz_coord_label = Tex(R"(x, y, z)", t2c=self.tex_to_color, font_size=30)
        xyz_coord_label.rotate(90 * DEG, RIGHT)
        xyz_coord_label.add_updater(lambda m: m.next_to(xyz_dot.get_center(), RIGHT, SMALL_BUFF))

        self.play(TransformMatchingTex(question, question_3d, run_time=1))
        self.play(FlashAround(question_3d[q3d_tex], time_width=1.5, run_time=2))
        self.play(
            frame.animate.reorient(-26, 79, 0, (-0.13, 0.08, 0.47), 10.09),
            Write(z_interval.number_line),
            Write(z_axis_label),
            run_time=2
        )
        self.play(
            TransformFromCopy(xy_dot, xyz_dot, suspend_mobject_updating=True),
            xy_dot.animate.set_opacity(0),
            TransformMatchingTex(xy_coord_label, xyz_coord_label),
            ShowCreation(z_line, suspend_mobject_updating=True),
            run_time=1
        )
        self.add(xyz_dot, z_line)
        self.wait()

        # Show cube and sphere
        cube = VCube()
        cube.match_width(square)
        cube.match_style(square)
        cube.set_fill(opacity=0.05)
        cube.move_to(square)
        cube.deactivate_depth_test()
        cube.save_state()
        cube.stretch(0, 2)

        sphere = Sphere()
        sphere.replace(cube)
        sphere_mesh = SurfaceMesh(sphere, resolution=(101, 51))
        sphere_mesh.set_stroke(WHITE, 2, 0.1)
        sphere_mesh.deactivate_depth_test()

        self.remove(square)
        self.play(
            Restore(cube),
            Write(sphere_mesh, lag_ratio=1e-3, time_span=(1, 3)),
            FadeOut(circle, time_span=(1, 3)),
            frame.animate.reorient(-41, 69, 0, (-0.34, -0.78, -0.59), 12.47).set_anim_args(run_time=4),
        )
        self.wait()

        # Even more randomness
        dot_history = DotHistory(xyz_dot.get_center)
        self.add(dot_history, xyz_dot, z_line)
        intervals.add(z_interval)
        self.play(
            self.randomize_intervals(intervals, frequency=0.2, run_time=20),
            frame.animate.reorient(14, 62, 0, (1.36, 0.96, 0.63), 12.10),
            run_time=20
        )
        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=20))


class AskAboutThreeSumsOfSquares(RandomSumsOfSquares):
    label_texs = "xyz"
    square_sum_tex = "x^2 + y^2 + z^2"

    def construct(self):
        intervals = self.get_labeled_intervals(self.label_texs, buff=LARGE_BUFF)
        intervals.to_edge(LEFT)
        question = self.get_probability_question(self.square_sum_tex)
        question.to_corner(UR)
        evaluation = self.get_evaluation_object(intervals)
        evaluation.set_width(6.5)
        evaluation.next_to(question, DOWN, LARGE_BUFF)
        evaluation.to_edge(RIGHT)

        self.add(intervals)
        self.add(question)
        self.add(evaluation)

        self.play(self.randomize_intervals(intervals, frequency=0.2, run_time=30))


class AskAboutFourSumsOfSquares(AskAboutThreeSumsOfSquares):
    label_texs = "xyzw"
    square_sum_tex = "x^2 + y^2 + z^2 + w^2"


class AskAboutLargeSumOfSquares(RandomSumsOfSquares):
    def construct(self):
        # Test
        intervals = self.get_intervals(
            8,
            width=6,
            numbers_font_size=24,
            dec_label_font_size=24,
            marker_length=0.15
        )
        intervals.set_height(7)
        intervals.to_edge(LEFT, buff=LARGE_BUFF)
        to_remove = intervals[-3:-1]
        dots = Tex(R"\\vdots", font_size=90)
        dots.move_to(to_remove)
        intervals.remove(*to_remove)

        labels = VGroup(
            Tex(Rf"x_{{{int(n)}}}", font_size=36)
            for n in [*range(1, 6), 100]
        )
        labels.set_submobject_colors_by_gradient(BLUE, YELLOW)
        for label, interval in zip(labels, intervals):
            label.next_to(interval.number_line.get_start(), LEFT, MED_SMALL_BUFF)

        question = Tex(
            R"P\\left(x_1^2 + x_2^2 + \\cdots + x_{100}^2 \\le 1 \\right)",
            t2c={"x_1": labels[0].get_color(), "x_2": labels[1].get_color(), "x_{100}": labels[-1].get_color()}
        )
        question.to_corner(UR)

        self.add(intervals)
        self.add(labels)
        self.add(dots)
        self.add(question)

        self.play(self.randomize_intervals(intervals, run_time=20, frequency=0.2))


class DotProductOfUnitVectors(InteractiveScene):
    random_seed = 2

    def construct(self):
        # Set up
        plane = NumberPlane((-4, 4), (-2, 2))
        plane.set_height(12)
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        unit_size = plane.x_axis.get_unit_size()

        circle = Circle(radius=unit_size)
        circle.set_stroke(GREY_C, 2)

        self.add(plane, circle)

        # Vectors
        v1, v2 = vects = Vector(unit_size * RIGHT, thickness=6).replicate(2)
        colors = [PINK, YELLOW]
        for vect, color, char in zip(vects, colors, "vw"):
            vect.set_fill(color)
            vect.label = Tex(R"\\vec{\\textbf{" + char + R"}}")
            vect.label.match_color(vect)
            vect.label.set_backstroke(BLACK, 5)
            vect.label.scale(2)
            vect.angle_tracker = ValueTracker(0)

        v1.angle_tracker.set_value(0.0)
        v2.angle_tracker.set_value(0.6)

        def update_vector(vector):
            vector.put_start_and_end_on(ORIGIN, circle.pfp(vector.angle_tracker.get_value() % 1))
            vector.label.move_to(vector.get_end() + 0.2 * vector.get_vector())

        v1.add_updater(update_vector)
        v2.add_updater(update_vector)

        self.add(v1, v2, v1.label, v2.label)

        self.play(v1.angle_tracker.animate.set_value(0.4), run_time=3)
        self.wait()

        # Show projection
        proj_group = always_redraw(lambda: self.get_projection_group(v1, v2))
        proj_group.suspend_updating()
        dashed_line, proj_line, proj_brace, label = proj_group

        self.play(LaggedStart(
            ShowCreation(dashed_line),
            FadeIn(proj_line),
            GrowFromCenter(proj_brace),
            AnimationGroup(
                TransformFromCopy(v1.label, label[0]),
                TransformFromCopy(v2.label, label[1]),
                Write(label[2])
            ),
            lag_ratio=0.5,
        ))
        self.wait()

        # Wander
        proj_group.resume_updating()
        self.add(proj_group)

        self.play(
            v1.angle_tracker.animate.set_value(0.2),
            run_time=3
        )
        self.wait()


        self.play(
            v2.angle_tracker.animate.set_value(0.25),
            run_time=3
        )
        self.wait()
        for value in [-0.23, 0.2]:
            self.play(
                v1.angle_tracker.animate.set_value(value),
                run_time=5
            )
            self.wait()

        # Randomness
        self.remove(proj_group)
        for n in range(50):
            v1.angle_tracker.set_value(random.random())
            v2.angle_tracker.set_value(random.random())
            self.wait(0.2)

    def get_projection_group(self, v1, v2):
        # Test
        dp = np.dot(v1.get_vector(), v2.get_vector()) / (v1.get_length() * v2.get_length())
        proj_line = Line(
            v2.get_start(),
            interpolate(v2.get_start(), v2.get_end(), dp)
        )
        proj_line.set_stroke(WHITE, 8)
        orientation = cross(v1.get_vector(), v2.get_vector())[2]
        proj_brace = LineBrace(
            proj_line,
            buff=SMALL_BUFF,
            # direction=UP if orientation > 0 else DOWN
            direction=UP if dp > 0 else DOWN
        )
        label = VGroup(v1.label, v2.label).copy()
        label.scale(0.5)
        label.arrange(RIGHT, buff=0.25)
        label.add(Tex(R"\\cdot").move_to(label))
        angle = (proj_line.get_angle() + (PI if orientation > 0 else 0))
        angle = (angle + PI / 2) % PI - PI / 2
        label.rotate(angle)
        label.move_to(proj_brace.get_center() + 3 * (proj_brace.get_center() - proj_line.get_center()))

        dashed_line = DashedLine(v1.get_end(), proj_line.get_end())
        dashed_line.set_stroke(WHITE, 1)

        result = VGroup(dashed_line, proj_line, proj_brace, label)

        return result


class Random3DVectors(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 2
        axes = ThreeDAxes(unit_size=radius)
        sphere = Sphere(radius=radius)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.1)

        v1, v2 = vects = Vector(RIGHT, thickness=4).replicate(2)
        colors = [PINK, YELLOW]
        for vect, color in zip(vects, colors):
            vect.set_fill(color)
            vect.always.set_perpendicular_to_camera(frame)
            self.set_vector_randomly(vect, radius)

        frame.reorient(-38, 74, 0, (0.04, -0.01, -0.06), 4.82)
        self.add(axes, mesh)
        self.add(vects)

        # Add dot product label
        def get_dp():
            vect1 = v1.get_vector()
            vect2 = v2.get_vector()
            return np.dot(vect1, vect2) / (get_norm(vect1) * get_norm(vect2))

        dp_label = Tex(R"\\vec{\\textbf{v}} \\cdot \\vec{\\textbf{w}} = 0.00")
        dp_label[R"\\vec{\\textbf{v}}"].set_color(PINK)
        dp_label[R"\\vec{\\textbf{w}}"].set_color(YELLOW)
        num = dp_label.make_number_changeable("0.00")
        num.f_always.set_value(get_dp)

        dp_label.fix_in_frame()
        dp_label.to_corner(UL)
        self.add(dp_label)

        # Randomize
        frame.add_ambient_rotation(2 * DEG)
        for n in range(100):
            for vect in vects:
                self.set_vector_randomly(vect, radius)
            self.wait(0.2)

        # Only one random vector
        v2.put_start_and_end_on(ORIGIN, radius * OUT)
        v2.set_fill(opacity=0.5)

        dot = GlowDot()
        dot.set_color(v1.get_color())
        dot_ghosts = Group()

        def project(points):
            points[:, :2] = 0
            return points

        z_line = Line(ORIGIN, OUT)
        z_line.set_stroke(WHITE, 8)
        self.add(dot, dot_ghosts, z_line)

        for n in range(100):
            self.set_vector_randomly(v1, radius)
            z_line.set_points_as_corners([ORIGIN, v1.get_end()[2] * OUT])
            dot.move_to(v1.get_end())
            dot_ghosts.add(dot.copy())
            for ghost in dot_ghosts:
                ghost.set_opacity(ghost.get_opacity() * 0.9)
            self.wait(0.2)

    def set_vector_randomly(self, vect, radius):
        point = radius * normalize(np.random.normal(0, 1, 3))
        vect.put_start_and_end_on(ORIGIN, point)
        return vect


class Distributions(InteractiveScene):
    def construct(self):
        # Test
        axes = Axes(
            (-1, 1, 0.2),
            (0, 1),
            width=6,
            height=4
        )
        axes.add_coordinate_labels(num_decimal_places=1, font_size=16)
        axes.y_axis.set_opacity(0)

        # graph = axes.get_graph(lambda x: np.exp(-5 * x**2))
        # graph = axes.get_graph(lambda x: np.sqrt(1 - x**2))
        # graph = axes.get_graph(lambda x: 0.3 * x**2 + 0.5)
        graph = axes.get_graph(lambda x: 0.5)

        rects = axes.get_riemann_rectangles(graph, dx=0.1)
        rects.set_fill(opacity=0.5)
        rects.set_stroke(WHITE, 1, 0.5)

        self.add(axes, rects)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "IntervalWithSample extends Group. Group holds heterogeneous mobjects (mix of VMobjects, Surfaces, Images) and transforms them together.",
      22: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      27: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      29: "DecimalNumber displays a formatted decimal that can be animated. Tracks a value and auto-updates display.",
      30: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      43: "Class DotHistory inherits from GlowDots.",
      48: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      64: "RandomSumsOfSquares extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      75: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      85: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      90: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      97: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      107: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      108: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      128: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      131: "Class SumOfTwoSquares inherits from RandomSumsOfSquares.",
      132: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      150: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      151: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      152: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      155: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      158: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      159: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      163: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      168: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      177: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      178: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      179: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      180: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      182: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      183: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      188: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      189: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      191: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      192: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      193: "FadeOut transitions a mobject from opaque to transparent.",
      194: "Transform smoothly morphs one mobject into another by interpolating their points.",
      195: "Transform smoothly morphs one mobject into another by interpolating their points.",
      196: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      197: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      198: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      199: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      200: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      201: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      202: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      203: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      207: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      208: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      385: "Class AskAboutThreeSumsOfSquares inherits from RandomSumsOfSquares.",
      389: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      406: "Class AskAboutFourSumsOfSquares inherits from AskAboutThreeSumsOfSquares.",
      411: "Class AskAboutLargeSumOfSquares inherits from RandomSumsOfSquares.",
      412: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      450: "DotProductOfUnitVectors extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      453: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      572: "Random3DVectors extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      573: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      647: "Distributions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      648: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/spheres/supplements.py"] = {
    description: "Supplementary sphere geometry scenes: additional surface area derivations, solid angle concepts, and spherical cap calculations.",
    code: `from manim_imports_ext import *


class CornerDistance(InteractiveScene):
    def construct(self):
        words = Text("Distance to corner")
        pythag = Tex(R"\\sqrt{1^2 + 1^2 + \\cdots + 1^2 + 1^2}")
        sqrt3 = Tex(R"= \\sqrt{N}")
        r_eq = Tex(R"r = \\sqrt{N} - 1")

        words.move_to(2 * UP)
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(words, DOWN)
        pythag.next_to(equals, DOWN)
        sqrt3.next_to(pythag, RIGHT)
        r_eq.next_to(pythag, DOWN, buff=LARGE_BUFF)

        self.add(words, equals, pythag, sqrt3, r_eq)


class REqExample(InteractiveScene):
    def construct(self):
        # Test
        tex = Tex(R"r = \\sqrt{10} - 1 \\approx 2.162")
        self.add(tex)


class HyperbolaSquare(InteractiveScene):
    def construct(self):
        # Label distances
        square = VGroup(
            Line(UL, UR),
            Line(UR, DR),
            Line(DR, DL),
            Line(DL, UL),
        )
        square.set_height(4)
        square_shadows = VGroup(
            square.copy().scale(0.95**n).shift(0.05 * DR * n).set_stroke(opacity=0.5 / n)
            for n in range(1, 20)
        )

        center_dot = Dot()
        center_dot.move_to(square)
        side_line = Line(square.get_center(), square.get_right())
        diag_line = Line(square.get_center(), square.get_corner(UR))
        side_line.set_stroke(BLUE, 3)
        diag_line.set_stroke(YELLOW, 3)

        side_label = Tex(R"1")
        side_label.next_to(side_line, DOWN, SMALL_BUFF)
        diag_label = Tex(R"\\sqrt{N}")
        diag_label.next_to(diag_line.get_center(), UL, SMALL_BUFF)
        diag_label.set_backstroke(BLACK, 5)

        self.add(square)
        self.add(square_shadows)

        self.play(
            GrowFromCenter(center_dot),
            ShowCreation(side_line),
            FadeIn(side_label, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            ShowCreation(diag_line),
            FadeIn(diag_label, 0.5 * UR),
        )
        self.wait()

        # Warp it
        warp_square = self.get_warp_square(square)
        warp_square_shadows = VGroup(self.get_warp_square(ss) for ss in square_shadows)

        side_line.target = side_line.generate_target()
        side_line.target.put_start_and_end_on(center_dot.get_center(), warp_square[1].pfp(0.5))
        diag_line.target = diag_line.generate_target()
        diag_line.target.put_start_and_end_on(center_dot.get_center(), warp_square.get_corner(UR))

        self.wait()
        self.play(
            Transform(square, warp_square),
            Transform(square_shadows, warp_square_shadows),
            MoveToTarget(side_line),
            MoveToTarget(diag_line),
            side_label.animate.next_to(side_line.target, DOWN, SMALL_BUFF),
            diag_label.animate.shift(0.4 * DOWN + 0.2 * LEFT),
            run_time=3
        )
        self.wait()

        # Corner spheres
        frame = self.frame
        lil_radius = 0.6
        big_radius = diag_line.get_length() - lil_radius
        circles = Circle(radius=lil_radius).replicate(4)
        circles.set_stroke(BLUE, 3)
        circles.set_fill(BLUE, 0.25)
        for circle, side in zip(circles, warp_square):
            circle.move_to(side.get_start())

        big_circle = Circle(radius=big_radius)
        big_circle.set_stroke(GREEN, 3).set_fill(GREEN, 0.25)

        self.play(
            LaggedStartMap(GrowFromCenter, circles),
            frame.animate.set_height(9),
        )
        self.wait()
        self.play(GrowFromCenter(big_circle))
        self.wait()

        # Show many more corners
        group = VGroup(warp_square, circles)
        n_new_groups = 12
        angles = np.linspace(0, 90 * DEG, n_new_groups + 2)[1:-1]
        alt_groups = VGroup(
            group.copy().rotate(theta, about_point=ORIGIN)
            for theta in angles
        )
        alt_groups.fade(0.75)

        corner_label = Tex(R"2^N \\text{ corners}", font_size=60)
        corner_label.to_corner(UR)
        corner_label.fix_in_frame()

        self.play(
            FadeIn(corner_label),
            LaggedStart(
                (TransformFromCopy(group.copy().set_fill(opacity=0), alt_group, path_arc=angle)
                for angle, alt_group in zip(angles, alt_groups)),
                lag_ratio=0.01,
                run_time=3
            ),
            frame.animate.set_height(11),
            run_time=3
        )
        self.wait()

    def get_warp_square(self, square, scale_factor=1.7):
        hyper = FunctionGraph(lambda x: math.sqrt(1 + x**2), x_range=(-2, 2, 0.02))
        warp_square = VGroup(
            hyper.copy().put_start_and_end_on(*side.get_start_and_end())
            for side in square
        )
        warp_square.scale(scale_factor)
        warp_square.match_style(square)
        return warp_square


class AreaCircleOverAreaSquareThenVolume(InteractiveScene):
    def construct(self):
        # Left hand side
        circle_color = TEAL
        square_color = GREEN
        circle_color = BLUE
        square_color = RED

        font_size = 72
        c_tex = R"{CC \\over CC}"
        s_tex = R"{SS \\over SS}"
        frac = Tex(
            fR"{{\\text{{Area}}\\left( {c_tex} \\right) \\over \\text{{Area}}\\left( {s_tex} \\right)}}",
            font_size=font_size
        )
        area_words = frac[R"\\text{Area}"]
        for part in area_words:
            part.scale(0.75, about_edge=RIGHT).shift(SMALL_BUFF * RIGHT)
        frac.next_to(ORIGIN, LEFT)
        frac.to_edge(UP, LARGE_BUFF)
        circle = Circle()
        circle.set_stroke(circle_color, 3)
        circle.set_fill(circle_color, 0.25)
        square = Square()
        square.set_stroke(square_color, 3)
        square.set_fill(square_color, 0.25)

        circle.replace(frac[c_tex])
        circle.shift(0.05 * UP)
        square.replace(frac[s_tex])
        frac[c_tex].scale(0).set_opacity(0)
        frac[s_tex].scale(0).set_opacity(0)

        self.wait(0.1)
        self.play(
            Write(frac, lag_ratio=1e-1),
            LaggedStart(
                Write(circle),
                Write(square),
                lag_ratio=0.5
            )
        )
        self.wait()

        # Right hand side
        equals = Tex(R"=", font_size=90)
        equals.rotate(90 * DEG)
        equals.next_to(frac, DOWN)
        approx = Tex(R"\\approx", font_size=font_size)
        value = DecimalNumber(PI / 4, num_decimal_places=3, font_size=font_size)
        rhs = Tex(
            R"{\\pi (1)^2 \\over 2 \\times 2}",
            font_size=font_size,
            t2c={R"\\pi (1)^2": circle_color, R"2 \\times 2": square_color}
        )
        rhs.next_to(equals, DOWN)
        approx.next_to(rhs, RIGHT)
        value.next_to(approx, RIGHT)

        self.play(Write(equals), Write(rhs))
        self.wait()
        self.play(Write(approx), FadeIn(value, RIGHT))
        self.wait()

        # Make it three d
        sphere = Sphere()
        sphere.set_color(circle_color, 1)
        sphere.rotate(90 * DEG, RIGHT)
        sphere.replace(circle)
        sphere_mesh = SurfaceMesh(sphere)
        sphere_mesh.set_stroke(WHITE, 1, 0.25)
        sphere_mesh.deactivate_depth_test()
        cube = VCube()
        cube.set_fill(square_color, 0.2)
        cube.set_stroke(square_color, 3)
        cube.deactivate_depth_test()
        cube.rotate(20 * DEG, RIGHT).rotate(0 * DEG, UP)
        cube.replace(square).scale(0.9)

        volume_words = Text("Volume").replicate(2)
        for v_word, a_word in zip(volume_words, area_words):
            v_word.move_to(a_word, RIGHT)

        new_frac = Tex(
            R"{(4/3) \\pi (1)^3 \\over 2 \\times 2 \\times 2}",
            t2c={R"(4/3) \\pi (1)^3": circle_color, R"2 \\times 2 \\times 2": square_color},
            font_size=font_size
        )
        new_frac.next_to(equals, DOWN)

        self.play(
            Write(sphere_mesh, lag_ratio=1e-2),
            FadeOut(VGroup(equals, rhs, approx, value)),
            FadeTransform(circle, sphere),
            FadeTransform(square, cube),
            *(
                FadeTransformPieces(a_word, v_word)
                for v_word, a_word in zip(volume_words, area_words)
            )
        )
        self.play(
            Write(equals),
            FadeIn(new_frac, DOWN),
        )
        self.wait()

        # New approx
        value = DecimalNumber((4 / 3) * PI / 8, font_size=font_size, num_decimal_places=3)
        approx = Tex(R"\\approx", font_size=font_size)
        approx.next_to(new_frac, RIGHT)
        value.next_to(approx, RIGHT)

        self.play(
            Write(approx),
            FadeIn(value, RIGHT),
        )


class BigVolumeRatio(InteractiveScene):
    def construct(self):
        # Test
        fraction = Tex(R"\\text{Vol}\\Big(\\text{100D Unit Ball}\\Big) \\over 2^{100}")
        fraction["2^{100}"].scale(1.5, about_edge=UP).shift(SMALL_BUFF * DOWN)

        numerator = fraction[R"\\text{Vol}\\Big(\\text{100D Unit Ball}\\Big)"]
        numerator_rect = SurroundingRectangle(numerator, buff=0)
        numerator_rect.set_stroke(BLUE, 3)
        randy = Randolph(height=2)
        randy.next_to(fraction, LEFT, MED_LARGE_BUFF)
        randy.shift(0.5 * DOWN)

        self.wait(0.1)
        self.play(FadeIn(fraction))
        self.wait(3)
        self.play(
            VFadeIn(randy),
            randy.change("confused"),
            ShowCreation(numerator_rect)
        )
        self.play(Blink(randy))
        self.wait()

        # Reveal answer
        new_numerator = Tex(R"\\pi^{50} / 50!", font_size=72)
        new_numerator.move_to(numerator, DOWN)

        self.play(
            FadeOut(numerator_rect),
            fraction[R"\\over"].animate.match_width(new_numerator, stretch=True),
            FadeTransform(numerator, new_numerator),
            randy.change("tease"),
        )
        self.play(Blink(randy))
        self.wait()


class Derivatives(InteractiveScene):
    def construct(self):
        kw = dict(t2c={"r": BLUE})
        power_rules = VGroup(
            Tex(R"\\frac{d}{dr} r = 1", **kw),
            Tex(R"\\frac{d}{dr} r^2 = 2r", **kw),
            Tex(R"\\frac{d}{dr} r^3 = 3r^2", **kw),
            Tex(R"\\frac{d}{dr} r^4 = 4r^3", **kw),
            Tex(R"\\vdots", **kw),
        )
        power_rules.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        power_rules[-1].match_x(power_rules[-2]["="])
        self.add(power_rules)


class Integrals(InteractiveScene):
    def construct(self):
        kw = dict(t2c={"r": BLUE, "R": TEAL})
        inv_power_rules = VGroup(
            Tex(R"\\int_0^R 1 \\, dr = R", **kw),
            Tex(R"\\int_0^R r \\, dr = \\frac{1}{2} R^2", **kw),
            Tex(R"\\int_0^R r^2 \\, dr = \\frac{1}{3} R^3", **kw),
            Tex(R"\\int_0^R r^3 \\, dr = \\frac{1}{4} R^4", **kw),
            Tex(R"\\vdots", **kw),
        )
        inv_power_rules.arrange(DOWN, aligned_edge=LEFT, buff=0.6)
        inv_power_rules[-1].match_x(inv_power_rules[-2]["="])
        self.add(inv_power_rules)


class CommentOnGeneralFormula(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        randy = Randolph()
        randy.next_to(morty, LEFT, buff=1.5)

        self.play(
            morty.says("Beautiful!", mode="hooray"),
            randy.change("hesitant", morty.eyes)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.debubble(mode="guilty"),
            randy.says("What about\\nodd n?", mode="confused", look_at=3 * LEFT + DOWN)
        )
        self.play(Blink(randy))
        self.wait()


class VolumeRatio(InteractiveScene):
    def construct(self):
        # 3D
        group = VGroup(
            Tex(R"{8  \\cdot \\big({4 \\over 3} \\pi \\big) \\over 2 \\times 2 \\times 2}"),
            Tex(R"="),
            Tex(R"{4 \\over 3} \\pi")
        )
        group.arrange(RIGHT, buff=0.25)
        self.add(group)

        # 100D
        self.clear()
        group = VGroup(
            Tex(R"{2^{100}  \\cdot \\big(\\pi^{50} / 50! \\big) \\over 2^{100}}"),
            Tex(R"="),
            Tex(R"{\\pi^{50} \\over 50!}"),
        )
        group.arrange(RIGHT, buff=0.5)
        self.add(group)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "CornerDistance extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      6: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      7: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      8: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      9: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      12: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      21: "REqExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      22: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      24: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      28: "HyperbolaSquare extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      29: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      50: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      52: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      59: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      61: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      62: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      64: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      65: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      66: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      67: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      69: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      80: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      81: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      151: "AreaCircleOverAreaSquareThenVolume extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      152: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      269: "BigVolumeRatio extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      270: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      307: "Derivatives extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      308: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      322: "Integrals extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      323: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      337: "CommentOnGeneralFormula extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      338: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      358: "VolumeRatio extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      359: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/spheres/volumes.py"] = {
    description: "Sphere volume derivation scenes: visualizing the integral approach to computing sphere volume using disk slicing and Cavalieri's principle.",
    code: `from manim_imports_ext import *


def get_boundary_volume_texs():
    return [
        R"0",
        R"2",
        R"2\\pi r",
        R"4\\pi r^2",
        R"2\\pi^2 r^3",
        R"{8 \\over 3}\\pi^2 r^4",
        R"\\pi^3 r^5",
        R"{16 \\pi^3 \\over 15} r^6",
        R"{\\pi^4 \\over 3} r^7",
        R"{32 \\pi^4 \\over 105} r^8",
    ]


def get_volume_texs():
    return [
        R"1",
        R"2 r",
        R"\\pi r^2",
        R"{4 \\over 3} \\pi r^3",
        R"{\\pi^2 \\over 2} r^4",
        R"{8 \\over 15} \\pi^2 r^5",
        R"{\\pi^3 \\over 6} r^6",
        R"{16 \\pi^3 \\over 105} r^7",
        R"{\\pi^4 \\over 24} r^8",
        R"{32 \\pi^4 \\over 945} r^9",
        R"{\\pi^5 \\over 120} r^10",
    ]


class CircumferenceToArea(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{r}": BLUE}
        circum_formula = Tex(R"\\text{Circumference} = 2 \\pi {r}", t2c=t2c)

        radius = 3
        circle = Circle(radius=radius)
        circle.set_stroke(YELLOW, 5)
        radius_line = Line(ORIGIN, radius * RIGHT)
        radius_line.set_stroke(WHITE, 3)
        r_label = Tex(R"r", font_size=72)
        r_label.set_color(BLUE)
        r_label.next_to(radius_line, UP, SMALL_BUFF)
        r_group = VGroup(radius_line, r_label)
        r_group.set_z_index(1)

        self.add(circle, r_group)
        self.play(
            ShowCreation(circle),
            Rotate(r_group, TAU, about_point=ORIGIN),
            run_time=3,
        )
        self.wait()

        # Inner circles
        circles = VGroup(
            circle.copy().set_width(a * circle.get_width())
            for a in np.linspace(1, 0, 100)
        )
        circles.set_stroke(YELLOW, 3, 0.25)
        self.play(
            ReplacementTransform(
                circle.replicate(len(circles)).set_stroke(width=0, opacity=0),
                circles,
                lag_ratio=0.1
            ),
            run_time=3
        )
        self.wait()


class SurfaceAreaToVolume(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.2)

        frame.reorient(44, 56, 0)
        self.play(
            ShowCreation(sphere),
            Write(mesh),
            run_time=2
        )
        self.wait()

        # Inner spheres
        inner_spheres = Group(
            sphere.copy().set_width(a * sphere.get_width())
            for a in np.linspace(0, 1, 50)
        )
        inner_spheres.set_color(BLUE, 0.2)
        inner_spheres.set_clip_plane(IN, 0)

        self.remove(sphere)
        self.add(inner_spheres, mesh)
        self.play(
            ShowCreation(inner_spheres, lag_ratio=0.5, run_time=3),
        )
        self.play(inner_spheres.animate.set_clip_plane(IN, radius), run_time=2)
        self.wait()


class VolumeGrid(InteractiveScene):
    tex_to_color = {"r": BLUE}

    def construct(self):
        # Write the grid
        frame = self.frame
        n_cols = 10
        grid = self.get_grid(n_cols)
        boundary_labels, volume_labels = self.get_volume_labels()
        labels = VGroup(
            VGroup(*pair)
            for pair in zip(boundary_labels, volume_labels)
        )

        for label_pair, col in zip(labels, grid):
            for label, square in zip(label_pair, col):
                label.move_to(square)
                square.label = label

        self.add(grid[1:4])

        # Row labels
        row_labels = VGroup(
            Tex(R"\\partial B^n"),
            Tex(R"B^n"),
        )
        for square, label in zip(grid[1], row_labels):
            label.next_to(square, LEFT)

        col_labels = VGroup(Integer(n, font_size=40) for n in range(n_cols))
        for col, label in zip(grid, col_labels):
            label.next_to(col, UP)

        cols_label = Text("Dimension")
        cols_label.next_to(col_labels[1:4], UP)
        cols_label.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(col_labels[1:4])
        self.add(cols_label)

        # Add for d=2 and d=3
        def highlight_cell(row, col, run_time=3, fill_color=TEAL_E, fill_opacity=0.5):
            kw = dict(rate_func=there_and_back_with_pause, run_time=run_time)
            cell = grid[col][row]
            return cell.animate.set_fill(fill_color, fill_opacity).set_anim_args(**kw)

        for d, n in it.product([2, 3], [0, 1]):
            self.play(
                highlight_cell(n, d),
                Write(labels[d][n]),
            )

        # Show derivatives
        self.show_derivative_and_integral(grid, 2)
        self.show_derivative_and_integral(grid, 3)

        # Add row labels
        rect = SurroundingRectangle(row_labels[1])
        rect.set_stroke(TEAL, 3)
        boundary_word = Text("“Boundary”")
        boundary_word.set_color(TEAL)
        boundary_word.next_to(row_labels[0], LEFT)

        self.play(ShowCreation(rect), Write(row_labels[1]))
        self.wait()
        self.play(
            rect.animate.surround(row_labels[0]),
            TransformMatchingTex(row_labels[1].copy(), row_labels[0]),
            run_time=1
        )
        self.wait()
        self.play(
            frame.animate.set_x(-3),
            FadeIn(boundary_word, lag_ratio=0.1),
            rect.animate.surround(row_labels[0][0])
        )
        self.wait()
        self.play(
            frame.animate.set_x(0),
            FadeOut(rect),
            FadeOut(boundary_word),
        )
        self.wait()

        # Add for d=1
        self.play(
            highlight_cell(1, 1),
            FadeIn(volume_labels[1])
        )
        self.play(
            highlight_cell(0, 1),
            FadeIn(boundary_labels[1])
        )

        # Ask about the rest
        q_marks = VGroup(
            VGroup(
                Tex(R"?", font_size=72).move_to(cell)
                for cell in col
            )
            for col in grid[4:]
        )
        q_marks.set_fill(YELLOW)
        self.play(
            Write(q_marks, lag_ratio=0.05),
            Write(grid[4:], lag_ratio=0.05),
            Write(col_labels[4:], lag_ratio=0.05),
            cols_label.animate.match_x(col_labels),
        )

        # Show knights move
        knight_group = self.get_knights_move_group(grid, 3)
        circle_cell = grid[2][0]

        self.play(FadeIn(knight_group))
        self.wait()
        self.play(circle_cell.animate.set_fill(RED, 0.35))
        self.wait()

        # New knights moves
        for d in range(4, 10):
            self.play(
                knight_group.animate.move_to(grid[d - 2][1], DL),
                FadeOut(q_marks[d - 4][0])
            )
            label_copy = boundary_labels[d].copy()
            self.play(
                TransformMatchingTex(boundary_labels[2].copy(), label_copy),
                TransformMatchingTex(volume_labels[d - 2].copy(), boundary_labels[d]),
                run_time=1
            )
            self.remove(label_copy)
            self.wait()
            self.show_derivative_and_integral(
                grid, d,
                int_added_anims=[
                    FadeOut(q_marks[d - 4][1]),
                    TransformMatchingTex(boundary_labels[d].copy(), volume_labels[d]),
                ],
                skip_derivative=True
            )
            self.wait()

        # Clean up
        self.play(
            FadeOut(knight_group),
            circle_cell.animate.set_fill(opacity=0)
        )

        # Show volume constants
        t2c = {"b_n": YELLOW, "b_{n - 2}": YELLOW}
        gen_formula = Tex(R"V(B^n) = b_n r^n", t2c=t2c, font_size=72)
        gen_formula["r"].set_color(BLUE)
        gen_formula.to_edge(DOWN)
        gen_b_part = gen_formula["b_n"][0]

        kw = dict(font_size=48)
        c_formulas = VGroup(
            Tex(R"b_0 = 1", **kw),
            Tex(R"b_1 = 2", **kw),
            Tex(R"b_2 = \\pi", **kw),
            Tex(R"b_3 = {4 \\over 3} \\pi", **kw),
            Tex(R"b_4 = {\\pi^2 \\over 2}", **kw),
            Tex(R"b_5 = {8 \\over 15} \\pi^2", **kw),
            Tex(R"b_6 = {\\pi^3 \\over 6}", **kw),
            Tex(R"b_7 = {16 \\over 105} \\pi^3", **kw),
            Tex(R"b_8 = {\\pi^4 \\over 24}", **kw),
        )

        self.play(Write(gen_formula))
        self.wait()

        last_highlight = VGroup()
        last_b_formula = VGroup()
        for col, b_formula, label in zip(grid[1:], c_formulas[1:], volume_labels[1:]):
            highlight = col[1].copy()
            highlight.set_fill(TEAL_E, 0.5)

            b_part = b_formula[re.compile(r"b_.")][0]
            b_part.set_color(YELLOW)
            b_formula.move_to(highlight)
            b_formula.shift(1.0 * highlight.get_height() * DOWN)

            group = VGroup(highlight, b_formula)
            self.play(
                FadeOut(last_highlight),
                FadeOut(last_b_formula),
                FadeIn(highlight),
                TransformFromCopy(gen_b_part, b_part),
                FadeTransform(
                    label[:len(b_formula) - 3].copy(),
                    b_formula[3:],
                    time_span=(0.25, 1)
                ),
                Write(b_formula[2], time_span=(0.25, 1.0)),
            )
            self.wait()

            last_highlight = highlight
            last_b_formula = b_formula
        self.play(
            FadeOut(last_highlight),
            FadeOut(last_b_formula),
        )

        # Show recursion formula
        recursion_formula = Tex(R"b_n = {2\\pi \\over n} b_{n - 2}", t2c=t2c, font_size=72)
        alt_recursion_formula = Tex(R"b_n = {\\pi \\over n / 2} b_{n - 2}", t2c=t2c, font_size=72)
        recursion_formula.to_corner(DR)
        alt_recursion_formula.move_to(recursion_formula)

        self.play(
            gen_formula.animate.match_y(recursion_formula).to_edge(LEFT, buff=LARGE_BUFF),
            TransformFromCopy(gen_formula["b_n"], recursion_formula["b_n"]),
        )
        self.play(Write(recursion_formula[2:]))
        self.wait()
        self.play(TransformMatchingTex(recursion_formula, alt_recursion_formula))
        self.wait()

        # Shrink
        zero_group = VGroup(grid[0], col_labels[0], labels[0])
        zero_group.set_fill(opacity=0)
        zero_group.set_stroke(opacity=0)
        grid_group = VGroup(grid, row_labels, col_labels, cols_label, labels)
        formula_group = VGroup(gen_formula, alt_recursion_formula)
        self.play(
            grid_group.animate.set_height(3.0, about_edge=UP),
            formula_group.animate.arrange(DOWN, buff=0.5, aligned_edge=LEFT).set_max_height(2).to_corner(DL)
        )
        self.wait()

        # Show recursion example
        stages = VGroup(
            Tex(R"b_8 = {\\pi \\over 4} b_6"),
            Tex(R"b_8 = {\\pi \\over 4} {\\pi \\over 3} b_4"),
            Tex(R"b_8 = {\\pi \\over 4} {\\pi \\over 3} {\\pi \\over 2} b_2"),
            Tex(R"b_8 = {\\pi \\over 4} {\\pi \\over 3} {\\pi \\over 2} {\\pi \\over 1} b_0"),
        )
        stages.set_height(1.2)
        stages.next_to(formula_group, RIGHT, buff=2.5)
        for stage in stages:
            stage.align_to(stages[-1], LEFT)
            stage[re.compile(r"b_.")].set_color(YELLOW)

        mult_arrows = VGroup(
            Arrow(
                col1.get_bottom(),
                col2.get_bottom(),
                path_arc=120 * DEG
            )
            for col1, col2 in zip(grid[0::2], grid[2::2])
        )
        arrow_label_texs = [
            R"\\times \\pi / 1",
            R"\\times \\pi / 2",
            R"\\times \\pi / 3",
            R"\\times \\pi / 4",
        ]
        for arrow, tex in zip(mult_arrows, arrow_label_texs):
            label = Tex(tex)
            label.next_to(arrow, DOWN, SMALL_BUFF)
            arrow.push_self_into_submobjects()
            arrow.add(label)

        zero_cells = grid[0]

        highlights = VGroup(col[1].copy() for col in grid[0::2])
        highlights.set_fill(TEAL, 0.5)

        self.play(
            FadeOut(boundary_labels[4:]),
            FadeOut(volume_labels[4:]),
        )
        self.play(
            Write(stages[0]),
        )
        self.wait()
        self.play(
            Write(mult_arrows[-1]),
            FadeIn(highlights[-2:]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[0], stages[1], key_map={"b_6": "b_4"}, run_time=1),
            Write(mult_arrows[-2]),
            FadeIn(highlights[-3]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[1], stages[2], key_map={"b_4": "b_2"}, run_time=1),
            Write(mult_arrows[-3]),
            FadeIn(highlights[-4]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(stages[2], stages[3], key_map={"b_2": "b_0"}, run_time=1),
            Write(mult_arrows[-4]),
            FadeIn(highlights[-5]),
            grid[0].animate.set_stroke(opacity=1),
            Write(col_labels[0]),
            row_labels.animate.next_to(grid, LEFT, MED_SMALL_BUFF)
        )
        self.wait()

        # Fill in zero terms
        self.play(volume_labels[0].animate.set_fill(opacity=1))
        self.wait()
        self.play(boundary_labels[0].animate.set_fill(opacity=1))
        self.wait()

        # General formula
        b8_form = stages[-1]

        gen_b_form = Tex(R"b_n = {\\pi^{n / 2} \\over (n / 2)!}", t2c=t2c, font_size=48)
        gen_b_form.move_to(b8_form)
        gen_b_form.to_edge(RIGHT, buff=LARGE_BUFF)

        small_b8_form = b8_form.copy()
        small_b8_form.generate_target()
        small_b8_form.target[-2:].set_opacity(0)
        small_b8_form.target.shift(0.75 * LEFT)
        small_b8_form.target.scale(gen_b_form[0].get_height() / small_b8_form[0].get_height())

        pis = b8_form[3:-2]
        pis_target = gen_b_form[R"{\\pi^{n / 2} \\over (n / 2)!}"][0]
        pis_rect = SurroundingRectangle(pis, buff=SMALL_BUFF)
        pis_rect.set_stroke(TEAL, 3)

        self.play(ShowCreation(pis_rect))
        self.wait()
        self.play(
            TransformMatchingTex(
                b8_form, gen_b_form,
                key_map={"b_8": "b_n"},
                matched_keys=[R"\\pi", R"\\over"]
            ),
            pis_rect.animate.surround(pis_target),
            MoveToTarget(small_b8_form),
            run_time=1
        )
        self.play(FadeOut(pis_rect))
        self.wait()

        # Substitute in
        final_formula = Tex(
            R"V(B^n) = {\\pi^{n / 2} \\over (n / 2)!} {r}^n",
            t2c={"{r}": BLUE},
            font_size=72
        )
        final_formula.next_to(grid, DOWN, buff=2.25)
        final_formula.to_edge(LEFT)

        bn_parts = VGroup(
            formula[R"{\\pi^{n / 2} \\over (n / 2)!}"]
            for formula in [gen_b_form, final_formula]
        )
        bn_rect = SurroundingRectangle(bn_parts[0])
        bn_rect.set_stroke(YELLOW, 1)

        self.play(
            FadeOut(small_b8_form),
            FadeOut(alt_recursion_formula),
        )
        self.play(ShowCreation(bn_rect))
        self.play(
            TransformFromCopy(*bn_parts),
            TransformFromCopy(
                gen_formula[R"V(B^n) = "].copy(),
                final_formula[R"V(B^n) = "],
            ),
            TransformFromCopy(
                gen_formula[R"r^n"],
                final_formula[R"{r}^n"],
            ),
            FadeOut(gen_b_form),
            FadeOut(gen_formula),
            bn_rect.animate.surround(bn_parts[1], buff=0.05),
        )
        self.wait()
        self.play(
            bn_rect.animate.surround(final_formula, buff=0.25).set_stroke(width=2),
        )
        self.add(final_formula)
        self.wait()

        # Fill in even volume labels
        def fill_every_other_label_from(n=2):
            for vl1, vl2 in zip(volume_labels[n::2], volume_labels[n + 2::2]):
                self.play(
                    TransformMatchingTex(vl1.copy(), vl2, path_arc=60 * DEG),
                    run_time=1
                )

            self.wait()
            self.play(LaggedStart(
                *(
                    TransformMatchingTex(vl.copy(), bl)
                    for vl, bl in zip(volume_labels[n + 2::2], boundary_labels[n + 2::2])
                ),
                run_time=1.5,
                lag_ratio=0.2
            ))

        fill_every_other_label_from(2)

        # Shift multiplication arrows
        pure_mult_arrows = VGroup(ma[0] for ma in mult_arrows)
        mult_arrow_labels = VGroup(ma[1] for ma in mult_arrows)

        shift_vect = grid[1].get_center() - grid[0].get_center()
        pure_mult_arrows.generate_target()
        pure_mult_arrows.target.shift(shift_vect)

        new_arrow_texs = [
            R"\\times {\\pi \\over 3 / 2}",
            R"\\times {\\pi \\over 5 / 2}",
            R"\\times {\\pi \\over 7 / 2}",
            R"\\times {\\pi \\over 9 / 2}",
        ]
        new_arrow_labels = VGroup(map(Tex, new_arrow_texs))
        for label, arrow in zip(new_arrow_labels, pure_mult_arrows.target):
            label.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            MoveToTarget(pure_mult_arrows, lag_ratio=0.2),
            LaggedStart(
                *(
                    TransformMatchingTex(l1, l2)
                    for l1, l2 in zip(mult_arrow_labels, new_arrow_labels)
                ),
                lag_ratio=0.2,
            ),
            run_time=1.5
        )
        self.wait()

        # Fill in odd volume labels
        fill_every_other_label_from(3)

        # Plug it in for n = 1
        d1_form = Tex(R"V(B^1) = {\\pi^{1/2} \\over (1/2)!} {r} = 2{r}", t2c={"{r}": BLUE})
        alt_d1_form = Tex(R"V(B^1) = {\\sqrt{\\pi} \\over (1/2)!} {r} = 2{r}", t2c={"{r}": BLUE})
        for form in d1_form, alt_d1_form:
            form.next_to(final_formula, RIGHT, buff=1.0, aligned_edge=DOWN)

        self.play(
            VGroup(final_formula, bn_rect).animate.scale(0.7, about_edge=DL),
            FadeTransform(final_formula.copy(), d1_form),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                d1_form,
                alt_d1_form,
                key_map={R"^{1/2}": R"\\sqrt"},
                matched_keys=[R"\\pi"],
                run_time=1,
            )
        )
        self.wait()

        # Half factoril fact
        half_fact = Tex(R"(1/2)! = {\\sqrt{\\pi} \\over 2}")
        half_fact.move_to(d1_form)

        self.play(TransformMatchingTex(alt_d1_form, half_fact, path_arc=-PI / 2))
        self.wait()

    def get_grid(self, n_cols=10, width=FRAME_WIDTH - 1):
        cell = Square()
        cell.set_stroke(WHITE, 2)
        col = cell.get_grid(2, 1, buff=0)
        grid = col.get_grid(1, n_cols, buff=0)
        grid.set_width(width)
        grid.to_edge(UP, buff=1.5)
        grid.set_z_index(-1)
        return grid

    def get_volume_labels(self):
        config = dict(
            t2c=self.tex_to_color,
            font_size=36,
        )
        return VGroup(
            VGroup(
                Tex(tex, **config)
                for tex in texs
            )
            for texs in [
                get_boundary_volume_texs(),
                get_volume_texs(),
            ]
        )

    def show_derivative_and_integral(
        self,
        grid,
        dim,
        upper_buff=1.25,
        deriv_added_anims=[],
        int_added_anims=[],
        skip_derivative=False
    ):
        top_cell = grid[dim][0]
        low_cell = grid[dim][1]
        right_point = VGroup(top_cell, low_cell).get_right()

        down_arrow = Arrow(
            top_cell.get_right(),
            low_cell.get_right(),
            buff=SMALL_BUFF,
            thickness=5,
            path_arc=-180 * DEG
        )
        down_arrow.scale(0.8, about_point=right_point)

        up_arrow = down_arrow.copy().flip(RIGHT)

        deriv_label = Tex(R"{d / dr}", t2c=self.tex_to_color)
        deriv_label.next_to(up_arrow, RIGHT, SMALL_BUFF)
        int_label = Tex(R"\\int \\dots dr", t2c=self.tex_to_color)
        int_label.next_to(down_arrow, RIGHT, SMALL_BUFF)

        cover_rect = Rectangle(width=grid.get_width(), height=grid.get_height() + upper_buff)
        cover_rect.set_fill(BLACK, 0.85)
        cover_rect.set_stroke(width=0)
        cover_rect.next_to(right_point, RIGHT, buff=5e-3)
        cover_rect.shift(1e-2 * RIGHT)

        if skip_derivative:
            self.play(
                FadeIn(cover_rect),
                Write(down_arrow),
                Write(int_label),
                *int_added_anims
            )
            self.wait()
        else:
            self.play(LaggedStart(
                FadeIn(cover_rect),
                Write(up_arrow),
                Write(deriv_label),
                *deriv_added_anims,
                lag_ratio=0.5,
            ))
            self.wait()
            self.play(
                TransformMatchingTex(deriv_label, int_label, run_time=1),
                ReplacementTransform(up_arrow, down_arrow),
                *int_added_anims,
            )
            self.wait()
        self.play(
            FadeOut(down_arrow),
            FadeOut(int_label),
            FadeOut(cover_rect),
        )

    def get_knights_move_group(self, grid, d, colors=[GREEN, YELLOW], opacity=0.4):
        # Test
        cells = VGroup(grid[d - 2][1], grid[d][0]).copy()
        for cell, color in zip(cells, colors):
            cell.set_fill(color, opacity)

        arrow = Arrow(cells[0], cells[1], thickness=5, buff=-0.25)
        arrow.set_backstroke(BLACK, 5)

        return VGroup(cells, arrow)


class ShowCircleAreaDerivative(InteractiveScene):
    def construct(self):
        # Shrinking difference
        r = 2
        dr_tracker = ValueTracker(0.5)
        get_dr = dr_tracker.get_value

        circle = self.get_circle(r)
        dA_group = always_redraw(lambda: self.get_dA_group(r, get_dr()))

        self.add(circle)
        self.add(dA_group)

        # Shrink
        dr_tracker.set_value(0)
        self.play(dr_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(dr_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

    def get_circle(self, r, fill_color=TEAL_E, fill_opacity=0.75, label="A"):
        result = VGroup()

        circle = Circle(radius=r)
        circle.set_fill(fill_color, fill_opacity)
        circle.set_stroke(WHITE, 0)
        result.add(circle)

        circle_label = Tex(label, font_size=72)
        circle_label.shift(0.5 * r * UP)
        result.add(circle_label)

        rad_line = Line(ORIGIN, r * RIGHT)
        rad_line.rotate(-45 * DEG, about_point=ORIGIN)
        r_label = Tex(R"r")
        r_label.next_to(rad_line.get_center(), UR, SMALL_BUFF)
        result.add(rad_line, r_label)

        return result

    def get_dA_group(self, r, dr, fill_color=RED_E, fill_opacity=0.5, label_color=WHITE):
        annulus = Annulus(r, r + dr)
        annulus.set_fill(fill_color, fill_opacity)
        annulus.set_stroke(width=0)
        line = Line(r * RIGHT, (r + dr) * RIGHT)
        dr_label = Tex(R"dr")
        dr_label.set_fill(label_color)
        dr_label.set_max_width(0.5 * line.get_width())
        dr_label.next_to(line, UP, buff=SMALL_BUFF)

        return VGroup(annulus, line, dr_label)


class CircleDerivativeFormula(InteractiveScene):
    def construct(self):
        # Test
        formulas = VGroup(
            Tex(tex, t2c={"dA": RED_D, "dr": BLUE})
            for tex in [
                R"dA = (2 \\pi r) dr",
                R"{dA \\over dr} = 2 \\pi r",
            ]
        )
        formulas.scale(3)

        self.add(formulas[0])
        self.play(TransformMatchingTex(*formulas, path_arc=-90 * DEG))
        self.wait()


class BuildCircleWithCombinedAnnulusses(ShowCircleAreaDerivative):
    def construct(self):
        # Test
        dr = 0.1
        radius = 3.9
        rings = VGroup(
            Annulus(r, r + dr)
            for r in np.arange(0, radius, dr)
        )
        rings.set_submobject_colors_by_gradient(TEAL_E, BLUE_E)
        rings.set_stroke(BLACK, 0.5, 1)
        for ring in rings:
            ring.insert_n_curves(100)

        self.play(FadeIn(rings, lag_ratio=0.5, run_time=3))


class ShowSphereVolumeDerivative(ShowCircleAreaDerivative):
    def construct(self):
        # Set up
        frame = self.frame
        self.set_floor_plane("xz")

        r = 3
        dr_tracker = ValueTracker(0)
        get_dr = dr_tracker.get_value

        circle = self.get_circle(r, label="V", fill_opacity=1)
        dV_group = always_redraw(lambda: self.get_dA_group(r, get_dr()))

        inner_sphere = Sphere(radius=r)
        inner_sphere.set_color(TEAL_E, 1)
        inner_sphere.set_clip_plane(IN, r)
        sphere_mesh = SurfaceMesh(inner_sphere, resolution=(51, 26))
        sphere_mesh.set_stroke(WHITE, 1, 0.2)
        sphere_mesh.rotate(90 * DEG, RIGHT)

        def get_outer_sphere():
            sphere = Sphere(radius=r + get_dr())
            sphere.set_color(RED_E, 0.5)
            sphere.set_clip_plane(IN, 0)
            sphere.sort_faces_back_to_front(LEFT)
            return sphere

        outer_sphere = always_redraw(get_outer_sphere)

        self.add(circle)
        self.add(inner_sphere, sphere_mesh)

        frame.reorient(-75, -21, 0, ORIGIN, 8.73)
        self.play(
            frame.animate.reorient(42, -15, 0, ORIGIN, 8.73),
            inner_sphere.animate.set_clip_plane(IN, 0),
            run_time=3,
        )
        self.add(inner_sphere, circle, sphere_mesh)
        self.play(FadeIn(circle))
        self.wait()

        # Show dV
        self.add(outer_sphere)
        self.add(dV_group)
        sphere_mesh.add_updater(lambda m: m.set_width(2 * (r + get_dr())).move_to(ORIGIN))
        self.play(dr_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(dr_tracker.animate.set_value(0.1), run_time=3)
        self.wait()

        # Clean shrinking
        self.clear()
        self.add(outer_sphere, sphere_mesh, dV_group)
        dV_group.add_updater(lambda m: m[1:].set_opacity(0))
        dr_tracker.set_value(0.5)
        self.play(dr_tracker.animate.set_value(0.0), run_time=5)



class SphereDerivativeFormula(InteractiveScene):
    def construct(self):
        # Test
        formulas = VGroup(
            Tex(tex, t2c={"dV": RED_D, "{r}": BLUE})
            for tex in [
                R"dV = (4 \\pi {r}^2) d{r}",
                R"{dV \\over d{r}} = 4 \\pi {r}^2",
            ]
        )
        formulas.scale(3)

        self.add(formulas[0])
        self.play(TransformMatchingTex(*formulas, path_arc=-90 * DEG))
        self.wait()


class SimpleLineWithEndPoints(InteractiveScene):
    def construct(self):
        # Test
        line = Line(LEFT, RIGHT)
        line.set_width(6)
        line.set_stroke(TEAL, 3)
        center_dot = Dot(ORIGIN, radius=0.05)

        brace = Brace(line, UP, buff=MED_SMALL_BUFF)
        brace.stretch(0.5, 0, about_edge=RIGHT)
        brace_label = brace.get_tex("r")

        end_points = Group(
            Group(Dot(), GlowDot()).move_to(point)
            for point in line.get_start_and_end()
        )
        end_points.set_color(YELLOW)

        self.add(line, center_dot)
        self.play(GrowFromCenter(brace), Write(brace_label))
        self.wait()
        self.play(FadeIn(end_points, lag_ratio=0.75))
        self.wait()


class ZAxisWithCircle(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.set_width(4)
        sphere = Sphere(radius=1)
        sphere.always_sort_to_camera(self.camera)
        sphere.set_color(BLUE, 0.2)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.25)

        z_tracker = ValueTracker(0.6)
        get_z = z_tracker.get_value

        z_line = Line(axes.c2p(0, 0, -1), axes.c2p(0, 0, 1))
        z_line.set_stroke(GREEN, 10)
        z_line.apply_depth_test()
        z_dot = TrueDot(color=GREEN, radius=0.05)
        z_dot.make_3d()
        z_dot.add_updater(lambda m: m.move_to(axes.z_axis.n2p(get_z())))

        circle = Circle(radius=0.8)
        circle.apply_depth_test()
        circle.set_stroke(RED, 10)
        circle.add_updater(lambda m: m.set_width(2.01 * math.sqrt(1 - get_z()**2)))
        circle.add_updater(lambda m: m.move_to(axes.z_axis.n2p(get_z())))

        circle_shadow = VGroup()

        def update_shadow(shadow):
            if len(shadow) > 0 and abs(shadow[-1].get_z() - circle.get_z()) < 5e-3:
                return
            shadow.add(circle.copy().clear_updaters().set_stroke(opacity=0.15, width=2).set_width(2))
            return shadow

        circle_shadow.add_updater(update_shadow)

        frame.reorient(23, 70, 0, (-0.06, 0.05, -0.19), 3.02)
        self.add(axes, z_line, z_dot, circle, sphere, mesh)
        # self.add(circle_shadow)
        self.play(z_tracker.animate.set_value(0.9), run_time=4)
        self.play(z_tracker.animate.set_value(-0.9), run_time=8)
        self.play(z_tracker.animate.set_value(0.6), run_time=6)


class SeparateRingsOfLatitude(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        sphere = Sphere(radius=1)
        sphere.set_color(BLUE_E)
        sphere.always_sort_to_camera(self.camera)

        n_rings = 50
        rings = VGroup(
            Circle(radius=math.sqrt(1 - z**2)).move_to(z * OUT)
            for z in np.linspace(-1, 1, n_rings)
        )
        rings.set_stroke(BLUE, 2, 0.5)

        frame.reorient(4, 78, 0, (-0.03, 0.01, 0.03), 2.88)
        self.add(sphere)
        self.play(
            sphere.animate.set_opacity(0.2),
            LaggedStartMap(FadeIn, rings),
            run_time=3
        )
        self.wait()
        self.play(
            rings[n_rings // 2].animate.set_stroke(YELLOW, 3, 1),
            rings[:n_rings // 2].animate.set_stroke(opacity=0.25),
            rings[n_rings // 2 + 1:].animate.set_stroke(opacity=0.25),
        )
        self.wait()


# TODO, too much code redundancy below?
class CrossLineWithCircle(InteractiveScene):
    def construct(self):
        # Equation
        line = Line(DOWN, UP)
        line.set_stroke(GREEN_E, 8)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = Group(
            Tex(R"\\partial B^3 = ", font_size=120),
            circle,
            Tex(R"\\times", font_size=120),
            line,
        )
        group[0].shift(MED_SMALL_BUFF * UL)
        group.arrange(RIGHT)
        group[-1].shift(0.5 * RIGHT)

        self.add(group)

        # Formulas
        sphere_3d_form = Tex(R"x^2 + y^2 + z^2 = 1")
        sphere_3d_form.to_corner(UL)
        sphere_3d_form.next_to(group[0], UP, buff=MED_LARGE_BUFF)
        sphere_3d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        line_form = Tex(R"z^2 \\le 1")
        line_form.next_to(line, DOWN)

        self.add(sphere_3d_form)
        self.add(circle_form)
        self.add(line_form)


class CrossDiskWithCircle(InteractiveScene):
    def construct(self):
        # Test
        disk = Circle()
        disk.set_fill(BLUE_E, 1)
        disk.set_stroke(WHITE, 1)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = VGroup(
            Tex(R"\\partial B^4 = ", font_size=120),
            circle,
            Tex(R"\\times", font_size=120),
            disk,
        )
        group[0].shift(SMALL_BUFF * UL)
        group.arrange(RIGHT)

        self.add(group)

        # Formulas
        sphere_4d_form = Tex(R"x^2 + y^2 + z^2 + w^2 = 1")
        sphere_4d_form.to_corner(UL)
        sphere_4d_form.next_to(group[0], UP, buff=LARGE_BUFF)
        sphere_4d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        disk_form = Tex(R"z^2 + w^2 \\le 1")
        disk_form.next_to(disk, DOWN)

        self.add(sphere_4d_form)
        self.add(circle_form)
        self.add(disk_form)


class CrossBallWithCircle(InteractiveScene):
    def construct(self):
        # Equation
        ball = Sphere()
        ball.set_color(BLUE_E, 1)
        circle = Circle()
        circle.set_stroke(RED, 3)

        group = Group(
            Tex(R"\\partial B^5 = ", font_size=120),
            circle,
            Tex(R"\\times", font_size=120),
            ball,
        )
        group[0].shift(SMALL_BUFF * UL)
        group.arrange(RIGHT)

        self.add(group)

        # Formulas
        sphere_4d_form = Tex(R"x^2 + y^2 + z^2 + w^2 + v^2 = 1")
        sphere_4d_form.to_corner(UL)
        sphere_4d_form.next_to(group[0], UP, buff=LARGE_BUFF)
        sphere_4d_form.shift(LEFT)

        circle_form = Tex(R"x^2 + y^2 = 1")
        circle_form.next_to(circle, DOWN)
        ball_form = Tex(R"z^2 + w^2 + v^2 \\le 1")
        ball_form.next_to(ball, DOWN)
        ball_form.shift(RIGHT)

        self.add(sphere_4d_form)
        self.add(circle_form)
        self.add(ball_form)


class ShowNumericalValues(InteractiveScene):
    def construct(self):
        # Set up
        axes = Axes((0, 25), (0, 6))
        axes.to_edge(UP, buff=LARGE_BUFF)
        axes.to_edge(LEFT, buff=MED_LARGE_BUFF)
        axes.x_axis.add_numbers()
        y_label = TexText("Volume of a\\nunit ball")
        y_label.next_to(axes.y_axis.get_top(), RIGHT)
        x_label = Text("Dimension")
        x_label.next_to(axes.x_axis.get_end(), UP)
        x_label.shift_onto_screen()
        axes.add(x_label)
        axes.add(y_label)

        def func(n):
            return math.pi**(n / 2) / math.gamma(n/2 + 1)

        graph = axes.get_graph(func)
        graph.set_stroke(BLUE, 2)

        self.add(axes)

        # Add terms
        formulas = VGroup(
            Tex(s.split(" r")[0])
            for s in get_volume_texs()
        )
        v_lines = VGroup(
            axes.get_v_line_to_graph(x, graph, line_func=Line)
            for x in range(len(formulas))
        )
        v_lines.set_stroke(BLUE, 5)
        dots = VGroup(Dot(line.get_end()) for line in v_lines)
        dots.set_fill(BLUE_E)

        expressions = VGroup()
        for n, formula, dot in zip(it.count(), formulas, dots):
            formula.next_to(dot, RIGHT)
            approx = VGroup(
                Tex(R"\\approx"),
                DecimalNumber(func(n))
            )
            approx.arrange(RIGHT)
            approx.next_to(formula, RIGHT)
            expressions.add(VGroup(formula, *approx))
            if n < 2:
                approx.set_fill(opacity=0)

        last_expression = VGroup()
        for v_line, dot, expression in zip(v_lines, dots, expressions):
            self.remove(last_expression)
            self.add(v_line, dot, expression)
            self.wait()
            last_expression = expression

        # Show general graph
        gen_formula = Tex(R"\\pi^{n/2} \\over (n/2)!")
        gen_formula.next_to(axes.i2gp(9, graph), UR)

        self.play(
            ShowCreation(graph),
            v_lines.animate.set_stroke(opacity=0.25),
            dots.animate.set_fill(opacity=0.25),
            FadeOut(last_expression[1:]),
            FadeTransform(last_expression[0], gen_formula),
            run_time=2
        )
        self.wait()
        self.play(
            self.frame.animate.reorient(0, 0, 0, (5.6, 0.15, 0.0), 14.77),
            run_time=3
        )
        self.wait()


class WriteB100Volume(InteractiveScene):
    def construct(self):
        # Test
        formula = Tex(R"B^{100} \\rightarrow {\\pi^{50} \\over 50!} \\approx 2.37 \\times 10^{-40}")
        self.add(formula)


class SphereEquator(InteractiveScene):
    def construct(self):
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        equator = Circle(radius=radius)
        sphere.set_color(BLUE_E, 0.7)
        sphere.always_sort_to_camera(self.camera)
        equator.set_stroke(YELLOW, 3)
        equator.apply_depth_test()

        self.add(equator, sphere)


class Distributions(InteractiveScene):
    def construct(self):
        # Test
        import matplotlib.pyplot as plt
        import torch

        # List of vectors in some dimension, with many
        # more vectors than there are dimensions
        num_vectors = 100000
        vector_len = 10000

        big_matrix = np.random.normal(size=(num_vectors, vector_len))
        norms = np.linalg.norm(big_matrix, axis=1)
        big_matrix /= norms[:, np.newaxis]

        plt.style.use('dark_background')
        plt.hist(big_matrix[:, -1], bins=1000, range=(-1, 1))
        plt.show()


class UnitCircleAndSquare(InteractiveScene):
    def construct(self):
        radius = 2
        circle = Circle(radius=radius)
        circle.set_fill(BLUE, 0.2)
        circle.set_stroke(BLUE, 3)
        square = Square(side_length=radius)
        square.set_stroke(RED, 3)
        square.set_fill(RED, 0.25)
        square.move_to(circle.get_center(), DL)
        one_label = Integer(1)
        one_label.next_to(square.get_bottom(), UP, SMALL_BUFF)

        self.add(circle, square, one_label)


class UniSphereAndSquare(InteractiveScene):
    def construct(self):
        radius = 2
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE, 0.2)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 1, 0.25)
        cube = VCube(side_length=radius)
        cube.move_to(ORIGIN, DL + IN)
        cube.set_color(RED, 0.25)
        cube.set_stroke(RED, 3)
        cube.deactivate_depth_test()
        one_label = Integer(1)
        one_label.rotate(90 * DEG, RIGHT)
        one_label.next_to(RIGHT, OUT, SMALL_BUFF)

        self.frame.reorient(25, 65, 0, (0.22, 0.13, 0.09), 4.66)
        self.add(sphere, mesh, cube, one_label)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      35: "CircumferenceToArea extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      36: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      39: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      46: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      53: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      54: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      58: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      63: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      66: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      67: "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
      74: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      77: "SurfaceAreaToVolume extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      78: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      82: "Sphere creates a parametric surface mesh. Can be textured, made transparent, and depth-tested for 3D rendering.",
      84: "Reorders transparent faces each frame for correct alpha blending from the current camera angle.",
      85: "SurfaceMesh draws wireframe grid lines on a Surface for spatial reference.",
      88: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      89: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      90: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      91: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      94: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      99: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      102: "Clips geometry to a half-space defined by a normal vector and offset. Used for cross-section reveals.",
      113: "VolumeGrid extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      116: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      684: "ShowCircleAreaDerivative extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      685: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      737: "CircleDerivativeFormula extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      738: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      754: "Class BuildCircleWithCombinedAnnulusses inherits from ShowCircleAreaDerivative.",
      755: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      771: "Class ShowSphereVolumeDerivative inherits from ShowCircleAreaDerivative.",
      772: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      831: "SphereDerivativeFormula extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      832: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      848: "SimpleLineWithEndPoints extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      849: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      873: "ZAxisWithCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      874: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      919: "SeparateRingsOfLatitude extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      920: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      951: "CrossLineWithCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      952: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      987: "CrossDiskWithCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      988: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1023: "CrossBallWithCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1024: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1059: "ShowNumericalValues extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1060: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1135: "WriteB100Volume extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1136: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1142: "SphereEquator extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1143: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1156: "Distributions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1157: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1176: "UnitCircleAndSquare extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1177: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1192: "UniSphereAndSquare extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1193: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();