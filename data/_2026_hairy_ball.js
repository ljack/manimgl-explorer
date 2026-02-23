(function() {
  const files = window.MANIM_DATA.files;

  files["_2026/hairy_ball/model3d.py"] = {
    description: "3D model scenes for the hairy ball video: sphere meshes, surface rendering, and tangent plane visualizations.",
    code: `from __future__ import annotations

from manim_imports_ext import *
from _2025.hairy_ball.spheres import fibonacci_sphere
from _2025.hairy_ball.spheres import get_sphereical_vector_field

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, Tuple
    # Something rom trimesh?


def faulty_perp(heading):
    return normalize(np.cross(heading, -OUT))


def get_position_vectors(trajectory, prop, d_prop=1e-4, use_curvature=False):
    if use_curvature:
        d_prop = 1e-2
    p0 = trajectory.pfp(clip(prop - d_prop, 0, 1))
    p1 = trajectory.pfp(prop)
    p2 = trajectory.pfp(clip(prop + d_prop, 0, 1))
    heading = normalize(p2 - p0)
    if use_curvature:
        acc = normalize(p2 - 2 * p1 + p0)
        top_vect = acc - IN
        wing_vect = normalize(np.cross(top_vect, heading))
    else:
        wing_vect = faulty_perp(heading)
    return (p1, heading, wing_vect)


class S3Viking(TexturedGeometry):
    offset = np.array([-0.2, 0, 0.2])

    def __init__(self, height=1):
        full_model = ThreeDModel("s3_viking/s3.obj")
        plane = full_model[-1]
        super().__init__(plane.geometry, plane.texture_file)
        self.set_height(height)
        self.apply_depth_test()
        self.rotate(PI).rotate(PI / 2, LEFT)

        # Trim, a bit hacky
        tube_index = 38_950
        idx = self.triangle_indices
        idx = idx[idx < tube_index]
        idx = idx[:-(len(idx) % 3)]
        self.triangle_indices = idx

        self.data = self.data[:tube_index]
        self.note_changed_data()
        self.refresh_bounding_box()
        self.move_to(height * self.offset)

        # Remoember position
        self.initial_points = self.get_points().copy()

    def reposition(self, center, heading, wing_vect):
        unit_heading = normalize(heading)
        roof_vect = normalize(np.cross(heading, wing_vect))
        true_wing = normalize(np.cross(roof_vect, heading))
        rot_mat_T = np.array([unit_heading, true_wing, roof_vect])

        self.set_points(np.dot(self.initial_points, rot_mat_T) + center)

    def place_on_path(self, trajectory, prop, use_curvature=False):
        self.reposition(*get_position_vectors(trajectory, prop, use_curvature=use_curvature))

    def set_partial(self, a, b):
        # Eh, no good
        opacities = self.data["opacity"].flatten()
        low_index = int(a * len(opacities))
        high_index = int(b * len(opacities))
        opacities[:] = 1
        opacities[:low_index] = 0
        opacities[high_index:] = 0
        self.set_opacity(opacities)
        return self


class RadioTower(VGroup):
    def __init__(self, height=4, stroke_color=GREY_A, stroke_width=2, **kwargs):
        self.legs = self.get_legs()
        self.struts = VGroup(
            self.get_struts(leg1, leg2)
            for leg1, leg2 in adjacent_pairs(self.legs)
        )

        super().__init__(self.legs, self.struts, **kwargs)

        self.set_stroke(stroke_color, stroke_width)
        self.set_depth(height)
        self.center()

    def get_legs(self):
        return VGroup(
            Line(point, 4 * OUT)
            for point in compass_directions(4, UR)
        )

    def get_struts(self, leg1, leg2, n_crosses=4):
        points1, points2 = [
            [leg.pfp(a) for a in np.linspace(0, 1, n_crosses + 1)]
            for leg in [leg1, leg2]
        ]
        return VGroup(
            *(Line(*pair) for pair in zip(points1, points2[1:])),
            *(Line(*pair) for pair in zip(points1[1:], points2)),
        )


class OrientAModel(InteractiveScene):
    def construct(self):
        # Add coordinate system
        frame = self.frame
        frame.reorient(11, 74, 0)
        axes = ThreeDAxes((-4, 4), (-4, 4), (-4, 4))
        axes.set_stroke(GREY_B, 1)
        xy_plane = NumberPlane((-4, 4), (-4, 4))
        xy_plane.axes.set_stroke(width=0)
        xy_plane.background_lines.set_stroke(GREY, 1, 0.5)
        xy_plane.faded_lines.set_stroke(GREY, 0.5, 0.25)

        frame.reorient(34, 76, 0, (-0.16, 0.01, 0.38), 8.16)

        # Add the plane
        plane = S3Viking(height=0.5)
        plane.scale(4, about_point=ORIGIN)

        frame.reorient(11, 49, 0)
        self.play(
            GrowFromPoint(plane, ORIGIN, time_span=(0, 2)),
            frame.animate.reorient(-33, 79, 0, (-0.16, 0.01, 0.38), 8.16),
            run_time=4
        )
        frame.add_ambient_rotation(1 * DEG)

        # Helix trajectory
        traj1 = ParametricCurve(
            lambda t: 2 * np.array([math.cos(t), math.sin(t), 0.04 * t**1.5 - 1]),
            t_range=(0, 15, 0.2)
        )
        traj1.set_stroke(YELLOW, 3, 0.5)
        traj1.set_z_index(1)
        prop_tracker = ValueTracker(0)

        def update_plane(s3, use_curvature=False):
            s3.place_on_path(traj1, prop_tracker.get_value(), use_curvature=use_curvature)

        self.play(
            plane.animate.place_on_path(traj1, prop_tracker.get_value()),
            ShowCreation(traj1),
            FadeIn(axes),
            FadeIn(xy_plane),
        )
        self.play(
            prop_tracker.animate.set_value(1),
            UpdateFromFunc(plane, update_plane),
            rate_func=linear,
            run_time=10
        )

        # Tweaked Gaussian trajectory
        traj2 = ParametricCurve(
            lambda t: t * RIGHT + np.exp(-0.1 * t**2) * (math.cos(2 * t) * UP + math.sin(2 * t) * OUT) + 2 * OUT,
            t_range=(-4, 4, 0.1)
        )
        traj2.match_style(traj1)
        prop_tracker.set_value(0)

        self.play(
            plane.animate.place_on_path(traj2, 0),
            Transform(traj1, traj2),
            run_time=2
        )
        self.play(
            prop_tracker.animate.set_value(1),
            UpdateFromFunc(plane, update_plane),
            frame.animate.reorient(41, 64, 0, (-0.17, 0.0, 0.38), 8.16),
            rate_func=linear,
            run_time=10
        )

        # Show a given point
        prop_tracker.set_value(0.2)
        traj1.insert_n_curves(1000)
        pre_traj = traj1.copy().pointwise_become_partial(traj1, 0, prop_tracker.get_value())
        pre_traj.set_stroke(WHITE, 0.75)
        plane.always_sort_to_camera(self.camera)

        vel_vector, wing_vector, center_dot = vect_group = Group(
            Vector(RIGHT, thickness=2).set_color(RED),
            Vector(RIGHT, thickness=2).set_color(PINK),
            TrueDot(radius=0.025, color=BLUE).make_3d(),
        )
        vect_group.set_z_index(2)
        vect_group.deactivate_depth_test()

        def update_vect_group(vect_group, use_curvature=False):
            center, vel, wing_dir = get_position_vectors(traj1, prop_tracker.get_value(), use_curvature=use_curvature)
            heading, wing, dot = vect_group
            dot.move_to(center)
            heading.put_start_and_end_on(center, center + vel)
            wing.put_start_and_end_on(center, center + wing_dir)

        update_vect_group(vect_group)
        vel_vector.always.set_perpendicular_to_camera(frame)
        wing_vector.always.set_perpendicular_to_camera(frame)

        self.add(pre_traj)
        self.play(
            plane.animate.reposition(center_dot.get_center(), RIGHT, UP).set_opacity(0.5),
            frame.animate.reorient(-7, 66, 0, (-1.99, -0.11, 1.79), 3.67),
            traj1.animate.set_stroke(GREY, 1, 0.5),
            FadeIn(center_dot),
            run_time=3
        )
        self.wait()
        self.play(Rotate(plane, TAU, axis=UR + OUT, about_point=center_dot.get_center()), run_time=6)
        self.wait()
        self.add(vel_vector, center_dot)
        self.play(GrowArrow(vel_vector))
        self.play(plane.animate.place_on_path(traj1, prop_tracker.get_value()), run_time=2)
        self.wait()
        self.play(Rotate(plane, TAU, axis=vel_vector.get_vector(), about_point=center_dot.get_center(), run_time=5))
        self.wait()
        self.play(GrowArrow(wing_vector))
        self.wait()
        self.play(
            Rotate(plane, TAU, axis=vel_vector.get_vector(), about_point=center_dot.get_center(), run_time=5),
            Rotate(wing_vector, TAU, axis=vel_vector.get_vector(), about_point=center_dot.get_center(), run_time=5),
        )
        self.wait()

        # Continue on the trajectory
        self.play(traj1.animate.set_stroke(WHITE, 1), FadeOut(pre_traj))
        self.play(
            UpdateFromFunc(plane, update_plane),
            UpdateFromFunc(vect_group, update_vect_group),
            prop_tracker.animate.set_value(1).set_anim_args(rate_func=linear),
            frame.animate.reorient(27, 69, 0, (-0.05, -0.3, 0.71), 8.18),
            run_time=10,
        )

        # One more trajectory
        traj3 = ParametricCurve(
            lambda t: 3 * math.cos(t) * RIGHT + 3 * math.sin(t) * UP + math.cos(3 * t) * OUT,
            t_range=(0, TAU, 0.01),
        )
        traj3.insert_n_curves(1000)
        traj3.set_stroke(WHITE, 1)
        center, vel, wing_dir = get_position_vectors(traj3, 0, use_curvature=False)

        prop_tracker.set_value(0)
        self.play(
            traj1.animate.become(traj3),
            plane.animate.place_on_path(traj3, 0).set_opacity(1),
            center_dot.animate.move_to(center),
            vel_vector.animate.put_start_and_end_on(center, center + vel),
            wing_vector.animate.put_start_and_end_on(center, center + wing_dir),
            frame.animate.reorient(-3, 47, 0, (-0.36, -0.81, -0.05), 8.18),
            run_time=2
        )
        self.play(
            UpdateFromFunc(plane, update_plane),
            UpdateFromFunc(vect_group, update_vect_group),
            prop_tracker.animate.set_value(1).set_anim_args(rate_func=linear),
            frame.animate.reorient(-49, 71, 0, (0.47, 0.37, 0.28), 6.45),
            run_time=10,
        )
        prop_tracker.set_value(0)
        self.play(
            prop_tracker.animate.set_value(1).set_anim_args(rate_func=linear),
            UpdateFromFunc(plane, update_plane),
            UpdateFromFunc(vect_group, update_vect_group),
            run_time=10
        )

        # Put plane in the center
        v_tracker = Point(normalize(vel_vector.get_vector()))
        self.play(
            plane.animate.reposition(ORIGIN, v_tracker.get_center(), faulty_perp(v_tracker.get_center())).set_opacity(0.3),
            center_dot.animate.move_to(ORIGIN),
            vel_vector.animate.put_start_and_end_on(ORIGIN, v_tracker.get_center()),
            wing_vector.animate.put_start_and_end_on(ORIGIN, faulty_perp(v_tracker.get_center())),
            FadeOut(traj1),
            frame.animate.reorient(-42, 74, 0, (-0.14, 0.44, -0.03), 3.68),
            run_time=5
        )

        # Add sphere of headings
        axis_tracker = Point(RIGHT)
        rot_group = Group(plane, vel_vector, wing_vector)
        wing_rotation = ValueTracker(0)
        wing_vector_offset = ValueTracker(0)

        sphere = Sphere(radius=1)
        sphere.set_color(GREY, 0.1)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(51, 26))
        mesh.set_stroke(WHITE, 2, 0.1)

        path = TracingTail(vel_vector.get_end, time_traced=10, stroke_color=RED)

        def update_v_tracker(v_tracker, dt):
            axis_tracker.rotate(dt * 10 * DEG, axis=DOWN, about_point=ORIGIN)
            v_tracker.rotate(dt * 31 * DEG, axis=axis_tracker.get_center(), about_point=ORIGIN)
            v_tracker.move_to(normalize(v_tracker.get_center()))

        def update_rot_group(group, dt):
            plane, vel_vector, wing_vector = group
            heading = v_tracker.get_center()
            wing_vect = rotate_vector(faulty_perp(heading), wing_rotation.get_value(), axis=heading)
            plane.reposition(ORIGIN, heading, wing_vect)
            vel_vector.put_start_and_end_on(ORIGIN, heading)
            wing_vector.put_start_and_end_on(ORIGIN, wing_vect)
            wing_vector.shift(wing_vector_offset.get_value() * heading)
            vel_vector.set_perpendicular_to_camera(frame)
            wing_vector.set_perpendicular_to_camera(frame)

        v_tracker.add_updater(update_v_tracker)
        rot_group.add_updater(update_rot_group)
        frame.add_ambient_rotation(2 * DEG)
        self.add(v_tracker, rot_group, path)
        self.wait(5)
        self.add(rot_group, mesh, sphere)
        self.play(ShowCreation(sphere), Write(mesh))
        self.wait(15)
        wing_rotation.set_value(90 * DEG)
        self.wait(4)
        wing_rotation.set_value(0)
        self.wait(6)
        self.play(VFadeOut(path))

        # Note infinitley many choices
        v_tracker.suspend_updating()
        frame.clear_updaters()
        self.play(v_tracker.animate.move_to(UP))

        wing_path = TracedPath(wing_vector.get_end, stroke_color=PINK)

        self.add(wing_path)
        wing_rotation.set_value(0)
        self.play(
            wing_rotation.animate.set_value(TAU).set_anim_args(run_time=6),
            frame.animate.reorient(135, 75, 0, (-0.14, 0.44, -0.03), 3.68).set_anim_args(run_time=10),
        )
        wing_path.clear_updaters()

        # Rotate heading around
        self.play(
            Rotate(
                Group(v_tracker, wing_path),
                TAU,
                axis=UR,
                about_point=ORIGIN,
            ),
            run_time=10
        )
        self.wait()

        # Note that wing vector is tangent
        tangent_plane = Square()
        tangent_plane.center()
        tangent_plane.set_width(2.5)
        tangent_plane.set_fill(WHITE, 0.15)
        tangent_plane.set_stroke(WHITE, 0.0)
        tangent_plane.save_state()
        wing_vector_offset.set_value(0)

        def update_tangent_plane(tangent_plane):
            tangent_plane.restore()
            tangent_plane.apply_matrix(rotation_between_vectors(OUT, v_tracker.get_center()))
            tangent_plane.shift(wing_vector_offset.get_value() * v_tracker.get_center())

        update_tangent_plane(tangent_plane)
        tangent_plane.set_opacity(0)

        self.add(tangent_plane, sphere, mesh)
        self.play(
            tangent_plane.animate.shift(1.01 * v_tracker.get_center()).set_fill(opacity=0.15),
            wing_path.animate.shift(1.01 * v_tracker.get_center()),
            wing_vector_offset.animate.set_value(1),
            run_time=2
        )
        self.play(
            frame.animate.reorient(73, 72, 0, (-0.14, 0.44, -0.03), 3.68),
            wing_rotation.animate.set_value(0),
            run_time=6
        )

        # Show the full vector field
        frame.clear_updaters()
        frame.add_ambient_rotation(-2 * DEG)

        points = fibonacci_sphere(1000)

        field = VectorField(
            lambda ps: np.array([faulty_perp(p) for p in ps]),
            axes,
            sample_coords=1.01 * points,
            color=PINK,
            max_vect_len_to_step_size=2,
            density=1,
            tip_width_ratio=4,
            tip_len_to_width=0.005,
        )
        field.set_stroke(opacity=0.8)
        field.apply_depth_test()
        field.add_updater(lambda m: m.update_vectors())
        field.set_stroke(PINK, opacity=0.5)

        tangent_plane.add_updater(update_tangent_plane)
        self.add(tangent_plane)
        self.play(FadeOut(wing_path))
        v_tracker.resume_updating()
        self.play(FadeIn(field, run_time=3))
        self.wait(10)
        tangent_plane.clear_updaters()
        self.play(FadeOut(tangent_plane))

        # Show the glitch
        v_tracker.suspend_updating()
        frame.clear_updaters()
        self.play(
            frame.animate.reorient(-22, 41, 0, (-0.1, -0.02, 0.13), 2.89),
            v_tracker.animate.move_to(normalize(LEFT + OUT)),
            plane.animate.set_opacity(0.5),
            run_time=2
        )
        self.play(
            Rotate(v_tracker, TAU, axis=UP, about_point=ORIGIN),
            run_time=10,
        )

    def alternate_trajectories(self):
        # Vertical loop
        traj1 = ParametricCurve(
            lambda t: 2 * np.array([math.sin(t), 0, -math.cos(t)]),
            t_range=(0, TAU, 0.1)
        )
        traj1.set_stroke(YELLOW, 3, 0.5)
        traj1.set_z_index(1)
        prop_tracker = ValueTracker(0)

        def update_plane(s3, use_curvature=False):
            s3.place_on_path(traj1, prop_tracker.get_value(), use_curvature=use_curvature)

        self.play(
            plane.animate.place_on_path(traj1, prop_tracker.get_value()),
            ShowCreation(traj1),
            FadeIn(axes),
            FadeIn(xy_plane),
            frame.animate.reorient(-52, 71, 0, (-0.29, 0.06, 0.48), 5.98),
        )
        self.play(
            prop_tracker.animate.set_value(1),
            UpdateFromFunc(plane, update_plane),
            rate_func=linear,
            run_time=8
        )


class RadioBroadcast(InteractiveScene):
    def construct(self):
        # Add tower
        frame = self.frame
        tower = RadioTower()
        tower.center()
        tower.move_to(ORIGIN, OUT)

        frame.reorient(-26, 93, 0, (0.44, -0.33, 1.02), 11.76)
        frame.add_ambient_rotation(3 * DEG)
        self.add(tower)

        # Add shells
        n_shells = 12
        shells = Group(Sphere() for n in range(n_shells))
        shells.set_color(RED)
        for shell in shells:
            shell.always_sort_to_camera(self.camera)

        time_tracker = ValueTracker(0)
        rate_tracker = ValueTracker(1)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt * rate_tracker.get_value()))

        def update_shells(shells):
            alpha = 1e-3 + time_tracker.get_value() % 1
            for n, shell in enumerate(shells):
                radius = n + alpha
                shell.set_width(2 * radius)
                shell.move_to(ORIGIN)
                dimmer = inverse_interpolate(n_shells, 0, radius)
                shell.set_opacity(0.25 * dimmer / radius)

        shells.add_updater(update_shells)

        self.add(shells, time_tracker)
        self.wait(8)

        # Show characters
        n_characters = 12
        characters = VGroup()
        modes = ["pondering", "thinking", "hesitant", "erm", "concentrating", "tease", "happy", "plain"]
        for theta in np.linspace(0, PI, n_characters):
            character = PiCreature(
                height=1.0,
                mode=random.choice(modes),
                color=random.choice([BLUE_A, BLUE_B, BLUE_C, BLUE_D, BLUE_D])
            )
            point = 6 * (math.cos(theta) * RIGHT + math.sin(theta) * UP)
            character.move_to(point)
            character.look_at(ORIGIN)
            characters.add(character)
        characters.rotate(90 * DEG, RIGHT, about_point=ORIGIN)

        frame.clear_updaters()
        self.play(
            LaggedStartMap(FadeIn, characters, lag_ratio=0.2, run_time=3),
            frame.animate.set_theta(0),
        )
        self.play(LaggedStartMap(Blink, characters, lag_ratio=0.2))
        self.wait(4)

        # Show a wave
        axes = ThreeDAxes()

        def wave_func(points, t, magnetic=False):
            real_time = time_tracker.get_value()
            normal = normalize_along_axis(points, 1)
            radii = np.linalg.norm(points, axis=1)
            perp = normalize_along_axis(np.cross(points, OUT), 1)
            if magnetic:
                direction = perp
            else:
                direction = np.cross(normal, perp)
            direction *= 1.0 - np.abs(np.dot(normal, OUT))[:, np.newaxis]
            return 0.25 * direction * np.cos(TAU * (radii - real_time))[:, np.newaxis]

        def E_wave_func(points, t):
            return wave_func(points, t, False)

        def B_wave_func(points, t):
            return wave_func(points, t, True)

        sample_points = np.linspace(ORIGIN, 10 * RIGHT, 100)
        E_wave, B_wave = [
            TimeVaryingVectorField(
                func, axes,
                sample_coords=sample_points,
                color=color,
                max_vect_len_to_step_size=np.inf,
                stroke_width=3
            )
            for color, func in zip([RED, TEAL], [E_wave_func, B_wave_func])
        ]

        points = sample_points

        self.play(
            VFadeIn(E_wave, time_span=(0, 1)),
            VFadeIn(B_wave, time_span=(0, 1)),
            FadeOut(characters, time_span=(0, 1)),
            frame.animate.reorient(49, 69, 0, (4.48, -0.2, -0.1), 2.71),
            rate_tracker.animate.set_value(0.5),
            run_time=3
        )
        self.play(
            frame.animate.reorient(122, 73, 0, (4.48, -0.2, -0.1), 2.71),
            run_time=11
        )

        # Show propagation direction
        radius = 5
        sample_point = radius * RIGHT
        prop_vect = Vector(sample_point, fill_color=YELLOW)

        def get_sample_vects(point):
            curr_time = time_tracker.get_value()
            result = VGroup(
                Vector(
                    func(point.reshape(1, -1), curr_time).flatten(),
                    fill_color=color
                )
                for color, func in zip([RED, TEAL], [E_wave_func, B_wave_func])
            )
            result.shift(point)
            return result

        sample_vects = always_redraw(lambda: get_sample_vects(sample_point))

        self.play(
            GrowArrow(prop_vect),
            E_wave.animate.set_stroke(opacity=0.1),
            B_wave.animate.set_stroke(opacity=0.1),
            VFadeIn(sample_vects),
        )
        self.wait(6)
        self.play(rate_tracker.animate.set_value(0))
        sample_vects.clear_updaters()

        # Show on the whole sphere
        sample_coords = radius * fibonacci_sphere(3000)
        sample_coords = np.array(list(sorted(sample_coords, key=lambda p: get_norm(p - sample_point))))
        fields = VGroup(
            VectorField(
                lambda p: func(p, time_tracker.get_value()),
                axes,
                sample_coords=sample_coords,
                stroke_width=3,
                max_vect_len_to_step_size=np.inf,
                max_vect_len=np.inf,
                color=color,
            )
            for func, color in zip([E_wave_func, B_wave_func], [RED, TEAL])
        )
        fields.set_stroke(opacity=0.5)
        fields.apply_depth_test()
        fields.set_scale_stroke_with_zoom(True)

        sphere = Sphere(radius=0.99 * radius)
        sphere.set_color(GREY_D, 0.5)
        sphere.always_sort_to_camera(self.camera)

        shells.clear_updaters()
        self.play(FadeOut(shells))
        self.play(
            FadeIn(sphere, time_span=(0, 2)),
            ShowCreation(fields, lag_ratio=0),
            frame.animate.reorient(36, 39, 0, (0.11, -0.24, 0.76), 11.24),
            run_time=12
        )
        self.play(frame.animate.reorient(-48, 69, 0, (0.4, -1.3, 0.14), 10.81), run_time=12)
        self.wait()`,
    annotations: {
      1: "Enables PEP 604 union types (X | Y) and postponed evaluation of annotations for cleaner type hints.",
      3: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "Imports fibonacci_sphere from the _2025.hairy_ball.spheres module within the 3b1b videos codebase.",
      5: "Imports get_sphereical_vector_field from the _2025.hairy_ball.spheres module within the 3b1b videos codebase.",
      15: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      28: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      34: "Class S3Viking inherits from TexturedGeometry.",
      42: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      62: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      63: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      66: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      83: "RadioTower extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      105: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      114: "OrientAModel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      115: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      118: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      119: "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
      121: "NumberPlane creates an infinite-looking 2D coordinate grid with major and minor gridlines.",
      126: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      132: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      133: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      135: "Smoothly animates the camera to a new orientation over the animation duration.",
      138: "Makes the camera slowly rotate at the given rate (in radians/second), providing 3D depth perception.",
      141: "ParametricCurve traces a function f(t) → (x,y,z) over a parameter range, producing a smooth 3D curve.",
      147: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      152: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      153: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      154: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      155: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      156: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      158: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      159: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      160: "UpdateFromFunc calls a function on each frame to update a mobject's state.",
      166: "ParametricCurve traces a function f(t) → (x,y,z) over a parameter range, producing a smooth 3D curve.",
      167: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      173: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      174: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      175: "Transform smoothly morphs one mobject into another by interpolating their points.",
      178: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      179: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      180: "UpdateFromFunc calls a function on each frame to update a mobject's state.",
      181: "Smoothly animates the camera to a new orientation over the animation duration.",
      188: "Subdivides existing bezier curves to increase point density. Needed before applying nonlinear transformations for smooth results.",
      191: "Reorders transparent faces each frame for correct alpha blending from the current camera angle.",
      213: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      214: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      215: "Smoothly animates the camera to a new orientation over the animation duration.",
      216: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      217: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      220: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      222: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      224: "GrowArrow animates an arrow growing from its start point to full length.",
      225: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      226: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      228: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      229: "GrowArrow animates an arrow growing from its start point to full length.",
      230: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      231: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      466: "RadioBroadcast extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      467: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2026/hairy_ball/old_functions.py"] = {
    description: "Earlier versions of hairy ball functions, preserved for reference. Contains alternative approaches to vector field visualization.",
    code: `from manim_imports_ext import *

def sanitize_3D_vector(pt):
    """
    Attempts to format input as a NumPy array of size (N,3)
    """

    # Try to convert the input to a NumPy array of the right size
    pts = np.array(pt,ndmin=2)

    if len(pts.shape) > 2:
        raise ValueError("3D vectors cannot have depth more than 2.")

    if pts.shape[1] != 3:
        raise ValueError("3D vectors must have 3 components.")

    return pts


def sanitize_scalar(x):
    """
    Attemps to format input as a NumPy array of size (N,)
    """

    # Try to convert input to a NumPy array of the right size
    x_array = np.array(x,ndmin=1)

    if len(x_array.shape) > 1:
        raise ValueError("Scalars cannot have depth more than 2.")

    return x_array


def direction_field(pt, discontinuities="equator", epsilon=0.01):
    """
    Parameters
    ----------
    pt : NumPy array (N,3), parallelizable
        Point on the unit sphere

    discontinuities : str
        Specifies where the direction field can be discontinuous.
        Three options:
            "equator"
            "two"
            "one"

    epsilon : float
        Determines error used in dealing with discontinuities

    Returns
    -------
    vec : NumPy array (N,3), parallelizable
        Associated unit vector
    """

    # Sanitize inputs
    pts = sanitize_3D_vector(pt)

    # Get coordinates of the points
    (x,y,z) = np.transpose(pts)

    # Split operation based on the number of discontinuities

    if discontinuities == "equator":
        # Define mask depending on whether we are close to the equator
        mask = (abs(z) < 0.1)

        # Define behavior on equator
        equator = np.stack((-y,x,0*z))

        # Define behavior off equator
        non_equator = np.stack((0*x,z,-y))

        # Combine using the mask
        perp = mask*equator + np.logical_not(mask)*non_equator

    elif discontinuities == "two":
        # Define mask depending on whether we are near the poles
        mask = abs(abs(z)-1) < epsilon

        # Define behavior around poles
        poles = np.stack((0*x,z,-y))

        # Define behavior away from poles
        non_poles = np.stack((-y,x,0*z))

        # Combine using the mask
        perp = mask*poles + np.logical_not(mask)*non_poles

    elif discontinuities == "one":
        # Define mask depending on whether we are near the north pole
        mask = abs(z-1) < epsilon

        # Define behavior around north pole
        north_pole = np.stack((0*x,z,-y))

        # Define behavior away from north pole
        X = x**2 - y**2 - (z-1)**2
        Y = 2*x*y
        Z = 2*x*(z-1)

        non_north_pole = np.stack((X,Y,Z))

        # Combine using the mask
        perp = mask*north_pole + np.logical_not(mask)*non_north_pole

    else:
        raise NotImplementedError()

    # Determine number of vectors
    field = np.transpose(perp)
    num_pts = field.shape[0]

    # normalize vector field before returning
    norm = np.linalg.norm(field, axis=1)
    norm = np.reshape(norm,[num_pts,1])
    vec = field/norm

    return vec


def distension(pt, t):
    """
    Helper function for computing amount to distend homotopy.

    Parameters
    ----------
    pt : NumPy array (M,3), parallelizable
        Point on the unit sphere
    t : float or NumPy array (N,)
        Time elapsed (between 0 and 1)

    Returns
    -------
    rho_factor : NumPy array (N,M)
        Adjustment to distance from the origin
    """

    # Compute the scaling factor from time
    time_factor = np.sin(np.pi * t)

    # Compute the scaling factor from space
    (x,y,z) = np.transpose(pt)
    space_factor = y*z

    # Combine the factors
    rho_factor = np.tensordot(time_factor, space_factor, axes = 0)

    return rho_factor


def great_circle_map(pt, t, discontinuities="one", distend=0, epsilon=0.01):
    """
    Parameters
    ----------
    pt : NumPy array (M,3), parallelizable
        Point on the unit sphere

    t : float or NumPy array (N,)
        Time elapsed (between 0 and 1)

    discontinuities : str
        Specifies where the direction field can be discontinuous.
        Three options:
            "equator"
            "two"
            "one"

    distend : float or NumPy array (T,)
        Amount to distend from spherical surface

    epsilon : float
        Determines error used in dealing with discontinuities

    Returns
    -------
    new_pts : NumPy array (N,T,M,3), parallelizable
        Location of pt after time t
    """

    # Sanitize inputs
    times = sanitize_scalar(t)
    dist_factors = sanitize_scalar(distend)
    pts = sanitize_3D_vector(pt)

    # Calculate initial unit vectors
    units = direction_field(pts, discontinuities, epsilon)

    # Compute weights for the great circle map
    scaled_times = np.pi * times
    u1 = np.cos(scaled_times)
    u2 = np.sin(scaled_times)

    # Compute linear combination using the constructed weights
    base_pts = (np.tensordot(u1, pts,axes=0) + np.tensordot(u2, units,axes=0))

    # Compute distension factors
    rho_factors = distension(pts, times)
    full_factors = 1 + np.tensordot(rho_factors, dist_factors,axes=0)

    # Reshape the base points and the scaling factors so that they can be multiplied together cleanly
    base_pts_reshaped = np.expand_dims(base_pts, axis=2)
    factors_reshaped = np.expand_dims(full_factors, axis=-1)

    # Combine and then reshape the array for convenience
    new_pts = base_pts_reshaped * factors_reshaped
    new_pts = new_pts.transpose((0,2,1,3))

    return new_pts


### TEST FUNCTIONS

def test_direction_field(num_pts=1000, epsilon=0.001):
    """
    Function to test correctness of direction_field

    Parameters
    ----------
    num_pts : int
        Number of points on sphere to test. The default is 1000.
    epsilon : float
        Acceptable error. The default is 0.001.

    Returns
    -------
    None
    """

    points = fibonacci_sphere(num_pts)


    failed_counts = {"equator":0,"two":0,"one":0}

    for disc in ["equator","two","one"]:
        field = direction_field(points,discontinuities=disc,epsilon=epsilon)
        dots = (points*field).sum(axis=1)

        failures = (abs(dots) >= epsilon).sum(axis=0)
        failed_counts[disc] = failures

    print("Number of points where the direction field failed to be orthogonal to the sphere.")
    print("")

    for key in failed_counts.keys():
        value = failed_counts[key]

        print(f"{key}: {value} points")

    print("")
    print("Count completed.")
    return None


def test_great_circle_map(discontinuities="one"):
    """
    Tests correctness of great_circle_map

    Parameters
    ----------
    discontinuities : str
        Selects base vector field. The default is "one".

    Returns
    -------
    None.

    """

    # Define acceptable error
    epsilon = 0.000001

    # Initialize points on the sphere to try
    pts = fibonacci_sphere(15)

    # Initialize distension amounts
    distends = [0.1*x for x in range(11)]

    # Initialize first batch of points
    end_pts = great_circle_map(pts,[0,1],discontinuities=discontinuities,distend=distends)

    # Check whether dimensions match what they should be
    pts_dims = end_pts.shape

    if len(pts_dims) != 4:
        print("Warning! Array has an incorrect number of dimensions.")

    elif pts_dims != (2,len(distends),len(pts),3):
        print("Warning: Dimensions of array are incorrect.")

    else:
        print("Array has expected dimensions.")
        print("")

    # Separate points belonging to the beginning and end of the homotopy
    beginning = end_pts[0]
    ending = end_pts[1]

    identity_pass = True
    antipode_pass = True

    # Go through all copies of beginning; see if they match the identity
    for distend,pts_copy in zip(distends,beginning):
        if np.linalg.norm(pts-pts_copy) > epsilon:
            if identity_pass:
                print("Warning! At time t=0, non-identity map at distensions:")

            identity_pass = False

            print(distend)

    # If there have been no errors, print success
    if identity_pass:
        print("Homotopy correctly defaults to identity at time t=0")
        print("No dependence on distension")

    print("")

    # Go through all copies of ending; see if they match antipode
    for distend,pts_copy in zip(distends,ending):
        if np.linalg.norm(pts+pts_copy) > epsilon:
            if antipode_pass :
                print("Warning! At time t=1, non-antipode map at distensions:")

            antipode_pass = False

            print(distend)

    # If there have been no errors, print success
    if antipode_pass:
        print("Homotopy correctly defaults to antipode at time t=1")
        print("No dependence on distension")

    print("")

    # Initialize second batch of points
    halfway_pts = great_circle_map(pts,1/2,discontinuities=discontinuities)[0][0]

    # At zero distension, halfway points should be sqrt(2) away from where they started

    # Compute distance between halfway points and original
    distances = np.linalg.norm(halfway_pts-pts,axis=1)

    # Compute discrepancy away from sqrt(2) expected distance
    discrepancies = np.abs(distances - np.sqrt(2))

    # Check whether maximum discrepancy is larger than the tolerance
    if np.max(discrepancies) > epsilon:
        print("Warning! Points tested at the halfway point at zero distension are in the wrong position.")
    else:
        print("Points are halfway around the great circle at t=1/2 with distension 0.")

    print("")

    # Initialize third batch of points
    infinitesimal_pts = great_circle_map(pts,epsilon**2,discontinuities=discontinuities)[0][0]

    # At zero distension, moving points an infinitesimal amount should agree with the underlying vector field

    # Compute difference in positions to get velocity
    velocities_actual = infinitesimal_pts - beginning[0]

    # Compute underlying field and rescale
    units = direction_field(pts,discontinuities=discontinuities)
    velocities_expected = units * np.pi * (epsilon**2)

    # Compute differences between velocities
    velocity_discrepancy = np.linalg.norm(velocities_actual - velocities_expected,axis=1)

    if np.max(velocity_discrepancy) > epsilon:
        print("Warning! Points do not appear to move in the direction of the vector field.")
    else:
        print("Points move in the direction of the vector field.")

    print("")

    # Compute fourth batch of points
    # Choose a selection of random times between 0 and 1
    times=np.random.rand(5)
    distension_pts = great_circle_map(pts,times,discontinuities=discontinuities,distend=distends)

    # Pull out subarray with zero distension
    zero_distension_pts = distension_pts[0:,0:1,0:,0:]

    # Compute norms of all points and normalize
    norms = np.expand_dims(np.linalg.norm(distension_pts,axis=-1),axis=-1)
    normalized_pts = distension_pts/norms

    # Normalized points should match zero distension ones
    differences = normalized_pts - zero_distension_pts
    discrepancies = np.linalg.norm(differences,axis=-1)

    if np.max(discrepancies) > epsilon:
        print("Warning! Discrepancy is shifting the directions of points, not just radially.")
    else:
        print("Discrepancy moves points only radially.")

    print("")
    print("All tests completed.")

    return None


## Manim Scenes

def spherical_surface(theta, phi):
    X = np.sin(phi) * np.cos(theta)
    Y = np.sin(phi) * np.sin(theta)
    Z = np.cos(phi)
    return np.array([[X, Y, Z]])


def spherical_eversion(theta, phi, t):
    pt = spherical_surface(theta, phi)
    new_pt = great_circle_map(pt, t, discontinuities="one", distend=0.5)
    return new_pt[0][0][0]`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      116: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      141: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      192: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      193: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      231: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      275: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      305: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      322: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      343: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      369: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      387: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      392: "Vector norm (magnitude/length). Used for normalization and distance calculations.",
      408: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      409: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      410: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
    }
  };

  files["_2026/hairy_ball/spheres.py"] = {
    description: "Core scenes for the Hairy Ball Theorem video. Includes Fibonacci sphere point generation, stereographic projection (and inverse), vector field transformations via the Jacobian, tangent vector fields on spheres, and animated stream lines demonstrating why every continuous tangent field on S² must have a zero.",
    code: `from manim_imports_ext import *
import numpy as np


if TYPE_CHECKING:
    from typing import Callable, Iterable, Sequence, TypeVar, Tuple, Optional
    from manimlib.typing import Vect2, Vect3, VectN, VectArray, Vect2Array, Vect3Array, Vect4Array


def fibonacci_sphere(samples=1000):
    """
    Create uniform-ish points on a sphere

    Parameters
    ----------
    samples : int
        Number of points to create. The default is 1000.

    Returns
    -------
    points : NumPy array
        Points on the unit sphere.

    """

    # Define the golden angle
    phi = np.pi * (np.sqrt(5) - 1)

    # Define y-values of points
    pos = np.array(range(samples), ndmin=2)
    y = 1 - (pos / (samples - 1)) * 2

    # Define radius of cross-section at y
    radius = np.sqrt(1 - y * y)

    # Define the golden angle increment
    theta = phi * pos

    # Define x- and z- values of poitns
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius

    # Merge together x,y,z
    points = np.concatenate((x, y, z))

    # Transpose to get coordinates in right place
    points = np.transpose(points)

    return points


def stereographic_proj(points3d, epsilon=1e-10):
    x, y, z = points3d.T

    denom = 1 - z
    denom[np.abs(denom) < epsilon] = np.inf
    return np.array([x / denom, y / denom, 0 * z]).T


def inv_streographic_proj(points2d):
    u, v = points2d.T
    norm_squared = u * u + v * v
    denom = 1 + norm_squared
    return np.array([
        2 * u / denom,
        2 * v / denom,
        (norm_squared - 1) / denom,
    ]).T


def right_func(points):
    return np.repeat([[1, 0]], len(points), axis=0)


def stereographic_vector_field(points3d, vector_field_2d):
    points2d = stereographic_proj(points3d)[:, :2]
    vects2d = vector_field_2d(points2d)
    u, v = points2d.T
    vect_u, vect_v = vects2d.T

    # Compute Jacobian
    r_squared = u**2 + v**2
    denom = 1 + r_squared
    denom_squared = denom**2

    # For x = 2u / (1 + u² + v²):
    dx_du = 2 * (1 + v**2 - u**2) / denom_squared
    dx_dv = -4 * u * v / denom_squared

    # For y = 2v / (1 + u² + v²):
    dy_du = -4 * u * v / denom_squared
    dy_dv = 2 * (1 + u**2 - v**2) / denom_squared

    # For z = (u² + v² - 1) / (1 + u² + v²):
    dz_du = 4 * u / denom_squared
    dz_dv = 4 * v / denom_squared

    # Apply the Jacobian: [v_x, v_y, v_z]^T = J × [v_u, v_v]^T
    vect_x = dx_du * vect_u + dx_dv * vect_v
    vect_y = dy_du * vect_u + dy_dv * vect_v
    vect_z = dz_du * vect_u + dz_dv * vect_v

    return np.array([vect_x, vect_y, vect_z]).T


def rotation_field(points3d, axis=IN):
    return np.cross(points3d, axis)


def flatten_field(points3d, vector_field_3d):
    vects = vector_field_3d(points3d)
    norms = normalize_along_axis(points3d, 1)
    return np.cross(vects, norms)


def get_sphereical_vector_field(
    v_func,
    axes,
    points,
    color=BLUE,
    stroke_width=1,
    mvltss=1.0,
    tip_width_ratio=4,
    tip_len_to_width=0.01,
):

    field = VectorField(
        v_func, axes,
        sample_coords=1.01 * points,
        max_vect_len_to_step_size=mvltss,
        density=1,
        stroke_width=stroke_width,
        tip_width_ratio=tip_width_ratio,
        tip_len_to_width=tip_len_to_width,
    )
    field.apply_depth_test()
    field.set_stroke(color, opacity=0.8)
    field.set_scale_stroke_with_zoom(True)
    return field


class SphereStreamLines(StreamLines):
    def __init__(self, func, coordinate_system, density=50, sample_coords=None, **kwargs):
        self.sample_coords = sample_coords
        super().__init__(func, coordinate_system, density=density, **kwargs)

    def get_sample_coords(self):
        if self.sample_coords is None:
            coords = fibonacci_sphere(int(4 * PI * self.density))
        else:
            coords = self.sample_coords
        return coords

    def draw_lines(self):
        super().draw_lines()
        for submob in self.submobjects:
            submob.set_points(normalize_along_axis(submob.get_points(), 1))
        return self


# Scenes


class TeddyHeadSwirl(InteractiveScene):
    def construct(self):
        img = ImageMobject("TeddyHead")
        img.set_height(FRAME_HEIGHT)
        img.fix_in_frame()
        # self.add(img)

        # Add sphere
        frame = self.frame
        axes = ThreeDAxes()

        def v_func(points):
            perp = np.cross(points, OUT)
            alt_perp = np.cross(points, perp)
            return 0.4 * (perp + alt_perp)

        top_sample_points = np.array([
            point for point in fibonacci_sphere(10_000) if point[2] > 0.95
        ])
        full_sample_points = fibonacci_sphere(1000)

        top_lines, full_lines = lines = VGroup(*(
            SphereStreamLines(
                v_func,
                axes,
                sample_coords=samples,
                arc_len=0.25,
                dt=0.05,
                max_time_steps=10,
            )
            for samples in [top_sample_points, full_sample_points]
        ))
        lines.set_stroke(WHITE, 2, 0.7)
        full_lines.set_stroke(WHITE, 2, 0.7)
        top_animated_lines = AnimatedStreamLines(top_lines)
        full_animated_lines = AnimatedStreamLines(full_lines)

        frame.reorient(-50, 12, 0, (0.15, -0.04, 0.0), 3.09)
        self.add(top_animated_lines)
        self.wait(4)
        self.add(full_animated_lines)
        self.play(
            frame.animate.reorient(-39, 75, 0),
            run_time=6
        )


class IntroduceVectorField(InteractiveScene):
    def construct(self):
        # Set up sphere
        frame = self.frame
        self.camera.light_source.move_to([0, -10, 10])
        radius = 3
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.scale(radius)
        sphere = Sphere(radius=radius)
        sphere.set_color(BLUE_B, 0.3)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, (51, 25))
        mesh.set_stroke(WHITE, 1, 0.15)

        frame.reorient(0, 90, 0)
        self.play(
            frame.animate.reorient(30, 65, 0),
            ShowCreation(sphere),
            Write(mesh, time_span=(1, 3), lag_ratio=1e-2),
            run_time=3
        )

        # Tangent plane
        v_tracker = Point()
        v_tracker.move_to(radius * OUT)

        def place_on_vect(mobject):
            matrix = z_to_vector(v_tracker.get_center())
            mobject.set_points(np.dot(mobject.points_at_zenith, matrix.T))

        v_dot = Dot(radius=0.05)
        v_dot.set_fill(YELLOW)

        plane = Square(side_length=2 * radius)
        plane.set_stroke(WHITE, 1)
        plane.set_fill(GREY, 0.5)

        for mob in [plane, v_dot]:
            mob.move_to(radius * OUT)
            mob.points_at_zenith = mob.get_points().copy()
            mob.add_updater(place_on_vect)

        self.play(
            VFadeIn(v_dot),
            v_tracker.animate.move_to(radius * normalize(RIGHT + OUT)),
            run_time=2
        )
        self.wait()
        plane.update()
        plane.suspend_updating()
        self.play(
            FadeIn(plane),
            frame.animate.reorient(13, 78, 0),
            run_time=2
        )
        self.play(frame.animate.reorient(53, 60, 0, (-0.29, 0.09, 0.31), 8.97), run_time=2)
        self.wait()

        # Show one vector
        def v_func(points3d):
            v1 = stereographic_vector_field(points3d, right_func)
            v2 = normalize_along_axis(rotation_field(points3d, RIGHT), 1)
            v3 = normalize_along_axis(rotation_field(points3d, RIGHT + IN), 1)
            return (3 * v1 + v2 + v3) / 6

        def vector_field_3d(points3d):
            x, y, z = points3d.T
            return np.array([
                np.cos(3 * x) * np.sin(y),
                np.cos(5 * z) * np.sin(x),
                -z**2 + x,
            ]).T

        def alt_v_func(points3d):
            return normalize_along_axis(flatten_field(points3d, vector_field_3d), 1)

        def get_tangent_vect():
            origin = v_tracker.get_center()
            vector = Arrow(origin, origin + radius * v_func(normalize(origin).reshape(1, -1)).flatten(), buff=0, thickness=3)
            vector.set_fill(BLUE)
            vector.set_perpendicular_to_camera(frame)
            return vector

        tangent_vect = get_tangent_vect()
        self.add(plane, tangent_vect, v_dot)
        self.play(GrowArrow(tangent_vect))
        self.play(Rotate(tangent_vect, TAU, axis=v_tracker.get_center(), about_point=tangent_vect.get_start(), run_time=2))
        self.wait()

        # Show more vectors
        plane.resume_updating()
        tangent_vect.add_updater(lambda m: m.match_points(get_tangent_vect()))
        og_vect = rotate_vector(v_tracker.get_center(), 10 * DEG, axis=DOWN)
        v_tracker.clear_updaters()
        v_tracker.add_updater(lambda m, dt: m.rotate(60 * DEG * dt, axis=og_vect, about_point=ORIGIN))
        v_tracker.add_updater(lambda m, dt: m.rotate(1 * DEG * dt, axis=RIGHT, about_point=ORIGIN))

        frame.clear_updaters()
        frame.add_ambient_rotation(-2.5 * DEG)

        self.wait(2)

        field = self.get_vector_field(axes, v_func, 4000, start_point=v_tracker.get_center())
        self.add(field, plane, v_tracker, tangent_vect, v_dot)
        self.play(
            ShowCreation(field),
            run_time=5
        )
        self.wait(4)
        self.play(
            FadeOut(plane),
            FadeOut(tangent_vect),
            FadeOut(v_dot),
            frame.animate.reorient(0, 59, 0, (-0.03, 0.15, -0.08), 6.80),
            run_time=4
        )
        self.wait(5)

        # Show denser field
        dots, dense_dots = [
            DotCloud(fibonacci_sphere(num), radius=0.01)
            for num in [4000, 400_000]
        ]
        for mob in [dots, dense_dots]:
            mob.make_3d()
            mob.set_color(WHITE)
            mob.scale(radius * 1.01)

        dense_field = self.get_vector_field(axes, v_func, 50_000, mvltss=5.0)
        dense_field.set_stroke(opacity=0.35)

        dots.set_radius(0.02)
        dense_dots.set_radius(0.01)

        self.play(ShowCreation(dots, run_time=3))
        self.wait()
        self.play(
            FadeOut(dots, time_span=(1.5, 2.5)),
            FadeOut(field, time_span=(2.5, 3)),
            ShowCreation(dense_field),
            run_time=3
        )
        frame.clear_updaters()
        self.play(frame.animate.reorient(-66, 52, 0, (-0.51, 0.27, 0.2), 2.81), run_time=4)
        self.wait()

        # Show plane and tangent vector again
        field.save_state()
        field.set_stroke(width=1e-6)
        self.play(
            dense_field.animate.set_stroke(width=1e-6),
            Restore(field, time_span=(1, 3)),
            frame.animate.reorient(48, 54, 0, (-2.74, -2.49, -0.1), 11.93),
            VFadeIn(plane),
            VFadeIn(tangent_vect),
            run_time=5
        )
        self.wait(5)

        v_tracker.clear_updaters()
        null_point = 3 * normalize(np.array([-1, -0.25, 0.8]))
        self.play(
            v_tracker.animate.move_to(null_point),
            frame.animate.reorient(-43, 55, 0, (-2.42, 2.76, -0.55), 9.22),
            run_time=3
        )
        self.remove(tangent_vect)
        v_dot.update()
        self.play(
            FadeIn(v_dot, scale=0.25),
            FadeOut(plane, scale=0.25),
        )
        self.wait()

        # Define streamlines
        static_stream_lines = SphereStreamLines(
            lambda p: v_func(np.array(p).reshape(-1, 3)).flatten(),
            axes,
            density=400,
            stroke_width=2,
            magnitude_range=(0, 10),
            solution_time=1,
        )
        static_stream_lines.scale(radius)
        static_stream_lines.set_stroke(BLUE_B, 3, 0.8)
        stream_lines = AnimatedStreamLines(static_stream_lines, lag_range=10, rate_multiple=0.5)
        stream_lines.apply_depth_test()

        # Show the earth
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.rotate(-90 * DEG, IN)
        earth.scale(1.001)
        frame.add_ambient_rotation(2 * DEG)

        self.add(sphere, earth, stream_lines, mesh, field)
        self.play(
            FadeOut(v_dot),
            FadeIn(earth, time_span=(0, 3)),
            field.animate.set_stroke(opacity=0.25).set_anim_args(time_span=(0, 2)),
            frame.animate.reorient(-90, 74, 0, (-1.37, 0.04, 0.37), 5.68),
        )
        self.wait(10)

        # Zoom in to null point
        frame.clear_updaters()
        dense_field = self.get_vector_field(axes, v_func, 10_000)
        dense_field.set_stroke(opacity=0.35)
        self.play(
            FadeOut(field, run_time=2),
            FadeIn(dense_field, run_time=2),
            frame.animate.reorient(-69, 54, 0, (-0.01, 0.19, -0.04), 4.81),
            run_time=5
        )
        self.wait(3)
        self.play(
            frame.animate.reorient(129, 70, 0, (-0.18, 0.18, -0.1), 8.22),
            run_time=20
        )
        self.wait(10)

    def get_vector_field(self, axes, v_func, n_points, start_point=None, mvltss=1.0, random_order=False):
        points = fibonacci_sphere(n_points)

        if start_point is not None:
            points = points[np.argsort(np.linalg.norm(points - start_point.reshape(-1, 3), axis=1))]
        if random_order:
            indices = list(range(len(points)))
            random.shuffle(indices)
            points = points[indices]

        alpha = clip(inverse_interpolate(10_000, 1000, n_points), 0, 1)
        stroke_width = interpolate(1, 3, alpha**2)

        return get_sphereical_vector_field(v_func, axes, points, mvltss=mvltss, stroke_width=stroke_width)

    def old(self):
        # Vary the density
        density_tracker = ValueTracker(2000)
        field.add_updater(
            lambda m: m.become(self.get_vector_field(int(density_tracker.get_value()), axes, v_func))
        )
        self.add(field)
        field.resume_updating()
        frame.suspend_updating()
        self.play(
            density_tracker.animate.set_value(50_000),
            frame.animate.reorient(-30, 50, 0, (-0.07, 0.02, 0.36), 3.01),
            run_time=5,
        )
        field.suspend_updating()
        self.wait(3)


class StereographicProjection(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        sphere = Sphere(radius=1)
        sphere.set_opacity(0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere)
        mesh.set_stroke(WHITE, 1, 0.25)

        self.add(sphere, mesh, axes, plane)
        frame.reorient(-15, 64, 0, (0.0, 0.1, -0.09), 4.0)

        # Show the 2d cross section
        frame.clear_updaters()
        sphere.set_clip_plane(UP, 1)
        n_dots = 20
        sample_points = np.array([
            math.cos(theta) * OUT + math.sin(theta) * RIGHT
            for theta in np.linspace(0, TAU, n_dots + 2)[1:-1]
        ])
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, 0),
            frame.animate.reorient(-43, 74, 0, (0.0, 0.0, -0.0), 3.50),
            FadeIn(sphere_dots, time_span=(1, 2)),
            ShowCreation(proj_lines, lag_ratio=0, time_span=(1, 2)),
            run_time=2
        )
        frame.add_ambient_rotation(2 * DEG)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)
        self.play(
            TransformFromCopy(sphere_dots, plane_dots, lag_ratio=0.5, run_time=10),
        )
        self.wait(3)

        planar_group = Group(sphere_dot_ghosts, plane_dots, proj_lines)

        # Show more points on the sphere
        sample_points = fibonacci_sphere(200)
        sphere_dots, plane_dots, proj_lines = self.get_dots_and_lines(sample_points)

        self.play(
            sphere.animate.set_clip_plane(UP, -1),
            frame.animate.reorient(-65, 73, 0, (-0.09, -0.01, -0.15), 5.08),
            ShowCreation(proj_lines, lag_ratio=0),
            FadeOut(planar_group),
            run_time=2,
        )
        self.wait(4)
        self.play(FadeIn(sphere_dots))
        self.wait(2)

        sphere_dot_ghosts = sphere_dots.copy().set_opacity(0.5)
        self.remove(sphere_dots)
        self.add(sphere_dot_ghosts)

        self.play(
            TransformFromCopy(sphere_dots, plane_dots, run_time=3),
        )
        self.wait(3)

        # Inverse projection
        plane.insert_n_curves(100)
        plane.save_state()
        proj_plane = plane.copy()
        proj_plane.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]))
        proj_plane.make_smooth()
        proj_plane.background_lines.set_stroke(BLUE, 2, 1)
        proj_plane.faded_lines.set_stroke(BLUE, 1, 0.5)

        self.play(
            Transform(plane_dots, sphere_dot_ghosts),
            FadeOut(sphere_dot_ghosts, scale=0.9),
            Transform(plane, proj_plane),
            proj_lines.animate.set_stroke(opacity=0.2),
            run_time=4,
        )
        self.play(
            frame.animate.reorient(-20, 38, 0, (-0.04, -0.03, 0.13), 3.54),
            run_time=5
        )
        self.wait(5)
        self.play(
            frame.animate.reorient(-27, 73, 0, (-0.03, 0.03, 0.04), 5.27),
            Restore(plane),
            FadeOut(plane_dots),
            run_time=5
        )
        self.wait(2)

        # Show a vector field
        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        xy_field.save_state()
        xy_field.set_stroke(width=1e-6)

        self.play(Restore(xy_field))
        self.wait(5)

        # Project the vector field up
        proj_field = xy_field.copy()
        proj_field.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proj_field.replace(sphere)
        proj_plane.background_lines.set_stroke(BLUE, 1, 0.5)
        proj_plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        proj_plane.axes.set_stroke(WHITE, 0)

        self.play(
            Transform(plane, proj_plane),
            Transform(xy_field, proj_field),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(-35, 31, 0, (0.05, 0.22, 0.22), 1.59),
            FadeOut(proj_lines),
            run_time=10
        )
        self.wait(8)

        # Show the flow (Maybe edit as a simple split-screen)
        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 20, y, 0]).insert_n_curves(25)
            for x in range(-100, 100, 10)
            for y in np.arange(-100, 100, 0.25)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)
        proto_stream_lines.apply_points_function(lambda p: inv_streographic_proj(p[:, :2]), about_point=ORIGIN)
        proto_stream_lines.scale(1.01)
        proto_stream_lines.make_smooth()
        animated_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        sphere.set_color(GREY_E, 1)
        sphere.set_clip_plane(UP, 1)
        sphere.set_height(1.98).center()
        xy_field.apply_depth_test()
        animated_lines.apply_depth_test()
        self.add(sphere, mesh, plane, animated_lines, xy_field)
        self.play(
            FadeIn(sphere),
            FadeOut(xy_field),
            plane.animate.fade(0.25),
            xy_field.animate.set_stroke(opacity=0.5),
            frame.animate.reorient(-30, 29, 0, ORIGIN, 3.0),
            run_time=3
        )
        self.wait(30)
        return

    def get_dots_and_lines(self, sample_points, color=YELLOW, radius=0.025, stroke_opacity=0.35):
        sphere_dots = Group(TrueDot(point) for point in sample_points)
        for dot in sphere_dots:
            dot.make_3d()
            dot.set_color(color)
            dot.set_radius(radius)

        plane_dots = sphere_dots.copy().apply_points_function(stereographic_proj)
        proj_lines = VGroup(
            VGroup(
                Line(OUT, dot.get_center())
                for dot in dots
            )
            for dots in [plane_dots, sphere_dots]
        )
        proj_lines.set_stroke(color, 1, stroke_opacity)

        return sphere_dots, plane_dots, proj_lines

    def flow_with_projection_insertion(self):
        # For an insertion
        frame.clear_updaters()
        frame.reorient(-18, 77, 0, (-0.04, 0.04, 0.09), 5.43)
        frame.clear_updaters()
        frame.add_ambient_rotation(1 * DEG)
        sphere.set_clip_plane(UP, 2)
        sphere.set_color(GREY_D, 0.5)
        xy_field.apply_depth_test()
        xy_field.set_stroke(opacity=0.)
        proj_lines.set_stroke(opacity=0)
        proj_plane.axes.set_stroke(width=1, opacity=0.5)
        self.clear()
        sphere.scale(0.99)
        self.add(axes, xy_field, proj_lines, sphere, plane, frame)

        # Particles
        n_samples = 50_000
        x_max = axes.x_range[1]
        sample_points = np.random.uniform(-x_max, x_max, (n_samples, 3))
        sample_points[:, 2] = 0
        particles = DotCloud(sample_points)
        particles.set_radius(0.015)
        particles.set_color(BLUE)
        particles.make_3d()

        particle_opacity_tracker = ValueTracker(1)
        proj_particle_radius_tracker = ValueTracker(0.01)

        proj_particles = particles.copy()
        proj_particles.set_opacity(1)

        x_vel = 0.5

        def update_particles(particles, dt):
            particles.shift(dt * x_vel * RIGHT)
            points = particles.get_points()
            points[points[:, 0] > x_max, 0] -= 2 * x_max
            particles.set_points(points)
            particles.set_opacity(particle_opacity_tracker.get_value())

            sphere_points = inv_streographic_proj(points[:, :2])
            zs = sphere_points[:, 2]
            proj_particles.set_points(sphere_points)
            proj_particles.set_radius((proj_particle_radius_tracker.get_value() * (1.5 - zs)).reshape(-1, 1))

        particles.add_updater(update_particles)
        self.add(particles, sphere)
        self.wait(7)

        # Project
        moving_particles = particles.copy()
        moving_particles_opacity_tracker = ValueTracker(0)
        moving_particles.add_updater(lambda m: m.set_opacity(moving_particles_opacity_tracker.get_value()))

        field = get_sphereical_vector_field(
            lambda p3d: stereographic_vector_field(p3d, right_func),
            axes,
            fibonacci_sphere(1000),
            stroke_width=0.5
        )
        field.set_stroke(WHITE, opacity=0.5)

        self.play(proj_lines.animate.set_opacity(0.4))
        self.play(
            particle_opacity_tracker.animate.set_value(0.15),
            Transform(moving_particles, proj_particles),
            moving_particles_opacity_tracker.animate.set_value(1),
            run_time=5
        )
        self.remove(moving_particles)
        self.add(sphere, particles, proj_particles)
        self.play(
            proj_lines.animate.set_opacity(0.1),
            frame.animate.reorient(-30, 29, 0, ORIGIN, 3.0),
            proj_particle_radius_tracker.animate.set_value(0.0075),
            FadeIn(field),
            run_time=3
        )
        self.wait(15)


    def old(self):
        earth = TexturedSurface(sphere, "EarthTextureMap")
        earth.set_opacity(1)

        earth_group = Group(earth)
        earth_group.save_state()
        proj_earth = earth_group.copy()
        proj_earth.apply_points_function(stereographic_proj)
        proj_earth.interpolate(proj_earth, earth_group, 0.01)

        self.remove(earth_group)
        self.play(TransformFromCopy(earth_group, proj_earth), run_time=3)
        self.wait()
        self.remove(proj_earth)
        self.play(TransformFromCopy(proj_earth, earth_group), run_time=3)


class SimpleRightwardFlow(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        x_max = 20
        axes = ThreeDAxes((-x_max, x_max), (-x_max, x_max), (-2, 2))
        plane = NumberPlane((-x_max, x_max), (-x_max, x_max))
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        axes.apply_depth_test()
        plane.apply_depth_test()

        # Simple flow
        frame.set_height(4)

        xy_field = VectorField(lambda ps: np.array([RIGHT for p in ps]), plane)
        xy_field.set_stroke(BLUE)
        self.add(xy_field)

        proto_stream_lines = VGroup(
            Line([x, y, 0], [x + 1, y, 0]).insert_n_curves(20)
            for x in np.arange(-10, 10, 0.5)
            for y in np.arange(-10, 10, 0.1)
        )
        for line in proto_stream_lines:
            line.virtual_time = 1
        proto_stream_lines.set_stroke(WHITE, 2, 0.8)

        animated_plane_lines = AnimatedStreamLines(proto_stream_lines, rate_multiple=0.2)

        self.add(animated_plane_lines)
        self.wait(30)


class SingleNullPointHairyBall(InteractiveScene):
    hide_top = True

    def construct(self):
        # Set up
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY_E, 1)
        sphere.set_shading(0.1, 0.1, 0.3)
        sphere.always_sort_to_camera(self.camera)
        axes = ThreeDAxes((-2, 2), (-2, 2), (-2, 2))
        axes.scale(radius)

        self.camera.light_source.move_to(3 * LEFT + 12 * UP + 3 * OUT)

        frame.reorient(-3, 161, 0)
        self.add(sphere)

        # Add vector field
        def v_func(points3d):
            new_points = stereographic_vector_field(points3d, right_func)
            norms = np.linalg.norm(new_points, axis=1)
            new_points *= 0.2 / norms[:, np.newaxis]
            return new_points

        def out_func(points3d):
            return 0.2 * points3d

        n_points = 50_000
        v_range = (-0.95, 1) if self.hide_top else (-1, 1)
        points = np.array([
            normalize(sphere.uv_func(TAU * random.random(), math.acos(pre_v)))
            for pre_v in np.random.uniform(*v_range, n_points)
        ])
        pre_field, field = fields = [
            get_sphereical_vector_field(
                func, axes, points,
                stroke_width=3,
                mvltss=3,
            )
            for func in [out_func, v_func]
        ]
        for lines in fields:
            lines.set_stroke(BLUE_E, opacity=0.75)
            lines.data['stroke_width'] = 0.5
            lines.note_changed_data()

        q_marks = Tex(R"???", font_size=72)
        q_marks.rotate(-90 * DEG)
        q_marks.move_to(sphere.get_zenith())
        disk = Circle(radius=1)
        disk = Sphere(radius=radius, v_range=(0.9 * PI, PI))
        disk.set_color(BLACK)
        disk.deactivate_depth_test()
        top_q = Group(disk, q_marks)
        top_q.set_z_index(1)

        if not self.hide_top:
            top_q.set_opacity(0)

        self.add(pre_field, sphere)
        self.play(
            ReplacementTransform(pre_field, field, run_time=2),
            frame.animate.reorient(-92, 11, 0).set_anim_args(run_time=7),
            FadeIn(top_q, time_span=(3.5, 5)),
        )
        self.play(
            frame.animate.reorient(-168, 127, 0),
            FadeOut(top_q, time_span=(4.2, 5)),
            run_time=10
        )


class SingleNullPointHairyBallRevealed(SingleNullPointHairyBall):
    hide_top = False


class PairsOfNullPoints(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 3
        sphere_scale = 0.99
        axis_range = (-4, 4)

        sphere = Sphere(radius=sphere_scale * radius)
        sphere.set_color(GREY_D, 0.5)
        sphere.always_sort_to_camera(self.camera)
        axes = ThreeDAxes(axis_range, axis_range, axis_range)
        axes.scale(radius)

        frame.reorient(-155, 74, 0)
        self.add(axes, sphere)

        # Vector field
        def source_and_sink(points2d, offset=-3):
            x, y = points2d.T
            return np.array([x - offset, y]).T

        def twirl_func(points2d, offset=-3):
            x, y = points2d.T
            return np.array([-y, x - offset]).T

        rotation = np.identity(3)[:, [0, 2, 1]]

        field = self.get_spherical_field(axes, twirl_func, rotation=rotation)
        offset_tracker = ValueTracker(-1)

        self.add(sphere, field)
        frame.reorient(-136, 77, 0, (0.91, 0.31, 0.79), 12.58)
        self.play(
            frame.animate.reorient(-177, 80, 0, (0.0, 0.0, 0.0), 8),
            run_time=4
        )
        frame.add_ambient_rotation(1 * DEG)

        # Change the field around
        new_params = [
            (rotation_matrix_transpose(90 * DEG, UP), -5),
            (rotation_matrix_transpose(30 * DEG, DOWN), -2),
            (rotation_matrix_transpose(120 * DEG, LEFT), 3),
            (rotation, -1),
        ]
        for new_rot, offset in new_params:
            new_field = self.get_spherical_field(
                axes, lambda ps: twirl_func(ps, offset), rotation=new_rot
            )
            self.play(Transform(field, new_field, run_time=2))

        self.play(
            offset_tracker.animate.set_value(-3),
            UpdateFromFunc(
                field,
                lambda m: m.become(
                    self.get_spherical_field(axes, lambda ps: twirl_func(ps, offset_tracker.get_value()), rotation=rotation)
                )
            ),
            run_time=4
        )

        # Show some flow
        streamlines = self.get_streamlines(field, axes, radius)
        self.add(streamlines)
        self.wait(8)

        # New field
        field2 = self.get_spherical_field(axes, source_and_sink)
        streamlines2 = self.get_streamlines(field2, axes, radius, density=100)

        self.play(FadeOut(streamlines))
        self.play(Transform(field, field2))
        self.play(FadeIn(streamlines2))
        self.wait(8)

    def get_streamlines(self, field, axes, radius, density=100):
        streamlines = SphereStreamLines(
            lambda p: field.func(np.array(p).reshape(-1, 3)).flatten(), axes,
            density=density,
            solution_time=1.0,
            dt=0.05,
        )
        streamlines.scale(radius, about_point=ORIGIN)
        streamlines.set_stroke(WHITE, 1, 0.5)
        animated_lines = AnimatedStreamLines(streamlines, rate_multiple=0.2)
        animated_lines.apply_depth_test()
        return animated_lines

    def get_spherical_field(
        self,
        axes,
        plane_func,
        n_sample_points=2000,
        stroke_width=2,
        rotation=np.identity(3),
    ):
        def v_func(points3d):
            rot_points = np.dot(points3d, rotation)
            new_points = stereographic_vector_field(rot_points, plane_func)
            norms = np.linalg.norm(new_points, axis=1)
            new_points *= 1.0 / norms[:, np.newaxis]
            new_points = np.dot(new_points, rotation.T)
            return new_points

        sample_points = fibonacci_sphere(n_sample_points)
        field = get_sphereical_vector_field(
            v_func, axes, sample_points,
            stroke_width=stroke_width
        )
        field.set_flat_stroke(False)

        return field


class AskAboutOutside(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        radius = 2
        axes = ThreeDAxes().scale(radius)
        axes.set_stroke(WHITE, 1, 0.5)
        axes.set_flat_stroke(False)
        axes.set_z_index(1)

        def sphere_uv(u, v):
            return [math.cos(u) * math.sin(v), math.sin(u) * math.sin(v), -math.cos(v)]

        sphere_group = self.get_warped_sphere_group(sphere_uv)
        sphere, mesh = sphere_group

        frame.reorient(-6, 70, 0)
        self.add(sphere, mesh, axes)

        # Show a character on the surface
        morty = Mortimer(mode="confused", height=0.1).flip()
        mirror_morty = morty.copy().flip(axis=RIGHT, about_edge=DOWN)
        mirror_morty.fade(0.5)
        out_arrow = Vector(0.15 * UP, thickness=0.5)
        out_arrow.next_to(morty, LEFT, buff=0.025, aligned_edge=DOWN)
        in_arrow = out_arrow.copy().flip(axis=RIGHT, about_edge=DOWN)
        out_arrow.set_color(BLUE)
        in_arrow.set_color(RED)
        morty_group = VGroup(morty, mirror_morty, out_arrow, in_arrow)

        morty_group.rotate(90 * DEG, RIGHT)
        morty_group.move_to(sphere.get_zenith())
        morty_group.rotate(30 * DEG, axis=UP, about_point=ORIGIN)

        sphere.set_clip_plane(UP, 2)

        self.play(
            FadeIn(morty, time_span=(0, 1)),
            sphere.animate.set_opacity(0.25),
            frame.animate.reorient(-5, 83, 0, (1.09, 0.51, 1.62), 1.15),
            run_time=3
        )
        self.play(GrowArrow(out_arrow))
        self.wait()
        self.play(
            TransformFromCopy(morty, mirror_morty),
            GrowArrow(in_arrow)
        )
        self.wait()

        # Show obvious outside and inside
        out_arrows, in_arrows = all_arrows = VGroup(
            VGroup(
                Arrow(radius * a1 * point, radius * a2 * point, fill_color=color, thickness=4, buff=0)
                for point in compass_directions(24)
            )
            for a1, a2, color in [(1.1, 1.5, BLUE), (0.9, 0.7, RED)]
        )
        all_arrows.rotate(90 * DEG, RIGHT, about_point=ORIGIN)

        self.play(
            frame.animate.reorient(-4, 82, 0, (0.15, -0.01, 0.02), 8.08),
            LaggedStartMap(GrowArrow, out_arrows, lag_ratio=0.01, time_span=(3, 6)),
            run_time=6
        )
        self.wait()
        self.play(
            FadeOut(out_arrows),
            FadeOut(morty_group),
            sphere.animate.set_opacity(1)
        )

        # Warp the sphere
        def twist_sphere_uv(u, v):
            return [math.cos(u + v - 0.5) * math.sin(v), math.sin(u + v - 0.5) * math.sin(v), -math.cos(v)]

        def squish_sphere_uv(u, v):
            x, y, z = twist_sphere_uv(u, v)
            dist = math.sqrt(x**2 + y**2)
            z *= -35 * (dist - 0.25) * (dist - 0.75) * (dist - 0) * (dist - 1.1)
            return (x, y, z)

        twisted_sphere = self.get_warped_sphere_group(twist_sphere_uv)
        squish_sphere = self.get_warped_sphere_group(squish_sphere_uv)

        self.play(
            Transform(sphere_group, twisted_sphere),
            frame.animate.reorient(-1, 73, 0, (-0.23, 0.06, -0.09), 6.00),
            run_time=5
        )
        self.play(
            frame.animate.reorient(12, 73, 0, (-0.09, 0.1, -0.06), 4.71),
            Transform(sphere_group, squish_sphere),
            run_time=5
        )

        # Example point
        index = 9200
        point = sphere.get_points()[index]
        normal = normalize(sphere.data["d_normal_point"][index] - point)
        dot = TrueDot(point, color=YELLOW)
        dot.make_3d()
        dot.deactivate_depth_test()
        dot.set_radius(0.01)
        dot.set_z_index(1)

        in_vect, out_vect = vects = VGroup(
            Vector(sign * 0.2 * normal, thickness=0.5, fill_color=color).shift(point)
            for sign, color in zip([1, -1], [RED, BLUE])
        )
        for vect in vects:
            vect.set_perpendicular_to_camera(frame)

        self.add(dot)
        self.play(
            FadeIn(dot),
            frame.animate.reorient(-47, 81, 0, (-0.55, 0.17, 0.13), 0.65),
            run_time=3
        )
        self.play(GrowArrow(in_vect))
        self.play(GrowArrow(out_vect))
        self.wait()

        # Show a homotopy
        def homotopy(x, y, z, t):
            alpha = clip((x + 3) / 6 - 1 + 2 * t, 0, 1)
            shift = wiggle(alpha, 3)
            return (x, y, z + 0.35 * shift)

        self.play(
            FadeOut(vects, time_span=(0, 1)),
            FadeOut(dot, time_span=(0, 1)),
            Homotopy(homotopy, sphere_group),
            frame.animate.reorient(-44, 78, 0, (-0.16, 0.02, 0.31), 4.18),
            run_time=6,
        )


    def get_warped_sphere_group(self, uv_func, radius=2, mesh_resolution=(61, 31), u_range=(0, TAU), v_range=(0, PI)):
        surface = ParametricSurface(uv_func, u_range=u_range, v_range=v_range, resolution=(201, 101))
        surface.always_sort_to_camera(self.camera)
        surface.set_color(GREY_D)
        mesh = SurfaceMesh(surface, resolution=mesh_resolution, normal_nudge=0)
        mesh.set_stroke(WHITE, 0.5, 0.25)
        mesh.deactivate_depth_test()
        result = Group(surface, mesh)
        result.scale(radius, about_point=ORIGIN)
        return result


class InsideOut(InteractiveScene):
    def construct(self):
        # Show sphere
        frame = self.frame
        self.camera.light_source.move_to([-3, 3, 3])
        radius = 3
        inner_scale = 0.999
        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))
        axes.set_stroke(WHITE, 1, 0.5)
        axes.apply_depth_test()
        axes.z_axis.rotate(0.1 * DEG, RIGHT)

        sphere = self.get_colored_sphere(radius, inner_scale)
        sphere.set_clip_plane(UP, radius)

        mesh = SurfaceMesh(sphere[0], resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)

        frame.reorient(-68, 70, 0)
        self.add(axes, sphere, mesh)
        self.play(sphere.animate.set_clip_plane(UP, 0), run_time=2)

        # Show point go to antipoode
        point = radius * normalize(LEFT + OUT)
        p_dot = TrueDot(point, color=YELLOW, radius=0.05).make_3d()
        p_label = Tex(R"p")
        p_label.rotate(90 * DEG, RIGHT).rotate(90 * DEG, IN)
        p_label.next_to(p_dot, OUT, SMALL_BUFF)

        neg_p_dot = p_dot.copy().move_to(-point)
        neg_p_label = Tex(R"-p")
        neg_p_label.rotate(90 * DEG, RIGHT)
        neg_p_label.next_to(neg_p_dot, IN, SMALL_BUFF)

        neg_p_dot.move_to(p_dot)

        dashed_line = DashedLine(point, -point, buff=0)
        dashed_line.set_stroke(YELLOW, 2)

        semi_circle = Arc(135 * DEG, 180 * DEG, radius=radius)
        semi_circle.set_stroke(YELLOW, 3)
        semi_circle.rotate(90 * DEG, RIGHT, about_point=ORIGIN)
        dashed_semi_circle = DashedVMobject(semi_circle, num_dashes=len(dashed_line))

        self.play(
            FadeIn(p_dot, scale=0.5),
            Write(p_label),
        )
        self.play(
            ShowCreation(dashed_semi_circle),
            MoveAlongPath(neg_p_dot, semi_circle, rate_func=linear),
            TransformFromCopy(p_label, neg_p_label, time_span=(2, 3)),
            Rotate(p_label, 90 * DEG, OUT),
            frame.animate.reorient(59, 87, 0),
            run_time=3,
        )
        frame.add_ambient_rotation(-2 * DEG)
        self.wait(2)
        self.play(ReplacementTransform(dashed_semi_circle, dashed_line))
        self.wait(3)

        # Show more antipodes
        angles = np.linspace(0, 90 * DEG, 15)[1:]
        top_dots, low_dots, lines = groups = [
            Group(
                template.copy().rotate(angle, axis=UP, about_point=ORIGIN)
                for angle in angles
            )
            for template in [p_dot, neg_p_dot, dashed_line]
        ]
        for group in groups:
            group.set_submobject_colors_by_gradient(YELLOW, BLUE, interp_by_hsl=True)

        self.play(FadeIn(top_dots, lag_ratio=0.1))
        frame.clear_updaters()
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(top_dot, low_dot, rate_func=linear)
                for top_dot, low_dot in zip(top_dots, low_dots)),
                lag_ratio=0.25,
                group_type=Group,
            ),
            frame.animate.reorient(-49, 60, 0),
            run_time=7
        )

        # Just show the cap
        cap = self.get_colored_sphere(radius, inner_scale, v_range=(0.75 * PI, PI))
        frame.clear_updaters()

        self.add(cap, sphere, mesh)
        self.play(
            FadeOut(p_label),
            FadeOut(neg_p_label),
            FadeOut(top_dots),
            FadeOut(lines),
            FadeOut(low_dots),
            FadeOut(p_dot),
            FadeOut(neg_p_dot),
            FadeOut(dashed_line),
            FadeOut(sphere, 0.1 * UP),
            FadeIn(cap),
        )
        self.wait()
        self.play(frame.animate.reorient(-48, 93, 0), run_time=3)
        self.wait()

        # Add normal vectors
        uv_samples = np.array([
            [(u + v) % TAU, v]
            for v in np.linspace(0.75 * PI, 0.95 * PI, 10)
            for u in np.linspace(0, TAU, 20)
        ])
        normal_vectors = VGroup(
            VMobject().set_points_as_corners([ORIGIN, RIGHT, RIGHT, 2 * RIGHT])
            for sample in uv_samples
        )
        for vect in normal_vectors:
            vect.set_stroke(BLUE_B, width=[1, 1, 1, 6, 3, 0], opacity=0.5)
        normal_vectors.apply_depth_test()

        def update_normal_vectors(normal_vectors):
            points = np.array([cap[0].uv_to_point(u, v) for u, v in uv_samples])
            du_points = np.array([cap[0].uv_to_point(u + 0.1, v) for u, v in uv_samples])
            dv_points = np.array([cap[0].uv_to_point(u, v + 0.1) for u, v in uv_samples])
            normals = normalize_along_axis(np.cross(du_points - points, dv_points - points), 1)
            for point, normal, vector in zip(points, normals, normal_vectors):
                vector.put_start_and_end_on(point, point + 0.3 * normal)

        update_normal_vectors(normal_vectors)
        self.play(FadeIn(normal_vectors))

        # Show transition to antipode
        anti_cap = cap.copy().rotate(PI, axis=OUT, about_point=ORIGIN).stretch(-1, 2, about_point=ORIGIN)
        anti_cap[0].shift(1e-2 * OUT)
        all_points = cap[0].get_points()
        indices = random.sample(list(range(len(all_points))), 200)
        pre_points = all_points[indices]
        post_points = -1 * pre_points

        antipode_lines = VGroup(
            Line(point, -point)
            for point in pre_points
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)
        antipode_lines.apply_depth_test()

        def update_lines(lines):
            points1 = cap[0].get_points()[indices]
            points2 = anti_cap[0].get_points()[indices]
            for line, p1, p2 in zip(lines, points1, points2):
                line.put_start_and_end_on(p1, p2)

        antipode_lines.add_updater(update_lines)

        rot_arcs = VGroup(
            Arrow(4 * RIGHT, 4 * LEFT, path_arc=180 * DEG, thickness=5),
            Arrow(4 * LEFT, 4 * RIGHT, path_arc=180 * DEG, thickness=5),
        )
        flip_arrows = VGroup(
            Arrow(3 * IN, 3.2 * OUT, thickness=5),
            Arrow(3 * OUT, 3.2 * IN, thickness=5),
        ).rotate(90 * DEG).shift(4 * RIGHT)

        self.play(
            ShowCreation(antipode_lines, lag_ratio=0, suspend_mobject_updating=True),
            frame.animate.reorient(8, 79, 0, (0.0, 0.02, 0.0)),
            run_time=4,
        )
        self.wait()
        normal_vectors.add_updater(update_normal_vectors)
        self.play(
            Write(rot_arcs, lag_ratio=0, run_time=1),
            Rotate(cap, PI, axis=OUT, run_time=3, about_point=ORIGIN),
            Rotate(mesh, PI, axis=OUT, run_time=3, about_point=ORIGIN),
        )
        self.play(FadeOut(rot_arcs))
        self.play(
            FadeIn(flip_arrows, time_span=(0, 1)),
            Transform(cap, anti_cap),
            mesh.animate.stretch(-1, 2, about_point=ORIGIN),
            VFadeOut(antipode_lines),
            run_time=3
        )
        self.play(FadeOut(flip_arrows))
        self.play(frame.animate.reorient(-9, 96, 0, (0.0, 0.02, 0.0)), run_time=5)
        self.wait()

        # Antipode homotopy

    def get_colored_sphere(
        self,
        radius=3,
        inner_scale=0.999,
        outer_color=BLUE_E,
        inner_color=GREY_BROWN,
        u_range=(0, TAU),
        v_range=(0, PI),
    ):
        outer_sphere = Sphere(radius=radius, u_range=u_range, v_range=v_range)
        inner_sphere = outer_sphere.copy()
        outer_sphere.set_color(outer_color, 1)
        inner_sphere.set_color(inner_color, 1)
        inner_sphere.scale(inner_scale)
        return Group(outer_sphere, inner_sphere)

    def old_homotopy(self):
        def homotopy(x, y, z, t, scale=-1):
            p = np.array([x, y, z])
            power = 1 + 0.2 * (x / radius)
            return interpolate(p, scale * p, t**power)

        antipode_lines = VGroup(
            Line(point, -point)
            for point in random.sample(list(cap[0].get_points()), 100)
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.25)

        self.play(
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -inner_scale), cap[0]),
            Homotopy(lambda x, y, z, t: homotopy(x, y, z, t, -1.0 / inner_scale), cap[1]),
            ShowCreation(antipode_lines, lag_ratio=0),
            frame.animate.reorient(-65, 68, 0),
            run_time=3
        )
        self.play(FadeOut(antipode_lines))
        self.play(frame.animate.reorient(-48, 148, 0), run_time=3)
        self.wait()
        self.play(frame.animate.reorient(-70, 83, 0), run_time=3)
        self.play(frame.animate.reorient(-124, 77, 0), run_time=10)


class UnitNormals(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        surface = Torus()
        surface.set_color(GREY_D)

        uv_samples = np.array([
            [u, v]
            for v in np.linspace(0, TAU, 25)
            for u in np.linspace(0, TAU, 50)
        ])
        points = np.array([surface.uv_to_point(u, v) for u, v in uv_samples])
        uv_samples = uv_samples[np.argsort(points[:, 0])]

        normal_vectors = VGroup(
            VMobject().set_points_as_corners([ORIGIN, RIGHT, RIGHT, 2 * RIGHT])
            for sample in uv_samples
        )
        for vect in normal_vectors:
            vect.set_stroke(BLUE_D, width=[2, 2, 2, 12, 6, 0], opacity=0.5)
        normal_vectors.set_stroke(WHITE)
        normal_vectors.apply_depth_test()
        normal_vectors.set_flat_stroke(False)

        def update_normal_vectors(normal_vectors):
            points = np.array([surface.uv_to_point(u, v) for u, v in uv_samples])
            du_points = np.array([surface.uv_to_point(u + 0.1, v) for u, v in uv_samples])
            dv_points = np.array([surface.uv_to_point(u, v + 0.1) for u, v in uv_samples])
            normals = normalize_along_axis(np.cross(du_points - points, dv_points - points), 1)
            for point, normal, vector in zip(points, normals, normal_vectors):
                vector.put_start_and_end_on(point, point + 0.5 * normal)

        update_normal_vectors(normal_vectors)

        frame.reorient(51, 57, 0, (0.39, 0.01, -0.5), 8.00)
        frame.add_ambient_rotation(5 * DEG)
        self.add(surface)
        self.add(normal_vectors)
        self.play(ShowCreation(normal_vectors, run_time=3))
        self.wait(6)


class DefineOrientation(InsideOut):
    def construct(self):
        # Latitude and Longitude
        frame = self.frame
        radius = 3
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY_E, 1)
        earth = TexturedSurface(sphere, "EarthTextureMap", "NightEarthTextureMap")
        earth.set_opacity(0.5)
        mesh = SurfaceMesh(sphere, resolution=(73, 37))
        mesh.set_stroke(WHITE, 1, 0.25)

        uv_tracker = ValueTracker(np.array([180 * DEG, 90 * DEG]))

        dot = TrueDot()
        dot.set_color(YELLOW)
        dot.add_updater(lambda m: m.move_to(sphere.uv_func(*uv_tracker.get_value())))
        dot.set_z_index(2)

        lat_label, lon_label = lat_lon_labels = VGroup(
            Tex(R"\\text{Lat: }\\, 10^\\circ"),
            Tex(R"\\text{Lon: }\\, 10^\\circ"),
        )
        lat_lon_labels.arrange(DOWN, aligned_edge=LEFT)
        lat_lon_labels.fix_in_frame()
        lat_lon_labels.to_corner(UL)
        lat_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round(self.get_lat_lon(*uv_tracker.get_value())[0]))
        )
        lon_label.make_number_changeable("10", edge_to_fix=RIGHT).add_updater(
            lambda m: m.set_value(np.round(self.get_lat_lon(*uv_tracker.get_value())[1]))
        )
        lat_lon_labels.add_updater(lambda m: m.fix_in_frame())

        self.add(sphere, mesh)
        self.add(lat_lon_labels)
        frame.reorient(-66, 85, 0, (-0.06, 0.18, 0.06), 6.78)
        self.play(FadeIn(dot))

        lon_line = TracedPath(dot.get_center, stroke_color=RED)
        self.add(lon_line, dot)
        self.play(uv_tracker.animate.increment_value([0, 30 * DEG]), run_time=4)
        lon_line.suspend_updating()

        lat_line = TracedPath(dot.get_center, stroke_color=TEAL)
        self.add(lat_line, dot)
        self.play(uv_tracker.animate.increment_value([45 * DEG, 0]), run_time=4)
        lat_line.suspend_updating()

        # Add labels to all the points
        u, v = uv_tracker.get_value()

        label_template = Tex(R"(10^\\circ, 10^\\circ)", isolate=["10"])
        label_template.set_backstroke(BLACK, 1)
        lon_num_template, lat_num_template = label_template.make_number_changeable("10", replace_all=True)

        def get_lat_lon_label(u, v, font_size=5):
            lat, lon = self.get_lat_lon(u, v)
            lat_num_template.set_value(np.round(lat))
            lon_num_template.set_value(np.round(lon))
            label = label_template.copy()
            label.scale(font_size / 48)
            label.move_to(sphere.get_zenith())
            label.rotate(PI - v, axis=RIGHT, about_point=ORIGIN)
            label.rotate(-270 * DEG + u, axis=OUT, about_point=ORIGIN)
            return label

        u_radius = 50 * DEG
        v_radius = 30 * DEG
        all_labels = VGroup(
            get_lat_lon_label(sub_u, sub_v)
            for sub_u in np.arange(u - u_radius, u + u_radius, 10 * DEG)
            for sub_v in np.arange(v - v_radius, v + v_radius, 5 * DEG)
        )
        all_labels.sort(lambda p: get_norm(p - dot.get_center()))

        self.play(
            frame.animate.reorient(-39, 60, 0, (-0.64, 0.45, 0.06), 4.33),
            run_time=3
        )
        self.play(
            FadeIn(all_labels, lag_ratio=0.001, run_time=3),
            dot.animate.set_opacity(0.25),
        )
        self.wait()

        # Show some kind of warping
        def homotopy(x, y, z, t):
            alpha = clip((x + 3) / 6 - 1 + 2 * t, 0, 1)
            shift = wiggle(alpha, 3)
            return (x, y, z + 0.35 * shift)

        dot.suspend_updating()
        group = Group(sphere, mesh, lat_line, lon_line, dot, all_labels)

        self.play(
            Homotopy(homotopy, group, run_time=10)
        )
        dot.resume_updating()
        self.wait()

        # Show tangent vectors
        u, v = uv_tracker.get_value()
        epsilon = 1e-4
        point = sphere.uv_func(u, v)
        u_step = normalize(sphere.uv_func(u + epsilon, v) - point)
        v_step = normalize(sphere.uv_func(u, v + epsilon) - point)

        u_vect = Arrow(point, point + 0.5 * u_step, buff=0, thickness=2).set_color(TEAL)
        v_vect = Arrow(point, point + 0.5 * v_step, buff=0, thickness=2).set_color(RED)
        tangent_vects = VGroup(u_vect, v_vect)
        tangent_vects.set_z_index(1)
        tangent_vects.set_fill(opacity=0.8)
        for vect in tangent_vects:
            vect.set_perpendicular_to_camera(frame)

        self.play(
            dot.animate.set_opacity(1).scale(0.5),
            all_labels.animate.set_stroke(width=0).set_fill(opacity=0.25),
        )
        self.wait()

        lat_line2 = TracedPath(dot.get_center, stroke_color=TEAL, stroke_width=1)
        lat_line2.set_scale_stroke_with_zoom(True)
        self.add(lat_line2)
        self.play(
            uv_tracker.animate.increment_value([30 * DEG, 0]).set_anim_args(rate_func=wiggle),
            FadeOut(lon_line),
            run_time=3
        )
        lat_line2.clear_updaters()
        self.play(GrowArrow(tangent_vects[0]))
        self.wait()

        lon_line2 = TracedPath(dot.get_center, stroke_color=RED, stroke_width=1)
        lon_line2.set_scale_stroke_with_zoom(True)
        self.add(lon_line2)
        self.play(
            tangent_vects[0].animate.set_fill(opacity=0.5).set_anim_args(time_span=(0, 1)),
            uv_tracker.animate.increment_value([0, 30 * DEG]).set_anim_args(rate_func=wiggle),
            FadeOut(lat_line),
            run_time=3
        )
        lon_line2.clear_updaters()
        self.play(GrowArrow(tangent_vects[1]))
        self.play(tangent_vects.animate.set_fill(opacity=1))
        self.wait()

        # Show normal vector
        normal_vect = Arrow(
            point, point + 0.5 * np.cross(u_step, v_step),
            thickness=2,
            buff=0
        )
        normal_vect.set_fill(BLUE, 0.8)
        normal_vect.rotate(90 * DEG, axis=normal_vect.get_vector())

        self.play(
            GrowArrow(normal_vect, time_span=(2, 4)),
            FadeOut(lat_lon_labels, time_span=(0, 2)),
            frame.animate.reorient(-96, 54, 0, (-0.23, -1.08, 0.25), 3.67),
            run_time=8
        )

        # Show a full vector field
        normal_field = get_sphereical_vector_field(
            lambda p: p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        normal_field.save_state()
        normal_field.set_stroke(width=1e-6)

        self.play(
            Restore(normal_field, time_span=(2, 5)),
            frame.animate.reorient(-103, 62, 0, (-0.09, 0.25, 0.23), 7.26),
            run_time=10
        )
        self.wait()
        self.add(self.camera.light_source)

        # Move around light
        light = GlowDot(radius=0.5, color=WHITE)
        light.move_to(self.camera.light_source)
        light.save_state()
        self.camera.light_source.always.move_to(light)

        self.add(self.camera.light_source)
        self.play(
            light.animate.move_to(4 * normalize(light.get_center())),
            run_time=3
        )
        self.play(Rotate(light, TAU, axis=OUT, about_point=ORIGIN, run_time=6))
        self.play(
            FadeOut(normal_field),
            FadeOut(normal_vect),
            Restore(light),
            all_labels.animate.set_fill(opacity=0.5),
            frame.animate.reorient(-63, 66, 0, (-0.09, 0.25, 0.23), 7.26),
            run_time=3
        )

        # Warp sphere
        dot.suspend_updating()
        group = Group(sphere, mesh, lat_line2, lon_line2, all_labels, tangent_vects, dot)
        group.save_state()
        group.target = group.generate_target()
        group.target.rotate(-45 * DEG)
        group.target.scale(0.5)
        group.target.stretch(0.25, 0)
        group.target.shift(RIGHT)
        group.target.apply_complex_function(lambda z: z**3)
        group.target.center()
        group.target.set_height(6)
        group.target.rotate(45 * DEG)

        self.play(Homotopy(homotopy, group, run_time=3))
        self.play(
            MoveToTarget(group),
            frame.animate.reorient(-94, 60, 0, (0.5, 0.84, 0.91), 1.06),
            run_time=8
        )
        self.wait()
        self.play(MoveAlongPath(dot, lat_line2, run_time=5))
        self.play(MoveAlongPath(dot, lon_line2, run_time=5))

        new_normal_vector = Vector(
            0.35 * normalize(np.cross(tangent_vects[0].get_vector(), tangent_vects[1].get_vector())),
            fill_color=BLUE,
            thickness=1
        )
        new_normal_vector.shift(tangent_vects[0].get_start())
        new_normal_vector.set_perpendicular_to_camera(self.frame)
        self.play(
            GrowArrow(new_normal_vector),
            frame.animate.reorient(-125, 52, 0, (0.5, 0.84, 0.91), 1.06),
            run_time=2
        )
        self.wait()
        self.play(
            FadeOut(new_normal_vector, time_span=(0, 1)),
            Restore(group),
            frame.animate.reorient(-78, 66, 0, (-0.13, 0.12, 0.13), 6.54),
            run_time=5
        )

        # Show antipode map
        group = Group(lat_line2, lon_line2, tangent_vects, dot)
        anti_group = group.copy().scale(-1, min_scale_factor=-np.inf, about_point=ORIGIN)

        antipode_lines = VGroup(
            Line(p1, p2)
            for index in [0, 1]
            for p1, p2 in zip(group[index].get_points(), anti_group[index].get_points())
        )
        antipode_lines.set_stroke(YELLOW, 1, 0.1)

        self.play(
            FadeOut(sphere, scale=0.9),
            FadeOut(all_labels),
            VGroup(lat_line2, lon_line2).animate.set_stroke(width=1),
            FadeIn(normal_vect),
            frame.animate.reorient(-93, 55, 0, (-0.26, -0.85, 0.24), 4.27),
            run_time=2
        )

        self.play(
            TransformFromCopy(group, anti_group),
            frame.animate.reorient(-194, 105, 0, (0.7, -0.22, -0.61), 4.64),
            ShowCreation(antipode_lines, lag_ratio=0),
            run_time=5
        )
        self.play(antipode_lines.animate.set_stroke(opacity=0.02))
        self.wait()

        # Map over labels
        all_labels.set_fill(opacity=0.5)
        anti_labels = all_labels.copy().scale(-1, min_scale_factor=-np.inf, about_point=ORIGIN)

        for label in anti_labels:
            label.rotate(PI, axis=np.cross(label.get_center(), IN))

        self.play(FadeIn(all_labels, lag_ratio=0.001, run_time=3))
        self.remove(all_labels)
        self.play(TransformFromCopy(all_labels, anti_labels, lag_ratio=0.0002, run_time=3))
        self.wait()
        self.play(
            MoveAlongPath(dot, anti_group[0], run_time=5),
            anti_group[1].animate.set_stroke(opacity=0.25),
            anti_group[2][1].animate.set_fill(opacity=0.25),
        )
        self.play(
            MoveAlongPath(dot, anti_group[1], run_time=5),
            anti_group[1].animate.set_stroke(opacity=1),
            anti_group[2][1].animate.set_fill(opacity=1),
            anti_group[0].animate.set_stroke(opacity=0.25),
            anti_group[2][0].animate.set_fill(opacity=0.25),
        )
        self.wait()
        self.play(
            anti_group[0].animate.set_stroke(opacity=1),
            anti_group[2][0].animate.set_fill(opacity=1),
        )

        # New normal
        new_normal = normal_vect.copy()
        new_normal.shift(-2 * point)

        self.play(
            GrowArrow(new_normal),
            frame.animate.reorient(-197, 90, 0, (1.16, -0.3, -0.99), 4.64),
            run_time=3
        )
        self.wait()

        # Show reverserd vector field
        anti_normal_field = get_sphereical_vector_field(
            lambda p: -p,
            ThreeDAxes(),
            points=np.array([
                sphere.uv_func(u, v)
                for u in np.arange(0, TAU, TAU / 60)
                for v in np.arange(0, PI, PI / 30)
            ])
        )

        frame.reorient(162, 89, 0, (1.16, -0.3, -0.99), 4.64)
        self.play(
            FadeOut(group),
            FadeOut(antipode_lines),
            FadeIn(anti_normal_field, time_span=(0, 2)),
            frame.animate.reorient(360 - 177, 82, 0, (-0.03, -0.11, 0.86), 9.51),
            run_time=10
        )

        # Show outward vectors again
        self.remove(anti_normal_field)
        self.remove(anti_group)
        self.remove(new_normal)
        self.add(group)
        self.add(normal_field)
        antipode_lines.set_stroke(YELLOW, 1, 0.1)
        self.play(ShowCreation(antipode_lines, lag_ratio=0, run_time=2))
        self.wait()

    def get_lat_lon(self, u, v):
        return np.array([
            v / DEG - 90,
            u / DEG - 180,
        ])


class FlowingWater(InteractiveScene):
    def construct(self):
        # Set up axes
        radius = 2
        frame = self.frame
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.scale(radius)
        frame.reorient(-89, 77, 0)
        frame.add_ambient_rotation(2 * DEG)
        self.add(axes)

        # Add water
        water = self.get_water(sigma0=0.2, n_droplets=1_000_000, opacity=0.1, refresh_ratio=0.015)
        water.scale(0.01, about_point=ORIGIN)
        source_dot = GlowDot(ORIGIN, color=BLUE)

        self.add(source_dot, water)
        water.refresh_sigma_tracker.set_value(0.2)
        water.opacity_tracker.set_value(0.05)
        water.radius_tracker.set_value(0.015)
        frame.reorient(-108, 73, 0, (0.01, 0.06, -0.03), 4.0),
        self.play(
            water.radius_tracker.animate.set_value(0.02),
            water.opacity_tracker.animate.set_value(0.1),
            water.refresh_sigma_tracker.animate.set_value(2.5),
            frame.animate.reorient(-120, 80, 0, ORIGIN, 8.00),
            run_time=10
        )
        self.wait(20)

        # Show full sphere
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY, 0.25)
        sphere.set_shading(0.5, 0.5, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.5)
        mesh.set_z_index(2)

        def get_unit_normal_field(u_range, v_range):
            return get_sphereical_vector_field(
                lambda p: p,
                ThreeDAxes(),
                points=np.array([
                    sphere.uv_func(u, v)
                    for u in u_range
                    for v in v_range
                ])
            )

        normal_field = get_unit_normal_field(
            np.arange(0, TAU, TAU / 60),
            np.arange(0, PI, PI / 30),
        )

        self.play(
            ShowCreation(sphere),
            Write(mesh, lag_ratio=1e-3),
            run_time=2
        )
        self.wait(2)
        self.play(FadeIn(normal_field))
        self.wait(7.2)

        # Show single patch
        u_range_params = (0 * DEG, 15 * DEG, 5 * DEG)
        v_range_params = (120 * DEG, 130 * DEG, 5 * DEG)
        patch = Sphere(
            radius=radius,
            u_range=u_range_params[:2],
            v_range=v_range_params[:2],
        )
        patch.set_color(WHITE, 0.6)

        patch_normals = get_unit_normal_field(np.arange(*u_range_params), np.arange(*v_range_params))

        self.play(
            FadeOut(sphere, time_span=(0, 1)),
            FadeOut(normal_field, time_span=(0, 1)),
            FadeIn(patch, time_span=(0, 1)),
            FadeIn(patch_normals, time_span=(0, 1)),
            mesh.animate.set_stroke(opacity=0.1),
            frame.animate.reorient(19, 55, 0, (1.65, 0.28, 0.98), 2.32),
            water.opacity_tracker.animate.set_value(0.125),
            water.radius_tracker.animate.set_value(0.01),
            run_time=5
        )
        frame.clear_updaters()
        self.wait(8)

        patch_group = Group(patch, patch_normals)

        # Original patch behavior
        self.play(
            Rotate(patch_group, PI, axis=UP),
            run_time=2
        )
        self.wait(2.5)
        self.play(
            Rotate(patch_group, PI, axis=UP, time_span=(0, 1)),
            FadeIn(sphere),
            FadeIn(normal_field),
            frame.animate.reorient(30, 78, 0, (0.19, -0.03, 0.09), 5.67),
            water.opacity_tracker.animate.set_value(0.15),
            water.radius_tracker.animate.set_value(0.015),
            run_time=5
        )
        self.play(FadeOut(patch_group))
        frame.add_ambient_rotation(-1 * DEG)
        self.wait(3)
        self.play(
            FadeOut(normal_field),
            sphere.animate.set_shading(1, 1, 1),
        )
        self.wait(5)

        # Show deformations
        sphere_group = Group(sphere, mesh)

        def alt_uv_func(u, v, params, wiggle_size=1.0, max_freq=4):
            x, y, z = sphere.uv_func(u, v)
            return (
                x + wiggle_size * params[0] * np.cos(max_freq * params[1] * y),
                y + wiggle_size * params[4] * np.cos(max_freq * params[5] * z),
                z + wiggle_size * params[2] * np.cos(max_freq * params[3] * x),
            )

        np.random.seed(3)
        for n in range(20):
            params = np.random.random(6)
            new_sphere = ParametricSurface(
                lambda u, v: alt_uv_func(u, v, params),
                u_range=sphere.u_range,
                v_range=sphere.v_range,
                resolution=sphere.resolution
            )
            new_sphere.match_style(sphere)
            new_mesh = SurfaceMesh(new_sphere, resolution=mesh.resolution)
            new_mesh.match_style(mesh)
            new_group = Group(new_sphere, new_mesh)

            if 10 < n < 13:
                new_group.shift(1.75 * radius * RIGHT)

            self.play(Transform(sphere_group, new_group, run_time=2))
            self.wait(2)

    def get_water(
        self,
        n_droplets=500_000,
        radius=0.02,
        opacity=0.2,
        sigma0=3,
        refresh_sigma=2.5,
        velocity=10,
        refresh_ratio=0.01,
    ):
        points = np.random.normal(0, sigma0, (n_droplets, 3))
        water = DotCloud(points)
        water.set_radius(radius)
        water.set_color(BLUE)
        water.opacity_tracker = ValueTracker(opacity)
        water.radius_tracker = ValueTracker(radius)
        water.refresh_sigma_tracker = ValueTracker(refresh_sigma)
        water.velocity = velocity

        def flow_out(water, dt):
            if dt == 0:
                pass
            points = water.get_points()
            radii = np.linalg.norm(points, axis=1)
            denom = 4 * PI * radii**2
            denom[denom == 0] = 1
            vels = points / denom[:, np.newaxis]
            new_points = points + water.velocity * vels * dt

            n_refreshes = int(refresh_ratio * len(points))
            indices = np.random.randint(0, len(points), n_refreshes)
            new_points[indices] = np.random.normal(0, water.refresh_sigma_tracker.get_value(), (n_refreshes, 3))
            water.set_points(new_points)
            water.set_opacity(water.opacity_tracker.get_value() / np.clip(radii, 1, np.inf))
            water.set_radius(water.radius_tracker.get_value())
            return water

        water.add_updater(flow_out)

        return water

    def rotate_patch(self):
        # to be inserted in "show single patch" above
        patch_group.clear_updaters()
        point = patch.get_center()
        path = Arc(0, PI)
        path.rotate(20 * DEG, RIGHT)
        path.put_start_and_end_on(point, -point)

        patch_group.move_to(path.get_start())
        self.wait(5)
        self.play(
            MoveAlongPath(patch_group, path),
            frame.animate.reorient(-25, 92, 0, (-0.75, -0.96, -0.16), 3.16),
            run_time=5
        )
        self.wait(5)


class SurfaceFoldedOverSelf(FlowingWater):
    def construct(self):
        # Set up axes
        radius = 2
        frame = self.frame
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        axes.scale(radius)
        frame.reorient(-20, 77, 0)
        self.add(axes)

        water = self.get_water()
        self.add(water)

        # Deform sphere
        sphere = Sphere(radius=radius)
        sphere.set_color(GREY, 0.25)
        sphere.set_shading(0.5, 0.5, 0.5)
        sphere.always_sort_to_camera(self.camera)
        mesh = SurfaceMesh(sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.25)
        mesh.set_z_index(2)

        sphere_group = Group(sphere, mesh)
        self.add(sphere_group)
        self.wait(10)

        # Morph
        def inflate(points):
            radii = np.linalg.norm(points, axis=1)
            new_radii = 2.0 * (1 - 1 / (radii + 1))
            return points * (new_radii / radii)[:, np.newaxis]

        sphere_target = sphere_group.copy()
        sphere_target.rotate(90 * DEG, LEFT)
        sphere_target.stretch(0.2, 0)
        sphere_target.stretch(0.5, 1)
        sphere_target.move_to(RIGHT)
        sphere_target.apply_complex_function(lambda z: z**3)
        sphere_target.rotate(90 * DEG, RIGHT)
        sphere_target.shift(radius * OUT)
        sphere_target.apply_points_function(inflate, about_point=ORIGIN)
        sphere_target.replace(sphere)

        sphere_group.save_state()
        self.play(
            frame.animate.reorient(-25, 76, 0, (-0.13, 0.29, 0.85), 3.38).set_anim_args(time_span=(2, 5)),
            Transform(sphere_group, sphere_target),
            water.radius_tracker.animate.set_value(0.01),
            water.opacity_tracker.animate.set_value(0.1),
            run_time=5
        )
        frame.add_ambient_rotation(0.5 * DEG)

        # Show vectors
        direction = OUT + 0.2 * LEFT

        vect = Vector(0.35 * direction, fill_color=GREEN)
        vect.set_perpendicular_to_camera(frame)
        vect.shift(1.1 * direction)
        vect.set_fill(opacity=0.75)

        vect1 = vect.copy()
        vect2 = vect.copy().shift(0.4 * direction).set_color(RED)
        vect3 = vect.copy().shift(0.8 * direction)

        dashed_line = DashedLine(ORIGIN, 5 * direction, dash_length=0.02)
        dashed_line.set_stroke(WHITE, 4)
        dashed_line_ghost = dashed_line.copy()
        dashed_line_ghost.set_stroke(opacity=0.25)
        dashed_line.apply_depth_test()

        self.play(
            ShowCreation(dashed_line, run_time=5),
            ShowCreation(dashed_line_ghost, run_time=5),
        )
        self.wait()
        self.play(
            GrowArrow(vect1),
            GrowArrow(vect3),
        )
        self.wait(3)
        self.play(GrowArrow(vect2))
        self.wait(4)
        self.play(
            FadeOut(VGroup(vect1, vect2, vect3)),
            FadeOut(dashed_line),
            FadeOut(dashed_line_ghost),
        )

        # Shift back
        frame.clear_updaters()
        self.play(
            Restore(sphere_group),
            frame.animate.reorient(10, 78, 0, ORIGIN, 7),
            run_time=7
        )


class InsideOutWithNormalField(InteractiveScene):
    def construct(self):
        # Axes and plane
        radius = 3
        frame = self.frame
        axes = ThreeDAxes()
        plane = NumberPlane((-8, 8), (-8, 8))
        plane.background_lines.set_stroke(GREY_C, 1)
        plane.faded_lines.set_stroke(GREY_C, 0.5, 0.5)
        plane.apply_depth_test()
        axes.apply_depth_test()
        self.add(axes, plane)

        # Add sphere
        outer_sphere = Sphere(radius=radius)
        inner_sphere = Sphere(radius=0.99 * radius)
        outer_sphere.set_color(BLUE_E, 1)
        inner_sphere.set_color(interpolate_color(GREY_BROWN, BLACK, 0.5), 1)
        for sphere in [inner_sphere, outer_sphere]:
            sphere.always_sort_to_camera(self.camera)
            sphere.set_shading(0.2, 0.2, 0.1)
        mesh = SurfaceMesh(outer_sphere, resolution=(61, 31))
        mesh.set_stroke(WHITE, 1, 0.5)
        mesh.set_z_index(2)

        def get_unit_normal_field(u_range, v_range, sign=1):
            return get_sphereical_vector_field(
                lambda p: sign * p,
                axes,
                points=np.array([
                    sphere.uv_func(u, v)
                    for u in u_range
                    for v in v_range
                ]),
                color=BLUE_D,
                stroke_width=3
            )

        u_range = np.arange(0, TAU, TAU / 30)
        v_range = np.arange(0, PI, PI / 15)
        outer_normals = get_unit_normal_field(u_range, v_range)
        inner_normals = get_unit_normal_field(u_range, v_range, -1)

        sign_tracker = ValueTracker(1)
        outer_normals.add_updater(lambda m: m.set_stroke(opacity=float(sign_tracker.get_value() > 0)))
        inner_normals.add_updater(lambda m: m.set_stroke(opacity=float(sign_tracker.get_value() < 0)))

        sphere_group = Group(inner_sphere, outer_sphere, mesh, outer_normals, inner_normals)
        sphere_group.set_clip_plane(UP, 0)
        frame.reorient(-23, 72, 0)
        self.add(sphere_group, axes)

        # Inversion homotopy
        def homotopy(x, y, z, t, scale=-1):
            p = np.array([x, y, z])
            power = 1 + 0.2 * (x / radius)
            return interpolate(p, scale * p, t**power)

        frame.add_ambient_rotation(3 * DEG)
        self.play(
            Homotopy(homotopy, sphere_group),
            sign_tracker.animate.set_value(-1),
            run_time=5
        )
        self.wait(3)


class ProjectedCombedHypersphere(InteractiveScene):
    def construct(self):
        # Set up
        frame = self.frame
        axes = ThreeDAxes((-5, 5), (-5, 5), (-5, 5))
        plane = NumberPlane((-5, 5), (-5, 5))
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        frame.reorient(86, 78, 0, (-0.13, 0.04, 0.63))
        self.add(axes)

        # Add lines
        flow_lines = VGroup(
            self.get_flow_line_from_point(normalize(np.random.normal(0, 1, 4)))
            for n in range(2000)
        )
        for line in flow_lines:
            line.virtual_time = TAU
            line.get_center()
            stroke_width = clip(get_norm(line.get_center()), 0, 3)
            color = random_bright_color(hue_range=(0.45, 0.55))
            line.set_stroke(color, stroke_width)

        self.add(flow_lines)

        frame.add_ambient_rotation(4 * DEG)
        self.play(
            LaggedStartMap(
                VShowPassingFlash,
                flow_lines,
                lag_ratio=3 / len(flow_lines),
                run_time=45,
                time_width=0.7,
                rate_func=linear
            )
        )

    def get_flow_line_from_point(self, point4d, stroke_color=WHITE, stroke_width=2):
        points4d = self.get_hypersphere_circle_points(point4d)
        points3d = self.streo_4d_to_3d(points4d)
        line = VMobject().set_points_smoothly(points3d)
        line.set_stroke(stroke_color, stroke_width)
        return line

    def get_hypersphere_circle_points(self, point4d, n_samples=100):
        x, y, z, w = point4d
        perp = np.array([-y, x, -w, z])
        return np.array([
            math.cos(a) * point4d + math.sin(a) * perp
            for a in np.linspace(0, TAU, n_samples)
        ])

    def streo_4d_to_3d(self, points4d):
        xyz = points4d[:, :3]
        w = points4d[:, 3]

        # Stereographic projection formula: (x, y, z) / (1 - w)
        # Reshape w for broadcasting
        denominator = (1 - w).reshape(-1, 1)

        # Handle potential division by zero (north pole)
        # Add small epsilon to avoid exact division by zero
        epsilon = 1e-10
        denominator = np.where(np.abs(denominator) < epsilon, epsilon, denominator)

        projected = xyz / denominator

        return projected`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      10: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      40: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      41: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      52: "Stereographic projection maps sphere points to the plane: (x,y,z) → (x/(1-z), y/(1-z)). Preserves angles (conformal).",
      76: "Stereographic projection maps sphere points to the plane: (x,y,z) → (x/(1-z), y/(1-z)). Preserves angles (conformal).",
      107: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      112: "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere.",
      113: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      127: "VectorField samples a vector function at grid points and renders arrows. Useful for visualizing flows and forces.",
      136: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      138: "When True, stroke width scales with camera zoom. When False, strokes maintain constant screen-space width.",
      142: "SphereStreamLines extends StreamLines. StreamLines traces curves that follow a vector field flow. Often wrapped in AnimatedStreamLines for dynamic visualization.",
      149: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      157: "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere.",
      164: "TeddyHeadSwirl extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      165: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      168: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      173: "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
      176: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      177: "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector.",
      181: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      183: "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids.",
      198: "Wraps StreamLines in a continuous animation that spawns, flows, and fades line segments.",
      199: "Wraps StreamLines in a continuous animation that spawns, flows, and fades line segments.",
      201: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      203: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      205: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      206: "Smoothly animates the camera to a new orientation over the animation duration.",
      211: "IntroduceVectorField extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      212: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      217: "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
      219: "Sphere creates a parametric surface mesh. Can be textured, made transparent, and depth-tested for 3D rendering.",
      221: "Reorders transparent faces each frame for correct alpha blending from the current camera angle.",
      222: "SurfaceMesh draws wireframe grid lines on a Surface for spatial reference.",
      225: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      226: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      227: "Smoothly animates the camera to a new orientation over the animation duration.",
      228: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      229: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      239: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      253: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      254: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      255: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      258: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      261: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      262: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      263: "Smoothly animates the camera to a new orientation over the animation duration.",
      266: "Smoothly animates the camera to a new orientation over the animation duration.",
      267: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      272: "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere.",
      273: "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere.",
      279: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      280: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      285: "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere.",
      464: "StereographicProjection extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      465: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      745: "SimpleRightwardFlow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      746: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      779: "SingleNullPointHairyBall extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      782: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      853: "Class SingleNullPointHairyBallRevealed inherits from SingleNullPointHairyBall.",
      857: "PairsOfNullPoints extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      858: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      973: "AskAboutOutside extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      974: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1123: "InsideOut extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1124: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1356: "UnitNormals extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1357: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1399: "Class DefineOrientation inherits from InsideOut.",
      1400: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1756: "FlowingWater extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1757: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1962: "Class SurfaceFoldedOverSelf inherits from FlowingWater.",
      1963: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2060: "InsideOutWithNormalField extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2061: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2127: "ProjectedCombedHypersphere extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2128: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2026/hairy_ball/supplements.py"] = {
    description: "Supplementary scenes for the hairy ball theorem: additional examples of vector fields with zeros, counterexamples on tori, and topological context.",
    code: `from manim_imports_ext import *


class WhyDoWeCare(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            self.change_students("confused", "erm", "concentrating", look_at=self.screen),
        )
        self.wait(3)
        self.play(
            stds[2].change("erm", stds[1].eyes),
            stds[1].says("I’m sorry, why\\ndo we care?", mode="sassy"),
            stds[0].change("thinking", self.screen),
            morty.change("well"),
        )
        self.wait(2)
        self.play(self.change_students("pondering", "maybe", "pondering", look_at=self.screen))

        # Answer
        self.play(
            morty.says("Topology has\\nmore subtle utility", mode="tease"),
            stds[0].animate.look_at(morty.eyes),
            stds[1].debubble(),
            stds[2].change("hesitant", morty.eyes)
        )
        self.wait(3)


class RenameTheorem(InteractiveScene):
    def construct(self):
        # Test
        name1, name2 = names = VGroup(
            Text("Hairy Ball Theorem"),
            Text("Sphere Vector Field Theorem"),
        )
        names.scale(1.25)
        names.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        names.to_edge(LEFT)

        lines = VGroup()
        for text in ["Hairy", "Ball"]:
            word = name1[text][0]
            line = Line(word.get_left(), word.get_right())
            line.set_stroke(RED, 8)
            lines.add(line)
        lines[0].align_to(lines[1], UP)

        self.add(name1)
        self.wait()
        self.play(
            ShowCreation(lines[1]),
            name1["Ball"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Ball"].copy(), name2["Sphere"]),
        )
        self.play(
            ShowCreation(lines[0]),
            name1["Hairy"].animate.set_opacity(0.5),
            FadeTransformPieces(name1["Hairy"].copy(), name2["Vector Field"]),
        )
        self.play(
            TransformFromCopy(name1["Theorem"], name2["Theorem"]),
        )
        self.wait()


class SimpleImplies(InteractiveScene):
    def construct(self):
        arrow = Tex(R"\\Rightarrow", font_size=120)
        self.play(Write(arrow))
        self.wait()


class CommentOnForce(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        morty.body.insert_n_curves(1000)
        equation = Tex(R"m x''(t) = \\text{Lift} + \\text{Gravity}", t2c={R"x''(t)": RED, R"\\text{Lift}": PINK, R"\\text{Gravity}": BLUE})
        equation.move_to(self.hold_up_spot, DOWN)
        equation.shift_onto_screen()

        self.play(
            morty.change("tease"),
            self.change_students("thinking", "erm", "concentrating", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", equation),
            self.change_students("pondering", "confused", "hesitant", look_at=equation),
            Write(equation),
        )
        self.wait(3)


class WingVectCodeSnippet(InteractiveScene):
    def construct(self):
        # Test
        code = Code("""
            def wing_vect(heading_vect):
                \\"\\"\\"
                Return 3d vector perpendicular
                to heading_vect
                \\"\\"\\"
                ...
        """, alignment="LEFT")
        self.play(ShowIncreasingSubsets(code, run_time=2, rate_func=linear))
        self.wait()


class LazyPerpCodeSnippet(InteractiveScene):
    def construct(self):
        # Test
        code = Code("""
            def lazy_perp(heading):
                # Returns the normalized cross product
                # between (0, 0, 1) and heading
                # Note the division by 0 for x=y=0
                x, y, z = heading
                return np.array([-y, x, 0]) / np.sqrt(x * x + y * y)
        """, alignment="LEFT")
        code.to_corner(UL)
        self.play(Write(code))
        self.wait()


class StatementOfTheorem(InteractiveScene):
    def construct(self):
        # Add text
        title = Text("Hairy Ball Theorem", font_size=72)
        title.to_corner(UL)
        underline = Underline(title)

        self.add(title, underline)

        statement = Text("""
            Any continuous vector field
            on a sphere must have at least
            one null vector.
        """, alignment="LEFT")
        statement.next_to(underline, DOWN, buff=MED_LARGE_BUFF)
        statement.to_edge(LEFT)

        self.play(Write(statement, run_time=3, lag_ratio=1e-1))
        self.wait()

        statement.set_backstroke(BLACK, 5)

        # Highlight text
        for text, color in [("continuous", BLUE), ("one null vector", YELLOW)]:
            self.play(
                FlashUnder(statement[text], time_width=1.5, run_time=2, color=color),
                statement[text].animate.set_fill(color)
            )
            self.wait()


class WriteAntipode(InteractiveScene):
    def construct(self):
        # Test
        text1 = Text("“Antipodes”")
        text2 = Text("Antipode map")
        for text in [text1, text2]:
            text.scale(1.5)
            text.to_corner(UL)

        self.play(Write(text1), run_time=2)
        self.wait()
        self.play(TransformMatchingStrings(text1, text2), run_time=1)
        self.wait()


class Programmer(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().fix_in_frame())
        laptop = Laptop()
        self.frame.reorient(60, 66, 0, (0.09, -0.5, 0.13), 4.12)

        randy = Randolph(height=5)
        randy.to_edge(LEFT)
        randy.add_updater(lambda m: m.fix_in_frame().look_at(4 * RIGHT))

        self.add(laptop)
        self.play(randy.change("hesitant"))
        self.play(Blink(randy))
        self.play(randy.change("concentrating"))
        self.play(Blink(randy))
        self.wait()


class PedanticStudent(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change('raise_right_hand'),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(LaggedStart(
            stds[2].says("But atmosphere\\nis 3D!", mode="angry", look_at=morty.eyes, bubble_direction=LEFT),
            morty.change("guilty"),
            stds[0].change("hesitant", look_at=stds[2].eyes),
            stds[1].change("hesitant", look_at=stds[2].eyes),
        ))
        self.wait(2)
        self.look_at(self.screen)
        self.wait(3)


class YouAsAMathematician(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=4)
        randy.move_to(3 * LEFT)
        label = VGroup(
            Text("You", font_size=72),
            Text("The mathematician").set_color(GREY_B)
        )
        label.arrange(DOWN)
        label.next_to(randy, DOWN)

        self.add(randy, label)
        self.play(randy.change("pondering", 3 * RIGHT))
        self.play(Blink(randy))
        self.play(randy.change("tease", 3 * RIGHT))
        self.wait(3)


class ThreeCases(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            VGroup(Text("2 null points"), Text("Obvious")),
            VGroup(Text("1 null point"), Text("Clever")),
            VGroup(Text("0 null points"), Text("Very clever")),
        )
        for title in titles:
            title[0].set_color(GREY_B)
            title[1].scale(1.25)
            title.arrange(DOWN)
        titles.arrange(RIGHT, buff=1.5, aligned_edge=UP)
        titles.to_edge(UP)

        vc_cross = Cross(titles[2][1])

        why_not = Text("Why not?")
        why_not.next_to(title)
        why_not.set_color(YELLOW)
        why_not.next_to(titles[2], DOWN, aligned_edge=RIGHT)

        for title in titles:
            self.add(title[0])
        for title in titles:
            self.play(FadeIn(title[1], lag_ratio=0.1))
        self.wait()
        self.play(ShowCreation(vc_cross))
        self.play(Write(why_not))


class ProofOutline(InteractiveScene):
    def construct(self):
        # Add outline
        title = Text("Proof by Contradiction", font_size=72)
        title.to_edge(UP)
        background = FullScreenRectangle()

        frames = Square().replicate(2)
        frames.set_height(4.5)
        frames.arrange(RIGHT, buff=3.5)
        frames.next_to(title, DOWN, buff=1.5)
        frames.set_fill(BLACK, 1)
        frames.set_stroke(WHITE, 2)

        implies = Tex(R"\\Longrightarrow", font_size=120)
        implies.move_to(frames)

        impossibility = Text("Impossibility", font_size=90)
        impossibility.next_to(implies, RIGHT, MED_LARGE_BUFF)
        impossibility.set_color(RED)

        assumption = Text("Assume there exists a non-zero\\ncontinuous vector field", font_size=30)
        assumption.set_color(BLUE)
        assumption.next_to(frames[0], UP)

        self.add(background)
        self.play(Write(title), run_time=2)
        self.wait()
        self.play(
            FadeIn(frames[0]),
            # FadeIn(assumption, lag_ratio=0.01)
        )
        self.wait()
        implies.save_state()
        implies.stretch(0, 0, about_edge=LEFT)
        self.play(Restore(implies))
        self.play(FadeIn(impossibility, lag_ratio=0.1))
        self.wait()
        self.play(
            DrawBorderThenFill(frames[1]),
            impossibility.animate.scale(0.5).next_to(frames[1], UP)
        )
        self.wait()
        self.play(FadeOut(impossibility))

        # Next part
        words = VGroup(Text("Assume the impossible"), Text("Find a contradiction"))
        brace = Brace(frames[0], RIGHT)
        question = Text("What do we\\nshow here?", font_size=72)
        question.next_to(brace, RIGHT)

        self.play(
            FadeOut(implies),
            FadeOut(frames[1]),
        )
        self.play(
            GrowFromCenter(brace),
            FadeIn(question, lag_ratio=0.1),
        )
        self.wait()


class AimingForRediscovery(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        goal = Text("Goal: A feeling\\nof rediscovery")
        goal.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("tease"),
            self.change_students("pondering", "happy", "hooray", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            FadeIn(goal, shift=UP),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=goal)
        )
        self.wait(2)
        self.play(self.change_students("thinking", "tease", "erm", look_at=self.screen))
        self.wait()


class TwoFactsForEachPoint(InteractiveScene):
    def construct(self):
        # Test
        features = VGroup(
            Tex(R"\\text{1)  } p \\rightarrow -p", font_size=72),
            Tex(R"\\text{2)  } &\\text{Motion varies }\\\\ &\\text{continuously with } p", font_size=72),
        )
        features[1][2:].scale(0.8, about_edge=UL)
        features.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        features.to_edge(LEFT)

        for feature in features:
            self.add(feature[:2])
        self.wait()
        for feature in features:
            self.play(FadeIn(feature[2:], lag_ratio=0.1))
            self.wait()
        self.wait()


class WaitWhat(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=4)
        randy.to_edge(LEFT, buff=2.5)
        randy.shift(DOWN)
        randy.body.insert_n_curves(1000)

        self.play(randy.says("Wait...what?", mode="confused", bubble_direction=RIGHT, look_at=RIGHT))
        self.play(Blink(randy))
        self.wait(3)


class TwoKeyFeatures(InteractiveScene):
    def construct(self):
        # Set up
        features = VGroup(
            # Text("1) Sphere turns\\ninside out"),
            # Text("2) No point touches\\nthe origin"),
            Text("1) Inside out"),
            Text("2) Avoids the origin"),
        )
        features[1]["the origin"].align_to(features[1]["No"], LEFT)
        features.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        features.to_edge(LEFT)

        for feature in features:
            self.play(FadeIn(feature, lag_ratio=0.1))
            self.wait()

        # Emphasize first point
        self.play(
            features[0].animate.scale(1.25, about_edge=LEFT),
            features[1].animate.scale(0.75, about_edge=LEFT).set_fill(opacity=0.5).shift(DOWN),
        )
        self.wait()

        # Ask why
        randy = Randolph(height=1.75)
        randy.next_to(features[0], DOWN, buff=MED_LARGE_BUFF)
        why = Text("Why?", font_size=36)
        why.next_to(randy, RIGHT, aligned_edge=UP)
        why.set_color(YELLOW)

        self.play(
            VFadeIn(randy),
            randy.change("maybe"),
            Write(why),
        )
        self.play(Blink(randy))
        self.wait()

        # Ask what "inside out" means
        rect = SurroundingRectangle(features[0][2:])
        rect.set_stroke(YELLOW, 2)
        self.play(
            randy.change("confused", rect),
            FadeTransform(why, rect)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(randy), FadeOut(rect))

        # Inside out implication
        rect0 = SurroundingRectangle(features[0])
        rect0.set_stroke(BLUE, 2)

        implies0 = Tex(R"\\Longrightarrow", font_size=72)
        implies0.next_to(rect0)
        net_flow_m1 = TexText("Final Flux = $-1.0$", t2c={"-1.0": RED}, font_size=60)
        net_flow_m1.next_to(implies0, RIGHT)

        self.play(
            ShowCreation(rect0),
            FadeIn(implies0, scale=2, shift=0.25 * RIGHT),
        )
        self.play(FadeIn(net_flow_m1, lag_ratio=0.1))
        self.wait()

        # No origin implication
        self.play(features[1].animate.scale(1.25 / 0.75, about_edge=UL).set_opacity(1))

        rect1 = SurroundingRectangle(features[1])
        rect1.match_style(rect0)
        implies1 = implies0.copy()
        implies1.next_to(rect1)
        net_flow_p1 = TexText(R"Final flux = $+1.0$", t2c={"+1.0": GREEN}, font_size=60)  
        net_flow_p1.next_to(implies1, RIGHT)

        self.play(
            ShowCreation(rect1),
            FadeIn(implies1, scale=2, shift=0.25 * RIGHT),
        )
        self.play(FadeIn(net_flow_p1, lag_ratio=0.1))
        self.wait()

        # Contradiction
        contra = Tex(R"\\bot", font_size=90)
        contra.to_corner(DR)

        self.play(Write(contra))
        self.wait()


class DumbQuestion(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[2].says("Isn’t it\\nobvious?", mode="confused", look_at=self.screen, bubble_direction=LEFT),
            stds[1].change("angry", look_at=morty.eyes),
            stds[0].change("erm", look_at=morty.eyes),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].change("sassy", self.screen),
            stds[2].change("maybe", morty.eyes),
        )
        self.wait(4)


class InsideOutsideQuestion(InteractiveScene):
    def construct(self):
        # Test
        inside = Text("Inside?", font_size=72)
        outside = Text("Outside?", font_size=72)
        VGroup(inside, outside).set_backstroke(BLACK, 3)
        self.play(FadeIn(inside, lag_ratio=0.1))
        self.wait()
        self.play(
            FadeIn(outside, lag_ratio=0.1),
            FadeOut(inside, lag_ratio=0.1),
        )
        self.wait()


class WhatIsInsideAndOutside(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says(
                "Hang on, what\\ndo you mean\\n“paint the outside”?",
                mode="maybe",
                bubble_direction=LEFT
            ),
            stds[1].change("erm", self.screen),
            stds[0].change("pondering", self.screen),
            morty.change("tease")
        )
        self.wait(5)


class PToNegP(InteractiveScene):
    def construct(self):
        # Test
        p, to, neg_p = expression = VGroup(
            Tex(R"p"), Tex(R"\\longrightarrow"), Tex(R"-p")
        )
        expression.arrange(RIGHT, buff=0.75)
        expression.scale(2.5)
        expression.to_edge(UP)

        to.save_state()
        to.stretch(0, 0, about_edge=LEFT)
        to.stretch(0.5, 1)

        self.play(Write(p))
        self.play(Restore(to))
        self.play(FadeTransformPieces(p.copy(), neg_p))
        self.wait()


class SimplerInsideOutProgression(InteractiveScene):
    def construct(self):
        # Test
        parts = VGroup(
            Tex(R"(x, y, z)"),
            Vector(0.75 * DOWN),
            Tex(R"(-x, -y, z)"),
            Vector(0.75 * DOWN),
            Tex(R"(-x, -y, -z)"),
        )
        parts.arrange(DOWN, buff=MED_SMALL_BUFF)
        self.add(parts[0])
        self.wait()

        for i in [0, 2]:
            src, arrow, trg = parts[i:i + 3]
            self.play(
                TransformMatchingStrings(src.copy(), trg),
                GrowArrow(arrow),
                run_time=1
            )
            self.wait()


class ReferenceInsideOutMovie(TeacherStudentsScene):
    def construct(self):
        # Complain
        morty = self.teacher
        stds = self.students
        self.screen.to_corner(UL)

        self.play(
            stds[0].change("pondering", self.screen),
            stds[1].change("erm", self.screen),
            stds[2].says(
                Text(
                    "Huh? I thought\\nyou can turn a\\nsphere inside out!",
                    t2s={"can": ITALIC},
                    font_size=42,
                ),
                mode="confused",
                look_at=self.screen,
                bubble_direction=LEFT
            ),
            morty.change("guilty")
        )
        self.wait(2)
        self.play(morty.change('tease'))
        self.wait(3)


class FluxDecimals(InteractiveScene):
    def construct(self):
        # Test
        label = TexText("Flux: +1.000 L/s", font_size=60)
        dec = label.make_number_changeable("+1.000", include_sign=True)
        label.to_corner(UR)
        dec.set_value(1)

        def update_color(dec, epsilon=1e-4):
            value = dec.get_value()
            if value > epsilon:
                dec.set_color(GREEN)
            elif abs(value) < epsilon:
                dec.set_color(YELLOW)
            else:
                dec.set_color(RED)

        dec.add_updater(update_color)

        self.add(label)
        self.wait()
        for value in [0.014, -0.014, 0.014]:
            self.play(ChangeDecimalToValue(dec, value))
            self.wait()
        self.play(ChangeDecimalToValue(dec, 1.0), run_time=3)
        self.wait()
        dec.set_value(0)
        self.wait()


class DivergenceTheorem(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        div_theorem = Tex(R"""
            \\displaystyle \\iiint_V(\\nabla \\cdot \\mathbf{F}) \\mathrm{d} V
            = \\oiint_S(\\mathbf{F} \\cdot \\hat{\\mathbf{n}}) \\mathrm{d} S
        """, t2c={R"\\mathbf{F}": BLUE})
        div_theorem.next_to(morty, UP, buff=1.5)
        div_theorem.to_edge(RIGHT)

        div_theorem_name = Text("Divergence Theorem", font_size=72)
        div_theorem_name.next_to(div_theorem, UP, buff=0.5)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)

        self.play(
            self.change_students("pondering", "confused", "tease", look_at=self.screen),
            morty.change("tease"),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand", div_theorem),
            FadeIn(div_theorem, shift=UP),
            self.change_students("thinking", "confused", "happy", look_at=div_theorem),
        )
        self.wait()
        self.play(Write(div_theorem_name))
        self.wait(3)


class ThinkAboutOrigin(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.says(Text("Think about why\\ncrossing the origin\\nis significant", font_size=36)),
            self.change_students('thinking', 'tease', 'pondering', look_at=self.screen)
        )
        self.wait(5)


class CommentOnContardiction(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(height=3)
        morty.body.insert_n_curves(1000)
        morty.to_corner(DR)
        morty.shift(2 * LEFT)

        qed = Text("Q.E.D.")
        qed.next_to(morty.get_corner(UR), UP, MED_SMALL_BUFF)

        self.play(morty.says("Contradiction!", mode="hooray"))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change('raise_left_hand', look_at=qed), FadeIn(qed, 0.25 * UP))
        self.play(Blink(morty))
        self.play(morty.change('tease'))
        self.wait()


class FrameIntuitionVsExamples(InteractiveScene):
    def construct(self):
        titles = VGroup(
            Text("Intuitive idea"),
            Text("Counterexample"),
            Text("Clever proof"),
        )
        for x, title in zip([-1, 1, 1], titles):
            title.scale(1.5)
            title.move_to(x * FRAME_WIDTH * RIGHT / 4)
            title.to_edge(UP)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.to_edge(UP, buff=1.5)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        VGroup(h_line, v_line).set_stroke(WHITE, 2)

        ideas = VGroup(
            Text("Turning a sphere\\ninside-out must crease it"),
            Text("All closed loops\\nhave inscribed rectangles"),
        )
        for idea in ideas:
            idea.next_to(h_line, DOWN)
            idea.set_color(GREY_A)
            idea.shift(FRAME_WIDTH * LEFT / 4)

        self.add(v_line, h_line)
        self.add(titles[0])
        self.wait()

        # Test
        self.play(FadeIn(ideas[0]))
        self.play(
            FadeIn(titles[1], lag_ratio=0.1)
        )
        self.wait()
        self.play(
            FadeOut(ideas[0]),
            FadeIn(ideas[1]),
            FadeOut(titles[1], lag_ratio=0.1),
            FadeIn(titles[2], lag_ratio=0.1),
        )
        self.wait()


class DimensionGeneralization(InteractiveScene):
    def construct(self):
        # Set up grid
        row_labels = VGroup(
            Text("Dimension"),
            Text("Can you\\ncomb a ball?"),
        )
        n_cols = 15
        cells = Square().get_grid(2, 1, buff=0).get_grid(1, n_cols, buff=0)
        cells.set_height(2.6)
        cells[0].set_width(row_labels.get_width() + MED_LARGE_BUFF, stretch=True, about_edge=RIGHT)
        cells.to_corner(UL, buff=LARGE_BUFF)
        for label, cell in zip(row_labels, cells[0]):
            label.move_to(cell)

        dim_labels = VGroup()
        mark_labels = VGroup()
        for n, cell in zip(it.count(2), cells[1:]):
            dim_label = Integer(n)
            mark_label = Checkmark().set_color(GREEN) if n % 2 == 0 else Exmark().set_color(RED)
            mark_label.set_height(0.5 * cell[1].get_height())
            dim_label.move_to(cell[0])
            mark_label.move_to(cell[1])

            dim_labels.add(dim_label)
            mark_labels.add(mark_label)

        self.add(cells[:3], row_labels, dim_labels[:2], mark_labels[1])
        self.play(
            LaggedStartMap(FadeIn, cells[3:], lag_ratio=0.5),
            LaggedStartMap(FadeIn, dim_labels[2:], lag_ratio=0.5),
            run_time=3
        )
        self.wait()

        # Show two
        self.play(Write(mark_labels[0]))
        self.wait()

        # General dimensions
        frame = self.frame
        for i in [0, 1]:
            self.play(
                LaggedStart(
                    (TransformFromCopy(mark_labels[i], mark_label, path_arc=30 * DEG)
                    for mark_label in mark_labels[i + 2::2]),
                    lag_ratio=0.25,
                ),
                frame.animate.reorient(0, 0, 0, (4.15, -2.53, 0.0), 12.34),
                run_time=3
            )
            self.wait()

        # Show determinants
        last_det = VGroup()
        highlight_rect = SurroundingRectangle(cells[1], buff=0)
        highlight_rect.set_opacity(0).shift(LEFT)
        for dim in range(2, 12):
            det_tex = self.get_det_neg_tex(dim)
            det_tex.scale(1.5)
            det_tex.move_to(5 * DOWN).to_edge(LEFT, LARGE_BUFF)
            rect = SurroundingRectangle(cells[dim - 1], buff=0)
            rect.set_stroke(YELLOW, 5)

            self.play(
                Transform(highlight_rect, rect),
                FadeIn(det_tex),
                FadeOut(last_det),
            )
            self.wait()
            last_det = det_tex

    def get_det_neg_tex(self, dim):
        mat = IntegerMatrix(-1 * np.identity(dim))
        det_text = Text("det")
        mat.set_max_height(5 * det_text.get_height())
        lp, rp = parens = Tex(R"()")
        parens.stretch(2, 1)
        parens.match_height(mat)
        lp.next_to(mat, LEFT, buff=0.1)
        rp.next_to(mat, RIGHT, buff=0.1)
        det_text.next_to(parens, LEFT, SMALL_BUFF)

        sign = ["+", "-"][dim % 2]
        rhs = Tex(Rf"= {sign}1")
        rhs.next_to(rp, RIGHT, SMALL_BUFF)
        rhs[1:].set_color([GREEN, RED][dim % 2])

        result = VGroup(det_text, lp, mat, rp, rhs)
        return result


class MoreRigorNeeded(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        words = Text("More rigor\\nneeded", font_size=60)
        arrow = Vector(1.5 * LEFT, thickness=8)
        label = VGroup(arrow, words)
        label.arrange(RIGHT, SMALL_BUFF)
        label.next_to(self.screen, RIGHT)

        self.add(words)
        self.play(
            morty.change("hesitant"),
            self.change_students("confused", "sassy", "erm", look_at=self.screen),
            Write(words),
        )
        self.play(
            GrowArrow(arrow)
        )
        self.wait(4)


class AskAboutHomology(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        chain = Tex(R"\\cdots C_{2} \\xrightarrow{\\partial_2} C_1 \\xrightarrow{\\partial_1} C_0 \\xrightarrow{\\partial_0} 0")
        chain.next_to(stds[2], UP, buff=MED_LARGE_BUFF)
        chain.align_to(stds[2].get_center(), RIGHT)

        self.play(
            morty.change("well", stds[2].eyes),
            stds[2].says(
                "What about\\nusing homology?",
                mode="tease",
                bubble_direction=LEFT,
                look_at=morty.eyes
            )
        )
        self.play(
            stds[2].change("raise_left_hand", chain),
            self.change_students("confused", "erm"),
            Write(chain),
        )
        self.wait(3)


class RotationIn2D(InteractiveScene):
    def construct(self):
        # Test
        grid = NumberPlane()
        back_grid = grid.copy()
        back_grid.background_lines.set_stroke(GREY, 1)
        back_grid.axes.set_stroke(GREY, 1, 1)
        back_grid.faded_lines.set_stroke(GREY, 0.5, 0.5)

        basis_vectors = VGroup(
            Vector(RIGHT).set_color(GREEN),
            Vector(UP).set_color(RED),
        )

        self.frame.set_height(4)
        self.add(back_grid, grid, basis_vectors)
        self.wait()
        self.play(
            Rotate(basis_vectors, PI, about_point=ORIGIN),
            Rotate(grid, PI, about_point=ORIGIN),
            run_time=5
        )
        self.wait()


class InversionIn3d(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        frame.add_ambient_rotation(1 * DEG)
        axes = ThreeDAxes((-3, 3), (-3, 3), (-3, 3))
        coord_range = list(range(-3, 3))
        cubes = VGroup(
            VCube(side_length=1).move_to([x, y, z], DL + IN)
            for x, y, z in it.product(* 3 * [coord_range])
        )
        cubes.set_fill(opacity=0)
        cubes.set_stroke(WHITE, 1, 0.25)

        basis_vectors = VGroup(
            Vector(RIGHT).set_color(GREEN),
            Vector(UP).set_color(RED),
            Vector(OUT).set_color(BLUE),
        )
        # for vect in basis_vectors:
        #     vect.set_perpendicular_to_camera(frame)

        frame.reorient(29, 72, 0, ORIGIN, 5)
        self.add(axes, cubes, basis_vectors)

        # Rotate
        rot_group = VGroup(cubes, basis_vectors)
        self.wait()
        self.play(
            Rotate(rot_group, PI, about_point=ORIGIN, run_time=3)
        )
        self.play(
            rot_group.animate.stretch(-1, 2, about_point=ORIGIN),
            run_time=2
        )
        self.wait(3)


class HypersphereWords(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Hairs on a neatly-combed 4d hypersphere", font_size=60),
            Text("Represented via streographic projection into 3d space", font_size=48).set_color(GREY_A)
        )
        words.set_backstroke(BLACK, 10)
        words.arrange(DOWN)
        words.to_corner(UL)

        self.add(words[0])
        self.wait()
        self.play(FadeIn(words[1], lag_ratio=0.1, run_time=2))
        self.wait()


class EndScreen2(SideScrollEndScreen):
    scroll_time = 23`,
    annotations: {
      4: "Class WhyDoWeCare inherits from TeacherStudentsScene.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      33: "RenameTheorem extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      34: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      70: "SimpleImplies extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      71: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      77: "Class CommentOnForce inherits from TeacherStudentsScene.",
      78: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      99: "WingVectCodeSnippet extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      100: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      114: "LazyPerpCodeSnippet extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      115: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      130: "StatementOfTheorem extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      131: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      161: "WriteAntipode extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      162: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      176: "Programmer extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      177: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      195: "Class PedanticStudent inherits from TeacherStudentsScene.",
      196: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      217: "YouAsAMathematician extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      218: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      236: "ThreeCases extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      237: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      267: "ProofOutline extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      268: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      329: "Class AimingForRediscovery inherits from TeacherStudentsScene.",
      330: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      353: "TwoFactsForEachPoint extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      354: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      373: "WaitWhat extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      374: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      386: "TwoKeyFeatures extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      387: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      477: "Class DumbQuestion inherits from TeacherStudentsScene.",
      478: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      498: "InsideOutsideQuestion extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      499: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      513: "Class WhatIsInsideAndOutside inherits from TeacherStudentsScene.",
      514: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      532: "PToNegP extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      533: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      552: "SimplerInsideOutProgression extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      553: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      576: "Class ReferenceInsideOutMovie inherits from TeacherStudentsScene.",
      577: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      603: "FluxDecimals extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      604: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      633: "Class DivergenceTheorem inherits from TeacherStudentsScene.",
      634: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      667: "Class ThinkAboutOrigin inherits from TeacherStudentsScene.",
      668: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      677: "CommentOnContardiction extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      678: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      697: "FrameIntuitionVsExamples extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      698: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      741: "DimensionGeneralization extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      742: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      833: "Class MoreRigorNeeded inherits from TeacherStudentsScene.",
      834: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      856: "Class AskAboutHomology inherits from TeacherStudentsScene.",
      857: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      883: "RotationIn2D extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      884: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      908: "InversionIn3d extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      909: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      946: "HypersphereWords extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      947: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      963: "Class EndScreen2 inherits from SideScrollEndScreen.",
    }
  };

  files["_2026/hairy_ball/talent.py"] = {
    description: "Talent/presentation scenes for the hairy ball video: intro sequences, visual hooks, and narrative transitions.",
    code: `from manim_imports_ext import *


class TalentPartnerProperties(InteractiveScene):
    def construct(self):
        # Company seeking you
        company = SVGMobject("company_building")
        company.set_height(3)
        company.set_fill(GREY_B)
        company.set_shading(0.5, 0.5)
        company.to_edge(LEFT, buff=0.5)

        laptop = Laptop(width=2.5)
        laptop.rotate(80 * DEG, LEFT)
        laptop.rotate(80 * DEG, DOWN)
        laptop.to_edge(RIGHT).shift(0.5 * DOWN)
        randy = Randolph(mode="pondering", height=2.5)
        randy.next_to(laptop, LEFT, buff=0)
        randy.align_to(company, DOWN)
        randy.look_at(laptop.screen)

        arrow = Arrow(randy.get_corner(DL), company.get_corner(DR), thickness=8, buff=0.5)
        arrow.match_y(randy)

        self.add(company)
        self.add(laptop)
        self.add(randy)
        self.play(Blink(randy))
        self.play(GrowArrow(arrow))
        self.play(randy.change("well", company))
        self.wait()
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("thinking", laptop.screen))
        self.play(Blink(randy))
        self.wait()

        # Others at the company
        others = VGroup(
            PiCreature(color=BLUE_D, mode="gracious"),
            PiCreature(color=BLUE_C, mode="tease"),
            PiCreature(color=BLUE_E, mode="well"),
        )
        others.set_height(0.8)
        others.arrange(RIGHT, buff=0.2)
        others.next_to(company, DOWN, buff=0.2)

        hearts = SuitSymbol("hearts").replicate(3)
        hearts.set_height(0.3)
        for heart, pi in zip(hearts, others):
            heart.move_to(pi.get_corner(UR))
            heart.shift(0.1 * RIGHT)
        hearts.shuffle()

        self.play(
            LaggedStartMap(FadeIn, others),
            randy.change("coin_flip_1", others),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, hearts, shift=0.25 * UP, lag_ratio=0.25),
            *(pi.animate.look_at(company) for pi in others)
        )
        self.play(Blink(others[1]))
        self.play(Blink(others[2]))
        self.wait()

        # Talking with them
        frame = self.frame

        morty = Mortimer().flip()
        morty.set_height(others.get_height() * 1.1)
        morty.next_to(others, LEFT, LARGE_BUFF, aligned_edge=DOWN)

        self.add(morty)
        self.play(
            FadeOut(randy, time_span=(0, 1)),
            frame.animate.reorient(0, 0, 0, (-6.75, -1.77, 0.0), 3.94),
            FadeOut(arrow),
            morty.change("well", others),
            FadeOut(hearts, lag_ratio=0.5),
            *(pi.animate.look_at(morty.eyes) for pi in others),
            run_time=3
        )
        bubble = morty.get_bubble("...", bubble_type=SpeechBubble)
        self.play(LaggedStart(
            Write(bubble),
            morty.change('speaking', others),
            others[0].change("tease"),
            others[1].change("happy"),
            others[1].change("happy"),
            lag_ratio=0.5
        ))
        self.play(Blink(morty))
        self.wait()

        # Response
        content = Text("...", font_size=12)
        new_bubble = others[0].get_bubble(
            content,
            bubble_type=SpeechBubble,
            direction=RIGHT
        )
        content.scale(3)
        self.play(
            others[0].change("hooray"),
            Write(new_bubble),
            FadeOut(bubble),
            morty.change('tease')
        )
        self.wait()
        self.play(Blink(others[0]))
        self.play(
            others[0].change("well", others[1].eyes),
            others[1].animate.look_at(others[0].eyes),
            FadeOut(new_bubble),
        )
        self.play(Blink(others[1]))
        self.wait()


class CareerFairBooths(InteractiveScene):
    def construct(self):
        # Add booths
        fair_words = Text("Virtual Career Fair", font_size=72)
        fair_words.to_edge(UP)

        booth = SVGMobject('booth')
        booth.set_height(1.35)
        booths = booth.get_grid(3, 2, h_buff=3.5, v_buff=0.75)
        booths.next_to(fair_words, DOWN, buff=LARGE_BUFF)
        for booth in booths:
            booth.set_color(random_bright_color(hue_range=(0.1, 0.2)))

        self.add(fair_words, booths)

        # Show job types
        job_types = BulletedList(
            "Senior roles",
            "New careers",
            "Internships",
            "Part-time",
            buff=0.75
        )
        job_types.move_to(booths).shift(0.5 * UP)

        for n, booth in enumerate(booths):
            booth.generate_target()
            booth.target.shift((-1)**(n + 1) * RIGHT)

        self.play(
            LaggedStartMap(FadeIn, job_types, lag_ratio=0.5, run_time=4),
            LaggedStartMap(MoveToTarget, booths, run_time=4)
        )
        self.wait()



class TalentContactCard(InteractiveScene):
    def construct(self):
        words = VGroup(
            VGroup(
                Text("Interested in joining?"),
                Text("Reach out to explore if your company is a good fit")
            ),
            VGroup(
                Text("Find a job through this page?"),
                Text("Let us know, we’d love to hear your story!")
            ),
        )
        for group in words:
            group[0].scale(1.25)
            group[1].scale(0.75)
            group[1].set_color(GREY_B)
            group.arrange(DOWN, buff=0.5)
        words.arrange(DOWN, buff=1.25)

        address = Text("talent@3blue1brown.com", font_size=48)
        address.set_color(BLUE)
        address.next_to(words, UP, LARGE_BUFF)

        # Test
        self.add(words[0][0])
        self.play(
            FadeIn(words[0][1], lag_ratio=0.1, run_time=1.5),
            FadeIn(address, 0.5 * UP),
        )
        self.play(LaggedStart(
            FadeIn(words[1][0]),
            FadeIn(words[1][1], lag_ratio=0.1, run_time=2)
        ))
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "TalentPartnerProperties extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      10: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      22: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      29: "GrowArrow animates an arrow growing from its start point to full length.",
      31: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      33: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      36: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      55: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      56: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      59: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      60: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      61: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      62: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      66: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      76: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      77: "FadeOut transitions a mobject from opaque to transparent.",
      78: "Smoothly animates the camera to a new orientation over the animation duration.",
      79: "FadeOut transitions a mobject from opaque to transparent.",
      81: "FadeOut transitions a mobject from opaque to transparent.",
      82: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      86: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      87: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      95: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      98: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      105: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      107: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      108: "FadeOut transitions a mobject from opaque to transparent.",
      111: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      113: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      115: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      116: "FadeOut transitions a mobject from opaque to transparent.",
      119: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      122: "CareerFairBooths extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      123: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      159: "TalentContactCard extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      160: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();