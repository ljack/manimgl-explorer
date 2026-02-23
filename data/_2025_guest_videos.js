(function() {
  const files = window.MANIM_DATA.files;

  files["_2025/guest_videos/burnside.py"] = {
    description: "Burnside's lemma animation: counting distinct objects under group symmetry, with colorful examples of necklace counting and rotational equivalence.",
    code: `from manim_imports_ext import *


class IncompleteSquares(InteractiveScene):
    def construct(self):
        # Set up top row
        squares = VGroup(self.get_square(n) for n in range(16))
        squares.arrange(RIGHT, buff=0.5)
        squares.center()
        squares.set_width(FRAME_WIDTH - 1)
        self.add(squares)

        # Reorder by groups
        groups = VGroup(
            VGroup(squares[i] for i in index_groups)
            for index_groups in [
                [0],
                [1, 2, 4, 8],
                [3, 6, 9, 12],
                [5, 10],
                [7, 11, 13, 14],
                [15],
            ]
        )
        groups.target = groups.generate_target()
        for group in groups.target:
            group.arrange(RIGHT, buff=0.25)
        groups.target.arrange(RIGHT, buff=0.5)
        groups.target.set_width(FRAME_WIDTH - 1)

        rects = VGroup(
            SurroundingRectangle(group, buff=0.15).set_stroke(width=3).round_corners()
            for group in groups.target
        )
        rects.set_submobject_colors_by_gradient(RED, YELLOW)

        self.play(
            MoveToTarget(groups, path_arc=45 * DEG, run_time=3),
        )
        self.play(LaggedStartMap(ShowCreation, rects, lag_ratio=0.5))
        self.wait()

        # Count the groups
        ones = VGroup(
            Integer(1).next_to(rect, UP)
            for rect in rects
        )
        plusses = VGroup(
            Tex(R"+").move_to(VGroup(pair))
            for pair in zip(ones, ones[1:])
        )
        brace = Brace(ones, UP)
        six = Integer(6).next_to(brace, UP)
        brace_group = VGroup(brace, six)

        self.play(
            LaggedStartMap(FadeIn, ones, shift=0.25 * UP),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * UP),
        )
        self.wait()
        self.play(
            GrowFromCenter(brace),
            Write(six)
        )
        self.wait()

        # Show the fractions
        all_fractions = VGroup(ones[0])
        fraction_groups = VGroup(ones[:1])
        ones_shift = 0.15 * UP
        for group in groups[1:-1]:
            n = len(group)
            new_fracs = VGroup(
                Tex(Rf"1 \\over {{{n}}}", font_size=36).next_to(square, UP).match_y(ones).shift(ones_shift)
                for square in group
            )
            all_fractions.add(*new_fracs)
            fraction_groups.add(new_fracs)

        all_fractions.add(ones[-1])
        fraction_groups.add(ones[-1:])

        new_plusses = VGroup(
            Tex(R"+", font_size=36).move_to(VGroup(*pair))
            for pair in zip(all_fractions, all_fractions[1:])
        )

        self.play(
            FadeOut(plusses),
            brace_group.animate.next_to(all_fractions, UP, SMALL_BUFF),
            ones.animate.shift(ones_shift)
        )
        for one, frac_group in zip(ones, fraction_groups):
            self.play(ReplacementTransform(one, frac_group, lag_ratio=0.01, run_time=1))

        self.play(FadeIn(new_plusses, lag_ratio=0.1))

        top_sum = VGroup(all_fractions, new_plusses)

        # Show the rotations
        frame = self.frame
        v_line = Line(rects.get_left(), rects.get_right()).next_to(rects, DOWN)
        v_line.set_stroke(WHITE, 2)
        v_line.next_to(rects, DOWN)
        v_line.scale(1.2, about_edge=RIGHT)

        def get_rot_sym(angle, label_tex):
            arcs = VGroup(
                Arc(0, angle),
                Arc(PI, angle)
            )
            for arc in arcs:
                arc.set_stroke(TEAL, 3)
                arc.add_tip()
            arcs.scale(0.45)
            label = Tex(label_tex, font_size=24)
            return VGroup(arcs, label)

        rot_symbols = VGroup(
            Text("Id"),
            get_rot_sym(90 * DEG, R"90^\\circ"),
            get_rot_sym(165 * DEG, R"180^\\circ"),
            get_rot_sym(-90 * DEG, R"-90^\\circ"),
        )
        rot_symbols.arrange(DOWN, buff=0.75)
        rot_symbols.next_to(v_line, DOWN, 0.75)
        rot_symbols.set_x(rects.get_x(LEFT) - rot_symbols.get_width() - MED_LARGE_BUFF)

        self.play(
            ShowCreation(v_line),
            frame.animate.reorient(0, 0, 0, (-1.55, -2.67, 0.0), 10.59),
            FadeIn(rot_symbols)
        )
        self.wait()

        # Create columns
        columns = VGroup()
        squares.sort(lambda p: p[0])  # Sort from left to right
        for square in squares:
            col = VGroup(
                square.copy().rotate(angle)
                for angle in np.arange(0, TAU, TAU / 4)
            )
            col.match_x(square)
            for part, sym in zip(col, rot_symbols):
                part.match_y(sym)

            columns.add(col)

        fixed_points = [col[0] for col in columns]
        fixed_points.extend(columns[0][1:])
        fixed_points.extend(columns[-1][1:])
        fixed_points.extend([
            columns[9][2],
            columns[10][2],
        ])
        fixed_point_dots = Group(GlowDot(radius=0.5).move_to(point) for point in fixed_points)

        # Show example columns
        low_opacity = 0.2
        ex_index = 4
        self.play(
            rects.animate.set_stroke(opacity=low_opacity),
            squares[:ex_index].animate.set_stroke(opacity=low_opacity),
            squares[ex_index + 1:].animate.set_stroke(opacity=low_opacity),
            all_fractions.animate.set_fill(opacity=low_opacity),
            top_sum.animate.set_opacity(low_opacity),
            brace_group.animate.set_opacity(low_opacity)
        )
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(
            squares[1:ex_index].animate.set_stroke(opacity=1),
            rects[1].animate.set_stroke(opacity=1)
        )
        self.wait()

        ex_index = 9
        self.play(squares[ex_index].animate.set_stroke(opacity=1))
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(
            rects[3].animate.set_stroke(opacity=1),
            squares[10].animate.set_stroke(opacity=1),
        )
        self.wait()

        ex_index = 15
        self.play(squares[ex_index].animate.set_stroke(opacity=1))
        for piece in columns[ex_index]:
            self.play(TransformFromCopy(squares[ex_index], piece, path_arc=30 * DEG))
        self.wait()
        self.play(rects[5].animate.set_stroke(opacity=1))

        # Show fixed point dots
        dot = GlowDot(radius=0.5)
        ex_dots1 = Group(dot.copy().move_to(part) for part in columns[9][0::2])
        ex_dots2 = Group(dot.copy().move_to(part) for part in columns[15])
        ex_dots3 = Group(dot.copy().move_to(columns[4][0]))

        for dots in [ex_dots1, ex_dots2, ex_dots3]:
            self.play(FadeIn(dots))
            self.wait()

        # Show fractions
        ex_fractions = VGroup(all_fractions[i] for i in [4, 9, 15])
        ex_dot_groups = Group(ex_dots3, ex_dots1, ex_dots2)

        def get_fourth_exprs(dot_group):
            return VGroup(
                Tex(R"1 / 4", font_size=24).next_to(dot, DOWN, buff=0)
                for dot in dot_group
            )

        ex_fourths = VGroup(
            get_fourth_exprs(dot_group)
            for dot_group in ex_dot_groups
        )

        self.play(ex_fractions.animate.set_opacity(1))
        self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(frac, fourth_group, path_arc=30 * DEG)
                for frac, fourth_group in zip(ex_fractions, ex_fourths)),
                lag_ratio=0.75,
            )
        )
        self.wait()

        # Light everything back up
        self.play(
            brace_group.animate.set_opacity(1),
            top_sum.animate.set_opacity(1),
            rects.animate.set_stroke(opacity=1),
            squares.animate.set_stroke(opacity=1),
        )
        self.wait()

        # Show all transform/shape pairs
        all_fourths = get_fourth_exprs(fixed_point_dots)

        self.play(LaggedStart(
            (ReplacementTransform(square.replicate(4), col, path_arc=30 * DEG)
            for square, col in zip(squares, columns)),
            lag_ratio=0.5,
            run_time=5
        ))
        self.play(
            FadeIn(fixed_point_dots),
            FadeOut(ex_dot_groups),
        )
        self.play(
            FadeIn(all_fourths),
            FadeOut(ex_fourths),
        )
        self.wait()

    def get_square(self, edge_pattern: int = 0, edge_color=BLUE, edge_stroke_width=5):
        outline = VGroup(
            DashedLine(p1, p2, dash_length=0.1).set_stroke(GREY_C, 1)
            for p1, p2 in adjacent_pairs([UL, UR, DR, DL])
        )
        pattern = [(edge_pattern >> (3 - i)) & 1 == 1 for i in range(4)]  # Pattern of bools
        bold_edges = VGroup()

        for edge, include in zip(outline, pattern):
            if include:
                line = Line(edge.get_start(), edge.get_end())
                line.set_stroke(edge_color, edge_stroke_width)
                line.scale(1.075)  # Bad hack for bevel
                bold_edges.add(line)

        return VGroup(outline, bold_edges)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "IncompleteSquares extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      37: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      40: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      41: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      45: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      49: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      53: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      56: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      57: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      58: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      60: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      61: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      63: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      65: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      74: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      84: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      88: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      89: "FadeOut transitions a mobject from opaque to transparent.",
      90: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      91: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      94: "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
      96: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      116: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      120: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      129: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      130: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      131: "Smoothly animates the camera to a new orientation over the animation duration.",
      132: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      134: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      157: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      162: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      163: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      164: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      165: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      166: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      167: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      168: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      171: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
    }
  };

  files["_2025/guest_videos/euclid.py"] = {
    description: "Euclid's Elements animations: visualizing classical geometric constructions, axioms, and proofs from the foundational mathematics text.",
    code: `from manim_imports_ext import *


class FlattenCone(InteractiveScene):
    def construct(self):
        # Add surfaces
        frame = self.frame
        radius = 3.0
        axes = ThreeDAxes()
        axes.set_stroke(WHITE, width=2, opacity=0.5)
        axes.set_anti_alias_width(5)
        axes.apply_depth_test()
        axes.set_z_index(1)
        self.add(axes)

        kw = dict(radius=0.1)
        tracking_dots = Group(
            TrueDot(2 * RIGHT, color=RED, **kw),
            TrueDot(2 * LEFT, color=RED, **kw),
            TrueDot(2 * UP, color=GREEN, **kw),
            TrueDot(2 * DOWN, color=GREEN, **kw),
            TrueDot(2 * IN, color=BLUE, **kw),
            TrueDot(2 * OUT, color=BLUE, **kw),
        )
        for dot in tracking_dots:
            dot.make_3d()
        tracking_dots.set_z_index(2)
        tracking_dots.deactivate_depth_test()
        self.add(tracking_dots)

        theta = TAU * math.sin(TAU / 8)  # Angle off slice of paper

        def cone_func(u, v):
            return np.array([
                u * math.cos(TAU * v),
                u * math.sin(TAU * v),
                0.5 * radius - u
            ])

        def flat_cone_func(u, v):
            return np.array([
                u * math.cos(theta * v + 0.5 * (TAU - theta)),
                u * math.sin(theta * v + 0.5 * (TAU - theta)),
                0,
            ])

        range_kw = dict(
            u_range=(0, radius),
            v_range=(0, 1)
        )
        cone = ParametricSurface(cone_func, **range_kw)
        flat_cone = ParametricSurface(flat_cone_func, **range_kw)

        for surface in [cone, flat_cone]:
            surface.set_color(GREY_D)
            surface.set_shading(0.5, 0.25, 0.5)

        frame.reorient(-25, 69, 0)
        frame.set_x(1e-1)
        self.play(
            frame.animate.reorient(50, 80, 0),
            ShowCreation(cone, time_span=(0, 2)),
            run_time=4
        )

        # Add line
        def get_line(uv_func):
            line = Line().set_stroke(RED, 5)
            line.set_points_as_corners([
                uv_func(radius, 0.75),
                uv_func(0, 0.75),
                uv_func(radius, 0.25),
            ])
            line.apply_depth_test()
            line.shift(1e-2 * OUT)
            return line

        def get_div_line(uv_func):
            line = DashedLine(uv_func(radius, 0.5), uv_func(0, 0)).set_stroke(YELLOW, 4)
            line.apply_depth_test()
            line.shift(1e-2 * OUT)
            return line

        cone_line = get_line(cone_func)
        flat_line = get_line(flat_cone_func)

        cone_div_line = get_div_line(cone_func)
        flat_div_line = get_div_line(flat_cone_func)

        self.play(
            ShowCreation(cone_line, time_span=(0, 3)),
            frame.animate.reorient(0, 2, 0),
            run_time=4,
        )
        self.play(ShowCreation(cone_div_line))
        self.wait()

        # Flatten
        kw = dict(time_span=(1.5, 3))
        self.play(
            Transform(cone, flat_cone, **kw),
            Transform(cone_line, flat_line, **kw),
            Transform(cone_div_line, flat_div_line, **kw),
            frame.animate.reorient(21, 74, 0),
            run_time=3
        )
        self.play(frame.animate.reorient(0, 0, 0), run_time=5)
        self.wait()


class SquareOnASphere(InteractiveScene):
    def construct(self):
        # Add sphere
        frame = self.frame
        self.camera.light_source.set_y(5)
        sphere = Sphere()
        sphere.set_color(GREY_D)
        sphere.set_shading(0.5, 0.25, 0.25)
        mesh = SurfaceMesh(sphere, resolution=(41, 21), normal_nudge=1e-3)
        mesh.set_stroke(WHITE, 1, 0.25)

        frame.reorient(0, 84, 0, ORIGIN, 2.50)
        self.add(sphere, mesh)

        # Show "square" lines
        arc_len = 40 * DEG
        u0 = 270 * DEG
        v0 = 100 * DEG

        line1 = ParametricCurve(lambda t: sphere.uv_func(u0, v0 + arc_len * t))
        line1.set_stroke(RED_D, 3)
        all_lines = VGroup(line1)

        self.play(
            ShowCreation(line1, time_span=(0, 2)),
            frame.animate.reorient(5, 57, 0),
            run_time=3
        )

        orientations = [
            (32, 65, 0),
            (23, 85, 0),
            (6, 84, 0),
        ]

        for orientation in orientations:
            last_line = all_lines[-1]
            elbow = self.get_elbow(last_line)
            new_line = last_line.copy()
            new_line = self.get_rotated_arc(last_line, 90 * DEG)
            new_line.reverse_points()
            self.play(
                ShowCreation(new_line, time_span=(0, 2)),
                ShowCreation(elbow, time_span=(0, 1)),
                frame.animate.reorient(*orientation),
                run_time=3
            )
            all_lines.add(new_line)

        # Show transitions
        for line in all_lines[:3]:
            anim = UpdateFromAlphaFunc(
                line.copy(),
                lambda m, a: m.match_points(self.get_rotated_arc(line, a * 90 * DEG)),
                run_time=3,
                time_span=(0, 2)
            )
            if line is all_lines[0]:
                self.play(
                    anim,
                    frame.animate.reorient(7, 61, 0, (-0.02, -0.01, -0.01), 2.68).set_anim_args(run_time=3)
                )
                frame.add_ambient_rotation(2 * DEG)
            else:
                self.play(anim)
        self.wait(5)

    def get_rotated_arc(self, arc, angle):
        return arc.copy().rotate(angle, about_point=arc.get_end(), axis=arc.get_end())

    def get_elbow(self, arc, prop=0.1):
        corner = arc.get_end()
        rot_arc = self.get_rotated_arc(arc, 90 * DEG)
        v1 = arc.get_points()[-2] - corner
        v2 = rot_arc.get_points()[-2] - corner
        elbow = VMobject()
        elbow.set_points_as_corners([corner + v1, corner + v1 + v2, corner + v2])
        elbow.set_stroke(WHITE, 2)
        return elbow`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "FlattenCone extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      9: "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
      12: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      56: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      58: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      60: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      61: "Smoothly animates the camera to a new orientation over the animation duration.",
      62: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      74: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      80: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      90: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      91: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      92: "Smoothly animates the camera to a new orientation over the animation duration.",
      95: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      96: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      100: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      101: "Transform smoothly morphs one mobject into another by interpolating their points.",
      102: "Transform smoothly morphs one mobject into another by interpolating their points.",
      103: "Transform smoothly morphs one mobject into another by interpolating their points.",
      104: "Smoothly animates the camera to a new orientation over the animation duration.",
      107: "Smoothly animates the camera to a new orientation over the animation duration.",
      108: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      111: "SquareOnASphere extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      112: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      116: "Sphere creates a parametric surface mesh. Can be textured, made transparent, and depth-tested for 3D rendering.",
      118: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      119: "SurfaceMesh draws wireframe grid lines on a Surface for spatial reference.",
      122: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      130: "ParametricCurve traces a function f(t) → (x,y,z) over a parameter range, producing a smooth 3D curve.",
      134: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      135: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      136: "Smoothly animates the camera to a new orientation over the animation duration.",
      152: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      153: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      154: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      155: "Smoothly animates the camera to a new orientation over the animation duration.",
    }
  };

  files["_2025/guest_videos/misc_animations.py"] = {
    description: "Miscellaneous animations produced for various guest collaborations and one-off projects.",
    code: `from manim_imports_ext import *


class WelchLabsIntroCard(InteractiveScene):
    title_words = "This is a guest video by Welch Labs"
    subtitle_words = ""
    logo_file = "WelchLabsLogo"
    banner_file = "WelchLabsBanner"
    bottom_words = "This is part of a series of guest videos, see the end for more details"

    def construct(self):
        # Add title and banners
        group = Group(
            VGroup(
                Text(self.title_words, font_size=48),
                Text(self.subtitle_words, font_size=36, fill_color=GREY_A),
            ).arrange(DOWN),
            ImageMobject(self.banner_file).set_width(0.7 * FRAME_WIDTH),
            Text(self.bottom_words, font_size=36),
        )
        group.arrange(DOWN, buff=MED_LARGE_BUFF)
        group[2].scale(1.1, about_edge=DOWN)
        group[-1].to_edge(DOWN, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            Write(group[0], run_time=1, lag_ratio=5e-2),
            FadeIn(group[1], scale=1.25),
            FadeIn(group[2], lag_ratio=0.01),
            lag_ratio=0.05
        ))
        self.wait(2)
        self.play(
            LaggedStartMap(FadeOut, group, shift=1.0 * DOWN, lag_ratio=0.1)
        )


class Aleph0IntroCard(WelchLabsIntroCard):
    name = "Aleph 0"
    logo_file = "Aleph0Logo"
    banner_file = "Aleph0Banner"
    bottom_words = "This is the 2nd of 5 of guest videos this summer while I'm on leave"


class VilasIntroCard(WelchLabsIntroCard):
    title_words = "This is a guest video by Vilas Winstein"
    subtitle_words = "(a PhD candidate in probability at UC Berkeley)"
    banner_file = "vilas_winstein"
    logo_file = "SpectralCollectiveLogo"
    bottom_words = "This is the 3rd of 5 of guest videos this summer while I am on leave"


class SubManifolds(InteractiveScene):
    def construct(self):
        # Set up spaces
        blob = Circle(radius=1).stretch(2.4, 0, about_edge=LEFT)
        blob.shift(0.5 * RIGHT)
        blob.set_fill(BLUE, 0.5)
        blob.set_stroke(BLUE, 1)
        blobs = VGroup()
        for angle in np.arange(0, TAU, TAU / 5):
            new_blob = blob.copy()
            new_blob.scale(random.uniform(0.5, 1), about_edge=LEFT)
            new_blob.stretch(random.uniform(0.9, 1.3), 1)
            new_blob.rotate(angle + random.uniform(-0.5, 0.5), about_point=ORIGIN)
            new_blob.set_color(random_bright_color())
            blobs.add(new_blob)

        middle = Intersection(*blobs)
        middle.match_style(blob)
        middle.reverse_points()

        big_circle = Circle(radius=3.8)
        big_circle.stretch(1.5, 0)
        big_circle.set_fill(TEAL, 0.2)
        big_circle.set_stroke(TEAL, 1)

        # Label all videos
        all_videos_text = Text("Space of\\nall videos")
        all_videos_text.next_to(big_circle.pfp(0.1), UR, SMALL_BUFF)
        self.add(all_videos_text)
        self.add(big_circle)

        # Show middle blob
        prompt_words = TexText(R"""
            Videos consistent with \\\\
            \`\`An astronaut on the moon\\\\
            riding a horse that turns\\\\
            into a giant cat''
        """, alignment="", font_size=24)
        prompt_words[len("Videos consistent with".replace(" ", "")):].set_color(BLUE_B)
        prompt_words.next_to(big_circle.pfp(3 / 8), DR, SMALL_BUFF)
        prompt_words.set_backstroke(BLACK, 5)
        arrow = Arrow(prompt_words.get_right(), middle.get_top(), buff=0.1, path_arc=-60 * DEG)

        self.play(LaggedStart(
            FadeIn(prompt_words, lag_ratio=0.1),
            Write(arrow),
            TransformFromCopy(big_circle, middle, run_time=2),
        ))
        self.wait()

        # No training data here
        no_training_words = Text("No training\\ndata here", font_size=24)
        no_training_words.next_to(middle, RIGHT, SMALL_BUFF)
        no_training_words.set_z_index(1)
        no_training_words.set_backstroke(BLACK, 2)

        def get_training_data_examples(n_samples, min_scale=0.2):
            training_data = DotCloud(np.array([
                big_circle.pfp(random.random()) * random.uniform(min_scale, 1)
                for n in range(n_samples)
            ]))
            training_data.set_radius(0.04)
            training_data.make_3d()
            training_data.set_z_index(-1)
            return training_data

        training_data = get_training_data_examples(100)

        self.play(
            ShowCreation(training_data, run_time=3),
            Write(no_training_words, run_time=1),
        )
        self.wait()

        # Show other blobs
        blobs.set_fill(opacity=0.25)
        more_data = get_training_data_examples(1000)

        blob_words = VGroup(
            Text(word, font_size=24) for word in [
                "cats",
                "horses",
                "astronauts",
                "transformation",
                "on the\\nmoon",
            ]
        )
        for word, blob in zip(blob_words, blobs):
            word.move_to(blob.get_center() * 1.5)
        blob_words.set_backstroke(BLACK, 2)

        self.play(
            ShowCreation(more_data, run_time=6),
            LaggedStartMap(Write, blobs),
            LaggedStartMap(FadeIn, blob_words),
            no_training_words.animate.scale(0.5).move_to(middle).set_backstroke(BLACK, 0).set_fill(BLACK),
            prompt_words.animate.set_backstroke(BLACK, 10),
        )
        self.wait()


class ComposingFeatures(InteractiveScene):
    def construct(self):
        pass


class IMOGoldOrganizations(InteractiveScene):
    def construct(self):
        # Test
        orgs = VGroup(
            Text("Google DeepMind"),
            Text("OpenAI"),
            Text("Harmonic"),
            Text("ByteDance"),
        )
        orgs.scale(1.5)
        orgs.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        orgs.to_edge(LEFT, buff=1)
        top_brace = Brace(orgs[:2], RIGHT)
        low_brace = Brace(orgs[2:], RIGHT)
        top_text = top_brace.get_text("Natural language", buff=0.5).set_color(BLUE_A)
        low_text = low_brace.get_text("Lean", buff=0.5).set_color(YELLOW_B)

        self.play(LaggedStartMap(FadeIn, orgs, shift=DOWN, lag_ratio=0.15))
        self.wait()
        self.play(LaggedStart(
            GrowFromCenter(top_brace),
            FadeIn(top_text, RIGHT),
            orgs[:2].animate.match_color(top_text),
            GrowFromCenter(low_brace),
            FadeIn(low_text, RIGHT),
            orgs[2:].animate.match_color(low_text),
        ))
        self.wait()


class AlephGeometryEndScreen(PatreonEndScreen):
    pass


class PhaseChangeEndScreen(PatreonEndScreen):
    def construct(self):
        # Attribution
        # background_color = interpolate_color("#FDF6E3", BLACK, 0)
        background_color = BLACK
        text_color = WHITE

        background_rect = FullScreenRectangle(fill_color=background_color)
        background_rect.set_z_index(-2)
        self.add(background_rect)

        elements = VGroup(
            ScreenRectangle(height=3),
            Circle(radius=1),
        )
        elements.arrange(RIGHT, buff=0.5)
        elements.set_width(6.5)
        elements.to_edge(RIGHT, buff=1.0)
        elements.set_y(1)
        elements.set_stroke(text_color, 1)
        # elements.insert_n_curves(100)

        attribution = Text("This guest video was\\nproduced by Vilas Winstein", font_size=48, fill_color=text_color)
        watch = Text("Watch part 2 now on Spectral Collective", font_size=36).set_fill(text_color, 0.95)
        attribution.next_to(elements, DOWN, buff=MED_LARGE_BUFF)
        watch.next_to(elements, UP, buff=MED_SMALL_BUFF)

        self.add(elements, attribution, watch)

        # Patron scroll
        v_line = Line(DOWN, UP).set_height(FRAME_HEIGHT)
        v_line.next_to(elements, LEFT, LARGE_BUFF)
        v_line.set_y(0)
        v_line.set_stroke(text_color, 1)

        thanks = Text(
            "This channel is funded via Patreon\\nSpecial thanks to these supporters",
            alignment="LEFT",
            font_size=32,
            fill_color=text_color
        )
        thanks.move_to(midpoint(v_line.get_center(), LEFT_SIDE))
        thanks.to_edge(UP)

        solid_rect = Square(side_length=8)
        solid_rect.set_fill(background_color, 1).set_stroke(text_color, 1)
        solid_rect.next_to(v_line, LEFT, 0)
        solid_rect.align_to(thanks, DOWN).shift(MED_SMALL_BUFF * DOWN)

        names = VGroup(map(Text, self.get_names()))
        names.scale(0.5)
        for name in names:
            name.set_width(min(name.get_width(), 2.0))
        names.set_fill(text_color)
        names.arrange_in_grid(n_cols=2, aligned_edge=LEFT)
        names.next_to(solid_rect, DOWN, buff=7).to_edge(LEFT)
        names.set_z_index(-1)

        self.add(solid_rect)
        self.add(v_line)
        self.add(thanks)
        self.add(names)
        self.play(
            names.animate.to_edge(DOWN, buff=1.5).set_anim_args(run_time=25, rate_func=linear),
            LaggedStartMap(
                VShowPassingFlash,
                elements.copy().set_stroke(WHITE, 5).insert_n_curves(1000),
                time_width=2,
                run_time=3
            ),
            FadeIn(elements, time_span=(1, 2))
        )


class EuclidEndScreen(SideScrollEndScreen):
    scroll_time = 30


class SeriesOfFiveVideos(InteractiveScene):
    def construct(self):
        # Add images
        self.add(FullScreenRectangle(fill_color=GREY_E))

        pure_images = Group(
            ImageMobject(filename)
            for filename in [
                "diffusion_TN",
                "alpha_geometry_TN",
                "phase_change_TN",
                "incomplete_cubes_TN",
                "euclid_TN",
            ]
        )
        borders = VGroup(SurroundingRectangle(image, buff=0) for image in pure_images)
        borders.set_stroke(WHITE, 3)

        images = Group(
            Group(border, image)
            for border, image in zip(borders, pure_images)
        )

        images.set_width(4)
        images.arrange_in_grid(n_cols=3, buff=1)
        images[3:].match_x(images[:3]).shift(MED_LARGE_BUFF * DOWN)
        images.set_width(FRAME_WIDTH - 1)

        # Add names
        names = VGroup(
            Text(f"Guest video by {text}", font_size=30)
            for text in [
                "Welch Labs",
                "Aleph0",
                "Vilas Winstein",
                "Paul Dancstep",
                "Ben Syversen",
            ]
        )
        for name, image in zip(names, images):
            name.next_to(image, UP, MED_SMALL_BUFF)
            image.add(name)

        # Animate in
        frame = self.frame
        frame.set_height(4).move_to(images[-1])
        self.add(images[-1])
        self.wait()
        self.play(
            frame.animate.to_default_state(),
            LaggedStartMap(FadeIn, images[3::-1], lag_ratio=0.25),
            run_time=3,
        )
        self.wait()

        # Swap out for topics
        titles = VGroup(
            Text("Diffusion models"),
            Text("AlphaGeometry"),
            Text("Statistical Mechanics"),
            Text("Group theory"),
            Text("Euclid"),
        )
        titles.scale(0.8)
        for title, name, image in zip(titles, names, images):
            title.move_to(name, DOWN)
            self.play(
                LaggedStart(FadeIn(title, 0.25 * UP), FadeOut(name, 0.25 * UP), lag_ratio=0.2),
                pure_images.animate.set_opacity(0.25),
                borders.animate.set_stroke(opacity=0.25),
            )
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "WelchLabsIntroCard extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      11: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      15: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      16: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      19: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      25: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      26: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      27: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      28: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      31: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      32: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      33: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      37: "Class Aleph0IntroCard inherits from WelchLabsIntroCard.",
      44: "Class VilasIntroCard inherits from WelchLabsIntroCard.",
      52: "SubManifolds extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      53: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      78: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      93: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      95: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      96: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      97: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      98: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      100: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      153: "ComposingFeatures extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      154: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      158: "IMOGoldOrganizations extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      159: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      188: "Class AlephGeometryEndScreen inherits from PatreonEndScreen.",
      192: "Class PhaseChangeEndScreen inherits from PatreonEndScreen.",
      193: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      266: "Class EuclidEndScreen inherits from SideScrollEndScreen.",
      270: "SeriesOfFiveVideos extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      271: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();