(function() {
  const files = window.MANIM_DATA.files;

  files["_2025/laplace/derivative_supplements.py"] = {
    description: "Additional derivative-related scenes for the Laplace series, covering higher-order derivatives and the general differentiation property.",
    code: `from manim_imports_ext import *
from _2025.laplace.derivatives import tex_to_color


def get_lt_group(src, trg, arrow_length=1.5, arrow_thickness=4, buff=MED_SMALL_BUFF, label_font_size=48):
    arrow = Vector(arrow_length * RIGHT, thickness=arrow_thickness)
    arrow.next_to(src, RIGHT, buff=buff)
    trg.next_to(arrow, RIGHT, buff=buff)

    label = Tex(R"\\mathcal{L}", font_size=label_font_size)
    label.next_to(arrow, UP, buff=SMALL_BUFF)

    return VGroup(src, arrow, label, trg)


class AnnotateIntro(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=1.0)
        left_section = Rectangle(5.25, 1.5).to_edge(LEFT)
        right_section = Rectangle(5, 1.3).next_to(left_section, RIGHT, buff=0)
        left_brace = Brace(left_section, UP)
        right_brace = Brace(right_section, RIGHT)

        q_marks = Tex(R"???", font_size=36)
        length_question = VGroup(Vector(LEFT), q_marks, Vector(RIGHT))
        length_question.arrange(RIGHT)
        length_question.match_width(left_section)
        length_question.move_to(left_brace, DOWN)

        randy.next_to(left_brace, UP, buff=0.25)

        self.play(
            randy.change("maybe", look_at=left_section),
            GrowFromCenter(left_brace)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.change("hesitant", look_at=left_section),
            FadeTransform(left_brace, length_question)
        )
        self.play(Blink(randy))
        self.wait()

        # Move to right section
        arrows = length_question[0::2]
        right_brace.save_state()
        right_brace.rotate(90 * DEG)
        right_brace.replace(arrows, stretch=True)
        right_brace.set_opacity(0)

        kw = dict(path_arc=-30 * DEG, run_time=2)
        self.play(
            Restore(right_brace, **kw),
            arrows.animate.rotate(90 * DEG).set_opacity(0).replace(right_brace.saved_state, stretch=True).set_anim_args(**kw),
            q_marks.animate.next_to(right_brace.saved_state, RIGHT).set_anim_args(**kw),
            randy.change("pondering", right_section),
        )

        h_lines = DashedLine(ORIGIN, 5 * LEFT).replicate(2)
        h_lines.set_stroke(WHITE, 1)
        h_lines[0].next_to(right_brace.get_corner(UL), LEFT, buff=0)
        h_lines[1].next_to(right_brace.get_corner(DL), LEFT, buff=0)
        self.play(*map(ShowCreation, h_lines))


class SimpleEToST(InteractiveScene):
    def construct(self):
        tex = Tex(R"e^{st}", t2c={"s": YELLOW, "t": GREY_A}, font_size=120)
        self.add(tex)


class MovingBrace(InteractiveScene):
    def construct(self):
        squares = Square().replicate(3)
        squares.arrange(RIGHT)
        squares.set_width(FRAME_WIDTH - 1)
        brace = Brace(squares[0], UP)
        self.play(GrowFromCenter(brace))
        self.wait()
        self.play(brace.animate.match_x(squares[1]))
        self.wait()


class TitleCard(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        title = Text("Laplace Transform ", font_size=72)
        tex = Tex(
            R"F({s}) = \\int^\\infty_0 f({t})e^{\\minus{s}{t}} d{t}",
            t2c=tex_to_color,
            font_size=60
        )
        # VGroup(title, tex).arrange(RIGHT, buff=MED_LARGE_BUFF).to_edge(UP)
        title.center().to_edge(UP)

        self.play(Write(title))
        # self.play(LaggedStart(
        #     FadeIn(title, 0.5 * LEFT, lag_ratio=0.01),
        #     FadeIn(tex, lag_ratio=0.1),
        #     lag_ratio=0.2
        # ))
        self.wait()


class QuickRecap(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        morty.to_edge(DOWN, buff=1.0).shift(LEFT)
        self.play(morty.says(Text("Quick Recap", font_size=72), bubble_direction=LEFT))
        self.play(Blink(morty))
        self.wait(2)


class KeyProperties(InteractiveScene):
    def construct(self):
        # Add title
        title = Text("Key Properties", font_size=72)
        title.to_edge(UP, buff=MED_SMALL_BUFF)
        title.set_backstroke(BLACK, 3)
        underline = Underline(title, buff=-0.05)
        underline.scale(1.25)
        self.add(underline, title)

        # Create
        t2c = dict(tex_to_color)
        number_labels = VGroup(Tex(Rf"{n})", font_size=72) for n in range(1, 4))
        number_labels.arrange(DOWN, aligned_edge=LEFT, buff=1.5)
        number_labels.next_to(title, DOWN, LARGE_BUFF)
        number_labels.to_edge(LEFT)

        properties = VGroup(
            get_lt_group(
                Tex(R"e^{a{t}}", t2c=t2c, font_size=60),
                Tex(R"{1 \\over {s} - a}", t2c=t2c)
            ),
            get_lt_group(
                Tex(R"a \\cdot f({t}) + b \\cdot g({t})", t2c=t2c),
                Tex(R"a \\cdot F({s}) + b \\cdot G({s})", t2c=t2c),
            ),
            get_lt_group(
                Tex(R"f'({t})", t2c=t2c),
                Tex(R"{s} F({s}) - f(0)", t2c=t2c),
            ),
        )
        exp_prop, lin_prop, deriv_prop = properties
        properties.scale(1.25)
        for num, prop in zip(number_labels, properties):
            prop.shift(num.get_right() + MED_SMALL_BUFF * RIGHT - prop[0].get_left())
        exp_prop.shift(SMALL_BUFF * UP)

        # Show first properties
        self.play(
            LaggedStartMap(FadeIn, number_labels[:2], shift=UP, lag_ratio=0.25),
            ShowCreation(underline),
        )
        self.wait()

        self.play(Write(exp_prop[0]))
        self.play(LaggedStart(
            GrowArrow(exp_prop[1]),
            FadeIn(exp_prop[2], 0.25 * RIGHT),
            Transform(exp_prop[0]["a"][0].copy(), exp_prop[3]["a"][0].copy(), remover=True),
            Write(exp_prop[3]),
            lag_ratio=0.5
        ))
        self.wait()

        # Show linearity
        f_rects = VGroup(
            SurroundingRectangle(lin_prop[0]["f({t})"], buff=SMALL_BUFF),
            SurroundingRectangle(lin_prop[3]["F({s})"], buff=SMALL_BUFF),
        )
        g_rects = VGroup(
            SurroundingRectangle(lin_prop[0]["g({t})"], buff=SMALL_BUFF),
            SurroundingRectangle(lin_prop[3]["G({s})"], buff=SMALL_BUFF),
        )
        VGroup(f_rects, g_rects).set_stroke(TEAL, 2)

        self.play(
            Write(lin_prop[0])
        )
        self.play(LaggedStart(
            GrowArrow(lin_prop[1]),
            FadeIn(lin_prop[2], 0.25 * RIGHT),
            TransformMatchingTex(
                lin_prop[0].copy(),
                lin_prop[3],
                key_map={"{t}": "{s}", "f": "F", "g": "G"},
                path_arc=45 * DEG,
                lag_ratio=0.01,
            )
        ))
        self.wait()
        self.play(ShowCreation(f_rects, lag_ratio=0))
        self.wait()
        self.play(ReplacementTransform(f_rects, g_rects, lag_ratio=0))
        self.wait()
        self.play(FadeOut(g_rects))

        # (Edited in, show combination transformed)

        # Show third property
        frame = self.frame
        morty = Mortimer().flip()
        morty.next_to(number_labels[2], DR, LARGE_BUFF)

        self.play(
            VFadeIn(morty),
            morty.change("raise_left_hand", number_labels[2]),
            properties[:2].animate.set_fill(opacity=0.5),
            number_labels[:2].animate.set_fill(opacity=0.5),
            Write(number_labels[2]),
            frame.animate.set_height(12, about_edge=UP)
        )
        self.wait()
        self.play(LaggedStart(
            Write(deriv_prop[0]),
            GrowArrow(deriv_prop[1]),
            FadeIn(deriv_prop[2], 0.25 * RIGHT),
            morty.change("pondering", deriv_prop[0]),
        ))
        self.play(Blink(morty))
        self.wait()
        morty.body.insert_n_curves(500)
        self.play(
            Write(deriv_prop[3]),
            morty.change("raise_right_hand", deriv_prop[3])
        )
        self.play(Blink(morty))
        self.wait()

        # TODO


class SimpleLTArrow(InteractiveScene):
    def insertion(self):
        # Test
        group = get_lt_group(VGroup(), VGroup(), arrow_length=3, arrow_thickness=5)
        group.scale(1.5).center()
        self.play(
            GrowArrow(group[1]),
            FadeIn(group[2], RIGHT),
        )
        self.wait()


class CombinationOfExponentials(InteractiveScene):
    def construct(self):
        # Test
        t2c = dict(tex_to_color)
        t2c["c_n"] = TEAL
        kw = dict(t2c=t2c, font_size=72)
        combination = Tex(R"\\sum_n c_n e^{a_n{t}}", **kw)
        result = Tex(R"\\sum_n {c_n \\over {s} - a_n}", **kw)

        group = get_lt_group(
            combination,
            result,
            arrow_length=4,
            label_font_size=60,
            arrow_thickness=6
        )
        group.center()

        self.play(FadeIn(combination, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            GrowArrow(group[1]),
            FadeIn(group[2], RIGHT),
            *(
                Transform(combination[tex][0].copy(), result[tex][0].copy(), remover=True)
                for tex in ["c_n", "a_n", R"\\sum_n"]
            ),
            Write(result[R"\\over {s} - "][0]),
            lag_ratio=0.2,
            run_time=2
        ))
        self.add(group)
        self.wait()


class SimpleFrameForExpDeriv(InteractiveScene):
    def construct(self):
        rect = Rectangle(6, 4)
        rect.set_stroke(BLUE, 3)

        self.play(ShowCreation(rect))
        self.wait()


class MortyReferencingTwoThings(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_edge(DOWN)
        self.play(morty.change("raise_left_hand", 3 * UL))
        self.play(Blink(morty))
        self.play(morty.change("raise_right_hand", 3 * UR))
        for _ in range(2):
            self.play(Blink(morty))
            self.wait(2)
        self.play(morty.change("tease", 3 * UR))
        self.play(Blink(morty))
        self.wait(2)


class StepByStep(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.background.fade(0.5)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("confused", "maybe", "erm", look_at=self.screen)
        )
        self.wait()
        self.play(
            morty.change("tease"),
            self.change_students("pondering", "thinking", "pondering", look_at=morty.eyes)
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(4)


class TryItAsAnExercise(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.body.insert_n_curves(500)
        self.play(morty.says("Try it as\\nan exercise", mode="tease"))
        self.play(Blink(morty))
        self.wait()


class SimpleRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(1.5, 1)
        rect.set_stroke(YELLOW, 3)
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.stretch(0.5, 1, about_edge=DOWN))
        self.wait()


class PolesAtOmegaI(InteractiveScene):
    def construct(self):
        t2c = {"{s}": YELLOW, R"\\omega": PINK}
        pole_words = VGroup(
            Tex(R"\\text{Pole at } {s} = +\\omega i", t2c=t2c),
            Tex(R"\\text{Pole at } {s} = -\\omega i", t2c=t2c),
        )
        pole_words.arrange(DOWN, aligned_edge=LEFT)
        for words in pole_words:
            self.play(Write(words))
            self.wait()


class WhatIsTheAnswer(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[1].says("Okay, okay, okay", mode="sassy"),
            stds[0].change("erm", stds[1].eyes),
            stds[2].change("hesitant", stds[1].eyes),
            morty.change("tease"),
        )
        self.wait(2)
        self.play(
            FadeOut(stds[1].bubble),
            stds[1].says("But...what is\\nthe actual answer?", mode="maybe"),
            morty.change("guilty"),
        )
        self.wait()
        self.play(self.change_students("sassy", "erm", "confused", look_at=self.screen))
        self.wait(3)

        self.wait(2)
        self.play(
            FadeOut(stds[1].bubble),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=4 * UR)
        )
        self.wait(4)


class InvertArrow(InteractiveScene):
    def construct(self):
        # Add arrow
        arrow = Vector(1.5 * RIGHT, thickness=4)
        arrow.set_fill(border_width=2)
        label = Tex(R"\\mathcal{L}", font_size=48)
        label.next_to(arrow, UP, SMALL_BUFF)
        inv_label = Tex(R"\\mathcal{L}^{-1}")
        inv_label.next_to(arrow, DOWN, buff=0)

        self.play(
            GrowArrow(arrow),
            FadeIn(label, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            Rotate(arrow, PI),
            ReplacementTransform(label, inv_label)
        )
        self.wait()


class ReferenceHomework(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("tease", "pondering", "thinking", look_at=self.screen)
        )
        self.wait(3)
        self.play(morty.change("tease"))
        self.play(self.change_students("pondering", "hesitant", "pondering", look_at=self.screen))
        self.wait(4)


class ODEToAlgebra(InteractiveScene):
    def construct(self):
        # Test
        arrow_config = dict(arrow_length=2.5, buff=0.75)
        ode_group = get_lt_group(
            Text("Differential\\nEquation"),
            Text("Algebra"),
            **arrow_config
        )
        ode_group.center()

        deriv_group = get_lt_group(
            Tex("d / dt", t2c={"t": BLUE}, font_size=72),
            Tex(R"\\times s", t2c={"s": YELLOW}, font_size=72),
            **arrow_config
        )
        deriv_group.next_to(ode_group, DOWN, LARGE_BUFF)

        self.animate_group(ode_group)
        self.wait()
        self.animate_group(deriv_group)
        self.wait()

    def animate_group(self, group):
        src, arrow, label, trg = group
        self.play(FadeIn(src))
        self.play(LaggedStart(
            GrowArrow(arrow),
            FadeIn(label, 0.5 * arrow.get_vector()),
            FadeIn(trg, arrow.get_vector())
        ))


class ThreeExplanations(InteractiveScene):
    def construct(self):
        # Add terms
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        title = Tex(R"\\text{Why } \\mathcal{L}\\big\\{f'({t})\\big\\} = {s} F({s}) - f(0)", t2c=t2c)
        title.to_edge(UP)
        title_underline = Underline(title)

        num_mobs = VGroup(Text(f"{n}) ") for n in range(1, 4))
        num_mobs.scale(1.25)
        num_mobs.arrange(DOWN, buff=1.25, aligned_edge=LEFT)
        num_mobs.next_to(title, DOWN, buff=1.25).to_edge(LEFT, buff=LARGE_BUFF)

        by_parts = Tex(R"\\text{Integration by parts}", t2c=t2c)
        inversion = Tex(R"\\frac{d}{d{t}} \\Big(\\text{Inverse Laplace Transform}\\Big)", t2c=t2c)

        descriptors = VGroup(
            Text("[Elementary, but limited]"),
            Text("[General, but opaque]"),
            Text("[My favorite, but theoretical]"),
        )
        descriptors.set_fill(GREY_B)
        explanations = VGroup(
            Tex(R"\\text{Start with } e^{a{t}}", t2c=t2c),
            by_parts,
            inversion,
        )

        for num, desc, expl in zip(num_mobs, descriptors, explanations):
            for mob in [desc, expl]:
                mob.next_to(num, RIGHT, buff=MED_LARGE_BUFF)
        explanations[0].shift(SMALL_BUFF * UP)

        self.add(title)
        self.play(ShowCreation(title_underline))
        self.wait()
        self.play(LaggedStartMap(FadeIn, num_mobs, shift=UP, lag_ratio=0.25))
        self.wait()
        for desc in descriptors:
            self.play(FadeIn(desc, lag_ratio=0.2))
            self.wait()

        # Show explanations
        for i, desc, expl in zip(it.count(), descriptors, explanations):
            self.play(
                FadeOut(desc, 0.5 * UP),
                FadeIn(expl, 0.5 * UP),
                num_mobs[:i].animate.set_opacity(0.25),
                explanations[:i].animate.set_opacity(0.25),
            )
            self.wait()

        # Show all of them
        num_mobs.set_fill(opacity=1)
        explanations.set_fill(opacity=1)
        self.play(
            LaggedStartMap(FadeIn, num_mobs, shift=UP, lag_ratio=0.25),
            LaggedStartMap(FadeIn, explanations, shift=UP, lag_ratio=0.25),
        )
        self.wait()
        fade_group = VGroup(title, title_underline, *num_mobs, *explanations)
        fade_group.sort(lambda p: np.dot(p, DOWN + 0.1 * RIGHT))
        self.play(LaggedStartMap(FadeOut, fade_group, shift=LEFT, lag_ratio=0.15, run_time=2))
        self.wait()


class ComplainAboutSpecificity(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[1].says("But, that’s just\\none example!", mode="angry"),
            stds[0].change("pondering", self.screen),
            stds[2].change("erm", self.screen),
            morty.change("guilty")
        )
        self.wait(3)
        self.play(
            stds[1].debubble("pondering"),
            morty.says("One with the\\nseeds of generality"),
            stds[2].change("thinking")
        )
        self.wait(3)


class ExplanationOneTitle(InteractiveScene):
    def construct(self):
        t2c = dict(tex_to_color)
        title = VGroup(
            Text("Explanation #1: Try it for"),
            Tex(R"e^{a{t}}", t2c=t2c)
        )
        title.arrange(RIGHT)
        title[1].shift(SMALL_BUFF * UP)
        title.to_corner(UL)
        alt_formula = Tex(R"c_1 e^{a_1 {t}} + c_2 e^{a_2 {t}} + \\cdots + c_n e^{a_n {t}}", t2c=t2c)
        alt_formula[re.compile(r"c_.")].set_color(TEAL_D)
        alt_formula.move_to(title[1], LEFT)
        ponder_words = Text("(Pause and ponder to taste)")
        ponder_words.set_fill(GREY_B)
        ponder_words.move_to(title)
        ponder_words.to_edge(RIGHT, buff=SMALL_BUFF)

        self.add(title)
        self.play(Write(ponder_words))
        self.wait()
        self.play(
            TransformMatchingTex(title[1], alt_formula),
            FadeOut(ponder_words, lag_ratio=0.1),
        )
        self.wait()


class IntuitionEvaporating(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        words = Text("Intuition", font_size=36).replicate(12)
        words.move_to(stds.get_top() + 0.5 * DOWN)
        for word in words:
            word.shift(random.uniform(-4, 4) * RIGHT)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        y0 = words.get_y()
        self.play(
            self.change_students("tired", "tired", "tired", look_at=self.screen),
            morty.change("guilty"),
            LaggedStart(
                *(
                    UpdateFromAlphaFunc(word, lambda m, a: m.set_y(y0 + 2 * a).set_fill(opacity=there_and_back(a)))
                    for word in words
                ),
                lag_ratio=0.1,
                run_time=6
            ),
        )


class ContourIntegralReference(InteractiveScene):
    def construct(self):
        # Test
        inv_words = Text("Inverse Laplace Transfor")
        inv_rect = SurroundingRectangle(inv_words)
        inv_rect.set_fill(BLACK, 1)
        inv_rect.set_stroke(width=0)

        contour_words = Text("Contour Integral")
        contour_words.move_to(inv_rect)

        integral = Tex(R"\\int_\\gamma")
        integral.next_to(contour_words, UP, MED_LARGE_BUFF)
        integral.shift(0.25 * RIGHT)
        integral_rect = SurroundingRectangle(integral)
        integral_rect.set_stroke(YELLOW, 2)

        self.add(inv_rect)
        self.play(
            Write(contour_words),
            ShowCreation(integral_rect),
            VShowPassingFlash(integral_rect.copy().insert_n_curves(100).set_stroke(YELLOW, 3)),
            run_time=2
        )
        self.wait()


class SimpleBigRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(4, 5)
        rect.set_stroke(YELLOW, 3)
        self.play(ShowCreation(rect))
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports tex_to_color from the _2025.laplace.derivatives module within the 3b1b videos codebase.",
      10: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      16: "AnnotateIntro extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      17: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      25: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      33: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      38: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      39: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      44: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      48: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      54: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      55: "Restore animates a mobject back to a previously saved state (from save_state()).",
      56: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      57: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      68: "SimpleEToST extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      69: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      74: "MovingBrace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      75: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      86: "TitleCard extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      87: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      108: "QuickRecap extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      109: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      118: "KeyProperties extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      119: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      239: "SimpleLTArrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      251: "CombinationOfExponentials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      252: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      286: "SimpleFrameForExpDeriv extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      287: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      295: "MortyReferencingTwoThings extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      296: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      310: "Class StepByStep inherits from TeacherStudentsScene.",
      311: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      333: "TryItAsAnExercise extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      334: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      342: "SimpleRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      343: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      352: "PolesAtOmegaI extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      353: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      365: "Class WhatIsTheAnswer inherits from TeacherStudentsScene.",
      366: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      396: "InvertArrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      397: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      418: "Class ReferenceHomework inherits from TeacherStudentsScene.",
      419: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      432: "ODEToAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      433: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      465: "ThreeExplanations extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      466: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      531: "Class ComplainAboutSpecificity inherits from TeacherStudentsScene.",
      532: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      552: "ExplanationOneTitle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      553: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      580: "Class IntuitionEvaporating inherits from TeacherStudentsScene.",
      581: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      613: "ContourIntegralReference extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      614: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      640: "SimpleBigRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      641: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/laplace/derivatives.py"] = {
    description: "Visualizes derivatives in the context of Laplace transforms. Shows how differentiation in the time domain corresponds to multiplication by s in the frequency domain.",
    code: `from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph


tex_to_color = {
    "{t}": BLUE,
    "{s}": YELLOW,
}


class ForcedOscillatorEquation(InteractiveScene):
    def construct(self):
        x_colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            "x(t)": x_colors[0],
            "x'(t)": x_colors[1],
            "x''(t)": x_colors[2],
            "x_0": x_colors[0],
            "v_0": x_colors[1],
            R"\\omega": PINK,
            "{s}": YELLOW,
        }
        ode = Tex(R"m x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega t)", t2c=t2c)
        ode.to_edge(UP)

        self.add(ode)

        # Comment on third force
        rect = SurroundingRectangle(ode[R"F_0 \\cos(\\omega t)"], buff=0.2)
        rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(ode[R"\\omega"], buff=0.1))
        self.wait()
        self.play(FadeOut(rect))
        self.wait()


class MoveAroundPolesSeeDynamics(InteractiveScene):
    def construct(self):
        # Add s_plane (and graph)
        frame = self.frame
        s_plane = ComplexPlane((-3, 3), (-3, 3))
        s_plane.add_coordinate_labels(font_size=20)

        s_trackers = Group(
            ComplexValueTracker(+2j),
            ComplexValueTracker(-2j),
            ComplexValueTracker(+1.0j - 0.2),
            ComplexValueTracker(-1.0j - 0.2),
            ComplexValueTracker(-1.0),
        )

        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        get_t = t_tracker.get_value

        s_points = Group(TrueDot(), GlowDot()).replicate(len(s_trackers))
        s_points.set_color(YELLOW)
        for point, s_tracker in zip(s_points, s_trackers):
            point.s_tracker = s_tracker
            point.add_updater(lambda m: m.move_to(s_plane.n2p(m.s_tracker.get_value())))

        frame.reorient(-14, 82, 0, (-4.89, 2.68, 3.02), 10.32)
        self.add(t_tracker)

        # Add title
        arrow = Vector(3 * RIGHT, thickness=6)
        kw = dict(
            t2c={"t": BLUE, "s": YELLOW},
            font_size=72
        )
        title = VGroup(Tex(R"f(t)", **kw), arrow, Tex(R"F(s)", **kw))
        title.arrange(RIGHT)
        title.to_edge(UP)
        lt_label = Tex(R"\\mathcal{L}")
        lt_label.next_to(arrow, UP, buff=0)
        arrow.add(lt_label)
        title.fix_in_frame()

        self.add(title[0])

        # Add time graph
        axes = Axes((0, 15), (-2, 2), width=6, height=3)
        axes.to_edge(LEFT)
        axes.shift(2 * DOWN)
        time_graph = axes.get_graph(self.get_time_func(s_trackers, amp=0.4))
        time_graph.set_stroke(BLUE, 2)
        axes.fix_in_frame()
        time_graph.fix_in_frame()

        self.add(axes, time_graph)

        # Add mass
        amp_factor_tracker = ValueTracker(2.0)

        number_line = NumberLine((-2, 2), width=6)
        number_line.next_to(axes, UP, LARGE_BUFF)
        number_line.add_numbers(font_size=16)
        number_line.fix_in_frame()

        tip = ArrowTip(angle=-90 * DEG)
        tip.stretch(0.25, 0)
        tip.func = self.get_time_func(s_trackers, amp=0.4)
        tip.add_updater(lambda m: m.move_to(number_line.n2p(m.func(get_t())), DOWN))
        tip.fix_in_frame()

        mass = Square(side_length=0.5)
        mass.set_fill(BLUE_E, 1)
        mass.set_stroke(WHITE, 2)
        mass.always.next_to(tip, UP)
        mass.fix_in_frame()
        mass_ghost = mass.copy()
        mass_ghost.set_opacity(0)

        v_vect = always_redraw(lambda: Arrow(
            mass.get_center(),
            mass.get_center() + np.clip(5 * (mass.get_center() - mass_ghost.get_center()), -1, 1),
            buff=0,
            thickness=4,
            fill_color=RED
        ).fix_in_frame())

        self.add(number_line, tip, mass, v_vect, mass_ghost)
        self.wait(5)

        # Show graph
        epsilon = 1e-5

        def get_graph():
            graph = get_complex_graph(
                s_plane,
                self.get_s_func(s_trackers),
                # resolution=(101, 101)
                # resolution=(31, 31)
            )
            for line in graph[1]:
                if line.get_z(OUT) > 1e3:
                    line.set_stroke(opacity=0)
            return graph

        def update_graph(graph):
            graph.become(get_graph())

        graph = get_graph()

        self.play(
            FadeIn(s_plane, lag_ratio=0.01),
            FadeIn(s_points),
            FadeIn(graph[0]),
            Write(graph[1]),
            Write(title[1]),
            FadeIn(title[2], RIGHT)
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (-5.0, 1.08, 0.0), 9.62).set_field_of_view(10 * DEG),
            run_time=3
        )
        self.wait(8)

        # Move s
        t_tracker.set_value(0)
        s_trackers.remove(*s_trackers[2:])
        s_trackers[1].add_updater(lambda m: m.set_value(s_trackers[0].get_value().conjugate()))
        s_tracker = s_trackers[0]
        s_tracker.set_value(2j)

        amp_tracker = ValueTracker(1)

        time_graph.add_updater(lambda m: m.match_points(axes.get_graph(self.get_time_func(s_trackers))))
        time_graph.set_clip_plane(DOWN, 0)

        def new_func(t):
            return amp_tracker.get_value() * math.cos(2 * t)

        tip.func = new_func

        update_graph(graph)
        self.add(*s_trackers)
        self.remove(s_points[2:])
        self.wait(10)

        amp_tracker.add_updater(lambda m: m.set_value(0.992 * m.get_value()))
        self.add(amp_tracker)
        self.play(
            s_tracker.animate.set_value(-0.5 + 2j),
            UpdateFromFunc(graph, update_graph),
            run_time=2
        )
        self.wait(4)

        amp_tracker.clear_updaters()
        amp_tracker.add_updater(lambda m: m.set_value(1.005 * m.get_value()))
        self.play(
            s_tracker.animate.set_value(0.2 + 2j),
            UpdateFromFunc(graph, update_graph),
            run_time=3
        )
        self.wait(15)

    def get_time_func(self, s_trackers, amp=0.5):
        def time_func(t):
            return amp * np.sum([np.exp(st.get_value() * t) for st in s_trackers]).real

        return time_func

    def get_s_func(self, s_trackers, epsilon=1e-5, amp=0.5):
        def s_func(s):
            s += epsilon
            return amp * np.sum([1.0 / (s - st.get_value() + epsilon) for st in s_trackers])

        return s_func


class DerivativeFormula(InteractiveScene):
    tex_config = dict(t2c=tex_to_color, font_size=72)

    def construct(self):
        # Set up commutative diagram
        kw = self.tex_config
        ft, Fs, dft, sFs = terms = VGroup(
            Tex(R"f({t})", **kw),
            Tex(R"F({s})", **kw),
            Tex(R"f'({t})", **kw),
            Tex(R"{s}F({s}) - f(0)", **kw),
        )
        terms.arrange_in_grid(
            h_buff=2.0,
            v_buff=3.0,
            aligned_edge=LEFT,
            fill_rows_first=False
        )
        terms.to_edge(UP, buff=LARGE_BUFF)
        terms.shift(RIGHT)

        dist = get_norm(ft.get_bottom() - Fs.get_top()) - MED_LARGE_BUFF
        down_arrow = Vector(dist * DOWN, thickness=6)
        arrow_kw = dict(thickness=6)
        lt_arrows = VGroup(
            down_arrow.copy().next_to(term, DOWN)
            for term in [ft, dft]
        )
        lt_arrows.set_fill(GREY_A)
        for arrow in lt_arrows:
            self.add_arrow_label(arrow, R"\\mathcal{L}", RIGHT)

        deriv_arrow = Arrow(ft, dft, thickness=6)
        s_mult_arrow = Arrow(Fs, sFs, thickness=6)

        self.add_arrow_label(deriv_arrow, R"d / d{t}", UP)
        self.add_arrow_label(s_mult_arrow, R"\\times {s}", UP)

        # Add terms
        L_df = Tex(R"\\mathcal{L}\\big\\{f'({t})\\big\\}", **kw)
        L_df.move_to(sFs, LEFT)
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(sFs["{s}F({s})"], DOWN)

        self.play(Write(ft), run_time=1)
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(deriv_arrow),
            TransformMatchingTex(ft.copy(), dft, run_time=1, path_arc=45 * DEG),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(lt_arrows[1]),
            TransformMatchingTex(
                dft.copy(),
                L_df,
                run_time=1.5,
                path_arc=45 * DEG,
                matched_keys=[R"f'({t})"]
            ),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(lt_arrows[0]),
            TransformFromCopy(ft.copy(), Fs, run_time=1.5, path_arc=45 * DEG),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            self.grow_arrow(s_mult_arrow),
            FadeTransform(Fs.copy(), sFs[1:5]),
            TransformFromCopy(Fs[2], sFs[0]),
            L_df.animate.scale(0.7).next_to(equals, DOWN),
            Write(equals),
            lag_ratio=0.1
        ))
        self.wait()

        # Correction
        almost = Text("Almost...")
        almost.set_color(RED)
        almost.next_to(equals, RIGHT, MED_SMALL_BUFF)

        self.play(FadeIn(almost, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(ft, sFs["f(0)"][0], path_arc=-45 * DEG, run_time=2),
            Write(sFs["-"]),
        ))
        self.play(
            FadeOut(almost),
            VGroup(equals, L_df).animate.space_out_submobjects(1.2).next_to(sFs, DOWN)
        )
        self.wait()

        # Highlight parts
        rect = SurroundingRectangle(deriv_arrow.label)
        rect.set_stroke(TEAL, 4)
        mid_lt_arrow = lt_arrows[0].copy().match_x(deriv_arrow)
        self.add_arrow_label(mid_lt_arrow, R"\\mathcal{L}", RIGHT)
        mid_arrow_group = VGroup(mid_lt_arrow, mid_lt_arrow.label)
        mid_arrow_group.shift(0.25 * UP)
        mid_arrow_group.set_fill(opacity=0.5)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            rect.animate.surround(s_mult_arrow.label),
            self.grow_arrow(mid_lt_arrow),
        )
        self.play(
            FadeOut(mid_arrow_group),
            FadeOut(rect),
        )
        self.wait()

        # Comment on -f(0)
        frame = self.frame
        randy = Randolph().flip()
        randy.next_to(sFs, DR, LARGE_BUFF)
        randy.shift(0.5 * LEFT)
        morty = Mortimer().flip()
        morty.next_to(randy, LEFT, buff=2.0)
        morty.body.insert_n_curves(500)
        quirk = sFs["- f(0)"][0]
        quirk_rect = SurroundingRectangle(quirk)
        quirk_rect.set_stroke(RED, 5)
        ic_words = Text("Initial\\nCondition", font_size=72)
        ic_words.next_to(quirk, UR, LARGE_BUFF)
        ic_arrow = Arrow(ic_words.get_bottom(), quirk.get_right(), path_arc=-90 * DEG, thickness=6)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (2.0, -0.79, 0.0), 9.49),
            FadeOut(equals),
            FadeOut(L_df),
            ShowCreation(quirk_rect),
            VFadeIn(randy),
            randy.change("angry", quirk),
            lag_ratio=0.1
        ))
        self.play(Blink(randy))
        self.wait()
        self.play(
            VFadeIn(morty),
            morty.change("tease", randy.eyes),
            randy.change('hesitant', morty.eyes),
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("raise_right_hand", ic_words),
            randy.change("pondering", ic_words),
            FadeIn(ic_words, lag_ratio=0.1, run_time=1),
            Write(ic_arrow)
        )
        self.wait()

        # Ask why
        bubble = randy.get_bubble("Why?", SpeechBubble, direction=LEFT)
        bubble.shift(0.5 * LEFT)

        exp_propety_frame = Rectangle(16, 9).replace(terms, 1)
        exp_propety_frame.next_to(terms, RIGHT, buff=1.0)
        exp_propety_frame.set_stroke(BLUE, 0)

        self.play(LaggedStart(
            randy.change("maybe"),
            Write(bubble),
            morty.change("thinking", lt_arrows),
            FadeOut(ic_words),
            FadeOut(ic_arrow),
            FadeOut(quirk_rect),
            lag_ratio=0.1
        ))
        self.play(Blink(randy))
        self.play(
            Group(frame, randy, morty, bubble).animate.scale(1.25, about_edge=UL),
        )
        self.play(
            randy.change("raise_left_hand", exp_propety_frame),
            morty.change('pondering', exp_propety_frame),
            FadeIn(exp_propety_frame),
            FadeOut(bubble),
        )
        self.wait()
        self.play(
            ShowCreationThenFadeOut(quirk_rect),
            morty.animate.look_at(quirk),
            randy.change('sassy', quirk),
        )
        self.play(Blink(randy))
        self.play(Blink(morty))
        self.wait()

        # Reset
        self.play(FadeOut(exp_propety_frame))
        self.play(
            frame.animate.reorient(0, 0, 0, (-1.0, -0.61, 0.0), 9.39),
            LaggedStartMap(FadeOut, VGroup(morty, randy), shift=DOWN),
            run_time=2
        )

        # Substitute in e^(at)
        fade_group = VGroup(
            deriv_arrow, deriv_arrow.label,
            s_mult_arrow, s_mult_arrow.label,
            lt_arrows[1], lt_arrows[1].label,
            dft,
            sFs
        )

        eat_terms = VGroup(
            Tex(R"f({t}) = e^{a{t}}", **kw),
            Tex(R"F({s}) = {1 \\over {s} - a}", **kw),
            Tex(R"f'({t}) = a \\cdot e^{a{t}}", **kw),
            Tex(R"{a \\over {s} - a}", **kw),
        )
        for term, eat_term, corner in zip(terms, eat_terms, [DR, UR, DL, UL]):
            eat_term.move_to(term, corner)
        eat_terms[3].align_to(eat_terms[1], DOWN)

        self.remove(ft)
        self.play(
            TransformFromCopy(ft, eat_terms[0][:len(ft)]),
            Write(eat_terms[0][len(ft):]),
            s_mult_arrow.animate.scale(0.8, about_edge=UP),
            fade_group.animate.set_fill(opacity=0.2),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(
                eat_terms[0][len(ft):].copy(),
                eat_terms[1][len(Fs):],
                path_arc=-45 * DEG,
                run_time=1.5,
            ),
            FadeTransform(Fs, eat_terms[1][:len(Fs)]),
            VGroup(s_mult_arrow, s_mult_arrow.label).animate.shift(0.4 * DOWN)
        )
        self.add(eat_terms[1])
        self.wait()
        self.play(
            VGroup(deriv_arrow, deriv_arrow.label, dft).animate.set_fill(opacity=1)
        )
        self.wait()
        self.remove(dft)
        self.play(
            TransformFromCopy(eat_terms[0][-3:], eat_terms[2][-3:], path_arc=-45 * DEG, run_time=1.5),
            TransformFromCopy(eat_terms[0][-4], eat_terms[2][-6], path_arc=-45 * DEG, run_time=1.5),
            TransformFromCopy(dft, eat_terms[2][:len(dft)])
        )
        self.play(
            Write(eat_terms[2][-4]),
            TransformFromCopy(eat_terms[2][-2], eat_terms[2][-5], path_arc=45 * DEG),
        )
        self.wait()

        # Show transform of right hand side
        left_group_copy = VGroup(
            eat_terms[0]["e^{a{t}}"][0],
            lt_arrows[0],
            lt_arrows[0].label,
            eat_terms[1][R"{1 \\over {s} - a}"][0]
        ).copy()
        a_dot_copy = eat_terms[2][R"a \\cdot"][0].copy()
        a_dot_rect = SurroundingRectangle(a_dot_copy).set_stroke(TEAL, 2)

        shift_value = eat_terms[2]["e^{a{t}}"].get_center() - left_group_copy[0].get_center()
        self.play(
            left_group_copy.animate.shift(shift_value).set_anim_args(path_arc=30 * DEG),
            sFs.animate.shift(3.0 * DOWN),
            run_time=1.5
        )
        self.play(ShowCreation(a_dot_rect))
        self.play(
            a_dot_copy.animate.next_to(left_group_copy[-1][1], LEFT),
            MaintainPositionRelativeTo(a_dot_rect, a_dot_copy),
        )
        self.wait()
        self.play(
            FadeOut(left_group_copy[:3]),
            ReplacementTransform(left_group_copy[3][1:], eat_terms[3][1:]),
            ReplacementTransform(a_dot_copy, eat_terms[3][:1]),
            FadeOut(left_group_copy[3][0], LEFT),
            FadeOut(a_dot_rect, LEFT),
            VGroup(lt_arrows[1], lt_arrows[1].label).animate.set_fill(opacity=1),
        )
        self.wait()

        # Differentiation to multiplication
        randy = Randolph(height=3)
        randy.next_to(eat_terms[1], DOWN, buff=LARGE_BUFF)
        randy.shift(2 * LEFT)
        mult_a_arrow = Arrow(
            eat_terms[1][-5:].get_bottom(),
            eat_terms[3].get_bottom(),
            path_arc=120 * DEG,
            thickness=6
        )
        rect = SurroundingRectangle(deriv_arrow.label)
        rect.set_stroke(TEAL, 4)
        mult_word = Text("Multiplication")
        times_a = Tex(R"\\times a", **kw)
        for mob in [mult_word, times_a]:
            mob.next_to(mult_a_arrow, DOWN)

        self.play(
            frame.animate.reorient(0, 0, 0, (-1.1, -1.43, 0.0), 11.38),
            VFadeIn(randy),
            randy.change("shruggie", eat_terms[3]),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(ShowCreation(rect))
        self.play(LaggedStart(
            randy.change("pondering", mult_word),
            rect.animate.surround(mult_word),
            Write(mult_word),
            Write(mult_a_arrow),
            lag_ratio=0.2
        ))
        self.play(FadeOut(rect))
        self.wait()
        self.play(randy.change("erm", mult_word))
        self.play(FadeTransformPieces(mult_word, times_a))
        self.play(Blink(randy))
        self.wait()

        # Contrast against multiplication by s
        equals.next_to(eat_terms[3], DOWN, MED_LARGE_BUFF)
        q_marks = Tex(R"???", font_size=60).replicate(2)
        q_marks.set_color(RED)
        q_marks[0].next_to(s_mult_arrow.label, UP)
        q_marks[1].next_to(equals, RIGHT)

        self.play(LaggedStart(
            randy.animate.change_mode("horrified").shift(0.25 * DL).set_opacity(0),
            # FadeOut(mult_a_arrow),
            ReplacementTransform(mult_a_arrow, s_mult_arrow),
            # FadeOut(times_a),
            ReplacementTransform(times_a, s_mult_arrow.label),
            Animation(Point()),
            VGroup(s_mult_arrow, s_mult_arrow.label).animate.set_fill(opacity=1),
            Write(equals),
            sFs.animate.set_fill(opacity=1).next_to(equals, DOWN, MED_LARGE_BUFF),
            Write(q_marks),
        ))
        self.remove(randy)
        self.wait()

        # Show algebra
        added_frac = Tex(R"+ {{s} - a \\over {s} - a}", **kw)
        minus_one = Tex(R"-1", **kw)
        added_frac.next_to(eat_terms[3], RIGHT, SMALL_BUFF)
        minus_one.next_to(added_frac[R"\\over"], RIGHT, MED_SMALL_BUFF)

        added_frac_rect = SurroundingRectangle(added_frac, SMALL_BUFF)
        added_frac_rect.set_stroke(BLUE, 1)

        plus_one = Tex(R"+ 1", font_size=60)
        plus_one.set_color(BLUE)
        plus_one.next_to(added_frac_rect, DOWN)

        cover_rect = BackgroundRectangle(VGroup(equals, sFs), buff=MED_SMALL_BUFF)
        cover_rect.set_fill(BLACK, 0.8)

        combined_fraction = Tex(R"{a + {s} - a \\over {s} - a}", **kw)
        clean_combined_fraction = Tex(R"{{s} \\over {s} - a}", **kw)
        for mob in [combined_fraction, clean_combined_fraction]:
            mob.move_to(eat_terms[3], LEFT)

        self.play(
            FadeIn(cover_rect),
            FadeIn(added_frac_rect),
            Write(added_frac["+"][0]),
            *(
                TransformFromCopy(eat_terms[3][tex], added_frac[tex])
                for tex in [R"{s} - a", R"\\over"]
            ),
        )
        self.play(Write(plus_one))
        self.wait()
        self.play(Write(minus_one))
        self.wait()
        self.remove(eat_terms[3], added_frac)
        self.play(
            TransformFromCopy(eat_terms[3]["a"][0], combined_fraction["a"][0]),
            TransformFromCopy(added_frac["+"][0], combined_fraction["+"][0]),
            TransformFromCopy(added_frac["{s} - a"][0], combined_fraction["{s} - a"][0]),
            TransformFromCopy(eat_terms[3][R"\\over"][0], combined_fraction[R"\\over"][0]),
            TransformFromCopy(added_frac[R"\\over"][0], combined_fraction[R"\\over"][0]),
            TransformFromCopy(eat_terms[3][R"{s} - a"][0], combined_fraction[R"{s} - a"][1]),
            TransformFromCopy(added_frac[R"{s} - a"][1], combined_fraction[R"{s} - a"][1]),
            added_frac_rect.animate.surround(combined_fraction),
            minus_one.animate.next_to(combined_fraction, RIGHT),
            FadeOut(plus_one),
            run_time=1.5
        )
        self.wait()
        self.play(
            TransformMatchingTex(combined_fraction, clean_combined_fraction, run_time=1),
            added_frac_rect.animate.surround(clean_combined_fraction),
            minus_one.animate.next_to(clean_combined_fraction, RIGHT)
        )
        self.wait()

        # Emphasize how this matches the rule
        pole = eat_terms[1][-5:]
        pole_rect = SurroundingRectangle(pole)
        pole_rect.match_style(added_frac_rect)

        self.play(
            FadeOut(added_frac_rect),
            FadeOut(q_marks[0], shift=0.25 * RIGHT, lag_ratio=0.1),
            ShowCreation(pole_rect),
        )
        self.play(
            pole_rect.animate.surround(clean_combined_fraction, SMALL_BUFF),
            TransformFromCopy(pole[1:], clean_combined_fraction[1:], path_arc=10 * DEG),
            TransformFromCopy(s_mult_arrow.label[1], clean_combined_fraction[0], path_arc=-10 * DEG),
            run_time=2
        )
        self.play(FadeOut(pole_rect, run_time=0.5))
        self.wait(0.5)

        # Emphasize minus f(0)
        low_eq_group = VGroup(equals, sFs)

        self.add(low_eq_group, cover_rect)
        self.play(
            FadeOut(cover_rect),
            low_eq_group.animate.match_x(VGroup(clean_combined_fraction, minus_one)),
            FadeOut(q_marks[1]),
        )

        quirk_rects = VGroup(
            SurroundingRectangle(minus_one, buff=SMALL_BUFF),
            SurroundingRectangle(sFs["- f(0)"][0], buff=SMALL_BUFF),
        )
        quirk_rects.set_stroke(RED, 2)
        minus_e_zero = Tex(R" - e^{a0}", **kw)
        minus_e_zero["0"].set_color(BLUE)
        minus_e_zero.next_to(quirk_rects[1], DOWN)

        self.play(ShowCreation(quirk_rects, lag_ratio=0))
        self.wait()
        self.play(
            TransformFromCopy(eat_terms[0]["= e^{a{t}}"][0], minus_e_zero, run_time=2)
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, VGroup(*quirk_rects, minus_e_zero), lag_ratio=0.5, run_time=1))

        # Maybe show more generally.
        kw["t2c"].update({"c_n": TEAL})
        exp_sum_terms = VGroup(
            Tex(R"f({t}) = \\sum_{n=1}^N c_n e^{a_n {t}}", **kw),
            Tex(R"F({s}) = \\sum_{n=1}^N {c_n \\over {s} - a_n}", **kw),
            Tex(R"f'({t}) = \\sum_{n=1}^N c_n \\cdot a_n e^{a_n {t}}", **kw),
            Tex(R"\\sum_{n=1}^N \\left( c_n {{s} \\over {s} - a_n} - c_n \\right)", **kw),
        )
        for exp_sum_term, eat_term, corner in zip(exp_sum_terms, eat_terms, [DR, UR, DL, UL]):
            exp_sum_term.move_to(eat_term, corner)
        eat_terms[3].align_to(eat_terms[1], DOWN)

        self.play(
            LaggedStart(
                (TransformMatchingTex(eat_term, exp_sum_term)
                for eat_term, exp_sum_term in zip(eat_terms[:3], exp_sum_terms)),
                lag_ratio=0.2,
            ),
            FadeTransform(VGroup(clean_combined_fraction, minus_one), exp_sum_terms[-1], time_span=(1, 2)),
            LaggedStart(
                *(
                    VGroup(arrow, arrow.label).animate.shift(vect)
                    for arrow, vect in [
                        (lt_arrows[0], LEFT),
                        (deriv_arrow, 0.5 * UP),
                        (lt_arrows[1], RIGHT),
                        (s_mult_arrow, 0.5 * DOWN),
                    ]
                ),
                lag_ratio=0.5,
            ),
            self.frame.animate.reorient(0, 0, 0, (-0.15, -0.39, 0.0), 13.58),
            low_eq_group.animate.shift(DR),
            run_time=3
        )
        self.wait()

    def add_arrow_label(self, arrow, label_tex, direction, buff=SMALL_BUFF):
        arrow.label = Tex(label_tex, **self.tex_config)
        arrow.label.next_to(arrow, direction, buff=buff)

    def grow_arrow(self, arrow, run_time=1):
        """
        Assumes the arrow has a .label attribute
        """
        return AnimationGroup(
            GrowArrow(arrow),
            FadeIn(arrow.label, shift=0.25 * arrow.get_vector()),
            run_time=run_time
        )


class PreviewStrategy(InteractiveScene):
    def construct(self):
        # Set up terms
        rect = Rectangle(6, 4)
        ode, lt_ode, lt_ans, ans = terms = VGroup(
            Text("Differential\\nEquation"),
            Text("Transformed\\nEquation"),
            Text("Transformed\\nSolution"),
            Text("Solution"),
        )
        VGroup(lt_ode, lt_ans).set_color(YELLOW)
        for term, corner in zip(terms, [UL, DL, DR, UR]):
            term.move_to(rect.get_corner(corner))

        # Set up arrows
        lt_arrow = self.get_lt_arrow(ode, lt_ode, buff=MED_SMALL_BUFF)
        solve_arrow = self.get_lt_arrow(lt_ode, lt_ans, label_tex=R"\\substack{\\text{Solve} \\\\ \\text{(Algebraically)}}")
        solve_arrow[1][:5].scale(1.5, about_edge=DOWN).shift(0.1 * UP)
        inv_lt_arrow = self.get_lt_arrow(lt_ans, ans, label_tex=R"\\mathcal{L}^{-1}", buff=MED_SMALL_BUFF)

        arrows = VGroup(lt_arrow, solve_arrow, inv_lt_arrow)

        # Show creation of terms
        self.add(ode)
        self.wait()
        self.play(
            self.grow_lt_arrow(lt_arrow),
            TransformMatchingStrings(ode.copy(), lt_ode, key_map={"Differential": "Transformed"}),
            run_time=1.5
        )
        self.wait()
        self.play(
            self.grow_lt_arrow(solve_arrow),
            FadeTransform(lt_ode.copy(), lt_ans),
            run_time=1.5
        )
        self.wait()
        self.play(
            self.grow_lt_arrow(inv_lt_arrow),
            FadeTransform(lt_ans["Solution"][0].copy(), ans),
            run_time=1.5
        )
        self.wait()

        # Add domain backgrounds
        time_domain = FullScreenRectangle()
        time_domain.stretch(0.55, 1, about_edge=UP)
        time_domain.set_stroke(BLUE, 3)
        time_domain.set_fill(opacity=0)
        s_domain = FullScreenRectangle()
        s_domain.stretch(0.45, 1, about_edge=DOWN)
        s_domain.set_stroke(YELLOW, 3)
        s_domain.set_fill(opacity=0)

        time_label = Text("Time domain")
        s_label = Text("s domain")
        s_label.set_fill(YELLOW)

        for label, domain in [(time_label, time_domain), (s_label, s_domain)]:
            label.next_to(domain.get_corner(UL), DR)

        self.play(LaggedStart(
            FadeIn(time_domain),
            FadeIn(time_label),
            FadeIn(s_domain),
            FadeIn(s_label),
        ))
        self.wait()

    def get_lt_arrow(self, m1, m2, thickness=4, label_font_size=36, buff=0.15, label_tex=R"\\mathcal{L}"):
        arrow = Arrow(m1, m2, buff=buff, thickness=thickness)
        arrow.set_fill(border_width=2)
        label = Tex(label_tex, font_size=label_font_size)
        label.move_to(arrow.get_center())
        shift_dir = rotate_vector(normalize(arrow.get_vector()), 90 * DEG)
        label.shift(1.25 * label.get_height() * shift_dir)
        return VGroup(arrow, label)

    def grow_lt_arrow(self, lt_arrow):
        return AnimationGroup(
            GrowArrow(lt_arrow[0]),
            FadeIn(lt_arrow[1], shift=0.25 * lt_arrow[0].get_vector())
        )


class WalkThroughEquationSolution(PreviewStrategy):
    def construct(self):
        # Add ode
        x_colors, t2c = self.get_x_colors_and_t2c()
        ode = Tex(R"m x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega t)", t2c=t2c)
        ode.to_edge(UP)

        xt = ode["x(t)"][0]
        dxt = ode["x'(t)"][0]
        ddxt = ode["x''(t)"][0]
        xt_group = VGroup(xt, dxt, ddxt)

        self.add(ode)

        # Transform of the full equation
        ode_lt_lhs = Tex(R"""
            m\\Big({s}^2 X({s}) - {s} x_0 - v_0 \\Big)
            + \\mu \\Big( {s} X({s}) - x_0 \\Big)
            + k X({s})
        """, t2c=t2c)
        factored_ode_lt_lhs = Tex(R"""
            X({s})\\big(m{s}^2 + \\mu{s} + k\\big)
            - m v_0 - (m{s} + \\mu)x_0
        """, t2c=t2c)
        ode_lt_lhs.next_to(xt_group, DOWN, buff=1.5)
        factored_ode_lt_lhs.next_to(ode_lt_lhs, DOWN, buff=1.5)

        x_lt = ode_lt_lhs[R"X({s})"][-1]
        dx_lt = ode_lt_lhs[R"{s} X({s}) - x_0"][-1]
        ddx_lt = ode_lt_lhs[R"{s}^2 X({s}) - {s} x_0 - v_0"][0]
        x_lt_parts = VGroup(x_lt, dx_lt, ddx_lt)

        # Show each transform
        for part in x_lt_parts:
            part.save_state()
        x_lt_parts[:2].match_x(xt_group)

        xt_rect = SurroundingRectangle(xt, buff=0.05)
        dxt_rect = SurroundingRectangle(dxt, buff=0.05)
        ddxt_rect = SurroundingRectangle(ddxt, buff=0.05)
        xt_rects = VGroup(xt_rect, dxt_rect, ddxt_rect)
        for rect, color in zip(xt_rects, x_colors):
            rect.set_stroke(color, width=2)

        xt_arrow = self.get_lt_arrow(xt_rect, x_lt)
        dxt_arrow = self.get_lt_arrow(dxt_rect, dx_lt)
        ddx_arrow = self.get_lt_arrow(ddxt_rect, ddx_lt)

        self.play(ShowCreation(xt_rect))
        self.play(
            self.grow_lt_arrow(xt_arrow),
            FadeTransform(xt.copy(), x_lt)
        )
        self.wait()
        self.play(ShowCreation(dxt_rect))
        self.play(
            self.grow_lt_arrow(dxt_arrow),
            FadeTransform(dxt.copy(), dx_lt)
        )
        self.wait()

        # Ask about L{x''(t)}
        ddx_lt_rect = SurroundingRectangle(ddx_lt, buff=SMALL_BUFF)
        ddx_lt_rect.set_stroke(RED, 1)

        self.play(ShowCreation(ddxt_rect)),
        self.play(LaggedStart(
            self.grow_lt_arrow(ddx_arrow),
            TransformFromCopy(ddxt_rect, ddx_lt_rect),
            Restore(x_lt),
            Transform(xt_arrow, self.get_lt_arrow(xt_rect, x_lt.saved_state)),
            Restore(dx_lt),
            Transform(dxt_arrow, self.get_lt_arrow(dxt_rect, dx_lt.saved_state)),
        ))
        self.wait()

        # Show second derivative rule
        ddx_lt_lhs = Tex(R"\\mathcal{L}\\Big\\{x''(t)\\Big\\}", t2c=t2c)
        ddx_lt_rhss = VGroup(
            Tex(R"= {s} \\mathcal{L}\\Big\\{x'(t)\\Big\\} - x'(0)", t2c=t2c),
            Tex(R"= {s} \\mathcal{L}\\Big\\{x'(t)\\Big\\} - v_0", t2c=t2c),
            Tex(R"= {s} \\Big({s}X({s}) - x_0 \\Big) - v_0", t2c=t2c),
            Tex(R"= {s}^2 X({s}) - {s} x_0 - v_0", t2c=t2c),
        )
        ddx_lt_lhs.to_edge(DOWN, buff=1.5)
        ddx_lt_lhs.to_edge(LEFT, buff=2.0)
        for rhs in ddx_lt_rhss:
            rhs.next_to(ddx_lt_lhs, RIGHT)
        for rhs in ddx_lt_rhss[2:]:
            rhs.next_to(ddx_lt_rhss[1], RIGHT)

        v0_rect = SurroundingRectangle(ddx_lt_rhss[0][R"x'(0)"])
        v0_rect.set_stroke(x_colors[1], 2)

        self.play(
            TransformFromCopy(ddx_arrow[1], ddx_lt_lhs[0]),
            TransformFromCopy(ddxt, ddx_lt_lhs[R"x''(t)"][0]),
            Write(ddx_lt_lhs[R"\\Big\\{"]),
            Write(ddx_lt_lhs[R"\\Big\\}"]),
        )
        self.wait()
        self.play(
            TransformMatchingTex(ddx_lt_lhs.copy(), ddx_lt_rhss[0], path_arc=30 * DEG, run_time=1)
        )
        self.wait()
        self.play(ShowCreation(v0_rect))
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[0],
                ddx_lt_rhss[1],
                matched_keys=[R"= {s} \\mathcal{L}\\Big\\{x'(t)\\Big\\} - "],
                key_map={R"x'(0)": R"v_0"},
                run_time=1,
            ),
            v0_rect.animate.surround(ddx_lt_rhss[1][R"v_0"])
        )
        self.play(FadeOut(v0_rect))
        self.wait()
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[1].copy(),
                ddx_lt_rhss[2],
                key_map={R"\\mathcal{L}\\Big\\{x'(t)\\Big\\}": R"\\Big({s}X({s}) - x_0 \\Big)"},
                matched_keys=[R"- v_0"],
                run_time=1.5,
                path_arc=30 * DEG,
            ),
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                ddx_lt_rhss[2],
                ddx_lt_rhss[3],
                matched_keys=[R"- v_0", "x_0", "X({s})"],
                run_time=1.5,
                path_arc=30 * DEG,
            ),
        )
        self.wait()
        self.play(
            FadeOut(ddx_lt_lhs),
            FadeOut(ddx_lt_rhss[1]),
            FadeOut(ddx_lt_rhss[3][0]),
            FadeTransform(ddx_lt_rhss[3][1:], ddx_lt),
            run_time=2
        )
        self.play(FadeOut(ddx_lt_rect))
        self.wait()

        # Bring along constants
        eq_index = ode.submobjects.index(ode["="][0][0])
        ode_lhs_rect = SurroundingRectangle(ode[:eq_index])
        ode_lhs_rect.set_stroke(BLUE, 2)
        ode_lt_lhs_rect = SurroundingRectangle(ode_lt_lhs)
        ode_lt_lhs_rect.set_stroke(YELLOW, 2)
        lhs_arrow = self.get_lt_arrow(ode_lhs_rect, ode_lt_lhs_rect.copy().shift(0.5 * RIGHT))

        self.play(
            LaggedStart(
                *(
                    TransformFromCopy(ode[tex][0], ode_lt_lhs[tex][0])
                    for tex in ["m", R"\\mu", "k"]
                ),
                lag_ratio=0.5
            ),
            AnimationGroup(*(
                FadeIn(ode_lt_lhs[tex])
                for tex in [R"\\Big(", R"\\Big)", "+"]
            ))
        )
        self.wait()
        self.play(LaggedStart(
            FadeOut(xt_arrow),
            FadeOut(ddx_arrow),
            FadeOut(xt_rect),
            FadeOut(ddxt_rect),
            ReplacementTransform(dxt_rect, ode_lhs_rect),
            ReplacementTransform(dxt_arrow, lhs_arrow),
            FadeIn(ode_lt_lhs_rect),
        ))
        self.wait()

        # Factor out X(s)
        Xs_parts = VGroup(
            ode_lt_lhs[R"m\\Big({s}^2 X({s})"][0],
            ode_lt_lhs[R"\\mu \\Big( {s} X({s})"][0],
            ode_lt_lhs[R"k X({s})"][0],
        )
        Xs_part_rects = VGroup(
            SurroundingRectangle(part, buff=0.1)
            for part in Xs_parts
        )
        Xs_part_rects[2].match_height(Xs_part_rects, stretch=True)
        Xs_part_rects.set_stroke(YELLOW, 2)

        ode_lt_lhs.set_fill(opacity=0.35)
        Xs_parts.set_fill(opacity=1)
        Xs_parts[0][1].set_fill(opacity=0.25)
        Xs_parts[1][1].set_fill(opacity=0.25)
        ode_lt_lhs.save_state()
        ode_lt_lhs.set_fill(opacity=1)

        self.remove(ode_lt_lhs_rect)
        self.play(
            FadeOut(ode_lt_lhs_rect),
            FadeIn(Xs_part_rects),
            Restore(ode_lt_lhs),
            lhs_arrow.animate.scale(0.75, about_edge=UL)
        )
        self.wait()
        self.play(LaggedStart(
            *[
                TransformFromCopy(ode_lt_lhs[tex][index0], factored_ode_lt_lhs[tex][index1])
                for tex in [R"X({s})", "m", R"{s}^2", R"\\mu", R"{s}", "k"]
                for index0 in [3 if tex == R"{s}" else 0]
                for index1 in [2 if tex == R"{s}" else 0]
            ] + [
                Write(factored_ode_lt_lhs["+"][:2]),
                Write(factored_ode_lt_lhs[R"\\big("]),
                Write(factored_ode_lt_lhs[R"\\big)"]),
            ],
            lag_ratio=0.1,
            run_time=3
        ))
        self.wait()

        # Show initial conditions
        ic_parts = VGroup(
            ode_lt_lhs[R"- {s} x_0 - v_0"][0],
            ode_lt_lhs[R"- x_0"][0],
        )
        ic_part_consts = VGroup(
            ode_lt_lhs[R"m\\Big("][0],
            ode_lt_lhs[R"\\Big)"][0],
            ode_lt_lhs[R"\\mu \\Big("][0],
            ode_lt_lhs[R"\\Big)"][1],
        ).copy().set_fill(opacity=1)
        ic_part_rects = VGroup(SurroundingRectangle(part, buff=SMALL_BUFF) for part in ic_parts)
        ic_part_rects.set_stroke(TEAL, 2)
        factored_ic_part = factored_ode_lt_lhs[R"- m v_0 - (m{s} + \\mu)x_0"]

        self.play(
            FadeOut(Xs_part_rects),
            FadeIn(ic_part_rects),
            Xs_parts.animate.set_fill(opacity=0.25),
            ic_parts.animate.set_fill(opacity=1),
            FadeIn(ic_part_consts),
        )
        self.play(
            Write(factored_ic_part),
        )
        self.wait()

        # Comment on initial conditions
        ic_rect = SurroundingRectangle(factored_ic_part, buff=0.15)
        ic_rect.set_stroke(TEAL, 3)

        ic_words = Text("Initial conditions")
        ic_words.next_to(ic_rect, DOWN)
        zero_ic = Tex(R"\\text{Let’s assume } x_0 = v_0 = 0", t2c=t2c)
        zero_ic.next_to(ic_words, DOWN, aligned_edge=LEFT)

        poly_part = Tex(R"X({s})\\big(m{s}^2 + \\mu{s} + k\\big)", t2c=t2c)
        poly_part.move_to(factored_ode_lt_lhs[poly_part.get_tex()][0])

        lt_equals = Tex(R"=")
        lt_equals.match_y(ode_lt_lhs)
        lt_equals.match_x(ode["="])

        self.play(
            ShowCreation(ic_rect),
            Write(ic_words),
            run_time=1
        )
        self.wait()
        self.play(FadeIn(zero_ic, 0.5 * DOWN))
        self.wait()

        self.play(
            FadeOut(ode_lt_lhs, UP),
            FadeOut(ic_part_rects, UP),
            FadeOut(ic_part_consts, UP),
            poly_part.animate.next_to(lt_equals, LEFT),
            lhs_arrow.animate.scale(1 / 0.75, about_edge=UL),
        )
        self.play(
            LaggedStartMap(FadeOut, VGroup(factored_ode_lt_lhs, ic_rect, ic_words, zero_ic), lag_ratio=0.25),
            run_time=2,
        )
        self.wait()

        # Mirror image
        ode_rhs = ode[R"F_0 \\cos(\\omega t)"][0]

        part_pairs = [
            # [ode[tex1][index].copy(), poly_part[tex2][index].copy()]
            [ode[tex1][index].copy(), poly_part_copy[tex2][index].copy()]
            for tex1, tex2, index in [
                ("m", "m", 0),
                ("x''(t)", R"{s}^2", 0),
                ("+", "+", 0),
                (R"\\mu", R"\\mu", 0),
                (R"x'(t)", R"{s}", -1),
                ("+", "+", 1),
                ("k", "k", 0),
            ]
        ]

        self.play(LaggedStart(
            ode_rhs.animate.set_fill(opacity=0.25),
            FadeOut(ode_lhs_rect.copy()),
            ShowCreation(ode_lhs_rect),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(LaggedStart(
            *(TransformFromCopy(*pair) for pair in part_pairs),
            lag_ratio=0.025
        ))
        self.wait()
        self.play(LaggedStart(
            *(TransformFromCopy(*reversed(pair)) for pair in part_pairs),
            lag_ratio=0.025
        ))
        self.wait()

        for pair in part_pairs:
            self.remove(*pair)
        self.add(poly_part)

        # Transform cosine
        ode_lt_rhs = Tex(R"{F_0 {s} \\over {s}^2 + \\omega^2}", t2c=t2c)
        ode_lt_rhs.next_to(lt_equals, RIGHT)

        rhs_arrow = self.get_lt_arrow(ode_rhs, ode_lt_rhs)

        self.play(
            Write(lt_equals),
            TransformFromCopy(lhs_arrow, rhs_arrow),
            ode_rhs.animate.set_fill(opacity=1),
            ode_lhs_rect.animate.surround(ode_rhs),
        )
        self.wait()
        self.play(LaggedStart(
            *(
                TransformFromCopy(ode[tex][0], ode_lt_rhs[tex][0])
                for tex in ["F_0", R"\\omega"]
            ),
            FadeIn(ode_lt_rhs[R"{s} \\over {s}^2 +"][0]),
            FadeIn(ode_lt_rhs[R"^2"][1]),
        ))
        self.add(ode_lt_rhs)
        self.wait()

        # Walk through cosine transform
        cos_transform_parts = VGroup(
            Tex(R"\\mathcal{L}\\big\\{\\cos(\\omega t)\\big\\}", t2c=t2c),
            Tex(R"= \\mathcal{L}\\left\\{\\frac{1}{2}e^{i \\omega t} + \\frac{1}{2} e^{\\minus i \\omega t} \\right\\}", t2c=t2c),
            Tex(R"= \\frac{1}{2} \\mathcal{L}\\big\\{e^{i \\omega t}\\big\\} + \\frac{1}{2} \\mathcal{L}\\big\\{e^{\\minus i \\omega t}\\big\\}", t2c=t2c),
            Tex(R"= \\frac{1}{2} {1 \\over {s} - \\omega i} + \\frac{1}{2} {1 \\over {s} + \\omega i}", t2c=t2c),
            Tex(R"= {{s} \\over {s}^2 + \\omega^2}", t2c=t2c),
        )
        cos_transform_parts.arrange(RIGHT)
        cos_transform_parts.to_edge(DOWN, buff=1.5)
        cos_transform_parts.to_edge(LEFT, buff=0.5)
        for part in cos_transform_parts[1:]:
            part.next_to(cos_transform_parts[0], RIGHT)
        cos_transform_parts[-1].next_to(cos_transform_parts[-2], RIGHT, aligned_edge=DOWN)

        self.play(LaggedStart(
            TransformFromCopy(ode[R"\\cos(\\omega t)"], cos_transform_parts[0][R"\\cos(\\omega t)"]),
            FadeTransform(rhs_arrow[1].copy(), cos_transform_parts[0][R"\\mathcal{L}"]),
            Write(cos_transform_parts[0][R"\\big\\{"]),
            Write(cos_transform_parts[0][R"\\big\\}"]),
        ))
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[0].copy(),
                cos_transform_parts[1],
                matched_keys=[R"\\omega"],
                run_time=1
            )
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[1],
                cos_transform_parts[2],
                matched_keys=[R"\\mathcal{L}", R"e^{i \\omega t}", R"e^{\\minus i \\omega t}"],
                key_map={R"\\left\\{": R"\\big\\{", R"\\right\\}": R"\\big\\}", },
                run_time=1
            )
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                cos_transform_parts[2],
                cos_transform_parts[3],
                matched_keys=[R"\\frac{1}{2}", "+"],
                key_map={
                    R"\\mathcal{L}\\big\\{e^{i \\omega t}\\big\\}": R"{1 \\over {s} - \\omega i}",
                    R"\\mathcal{L}\\big\\{e^{\\minus i \\omega t}\\big\\}": R"{1 \\over {s} + \\omega i}",
                },
                run_time=1,
                path_arc=30 * DEG,
            )
        )
        self.wait()
        self.play(Write(cos_transform_parts[4], run_time=1))
        self.wait()
        self.play(
            *(FadeOut(cos_transform_parts[i]) for i in [0, 3, 4]),
            FadeOut(ode_lhs_rect)
        )
        self.wait()

        # Divide out
        poly_tex = R"\\big(m{s}^2 + \\mu{s} + k\\big)"
        cos_lt_denom_tex = R"\\left({s}^2 + \\omega^2 \\right)"
        true_poly_part = poly_part[poly_tex][0]
        final_answer = Tex(R"X({s}) = {F_0 {s} \\over " + cos_lt_denom_tex + poly_tex + "}", t2c=t2c)
        final_answer.next_to(lt_equals, DOWN, buff=1.5)

        poly_rect = SurroundingRectangle(true_poly_part, buff=SMALL_BUFF)
        poly_rect.set_stroke(TEAL, 2)

        self.play(ShowCreation(poly_rect))
        self.play(LaggedStart(
            AnimationGroup(
                TransformFromCopy(true_poly_part, final_answer[poly_tex][0]),
                poly_rect.animate.surround(final_answer[poly_tex][0])
            ),
            TransformFromCopy(poly_part["X({s})"], final_answer["X({s})"]),
            AnimationGroup(
                TransformFromCopy(ode_lt_rhs[R"F_0 {s} \\over"], final_answer[R"F_0 {s} \\over"]),
                TransformFromCopy(ode_lt_rhs[R"{s}^2 + \\omega^2"], final_answer[R"{s}^2 + \\omega^2"]),
            ),
            Write(final_answer["="]),
            Write(final_answer[R"\\left("]),
            Write(final_answer[R"\\right)"]),
            lag_ratio=0.1,
            run_time=3
        ))
        self.play(FadeOut(poly_rect))
        self.wait()

        # Pull up final answer
        self.play(
            LaggedStartMap(FadeOut, VGroup(lhs_arrow, rhs_arrow, poly_part, lt_equals, ode_lt_rhs), shift=0.2 * UP, lag_ratio=0.2, run_time=1),
            LaggedStart(
                final_answer.animate.next_to(ode, DOWN, MED_LARGE_BUFF).shift(RIGHT),
                ode.animate.scale(0.75, about_edge=UP),
                run_time=2,
                lag_ratio=0.25,
            )
        )
        self.wait()

        # Write L{x(t)}
        final_lhs = Tex(R"\\mathcal{L}\\left\\{x(t)\\right\\} = ", t2c=t2c)
        final_lhs.next_to(final_answer, LEFT)
        final_lhs.shift(SMALL_BUFF * UP)
        xt_copy = ode["x(t)"][0].copy()

        self.play(xt_copy.animate.replace(final_lhs["x(t)"]))
        self.play(Write(final_lhs))
        self.remove(xt_copy)
        self.wait()

        # Reference inversion
        rhs_tex = final_answer.get_tex().split( "= ")[1]
        inverse_equation = Tex(
            R"x(t) = \\mathcal{L}^{-1}\\left\\{" + rhs_tex + R"\\right\\}",
            t2c=t2c
        )
        inverse_equation.next_to(final_answer, DOWN, LARGE_BUFF)
        inverse_equation.set_x(0)

        self.play(LaggedStart(
            *(
                TransformFromCopy(final_lhs[tex], inverse_equation[tex])
                for tex in ["x(t)", R"\\mathcal{L}", R"\\left\\{", R"\\right\\}", "="]
            ),
            FadeInFromPoint(inverse_equation["-1"], final_lhs.get_center()),
            TransformFromCopy(final_answer[rhs_tex], inverse_equation[rhs_tex]),
            lag_ratio=0.025,
            run_time=1.5
        ))
        self.wait()
        self.play(FadeOut(inverse_equation, DOWN))

        # Ask about denominator
        denom_tex = cos_lt_denom_tex + poly_tex
        sub_texs = [rhs_tex, denom_tex, poly_tex, cos_lt_denom_tex]
        rhs, denom, poly_part, cos_lt_denom = answer_parts = VGroup(
            final_answer[tex][0]
            for tex in sub_texs
        )
        rect = SurroundingRectangle(rhs, buff=0.05)
        rect.set_stroke(YELLOW, 2)

        zero_question = Text("When is this 0?")
        zero_question.next_to(rect, DOWN)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            final_lhs.animate.set_fill(opacity=0.5),
            final_answer[:final_answer.submobjects.index(cos_lt_denom[0])].animate.set_fill(opacity=0.5),
            rect.animate.surround(denom),
            FadeIn(zero_question, lag_ratio=0.1)
        )
        self.wait()

        # Show quadratic formula
        implies = Tex(R"\\Longrightarrow", font_size=72)
        implies.rotate(-90 * DEG)
        implies.next_to(poly_part, DOWN)
        eq_0 = Tex(R"=0", font_size=36)
        eq_0.next_to(implies, RIGHT, buff=0)
        implies.add(eq_0)
        implies.add(eq_0.copy().fade(1).next_to(implies, LEFT, buff=0))

        quadratic_form = Tex(R"{s} = {-\\mu \\pm \\sqrt{\\mu^2 - 4mk} \\over 2m}", t2c=t2c, font_size=36)
        quadratic_form.next_to(implies, DOWN)

        poly_part_copy = Tex(poly_tex, t2c=t2c)
        poly_part_copy.replace(poly_part)

        self.play(
            FadeTransformPieces(zero_question, eq_0),
            Write(implies),
            rect.animate.surround(poly_part),
            cos_lt_denom.animate.set_fill(opacity=0.5)
        )
        self.wait()
        self.play(
            TransformMatchingTex(poly_part_copy, quadratic_form, lag_ratio=0.01)
        )
        self.wait()

        # Show omega i and -omega i roots
        cos_poles = Tex(R"{s} = \\pm \\omega i", t2c=t2c)
        cos_poles.next_to(implies, DOWN)
        cos_poles.match_x(cos_lt_denom)

        self.play(
            cos_lt_denom.animate.set_fill(opacity=1),
            poly_part.animate.set_fill(opacity=0.5),
            rect.animate.surround(cos_lt_denom),
            implies.animate.match_x(cos_lt_denom),
            FadeTransformPieces(quadratic_form, cos_poles)
        )
        self.wait()

    def get_x_colors_and_t2c(self):
        x_colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            "x(t)": x_colors[0],
            "x'(t)": x_colors[1],
            "x''(t)": x_colors[2],
            "x_0": x_colors[0],
            "v_0": x_colors[1],
            R"\\omega": PINK,
            "{s}": YELLOW,
        }
        return x_colors, t2c


class ShowSolutionPoles(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        F_0 = 1
        m = 1
        k = 2
        mu = 2
        omega = 2

        def func(s):
            denom = (s**2 + omega**2) * (m * s**2 + mu * s + k)
            return np.divide(F_0 * s, denom)

        plane = ComplexPlane((-3, 3), (-3, 3), faded_line_ratio=0)
        plane.add_coordinate_labels(font_size=16)
        plot = get_complex_graph(plane, func)

        self.add(plane, plot)

        # Pan around
        self.play(
            frame.animate.reorient(45, 82, 0, (-0.54, 0.3, 1.06), 5.71),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-31, 83, 0, (-0.13, 0.5, 1.02), 5.71),
            run_time=13,
        )
        self.wait()

        # Emphasize the four roots
        roots = [-1 + 1j, -1 - 1j, 2j, -2j]
        s_points = [plane.n2p(s) for s in roots]
        s_dots = Group(Group(TrueDot(p), GlowDot(p)) for p in s_points)
        s_dots.set_color(YELLOW)
        s_labels = VGroup(
            Tex(Rf"s_{n}").next_to(point, RIGHT)
            for n, point in enumerate(s_points, 1)
        )
        s_labels.set_color(YELLOW)
        s_labels.set_backstroke(BLACK, 5)

        self.play(
            frame.animate.to_default_state().set_field_of_view(1 * DEG),
            plot[0].animate.set_opacity(0.25),
            plot[1].animate.set_stroke(opacity=0.01),
            run_time=2
        )
        self.play(
            FadeIn(s_labels, lag_ratio=0.25),
            FadeIn(s_dots, lag_ratio=0.25),
        )
        self.wait()

        # Highlight certain roots
        rect = SurroundingRectangle(Group(s_labels[:2], s_dots[:2]))
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(Group(s_labels[2:], s_dots[2:])))
        self.wait()
        self.play(FadeOut(rect))


class PartialFractionDecomposition(WalkThroughEquationSolution):
    def construct(self):
        # Set up equation
        x_colors, t2c = self.get_x_colors_and_t2c()
        poly_tex = R"\\big(m{s}^2 + \\mu{s} + k\\big)"
        cos_lt_denom_tex = R"\\left({s}^2 + \\omega^2 \\right)"
        denom_tex = cos_lt_denom_tex + poly_tex
        tex_kw = dict(t2c=t2c, font_size=42)

        final_answer = Tex(R"X({s}) = {F_0 {s} \\over " + denom_tex + "}", **tex_kw)
        final_answer.to_corner(UL)
        final_answer.save_state()
        final_answer.center().scale(1.5)
        self.add(final_answer)

        # Show the roots
        t2c.update({f"r_{n}": TEAL for n in range(1, 5)})
        denom_rect = SurroundingRectangle(final_answer[denom_tex], buff=0.1)
        denom_rect.target = SurroundingRectangle(final_answer.saved_state[denom_tex], buff=0.1)
        VGroup(denom_rect, denom_rect.target).set_stroke(TEAL, 3)

        denom_roots_title = Text("Roots of denominator", font_size=36)
        denom_roots_title.next_to(final_answer.saved_state, DOWN, buff=1.25, aligned_edge=LEFT)
        denom_roots_title.add(Underline(denom_roots_title))

        roots = VGroup(
            Tex(R"r_1 = +\\omega i", **tex_kw),
            Tex(R"r_2 = -\\omega i", **tex_kw),
            Tex(R"r_3 = \\left(-\\mu + \\sqrt{\\mu^2 - 4mk} \\right) / 2m", **tex_kw),
            Tex(R"r_4 = \\left(-\\mu - \\sqrt{\\mu^2 - 4mk} \\right) / 2m", **tex_kw),
        )
        for root in roots[2:]:
            root[3:].scale(0.8, about_edge=LEFT)
        roots.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        roots.next_to(denom_roots_title[0], DOWN, aligned_edge=LEFT, buff=0.75)

        self.play(ShowCreation(denom_rect))
        self.play(
            Restore(final_answer),
            MoveToTarget(denom_rect),
            FadeIn(denom_roots_title),
            LaggedStart(*(
                FadeIn(root[:3])
                for root in roots
            ), lag_ratio=0.15),
            LaggedStart(
                FadeTransform(final_answer[cos_lt_denom_tex].copy(), roots[0][3:]),
                FadeTransform(final_answer[cos_lt_denom_tex].copy(), roots[1][3:]),
                FadeTransform(final_answer[poly_tex].copy(), roots[2][3:]),
                FadeTransform(final_answer[poly_tex].copy(), roots[3][3:]),
                lag_ratio=0.15,
            ),
        )
        self.play(FadeOut(denom_rect))
        self.wait()

        # Break apart fraction
        rhs_str = " = " + " + ".join([
            fR"{{c_{n} \\over {{s}} - r_{n}}}"
            for n in range(1, 5)
        ])
        rhs = Tex(rhs_str, **tex_kw)
        rhs.next_to(final_answer, RIGHT)

        self.play(
            LaggedStart(
                *(
                    Transform(root[:2].copy(), rn, remover=True)
                    for root, rn in zip(roots, rhs[re.compile(r"r_.")])
                ),
                lag_ratio=0.2
            ),
            FadeIn(rhs, lag_ratio=0.1, time_span=(0.5, 1.5))
        )
        self.add(rhs)
        self.wait()

        # Highlight c terms
        c_terms = rhs[re.compile(r"c_.")]
        c_rects = VGroup(SurroundingRectangle(c_term, buff=0.1) for c_term in c_terms)
        c_rects.set_stroke(BLUE, 2)
        q_marks = VGroup(Tex(R"???", font_size=36).next_to(rect, UP, SMALL_BUFF) for rect in c_rects)

        self.play(
            LaggedStartMap(ShowCreation, c_rects, lag_ratio=0.2),
            LaggedStartMap(FadeIn, q_marks, lag_ratio=0.2),
        )
        self.wait()

        # Write up the partial fraction decomposition idea
        clean_fraction = Tex(
            R"F_0 {s} / m \\over ({s} - r_1)({s} - r_2)({s} - r_3)({s} - r_4)",
            **tex_kw
        )
        example = Tex(R"c_1 = {F_0 r_1 / m \\over (r_1 - r_2)(r_1 - r_3)(r_1 - r_4)}", **tex_kw)
        pfd_group = VGroup(
            Text("It’s easiest to rewrite our fraction as", **tex_kw),
            clean_fraction,
            TexText(R"""
                To calculate each term $c_i$, remove the $({s} - r_i)$ term \\\\
                from our fraction and plug in \${s} = r_i$. For example,
            """, alignment="", **tex_kw),
            example,
            TexText(R"""
                Why? This essentially amounts to multiplying both sides\\\\
                of our equation by $({s} - r_1)$ and taking the limit as \${s} \\to r_1$
            """, alignment="", **tex_kw),
        )
        pfd_group.arrange(DOWN, buff=0.5)
        for part in pfd_group[0::2]:
            part.align_to(pfd_group, LEFT)

        pfd_box = SurroundingRectangle(pfd_group, buff=0.5)
        pfd_box.set_fill(GREY_E, 1)
        pfd_box.set_stroke(WHITE, 2)
        pfd_group.add_to_back(pfd_box)
        pfd_group.set_width(7.0)
        pfd_group.to_corner(DR)

        pfd_title = Text("Partial Fraction Decomposition")
        pfd_title.next_to(pfd_box, UP)
        pfd_group.add_to_back(pfd_title)

        self.play(
            FadeIn(pfd_group, lag_ratio=5e-3, run_time=2),
        )
        self.wait()
        self.play(
            FadeOut(pfd_group),
            FadeOut(c_rects),
            FadeOut(q_marks),
        )

        # Show inverse laplace process
        t2c["{t}"] = BLUE
        exp_sum = Tex("+".join([
            Rf"c_{i} e^{{r_{i} {{t}} }}"
            for i in range(1, 5)
        ]), t2c=t2c, font_size=46)

        exp_sum.next_to(rhs[1:], DOWN, buff=1.5)
        frac_parts = rhs[re.compile(r"{c_. \\\\over {s} - r_.}")]
        exp_parts = exp_sum[re.compile(r"c_. e\\^{r_. {t} }")]

        inv_lt_arrows = VGroup(
            self.get_lt_arrow(frac_part, exp_part, label_tex=R"\\mathcal{L}^{-1}")
            for frac_part, exp_part in zip(frac_parts, exp_parts)
        )
        frac_rects = VGroup(SurroundingRectangle(part) for part in frac_parts)
        frac_rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(frac_rects, lag_ratio=0.2))
        self.play(
            LaggedStartMap(self.grow_lt_arrow, inv_lt_arrows),
            LaggedStart(*(
                TransformFromCopy(rhs[re.compile(p1)], exp_sum[re.compile(p2)], lag_ratio=0.05)
                for p1, p2 in [
                    (r"c_.", r"c_."),
                    (r"\\\\over", r"e"),
                    (r"{s}", r"{t}"),
                    (r"r_.", r"r_."),
                    (r"\\+", r"\\+"),
                ]
            ), lag_ratio=0.1, run_time=2),
            # TransformMatchingTex(rhs.copy(), exp_sum),
        )
        self.add(exp_sum)
        self.wait()

        # Highlight rs then cs
        upper_r_rects, lower_r_rects, upper_c_rects, lower_c_rects = [
            VGroup(SurroundingRectangle(term, buff=0.025) for term in group[pattern]).set_stroke(YELLOW, 2)
            for pattern in [re.compile(R"r_."), re.compile(R"c_.")]
            for group in [rhs, exp_sum]
        ]

        self.play(ReplacementTransform(frac_rects, upper_r_rects))
        self.wait()
        self.play(TransformFromCopy(upper_r_rects, lower_r_rects))
        self.wait()
        self.play(
            ReplacementTransform(upper_r_rects, upper_c_rects),
            ReplacementTransform(lower_r_rects, lower_c_rects),
        )
        self.wait()
        self.play(FadeOut(upper_c_rects), FadeOut(lower_c_rects))

        # Highlight cosine part
        index = exp_sum.submobjects.index(exp_sum["+"][1][0])
        cos_part_rect = SurroundingRectangle(exp_sum[:index], buff=0.1)
        cos_part_rect.set_stroke(BLUE, 2)
        c1_tex = R"{F_0 \\over 2m(k/m - \\omega^2) + 2\\mu \\omega i}"
        c2_tex = R"{F_0 \\over 2m(k/m - \\omega^2) - 2\\mu \\omega i}"
        const_tex = R"{F_0 \\over 2m(k/m - \\omega^2)}"
        const_tex2 = R"{F_0 \\over m(k/m - \\omega^2)}"
        two_exp_tex = R" \\left(e^{+\\omega i t} + e^{-\\omega i t} \\right)"
        cos_exprs = VGroup(
            Tex(R"c_1 e^{+\\omega i t} + c_2 e^{-\\omega i t}", t2c=t2c),
            Tex(const_tex + two_exp_tex, t2c=t2c),
            Tex(const_tex + R"2\\cos(\\omega t)", t2c=t2c),
            Tex(const_tex2 + R"\\cos(\\omega t)", t2c=t2c),
        )
        cos_exprs.next_to(cos_part_rect, DOWN, LARGE_BUFF)
        const_part = cos_exprs[1][const_tex][0]
        const_rect = SurroundingRectangle(const_part)
        two_exp_rect = SurroundingRectangle(cos_exprs[1][two_exp_tex])
        cos_rect = SurroundingRectangle(cos_exprs[2][R"2\\cos(\\omega t)"])
        amp_rect = SurroundingRectangle(cos_exprs[3][const_tex2])
        diff_rect = SurroundingRectangle(cos_exprs[3][R"k/m - \\omega^2"])
        VGroup(c_rects, const_rect, two_exp_rect, cos_rect, amp_rect, diff_rect).set_stroke(YELLOW, 2)

        const_values = VGroup(
            Tex(c1_tex, t2c=t2c, font_size=24),
            Tex(c2_tex, t2c=t2c, font_size=24),
        )
        const_values.arrange(RIGHT, buff=LARGE_BUFF)
        const_values.next_to(cos_exprs[0], DOWN, LARGE_BUFF, aligned_edge=LEFT)
        const_value_rects = VGroup(SurroundingRectangle(value, buff=0.05) for value in const_values)
        c_terms = cos_exprs[0][re.compile("c_.")]
        c_rects = VGroup(SurroundingRectangle(c_term, buff=0.05) for c_term in c_terms)
        rect_lines = VGroup(
            Line(r2.get_bottom(), r1.get_top())
            for r1, r2 in zip(const_value_rects, c_rects)
        )
        VGroup(const_value_rects, c_rects, rect_lines).set_stroke(RED, 1)

        mu_assumption = Tex(R"\\text{Assume } \\mu \\approx 0", font_size=36)
        mu_assumption.next_to(const_value_rects, DOWN)

        self.play(
            ShowCreation(cos_part_rect),
            exp_sum[index:].animate.set_opacity(0.25),
            inv_lt_arrows.animate.set_fill(opacity=0.25),
            roots[2:].animate.set_opacity(0.25),
            denom_roots_title.animate.set_opacity(0.25),
        )
        self.wait()
        self.play(FadeIn(cos_exprs[0], DOWN))
        self.wait()
        self.play(
            LaggedStartMap(ShowCreation, c_rects, lag_ratio=0.35),
            LaggedStartMap(ShowCreation, const_value_rects, lag_ratio=0.35),
            LaggedStartMap(ShowCreation, rect_lines, lag_ratio=0.35),
            LaggedStart(
                *(FadeTransform(c_term.copy(), const_value)
                for c_term, const_value in zip(c_terms, const_values)),
                lag_ratio=0.35,
            )
        )
        self.wait()
        self.play(Write(mu_assumption))
        self.play(
            const_values[0][R"+ 2\\mu \\omega i}"].animate.set_fill(opacity=0.25),
            const_values[1][R"- 2\\mu \\omega i}"].animate.set_fill(opacity=0.25),
        )
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(const_values[0][:len(const_part)], const_part),
            TransformFromCopy(const_values[1][:len(const_part)], const_part),
            FadeOut(VGroup(const_values, rect_lines, c_rects)),
            FadeOut(c_terms),
            ReplacementTransform(cos_exprs[0][R"e^{+\\omega i t}"], cos_exprs[1][R"e^{+\\omega i t}"]),
            ReplacementTransform(const_value_rects[0], const_rect),
            ReplacementTransform(cos_exprs[0][R"e^{-\\omega i t}"], cos_exprs[1][R"e^{-\\omega i t}"]),
            ReplacementTransform(const_value_rects[1], const_rect),
            ReplacementTransform(cos_exprs[0][R"+"][1], cos_exprs[1][R"+"][1]),
            FadeIn(cos_exprs[1][R"\\left("]),
            FadeIn(cos_exprs[1][R"\\right)"]),
            mu_assumption.animate.next_to(cos_exprs[1], DOWN, MED_LARGE_BUFF),
            lag_ratio=0.01
        ))
        self.add(cos_exprs[1])
        self.wait()
        self.play(ReplacementTransform(const_rect, two_exp_rect))
        self.wait()
        self.play(
            ReplacementTransform(two_exp_rect, cos_rect),
            TransformMatchingTex(*cos_exprs[1:3], key_map={two_exp_tex: R"2\\cos(\\omega t)"}),
            run_time=1
        )
        self.wait()
        self.play(
            ReplacementTransform(cos_rect, amp_rect),
            TransformMatchingTex(*cos_exprs[2:4]),
            run_time=1
        )
        self.wait()
        self.play(ReplacementTransform(amp_rect, diff_rect))
        self.wait()


class IntegrateByParts(InteractiveScene):
    def construct(self):
        # Show formulas
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        ibp_rule = VGroup(
            Text("Integration by parts: "),
            Tex(R"\\int u dv = uv - \\int v du", t2c={"u": PINK, "v": RED})
        )
        ibp_rule.arrange(RIGHT, buff=MED_LARGE_BUFF)
        ibp_rule.to_edge(UP, buff=MED_LARGE_BUFF)

        lt_deriv_lines = VGroup(
            Tex(R"\\mathcal{L}\\big\\{f'({t})\\big\\} =\\int_0^\\infty f'({t}) e^{\\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"=\\Big[f({t}) e^{\\minus {s}{t}} \\Big]_0^\\infty - \\int_0^\\infty f({t}) (-{s}) e^{\\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"=-f(0) + {s} \\int_0^\\infty f({t}) e^{\\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"={s} \\mathcal{L}\\big\\{f({t})\\big\\} -f(0)", t2c=t2c),
        )
        lt_deriv_lines.arrange(DOWN, aligned_edge=LEFT)
        lt_deriv_lines[1:].align_to(lt_deriv_lines[0]["="], LEFT)
        lt_deriv_lines.next_to(ibp_rule, DOWN, LARGE_BUFF, aligned_edge=LEFT)

        # Show Definition
        definition = lt_deriv_lines[0]
        definition.save_state()
        definition.set_height(1.6).center()
        eq_index = definition.submobjects.index(definition["="][0][0])

        def_brace = Brace(definition[eq_index:], UP)
        def_words = Text("Definition", font_size=60)
        def_words.next_to(def_brace, UP)

        self.add(definition[:eq_index])
        self.wait()
        self.play(LaggedStart(
            Transform(*definition["f'({t})"].copy(), path_arc=60 * DEG, remover=True),
            GrowFromCenter(def_brace),
            FadeIn(def_words, 0.5 * UP),
            Write(definition[eq_index:], lag_ratio=0.1),
            lag_ratio=0.2
        ))
        self.wait()
        self.play(LaggedStart(
            Transform(
                definition[R"\\int"][0].copy(),
                ibp_rule[1][R"\\int"][0].copy(),
                remover=True,
                path_arc=-45 * DEG,
            ),
            FadeOut(def_brace),
            FadeOut(def_words),
            FadeIn(ibp_rule, lag_ratio=0.1),
            lag_ratio=0.2
        ))
        self.wait()

        # Show remaining lines
        self.play(Restore(definition))
        self.play(
            FadeTransform(ibp_rule[1][4:].copy(), lt_deriv_lines[1])
        )
        self.wait()
        for line in lt_deriv_lines[2:]:
            self.play(FadeIn(line, 0.5 * DOWN))
            self.wait()

        # Ask about times s and minus f(0)
        final_line = lt_deriv_lines[-1]
        s_rect = SurroundingRectangle(final_line["{s}"][0], buff=0.05)
        s_rect.set_stroke(YELLOW, 2)
        minus_f0_rect = SurroundingRectangle(final_line["-f(0)"][0], buff=0.05)
        minus_f0_rect.set_stroke(BLUE, 2)

        why = Text("Why?")
        why.next_to(s_rect, LEFT, buff=1.5).shift(0.5 * UP)
        why_arrow = Arrow(why.get_right() + SMALL_BUFF * RIGHT, s_rect.get_top(), path_arc=-90 * DEG, buff=0.05)
        VGroup(why, why_arrow).set_color(YELLOW)

        why2 = Text("Why?")
        why2.next_to(minus_f0_rect, RIGHT)
        why2.set_color(BLUE)

        self.play(
            Write(why),
            Write(why_arrow),
            ShowCreation(s_rect),
            lt_deriv_lines[1:3].animate.set_fill(opacity=0.3),
        )
        self.wait()
        self.play(
            ShowCreation(minus_f0_rect),
            Write(why2),
        )
        self.wait()


class FromPropertyToLaplaceTransform(InteractiveScene):
    def construct(self):
        # Add desired property
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        prop = VGroup(
            Tex(R"d / d{t}", font_size=72, t2c=t2c),
            Vector(DOWN, thickness=5),
            Tex(R"\\times {s}", font_size=72, t2c=t2c),
        )
        prop.arrange(DOWN)
        prop.to_edge(LEFT, buff=LARGE_BUFF)
        prop_rect = SurroundingRectangle(prop, buff=MED_LARGE_BUFF)
        prop_rect.set_stroke(GREEN, 2)
        goal_word = Text("Goal", font_size=72)
        goal_word.set_fill(GREEN)
        goal_word.next_to(prop_rect, UP)

        self.add(prop)
        self.play(
            ShowCreation(prop_rect),
            Write(goal_word)
        )

        # Show threads
        n_threads = 10
        threads = VGroup(
            self.get_thread(prop_rect.get_right() + shift)
            for shift in np.linspace(0.5 * UP, 0.5 * DOWN, n_threads)
        )
        kw = dict(
            lag_ratio=1.0 / n_threads,
            run_time=4
        )
        self.play(
            ShowCreation(threads, **kw),
            LaggedStartMap(VShowPassingFlash, threads.copy().set_stroke(WHITE, 2), time_width=2.0, **kw)
        )
        self.wait()

        # Transforms
        lt, inv_lt = transforms = VGroup(
            Tex(R"F({s}) = \\int_0^\\infty f({t}) e^{\\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"f({t}) = \\frac{1}{2\\pi i} \\int_{a - i\\infty}^{a + i\\infty} F({s}) e^{{s}{t}} d{s}", t2c=t2c),
        )

        transforms.arrange(DOWN, buff=2.5)
        transforms.to_edge(RIGHT)

        transform_rects = VGroup(SurroundingRectangle(term) for term in transforms)

        v_line = Line(*transform_rects, buff=0)
        h_line = Line(prop_rect.get_right(), v_line.get_center())
        VGroup(v_line, h_line, transform_rects).set_stroke(GREY, 1)

        self.play(
            Transform(threads, h_line.replicate(len(threads)), remover=True, lag_ratio=0.1, run_time=3)
        )
        self.add(h_line)
        self.play(GrowFromCenter(v_line, run_time=0.5))
        self.play(
            FadeIn(transform_rects),
            FadeIn(transforms),
        )
        self.wait()

        # Unify
        transforms.target = transforms.generate_target()
        transforms.target.space_out_submobjects(0.7)
        new_rect = SurroundingRectangle(transforms.target)
        new_rect.set_stroke(GREY, 2)

        self.play(
            MoveToTarget(transforms),
            ReplacementTransform(transform_rects[0], new_rect),
            ReplacementTransform(transform_rects[1], new_rect),
            v_line.animate.scale(0),
            h_line.animate.put_start_and_end_on(prop_rect.get_right(), new_rect.get_left()),
        )
        self.wait()

    def get_thread(self, start_point, step_size=0.15, angle_range=(-45 * DEG, 45 * DEG), n_steps=20):
        points = [start_point]
        for n in range(n_steps):
            step = rotate_vector(RIGHT, random.uniform(*angle_range))
            points.append(points[-1] + step)

        path = VMobject().set_points_smoothly(points, approx=False)
        path.set_stroke(GREY, 1)
        return path


class EndScreen(SideScrollEndScreen):
    pass`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports get_complex_graph from the _2025.laplace.integration module within the 3b1b videos codebase.",
      11: "ForcedOscillatorEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      12: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      13: "Creates a list of colors smoothly interpolated between the given endpoints.",
      23: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      32: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      33: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      34: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      35: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      36: "FadeOut transitions a mobject from opaque to transparent.",
      37: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      40: "MoveAroundPolesSeeDynamics extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      41: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      44: "ComplexPlane extends NumberPlane for complex number visualization. Points map to complex numbers directly.",
      55: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      56: "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion.",
      59: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      63: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      65: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      74: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      77: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      80: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      85: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      90: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      91: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      96: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      101: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      106: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      107: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      113: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      117: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      123: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      126: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      148: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      149: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      150: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      151: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      152: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      153: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      154: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      156: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      157: "Smoothly animates the camera to a new orientation over the animation duration.",
      160: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      165: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      169: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      171: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      172: "Clips geometry to a half-space defined by a normal vector and offset. Used for cross-section reveals.",
      182: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      184: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      186: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      187: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      188: "UpdateFromFunc calls a function on each frame to update a mobject's state.",
      191: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      194: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      195: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      196: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      197: "UpdateFromFunc calls a function on each frame to update a mobject's state.",
      200: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      204: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      216: "DerivativeFormula extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      219: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      223: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      224: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      225: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      226: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      237: "Returns the Euclidean length of a vector. ManimGL utility wrapping np.linalg.norm.",
      722: "PreviewStrategy extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      723: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      807: "Class WalkThroughEquationSolution inherits from PreviewStrategy.",
      808: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1377: "ShowSolutionPoles extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1378: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1441: "Class PartialFractionDecomposition inherits from WalkThroughEquationSolution.",
      1442: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1732: "IntegrateByParts extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1733: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1827: "FromPropertyToLaplaceTransform extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1828: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1918: "Class EndScreen inherits from SideScrollEndScreen.",
    }
  };

  files["_2025/laplace/exponentials.py"] = {
    description: "Exponential functions and their Laplace transforms. Demonstrates the fundamental relationship between e^(st) and the s-domain, building intuition for the transform.",
    code: `from manim_imports_ext import *
from _2025.laplace.shm import ShowFamilyOfComplexSolutions


S_COLOR = YELLOW
T_COLOR = BLUE


def get_exp_graph_icon(s, t_range=(0, 7), y_max=4, pos_real_scalar=0.1, neg_real_scalar=0.2, width=1, height=1):
    axes = Axes(
        t_range,
        (-y_max, y_max),
        width=width,
        height=height,
        axis_config=dict(tick_size=0.035, stroke_width=1)
    )
    scalar = pos_real_scalar if s.real > 0 else neg_real_scalar
    new_s = complex(s.real * scalar, s.imag)
    graph = axes.get_graph(lambda t: np.exp(new_s * t).real)
    graph.set_stroke(YELLOW, 2)
    rect = SurroundingRectangle(axes)
    rect.set_fill(BLACK, 1)
    rect.set_stroke(WHITE, 1)
    return VGroup(rect, axes, graph)


class IntroduceEulersFormula(InteractiveScene):
    def construct(self):
        # Add plane
        plane = ComplexPlane(
            (-2, 2), (-2, 2),
            width=6, height=6,
        )
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.5)
        plane.to_edge(LEFT)

        plane.add_coordinate_labels([1, -1])
        i_labels = VGroup(
            Tex(R"i", font_size=36).next_to(plane.n2p(1j), UL, SMALL_BUFF),
            Tex(R"-i", font_size=36).next_to(plane.n2p(-1j), DL, SMALL_BUFF),
        )
        plane.add(i_labels)

        self.add(plane)

        # Show pi
        pi_color = RED
        arc = Arc(0, PI, radius=plane.x_axis.get_unit_size(), arc_center=plane.n2p(0))
        arc.set_stroke(pi_color, 5)
        t_tracker = ValueTracker(0)
        t_dec = DecimalNumber(0)
        t_dec.set_color(pi_color)
        t_dec.add_updater(lambda m: m.set_value(t_tracker.get_value()))
        t_dec.add_updater(lambda m: m.move_to(plane.n2p(1.3 * np.exp(0.9 * t_tracker.get_value() * 1j))))

        pi = Tex(R"\\pi", font_size=72)
        pi.set_color(pi_color)
        pi.set_backstroke(BLACK, 3)

        self.play(
            ShowCreation(arc),
            t_tracker.animate.set_value(PI),
            VFadeIn(t_dec, time_span=(0, 1)),
            run_time=2
        )
        pi.move_to(t_dec, DR)
        self.play(
            FadeOut(t_dec),
            FadeIn(pi),
            run_time=0.5
        )

        # Write formula
        formula = Tex(R"e^{\\pi i} = -1", font_size=90, t2c={R"\\pi": RED, "i": BLUE})
        formula.set_x(FRAME_WIDTH / 4).to_edge(UP)
        cliche = Text("Cliché?", font_size=72)
        cliche.next_to(formula, DOWN, LARGE_BUFF)

        randy = Randolph(height=2)
        randy.next_to(plane, RIGHT, LARGE_BUFF, aligned_edge=DOWN)
        randy.body.set_backstroke(BLACK)

        self.play(LaggedStart(
            TransformFromCopy(pi, formula[R"\\pi"][0]),
            FadeTransform(i_labels[0].copy(), formula["i"][0]),
            Write(formula["="][0]),
            TransformFromCopy(plane.coordinate_labels[1], formula["-1"][0]),
            Write(formula["e"][0]),
            lag_ratio=0.2,
            run_time=3
        ))
        self.wait()

        self.play(
            LaggedStart(
                *(
                    # TransformFromCopy(formula[c1][0], cliche[c2][0])
                    FadeTransform(formula[c1][0].copy(), cliche[c2][0])
                    for c1, c2 in zip(
                        ["e", "1", "i", "e", R"\\pi", "e", "1"],
                        "Cliché?",
                    )
                ),
                lag_ratio=0.1,
                run_time=3
            ),
            VFadeIn(randy),
            randy.says("This again?", mode="sassy", bubble_direction=LEFT)
        )
        self.play(Blink(randy))
        self.add(cliche)

        # Show many thumbnails
        plane_group = VGroup(plane, arc, pi)
        plane_group.set_z_index(-1)
        thumbnails = Group(
            Group(ImageMobject(f"https://img.youtube.com/vi/{slug}/maxresdefault.jpg"))
            for slug in [
                "-dhHrg-KbJ0",  # Mathologer
                "f8CXG7dS-D0",  # Welch Labs
                "ZxYOEwM6Wbk",  # 3b1b
                "LE2uwd9V5vw",  # Khan Academy
                "CRj-sbi2i2I",  # Numberphile
                "v0YEaeIClKY",  # Other 3b1b
                "sKtloBAuP74",
                "IUTGFQpKaPU",  # Po shen lo
            ]
        )
        thumbnails.set_width(4)
        thumbnails.arrange(DOWN, buff=-0.8)
        thumbnails[4:].align_to(thumbnails, UP).shift(0.5 * DOWN)
        thumbnails.to_corner(UL)
        for n, tn in enumerate(thumbnails):
            tn.add_to_back(SurroundingRectangle(tn, buff=0).set_stroke(WHITE, 1))
            tn.shift(0.4 * n * RIGHT)

        self.play(
            FadeOut(randy.bubble, time_span=(0, 1)),
            randy.change("raise_left_hand", thumbnails).set_anim_args(time_span=(0, 1)),
            plane_group.animate.set_width(3.5).next_to(formula, DOWN, MED_LARGE_BUFF).set_anim_args(time_span=(0, 2)),
            FadeOut(cliche, 3 * RIGHT, lag_ratio=-0.02, time_span=(0.5, 2.0)),
            LaggedStartMap(FadeIn, thumbnails, shift=UP, lag_ratio=0.5),
            run_time=3
        )

        # Fail to explain
        thumbnails.generate_target()
        for tn, vect in zip(thumbnails.target, compass_directions(len(thumbnails))):
            vect[0] *= 1.5
            tn.set_height(1.75)
            tn.move_to(3 * vect)

        formula.generate_target()
        q_marks = Tex(R"???", font_size=90)
        VGroup(formula.target, q_marks).arrange(DOWN, buff=MED_LARGE_BUFF).center()

        self.play(
            MoveToTarget(thumbnails, lag_ratio=0.01, run_time=2),
            FadeOut(randy, DOWN),
            FadeOut(plane_group, DOWN),
            MoveToTarget(formula),
            Write(q_marks)
        )
        self.wait()
        self.play(LaggedStartMap(FadeOut, thumbnails, shift=DOWN, lag_ratio=0.5, run_time=4))
        self.wait()

        # Show constant meanings
        e_copy = formula["e"][0].copy()

        circle = Circle(radius=1)
        circle.to_edge(UP, buff=LARGE_BUFF)
        circle.set_stroke(WHITE, 1)
        arc = circle.copy().pointwise_become_partial(circle, 0, 0.5)
        arc.set_stroke(pi_color, 5)
        radius = Line(circle.get_center(), circle.get_right())
        radius.set_stroke(WHITE, 1)
        radius_label = Tex(R"1", font_size=24)
        radius_label.next_to(radius, DOWN, SMALL_BUFF)
        pi_label = Tex(R"\\pi").set_color(pi_color)
        pi_label.next_to(circle, UP, buff=SMALL_BUFF)
        circle_group = VGroup(circle, arc, radius_label, radius, pi_label)

        i_eq = Tex(R"i^2 = -1", t2c={"i": BLUE}, font_size=90)
        i_eq.move_to(circle).set_x(5)

        self.play(
            formula.animate.shift(DOWN),
            FadeOut(e_copy, 3 * UP + 5 * LEFT),
            FadeOut(q_marks, DOWN)
        )
        self.play(
            TransformFromCopy(formula[R"\\pi"][0], pi_label),
            LaggedStartMap(FadeIn, VGroup(circle, radius, radius_label)),
            ShowCreation(arc),
        )
        self.play(
            FadeTransform(formula["i"][0].copy(), i_eq["i"][0]),
            Write(i_eq[1:], time_span=(0.75, 1.75)),
        )
        self.wait()

        # Question marks over i
        i_rect = SurroundingRectangle(formula["i"], buff=0.05)
        i_rect.set_stroke(YELLOW, 2)
        q_marks = Tex(R"???", font_size=24)
        q_marks.match_color(i_rect)
        q_marks.next_to(i_rect, UP, SMALL_BUFF)

        self.play(
            ShowCreation(i_rect),
            FadeIn(q_marks, 0.25 * UP, lag_ratio=0.25)
        )
        self.wait()

        # Who cares (To overlay)
        frame = self.frame
        back_rect = FullScreenRectangle()
        back_rect.fix_in_frame()
        back_rect.set_z_index(-1),

        self.play(
            LaggedStartMap(FadeOut, VGroup(circle_group, i_eq, VGroup(i_rect, q_marks))),
            FadeOut(circle_group),
            frame.animate.set_y(-3.5),
            FadeIn(back_rect),
            formula.animate.set_fill(WHITE),
            run_time=2
        )
        self.wait()


class ExpGraph(InteractiveScene):
    def construct(self):
        # Set up graph
        axes = Axes((-1, 4), (0, 20), width=10, height=6)
        axes.to_edge(RIGHT)
        x_axis_label = Tex("t")
        x_axis_label.next_to(axes.x_axis.get_right(), UL, MED_SMALL_BUFF)
        axes.add(x_axis_label)

        graph = axes.get_graph(np.exp)
        graph.set_stroke(BLUE, 3)

        title = Tex(R"\\frac{d}{dt} e^t = e^t", t2c={"t": GREY_B}, font_size=60)
        title.to_edge(UP)
        title.match_x(axes.c2p(1.5, 0))

        self.add(axes)
        self.add(graph)
        self.add(title)

        # Add height tracker
        t_tracker = ValueTracker(1)
        get_t = t_tracker.get_value
        v_line = always_redraw(
            lambda: axes.get_v_line_to_graph(get_t(), graph, line_func=Line).set_stroke(RED, 3)
        )
        height_label = Tex(R"e^t", font_size=42)
        height_label.always.next_to(v_line, RIGHT, SMALL_BUFF)
        height_label_height = height_label.get_height()
        height_label.add_updater(lambda m: m.set_height(
            min(height_label_height, 0.7 * v_line.get_height())
        ))

        self.play(
            ShowCreation(v_line, suspend_mobject_updating=True),
            FadeIn(height_label, UP, suspend_mobject_updating=True),
        )
        self.wait()

        # Add tangent line
        tangent_line = always_redraw(
            lambda: axes.get_tangent_line(get_t(), graph, length=10).set_stroke(BLUE_A, 1)
        )
        unit_size = axes.x_axis.get_unit_size()
        unit_line = Line(axes.c2p(0, 0), axes.c2p(1, 0))
        unit_line.add_updater(lambda m: m.move_to(v_line.get_end(), LEFT))
        unit_line.set_stroke(WHITE, 2)
        unit_label = Integer(1, font_size=24)
        unit_label.add_updater(lambda m: m.next_to(unit_line.pfp(0.6), UP, 0.5 * SMALL_BUFF))
        tan_v_line = always_redraw(
            lambda: v_line.copy().shift(v_line.get_vector() + unit_size * RIGHT)
        )

        deriv_label = Tex(R"\\frac{d}{dt} e^t = e^t", font_size=42)
        deriv_label[R"\\frac{d}{dt}"].scale(0.75, about_edge=RIGHT)
        deriv_label_height = deriv_label.get_height()
        deriv_label.add_updater(lambda m: m.set_height(
            min(deriv_label_height, 0.8 * v_line.get_height())
        ))
        deriv_label.always.next_to(tan_v_line, RIGHT, SMALL_BUFF)

        self.play(ShowCreation(tangent_line, suspend_mobject_updating=True))
        self.play(
            VFadeIn(unit_line),
            VFadeIn(unit_label),
            VFadeIn(tan_v_line, suspend_mobject_updating=True),
            TransformFromCopy(title, deriv_label),
        )
        self.play(
            ReplacementTransform(v_line.copy().clear_updaters(), tan_v_line, path_arc=45 * DEG),
            FadeTransform(height_label.copy(), deriv_label["e^t"][1], path_arc=45 * DEG, remover=True),
        )
        self.wait()

        # Move it around
        for t in [2.35, 0, 1, 2]:
            self.play(t_tracker.animate.set_value(t), run_time=5)


class DefiningPropertyOfExp(InteractiveScene):
    def construct(self):
        # Key property
        tex_kw = dict(t2c={"{t}": GREY_B, "x": BLUE})
        equation = Tex(R"\\frac{d}{d{t}} e^{t} = e^{t}", font_size=90, **tex_kw)

        exp_parts = equation["e^{t}"]
        ddt = equation[R"\\frac{d}{d{t}}"]

        self.play(Write(exp_parts[0]))
        self.wait()
        self.play(FadeIn(ddt, scale=2))
        self.play(
            Write(equation["="]),
            TransformFromCopy(*exp_parts, path_arc=PI / 2),
        )
        self.wait()

        # Differential Equation
        ode = Tex(R"x'(t) = x(t)", font_size=72, **tex_kw)
        ode.move_to(equation).to_edge(UP)
        ode_label = Text("Differential\\nEquation", font_size=36)
        ode_label.next_to(ode, LEFT, LARGE_BUFF, aligned_edge=DOWN)

        self.play(
            FadeTransform(equation.copy(), ode),
            FadeIn(ode_label)
        )
        self.wait()

        # Initial condition
        frame = self.frame
        abs_ic = Tex(R"x(0) = 1", font_size=72, **tex_kw)
        exp_ic = Tex(R"e^{0} = 1", font_size=90, t2c={"0": GREY_B})
        abs_ic.next_to(ode, RIGHT, buff=2.0)
        exp_ic.match_x(abs_ic).match_y(equation).shift(0.1 * UP)
        ic_label = Text("Initial\\nCondition", font_size=36)
        ic_label.next_to(abs_ic, RIGHT, buff=0.75)

        self.play(
            FadeIn(abs_ic, RIGHT),
            FadeIn(exp_ic, RIGHT),
            frame.animate.set_x(2),
            Write(ic_label)
        )
        self.wait()

        # Scroll down
        self.play(frame.animate.set_y(-2.5), run_time=2)
        self.wait()


class ExampleExponentials(InteractiveScene):
    def construct(self):
        # Show the family
        pass

        # Highlight -1 + i term

        # Show e^t as its own derivative


class ImaginaryInputsToTheTaylorSeries(InteractiveScene):
    def construct(self):
        # Add complex plane
        plane = ComplexPlane(
            (-6, 6),
            (-4, 4),
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.set_height(5)
        plane.to_edge(DOWN, buff=0)
        plane.add_coordinate_labels(font_size=16)

        self.add(plane)

        # Add πi dot
        dot = GlowDot(color=YELLOW)
        dot.move_to(plane.n2p(PI * 1j))
        pi_i_label = Tex(R"\\pi i", font_size=30).set_color(YELLOW)
        pi_i_label.next_to(dot, RIGHT, buff=-0.1).align_to(plane.n2p(3j), DOWN)

        self.add(dot, pi_i_label)

        # Show false equation
        false_eq = Tex(R"e^x = e \\cdot e \\cdots e \\cdot e", t2c={"x": BLUE}, font_size=60)
        false_eq.to_edge(UP).shift(2 * LEFT)
        brace = Brace(false_eq[3:], DOWN)
        brace_tex = brace.get_tex(R"x \\text{ times}")
        brace_tex[0].set_color(BLUE)

        nonsense = TexText(R"Nonsense if $x$ \\\\ is complex")
        nonsense.next_to(VGroup(false_eq, brace_tex), RIGHT, LARGE_BUFF)
        nonsense.set_color(RED)

        self.add(false_eq)
        self.play(GrowFromCenter(brace), FadeIn(brace_tex, lag_ratio=0.1))
        self.play(FadeIn(nonsense, lag_ratio=0.1))
        self.wait()

        # Make it the real equation
        gen_poly = self.get_series("x")
        gen_poly.to_edge(LEFT).to_edge(UP, MED_SMALL_BUFF)

        epii = self.get_series(R"\\pi i", use_parens=True, in_tex_color=YELLOW)
        epii.next_to(gen_poly, DOWN, aligned_edge=LEFT)

        self.remove(false_eq)
        self.play(
            TransformFromCopy(false_eq[:2], gen_poly[0]),
            FadeOut(false_eq[2:], 0.5 * DOWN, lag_ratio=0.05),
            FadeOut(nonsense),
            FadeOut(brace, 0.5 * DOWN),
            FadeOut(brace_tex, 0.25 * DOWN),
            Write(gen_poly[1:])
        )
        self.wait()

        # Plug in πi
        vectors = self.get_spiral_vectors(plane, PI)
        buff = 0.5 * SMALL_BUFF
        labels = VGroup(
            Tex(R"\\pi i", font_size=30).next_to(vectors[1], RIGHT, buff),
            Tex(R"(\\pi^2 / 2) \\cdot i^2", font_size=30).next_to(vectors[2], UP, buff),
            Tex(R"(\\pi^3 / 6) \\cdot i^3", font_size=30).next_to(vectors[3], LEFT, buff),
            Tex(R"(\\pi^4 / 24) \\cdot i^4", font_size=30).next_to(vectors[4], DOWN, buff),
        )
        labels.set_color(YELLOW)
        labels.set_backstroke(BLACK, 5)

        for n in range(0, len(gen_poly), 2):
            anims = [
                LaggedStart(
                    TransformMatchingTex(gen_poly[n].copy(), epii[n], run_time=1),
                    TransformFromCopy(gen_poly[n + 1], epii[n + 1]),
                    gen_poly[n + 2:].animate.align_to(epii[n + 2:], LEFT),
                    lag_ratio=0.05
                ),
            ]
            k = (n - 1) // 2
            if k >= 0:
                anims.append(GrowArrow(vectors[k]))
            if k == 1:
                anims.append(FadeTransform(pi_i_label, labels[0]))
            elif 2 <= k <= len(labels):
                anims.append(FadeIn(labels[k - 1]))
            if k >= 1:
                anims.append(dot.animate.set_width(0.5).move_to(vectors[k].get_end()))
            self.play(*anims)
        for vector in vectors[7:]:
            self.play(GrowArrow(vector), dot.animate.move_to(vector.get_end()))
        self.wait()

        # Step through terms individually
        labels.add_to_back(VectorizedPoint().move_to(vectors[0]))
        for n in range(5):
            rect1 = SurroundingRectangle(epii[2 * n + 2])
            rect2 = SurroundingRectangle(VGroup(vectors[n], labels[n]))
            self.play(
                FadeIn(rect1),
                self.fade_all_but(epii, 2 * n + 2),
                self.fade_all_but(vectors, n),
                self.fade_all_but(labels, n),
                dot.animate.set_opacity(0.1),
            )
            self.play(Transform(rect1, rect2))
            self.play(FadeOut(rect1))
        self.play(*(
            mob.animate.set_fill(opacity=1)
            for mob in [epii, vectors, labels]
        ))
        self.wait()

        # Swap out i for t
        e_to_it = self.get_series("it", use_parens=True, in_tex_color=GREEN)
        for sm1, sm2 in zip(e_to_it, epii):
            sm1.move_to(sm2)

        t_tracker = ValueTracker(PI)
        get_t = t_tracker.get_value

        t_label = Tex(R"t = 3.14", t2c={"t": GREEN})
        t_label.next_to(e_to_it, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        t_rhs = t_label.make_number_changeable("3.14")
        t_rhs.add_updater(lambda m: m.set_value(get_t()))

        vectors.add_updater(lambda m: m.become(self.get_spiral_vectors(plane, get_t(), 20)))
        dot.f_always.move_to(vectors[-1].get_end)

        max_theta = TAU
        semi_circle = Arc(0, max_theta, radius=plane.x_axis.get_unit_size(), arc_center=plane.n2p(0))
        semi_circle.set_stroke(TEAL, 3)

        self.play(
            ReplacementTransform(epii, e_to_it, lag_ratio=0.01, run_time=2),
            FadeOut(labels),
            FadeIn(t_label),
        )
        self.add(vectors)
        self.play(t_tracker.animate.set_value(0), run_time=5)
        self.play(
            t_tracker.animate.set_value(max_theta),
            ShowCreation(semi_circle),
            run_time=12
        )
        self.play(t_tracker.animate.set_value(PI), run_time=6)
        self.wait()

    def get_series(self, in_tex="x", use_parens=False, in_tex_color=BLUE, buff=0.2):
        paren_tex = f"({in_tex})" if use_parens else in_tex
        kw = dict(t2c={in_tex: in_tex_color})
        terms = VGroup(
            Tex(fR"e^{{{in_tex}}}", **kw),
            Tex("="),
            Tex(fR"1"),
            Tex(R"+"),
            Tex(fR"{in_tex}", **kw),
            Tex(R"+"),
            Tex(fR"\\frac{{1}}{{2}} {paren_tex}^2", **kw),
            Tex(R"+"),
            Tex(fR"\\frac{{1}}{{6}} {paren_tex}^3", **kw),
            Tex(R"+"),
            Tex(fR"\\frac{{1}}{{24}} {paren_tex}^4", **kw),
            Tex(R"+"),
            Tex(R"\\cdots", **kw),
            Tex(R"+"),
            Tex(fR"\\frac{{1}}{{n!}} {paren_tex}^n", **kw),
            Tex(R"+"),
        )
        terms.arrange(RIGHT, buff=buff)
        terms[0].scale(1.25, about_edge=DR)
        return terms

    def get_spiral_vectors(
        self,
        plane,
        t,
        n_terms=10,
        # colors=[GREEN, YELLOW, GREEN_E, YELLOW_E]
        colors=[GREEN_E, GREEN_C, GREEN_B, GREEN_A],
    ):
        values = [(t * 1j)**n / math.factorial(n) for n in range(n_terms)]
        vectors = VGroup(
            Arrow(plane.n2p(0), plane.n2p(value), buff=0, fill_color=color)
            for value, color in zip(values, it.cycle(colors))
        )
        for v1, v2 in zip(vectors, vectors[1:]):
            v2.shift(v1.get_end() - v2.get_start())
        return vectors

    def fade_all_but(self, group, index):
        group.target = group.generate_target()
        group.target.set_fill(opacity=0.4)
        group.target[index].set_fill(opacity=1)
        return MoveToTarget(group)


class ComplexExpGraph(InteractiveScene):
    s_value = 1j
    orientation1 = (-77, -1, 0, (1.01, -0.1, 3.21), 7.55)
    orientation2 = (-33, -2, 0, (1.68, -0.09, 3.79), 10.88)

    def construct(self):
        # Set up parts
        self.set_floor_plane("xz")
        frame = self.frame

        plane = ComplexPlane((-2, 2), (-2, 2))
        plane.scale(0.75)
        moving_plane = plane.copy()

        t_axis = NumberLine((0, 12))
        t_axis.rotate(90 * DEG, DOWN)
        t_axis.shift(-t_axis.n2p(0))

        self.add(plane)
        self.add(t_axis)

        # Trackers and graph
        t_tracker = ValueTracker(0)
        get_t = t_tracker.get_value
        s = self.s_value

        def get_z():
            return np.exp(s * get_t())

        def z_to_point(z):
            return plane.n2p(z) + get_t() * OUT

        moving_plane.add_updater(lambda m: m.move_to(t_axis.n2p(get_t())))
        point = GlowDot()
        point.add_updater(lambda m: m.move_to(z_to_point(get_z())))
        vector = Vector()
        vector.add_updater(lambda m: m.put_start_and_end_on(
            z_to_point(0),
            z_to_point(get_z())
        ))
        graph = TracedPath(vector.get_end, stroke_color=TEAL)

        t_label = Tex("t = 0.00", font_size=30)
        t_label_rhs = t_label.make_number_changeable("0.00")
        t_label_rhs.add_updater(lambda m: m.set_value(get_t()))
        t_label.add_updater(lambda m: m.next_to(moving_plane, UP, SMALL_BUFF))

        self.add(t_tracker, moving_plane, vector, point, graph)
        self.add(t_label)
        frame.reorient(*self.orientation1)
        self.play(
            frame.animate.reorient(*self.orientation2),
            t_tracker.animate.set_value(12).set_anim_args(rate_func=linear),
            VFadeIn(t_axis, time_span=(0, 1)),
            run_time=12
        )
        self.play(
            frame.animate.reorient(0, -89, -90, (0.06, -0.62, 5.27), 7.42).set_field_of_view(1 * DEG),
            FadeOut(plane),
            FadeOut(moving_plane),
            FadeOut(t_label),
            FadeOut(point),
            FadeOut(vector),
            run_time=3
        )


class AltComplexExpGraph(ComplexExpGraph):
    s_value = -0.2 + 1j
    orientation1 = (-37, -1, 0, (0.08, 0.1, 0.08), 6)
    orientation2 = (-21, -5, 0, (1.47, -0.44, 3.88), 12.29)


class SPlane(InteractiveScene):
    tex_to_color_map = {"s": YELLOW, "t": BLUE, R"\\omega": PINK}
    s_plane_x_range = (-2, 2)
    s_label_font_size = 36
    s_label_config = dict(
        hide_zero_components_on_complex=True,
        include_sign=True,
        num_decimal_places=1,
    )

    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_path = self.get_output_path(exp_plane, get_t, get_s)

        self.add(exp_plane, exp_plane_label, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)

        self.add(axes, graph, v_line)

        # Move s around, end at i
        s_tracker.set_value(-1)
        self.play(s_tracker.animate.set_value(0.2), run_time=4)
        self.play(s_tracker.animate.set_value(0), run_time=2)
        self.play(s_tracker.animate.set_value(1j), run_time=3)
        self.wait()

        # Let time tick forward
        frame = self.frame
        self.play_time_forward(
            3 * TAU,
            added_anims=[frame.animate.set_x(3).set_height(12).set_anim_args(time_span=(6, 15))],
        )
        self.wait()
        self.play(
            t_tracker.animate.set_value(0),
            frame.animate.set_x(1.5).set_height(10),
            run_time=3
        )

        # Set s to 2i, then add vectors
        t2c = {"2i": YELLOW, **self.tex_to_color_map}
        exp_2it_label = Tex(R"e^{2i t}", t2c=t2c)
        exp_2it_label.move_to(exp_plane_label, RIGHT)

        self.play(s_tracker.animate.set_value(2j), run_time=2)
        self.play(
            FadeOut(exp_plane_label, 0.5 * UP),
            FadeIn(exp_2it_label, 0.5 * UP),
        )
        self.play_time_forward(TAU)

        # Show the derivative
        exp_plane.target = exp_plane.generate_target()
        exp_plane.target.align_to(axes.c2p(0, 0), LEFT)

        deriv_expression = Tex(R"\\frac{d}{dt} e^{2i t} = 2i \\cdot e^{2i t}", t2c=t2c)
        deriv_expression.next_to(exp_plane.target, RIGHT, aligned_edge=UP)

        self.play(
            LaggedStart(
                ReplacementTransform(exp_2it_label, deriv_expression["e^{2i t}"][0], path_arc=-90 * DEG),
                MoveToTarget(exp_plane),
                Write(deriv_expression[R"\\frac{d}{dt}"]),
                lag_ratio=0.2
            ),
            frame.animate.reorient(0, 0, 0, (1, 0, 0.0), 9.25),
        )
        self.play(LaggedStart(
            Write(deriv_expression["="]),
            TransformFromCopy(*deriv_expression["e^{2i t}"], path_arc=-90 * DEGREES),
            FadeTransform(deriv_expression["e^{2i t}"][0][1:3].copy(), deriv_expression[R"2i"][1], path_arc=-90 * DEG),
            Write(deriv_expression[R"\\cdot"]),
            lag_ratio=0.25,
            run_time=1.5
        ))
        self.add(deriv_expression)
        self.wait()

        # Step through derivative parts
        v_part, p_part, i_part, two_part = parts = VGroup(
            deriv_expression[R"\\frac{d}{dt} e^{2i t}"][0],
            deriv_expression[R"e^{2i t}"][1],
            deriv_expression[R"i"][1],
            deriv_expression[R"2"][1],
        )
        colors = [GREEN, BLUE, YELLOW, YELLOW]
        labels = VGroup(Text("Velocity"), Text("Position"), Tex(R"90^{\\circ}"), Text("Stretch"))
        for part, color, label in zip(parts, colors, labels):
            part.rect = SurroundingRectangle(part, buff=0.05)
            part.rect.set_stroke(color, 2)
            label.set_color(color)
            label.next_to(part.rect, DOWN)
            part.label = label

        p_vect, v_vect = vectors = self.get_pv_vectors(exp_plane, get_t, get_s)
        for vector in vectors:
            vector.suspend_updating()
        p_vect_copy = p_vect.copy().clear_updaters()

        self.play(
            ShowCreation(v_part.rect),
            FadeIn(v_part.label),
            GrowArrow(v_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(v_part.rect, p_part.rect),
            FadeTransform(v_part.label, p_part.label),
            GrowArrow(p_vect),
        )
        self.wait()
        self.play(
            ReplacementTransform(p_part.rect, i_part.rect),
            FadeTransform(p_part.label, i_part.label),
            p_vect_copy.animate.rotate(90 * DEG, about_point=exp_plane.n2p(0)).shift(p_vect.get_vector())
        )
        self.wait()
        self.play(
            ReplacementTransform(i_part.rect, two_part.rect),
            FadeTransform(i_part.label, two_part.label),
            Transform(p_vect_copy, v_vect, remover=True)
        )
        self.wait()
        self.play(FadeOut(two_part.rect), FadeOut(two_part.label))

        vectors.resume_updating()
        self.play_time_forward(TAU)

        # Label this angular frequency with omega
        imag_exp = Tex(R"e^{i \\omega t}", t2c=self.tex_to_color_map, font_size=60)
        imag_exp.move_to(deriv_expression, LEFT)

        self.play(
            FadeOut(deriv_expression, 0.5 * UP),
            FadeIn(imag_exp, 0.5 * UP),
        )
        t_tracker.set_value(0)
        output_path.suspend_updating()
        self.play(s_tracker.animate.set_value(1.5j), run_time=3)
        output_path.resume_updating()
        self.play_time_forward(TAU * 4 / 3)

        # Move to other complex values, end at -0.5 + i
        t_max_tracker = ValueTracker(20 * TAU)
        new_output_path = self.get_output_path(exp_plane, t_max_tracker.get_value, get_s)
        output_path.match_updaters(new_output_path)
        t_tracker.set_value(0)

        self.play(
            FadeOut(imag_exp, time_span=(0, 1)),
            s_tracker.animate.set_value(-0.2 + 1.5j),
            run_time=2
        )
        self.play(s_tracker.animate.set_value(-0.2 + 1j), run_time=2)
        self.play(s_tracker.animate.set_value(-0.5 + 1j), run_time=2)
        self.play_time_forward(TAU)

        # Split up the exponential to e^{-0.5t} * e^{it}
        t2c = {"-0.5": YELLOW, "i": YELLOW, "t": BLUE}
        lines = VGroup(
            Tex(R"e^{(-0.5 + i)t}", t2c=t2c),
            Tex(R"\\left(e^{-0.5t} \\right) \\left(e^{it} \\right)", t2c=t2c)
        )
        lines.arrange(DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        lines.next_to(exp_plane, RIGHT, LARGE_BUFF, aligned_edge=UP)

        dec_brace = Brace(lines[1][R"\\left(e^{-0.5t} \\right)"], DOWN, SMALL_BUFF)
        rot_brace = Brace(lines[1][R"\\left(e^{it} \\right)"], DOWN, SMALL_BUFF)
        dec_label = dec_brace.get_text("Decay")
        rot_label = rot_brace.get_text("Rotation")

        self.play(
            FadeIn(lines[0], time_span=(0.5, 1)),
            FadeTransform(s_label[-1].copy(), lines[0]["-0.5 + i"])
        )
        self.wait()
        self.play(
            TransformMatchingTex(lines[0].copy(), lines[1], run_time=1, lag_ratio=0.01)
        )
        self.wait()
        self.play(
            GrowFromCenter(dec_brace),
            FadeIn(dec_label)
        )
        self.wait()
        self.play(
            ReplacementTransform(dec_brace, rot_brace),
            FadeTransform(dec_label, rot_label),
        )
        self.wait()
        self.play(
            FadeOut(rot_brace),
            FadeOut(rot_label),
            t_tracker.animate.set_value(0).set_anim_args(run_time=3)
        )

        # Show multiplication by s
        s_vect = Arrow(s_plane.n2p(0), s_plane.n2p(get_s()), buff=0, fill_color=YELLOW)
        one_vect = Arrow(s_plane.n2p(0), s_plane.n2p(1), buff=0, fill_color=BLUE)
        arc = Line(
            s_plane.n2p(0.3),
            s_plane.n2p(0.3 * get_s()),
            path_arc=s_vect.get_angle(),
            buff=0.1
        )
        arc.add_tip(length=0.25, width=0.25)
        times_s_label = Tex(R"\\times s")
        times_s_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.play(LaggedStart(
            FadeIn(one_vect),
            FadeIn(arc),
            FadeIn(times_s_label),
            FadeIn(s_vect),
        ))
        self.wait()
        self.play(
            TransformFromCopy(one_vect, s_vect, path_arc=s_vect.get_angle()),
            run_time=2
        )
        self.wait()
        self.play(
            TransformFromCopy(one_vect, p_vect),
            TransformFromCopy(s_vect, v_vect),
            run_time=2
        )

        # Show spiraling inward
        self.play_time_forward(2 * TAU)

        self.play(FadeOut(VGroup(arc, times_s_label, lines)))
        t_tracker.set_value(0)

        s_vect.add_updater(lambda m: m.put_start_and_end_on(s_plane.n2p(0), s_plane.n2p(get_s())))
        self.add(s_vect)

        # Tour various values on the s plane
        values = [
            -0.1 + 2j,
            -0.1 - 2j,
            -0.1 + 0.5j,
            +0.05 + 0.5j,
            -0.5 + 0.5j,
            -0.1 + 0.5j,
        ]
        for value in values:
            self.play(s_tracker.animate.set_value(value), run_time=5)
            if value == values[0]:
                self.play_time_forward(TAU)
                self.t_tracker.set_value(0)

        self.play_time_forward(4 * TAU)

    def get_s_plane(self):
        s_plane = ComplexPlane(self.s_plane_x_range, self.s_plane_x_range)
        s_plane.set_width(7)
        s_plane.to_edge(LEFT, buff=SMALL_BUFF)
        s_plane.add_coordinate_labels(font_size=16)
        return s_plane

    def get_s_dot_and_label(self, s_plane, get_s):
        s_dot = Group(
            Dot(radius=0.05, fill_color=YELLOW),
            GlowDot(color=YELLOW),
        )
        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))

        s_label = Tex(R"s = +0.5", font_size=self.s_label_font_size)
        s_rhs = s_label.make_number_changeable("+0.5", **self.s_label_config)
        s_rhs.f_always.set_value(get_s)
        s_label.set_color(S_COLOR)
        s_label.set_backstroke(BLACK, 5)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        return Group(s_dot, s_label)

    def get_exp_plane(self, x_range=(-2, 2)):
        exp_plane = ComplexPlane(x_range, x_range)
        exp_plane.background_lines.set_stroke(width=1)
        exp_plane.faded_lines.set_stroke(opacity=0.25)
        exp_plane.set_width(4)
        exp_plane.to_corner(DR).shift(0.5 * LEFT)

        return exp_plane

    def get_exp_plane_label(self, exp_plane, font_size=60):
        label = Tex(R"e^{st}", font_size=font_size, t2c=self.tex_to_color_map)
        label.set_backstroke(BLACK, 5)
        label.next_to(exp_plane.get_corner(UL), DL, 0.2)
        return label

    def get_output_dot_and_label(self, exp_plane, get_s, get_t, label_direction=UR, s_tex="s"):
        output_dot = Group(
            TrueDot(color=GREEN),
            GlowDot(color=GREEN)
        )
        output_dot.add_updater(lambda m: m.move_to(exp_plane.n2p(np.exp(get_s() * get_t()))))

        output_label = Tex(Rf"e^{{{s_tex} \\cdot 0.00}}", font_size=36, t2c=self.tex_to_color_map)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        s_term = output_label[s_tex][0][0]
        t_label.set_height(s_term.get_height() * 1.2, about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        t_label.always.match_y(s_term, DOWN)
        output_label.always.next_to(output_dot, label_direction, buff=SMALL_BUFF, aligned_edge=LEFT, index_of_submobject_to_align=0),
        output_label.set_backstroke(BLACK, 3)

        return Group(output_dot, output_label)

    def get_graph_axes(
        self,
        x_range=(0, 24),
        y_range=(-2, 2),
        width=15,
        height=2,
    ):
        axes = Axes(x_range, y_range, width=width, height=height)
        x_axis_label = Tex(R"t", font_size=36, t2c=self.tex_to_color_map)
        y_axis_label = Tex(R"\\text{Re}\\left[e^{st}\\right]", font_size=36, t2c=self.tex_to_color_map)
        x_axis_label.next_to(axes.x_axis.get_right(), UP, buff=0.15)
        y_axis_label.next_to(axes.y_axis.get_top(), UP, SMALL_BUFF)
        axes.add(x_axis_label)
        axes.add(y_axis_label)
        axes.next_to(ORIGIN, RIGHT, MED_LARGE_BUFF)
        axes.to_edge(UP, buff=0.5)
        x_axis_label.shift_onto_screen(buff=MED_LARGE_BUFF)
        return axes

    def get_dynamic_exp_graph(self, axes, get_s, delta_t=0.1, stroke_color=TEAL, stroke_width=3):
        graph = Line().set_stroke(stroke_color, stroke_width)
        t_samples = np.arange(*axes.x_range[:2], 0.1)

        def update_graph(graph):
            s = get_s()
            values = np.exp(s * t_samples)
            xs = values.astype(np.complex128).real
            graph.set_points_smoothly(axes.c2p(t_samples, xs))

        graph.add_updater(update_graph)
        return graph

    def get_graph_v_line(self, axes, get_t, get_s):
        v_line = Line(DOWN, UP)
        v_line.set_stroke(WHITE, 2)
        v_line.f_always.put_start_and_end_on(
            lambda: axes.c2p(get_t(), 0),
            lambda: axes.c2p(get_t(), np.exp(get_s() * get_t()).real),
        )
        return v_line

    def get_output_path(self, exp_plane, get_t, get_s, delta_t=1 / 30, color=TEAL, stroke_width=2):
        path = VMobject()
        path.set_points([ORIGIN])
        path.set_stroke(color, stroke_width)

        def get_path_points():
            t_range = np.arange(0, get_t(), delta_t)
            if len(t_range) == 0:
                t_range = np.array([0])
            values = np.exp(t_range * get_s())
            return np.array([exp_plane.n2p(z) for z in values])

        path.f_always.set_points_smoothly(get_path_points)
        return path

    def play_time_forward(self, time, added_anims=[]):
        self.t_tracker.set_value(0)
        self.play(
            self.t_tracker.animate.set_value(time).set_anim_args(rate_func=linear),
            *added_anims,
            run_time=time,
        )

    def get_pv_vectors(self, exp_plane, get_t, get_s, thickness=3, colors=[BLUE, YELLOW]):
        p_vect = Vector(RIGHT, fill_color=colors[0], thickness=thickness)
        v_vect = Vector(RIGHT, fill_color=colors[1], thickness=thickness)
        p_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(np.exp(get_t() * get_s()))
        ))
        v_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(get_s() * np.exp(get_t() * get_s()))
        ).shift(p_vect.get_vector()))

        return VGroup(p_vect, v_vect)

    ###

    def setup_for_square_frame(self):
        # For an insert
        axes.next_to(s_plane, UP, LARGE_BUFF, aligned_edge=LEFT)
        exp_plane.match_height(s_plane).next_to(s_plane, RIGHT, buff=1.5)
        exp_plane_label.set_height(1).next_to(exp_plane, RIGHT, aligned_edge=UP)
        output_label.set_fill(opacity=0).set_stroke(opacity=0)
        self.add(exp_plane_label)

        axes[:-2].stretch(2.5, 1, about_edge=DOWN)
        axes.x_axis.ticks.stretch(1 / 2.5, 1)
        axes[-2].next_to(axes.x_axis.get_right(), UP, MED_SMALL_BUFF)
        axes[-1].next_to(axes.y_axis.get_top(), RIGHT, MED_SMALL_BUFF)

        self.frame.reorient(0, 0, 0, (1.0, 2.68, 0.0), 16.00)

    def old_material(self):
        # Collapse the graph
        output_dot = GlowDot(color=GREEN)
        output_dot.move_to(axes.x_axis.n2p(1))
        output_label = Tex(R"e^{s \\cdot 0.00}", **tex_kw, font_size=36)
        t_tracker.set_value(0)
        t_label = output_label.make_number_changeable("0.00")
        t_label.set_color(BLUE)
        t_label.match_height(output_label["s"], about_edge=LEFT)
        t_label.f_always.set_value(get_t)
        output_label.add_updater(lambda m: m.next_to(output_dot, UP, SMALL_BUFF, LEFT).shift(0.2 * DR))

        graph.clear_updaters()
        self.remove(axes)
        self.play(LaggedStart(
            FadeOut(VGroup(axes.x_axis, x_axis_label, graph)),
            AnimationGroup(
                Rotate(axes.y_axis, -90 * DEG),
                TransformMatchingTex(y_axis_label, output_label, run_time=1.5),
            ),
            FadeIn(output_dot),
            lag_ratio=0.5
        ))


class FamilyOfRealExp(InteractiveScene):
    def construct(self):
        # Graphs
        axes = Axes((-1, 8), (-1, 5))
        axes.set_height(FRAME_HEIGHT - 1)

        s_tracker = ValueTracker(0.5)
        get_s = s_tracker.get_value
        graph = axes.get_graph(lambda t: np.exp(t))
        graph.set_stroke(BLUE)
        axes.bind_graph_to_func(graph, lambda t: np.exp(get_s() * t))

        label = Tex(R"e^{st}", font_size=90)
        label.move_to(UP)
        label["s"].set_color(YELLOW)

        self.add(axes, label)
        self.play(ShowCreation(graph, suspend_mobject_updating=True))
        self.play(
            s_tracker.animate.set_value(-1),
            graph.animate.set_color(YELLOW),
            run_time=4
        )
        self.wait()


class ForcedOscillatorSolutionForm(InteractiveScene):
    def construct(self):
        # Create linear combination
        exp_texs = [Rf"e^{{s_{n} t}}" for n in range(1, 5)]
        const_texs = [Rf"c_{n}" for n in range(1, 5)]
        terms = [" ".join(pair) for pair in zip(const_texs, exp_texs)]
        solution = Tex("x(t) = " + " + ".join(terms), isolate=[*exp_texs, *const_texs])
        solution.to_edge(RIGHT)

        solution[re.compile(r's_\\w+')].set_color(YELLOW)
        solution[re.compile(r'c_\\w+')].set_color(BLUE)

        cut_index = solution.submobjects.index(solution["+"][1][0])
        first_two = solution[:cut_index]
        last_two = solution[cut_index:]

        first_two.save_state()
        first_two.to_edge(RIGHT, buff=1.5)

        self.add(first_two)

        # Not this
        ex_mark = Exmark(font_size=72).set_color(RED)
        checkmark = Checkmark(font_size=72).set_color(GREEN)
        ex_mark.next_to(first_two, UP, MED_LARGE_BUFF, aligned_edge=LEFT)
        checkmark.next_to(first_two.saved_state, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        nope = Text("Nope!", font_size=60).set_fill(border_width=4)
        nope.match_color(ex_mark)
        nope.next_to(ex_mark, RIGHT)

        actually = Text("Actually...", font_size=60)
        actually.set_fill(border_width=2)
        actually.match_color(checkmark)
        actually.next_to(checkmark, RIGHT, SMALL_BUFF, aligned_edge=DOWN, index_of_submobject_to_align=0)

        self.play(Write(ex_mark), Write(nope))

        # Freely tune coefficients
        c_trackers = ValueTracker(0).replicate(2)

        def get_c_values():
            return [tracker.get_value() for tracker in c_trackers]

        number_lines = VGroup(
            NumberLine((-3, 3), width=2).rotate(90 * DEG).next_to(solution[c_tex], DOWN)
            for c_tex in const_texs[:2]
        )
        for line in number_lines:
            line.set_width(0.1, stretch=True)
            line.add_numbers(font_size=12, direction=LEFT, buff=0.1)

        tips = ArrowTip().rotate(PI).set_height(0.2).replicate(2)
        tips.set_color(BLUE)

        def update_tips(tips):
            for tip, line, value in zip(tips, number_lines, get_c_values()):
                tip.move_to(line.n2p(value), LEFT)
            return tips

        tips.add_updater(update_tips)

        c_labels = VGroup(DecimalNumber(0, font_size=24) for _ in range(2))

        def update_c_labels(c_labels):
            for label, tip, value in zip(c_labels, tips, get_c_values()):
                label.set_value(value)
                label.next_to(tip, RIGHT, SMALL_BUFF)

        c_labels.add_updater(update_c_labels)

        def random_tuning_animation(run_time=2, lag_ratio=0.25):
            return LaggedStart(
                *(
                    tracker.animate.set_value(random.uniform(-3, 3))
                    for tracker in c_trackers
                ),
                lag_ratio=lag_ratio,
                run_time=run_time,
            )

        self.play(
            FadeIn(number_lines),
            VFadeIn(tips),
            VFadeIn(c_labels),
            random_tuning_animation()
        )
        for _ in range(6):
            self.play(random_tuning_animation())
        self.wait()

        # Show four particulcar exponentials
        plane = ComplexPlane((-3, 3), (-2, 2))
        plane.set_height(3.25)
        plane.to_corner(UL)
        plane.add_coordinate_labels(font_size=16)
        plane.coordinate_labels[-1].set_opacity(0)

        s_values = [1.5j, -1.5j, -0.3 + 1.0j, -0.3 - 1.0j]
        s_dots = Group(
            GlowDot(plane.n2p(s))
            for s in s_values
        )
        s_labels = VGroup(
            Tex(Rf"s_{n}", font_size=24).set_color(YELLOW).next_to(dot, vect, buff=-0.1)
            for n, dot, vect in zip(it.count(1), s_dots, [RIGHT, RIGHT, LEFT, LEFT])
        )

        self.play(LaggedStart(
            FadeOut(number_lines, lag_ratio=0.1),
            FadeOut(tips, lag_ratio=0.1),
            FadeOut(c_labels, lag_ratio=0.1),
            FadeOut(VGroup(ex_mark, nope), LEFT),
            FadeIn(VGroup(checkmark, actually), LEFT),
            Restore(first_two),
            FadeIn(last_two, LEFT),
            run_time=2
        ))
        self.play(
            FadeIn(plane),
            LaggedStartMap(FadeIn, s_dots),
            LaggedStart(
                *(
                    FadeTransform(solution[f"s_{n + 1}"].copy(), s_labels[n])
                    for n in range(4)
                )
            ),
        )
        self.wait()

        # Comment on constants
        const_rects = VGroup(
            SurroundingRectangle(solution[c_tex], buff=0.075)
            for c_tex in const_texs
        )
        const_rects.set_stroke(BLUE, 2)

        underlines = VGroup(
            Line(c1.get_bottom(), c2.get_bottom(), path_arc=40 * DEG)
            for c1, c2 in it.combinations(const_rects, 2)
        )
        underlines.set_stroke(TEAL, 2)
        underlines.insert_n_curves(10)

        underlines = VGroup(
            Vector(0.75 * UP, thickness=4).next_to(rect, DOWN, buff=0)
            for rect in const_rects
        )
        underlines.set_fill(BLUE)

        constraint_words = TexText("Only specific $c_n$ work")
        constraint_words.set_fill(BLUE, border_width=1)
        constraint_words.match_width(underlines)
        constraint_words.next_to(underlines, DOWN, buff=SMALL_BUFF)

        self.play(
            FadeIn(constraint_words, lag_ratio=0.1),
            FadeOut(checkmark),
            FadeOut(actually),
            LaggedStartMap(ShowCreation, const_rects, lag_ratio=0.25),
            LaggedStartMap(GrowArrow, underlines),
        )
        self.play(FadeOut(const_rects, lag_ratio=0.1))

        # Add exponential parts
        if False:
            # For an insertion
            for term, s in zip(exp_texs, s_values):
                exp_diagram = self.get_exponential_diagram(solution[term], s)
                self.add(exp_diagram)
            self.wait(24)

        # Ask about each part
        term_rects = VGroup(
            SurroundingRectangle(solution[term], buff=0.1).set_stroke(TEAL, 2)
            for term in terms
        )
        s_rects = VGroup(
            SurroundingRectangle(solution[exp_tex][0][1:3], buff=0.05).set_stroke(YELLOW, 2)
            for exp_tex in exp_texs
        )

        moving_rects = const_rects.copy()
        self.remove(const_rects)

        anim_kw = dict(lag_ratio=0.25, run_time=1.5)
        self.play(
            FadeOut(constraint_words),
            FadeOut(underlines),
            Transform(moving_rects, term_rects, **anim_kw)
        )
        self.wait()
        self.play(Transform(moving_rects, s_rects, **anim_kw))
        self.wait()
        self.play(Transform(moving_rects, const_rects, **anim_kw))
        self.wait()
        self.play(FadeOut(moving_rects, **anim_kw))

    def get_exponential_diagram(self, term, s, c=1.0, color=PINK):
        plane = ComplexPlane((-1, 1), (-1, 1))
        plane.set_width(1.25)
        plane.next_to(term, UP)

        t_tracker = ValueTracker()
        get_t = t_tracker.get_value
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        vector = Vector(thickness=2, fill_color=color)
        vector.add_updater(lambda m: m.put_start_and_end_on(
            plane.n2p(0),
            plane.n2p(c * np.exp(s * get_t())),
        ))

        tail = TracingTail(vector.get_end, stroke_color=color, time_traced=2, stroke_width=(0, 4))
        path = TracedPath(vector.get_end, stroke_color=color, stroke_width=1, stroke_opacity=0.75)

        return Group(plane, t_tracker, vector, tail, path)


class BreakingDownFunctions(ForcedOscillatorSolutionForm):
    def construct(self):
        # A s plane on the left, output plane on the right
        s_plane, out_plane = planes = VGroup(
            ComplexPlane((-2, 2), (-2, 2)),
            ComplexPlane((-2, 2), (-2, 2)),
        )
        for plane in planes:
            plane.set_height(5)
            plane.add_coordinate_labels(font_size=16)

        out_plane.center().to_edge(DOWN)
        s_plane.move_to(out_plane).to_edge(LEFT)

        self.add(out_plane)

        # Write a function as a sum, above the output plane
        n_range = list(range(1, 6))
        exp_texs = [Rf"e^{{s_{n} t}}" for n in n_range]
        const_texs = [Rf"c_{n}" for n in n_range]
        terms = [" ".join(pair) for pair in zip(const_texs, exp_texs)]
        solution = Tex("x(t) = " + " + ".join(terms), isolate=[*exp_texs, *const_texs])

        solution[re.compile(r's_\\w+')].set_color(YELLOW)
        solution[re.compile(r'c_\\w+')].set_color(BLUE)

        solution.next_to(out_plane, UP)

        self.add(solution)

        # Exp animations
        s_values = [-0.2 + 2j, -0.2 - 2j, -0.1 + 1j, -0.1 - 1j, -0.2]
        c_values = [1j, -1j, 0.8, 0.8, -0.75]
        colors = color_gradient([PINK, MAROON_B], len(n_range), interp_by_hsl=False)
        exp_diagrams = Group()
        for term, s, c, color in zip(terms, s_values, c_values, colors):
            part = solution[term]
            if len(part) == 0:
                continue
            exp_diagram = self.get_exponential_diagram(part, s, c, color)
            self.add(*exp_diagram)
            exp_diagrams.add(exp_diagram)

        # Set up the output
        center_point = VectorizedPoint(out_plane.n2p(0))
        all_vects = Vector().replicate(len(n_range))
        scale_factor = out_plane.x_axis.get_unit_size() / exp_diagrams[0][0].x_axis.get_unit_size()

        for diagram, vect, previous in zip(exp_diagrams, all_vects, [center_point, *all_vects]):
            vect.clone = diagram[2]
            vect.previous = previous
            vect.add_updater(lambda m: m.become(m.clone).scale(scale_factor))
            vect.add_updater(lambda m: m.shift(m.previous.get_end() - m.get_start()))

        self.add(all_vects)

        # Add output graph
        graph = VMobject()
        graph.set_stroke(RED, 3)
        graph.start_new_path(all_vects[-1].get_end())

        def update_graph(graph, dt):
            graph.shift(0.25 * dt * DOWN)
            graph.add_line_to(all_vects[-1].get_end())

        graph.add_updater(update_graph)

        self.add(graph)

        self.wait(30)
        self.play(VFadeOut(graph), VFadeOut(all_vects))
        exp_diagrams.suspend_updating()

        # Collapse to the s plane
        out_plane.generate_target()
        plane_group = VGroup(s_plane, out_plane.target)
        plane_group.arrange(RIGHT, buff=2)
        plane_group.to_edge(DOWN)
        compact_equation = Tex(
            R"x(t) = \\sum_{n=1}^{N} c_n e^{{s_n} t}",
            t2c={"c_n": BLUE, "s_n": YELLOW},
            isolate=["n=1", "N"]
        )
        compact_equation.next_to(out_plane.target, UP)
        compact_equation_start = compact_equation[:-4]

        s_dot = GlowDot()
        s_dot.move_to(s_plane.n2p(s_values[2]))
        s_label = Tex(R"s").set_color(YELLOW)
        s_label.always.next_to(s_dot, UL, buff=-0.1)
        s_plane_title = Text(R"S-plane", font_size=60)
        s_plane_title.next_to(s_plane, UP)
        s_plane_title.set_color(YELLOW)

        exp_graph = VMobject()
        exp_graph.set_stroke(PINK, 2)

        def update_exp_graph(exp_graph):
            s = s_plane.p2n(s_dot.get_center())
            anchors = np.array([
                out_plane.n2p(np.exp(s * t))
                for t in np.arange(0, 100, 0.1)
            ])
            exp_graph.set_points_as_corners(anchors)

        exp_graph.add_updater(update_exp_graph)

        self.play(LaggedStart(
            MoveToTarget(out_plane),
            LaggedStart(*(
                exp_diagram.animate.scale(scale_factor).move_to(out_plane.target)
                for exp_diagram in exp_diagrams
            ), lag_ratio=0.01),
            AnimationGroup(*(
                ReplacementTransform(solution[t1], compact_equation[t2])
                for t1, t2 in [
                    ("x(t) = ", "x(t) = "),
                    (re.compile(r'c_\\w+'), "c_n"),
                    (re.compile(r's_\\w+'), "s_n"),
                    ("e", "e"),
                    ("t", "t"),
                    ("+", R"\\sum_{n=1}^{N}"),
                ]
            )),
            lag_ratio=0.2,
            run_time=2
        ))
        exp_graph.update()
        self.play(
            FadeIn(s_plane),
            TransformFromCopy(compact_equation["s_n"][0], s_label),
            FadeTransform(compact_equation["s_n"][0].copy(), s_dot),
            Write(s_plane_title),
            FadeOut(exp_diagrams[2]),
            FadeIn(exp_graph, suspend_mobject_updating=True),
            compact_equation_start.animate.set_opacity(0.4)
        )
        self.remove(exp_diagrams)
        self.wait()

        # Growth, decay and oscillation
        arrows = VGroup(
            Arrow(s_plane.n2p(0), s_plane.n2p(z), thickness=4, fill_color=GREY_A)
            for z in [2, -2, 2j, -2j]
        )
        arrows.set_fill(GREY_A, 1)
        arrow_labels = VGroup(
            Text("Growth", font_size=36).next_to(arrows[0], UP, buff=0),
            Text("Decay", font_size=36).next_to(arrows[1], UP, buff=0),
            Text("Oscillation", font_size=36).rotate(-90 * DEG).next_to(arrows[3], RIGHT, buff=SMALL_BUFF),
        )
        rot_vect = Vector(2 * RIGHT, fill_color=RED, thickness=5)
        rot_vect.shift(out_plane.n2p(0) - rot_vect.get_start())
        rot_vect.rotate(-45 * DEG, about_point=out_plane.n2p(0))
        rot_vect.add_updater(lambda m, dt: m.rotate(2 * dt, about_point=out_plane.n2p(0)))

        self.play(
            s_dot.animate.shift(0.2 * RIGHT).set_anim_args(run_time=1),
            GrowArrow(arrows[0]),
            FadeIn(arrow_labels[0]),
        )
        self.play(
            s_dot.animate.shift(0.2 * LEFT).set_anim_args(run_time=1),
            GrowArrow(arrows[1]),
            FadeIn(arrow_labels[1]),
        )
        self.play(
            # s_dot.animate.shift(3 * DOWN).set_anim_args(run_time=4, rate_func=there_and_back),
            GrowArrow(arrows[2]),
            GrowArrow(arrows[3]),
            FadeIn(arrow_labels[2]),
            VFadeIn(rot_vect)
        )
        self.wait()
        self.play(VFadeOut(rot_vect))

        # Show multiple s
        frame = self.frame
        s_values[:2] = [-1.5 + 0.5j, -1.5 - 0.5j]
        s_values.extend([-0.8 + 1.5j, -0.8 - 1.5j])
        s_dots = Group(GlowDot(s_plane.n2p(s)) for s in s_values)

        self.play(
            FadeOut(arrows),
            FadeOut(arrow_labels),
            FadeOut(out_plane),
            FadeOut(exp_graph),
            FadeIn(s_dots, lag_ratio=0.5),
            FadeOut(s_dot),
            FadeOut(s_label),
            compact_equation.animate.set_height(2.0).set_opacity(1).next_to(s_plane, RIGHT, LARGE_BUFF),
            frame.animate.match_y(s_plane),
        )
        self.wait()

        # Infinite
        inf = Tex(R"\\infty", font_size=60)
        N = compact_equation["N"][0]
        inf.move_to(N)

        dot_line = Group(
            GlowDot(s_plane.n2p(complex(-0.5, b)))
            for b in np.linspace(-2, 2, 25)
        )

        self.play(
            FlashAround(N, time_width=1),
            # Transform(N, inf, path_arc=20 * DEG),
            FadeTransform(N, inf, path_arc=20 * DEG),
            ShowIncreasingSubsets(dot_line, run_time=5, rate_func=linear),
            ReplacementTransform(s_dots, dot_line[:len(s_dots)].copy().set_opacity(0))
        )

        # Continuous range
        integral_eq = Tex(
            R"x(t) = \\int_{\\gamma} c(s) e^{st} ds",
            t2c={"s": YELLOW, R"\\gamma": YELLOW}
        )
        integral_eq.replace(compact_equation, dim_to_match=1)
        integral_eq.shift(0.5 * RIGHT)
        line = Line(dot_line.get_bottom(), dot_line.get_top())
        line.set_stroke(YELLOW, 2)
        thick_line = line.copy().set_stroke(width=6)
        thick_line.insert_n_curves(100)

        self.play(
            LaggedStart(*(
                FadeTransform(compact_equation[t1], integral_eq[t2])
                for t1, t2 in [
                    ("x(t) = ", "x(t) ="),
                    (R"\\sum_{n=1}^{N}", R"\\int_{\\gamma}"),
                    (R"n=1", R"\\gamma"),
                    ("c_n", "c(s)"),
                    (R"e^{{s_n} t}", R"e^{st}"),
                ]
            )),
            FadeIn(integral_eq[R"ds"]),
            LaggedStartMap(FadeOut, dot_line, lag_ratio=0.5, scale=0.25, time_span=(0.3, 2)),
            VShowPassingFlash(thick_line, run_time=2),
            ShowCreation(line, run_time=2),
        )
        self.wait()


class Thumbnail(InteractiveScene):
    def construct(self):
        # Spiral
        spiral_color = TEAL
        s = -0.15 + 2j
        max_t = 3
        thick_stroke_width = (5, 25)

        plane = ComplexPlane(
            (-4, 4),
            (-2, 2)
        )
        plane.set_height(11)
        plane.center()
        plane.add_coordinate_labels(font_size=24)

        plane.axes.set_stroke(WHITE, 5)
        plane.background_lines.set_stroke(BLUE, 3)
        plane.faded_lines.set_stroke(BLUE, 2, 0.25)

        curve = ParametricCurve(
            lambda t: plane.n2p(np.exp(s * t)),
            t_range=(0, 40, 0.1)
        )
        partial_curve = ParametricCurve(
            lambda t: plane.n2p(np.exp(s * t)),
            t_range=(0, max_t, 0.1)
        )
        curve.set_stroke(spiral_color, 2, 0.5)
        partial_curve.set_stroke(spiral_color, width=thick_stroke_width, opacity=(0.5, 1))

        dot = Group(TrueDot(radius=0.2), GlowDot(radius=0.75))
        dot.set_color(spiral_color)
        dot.move_to(partial_curve.get_end())

        self.add(plane, curve)
        self.add(partial_curve)
        self.add(dot)

        vectors = VGroup(
            self.get_vector(plane, s, t, color=BLUE)
            for t in np.linspace(0, max_t, 10)
        )
        self.add(vectors)

    def get_vector(self, plane, s, t, scale_factor=0.5, thickness=5, color=YELLOW):
        vect = Vector(RIGHT, thickness=thickness, fill_color=color)
        vect.put_start_and_end_on(plane.n2p(0), scale_factor * plane.n2p(s * np.exp(s * t)))
        vect.shift(plane.n2p(np.exp(s * t)) - plane.n2p(0))
        return vect

    def get_formula(self):
        # Formula
        formula = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=200)
        formula.next_to(plane, LEFT, buff=2.0)

        s_rect = SurroundingRectangle(formula["s"], buff=0.1)
        s_rect.set_stroke(WHITE, 2)

        abi = Tex("a + bi", font_size=72)
        abi.next_to(formula, UP, LARGE_BUFF)
        abi.to_edge(LEFT, buff=LARGE_BUFF)

        arrow = Arrow(s_rect.get_top(), abi.get_corner(DR), buff=0.05)

        self.add(formula)
        self.add(s_rect)
        self.add(arrow)
        self.add(abi)


class Thumbnail2(InteractiveScene):
    def construct(self):
        # Test
        theta = 140 * DEG
        z = np.exp(theta * 1j)

        plane_color = BLUE
        path_color = YELLOW
        vect_color = WHITE

        plane = ComplexPlane(
            (-4, 4),
            (-2, 2)
        )
        plane.set_height(10)
        plane.center()
        plane.add_coordinate_labels(font_size=24)
        plane.axes.set_stroke(WHITE, 3)
        plane.background_lines.set_stroke(plane_color, 4)
        plane.faded_lines.set_stroke(plane_color, 3, 0.35)

        unit_size = plane.x_axis.get_unit_size()

        arrow = Arrow(
            plane.n2p(1),
            plane.n2p(z),
            buff=0,
            path_arc=theta,
            thickness=12,
            fill_color=YELLOW,
        )
        arrow.scale(0.965, about_point=plane.n2p(1))
        arrow.set_fill(border_width=3)

        path = Arc(0, theta, radius=unit_size)
        path.set_stroke(path_color, width=(2, 30))
        dot = Group(TrueDot(radius=0.2), GlowDot(radius=0.75))
        dot.set_color(path_color)
        dot.move_to(path.get_end())

        n_vects = 8
        vectors = VGroup(
            Vector(2.5 * UP, thickness=8).set_fill(vect_color, opacity**2).put_start_on(plane.n2p(1)).rotate(phi, about_point=plane.n2p(0))
            for phi, opacity in zip(np.linspace(0, theta, n_vects), np.linspace(0.5, 1, n_vects))
        )
        vectors.set_fill(border_width=3)

        circle = Circle(radius=unit_size)
        circle.set_stroke(GREY, 5)

        d_line = DashedLine(plane.n2p(0), plane.n2p(z))
        d_line.set_stroke(WHITE, 3)
        arc = Arc(0, theta, radius=0.5)
        arc.set_stroke(WHITE, 5)
        theta_label = Tex(R"\\theta", font_size=72)
        theta_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)

        self.add(plane)
        self.add(circle)
        self.add(d_line)
        self.add(vectors)
        self.add(path)
        self.add(dot)
        self.add(arc)
        self.add(theta_label)


### For Main Laplace video


class RecapSPlane(SPlane):
    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_path = self.get_output_path(exp_plane, get_t, get_s)
        output_path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        max_t = 50
        output_path_preview = self.get_output_path(exp_plane, lambda: max_t, get_s)
        output_path_preview.set_stroke(opacity=0.5)
        for path in [output_path, output_path_preview]:
            path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        self.add(exp_plane, exp_plane_label, output_path_preview, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        axes.x_axis.scale(0.5, 0, about_edge=LEFT)
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)
        graph.set_clip_plane(UP, -axes.get_y(DOWN))
        v_line.set_clip_plane(UP, -axes.get_y(DOWN))
        axes_background = BackgroundRectangle(axes)
        axes_background.set_fill(BLACK, 1)
        axes_background.align_to(s_plane.get_right(), LEFT).shift(1e-2 * RIGHT)
        axes_background.stretch(1.2, 1, about_edge=DOWN)

        self.add(axes_background, axes, graph, v_line)

        # Pre-preamble
        s_tracker.set_value(-0.3)
        self.play(s_tracker.animate.increment_value(1.5j), run_time=4)
        self.play(t_tracker.animate.set_value(TAU / 1.5), run_time=5, rate_func=linear)
        self.play(s_tracker.animate.set_value(1.5j), run_time=3)
        t_tracker.set_value(0)

        # Play around
        s_tracker.set_value(0)
        self.play(s_tracker.animate.increment_value(1.5j), run_time=4)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        self.wait(4.5)
        self.play(s_tracker.animate.increment_value(-0.2))
        self.wait(2)
        for step in [-0.8, 1.2, -0.4]:
            self.play(s_tracker.animate.increment_value(step), run_time=3)
        self.wait()


class DefineSPlane(SPlane):
    def construct(self):
        # Trackers
        s_tracker = self.s_tracker = ComplexValueTracker(-1)
        t_tracker = self.t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        # Add s plane
        s_plane = self.get_s_plane()
        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s)
        self.add(s_plane, s_dot, s_label)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane_label = self.get_exp_plane_label(exp_plane)
        # output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t)
        output_dot, output_label = self.get_output_dot_and_label(exp_plane, get_s, get_t, s_tex="a")
        output_path = self.get_output_path(exp_plane, get_t, get_s)
        output_path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        max_t = 50
        output_path_preview = self.get_output_path(exp_plane, lambda: max_t, get_s)
        output_path_preview.set_stroke(opacity=0.5)
        for path in [output_path, output_path_preview]:
            path.set_clip_plane(RIGHT, -s_plane.get_x(RIGHT))

        self.add(exp_plane, exp_plane_label, output_path_preview, output_path, output_dot, output_label)

        # Add e^{st} graph
        axes = self.get_graph_axes()
        axes.x_axis.scale(0.5, 0, about_edge=LEFT)
        graph = self.get_dynamic_exp_graph(axes, get_s)
        v_line = self.get_graph_v_line(axes, get_t, get_s)
        graph.set_clip_plane(UP, -axes.get_y(DOWN))
        v_line.set_clip_plane(UP, -axes.get_y(DOWN))
        axes_background = BackgroundRectangle(axes)
        axes_background.set_fill(BLACK, 1)
        axes_background.align_to(s_plane.get_right(), LEFT).shift(1e-2 * RIGHT)
        axes_background.stretch(1.2, 1, about_edge=DOWN)

        self.add(axes_background, axes, graph, v_line)

        # Test
        s_tracker.set_value(-0.1 + 1.5j)
        t_tracker.clear_updaters()
        t_tracker.set_value(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        self.wait(20)

        # Go!
        t_tracker.clear_updaters()
        t_tracker.set_value(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        self.add(t_tracker)
        s_tracker.set_value(0)
        self.play(s_tracker.animate.set_value(-0.1 + 1.5j), run_time=3)
        self.wait(3)

        # Name the plane
        frame = self.frame
        s_plane_name = Text("S-plane", font_size=90)
        s_plane_name.next_to(s_plane, UP)
        s_plane_name.set_color(YELLOW)

        self.play(
            frame.animate.reorient(0, 0, 0, (0.49, 0.34, 0.0), 9.65).set_anim_args(time_span=(0, 2)),
            FlashAround(s_plane, time_width=1.5, buff=MED_SMALL_BUFF, stroke_width=5),
            run_time=4,
        )
        self.wait()
        self.play(s_tracker.animate.set_value(-0.2 - 1j), run_time=3)
        self.wait()
        self.play(Write(s_plane_name))
        self.wait(3)

        # Show exp pieces
        s_samples = [complex(a, b) for a in range(-2, 3) for b in range(-2, 3)]
        exp_pieces = VGroup(
            get_exp_graph_icon(s).move_to(s_plane.n2p(0.85 * s))
            for s in s_samples
        )

        dots = VGroup(Dot(radius=0.1).move_to(piece) for piece in exp_pieces)
        dots.set_color(YELLOW)

        self.play(
            LaggedStartMap(FadeIn, dots, scale=5, lag_ratio=0.05, run_time=2),
            VFadeOut(output_label),
            FadeOut(output_dot),
            VFadeOut(s_label),
            FadeOut(s_dot),
        )
        t_tracker.clear_updaters()
        self.wait()
        self.play(LaggedStart(
            (FadeTransform(dot, piece)
            for dot, piece in zip(dots, exp_pieces)),
            lag_ratio=0.05,
            run_time=3,
            group_type=Group,
        ))
        self.wait()

        # Associate with complex
        rect = SurroundingRectangle(exp_pieces[6], buff=SMALL_BUFF)
        rect.set_stroke(TEAL, 3)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.surround(VGroup(exp_plane, exp_plane_label)))
        self.play(FadeOut(rect))

        # Go through imaginary partsg
        rows = VGroup(exp_pieces[n::5] for n in range(0, 5))
        for row in rows:
            row.save_state()
        self.play(rows.animate.fade(0.7))
        self.play(Restore(rows[2]))
        self.play(
            rows[2].animate.fade(0.7),
            Restore(rows[1]),
            Restore(rows[3]),
        )
        self.play(
            Restore(rows[0]),
            Restore(rows[4]),
            rows[1].animate.fade(0.7),
            rows[3].animate.fade(0.7),
        )
        self.wait()
        self.play(*(Restore(row) for row in rows))

        # Show columns
        cols = VGroup(exp_pieces[n:n + 5] for n in range(0, 25, 5))
        for col in cols:
            col.save_state()
        last = VectorizedPoint()
        self.play(cols.animate.fade(0.7))
        for col in cols:
            self.play(
                last.animate.fade(0.7),
                Restore(col),
            )
            last = col

        self.play(*(Restore(col) for col in cols))
        self.wait()


class BreakingDownCosine(ShowFamilyOfComplexSolutions):
    tex_to_color_map = {"+i": YELLOW, "-i": YELLOW}

    def construct(self):
        # Set up various planes
        left_planes, left_plane_labels = self.get_left_planes(label_texs=[R"e^{+it}", R"e^{-it}"])
        rot_vects, tails, t_tracker = self.get_rot_vects(left_planes)

        right_plane = self.get_right_plane(x_range=(-2, 2))
        right_plane.next_to(left_planes, RIGHT, buff=2.0)
        right_plane.add_coordinate_labels(font_size=16)
        vect_sum = self.get_rot_vect_sum(right_plane, t_tracker)
        for vect in vect_sum:
            vect.coef_tracker.set_value(0.5)

        output_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        output_dot.f_always.move_to(vect_sum[1].get_end)

        self.add(t_tracker)
        self.add(left_planes, left_plane_labels)
        self.add(rot_vects, tails, t_tracker)

        self.add(right_plane)
        self.add(vect_sum)
        self.add(output_dot)
        self.wait(12)

        # Show each part
        for vect in vect_sum:
            vect.coef_tracker.set_value(1)

        inner_arrows = VGroup(Arrow(2 * v, v, buff=0) for v in compass_directions(8))
        inner_arrows.set_fill(YELLOW)
        inner_arrows.move_to(right_plane)

        self.remove(vect_sum, output_dot)
        for i in [0, 1]:
            # self.play(ReplacementTransform(rot_vects[i].copy().clear_updaters(), vect_sum[i]))
            self.play(TransformFromCopy(rot_vects[i], vect_sum[i], suspend_mobject_updating=True))
            self.wait(3)

        self.wait(15)
        self.play(
            *(vs.coef_tracker.animate.set_value(0.5) for vs in vect_sum),
            LaggedStartMap(GrowArrow, inner_arrows, lag_ratio=1e-2, run_time=1)
        )
        self.play(FadeOut(inner_arrows))
        self.wait(12)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports ShowFamilyOfComplexSolutions from the _2025.laplace.shm module within the 3b1b videos codebase.",
      10: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      19: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      27: "IntroduceEulersFormula extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      28: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      30: "ComplexPlane extends NumberPlane for complex number visualization. Points map to complex numbers directly.",
      40: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      41: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      51: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      52: "DecimalNumber displays a formatted decimal that can be animated. Tracks a value and auto-updates display.",
      54: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      55: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      57: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      61: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      62: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      63: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      64: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      68: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      69: "FadeOut transitions a mobject from opaque to transparent.",
      70: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      75: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      77: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      84: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      85: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      87: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      88: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      89: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      93: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      95: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      96: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      108: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      138: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      139: "FadeOut transitions a mobject from opaque to transparent.",
      141: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      142: "FadeOut transitions a mobject from opaque to transparent.",
      143: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      155: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      158: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      160: "FadeOut transitions a mobject from opaque to transparent.",
      161: "FadeOut transitions a mobject from opaque to transparent.",
      163: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      165: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      166: "FadeOut transitions a mobject from opaque to transparent.",
      167: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      179: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      181: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      185: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      188: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      189: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      190: "FadeOut transitions a mobject from opaque to transparent.",
      234: "ExpGraph extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      235: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      313: "DefiningPropertyOfExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      314: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      365: "ExampleExponentials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      366: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      375: "ImaginaryInputsToTheTaylorSeries extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      376: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      571: "ComplexExpGraph extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      576: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      638: "Class AltComplexExpGraph inherits from ComplexExpGraph.",
      644: "SPlane extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      654: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1092: "FamilyOfRealExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1093: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1118: "ForcedOscillatorSolutionForm extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1119: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1337: "Class BreakingDownFunctions inherits from ForcedOscillatorSolutionForm.",
      1338: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1581: "Thumbnail extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1582: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1652: "Thumbnail2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1653: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1722: "Class RecapSPlane inherits from SPlane.",
      1723: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1784: "Class DefineSPlane inherits from SPlane.",
      1785: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1935: "Class BreakingDownCosine inherits from ShowFamilyOfComplexSolutions.",
      1938: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/laplace/integration.py"] = {
    description: "Integration and the Laplace transform: visualizes how integration in time maps to division by s, with applications to solving ODEs.",
    code: `from manim_imports_ext import *
from _2025.laplace.exponentials import *


def z_to_color(z, sat=0.5, lum=0.5):
    angle = math.atan2(z.imag, z.real)
    return Color(hsl=(angle / TAU, sat, lum))


def get_complex_graph(
    s_plane,
    func,
    min_real=None,
    pole_buff=1e-3,
    color_by_phase=True,
    opacity=0.7,
    shading=(0.1, 0.1, 0.1),
    resolution=(301, 301),
    saturation=0.5,
    luminance=0.5,
    face_sort_direction=UP,
    mesh_resolution=(61, 61),
    mesh_stroke_style=dict(
        color=WHITE,
        width=1,
        opacity=0.15
    )
):
    u_range = list(s_plane.x_range[:2])
    v_range = list(s_plane.y_range[:2])

    if min_real is not None:
        u_range[0] = min_real + pole_buff

    unit_size = s_plane.x_axis.get_unit_size()
    graph = ParametricSurface(
        lambda u, v: [
            *s_plane.c2p(u, v)[:2],
            unit_size * abs(func(complex(u, v)))
        ],
        u_range=u_range,
        v_range=v_range,
        resolution=resolution
    )
    graph.set_shading(*shading)

    if color_by_phase:
        graph.color_by_uv_function(
            lambda u, v: z_to_color(func(complex(u, v)), sat=saturation, lum=luminance)
        )

    graph.set_opacity(opacity)
    graph.sort_faces_back_to_front(face_sort_direction)

    # Add mesh
    mesh = SurfaceMesh(graph, resolution=mesh_resolution)
    mesh.set_stroke(**mesh_stroke_style)

    return Group(graph, mesh)


class IntegrateConstant(InteractiveScene):
    def construct(self):
        # Axes and graph
        axes = Axes((0, 100), (0, 2, 0.25), width=100, height=3)
        axes.add_coordinate_labels(num_decimal_places=2, font_size=20)
        axes.to_corner(DL)

        graph = axes.get_graph(lambda t: 1)
        graph.set_stroke(BLUE, 3)
        graph_label = Tex(R"f(t) = 1")
        graph_label.next_to(graph, UP)
        graph_label.to_edge(RIGHT)

        self.add(axes)
        self.add(graph, graph_label)

        # Integral
        t_tracker = ValueTracker(0.01)
        get_t = t_tracker.get_value

        v_tracker = ValueTracker(1)
        t_tracker.add_updater(lambda m, dt: m.increment_value(v_tracker.get_value() * dt))

        integral = Tex(R"\\int^{0.01}_0 1 \\, dt = 0.01", font_size=72)
        integral.to_edge(UP, buff=3.0)
        integral["1"].set_color(BLUE)
        decimals = integral.make_number_changeable("0.01", replace_all=True)
        decimals[0].scale(0.5, about_edge=LEFT)
        integral[2].scale(0.5, about_edge=LEFT)
        integral[:3].shift(0.5 * RIGHT)
        integral.set_scale_stroke_with_zoom(True)
        for dec in decimals:
            dec.add_updater(lambda m: m.set_value(get_t()))

        rect = Rectangle()
        rect.set_fill(BLUE, 0.5)
        rect.set_stroke(WHITE, 2)
        rect.set_z_index(-1)
        x_unit = axes.x_axis.get_unit_size()
        y_unit = axes.y_axis.get_unit_size()
        origin = axes.get_origin()
        rect.add_updater(
            lambda m: m.set_shape(get_t() * x_unit, y_unit).move_to(origin, DL)
        )

        integral.fix_in_frame()

        self.add(rect, integral, t_tracker)

        # Grow
        v_tracker.set_value(1)
        self.play(
            v_tracker.animate.set_value(12),
            self.frame.animate.reorient(0, 0, 0, (42.64, 14.41, 0.0), 58.35).set_anim_args(time_span=(6, 15)),
            run_time=20
        )


class IntegrateRealExponential(InteractiveScene):
    def construct(self):
        # Add integral expression
        t2c = {R"{s}": YELLOW}
        integral = Tex(R"\\int^\\infty_0 e^{\\minus {s}t} dt", t2c=t2c)
        integral.set_x(1)
        integral.to_edge(UP)
        integral.save_state()
        integral.scale(1.5, about_edge=UP)
        self.add(integral)

        # Add the graph of e^{-st}
        max_x = 15
        unit_size = 4
        axes = Axes((0, max_x, 0.25), (0, 1, 0.25), unit_size=unit_size)
        axes.to_edge(DL, buff=1.0)
        axes.add_coordinate_labels(num_decimal_places=2, font_size=20)

        def exp_func(t):
            return np.exp(-get_s() * t)

        s_tracker = ValueTracker(1)
        get_s = s_tracker.get_value
        graph = axes.get_graph(np.exp)
        graph.set_stroke(BLUE, 3)
        axes.bind_graph_to_func(graph, exp_func)

        graph_label = Tex(R"e^{\\minus {s}t}", t2c=t2c, font_size=72)
        graph_label.next_to(axes.y_axis.get_top(), UR).shift(0.5 * RIGHT)

        self.play(
            FadeIn(axes),
            TransformFromCopy(integral[R"e^{\\minus {s}t}"], graph_label),
            ShowCreation(graph, suspend_mobject_updating=True, run_time=3),
            Restore(integral),
        )

        # Add a slider for s
        s_slider = Slider(s_tracker, x_range=(0, 5), var_name="s")
        s_slider.scale(1.5)
        s_slider.to_edge(UP, buff=MED_LARGE_BUFF)
        s_slider.align_to(axes.c2p(0, 0), LEFT)

        self.play(VFadeIn(s_slider))
        for value in [5, 0.25, 1]:
            self.play(
                s_tracker.animate.set_value(value),
                run_time=4
            )
            self.wait()

        # Show integral as area
        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.next_to(integral, DOWN)
        area_word = Text("Area", font_size=60)
        area_word.next_to(equals, DOWN)

        area = axes.get_area_under_graph(graph)

        def update_area(area):
            area.become(axes.get_area_under_graph(graph))

        arrow = Arrow(area_word.get_corner(DL), axes.c2p(0.75, 0.5), thickness=4)

        self.play(
            LaggedStart(
                Animation(graph.copy(), remover=True),
                Write(equals),
                FadeIn(area_word, DOWN),
                GrowArrow(arrow),
                UpdateFromFunc(area, update_area),
                lag_ratio=0.25
            ),
            ShowCreation(graph, suspend_mobject_updating=True, run_time=3),
        )
        self.wait()

        # Try altenrate s value
        frame = self.frame

        area.clear_updaters()
        area.add_updater(update_area)
        self.play(
            s_tracker.animate.set_value(-0.2),
            frame.animate.reorient(0, 0, 0, (16.79, 7.82, 0.0), 28.01).set_anim_args(time_span=(2, 5)),
            run_time=5
        )

        # Show area = 1 from s = 1
        simple_integral = Tex(R"\\int^\\infty_0 e^{\\minus t} dt")
        simple_integral.move_to(integral)
        simple_exp = Tex(R"e^{\\minus t}")
        simple_exp.move_to(graph_label)

        anti_deriv = Tex(R"=\\big[\\minus e^{\\minus t} \\big]^\\infty_0")
        simple_rhs = Tex(R"=0 - (\\minus 1)")
        anti_deriv.next_to(simple_integral, RIGHT)
        simple_rhs.next_to(anti_deriv, RIGHT)

        equals_one = Tex(R"= 1", font_size=60)
        equals_one.next_to(area_word)

        area_one_label = Tex(R"1", font_size=60)
        area_one_label.move_to(axes.c2p(0.35, 0.35))
        area_one_label.set_z_index(1)

        self.remove(integral, graph_label)
        self.play(
            TransformMatchingTex(integral.copy(), simple_integral),
            TransformMatchingTex(graph_label.copy(), simple_exp),
            run_time=1
        )
        self.play(
            TransformMatchingTex(simple_integral.copy(), anti_deriv, run_time=1.5, path_arc=30 * DEG),
        )
        rect_kw = dict(buff=0.05, stroke_width=1.5)
        self.play(
            FadeIn(simple_rhs[:2], time_span=(0, 1)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\\minus e^{\\minus t}"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\\infty"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(simple_rhs[:2], **rect_kw)),
            run_time=1.5
        )
        self.play(
            FadeIn(simple_rhs[2:], time_span=(0, 1)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"\\minus e^{\\minus t}"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(anti_deriv[R"0"], **rect_kw)),
            VFadeInThenOut(SurroundingRectangle(simple_rhs[2:], **rect_kw)),
            run_time=1.5
        )
        self.wait()
        self.play(TransformMatchingTex(simple_rhs.copy(), equals_one, run_time=1))
        self.play(TransformFromCopy(equals_one["1"], area_one_label))
        self.wait()

        # Comapre to unit square
        square = Polygon(
            axes.c2p(0, 1),
            axes.c2p(1, 1),
            axes.c2p(1, 0),
            axes.c2p(0, 0),
        )
        square.set_stroke(WHITE, 3)
        square.set_fill(BLUE, 0.0)

        area_s_tracker = ValueTracker(get_s())
        area_x_max_tracker = ValueTracker(graph.x_range[1])
        squishy_area = always_redraw(  # Currently unused
            lambda: axes.get_area_under_graph(axes.get_graph(
                lambda t: np.exp(-area_s_tracker.get_value() * t),
                x_range=(0, area_x_max_tracker.get_value())
            ))
        )

        tail_area = axes.get_area_under_graph(graph, x_range=(1, 6))
        tail_area.set_fill(RED_E, 0.75)
        corner_area = axes.get_graph(lambda t: np.exp(-get_s() * t), (0, 1))
        corner_area.add_line_to(axes.c2p(1, 1))
        corner_area.add_line_to(axes.c2p(0, 1))
        corner_area.match_style(tail_area)
        corner_area.set_z_index(-2)

        area_one_label.save_state()

        self.play(FadeIn(tail_area))
        self.wait()
        self.play(
            ShowCreation(square),
            FadeIn(corner_area),
            area_one_label.animate.move_to(square),
        )
        self.wait()
        self.play(
            FadeOut(square),
            FadeOut(corner_area),
            FadeOut(tail_area),
            Restore(area_one_label),
            FadeOut(anti_deriv),
            FadeOut(simple_rhs)
        )
        self.wait()

        # Show squishing the area
        stretch_label = VGroup(
            TexText(R"Squish by $\\frac{1}{s}$", t2c=t2c),
            Vector(2 * LEFT, thickness=5, fill_color=YELLOW)
        )
        stretch_label.arrange(DOWN, buff=MED_SMALL_BUFF)
        stretch_label.move_to(axes.c2p(0.5, 0.5))

        area_word.set_z_index(1)
        area_word.target = area_word.generate_target()
        area_word.target.move_to(axes.c2p(0.6, 0.33))

        rhs = Tex(R"= \\frac{1}{s}", t2c=t2c, font_size=60)
        rhs.next_to(area_word.target, RIGHT)

        area.set_z_index(-1)
        area.add_updater(update_area)

        self.play(LaggedStart(
            TransformMatchingTex(simple_integral, integral, run_time=1),
            FadeIn(graph_label, 0.5 * DOWN),
            FadeOut(simple_exp, 0.5 * DOWN),
            FadeOut(equals_one, 0.5 * DOWN),
            FadeOut(area_one_label),
            lag_ratio=0.1
        ))
        self.play(
            s_tracker.animate.set_value(5).set_anim_args(run_time=8),
            FadeIn(stretch_label, 1.5 * LEFT, time_span=(1, 5)),
            FadeOut(arrow),
        )
        self.wait()
        self.play(
            LaggedStart(
                FadeOut(stretch_label),
                MoveToTarget(area_word),
                FadeTransform(stretch_label[0][-3:].copy(), rhs[1:]),
                FadeTransform(stretch_label[1].copy(), rhs[0]),
                FadeOut(equals),
                run_time=2,
                lag_ratio=0.1
            )
        )

        # Show area value
        dec_rhs = Tex(R"= 1.00", font_size=60)
        dec_rhs.make_number_changeable("1.00").add_updater(lambda m: m.set_value(1 / get_s()))
        dec_rhs.always.next_to(rhs, RIGHT)

        self.play(
            VFadeIn(dec_rhs),
            s_tracker.animate.set_value(0.01).set_anim_args(run_time=12, rate_func=bezier([0, 1, 1, 1])),
            self.frame.animate.reorient(0, 0, 0, (15.36, 0, 0.0), 30).set_anim_args(time_span=(6, 11)),
        )
        self.wait()
        self.play(
            VGroup(area_word, rhs).animate.next_to(axes.c2p(0, 0), UR, MED_LARGE_BUFF).set_anim_args(time_span=(1, 3)),
            s_tracker.animate.set_value(0.75),
            VFadeOut(dec_rhs, time_span=(2, 4)),
            self.frame.animate.to_default_state(),
            run_time=4,
        )
        self.wait()

        # Averages over intervals
        v_lines = VGroup(
            DashedLine(axes.c2p(0, 0), axes.c2p(0, 1.25)),
            DashedLine(axes.c2p(1, 0), axes.c2p(1, 1.25)),
        )
        v_lines.set_stroke(WHITE, 1)

        unit_int = Tex(R"\\int^1_0 e^{\\minus {s}t} dt", t2c=t2c, font_size=60)
        unit_int.move_to(v_lines, UP)

        graph_for_unit_area = axes.get_graph(exp_func)
        graph_for_unit_area.set_stroke(width=0)
        unit_int_area = always_redraw(
            lambda: axes.get_area_under_graph(graph_for_unit_area, (0, 1))
        )
        avg_value = exp_func(np.linspace(0, 1, 1000)).mean()

        def slosh_rate_func(t, cycles=3):
            return min(smooth(2 * cycles * t), 1) + 0.7 * math.sin(cycles * TAU * t) * (t - 1)**2

        self.remove(integral)
        area.clear_updaters()
        self.play(
            *map(FadeOut, [area_word, rhs, graph_label, s_slider]),
            *map(ShowCreation, v_lines),
            TransformMatchingTex(integral.copy(), unit_int),
            FadeOut(area),
            FadeIn(unit_int_area, time_span=(0.5, 1), suspend_mobject_updating=True),
        )
        self.wait()
        self.play(
            Transform(
                graph_for_unit_area,
                axes.get_graph(lambda t: avg_value).set_stroke(width=0),
                run_time=3,
                rate_func=slosh_rate_func,
            ),
        )
        unit_int_area.suspend_updating()

        # Show average
        avg_label1 = self.get_avg_label(unit_int_area, 0, 1)
        avg_label1.save_state()
        avg_label1.space_out_submobjects(0.5)
        avg_label1.set_opacity(0)
        self.play(Restore(avg_label1))
        self.wait()

        # Emphasize that it's a unit interval
        frame = self.frame
        unit_line = Line(axes.c2p(0, 0), axes.c2p(1, 0))
        unit_line.set_stroke(YELLOW, 5)
        brace = Brace(unit_line, DOWN, buff=MED_LARGE_BUFF)
        unit_label = brace.get_tex("1", buff=MED_SMALL_BUFF, font_size=72)
        unit_group = VGroup(brace, unit_line, unit_label)

        self.play(LaggedStart(
            GrowFromCenter(brace),
            ShowCreation(unit_line),
            FadeIn(unit_label, 0.25 * DOWN),
            frame.animate.set_y(-1.5),
            run_time=1.5
        ))
        self.wait()

        # Area = height
        height_line = Line(unit_int_area.get_corner(DL), unit_int_area.get_corner(UL))
        height_line.set_stroke(RED, 5)

        area_eq_height = TexText(R"= Area = Width $\\times$ Height", t2c={"Area": BLUE, "Height": RED, "Width": YELLOW})
        area_eq_height.next_to(unit_int, RIGHT)
        fade_rect = BackgroundRectangle(area_eq_height, buff=0.25, fill_opacity=1)
        fade_rect.stretch(2, 1, about_edge=DOWN)

        sample_dots = DotCloud([axes.c2p(a, exp_func(a)) for a in np.linspace(0, 1, 30)])
        sample_dots.set_color(WHITE)
        sample_dots.set_glow_factor(2)
        sample_dots.set_radius(0.15)

        self.play(
            FadeIn(fade_rect),
            FadeIn(area_eq_height),
            ShowCreation(height_line),
        )
        self.wait()
        self.play(
            area_eq_height[R"Width $\\times$"].animate.set_opacity(0),
            area_eq_height[R"Height"].animate.move_to(area_eq_height["Width"], UL),
        )
        self.wait()
        self.play(FlashAround(unit_int, run_time=2, time_width=1.5))
        self.wait()

        self.play(ShowCreation(sample_dots))
        self.play(sample_dots.animate.stretch(0, 1).match_y(axes.c2p(0, avg_value)), rate_func=lambda t: slosh_rate_func(t, 2), run_time=3)
        self.play(FadeOut(sample_dots), FadeOut(height_line), FadeOut(area_eq_height))

        # Average over next interval
        v_lines2 = v_lines.copy()
        v_lines2.move_to(axes.c2p(1, 0), DL)
        unit_int2 = Tex(R"\\int^2_1 e^{\\minus {s}t} dt", t2c=t2c, font_size=60)
        unit_int2.move_to(v_lines2, UP)

        area2_graph = graph.copy().clear_updaters().set_stroke(width=0)
        pile2 = always_redraw(lambda: axes.get_area_under_graph(area2_graph, (1, 2)))
        avg_value2 = get_norm(pile2.get_area_vector()) / (axes.x_axis.get_unit_size()**2)
        avg_label2 = self.get_avg_label(
            pile2.copy().set_height(axes.y_axis.get_unit_size() * avg_value2, stretch=True, about_edge=DOWN),
            1, 2
        )

        self.add(v_lines[0])
        self.play(
            TransformFromCopy(v_lines, v_lines2, path_arc=-20 * DEG),
            TransformMatchingTex(
                unit_int.copy(),
                unit_int2,
                path_arc=-20 * DEG,
                run_time=1,
                key_map={"0": "1", "1": 2},
            ),
            FadeIn(pile2, suspend_mobject_updating=True),
            unit_group.animate.match_x(pile2),
        )
        self.play(
            area2_graph.animate.stretch(0, 1).match_y(axes.y_axis.n2p(avg_value2)).set_anim_args(
                rate_func=slosh_rate_func,
                run_time=3,
            ),
            FadeIn(avg_label2)
        )
        pile2.clear_updaters()
        self.wait()

        # Show all integrals
        new_groups = VGroup()
        piles = VGroup(unit_int_area, pile2)
        avg_labels = VGroup(avg_label1, avg_label2)

        for n in range(2, 6):
            new_v_lines = v_lines.copy()
            new_v_lines.move_to(axes.c2p(n, 0), DL)
            new_unit_int = Tex(Rf"\\int^{n + 1}_{n} " + R"e^{\\minus {s}t} dt", t2c=t2c, font_size=60)
            if n == 5:
                new_unit_int = Tex(R"\\cdots", font_size=90)
            new_unit_int.match_y(unit_int)
            new_unit_int.match_x(new_v_lines)
            s = get_s()
            avg_y = np.mean([np.exp(-s * t) for t in np.arange(n, n + 1, 1e-3)])
            new_pile = pile2.copy().clear_updaters()
            new_pile.set_height(avg_y * axes.y_axis.get_unit_size(), stretch=True)
            new_pile.move_to(axes.c2p(n, 0), DL)
            new_avg_label = self.get_avg_label(new_pile, n, n + 1)

            new_group = VGroup(new_v_lines, new_unit_int, new_pile, new_avg_label)
            new_groups.add(new_group)
            piles.add(new_pile)
            avg_labels.add(new_avg_label)

        big_brace = Brace(VGroup(unit_int, new_groups[-1][1]), UP, font_size=90, buff=LARGE_BUFF)
        integral.set_height(3)
        integral.next_to(big_brace, UP, buff=LARGE_BUFF)

        self.play(
            self.frame.animate.reorient(0, 0, 0, (5.34, 2.63, 0.0), 13.89),
            LaggedStartMap(FadeIn, new_groups, lag_ratio=0.75),
            FadeOut(unit_group, time_span=(0, 2)),
            run_time=4
        )
        self.play(
            GrowFromCenter(big_brace),
            FadeIn(integral),
        )
        self.wait()

        # Show adding all the heights
        height_vects = VGroup(
            Arrow(pile.get_bottom(), pile.get_top(), buff=0, thickness=6, fill_color=RED)
            for pile in piles
        )

        equals = Tex(R"=", font_size=120)
        equals.move_to(integral).shift(RIGHT)

        stacked_vects = height_vects.copy()
        stacked_vects.arrange(UP, buff=SMALL_BUFF)
        stacked_vects.scale(0.9)
        stacked_vects.next_to(equals, RIGHT, LARGE_BUFF)

        self.play(
            LaggedStartMap(FadeOut, avg_labels, lag_ratio=0.2),
            LaggedStartMap(FadeIn, height_vects, lag_ratio=0.2),
            run_time=1,
        )
        self.wait()
        self.play(
            TransformFromCopy(height_vects, stacked_vects, lag_ratio=0.05, run_time=2),
            integral.animate.next_to(equals, LEFT, LARGE_BUFF),
            Write(equals),
        )
        self.wait()

    def get_avg_label(self, pile, start=0, end=1, font_size=30):
        word = Text(f"Average over [{start}, {end}]", font_size=font_size)
        word.move_to(pile)
        word.set_max_height(pile.get_height() / 4)
        buff = min(SMALL_BUFF, 0.05 * pile.get_height())
        arrows = VGroup(
            Arrow(word.get_top() + buff * UP, pile.get_top(), buff=0),
            Arrow(word.get_bottom() + buff * DOWN, pile.get_bottom(), buff=0),
        )
        return VGroup(word, arrows)


class IntegrateComplexExponential(SPlane):
    staggered_path_colors = [MAROON_B, MAROON_C]
    t_max = 100  # For dynamic path and vector sum.  Change to 100 for final render
    initial_s = 0.2 + 1j
    s_label_config = dict(
        hide_zero_components_on_complex=False,
        include_sign=True,
        num_decimal_places=1,
    )

    def setup(self):
        super().setup()
        # Trackers
        self.s_tracker = ComplexValueTracker(self.initial_s)
        self.t_tracker = ValueTracker(0)
        get_s = self.s_tracker.get_value
        get_t = self.t_tracker.get_value

        def exp_func(t):
            return np.exp(-get_s() * t)

        self.exp_func = exp_func

    def construct(self):
        # Trackers
        s_tracker = self.s_tracker
        t_tracker = self.t_tracker
        exp_func = self.exp_func
        t2c = self.tex_to_color_map

        get_s = self.s_tracker.get_value
        get_t = self.t_tracker.get_value

        # Add s plane
        s_plane, s_dot, s_label, s_plane_label = s_group = self.get_s_group()
        self.add(*s_group)

        # Add exp plane
        exp_plane = self.get_exp_plane()
        exp_plane.set_width(6)
        exp_plane.next_to(s_plane, RIGHT, buff=1.0)
        exp_plane.add_coordinate_labels(font_size=16)
        exp_plane.axes.set_stroke(width=1)
        exp_plane_label = Tex(R"e^{\\minus st}", font_size=72, t2c=t2c)
        exp_plane_label.next_to(exp_plane, UP)

        output_dot, output_label = self.get_output_dot_and_label(exp_plane, lambda: -get_s(), get_t, label_direction=DR, s_tex=R"\\minus s")
        output_label.set_z_index(1)
        output_label_added_shift = Point(ORIGIN)
        output_label.add_updater(lambda m: m.shift(output_label_added_shift.get_center()))

        self.add(exp_plane, exp_plane_label, output_dot, output_label)

        # Show inital path
        path = always_redraw(lambda: self.get_output_path(exp_plane, exp_func, 0, 20))

        self.play(
            ShowCreation(path, suspend_mobject_updating=True),
            t_tracker.animate.set_value(path.t_range[1]),
            rate_func=linear,
            run_time=10,
        )
        self.play(
            s_tracker.animate.set_value(0.2 - 1j),
            rate_func=there_and_back,
            run_time=6,
        )
        self.play(
            s_tracker.animate.set_value(-0.2 + 1j),
            rate_func=there_and_back,
            run_time=6,
        )

        path.suspend_updating()

        # Fadeable transition to the start
        path_ghost = path.copy().set_stroke(opacity=0.4)
        self.wait()
        t_tracker.set_value(0)
        self.remove(path)
        self.add(path_ghost)
        self.wait()

        # Show [0, 1]
        subpath_0 = self.get_output_path(exp_plane, exp_func, 0, 1, stroke_color=self.staggered_path_colors[0])
        avg_vect0 = self.get_mean_vector(exp_plane, exp_func, 0, 1)
        many_points = self.get_sample_dots(subpath_0)
        avg_dot0 = TrueDot(avg_vect0.get_end(), color=RED)

        int_tex0 = Tex(R"\\int^1_0 e^{\\minus st} dt", t2c=t2c, font_size=48)
        int_tex0.next_to(exp_plane.n2p(1), UP, MED_SMALL_BUFF)
        int_tex0.set_backstroke(BLACK, 5)

        self.play(FadeIn(int_tex0, 0.25 * UP))
        self.play(
            t_tracker.animate.set_value(1),
            ShowCreation(subpath_0),
            rate_func=linear,
            run_time=2
        )
        self.wait()
        self.play(ShowCreation(many_points))
        self.wait()
        self.play(ReplacementTransform(many_points, avg_dot0))
        self.wait()
        self.play(
            GrowArrow(avg_vect0),
            int_tex0.animate.set_height(0.75).next_to(avg_vect0.get_center(), UR, buff=SMALL_BUFF)
        )
        self.play(FadeOut(avg_dot0))
        self.wait()

        # Bring in integral plane
        lil_exp_plane = self.get_exp_plane(x_range=(-1, 1))
        lil_exp_plane.set_width(3)
        lil_exp_plane.move_to(exp_plane, UR).shift(0.5 * UP + 0.75 * LEFT)
        lil_exp_plane.save_state()
        lil_exp_plane.axes.set_stroke(width=1)

        int_plane = self.get_exp_plane()
        int_plane.set_width(3)
        int_plane.next_to(lil_exp_plane, DOWN, MED_LARGE_BUFF)
        int_plane.axes.set_stroke(width=1)
        int_plane.add_coordinate_labels(font_size=12)

        to_upper = lil_exp_plane.get_center() - exp_plane.get_center()
        to_lower = int_plane.get_center() - exp_plane.get_center()

        int_plane_label = Tex(R"\\int^\\infty_0 e^{\\minus st} dt", t2c=t2c, font_size=48)
        int_plane_label.next_to(int_plane, LEFT, aligned_edge=UP)

        lower_avg_vect0 = self.get_mean_vector(int_plane, exp_func, 0, 1, thickness=1, fill_color=self.staggered_path_colors[0])

        self.play(
            exp_plane.animate.move_to(lil_exp_plane.saved_state).set_opacity(0),
            FadeIn(lil_exp_plane, to_upper),
            VGroup(path_ghost, subpath_0, avg_vect0).animate.shift(to_upper),
            int_tex0.animate.scale(0.5, about_point=exp_plane.n2p(0)).shift(to_lower),
            TransformFromCopy(avg_vect0, lower_avg_vect0),
            FadeIn(int_plane),
            exp_plane_label.animate.next_to(lil_exp_plane, LEFT, aligned_edge=UP),
        )
        self.wait()

        # Add the next few arrows
        avg_vects = VGroup(avg_vect0)
        lower_avg_vects = VGroup(lower_avg_vect0)
        subpaths = VGroup(subpath_0)
        int_texs = VGroup(int_tex0)
        shifts = [
            0.8 * LEFT + 0.1 * DOWN,
            LEFT,
            0.2 * LEFT + 0.4 * UP,
            0.4 * UP,
        ]

        def get_subpath(n):
            result = self.get_output_path(exp_plane, exp_func, n, n + 1, stroke_width=3)
            result.set_stroke(self.staggered_path_colors[n % 2])
            return result

        for n in range(1, 5):
            subpath_n = get_subpath(n)
            sample_points = self.get_sample_dots(subpath_n)
            avg_vect, lower_avg_vect = [
                self.get_mean_vector(plane, exp_func, n, n + 1, thickness=thickness)
                for plane, thickness in [(exp_plane, 3), (int_plane, 1)]
            ]
            lower_avg_vect.set_fill(self.staggered_path_colors[n % 2], border_width=0.5)
            lower_avg_vect.set_stroke(width=0)
            lower_avg_vect.put_start_on(lower_avg_vects[-1].get_end())
            avg_dot = sample_points.copy().set_points([avg_vect.get_end()])

            int_tex = Tex(Rf"\\int_{n}^{n + 1} e^{{\\minus s t}} dt", t2c=t2c)
            int_tex.match_height(int_tex0)
            int_tex.scale(1.0 - 0.15 * n)
            int_tex.next_to(lower_avg_vect.get_center(), rotate_vector(lower_avg_vect.get_vector(), 90 * DEG), buff=SMALL_BUFF)

            self.play(
                t_tracker.animate.set_value(n + 1),
                ShowCreation(subpath_n),
                output_label_added_shift.animate.move_to(shifts[n - 1]),
                ShowCreation(sample_points),
                run_time=2,
                rate_func=linear,
            )
            self.play(
                avg_vects[-1].animate.set_fill(opacity=0.5).set_stroke(width=0),
                GrowArrow(avg_vect),
                Transform(sample_points, avg_dot),
            )
            self.play(
                FadeOut(sample_points),
                TransformFromCopy(avg_vect, lower_avg_vect),
                FadeIn(int_tex),
            )

            avg_vects.add(avg_vect)
            lower_avg_vects.add(lower_avg_vect)
            subpaths.add(subpath_n)
            int_texs.add(int_tex)

        # Show numerous more
        for n in range(5, 20):
            subpath = get_subpath(n)
            avg_vect = self.get_mean_vector(exp_plane, exp_func, n, n + 1, thickness=3)
            lower_avg_vect = self.get_mean_vector(int_plane, exp_func, n, n + 1, thickness=1)
            lower_avg_vect.set_fill(self.staggered_path_colors[n % 2], border_width=0.5)
            lower_avg_vect.set_stroke(width=0)
            lower_avg_vect.put_start_on(lower_avg_vects[-1].get_end())

            anims = [
                ShowCreation(subpath, rate_func=linear),
                t_tracker.animate.set_value(n + 1).set_anim_args(rate_func=linear),
                FadeIn(avg_vect),
            ]
            if n == 5:
                anims.append(avg_vects.animate.set_opacity(0.25).set_stroke(width=0))
            else:
                anims.append(avg_vects[-1].animate.set_opacity(0.25).set_stroke(width=0))
                anims.append(TransformFromCopy(avg_vects[-1], lower_avg_vects[-1]))
            self.play(*anims)

            avg_vects.add(avg_vect)
            lower_avg_vects.add(lower_avg_vect)
            subpaths.add(subpath)

            self.add(avg_vects, subpaths, *lower_avg_vects[:-1])

        # Show the full integral
        int_dot = Group(TrueDot(radius=0.025), GlowDot())
        int_dot.set_color(MAROON_B)
        int_dot.move_to(int_plane.n2p(1 / get_s()))
        int_rect = SurroundingRectangle(int_plane_label)
        int_rect.set_stroke(YELLOW, 2)

        self.play(
            LaggedStart(
                (FadeTransform(int_tex, int_plane_label)
                for int_tex in int_texs),
                lag_ratio=0.025,
                group_type=Group,
            ),
            *map(FadeOut, [output_label, output_dot, avg_vects]),
        )
        self.play(ShowCreation(int_rect))
        self.wait()
        self.play(
            int_rect.animate.surround(int_dot[0], buff=0),
            FadeIn(int_dot),
        )
        self.play(FadeOut(int_rect))
        self.wait()

        # Make the diagram dynamic
        staggered_path = self.get_dynamic_output_path(exp_plane, exp_func)
        int_pieces = self.get_dynamic_vector_sum(int_plane, exp_func)
        int_dot.add_updater(lambda m: m.move_to(int_plane.n2p(1.0 / get_s())))

        dynamic_pieces = Group(staggered_path, int_pieces)

        self.play(FlashAround(Group(s_label, s_dot), time_width=1.5, run_time=2))
        self.remove(path_ghost, subpaths, lower_avg_vects)
        self.add(staggered_path, int_pieces, int_dot)

        # Move around the path
        s_rect = int_rect.copy()

        self.play(s_tracker.animate.set_value(0.2 - 1j), run_time=6)
        self.wait()
        self.play(s_tracker.animate.set_value(1), run_time=4)
        s_rect.surround(Group(s_label))
        self.play(ShowCreation(s_rect))
        self.play(s_rect.animate.surround(int_dot[0], buff=0))
        self.play(FadeOut(s_rect))
        self.wait()

        # Approach zero, then go vertical
        frame = self.frame
        self.play(
            s_tracker.animate.set_value(0.1).set_anim_args(rate_func=bezier([0, 1, 1, 1])),
            frame.animate.reorient(0, 0, 0, (2.75, 0, 0.0), 11).set_anim_args(time_span=(2, 8)),
            run_time=8,
        )
        self.wait()
        self.play(
            s_tracker.animate.set_value(0.1 + 2j),
            frame.animate.to_default_state(),
            run_time=10
        )
        self.play(s_tracker.animate.set_value(0.1 - 2j), run_time=5)
        self.wait()
        self.play(s_tracker.animate.set_value(0.2 + 1j), run_time=5)
        self.wait()

        # Emphasize output
        int_vect = self.get_vector(int_plane, 1 / get_s(), thickness=2, fill_color=WHITE)
        int_vect.set_stroke(width=0).set_fill(border_width=0.5)
        rect = SurroundingRectangle(int_plane_label)
        rect.set_stroke(YELLOW, 2)
        int_vect_outline = int_vect.copy().set_fill(opacity=0).set_stroke(WHITE, 1)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            rect.animate.surround(int_dot[0], buff=0),
            int_pieces.animate.set_fill(opacity=0.5),
        )
        self.wait()
        self.play(
            GrowArrow(int_vect),
            FadeOut(rect),
            FadeOut(int_dot),
        )
        self.wait()

        # Plot it
        frame = self.frame
        s_plane_group = Group(s_plane, s_dot, s_label, s_plane_label)
        s_label.set_flat_stroke(True)
        s_plane.set_flat_stroke(False)
        s_plane_group.save_state()
        s_plane_group.center()
        self.remove(s_plane_group)

        for mob in self.get_mobjects():
            mob.fix_in_frame()
            mob.set_z_index(2)

        right_rect = FullScreenRectangle()
        right_rect.set_fill(BLACK, 1).set_stroke(width=0)
        right_rect.set_width(lil_exp_plane.get_width() + 2, about_edge=RIGHT, stretch=True)
        v_line = DashedLine(right_rect.get_corner(DL), right_rect.get_corner(UL))
        v_line.set_stroke(WHITE, 1)
        right_pannel = VGroup(right_rect, v_line)
        right_pannel.fix_in_frame()
        right_pannel.set_z_index(1)
        self.add(right_pannel)

        def get_output_magnitude(s):
            return abs(1.0 / s)

        z_line = Line(ORIGIN, OUT)
        z_line.set_stroke(WHITE, 2)

        unit_size = s_plane.x_axis.get_unit_size()
        z_line.add_updater(lambda m: m.put_start_and_end_on(
            s_plane.n2p(get_s()),
            s_plane.n2p(get_s()) + unit_size * get_output_magnitude(get_s()) * OUT,
        ))

        out_dot = Group(TrueDot(), GlowDot())
        out_dot.set_color(TEAL)
        out_dot.f_always.move_to(z_line.get_end)

        traced_graph = TracedPath(z_line.get_end, stroke_color=TEAL)
        traced_graph.update()

        int_vect.add_updater(
            lambda m: m.put_start_and_end_on(
                int_plane.n2p(0),
                int_plane.n2p(1.0 / get_s()),
            )
        )

        pre_z_line = Line(int_vect.get_start(), int_vect.get_end())
        pre_z_line.set_stroke(WHITE, 5)
        pre_z_line.set_z_index(2)

        new_exp_plane_label = exp_plane_label.copy().scale(0.5).next_to(lil_exp_plane, RIGHT, aligned_edge=UP)
        new_int_plane_label = int_plane_label.copy().scale(0.5).next_to(int_plane, RIGHT, SMALL_BUFF, aligned_edge=UP)

        s_plane_group.restore()
        self.play(
            frame.animate.reorient(-3, 68, 0, (1.43, 0.51, -0.28), 6.50),
            s_plane_group.animate.center(),
            ReplacementTransform(pre_z_line, z_line),
            FadeOut(exp_plane_label, time_span=(1, 2)),
            FadeIn(new_exp_plane_label, time_span=(1, 2)),
            FadeOut(int_plane_label, time_span=(1, 2)),
            FadeIn(new_int_plane_label, time_span=(1, 2)),
            VFadeIn(v_line, time_span=(2, 3)),
            run_time=3,
        )
        self.play(
            FadeIn(out_dot, time_span=(0, 1)),
            frame.animate.reorient(8, 64, 0, (1.43, 0.51, -0.28), 6.50),
            run_time=3
        )

        self.add(traced_graph)
        self.play(
            s_tracker.animate.set_value(0.2 - 2j),
            frame.animate.reorient(25, 87, 0, (3.18, 1.73, 2.63), 11.08).set_anim_args(time_span=(1, 7.5)),
            run_time=10
        )
        self.play(s_tracker.animate.set_value(0.4 - 2j))
        self.play(
            s_tracker.animate.set_value(0.4 + 2j),
            frame.animate.reorient(44, 81, 0, (3.16, 1.71, 2.63), 11.08),
            run_time=5,
        )
        self.play(s_tracker.animate.set_value(0.6 + 2j))
        self.play(
            s_tracker.animate.set_value(0.6 - 2j),
            frame.animate.reorient(66, 89, 0, (3.31, 2.07, 2.96), 9.69),
            run_time=5,
        )
        self.play(s_tracker.animate.set_value(0.8 - 2j))
        self.play(
            s_tracker.animate.set_value(0.8 + 2j),
            frame.animate.reorient(22, 84, 0, (2.56, 1.53, 1.72), 10.14),
            run_time=5
        )
        self.play(s_tracker.animate.set_value(1.0 + 2j))
        self.play(
            s_tracker.animate.set_value(1.0 - 2j),
            frame.animate.reorient(22, 84, 0, (2.56, 1.53, 1.72), 10.14),
            run_time=5
        )
        self.wait()

        traced_graph.suspend_updating()

        # Add the graph
        graph = get_complex_graph(s_plane, lambda s: 1.0 / s, min_real=0)[0]
        graph.set_shading(0.1, 0.1, 0)
        graph.save_state()
        graph.set_color(GREY_B)
        mesh = SurfaceMesh(graph, resolution=(11, 21))
        mesh.set_stroke(WHITE, 0.5, 0.5)
        for line in mesh:
            if line.get_z(IN) < 0:
                mesh.remove(line)

        self.play(
            ShowCreation(graph),
            ShowCreation(mesh, lag_ratio=1e-2, time_span=(1, 4)),
            VFadeOut(traced_graph),
            frame.animate.reorient(34, 74, 0, (2.02, 1.44, 1.88), 9.87),
            out_dot.animate.set_color(WHITE),
            run_time=4
        )

        self.play(
            s_tracker.animate.set_value(0.1 - 0.1j),
            frame.animate.reorient(38, 82, 0, (2.84, 2.62, 4.72), 14.21).set_anim_args(time_span=(0, 6)),
            run_time=10
        )
        self.wait()
        self.play(s_tracker.animate.set_value(0.5 - 0.1j), run_time=5)
        self.play(frame.animate.reorient(26, 72, 0, (2.33, 1.59, 1.91), 10.76), run_time=3)
        self.wait()
        self.play(
            s_tracker.animate.set_value(0.5 - 1j),
            run_time=3
        )
        self.wait()

        # Explain color
        hue_circle = Circle(radius=1.5 * int_plane.x_axis.get_unit_size())
        hue_circle.set_color_by_proportion(lambda h: Color(hsl=(h, 0.5, 0.5)))
        hue_circle.set_stroke(width=5)
        hue_circle.move_to(int_plane)
        hue_circle.fix_in_frame().set_z_index(2)

        brace = LineBrace(int_vect, direction=UP, buff=0)
        brace.set_fill(border_width=1)
        brace.set_z_index(2).fix_in_frame()

        self.play(GrowFromCenter(brace))
        self.wait()
        self.play(FadeOut(brace))

        self.play(ShowCreation(hue_circle, run_time=2))
        self.wait()
        self.play(
            Restore(graph, time_span=(0, 1)),
            frame.animate.reorient(29, 72, 0, (1.79, 1.9, 1.65), 11.21),
            run_time=4,
        )
        self.play(s_tracker.animate.set_value(0.5 + 1j), run_time=5)
        self.wait()

        # Show negative real part
        left_plane = Rectangle()
        left_plane.set_stroke(width=0).set_fill(RED, 0.5)
        left_plane.replace(s_plane, stretch=True)
        left_plane.stretch(0.5, 0, about_edge=LEFT)
        left_plane.save_state()
        left_plane.stretch(0, 0, about_edge=RIGHT)

        staggered_path.set_clip_plane(RIGHT, -v_line.get_x())
        int_pieces.set_clip_plane(RIGHT, -v_line.get_x())

        self.play(
            frame.animate.reorient(-38, 70, 5, (3.17, 1.96, -0.36), 9.16),
            FadeOut(hue_circle),
            Restore(left_plane, time_span=(3, 5)),
            run_time=5
        )
        self.wait()

        # Move to negative real
        self.play(
            s_tracker.animate.set_value(-0.5),
            VFadeOut(z_line, time_span=(0, 1.5)),
            FadeOut(out_dot, time_span=(0, 1.5)),
            VFadeOut(int_vect, time_span=(0, 1.0)),
            run_time=3
        )
        self.play(s_tracker.animate.set_value(-0.1 - 1j).set_anim_args(run_time=3))
        self.play(
            *(
                plane.animate.scale(0.1, about_point=plane.n2p(0))
                for plane in [exp_plane, lil_exp_plane, int_plane]
            ),
        )
        self.play(
            ShowCreation(staggered_path, suspend_mobject_updating=True),
            ShowIncreasingSubsets(int_pieces, suspend_mobject_updating=True),
            run_time=4
        )
        self.play(
            UpdateFromAlphaFunc(
                s_tracker,
                lambda m, a: m.set_value(complex(
                    -0.1 - math.sin(PI * a),
                    interpolate(-1, 1, a)
                ))
            ),
            run_time=8
        )
        self.play(
            *(
                plane.animate.scale(10, about_point=plane.n2p(0)).set_anim_args(time_span=(2, 4))
                for plane in [exp_plane, lil_exp_plane, int_plane]
            ),
            s_tracker.animate.set_value(1j),
            run_time=4
        )
        self.play(FadeIn(z_line), FadeIn(out_dot))
        self.wait()

        # Show imaginary input
        self.remove(staggered_path, int_pieces)
        int_pieces.set_fill(opacity=1)
        t_tracker.set_value(0)
        output_dot.fix_in_frame().set_z_index(2)
        output_label.add_updater(lambda m: m.fix_in_frame().set_z_index(2))
        self.play(
            FadeIn(output_dot),
            VFadeIn(output_label),
            left_plane.animate.set_fill(opacity=0.2),
        )
        max_n = 25
        self.play(
            ShowCreation(staggered_path[:max_n].set_z_index(2)),
            ShowIncreasingSubsets(int_pieces[:max_n].set_z_index(2), int_func=np.floor),
            t_tracker.animate.set_value(max_n).set_anim_args(time_span=(max_n / (max_n + 1), max_n)),
            rate_func=linear,
            run_time=max_n
        )
        self.play(
            VFadeOut(output_label),
            FadeOut(output_dot),
            t_tracker.animate.set_value(max_n + 1).set_anim_args(rate_func=linear),
            ShowIncreasingSubsets(int_pieces[max_n:].set_z_index(2)),
            FadeIn(staggered_path[max_n:].set_z_index(2))
        )

        # Show the limiting value
        self.add(staggered_path, int_pieces)
        self.add(int_vect)
        self.play(
            s_tracker.animate.set_value(0.1 + 1j),
            int_pieces.animate.set_fill(opacity=0.75),
            VFadeIn(int_vect),
            FadeOut(left_plane),
            run_time=2
        )
        rect = SurroundingRectangle(int_vect).set_z_index(2)
        self.play(ShowCreation(rect))
        self.play(FadeOut(rect))
        self.wait()
        self.play(
            s_tracker.animate.set_value(1j),
            run_time=5
        )
        self.wait()

        # Move along imaginary line
        boundary = TracedPath(z_line.get_end, stroke_color=RED, stroke_width=3)

        self.add(boundary, out_dot)
        self.play(s_tracker.animate.set_value(2j), run_time=3)
        self.play(s_tracker.animate.set_value(0.1j), run_time=3)
        self.play(s_tracker.animate.set_value(1j), run_time=3)
        boundary.clear_updaters()
        self.play(FadeOut(boundary))

        # Show this as 1 / s
        int_equation = Tex(R"\\int^\\infty_0 e^{\\minus st} dt = \\frac{1}{s}", t2c=t2c)
        int_equation.to_edge(UP).shift(LEFT)
        int_equation.fix_in_frame()
        lhs_mover = new_int_plane_label.copy()

        self.play(
            frame.animate.reorient(0, 85, 1, (5.32, 3.46, 3.69), 13.49),
            s_tracker.animate.set_value(0.2 + 1j),
            run_time=3,
        )
        self.play(
            lhs_mover.animate.replace(int_equation[:-4]),
            Write(int_equation[-4:], time_span=(0.75, 1.75)),
        )
        self.remove(lhs_mover)
        self.add(int_equation)
        self.wait()

        # Look closely at s = i again
        int_plane_rect = SurroundingRectangle(int_plane)
        int_plane_rect.fix_in_frame()
        int_plane_rect.set_stroke(YELLOW, 5).insert_n_curves(100)
        int_plane_rect.set_z_index(3)
        int_plane_rect.scale(0.5, about_edge=DOWN).scale(1.1)

        i_eq = Tex(R"\\frac{1}{i} = -i")
        i_eq.next_to(int_equation[-3:], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        i_eq.fix_in_frame()

        left_arrow = Vector(LEFT).set_color(YELLOW)
        left_arrow.next_to(s_plane.n2p(0.4 + 1j), DOWN)

        self.play(s_tracker.animate.set_value(0.2 - 1j), run_time=6)
        self.play(
            s_tracker.animate.set_value(0.5 + 1j),
            frame.animate.reorient(-49, 79, 0, (6.24, 2.45, -0.13), 10.06),
            run_time=4
        )
        self.play(
            GrowArrow(left_arrow, time_span=(0, 1)),
            VShowPassingFlash(int_plane_rect, time_width=1.5, time_span=(4, 7)),
            Write(i_eq, time_span=(10, 12)),
            s_tracker.animate.set_value(1j),
            run_time=15
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 85, 1, (5.32, 3.46, 3.69), 13.49),
            FadeOut(i_eq),
            FadeOut(left_arrow),
            run_time=3
        )

        # Ambient s movement
        for z in [0.2 - 1j, 1 - 1j, 1 + 1j, 0.2 + 1j]:
            self.play(s_tracker.animate.set_value(z), run_time=6)

        # Talk again about the left plane
        full_plane = Rectangle().set_stroke(width=0).set_fill(RED, 0.5)
        full_plane.replace(s_plane, stretch=True)
        full_plane.set_fill(GREEN, 0.5)

        left_plane = full_plane.copy()
        left_plane.set_fill(RED, 0.5)
        left_plane.stretch(0.5, 0, about_edge=LEFT)
        left_plane.save_state()
        left_plane.stretch(0, 0, about_edge=RIGHT)

        equation_rect = SurroundingRectangle(int_equation)
        equation_rect.set_stroke(YELLOW, 2)
        equation_rect.fix_in_frame()

        self.play(
            Restore(left_plane),
            frame.animate.reorient(-36, 79, 4, (7.71, 2.4, 1.86), 15.36),
            run_time=3
        )
        self.wait()
        self.play(ShowCreation(equation_rect))
        self.wait()
        self.play(equation_rect.animate.surround(int_equation[R"\\frac{1}{s}"], buff=SMALL_BUFF))
        self.wait()
        self.play(
            ReplacementTransform(left_plane, full_plane)
        )
        self.wait()

        # Describe continuation
        lhs_rect = SurroundingRectangle(int_equation[:-4])
        rhs_rect = SurroundingRectangle(int_equation[-3:])
        VGroup(lhs_rect, rhs_rect).fix_in_frame().set_stroke(YELLOW, 2)

        def pole_func(s):
            if s != 0:
                return 1.0 / s
            return 100

        extended_graph = get_complex_graph(s_plane, pole_func)
        extended_mesh = SurfaceMesh(extended_graph, resolution=(21, 21))
        extended_mesh.remove(*(line for line in extended_mesh if line.get_z(IN) < 0))
        extended_mesh.set_stroke(WHITE, 0.5, 0.15)
        extended_mesh.shift(1e-2 * OUT)

        self.add(extended_graph, graph, extended_mesh, mesh)
        self.play(
            FadeOut(graph, 1e-2 * IN),
            FadeIn(extended_graph),
            FadeOut(mesh),
            FadeIn(extended_mesh),
            frame.animate.reorient(0, 85, 0, (5.32, 3.46, 3.69), 13.49).set_anim_args(run_time=5),
            FadeOut(full_plane),
        )
        self.play(
            frame.animate.reorient(-17, 62, 9, (6.29, -0.4, 3.75), 13.87),
            FadeOut(equation_rect),
            run_time=5
        )
        curr_s = complex(get_s())
        self.play(
            UpdateFromAlphaFunc(s_tracker, lambda m, a: m.set_value(curr_s * np.exp(a * TAU * 1j))),
            run_time=30,
        )
        self.wait()

        # Discuss pole (Mostly with overlays)
        self.play(
            frame.animate.reorient(-36, 83, 1, (5.84, -1.06, 3.19), 10.63),
            s_tracker.animate.set_value(0.1 + 0.1j),
            run_time=20
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (1.95, 0.95, -0.0), 15.34),
            s_tracker.animate.set_value(1e-3).set_anim_args(time_span=(0, 5)),
            extended_mesh.animate.set_stroke(opacity=0),
            VFadeOut(int_vect, time_span=(0, 7)),
            run_time=10,
        )
        self.play(frame.animate.reorient(0, 83, 0, (5.18, 4.04, 3.15), 14.50), run_time=10)

    def get_s_group(self):
        s_plane = self.get_s_plane()
        s_plane.set_width(6)
        s_plane.to_corner(DL)
        s_plane.axes.set_stroke(width=1)

        s_dot, s_label = self.get_s_dot_and_label(s_plane, get_s=self.s_tracker.get_value)
        s_plane_label = Tex(R"s", t2c=self.tex_to_color_map, font_size=72)
        s_plane_label.next_to(s_plane, UP)

        s_group = Group(s_plane, s_dot, s_label, s_plane_label)
        return s_group

    def get_output_path(self, plane, func, t_min=0, t_max=20, stroke_color=TEAL, stroke_width=2, step_size=1e-1):
        return ParametricCurve(
            lambda t: plane.n2p(func(t)),
            t_range=[t_min, t_max, step_size],
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )

    def get_vector(self, plane, value, backstroke_width=2, **kwargs):
        vect = Arrow(plane.n2p(0), plane.n2p(value), buff=0, **kwargs)
        vect.set_backstroke(BLACK, width=backstroke_width)
        return vect

    def get_mean_vector(self, plane, func, t_min, t_max, thickness=3, fill_color=RED, n_samples=1000, **kwargs):
        x_range = np.linspace(t_min, t_max, n_samples)
        x_mean = np.mean(func(x_range))
        return self.get_vector(plane, x_mean, thickness=thickness, fill_color=fill_color, **kwargs)
        return vect

    def get_sample_dots(self, subpath, n_samples=20, radius=0.05, glow_factor=0.5, color=RED):
        return DotCloud(
            [subpath.pfp(a) for a in np.linspace(0, 1, n_samples)],
            radius=radius,
            glow_factor=glow_factor,
            color=color
        )

    def get_dynamic_output_path(self, plane, func, stroke_width=3, step_size=1e-2):
        t_range = list(range(0, self.t_max))
        t_samples = [np.arange(t, t + 1 + step_size, step_size) for t in t_range]
        path = VGroup(VMobject() for t in t_range)

        def update_path(path):
            for piece, samples in zip(path, t_samples):
                piece.set_points_as_corners(plane.n2p(func(samples)))
            return path

        path.add_updater(update_path)
        for piece, color in zip(path, it.cycle(self.staggered_path_colors)):
            piece.set_stroke(color, stroke_width)

        return path

    def get_dynamic_vector_sum(self, plane, func, thickness=1, backstroke_width=0, n_samples=100, border_width=0.5):
        t_range = list(range(0, self.t_max))
        vects = VGroup(
            self.get_vector(
                plane, 1,
                thickness=thickness,
                backstroke_width=backstroke_width,
                fill_color=color,
            )
            for t, color in zip(t_range, it.cycle(self.staggered_path_colors))
        )
        vects.set_fill(border_width=border_width)
        t_samples = [np.linspace(t, t + 1, n_samples) for t in t_range]

        def update_vects(vects):
            avg_values = [0] + [func(samples).mean() for samples in t_samples]
            end_values = np.cumsum(avg_values)
            end_points = plane.n2p(end_values)
            for vect, p0, p1 in zip(vects, end_points, end_points[1:]):
                vect.put_start_and_end_on(p0, p1)
            return vect

        vects.add_updater(update_vects)

        return vects


class BreakDownLaplaceTransform(IntegrateComplexExponential):
    func_tex = R"e^{1.5 {t}}"
    initial_s = 2 + 1j
    s_plane_x_range = (-3, 3)
    s_label_font_size = 24
    t_max = 100  # For dynamic path and vector sum.  Change to 100 for final render
    # pole_value = 1.5
    pole_value = -0.2 + 1.5j

    def construct(self):
        self.add_core_pieces()

        frame = self.frame
        s_tracker = self.s_tracker
        exp_plane, int_plane = self.output_planes

        # Talk through the parts
        frame.reorient(0, 52, 0, (-0.69, 0.41, 0.56), 10.93)
        int_rect = SurroundingRectangle(int_plane.label)
        int_rect.set_stroke(YELLOW, 2)
        int_rect.fix_in_frame()

        self.play(
            frame.animate.reorient(20, 68, 0, (-0.45, 0.24, 0.17), 6.28),
            s_tracker.animate.set_value(1.6 - 1j),
            ShowCreation(int_rect, time_span=(4, 5)),
            run_time=8
        )
        self.play(int_rect.animate.surround(exp_plane.label))
        self.draw_upper_plot(draw_time=8)
        self.play(
            s_tracker.animate.set_value(1.6),
            frame.animate.reorient(24, 65, 0, (-0.35, 1.85, 1.61), 11.20),
            run_time=5
        )
        self.play(
            s_tracker.animate.set_value(1.501),
            run_time=5
        )

        # Alternate pan to pole
        self.clear()
        to_add = Group(self.s_group, self.graph_tracer, self.graph)
        to_add.shift(-self.s_group[0].n2p(0))
        self.add(to_add)

        frame.reorient(-35, 84, 0, (-1.14, 1.42, 4.73), 14.55)
        self.s_tracker.set_value(1 + 1j)
        self.play(
            frame.animate.reorient(0, 43, 0, (-0.42, 0.6, 2.1), 9.91),
            self.s_tracker.animate.set_value(1.5001),
            run_time=10
        )

    def add_core_pieces(self):
        # Planes and their contenst
        s_group = self.get_s_group()
        s_plane = s_group[0]
        s_plane.faded_lines.set_opacity(0)

        output_planes = self.get_output_planes()
        exp_plane, int_plane = output_planes

        right_rect = FullScreenRectangle()
        right_rect.set_fill(BLACK, 1)
        right_rect.set_width(RIGHT_SIDE[0] - output_planes.get_left()[0] + SMALL_BUFF, stretch=True, about_edge=RIGHT)

        # Dynamic pieces
        output_path = self.get_dynamic_output_path(exp_plane, self.inner_func)
        vect_sum = self.get_dynamic_vector_sum(int_plane, self.inner_func)
        integral_vect = self.get_integral_vect(int_plane)

        for mob in [output_path, vect_sum, integral_vect]:
            mob.set_clip_plane(RIGHT, -exp_plane.get_x(LEFT))

        graph_tracer = self.get_graph_tracer(s_plane)

        graph = get_complex_graph(s_plane, self.transformed_func)

        # Add everything
        right_group = VGroup(right_rect, output_planes, output_path, vect_sum)
        right_group.fix_in_frame()
        right_group.set_scale_stroke_with_zoom(True)

        self.add(
            s_group,
            graph_tracer,
            graph,
            right_rect,
            output_planes,
            output_path,
            vect_sum,
            integral_vect,
        )

        self.s_group = s_group
        self.output_planes = output_planes
        self.output_path = output_path
        self.vect_sum = vect_sum
        self.integral_vect = integral_vect
        self.graph_tracer = graph_tracer
        self.graph = graph

    def get_integral_vect(self, int_plane):
        vect = Vector(RIGHT, fill_color=WHITE, thickness=2)
        dot = GlowDot(color=PINK)
        origin = int_plane.n2p(0)

        vect.add_updater(lambda m: m.put_start_and_end_on(
            origin, int_plane.n2p(self.transformed_func(self.s_tracker.get_value()))
        ))
        dot.f_always.move_to(vect.get_end)

        group = Group(vect, dot)
        group.fix_in_frame()
        return group

    def get_graph_tracer(self, s_plane):
        v_line = Line(ORIGIN, OUT).set_stroke(WHITE, 3)
        graph_dot = Group(TrueDot(), GlowDot())
        graph_dot.set_color(TEAL)

        group = Group(v_line, graph_dot)
        s_plane_unit = s_plane.x_axis.get_unit_size()

        def update_tracer(group):
            s = self.s_tracker.get_value()
            s_point = s_plane.n2p(s)
            int_value = self.transformed_func(s)

            line, dot = group
            line.put_start_and_end_on(s_point, s_point + s_plane_unit * abs(int_value) * OUT)
            dot.move_to(line.get_end())

        group.add_updater(update_tracer)
        return group

    def get_output_planes(self, width=3, edge_buff=2.5):
        exp_plane = self.get_exp_plane(x_range=(-1, 1))
        exp_plane.set_width(width)
        exp_plane.axes.set_stroke(width=1)
        exp_plane.to_corner(UP, MED_LARGE_BUFF)
        exp_plane.to_edge(RIGHT, buff=edge_buff)

        int_plane = self.get_exp_plane()
        int_plane.set_width(width)
        int_plane.move_to(exp_plane).to_edge(DOWN)
        int_plane.axes.set_stroke(width=1)
        int_plane.add_coordinate_labels(font_size=12)

        kw = dict(t2c=self.tex_to_color_map)
        exp_plane.label = Tex(self.func_tex, R"e^{\\minus {s} {t}}", **kw)
        exp_plane.label.next_to(exp_plane, RIGHT, aligned_edge=UP)

        int_plane.label = Tex(R"\\int^\\infty_0 " + self.func_tex + R"e^{\\minus {s} {t}} d{t}", **kw)
        int_plane.label.scale(0.6)
        int_plane.label.next_to(int_plane, RIGHT, aligned_edge=UP)

        planes = VGroup(exp_plane, int_plane)
        for plane in planes:
            plane.add(plane.label)

        return planes

    def func(self, t):
        return np.exp(1.5 * t)

    def inner_func(self, t):
        return self.func(t) * np.exp(-self.s_tracker.get_value() * t)

    def transformed_func(self, s):
        return np.divide(1.0, (s - self.pole_value))

    def draw_upper_plot(self, draw_time=12, rate_multiple=2):
        exp_plane = self.output_planes[0]
        n_pieces = int(rate_multiple * draw_time)
        path_copy = VMobject()
        path_copy.fix_in_frame()
        path_copy.start_new_path(self.output_path[0].get_start())
        for part in self.output_path[:n_pieces]:
            path_copy.append_vectorized_mobject(part)

        tracing_vect = Vector(fill_color=YELLOW)
        tracing_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            path_copy.get_end(),
        ))
        tracing_vect.fix_in_frame()

        vect_sum_copy = self.vect_sum.copy()[:n_pieces]
        vect_sum_copy.clear_updaters()
        self.remove(self.output_path, self.vect_sum, self.integral_vect)

        self.add(tracing_vect)
        self.play(
            ShowCreation(path_copy),
            ShowIncreasingSubsets(self.vect_sum[:n_pieces], int_func=np.ceil),
            run_time=draw_time,
            rate_func=linear
        )
        self.remove(path_copy, tracing_vect)
        self.add(self.output_path, self.vect_sum, self.integral_vect)
        self.wait()

    def show_simple_pole(self):
        # Clean the board
        frame = self.frame
        for line in self.graph[1]:
            if line.get_z(OUT) > 100 or line.get_z(IN) < -100:
                line.set_stroke(opacity=0)
        self.clear()
        self.add(self.s_group, self.graph, self.graph_tracer)
        self.graph_tracer[0].set_stroke(width=1)

        # Move
        frame.reorient(-30, 83, 0, (-4.35, 1.2, 2.52), 11.47)
        self.play(
            frame.animate.reorient(0, 0, 0, (-5.17, -0.19, 0.0), 9.88).set_field_of_view(20 * DEG),
            self.s_tracker.animate.set_value(self.pole_value + 1e-5),
            run_time=10
        )
        self.wait()




class LaplaceTransformOfCos(BreakDownLaplaceTransform):
    func_tex = R"\\cos(t)"
    s_label_config = dict(
        hide_zero_components_on_complex=True,
        include_sign=True,
        num_decimal_places=2,
    )
    t_max = 200  # For dynamic path and vector sum.

    def construct(self):
        self.add_core_pieces()

        frame = self.frame
        s_tracker = self.s_tracker

        # Pan around, and highlight the two poles
        frame.reorient(0, 0, 0, (-0.44, -0.47, 0.0), 8.00)
        self.play(
            frame.animate.reorient(28, 69, 0, (-0.42, 0.29, 1.36), 7.45),
            s_tracker.animate.set_value(0.02 + 1j),
            run_time=10
        )
        self.play(
            frame.animate.reorient(-38, 81, 0, (-0.24, 0.38, 1.05), 8.77),
            s_tracker.animate.set_value(0.02 - 1j),
            run_time=10
        )
        self.play(
            frame.animate.reorient(13, 72, 0, (-1.41, 0.26, 1.11), 7.37),
            s_tracker.animate.set_value(0.1 + 0.57j),
            run_time=8
        )

        # Further panning
        self.play(
            frame.animate.reorient(0, 31, 0, (-1.08, -0.52, 0.3), 7.37),
            s_tracker.animate.set_value(0.01 + 1j),
            run_time=4
        )
        self.play(
            s_tracker.animate.set_value(0.2 - 1j),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(21, 71, 0, (-1.48, -0.28, 0.8), 5.07),
            s_tracker.animate.set_value(0.5 + 1j),
            run_time=20
        )

        # Highlight the integral (Start here)
        exp_plane, int_plane = self.output_planes
        int_rect = SurroundingRectangle(int_plane.label, buff=SMALL_BUFF)
        int_rect.set_stroke(YELLOW, 2)
        int_rect.fix_in_frame()

        self.play(ShowCreation(int_rect))
        self.wait()
        self.play(int_rect.animate.surround(exp_plane.label, buff=SMALL_BUFF))
        self.wait()

        # Draw the upper plot
        self.draw_upper_plot()

        # Move s to 0.2i
        self.play(
            s_tracker.animate.set_value(0.2j),
            self.graph[0].animate.set_opacity(0.5),
            self.graph[1].animate.set_stroke(opacity=0.05),
            frame.animate.reorient(0, 15, 0, (-1.05, -1.03, 0.59), 4.66),
            run_time=4,
        )

        # Trace the path
        t_tracker = self.t_tracker
        t_tracker.set_value(0)
        get_t = t_tracker.get_value
        get_s = s_tracker.get_value

        output_path = self.output_path
        vect_sum = self.vect_sum
        int_vect = self.integral_vect
        exp_vect = Arrow(exp_plane.n2p(0), exp_plane.n2p(1), buff=0)
        exp_vect.set_fill(YELLOW)
        exp_vect.fix_in_frame()
        exp_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(np.exp(-get_s() * get_t())),
        ))
        full_vect = exp_vect.copy().clear_updaters()
        full_vect.add_updater(lambda m: m.put_start_and_end_on(
            exp_plane.n2p(0),
            exp_plane.n2p(self.func(get_t()) * np.exp(-get_s() * get_t())),
        ))

        t_label = Tex(R"{t} = 0.0", t2c=self.tex_to_color_map)
        t_label.next_to(exp_plane.label, DOWN, LARGE_BUFF, aligned_edge=LEFT)
        t_label.make_number_changeable("0.0").f_always.set_value(t_tracker.get_value)
        t_label.fix_in_frame()

        output_path_copy = VMobject()
        output_path_copy.start_new_path(self.output_path[0].get_start())
        output_path_copy.fix_in_frame()
        for part in output_path:
            output_path_copy.append_vectorized_mobject(part)
        partial_path = output_path_copy.copy()
        partial_path.fix_in_frame()
        partial_path.add_updater(lambda m: m.pointwise_become_partial(
            output_path_copy, 0, get_t() / self.t_max
        ))

        vect_sum_copy = vect_sum.copy()
        vect_sum_copy.clear_updaters()
        growing_vect_sum = VGroup(*vect_sum_copy)
        growing_vect_sum.fix_in_frame()

        def update_growing_sum(group):
            group.set_submobjects(vect_sum_copy.submobjects[:int(get_t())])
            if len(group) == 0:
                return
            group[:-1].set_fill(opacity=0.35, border_width=0.5)
            group[-1].set_fill(opacity=1, border_width=2)

        growing_vect_sum.add_updater(update_growing_sum)

        t_tracker.clear_updaters()
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        self.play(
            int_rect.animate.surround(exp_plane.label[-4:], SMALL_BUFF),
            FadeOut(output_path, suspend_mobject_updating=True),
            FadeOut(vect_sum, suspend_mobject_updating=True),
            FadeOut(int_vect, suspend_mobject_updating=True),
            FadeIn(exp_vect),
            FadeIn(t_label),
        )
        self.add(t_tracker)
        self.wait(12)
        self.add(partial_path)
        self.play(
            int_rect.animate.surround(exp_plane.label, SMALL_BUFF),
            exp_vect.animate.set_fill(opacity=0.25),
            VFadeIn(full_vect),
        )
        self.wait(12)
        self.add(growing_vect_sum)
        self.play(int_rect.animate.surround(int_plane.label, buff=SMALL_BUFF))
        self.wait(20)
        self.play(
            frame.animate.reorient(25, 55, 0, (-0.91, -0.76, 0.83), 5.93),
            self.graph[0].animate.set_opacity(0.7),
            self.graph[1].animate.set_stroke(opacity=0.1),
            run_time=5
        )
        self.wait(5)
        growing_vect_sum.clear_updaters()
        self.play(
            VFadeOut(growing_vect_sum),
            VFadeOut(partial_path),
            VFadeOut(t_label),
            VFadeOut(exp_vect),
            VFadeOut(full_vect),
            VFadeOut(int_rect),
            VFadeIn(output_path),
            VFadeIn(vect_sum),
            FadeIn(int_vect)
        )

        # Add a small real part, increase imaginary
        self.play(
            s_tracker.animate.increment_value(0.05),
            run_time=3
        )
        self.draw_upper_plot(draw_time=12, rate_multiple=6)
        self.play(
            s_tracker.animate.increment_value(0.8j),
            run_time=20,
        )
        self.draw_upper_plot(draw_time=12, rate_multiple=4)
        self.play(
            s_tracker.animate.increment_value(-0.03),
            frame.animate.reorient(29, 78, 0, (-0.48, 0.16, 2.64), 10.00),
            run_time=7,
        )

        # Walk the imaginary line
        self.play(s_tracker.animate.increment_value(0.05), run_time=3)
        self.wait()
        self.play(s_tracker.animate.increment_value(1.3j), run_time=20)
        self.play(s_tracker.animate.increment_value(-0.5j), run_time=10)
        self.play(s_tracker.animate.increment_value(1), run_time=6)
        self.play(s_tracker.animate.increment_value(-0.8), run_time=6)
        self.play(s_tracker.animate.increment_value(-0.2j), run_time=6)
        self.play(s_tracker.animate.set_value(0.2j), run_time=3)

        # Pan
        self.play(frame.animate.reorient(-50, 67, 5, (0.47, -1.1, 1.08), 11.24), run_time=10)

    def func(self, t):
        return np.cos(t)

    def transformed_func(self, s):
        return np.divide(s, s**2 + 1**2)


class SimplePole(InteractiveScene):
    def construct(self):
        frame = self.frame
        plane = ComplexPlane((-3, 3), (-3, 3))
        a = complex(-0.5 + 1.5j)
        graph = get_complex_graph(plane, lambda s: np.divide(1.0, (s - a)))

        self.add(plane, graph)
        frame.reorient(-22, 89, 0, (0.31, 0.85, 3.73), 12.49)
        self.play(frame.animate.reorient(23, 88, 0, (0.31, 0.85, 3.73), 12.49), run_time=10)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2025.laplace.exponentials module within the 3b1b videos codebase.",
      38: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      45: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      56: "SurfaceMesh draws wireframe grid lines on a Surface for spatial reference.",
      62: "IntegrateConstant extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      63: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      65: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      71: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      79: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      82: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      83: "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion.",
      85: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      92: "When True, stroke width scales with camera zoom. When False, strokes maintain constant screen-space width.",
      94: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      107: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      113: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      114: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      115: "Smoothly animates the camera to a new orientation over the animation duration.",
      120: "IntegrateRealExponential extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      121: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      124: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      127: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      134: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      139: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      141: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      147: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      150: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      151: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      152: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      153: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      154: "Restore animates a mobject back to a previously saved state (from save_state()).",
      161: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      163: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      165: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      166: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      169: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      172: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      174: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      182: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      184: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      185: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      187: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      188: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      189: "GrowArrow animates an arrow growing from its start point to full length.",
      190: "UpdateFromFunc calls a function on each frame to update a mobject's state.",
      193: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      195: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      202: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      203: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      204: "Smoothly animates the camera to a new orientation over the animation duration.",
      209: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      211: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      214: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      215: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      219: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      222: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      223: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      227: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      232: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      236: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      237: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      238: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      239: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      240: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      243: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      244: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      245: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      246: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      247: "VFadeInThenOut fades a mobject in then back out within a single animation.",
      250: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      252: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      580: "Class IntegrateComplexExponential inherits from SPlane.",
      603: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1404: "Class BreakDownLaplaceTransform inherits from IntegrateComplexExponential.",
      1413: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1630: "Class LaplaceTransformOfCos inherits from BreakDownLaplaceTransform.",
      1639: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1834: "SimplePole extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1835: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/laplace/main_equations.py"] = {
    description: "Core equation scenes for the Laplace series. Presents the key Laplace transform formulas, inverse transforms, and their derivations with step-by-step animations.",
    code: `from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph
from _2025.laplace.exponentials import SPlane
from _2025.laplace.exponentials import get_exp_graph_icon


class DrivenHarmonicOscillatorEquation(InteractiveScene):
    def construct(self):
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        equation = Tex(
            R"m x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega t)",
            t2c={
                "x(t)": colors[0],
                "x'(t)": colors[1],
                "x''(t)": colors[2],
                R"\\omega": PINK,
            }
        )
        self.add(equation)


class SimpleCosGraph(InteractiveScene):
    rotation_frequency = TAU / 4

    def construct(self):
        # Show graph
        def get_t():
            return self.time

        t_max = 20
        axes = Axes((0, t_max), (-1, 1), x_axis_config=dict(unit_size=0.6))
        axes.scale(1.25).to_edge(LEFT)

        x_axis = axes.x_axis

        x_axis.add(VGroup(
            Tex(tex, font_size=20).next_to(x_axis.n2p(n), DOWN)
            for n, tex in zip(it.count(1, 2), self.get_pi_frac_texs())
        ))

        def cos_func(t):
            return np.cos(self.rotation_frequency * t)

        graph = axes.get_graph(cos_func)
        graph.set_stroke(TEAL, 3)
        graph_ghost = graph.copy()
        graph_ghost.set_stroke(opacity=0.5)
        shown_graph = graph.copy()
        shown_graph.add_updater(lambda m: m.pointwise_become_partial(graph, 0, get_t() / t_max))

        output_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        output_dot.add_updater(lambda m: m.move_to(axes.y_axis.n2p(cos_func(get_t()))))

        h_line = Line()
        h_line.set_stroke(WHITE, 1)
        h_line.f_always.put_start_and_end_on(output_dot.get_center, shown_graph.get_end)

        cos_label = Tex(R"\\cos(t)", font_size=60)
        cos_label.to_edge(UP)

        self.add(axes, graph_ghost, output_dot, shown_graph, h_line)
        self.wait(10)
        self.play(FadeOut(x_axis))
        self.wait(14)

    def get_pi_frac_texs(self):
        return [
            R"\\pi / 2", R"\\pi", R"3 \\pi / 2", R"2\\pi",
            R"5\\pi / 2", R"3\\pi", R"7 \\pi / 2", R"4\\pi",
            R"9\\pi / 2", R"5\\pi",
        ]


class BreakUpCosineTex(InteractiveScene):
    def construct(self):
        # Show sum
        pure_cos = Tex(R"\\cos(t)", font_size=72)
        pure_cos.to_edge(UP)
        tex_pieces = [R"\\cos(t)", "=", R"\\frac{1}{2}", "e^{+it}", "{+}", R"{1 \\over 2}", "e^{-it}"]
        sum_tex = Tex(" ".join(tex_pieces), t2c={"+i": YELLOW, "-i": YELLOW})
        sum_tex.to_edge(UP, buff=MED_SMALL_BUFF).shift(0.5 * RIGHT)
        cos, equals, half1, eit, plus, half2, enit = pieces = VGroup(
            sum_tex[tex][0] for tex in remove_list_redundancies(tex_pieces)
        )

        self.add(pure_cos)
        self.wait()
        self.play(
            FadeTransform(pure_cos, cos),
            Write(pieces[1:])
        )
        self.wait()

        # Fade parts
        pieces.generate_target()
        pieces.target.set_fill(opacity=0.35)
        pieces.target[3].set_fill(opacity=1)
        pieces.target.space_out_submobjects(1.2)
        self.play(MoveToTarget(pieces))
        self.wait()
        self.play(VGroup(plus, enit).animate.set_fill(opacity=1))
        self.wait()
        self.play(pieces.animate.set_fill(opacity=1))
        self.play(pieces.animate.space_out_submobjects(1 / 1.2))
        self.wait()


class TranslateToNewLanguage(InteractiveScene):
    graph_resolution = (301, 301)
    show_integral = True
    label_config = dict(
        font_size=72,
        t2c={"{t}": BLUE, "{s}": YELLOW}
    )

    def construct(self):
        # Set up a functions
        full_s_samples = self.get_s_samples()
        func_s_samples = [
            complex(-2, 2),
            complex(-2, -2),
            complex(0, 1),  # Changed
            complex(-1, 0),
            complex(0, -1),  # Changed
        ]
        func_weights = [-1, -1, 1j, 2, -1j]

        def func(t):
            return sum([
                (weight * np.exp(complex(0.1 * s.real, s.imag) * t)).real
                for s, weight in zip(func_s_samples, func_weights)
            ])

        # Graph
        axes, graph, graph_label = self.get_graph_group(func)

        # Show the S-plane pieces
        frame = self.frame
        frame.set_y(0.5)
        s_plane, exp_pieces, s_plane_name = self.get_s_plane_and_exp_pieces(full_s_samples)

        self.play(LaggedStart(
            FadeIn(axes),
            ShowCreation(graph),
            FadeIn(graph_label),
            FadeIn(s_plane_name, lag_ratio=0.1),
            LaggedStartMap(FadeIn, exp_pieces, lag_ratio=0.1),
        ))
        self.wait()

        # Narrow down specific pieces
        exp_pieces.save_state()
        exp_pieces.generate_target()
        key_pieces = VGroup()
        for piece, s_sample in zip(exp_pieces.target, full_s_samples):
            if s_sample not in func_s_samples:
                piece.fade(0.7)
            else:
                key_pieces.add(piece)

        weight_labels = VGroup(
            Tex(Rf"\\times {w}", font_size=24).next_to(piece.get_top(), DOWN, SMALL_BUFF)
            for w, piece in zip([R"\\minus 1", R"\\minus 1", R"\\minus i", "2", "i"], key_pieces)
        )
        self.play(
            MoveToTarget(exp_pieces),
            LaggedStartMap(FadeIn, weight_labels),
        )
        self.play(LaggedStart(
            (Transform(graph.copy(), piece[-1].copy().insert_n_curves(100), remover=True)
            for piece in key_pieces),
            lag_ratio=0.1,
            group_type=Group,
            run_time=2
        ))
        self.wait()

        # Reveal plane
        frame = self.frame
        arrow, fancy_L, Fs_label = self.get_arrow_to_Fs(graph_label)
        Fs_label.save_state()
        Fs_label.become(graph_label)

        def Func(s):
            result = sum([
                np.divide(w, (s - s0))
                for s0, w in zip(func_s_samples, func_weights)
            ])
            return min(100, result)

        lt_graph = get_complex_graph(s_plane, Func, resolution=self.graph_resolution, face_sort_direction=DOWN)
        lt_graph.stretch(0.25, 2, about_point=s_plane.n2p(0))
        lt_graph.save_state()
        lt_graph.stretch(0, 2, about_point=s_plane.n2p(0))
        lt_graph.set_opacity(0)

        exp_pieces.target = exp_pieces.saved_state.copy()
        for piece in exp_pieces.target:
            piece.scale(0.35)

        self.add(exp_pieces, lt_graph, graph_label, arrow, Fs_label, Point(), weight_labels)
        self.play(
            FadeOut(s_plane_name),
            GrowArrow(arrow),
            Write(fancy_L),
            Restore(Fs_label, time_span=(1, 2), path_arc=-10 * DEG),
            FadeIn(s_plane),
            MoveToTarget(exp_pieces, lag_ratio=1e-3),
            FadeOut(weight_labels),
            Restore(lt_graph, time_span=(1.5, 3)),
            frame.animate.reorient(70, 86, 0, (-4.94, -2.45, 3.51), 19.43),
            run_time=3
        )

        # Show interal and continuation
        if self.show_integral:
            # For an insertion
            integral = Tex(R"= \\int^\\infty_0 f({t})e^{\\minus{s}{t}}d{t}", t2c=self.label_config["t2c"])
            integral.fix_in_frame()
            integral.next_to(Fs_label, RIGHT)
            integral.set_backstroke(BLACK, 5)
            rect = BackgroundRectangle(VGroup(Fs_label, integral))
            rect.set_fill(BLACK, 0.8)
            rect.scale(2, about_edge=DL)
            rect.shift(0.25 * DOWN)

            graph_copy = lt_graph[0].copy()
            graph_copy.set_clip_plane(RIGHT, -s_plane.get_left()[0])
            graph_copy.fade(0.5)

            self.add(rect, Fs_label, integral)
            self.play(
                FadeIn(rect),
                Write(integral, run_time=1)
            )
            self.wait()
            lt_graph.set_clip_plane(RIGHT, s_plane.get_left()[0])
            self.play(
                frame.animate.reorient(-10, 85, 0, (-2.18, 0.45, 3.12), 11.52),
                lt_graph.animate.set_clip_plane(RIGHT, -s_plane.n2p(0)[0]),
                run_time=3
            )
            self.play(
                frame.animate.reorient(33, 85, 0, (-1.81, 0.13, 2.44), 12.25),
                run_time=6
            )
            self.wait()
            self.add(graph_copy, rect, Fs_label, integral)
            self.play(
                FadeOut(rect),
                ShowCreation(graph_copy),
                frame.animate.reorient(-2, 67, 0, (-0.95, -0.06, 1.91), 9.45),
                run_time=8
            )

            # Show key exponentials below poles
            for piece in key_pieces:
                piece.set_height(1)
            self.add(key_pieces, graph_copy, graph)
            self.play(
                frame.animate.reorient(0, 0, 0, (-1.23, 1.49, 1.9), 9.36),
                FadeIn(key_pieces),
                exp_pieces.animate.fade(0.5),
                run_time=3,
            )

        # Reorient
        self.play(frame.animate.reorient(-39, 90, 0, (-1.37, 1.1, 4.12), 10.93), run_time=15)
        self.play(frame.animate.reorient(0, 0, 0, (-2.13, 0.07, 2.2), 9.79), run_time=10)
        self.play(frame.animate.reorient(84, 87, 0, (-4.22, -3.48, 5.26), 22.43), run_time=10)

        # Poles as lines
        pole_lines = VGroup(
            Line(s_plane.n2p(s), s_plane.n2p(s) + 20 * OUT)
            for s in func_s_samples
        )
        pole_lines.set_stroke(WHITE, 3)

        key_pieces.target = key_pieces.generate_target()
        target_rects = VGroup()
        for piece in key_pieces.target:
            piece.set_height(1.2)
            target_rect = piece[0].copy()
            target_rect.set_fill(opacity=0)
            target_rects.add(target_rect)

        self.add(pole_lines, key_pieces, lt_graph)
        self.play(
            ShowCreation(pole_lines, lag_ratio=0.1),
        )
        self.play(
            frame.animate.reorient(0, 0, 0, (-0.98, 0.82, 0.0), 10.00),
            lt_graph[0].animate.set_opacity(0.2),
            lt_graph[1].animate.set_opacity(0.05),
            pole_lines.animate.stretch(0, 2, about_edge=IN),
            MoveToTarget(key_pieces),
            FadeIn(weight_labels),
            run_time=3,
        )
        self.wait()

        # Shift things down
        top_rect = FullScreenRectangle()
        top_rect.set_fill(BLACK, 1).set_stroke(width=0)
        top_rect.set_height(2.5, about_edge=UP, stretch=True)
        top_rect.fix_in_frame()

        h_line = DashedLine(top_rect.get_corner(DL), top_rect.get_corner(DR))
        h_line.set_stroke(WHITE, 1)
        h_line.fix_in_frame()

        top_rect.save_state()
        top_rect.stretch(0, 1, about_edge=UP)

        self.play(
            Restore(top_rect),
            ShowCreation(h_line, time_span=(1, 2)),
            VGroup(axes, graph).animate.shift(2 * DOWN),
            VGroup(graph_label, arrow, fancy_L, Fs_label).animate.shift(3 * DOWN),
            frame.animate.reorient(-17, 90, 0, (-2.86, 1.57, 3.14), 10.95),
            run_time=2
        )
        self.play(frame.animate.reorient(39, 92, 0, (-4.35, 0.64, 3.03), 14.99), run_time=20)

    def get_graph_group(self, func, func_tex=R"f({t})"):
        # axes = Axes((0, 7), (-4, 4), width=0.5 * FRAME_WIDTH - 1, height=5)
        axes = Axes((0, 8), (-1, 6, 0.5), width=0.3 * FRAME_WIDTH - 1, height=7)
        axes.to_edge(LEFT).shift(0.5 * DOWN)
        graph = axes.get_graph(func)
        graph.set_stroke(BLUE, 5)
        graph.set_scale_stroke_with_zoom(True)
        axes.set_scale_stroke_with_zoom(True)
        graph_label = Tex(func_tex, **self.label_config)
        graph_label.move_to(axes).to_edge(UP, buff=LARGE_BUFF)

        graph_group = VGroup(axes, graph, graph_label)
        graph_group.fix_in_frame()
        return graph_group

    def get_s_samples(self):
        return [complex(a, b) for a in range(-2, 3) for b in range(-2, 3)]

    def get_s_plane_and_exp_pieces(self, s_samples):
        s_plane = ComplexPlane((-3, 3), (-3, 3))
        s_plane.set_height(7.5)
        s_plane.move_to(3.75 * RIGHT)
        s_plane.set_z_index(-1)

        exp_pieces = VGroup(
            self.get_exp_graph(s).move_to(s_plane.n2p(s))
            for s in s_samples
        )
        s_plane_name = Text("S-plane", font_size=72)
        s_plane_name.next_to(exp_pieces, UP, MED_SMALL_BUFF)
        return s_plane, exp_pieces, s_plane_name

    def get_arrow_to_Fs(self, graph_label):
        arrow = Vector(2 * RIGHT, thickness=5, fill_color=WHITE)
        arrow.fix_in_frame()
        arrow.next_to(graph_label, RIGHT, buff=MED_LARGE_BUFF)

        fancy_L = Tex(R"\\mathcal{L}", font_size=60)
        fancy_L.next_to(arrow, UP, buff=0)
        fancy_L.fix_in_frame()

        Fs_label = Tex(R"F({s})", **self.label_config)
        Fs_label.next_to(arrow, RIGHT, MED_LARGE_BUFF)
        Fs_label.fix_in_frame()
        Fs_label.set_z_index(1)
        Fs_label.set_backstroke(BLACK, 5)

        return VGroup(arrow, fancy_L, Fs_label)

    def get_exp_graph(self, s, **kwargs):
        return get_exp_graph_icon(s, **kwargs)


class TranslateDifferentialEquationAndInvert(InteractiveScene):
    def construct(self):
        # Translate the equations
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            R"{s}": YELLOW,
            R"x(t)": colors[0],
            R"x'(t)": colors[1],
            R"x''(t)": colors[2],
            R"\\omega": PINK,
        }
        kw = dict(t2c=t2c, font_size=30)
        lhs = Tex(R"m x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega t)", **kw)
        rhs1 = Tex(R"m {s}^2 X({s}) + \\mu {s} X({s}) + k X({s}) = \\frac{F_0 {s}}{({s}^2 + \\omega^2)}", **kw)
        rhs2 = Tex(R"X({s}) \\left( m {s}^2 + \\mu {s} + k \\right) = \\frac{F_0 {s}}{({s}^2 + \\omega^2)}", **kw)
        rhs3 = Tex(R"X({s}) = \\frac{F_0 {s}}{\\left({s}^2 + \\omega^2\\right) \\left( m {s}^2 + \\mu {s} + k \\right)}", **kw)

        for sign, term in zip([-1, 1, 1, 1], [lhs, rhs1, rhs2, rhs3]):
            term.set_x(sign * FRAME_WIDTH / 4)
            term.set_y(3.25)
            term.scale(2)
            term.set_x(0)
            term.set_y(-sign * 2)

        arrow = Arrow(lhs, rhs1, thickness=6, buff=0.5)
        arrow_label = Tex(R"\\mathcal{L}", font_size=72)
        arrow_label.next_to(arrow, RIGHT, SMALL_BUFF)

        ode_word = Text("Differential Equation")
        algebra_word = Text("Algebra")
        ode_word.next_to(lhs, DOWN)
        algebra_word.next_to(rhs1, DOWN)

        VGroup(ode_word, algebra_word).set_opacity(0)

        # Add domain backgrounds
        time_domain = FullScreenRectangle()
        time_domain.set_stroke(BLUE, 2)
        time_domain.set_fill(BLACK, 1)
        time_domain.stretch(0.5, 1, about_edge=UP)
        time_label = Text("Time domain")

        s_domain = time_domain.copy()
        s_domain.to_edge(DOWN, buff=0)
        s_domain.set_fill(GREY_E, 1)
        s_domain.set_stroke(YELLOW, 2)
        s_label = Text("s domain")

        for label, domain in [(time_label, time_domain), (s_label, s_domain)]:
            label.next_to(domain.get_corner(UL), DR)

        self.add(time_domain, time_label)
        self.add(s_domain, s_label)

        # Do the algebra
        self.play(
            FadeIn(lhs, lag_ratio=0.1),
            FadeIn(ode_word, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label, DOWN),
        )
        self.play(
            TransformMatchingTex(
                lhs.copy(),
                rhs1,
                path_arc=-10 * DEG,
                lag_ratio=3e-2,
                key_map={
                    R"F_0 \\cos(\\omega t)": R"\\frac{F_0 {s}}{({s}^2 + \\omega^2)}",
                    "x(t) = ": "X({s}) = ",
                    "x'(t)": R"{s} X({s})",
                    "x''(t)": "{s}^2 X({s})",
                    "(t)": "({s})",
                }
            )
        )
        self.wait()
        self.play(
            FadeIn(algebra_word, 0.5 * DOWN),
            TransformMatchingTex(rhs1, rhs2, path_arc=-30 * DEG),
        )
        self.play(
            TransformMatchingTex(
                rhs2,
                rhs3,
                path_arc=-10 * DEG,
                matched_keys=[R"\\left( m {s}^2 \\mu s + k \\right)", R"X({s})"]
            )
        )
        self.wait()

        # Show inversion
        inv_L = Tex(R"\\mathcal{L}^{-1}", font_size=72)
        inv_L.next_to(arrow, LEFT, buff=0)

        xt = Text(R"Solution", font_size=72)
        xt.next_to(arrow, UP, MED_LARGE_BUFF)

        self.play(LaggedStart(
            Rotate(arrow, PI),
            ReplacementTransform(arrow_label, inv_L),
            lhs.animate.scale(0.5).to_corner(UR),
        ))
        self.play(FadeIn(xt, 2 * UP))
        self.wait()


class DesiredMachine(InteractiveScene):
    show_ode = True

    def construct(self):
        # Add machine
        machine = self.get_machine()
        machine.rotate(90 * DEG)
        machine.center()
        fancy_L = Tex(R"\\mathcal{L}", font_size=120)
        fancy_L.move_to(machine)
        machine.set_z_index(1)
        fancy_L.set_z_index(1)

        self.add(machine, fancy_L)

        # Pump in a function
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        in_func = Tex(R"x({t})", t2c=t2c, font_size=90)
        in_func[re.compile("s_.")].set_color(YELLOW)
        in_func.next_to(machine, UP, MED_LARGE_BUFF)
        in_func_ghost = in_func.copy().set_fill(opacity=0.5)

        self.play(Write(in_func))
        self.wait()
        self.add(in_func_ghost)
        self.play(
            FadeOutToPoint(in_func, machine.get_bottom(), lag_ratio=0.025)
        )
        self.wait()

        # Pump in a differential equation
        if show_ode:
            ode = Tex(R"m x''({t}) + \\mu x'({t}) + k x(t) = F_0 \\cos(\\omega{t})", t2c=t2c, font_size=60)
            ode.next_to(machine, UP, MED_LARGE_BUFF)
            ode_ghost = ode.copy().set_fill(opacity=0.5)

            self.play(
                in_func_ghost.animate.to_edge(UP, buff=MED_SMALL_BUFF),
                FadeIn(ode, lag_ratio=0.1),
            )
            self.wait()
            self.add(ode_ghost)
            self.play(LaggedStart(
                (FadeOutToPoint(piece, machine.get_top() + 0.5 * DOWN, path_arc=arc)
                for piece, arc in zip(ode, np.linspace(-70 * DEG, 70 * DEG, len(ode)))),
                lag_ratio=0.05,
                run_time=2
            ))
            self.wait()

        # Result
        out_func = Tex(R"x({t}) = c_1 e^{s_1 {t}} + c_2 e^{s_2 {t}} + c_3 e^{s_3 {t}} + c_4 e^{s_4 {t}}", t2c=t2c, font_size=72)
        # out_func = Tex(R"x({t}) = \\sum_{n=1}^N c_n e^{s_n {t}}", t2c=t2c, font_size=72)
        s_parts = out_func[re.compile("s_.")]
        c_parts = out_func[re.compile("c_.")]
        s_parts.set_color(YELLOW)
        c_parts.set_color(GREY_A)
        out_func.next_to(machine, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            (FadeInFromPoint(piece, machine.get_bottom() + 0.5 * UP, path_arc=arc)
            for piece, arc in zip(out_func, np.linspace(-70 * DEG, 70 * DEG, len(out_func)))),
            lag_ratio=0.05,
            run_time=2
        ))

        # Make way for Laplace Transform words
        if False:
            # For an insert
            text = Text("Laplace Transform", font_size=72)
            machine.set_z_index(0)
            self.play(
                LaggedStart(
                    FadeOut(in_func_ghost, UP),
                    FadeOut(ode_ghost, 2 * UP),
                    FadeOut(out_func, DOWN),
                    FadeTransform(fancy_L[0], text[0]),
                    FadeIn(text[1:], lag_ratio=0.1),
                    run_time=2,
                    lag_ratio=0.2
                ),
                FadeOut(machine, scale=3, run_time=2)
            )
            self.wait()

        # Highlight s and c
        s_rects = VGroup(SurroundingRectangle(part, buff=0.05) for part in s_parts)
        c_rects = VGroup(SurroundingRectangle(part, buff=0.05) for part in c_parts)
        s_rects.set_stroke(YELLOW, 2)
        c_rects.set_stroke(WHITE, 2)

        s_part_copies = s_parts.copy()
        c_part_copies = c_parts.copy()

        self.add(s_part_copies)
        self.play(
            Write(s_rects),
            out_func.animate.set_opacity(0.75),
        )
        self.wait()
        self.play(
            ReplacementTransform(s_rects, c_rects, lag_ratio=0.1),
            FadeOut(s_part_copies),
            FadeIn(c_part_copies),
        )
        self.wait()
        self.play(FadeOut(c_rects), out_func.animate.set_fill(opacity=1))
        self.remove(c_part_copies)

        # Ask about exponential pieces
        mobs = Group(*self.mobjects)
        randy = Randolph()
        randy.move_to(5 * LEFT + 3 * DOWN, DL)
        randy.look_at(out_func),
        exp_piece = Tex(R"e^{{s}{t}}", t2c=t2c, font_size=90)
        exp_piece.next_to(randy, UR, LARGE_BUFF).shift(0.5 * DOWN)
        exp_piece.insert_submobject(2, VectorizedPoint(exp_piece[2].get_right()))

        self.play(
            LaggedStartMap(FadeOut, mobs, run_time=2),
            TransformFromCopy(out_func["e^{s_1 {t}}"][0], exp_piece, run_time=2),
            VFadeIn(randy, time_span=(0.5, 2.0)),
            randy.change("confused", exp_piece).set_anim_args(run_time=2),
        )
        self.play(Blink(randy))
        self.wait()
        for mode in ["pondering", "thinking", "tease"]:
            self.play(randy.change(mode, exp_piece))
            self.play(Blink(randy))
            self.wait(2)

    def get_machine(self, width=1.5, height=2, color=GREY_D):
        square = Rectangle(width, height)
        in_tri = ArrowTip().set_height(0.5 * height)
        in_tri.stretch(2, 1)
        out_tri = in_tri.copy().rotate(PI)
        in_tri.move_to(square.get_left())
        out_tri.move_to(square.get_right())
        machine = Union(square, in_tri, out_tri)
        machine.set_fill(color, 1)
        machine.set_stroke(WHITE, 2)
        return machine


class ExpDeriv(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        lhs, rhs = terms = VGroup(
            Tex(tex, t2c=t2c, font_size=90)
            for tex in [R"e^{{s}{t}}", R"{s} e^{{s}{t}}"]
        )
        terms.arrange(RIGHT, buff=4)

        arrows = VGroup(
            Arrow(
                lhs.get_corner(sign * UP + RIGHT),
                rhs.get_corner(sign * UP + LEFT),
                path_arc=-sign * 75 * DEG,
                thickness=4
            )
            for sign in [1, -1]
        )
        arrows.set_fill(border_width=2)
        arrows[1].shift(0.25 * DOWN)
        arrow_labels = VGroup(
            Tex(tex, t2c=t2c, font_size=60).next_to(arrow, vect)
            for arrow, tex, vect in zip(arrows, ["d / d{t}", R"\\times {s}"], [UP, DOWN])
        )
        self.add(terms[0])
        self.play(
            TransformFromCopy(terms[0].copy(), terms[1][1:], path_arc=-75 * DEG),
            Write(arrows[0], time_span=(0.5, 1.5)),
            FadeIn(arrow_labels[0], lag_ratio=0.1),
            run_time=1.5,
        )
        self.play(FadeTransform(terms[1][2].copy(), terms[1][0], path_arc=90 * DEG, run_time=0.75))
        self.play(LaggedStart(
            TransformFromCopy(*arrows),
            TransformFromCopy(*arrow_labels),
        ))
        self.wait(2)


class IntroduceTransform(SPlane):
    long_ambient_graph_display = False

    def construct(self):
        # Write "Laplace Transform"
        text = Text("Laplace Transform", font_size=72)
        laplace_word = text["Laplace"][0]
        transform_word = text["Transform"][0]
        laplace_transform_word = VGroup(laplace_word, transform_word)
        transform_word_rect = SurroundingRectangle(transform_word, buff=SMALL_BUFF)
        randy = Randolph().flip()
        randy.next_to(laplace_transform_word, DR)
        q_marks = Tex(R"???", font_size=72).space_out_submobjects(1.5)
        q_marks.next_to(transform_word_rect, UP)
        q_marks.set_color(YELLOW)

        self.add(laplace_transform_word)
        self.wait()
        self.play(LaggedStart(
            VFadeIn(randy),
            randy.change("confused", transform_word),
            ShowCreation(transform_word_rect),
            laplace_word.animate.set_opacity(0.5),
            LaggedStartMap(FadeIn, q_marks, shift=0.5 * UP, lag_ratio=0.25),
            lag_ratio=0.1
        ))
        self.wait()

        # Show a function with an arrow
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        name_group = VGroup(laplace_transform_word, transform_word_rect, q_marks)
        name_group.target = name_group.generate_target(use_deepcopy=True)
        name_group.target[2].set_fill(opacity=0).move_to(transform_word_rect)
        name_group.target.scale(0.5).to_edge(UP)
        name_group.target[1].set_stroke(width=0)

        mapsto = Tex(R"\\xmapsto{\\qquad}", additional_preamble=R"\\usepackage{mathtools}")
        mapsto.rotate(-90 * DEG)
        mapsto.set_fill(border_width=2)

        t_mob, _, ft_mob = func_group = VGroup(Tex(R"{t}", t2c=t2c), mapsto, Tex(R"f({t})", t2c=t2c))
        func_group.arrange(DOWN)
        func_name = Text("function")
        func_name.next_to(mapsto, LEFT, SMALL_BUFF)

        mapsto.save_state()
        mapsto.stretch(0, 1, about_edge=UP).set_fill(opacity=0)

        self.play(
            MoveToTarget(name_group),
            randy.change("pondering", func_group),
            FadeIn(func_name, lag_ratio=0.1),
        )
        self.play(
            Restore(mapsto),
            FadeIn(t_mob, 0.1 * UP)
        )
        self.play(LaggedStart(
            TransformFromCopy(t_mob, ft_mob[2], path_arc=45 * DEG, run_time=1),
            FadeTransform(func_name[0].copy(), ft_mob[0], path_arc=45 * DEG, run_time=1),
            Write(ft_mob[1::2]),
            lag_ratio=0.25,
        ))
        self.wait()

        # Show the meta-notion
        func_group.generate_target()
        func_group.target.arrange(DOWN)
        func_group.target.move_to(4 * LEFT)

        braces = VGroup(
            Brace(func_group.target, direction, SMALL_BUFF).scale(1.25, about_point=func_group.target.get_center())
            for direction in [LEFT, RIGHT]
        )
        braces.save_state()
        braces.set_opacity(0)
        for brace in braces:
            brace.replace(func_group, dim_to_match=1)

        short_func_name = Tex(R"f")
        short_func_name.next_to(func_group.target[1], LEFT, buff=0)

        right_arrow = Vector(3 * RIGHT, thickness=6)
        right_arrow.next_to(braces.saved_state, RIGHT, MED_LARGE_BUFF)

        self.play(LaggedStart(
            MoveToTarget(func_group),
            Restore(braces),
            LaggedStart(
                [FadeTransform(func_name[0], short_func_name)] + [
                    char.animate.set_opacity(0).replace(short_func_name)
                    for char in func_name[1:]
                ],
                lag_ratio=0.02,
                remover=True
            ),
            GrowArrow(right_arrow),
            lag_ratio=0.05
        ))
        self.play(transform_word.animate.scale(1.5).next_to(right_arrow, UP))
        self.play(Blink(randy))
        self.wait()

        # Show output function
        f_group = VGroup(braces, t_mob, short_func_name, mapsto, ft_mob)
        F_braces, s_mob, F_mob, right_mapsto, Fs_mob = F_group = VGroup(
            braces.copy(),
            Tex(R"{s}", t2c=t2c),
            Tex(R"F"),
            mapsto.copy(),
            Tex(R"F({s})", t2c=t2c),
        )
        for mob1, mob2 in zip(f_group, F_group):
            mob2.move_to(mob1)
        F_group[2].shift(SMALL_BUFF * LEFT)

        F_group.next_to(right_arrow, RIGHT, MED_LARGE_BUFF)

        self.play(
            LaggedStart(
                (TransformFromCopy(*pair, path_arc=-60 * DEG)
                for pair in zip(f_group, F_group)),
                lag_ratio=0.025,
                # run_time=1.5
                run_time=8
            ),
            randy.change("tease").scale(0.75).to_corner(DR),
        )
        self.wait()

        # Talk through parts
        rect = SurroundingRectangle(F_group[-1]["F"], buff=0.05)
        rect.set_stroke(RED, 3)
        vect = Vector(0.5 * UP, thickness=4)
        vect.set_color(RED)
        vect.next_to(rect, DOWN, SMALL_BUFF)

        self.play(laplace_word.animate.set_fill(opacity=1).scale(1.5).next_to(transform_word, UP, MED_SMALL_BUFF, LEFT))
        self.play(
            ShowCreation(rect),
            GrowArrow(vect),
            randy.change("pondering", rect),
        )
        self.wait()
        self.play(
            rect.animate.surround(f_group[-1][0], buff=0.05),
            MaintainPositionRelativeTo(vect, rect),
        )
        self.wait()
        self.play(Blink(randy))
        self.play(
            rect.animate.surround(f_group[1], buff=0.05).set_anim_args(path_arc=-90 * DEG),
            vect.animate.rotate(PI).next_to(f_group[1], UP, SMALL_BUFF).set_anim_args(path_arc=-90 * DEG),
        )
        self.wait()
        self.play(
            rect.animate.surround(F_group[1], buff=0.05).set_anim_args(path_arc=-90 * DEG),
            MaintainPositionRelativeTo(vect, rect),
            randy.animate.look_at(F_group),
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(vect), FadeOut(rect))

        # Show Laplace Transform expression
        lt_def = Tex(R"\\int^\\infty_0 f({t}) e^{\\minus {s}{t}} d{t}", t2c=t2c)
        lt_def.move_to(Fs_mob, LEFT)
        new_interior = VGroup(s_mob, F_mob, right_mapsto, lt_def)
        F_braces.generate_target()
        F_braces.target.set_height(new_interior.get_height() + MED_LARGE_BUFF, stretch=True)
        F_braces.target.match_y(right_arrow)
        F_braces.target[1].next_to(new_interior, RIGHT).match_y(F_braces.target[0])

        q_marks = Tex(R"???").set_color(YELLOW)
        q_marks.next_to(randy, UP, SMALL_BUFF).shift(0.25 * RIGHT + 0.5 * DOWN)

        self.play(
            FadeOut(Fs_mob),
            Write(lt_def),
            randy.change("pleading").scale(0.75, about_edge=DR),
            FadeIn(q_marks, 0.1 * UP, lag_ratio=0.1),
            MoveToTarget(F_braces),
            new_interior[:3].animate.match_x(lt_def).shift(0.1 * UP),
        )
        self.play(Blink(randy))
        self.wait()

        # Discuss expression
        rect = SurroundingRectangle(lt_def)
        rect.set_stroke(YELLOW, 2)

        s_vect = Vector(0.25 * DOWN)
        s_vect.set_fill(YELLOW)
        s_vect.next_to(lt_def["{s}"][0], UP, SMALL_BUFF)

        self.play(
            ShowCreation(rect),
            randy.change('confused')
        )
        self.wait()
        self.play(
            Transform(rect, s_vect.copy().match_style(rect)),
            FadeIn(s_vect, time_span=(0.75, 1)),
        )
        self.play(
            randy.change("pondering", s_vect),
            FadeOut(q_marks),
            FadeOut(rect),
        )
        self.play(Blink(randy))
        self.wait()

        # Show F(s) left hand side
        F_lhs = Tex(R"F({s}) = ", t2c=t2c)
        F_lhs.move_to(lt_def, LEFT)

        right_shift = F_lhs.get_width() * RIGHT

        self.play(
            TransformFromCopy(s_mob, F_lhs["{s}"][0]),
            TransformFromCopy(F_mob, F_lhs["F"][0]),
            Write(F_lhs[re.compile(r"\\(|\\)|=")]),
            lt_def.animate.shift(right_shift),
            MaintainPositionRelativeTo(s_vect, lt_def),
            F_braces[1].animate.shift(right_shift),
            new_interior[:3].animate.shift(right_shift + 0.1 * UP),
            randy.animate.shift(0.5 * DOWN),
            self.frame.animate.shift(0.5 * DOWN),
        )

        # Reference the naming convention
        F_rect = SurroundingRectangle(F_lhs[0], buff=0.05)
        F_rect.set_stroke(RED, 2)

        self.play(
            ShowCreation(F_rect),
            s_vect.animate.next_to(F_rect, UP, SMALL_BUFF).set_color(RED)
        )
        self.wait()
        self.play(
            F_rect.animate.surround(ft_mob[0], buff=0.05),
            s_vect.animate.next_to(ft_mob[0], UP, SMALL_BUFF),
            randy.animate.look_at(ft_mob)
        )
        self.play(Blink(randy))
        self.wait()
        self.play(FadeOut(F_rect), FadeOut(s_vect))

        # Clear the board
        lt_group = VGroup(F_lhs, lt_def)

        self.remove(lt_group)
        everything = Group(*self.get_mobjects())
        everything.sort(lambda p: np.dot(p, UL))

        self.play(
            LaggedStartMap(FadeOut, everything, shift=0.1 * DOWN, lag_ratio=0.1),
            lt_group.animate.center().set_height(2),
            self.frame.animate.to_default_state(),
            run_time=2,
        )

        # Highlight inner part
        outer_part = VGroup(lt_def[R"\\int^\\infty_0"][0], lt_def[R"d{t}"][0])
        inner_part = lt_def[R"f({t}) e^{\\minus {s}{t}}"][0]

        rect = SurroundingRectangle(inner_part, buff=SMALL_BUFF)
        outer_rects = VGroup(SurroundingRectangle(piece, buff=SMALL_BUFF) for piece in outer_part)
        VGroup(rect, outer_rects).set_stroke(RED, 2)

        self.play(
            ShowCreation(rect),
            outer_part.animate.set_fill(opacity=0.2),
            F_lhs.animate.set_fill(opacity=0.2),
        )
        self.wait()
        self.play(
            # TransformFromCopy(rect.replicate(2), outer_rects),
            rect.animate.surround(outer_part, buff=SMALL_BUFF),
            outer_part.animate.set_fill(opacity=1),
            inner_part.animate.set_fill(opacity=0.5),
        )
        self.wait()

        # Put at the top
        lt_group = VGroup(F_lhs, outer_part, inner_part)
        lt_group.target = lt_group.generate_target()
        lt_group.target.set_height(1.2).to_edge(UP, buff=MED_SMALL_BUFF)
        lt_group.target[:2].set_fill(opacity=0.2)
        lt_group.target[2].set_fill(opacity=1)

        lt_group.target.set_fill(opacity=1)

        self.play(
            MoveToTarget(lt_group),
            # rect.animate.surround(lt_group.target[2]).set_stroke(width=1),
            rect.animate.surround(lt_group.target).set_stroke(width=0),
        )
        self.wait()
        self.play(
            rect.animate.surround(lt_def["s"][0], buff=0.05).set_stroke(YELLOW)
        )
        self.wait()

        # Show s as a complex number
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.next_to(lt_group, DOWN)
        plane.add_coordinate_labels(font_size=16)

        s_dot = Group(TrueDot(), GlowDot()).set_color(YELLOW)
        s_dot.move_to(plane.n2p(2 + 1j))
        s_label = Tex(R"s").set_color(YELLOW)
        s_label.always.next_to(s_dot[0], UR, SMALL_BUFF)

        self.play(
            Write(plane, lag_ratio=1e-2),
            TransformFromCopy(lt_def["s"][0], s_label),
            FadeInFromPoint(s_dot, rect.get_center()),
            FadeOut(rect),
        )
        self.wait()

        # Wander
        final_z = -2 + 1j
        n_iterations = 12
        for n in range(12):
            z = complex(*np.random.uniform(-3, 3, 2))
            if n == n_iterations - 1:
                z = final_z
            self.play(s_dot.animate.move_to(plane.n2p(z)).set_anim_args(path_arc=45 * DEG))
            self.wait()

        # Plug in cos(t)
        lt_def.refresh_bounding_box()
        lt_of_cos = Tex(lt_def.get_tex().replace("f({t})", R"\\cos({t})"), t2c=t2c)
        lt_of_cos.move_to(lt_def, LEFT)
        cos_lt_group = VGroup(
            F_lhs,
            VGroup(lt_of_cos[R"\\int^\\infty_0"][0], lt_of_cos[R"d{t}"][0]),
            lt_of_cos[R"\\cos({t}) e^{\\minus {s}{t}}"][0]
        )
        cos_lt_group[1].set_fill(opacity=0.2)

        imag_circles = VGroup(
            Dot(plane.n2p(z)).set_stroke(YELLOW, 1).set_fill(opacity=0.25)
            for z in [1j, -1j]
        )

        f_term = lt_def["f"][-1]
        cos_term = lt_of_cos[R"\\cos"][0]
        f_rect = SurroundingRectangle(f_term, buff=0.05)
        f_rect.set_stroke(PINK, 2)

        self.play(ShowCreation(f_rect))
        self.play(
            TransformMatchingTex(
                lt_def,
                lt_of_cos,
                matched_pairs=[(f_term, cos_term)],
                run_time=1
            ),
            f_rect.animate.surround(cos_term, buff=0.05),
            F_lhs.animate.shift((lt_of_cos.get_width() - lt_def.get_width()) * 0.5 * LEFT),
            *map(ShowCreation, imag_circles),
        )
        self.play(FadeOut(f_rect))
        self.wait()

        # Create the graph
        frame = self.frame
        frame.to_default_state()
        cos_lt_group.fix_in_frame()
        self.add(cos_lt_group)

        graph = get_complex_graph(
            plane,
            lambda s: s / (s**2 + 1),
            resolution=(501, 501),
            mesh_resolution=(31, 31)
        )
        graph.stretch(0.25, 2, about_point=plane.n2p(0))

        # Show the graph
        self.play(
            cos_lt_group.animate.to_edge(LEFT, buff=MED_SMALL_BUFF).set_fill(opacity=1),
            ShowCreation(graph[0]),
            Write(graph[1], stroke_width=1, time_span=(2, 4)),
            frame.animate.reorient(-28, 74, 0, OUT, 8.37),
            run_time=4,
        )
        frame.clear_updaters()
        frame.add_ambient_rotation()
        for z in [1j, -1j]:
            self.play(s_dot.animate.move_to(plane.n2p(z)), run_time=2)
            self.wait(2)

        # Long ambient graph rotation (one branch here)
        if self.long_ambient_graph_display:
            self.remove(cos_lt_group)
            self.wait(19)
            self.add(imag_circles, graph)
            self.play(FadeOut(graph), FadeOut(imag_circles))
            frame.clear_updaters()
            s_plane = plane
            s_plane.save_state()
            s_plane.set_height(3.5)
            s_plane.to_corner(UR, buff=MED_SMALL_BUFF).shift(LEFT)
            s_plane.background_lines.set_stroke(BLUE, 1, 1)
            s_plane.target = s_plane.copy()

            s_dot.target = s_dot.generate_target()
            s_dot.target.move_to(s_plane.n2p(0.2 + 2j))

            s_plane.restore()
            self.play(
                frame.animate.to_default_state(),
                MoveToTarget(s_plane),
                MoveToTarget(s_dot),
                run_time=2
            )

        # Remove graphs
        self.remove(plane, graph, s_dot, s_label, imag_circles)
        frame.to_default_state()
        frame.clear_updaters()

        # Ignore the integral
        interior = cos_lt_group[2]
        interior_rect = SurroundingRectangle(interior, buff=0.05)
        interior_rect.set_stroke(TEAL, 2)

        self.add(interior)
        self.play(
            ShowCreation(interior_rect),
            cos_lt_group[:2].animate.set_fill(opacity=0.2),
        )
        self.wait()

        # Replace break down cos(t) with exponentials
        arrow = Vector(DOWN)
        arrow.match_color(interior_rect)
        arrow.next_to(interior_rect, DOWN, SMALL_BUFF)
        expanded = Tex(
            R"\\frac{1}{2} \\left(e^{i{t}} + e^{\\minus i{t}} \\right) e^{\\minus{s}{t}}",
            t2c={"i": WHITE, "-i": WHITE, **t2c},
        )
        expanded.next_to(arrow, DOWN, MED_LARGE_BUFF)
        expanded_brace = Brace(expanded, UP, SMALL_BUFF)
        expanded_brace.set_color(TEAL)

        index = -4
        self.play(
            FadeTransform(interior[:index].copy(), expanded[:index]),
            TransformFromCopy(interior[index:], expanded[index:]),
            GrowArrow(arrow),
            FadeInFromPoint(expanded_brace, arrow.get_start()),
        )
        self.wait()

        # Focus on just part
        eit, est = pair = VGroup(expanded["e^{i{t}}"][0], expanded[R"e^{\\minus{s}{t}}"][0])
        pair_rects = VGroup(SurroundingRectangle(p, buff=0.05) for p in pair)
        pair_rects.set_stroke(BLUE, 2)
        pair_copy = pair.copy()

        self.add(pair_copy),
        self.play(
            expanded.animate.set_fill(opacity=0.5),
            ShowCreation(pair_rects, lag_ratio=0.5)
        )
        self.wait()

        # Combine
        combined_term = Tex(R"e^{(i - {s}){t}}", t2c=t2c)
        combined_term.next_to(pair, DOWN, LARGE_BUFF)
        comb_arrows = VGroup(Arrow(p, combined_term, buff=0.25, thickness=2) for p in pair)
        comb_arrows.set_fill(BLUE)

        self.play(
            TransformMatchingShapes(pair.copy(), combined_term),
            *map(GrowArrow, comb_arrows),
            run_time=1
        )
        self.wait()

        # Pull up s-plane and output plane
        s_plane = plane
        s_plane.set_height(3.5)
        s_plane.to_corner(UR, buff=MED_SMALL_BUFF).shift(LEFT)
        s_plane.background_lines.set_stroke(BLUE, 1, 1)

        s_tracker = ComplexValueTracker(0.2 + 2j)
        t_tracker = ValueTracker(0)
        get_s = s_tracker.get_value
        get_t = t_tracker.get_value

        def get_ims():
            return 1j - get_s()

        s_dot.add_updater(lambda m: m.move_to(s_plane.n2p(get_s())))

        exp_plane = self.get_exp_plane()
        exp_plane.set_height(3.5)
        exp_plane.next_to(s_plane, DOWN, MED_LARGE_BUFF)
        output_label = self.get_output_dot_and_label(
            exp_plane,
            get_s=get_ims,
            get_t=get_t,
            s_tex="(i - {s})"
        )
        output_path = self.get_output_path(exp_plane, get_t, get_ims)

        self.play(FadeIn(s_plane), FadeIn(s_dot), FadeIn(s_label))
        self.play(
            FadeIn(exp_plane),
            FadeTransform(combined_term.copy(), output_label[1]),
            FadeIn(output_label[0]),
        )
        self.add(output_path)
        self.play(t_tracker.animate.set_value(2 * TAU), rate_func=linear, run_time=10)
        self.play(s_tracker.animate.increment_value(-0.3), rate_func=there_and_back, run_time=4)
        self.play(s_tracker.animate.set_value(0.2), run_time=4)
        self.play(s_tracker.animate.set_value(1j), run_time=4)
        self.wait()

        # Let t play
        eq_i_rhs = Tex(R"= i")
        eq_i_rhs.next_to(s_label, RIGHT, SMALL_BUFF).shift(0.04 * UP)
        self.play(Write(eq_i_rhs), run_time=1)

        t_tracker.set_value(0)
        self.play(t_tracker.animate.set_value(20), run_time=20, rate_func=linear)

        # Show s = i
        down_arrow = Vector(0.75 * DOWN)
        down_arrow.next_to(combined_term, DOWN, SMALL_BUFF)
        arrow_label = Tex(R"{s} = i", font_size=24, t2c=t2c)
        arrow_label.next_to(down_arrow, RIGHT, buff=0)
        const = Tex(R"e^{0{t}}", t2c=t2c)
        const.next_to(down_arrow, DOWN, SMALL_BUFF)
        eq_1 = Tex(R"= 1")
        eq_1.next_to(const, RIGHT, SMALL_BUFF, aligned_edge=DOWN)

        self.play(
            GrowArrow(down_arrow),
            FadeIn(arrow_label),
            FadeIn(const, DOWN),
        )
        self.wait()
        self.play(Write(eq_1, run_time=1))
        self.wait()

        # Move around s
        s_tracker.clear_updaters()
        t_tracker.set_value(100)
        output_label.clear_updaters()

        self.play(
            FadeOut(eq_i_rhs),
            FadeOut(output_label),
        )

        self.play()
        for _ in range(6):
            self.play(
                s_tracker.animate.set_value(complex(random.uniform(-0.1, 0.5), random.uniform(-3, 3))),
                run_time=2
            )
            self.wait()
        self.play(s_tracker.animate.set_value(1j), run_time=2)
        self.remove(output_path)

        self.wait()

        # State the goal
        big_rect = SurroundingRectangle(expanded[:-4])
        big_rect.set_stroke(BLUE, 2)
        goal = Text("Goal:\\nReveal these terms", alignment="LEFT")
        goal.next_to(big_rect, DOWN, aligned_edge=LEFT)

        self.play(
            ReplacementTransform(pair_rects, VGroup(big_rect)),
            LaggedStartMap(FadeOut, VGroup(comb_arrows, combined_term, down_arrow, arrow_label, const, eq_1)),
            FadeOut(Group(exp_plane, s_plane, s_label, s_dot)),
            expanded.animate.set_fill(opacity=1),
        )
        self.play(FadeIn(goal, lag_ratio=0.1))
        self.wait()
        self.play(
            FadeOut(goal),
            big_rect.animate.surround(expanded),
        )
        self.wait()

        # Highlight the integral again
        self.add(expanded)
        self.remove(pair_copy)
        self.play(
            LaggedStartMap(FadeOut, VGroup(interior_rect, arrow, expanded_brace, expanded, big_rect)),
            cos_lt_group.animate.set_fill(opacity=1),
        )
        self.wait()


class SimpleToComplex(InteractiveScene):
    def construct(self):
        # Add key expressions
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        terms = VGroup(
            Tex(R"f({t}) = 1", t2c=t2c),
            Tex(R"f({t}) = e^{at}", t2c=t2c),
            Tex(R"f({t}) = \\sum_{n=1}^N c_n e^{s_n {t}}", t2c=t2c),
        )
        terms.scale(1.5)
        terms.arrange(DOWN, buff=1.5)
        arrows = VGroup(
            Arrow(*pair, buff=0.25, thickness=5)
            for pair in zip(terms, terms[1:])
        )
        terms[2].shift(SMALL_BUFF * DOWN)

        self.add(terms[0])
        for arrow, term1, term2 in zip(arrows, terms, terms[1:]):
            self.play(
                TransformMatchingTex(
                    term1.copy(),
                    term2,
                    key_map={"1": "e^{at}", "a": "s_n"}
                ),
                GrowArrow(arrow),
                run_time=1
            )
            self.wait()

        # Set up Laplace Transforms
        lt_left_x = terms.get_x(RIGHT) + 3
        lt_arrows = VGroup(
            Arrow(
                term.get_right(),
                lt_left_x * RIGHT + term.get_y() * UP,
                buff=0.5,
                thickness=6
            )
            for term in terms
        )
        lt_arrows.set_color(GREY_A)
        lt_arrow_labels = VGroup(
            Tex(R"\\mathcal{L}").next_to(arrow, UP, buff=0)
            for arrow in lt_arrows
        )
        lt_integrals = VGroup(
            Tex(
                R"\\int^\\infty_0 " + tex + R"\\cdot e^{\\minus{s}{t}}d{t}",
                t2c=t2c
            )
            for tex in ["1", "e^{a{t}}", R"\\sum_{n=1}^N c_n e^{s_n {t}}"]
        )
        lt_rhss = VGroup(
            Tex(R"= \\frac{1}{{s}}", t2c=t2c),
            Tex(R"= \\frac{1}{{s} - a}", t2c=t2c),
            Tex(R"= \\sum_{n=1}^N \\frac{c_n}{{s} - s_n}", t2c=t2c),
        )
        for integral, arrow, rhs in zip(lt_integrals, lt_arrows, lt_rhss):
            integral.next_to(arrow, RIGHT)
            rhs.scale(1.25)
            rhs.next_to(integral, RIGHT)

        # Show laplace transforms
        frame = self.frame

        self.play(
            LaggedStart(
                GrowArrow(lt_arrows[0]),
                Write(lt_arrow_labels[0]),
                FadeIn(lt_integrals[0], RIGHT),
                lag_ratio=0.3
            ),
            terms[1:].animate.set_fill(opacity=0.1),
            arrows.animate.set_fill(opacity=0.1),
            frame.animate.set_x(4),
            run_time=2
        )
        self.wait()
        self.play(Write(lt_rhss[0]))
        self.wait()
        for index in [1, 2]:
            self.play(
                arrows[index - 1].animate.set_fill(opacity=1),
                terms[index].animate.set_fill(opacity=1),
            )
            self.play(
                LaggedStart(
                    GrowArrow(lt_arrows[index]),
                    Write(lt_arrow_labels[index]),
                    FadeIn(lt_integrals[index], RIGHT),
                    lag_ratio=0.3
                ),
            )
            self.wait()
            self.play(Write(lt_rhss[index]))
            self.wait()
        self.play(frame.reorient(0, 0, 0, (5.5, 0.04, 0.0), 10), run_time=2)


class SetSToMinus1(InteractiveScene):
    def construct(self):
        eq = Tex(R"\\frac{1}{\\minus 1} = \\minus 1")
        eq[R"\\minus 1"][0].set_color(YELLOW)
        self.play(Write(eq))
        self.wait()


class RealExtension(InteractiveScene):
    def construct(self):
        # Show limited domain
        axes = Axes((-1, 10), (-1, 5), width=FRAME_WIDTH - 1, height=6)
        self.add(axes)

        def func(x):
            decay = math.exp(-0.05 * (x + 1))
            poly = -0.003 * x**3 - 0.2 * (0.15 * x)**2 + 0.2 * x
            return (decay + 0.5) * (math.cos(1.0 * x) + 1.5) + poly

        limited_domain = (2, 6)

        partial_graph = axes.get_graph(func, x_range=limited_domain)
        partial_graph.set_stroke(BLUE, 5)
        f_label = Tex(R"f(x)")
        f_label.next_to(partial_graph.get_end(), UL)

        limited_domain_line = Line(
            axes.c2p(limited_domain[0], 0),
            axes.c2p(limited_domain[1], 0),
        )
        limited_domain_line.set_stroke(BLUE, 5)
        limited_domain_words = Text("Limited Domain")
        limited_domain_words.next_to(limited_domain_line, UP, SMALL_BUFF)

        self.add(axes)
        self.play(
            ShowCreation(partial_graph),
            Write(f_label)
        )
        self.play(
            ShowCreation(limited_domain_line),
            FadeIn(limited_domain_words, lag_ratio=0.1)
        )
        self.wait()
        self.play(TransformFromCopy(limited_domain_line, partial_graph))
        self.wait()

        # Extend the graph
        points = partial_graph.get_anchors()

        def get_extension(nudge_size=0):
            pre_xs = np.arange(1, -2, -1)
            post_xs = np.arange(7, 11)
            result = VGroup(
                self.get_extension(axes, points[3::-1], pre_xs, func, nudge_size=nudge_size),
                self.get_extension(axes, points[-4:], post_xs, func, nudge_size=nudge_size),
            )
            result[0].set_clip_plane(LEFT, axes.c2p(limited_domain[0], 0)[0])
            result[1].set_clip_plane(RIGHT, -axes.c2p(limited_domain[1], 0)[0])
            return result

        extension = get_extension()
        self.play(ShowCreation(extension, lag_ratio=0, run_time=4))

        # Change around
        extension.save_state()
        for n in range(5):
            new_extension = get_extension(nudge_size=3)
            self.play(extension.animate.become(new_extension), run_time=1)
        self.play(Restore(extension))

        # Show a derivative
        x_tracker = ValueTracker(limited_domain[0])
        tan_line = always_redraw(lambda : axes.get_tangent_line(
            x_tracker.get_value(), partial_graph, length=2
        ).set_stroke(WHITE, 3))

        self.play(GrowFromCenter(tan_line, suspend_mobject_updating=True))
        self.play(x_tracker.animate.set_value(limited_domain[1]), run_time=5)
        self.play(FadeOut(tan_line, suspend_mobject_updating=True))

        # Wiggly spaghetti
        def tweaked_func(x):
            x0, x1 = limited_domain
            if x < x0:
                return func(x) + 0.5 * (x - x0)**2
            elif x < x1:
                return func(x)
            else:
                return func(x) - 0.5 * (x - x1)**2

        full_graph = axes.get_graph(func)
        modifed_graph = axes.get_graph(tweaked_func)

        group = VGroup(full_graph, modifed_graph)
        group.set_stroke(RED, 5)
        group.set_z_index(-1)

        self.play(
            FadeOut(extension),
            FadeIn(full_graph),
        )
        self.play(
            full_graph.animate.become(modifed_graph),
            rate_func=lambda t: wiggle(t, 5),
            run_time=8
        )

    def get_extension(self, axes, pre_start, xs, func, nudge_size=0, stroke_width=5, stroke_color=RED):
        ys = np.array([func(x) + (nudge_size * (random.random() - 0.5)) for x in xs])
        new_points = axes.c2p(xs, ys)

        result = VMobject()
        result.set_points_smoothly([*pre_start, *new_points], approx=False)
        result.insert_n_curves(100)

        result.set_stroke(stroke_color, stroke_width)
        result.set_z_index(-1)
        return result


class ComplexExtension(InteractiveScene):
    def construct(self):
        # Set up input and output planes
        input_plane, output_plane = planes = VGroup(
            ComplexPlane((-3, 3), (-3, 3))
            for n in range(2)
        )
        planes.set_height(5)
        planes.arrange(RIGHT, buff=1.5)
        for plane in planes:
            plane.axes.set_stroke(WHITE, 1, 1)
            plane.background_lines.set_stroke(BLUE, 1, 0.5)
            plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        self.add(planes)

        # Set up limited domain
        domain = self.get_rect_group(2, 1, input_plane, 0.2 + 0.2j)
        domain.set_z_index(2)
        self.add(domain)

        # Show a mapping
        def func(z):
            return -0.05j * z**3

        def point_func(points):
            return np.array([
                output_plane.n2p(func(input_plane.p2n(p)))
                for p in points
            ])

        mapped_domain = domain.copy().apply_points_function(point_func, about_edge=None)
        mapped_domain.set_z_index(2)

        arrow = Arrow(
            domain.get_top(),
            mapped_domain.get_top(),
            path_arc=-90 * DEG,
            fill_color=TEAL,
            thickness=5
        )
        arrow.set_z_index(1)
        func_label = Tex(R"f(z)")
        func_label.next_to(arrow, UP, SMALL_BUFF)
        func_label.set_backstroke()

        self.play(
            Write(arrow, time_span=(0, 1)),
            Write(func_label, time_span=(0.5, 1.5)),
            ReplacementTransform(
                domain.copy().set_fill(opacity=0),
                mapped_domain,
                path_arc=-90 * DEG,
                run_time=3
            )
        )

        # Show the extension
        dark_red = interpolate_color(RED_E, BLACK, 0.5)
        extension = self.get_extended_domain(input_plane, domain, 4, 2, corner_value=-1, color=dark_red)
        mapped_extension = extension.copy().apply_points_function(point_func, about_edge=None)

        self.play(Write(extension, run_time=1, lag_ratio=1e-2))
        self.wait()
        self.play(
            TransformFromCopy(extension.copy().set_fill(opacity=0), mapped_extension, path_arc=-90 * DEG),
            run_time=3
        )
        self.wait()

        # Two possibilities
        frame = self.frame
        possibilities = VGroup(
            Text("1) There is no such extension", font_size=72),
            Text("2) There is only one extension", font_size=72),
        )
        possibilities.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        possibilities.next_to(planes, UP, buff=LARGE_BUFF)
        only_one = possibilities[1]["only one"][0]
        underline = Underline(only_one, buff=-SMALL_BUFF).set_stroke(YELLOW)

        self.play(
            Write(possibilities[0][:2]),
            Write(possibilities[1][:2]),
            frame.animate.reorient(0, 0, 0, (0, 1.5, 0.0), 9).set_anim_args(time_span=(0, 1)),
            run_time=2
        )
        self.wait()
        self.play(FadeIn(possibilities[0][2:], lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(possibilities[1][2:], lag_ratio=0.1))
        self.play(
            ShowCreation(underline),
            only_one.animate.set_fill(YELLOW)
        )
        self.wait()

        # Analytic continuation
        ac_words = Text("Analytic Continuation", font_size=72)
        ac_words.next_to(output_plane, UP, MED_LARGE_BUFF)

        darkest_red = interpolate_color(RED_E, BLACK, 0.9)
        big_domain = self.get_extended_domain(input_plane, extension, 6, 6, corner_value=-3 - 3j, color=darkest_red)
        big_domain.set_stroke(WHITE, 0.5, 0.1)
        big_domain.set_fill(dark_red, 0.5)
        mapped_big_domain = big_domain.copy().apply_points_function(point_func, about_edge=None)

        self.play(
            FadeOut(possibilities, lag_ratio=0.05),
            FadeOut(underline),
            Write(ac_words, time_span=(0.5, 2.)),
        )
        self.play(
            Write(mapped_big_domain, lag_ratio=5e-2, stroke_color=RED, stroke_width=0.5),
            Write(big_domain, lag_ratio=5e-2, stroke_color=RED, stroke_width=0.5),
            run_time=8
        )
        self.wait()

    def get_extended_domain(self, plane, domain, width, height, corner_value, color=RED_E):
        extension = self.get_rect_group(width, height, plane, corner_value=corner_value, color=color)

        min_x = domain.get_x(LEFT)
        max_x = domain.get_x(RIGHT)
        min_y = domain.get_y(DOWN)
        max_y = domain.get_y(UP)
        to_remove = list()
        for square in extension:
            if (min_x < square.get_x() < max_x) and (min_y < square.get_y() < max_y):
                to_remove.append(square)
        extension.remove(*to_remove)
        extension.sort(lambda p: get_norm(p - domain.get_center()))
        return extension

    def get_rect_group(self, width, height, plane, corner_value=0, square_density=5, color=BLUE_E):
        square = Square(side_length=plane.x_axis.get_unit_size() / square_density)
        square.set_stroke(WHITE, 0.5)
        square.set_fill(color, 1)
        square.insert_n_curves(20)
        grid = square.get_grid(square_density * height, square_density * width, buff=0)
        grid.move_to(plane.n2p(corner_value), DL)
        return grid

        rect = Rectangle(width, height)
        rect.set_width(width * plane.x_axis.get_unit_size())
        rect.move_to(plane.n2p(corner_value), DL)
        rect.set_stroke(color, 2)
        rect.set_fill(color, 1)
        rect.insert_n_curves(200)

        rect_lines = VGroup(
            Line(DOWN, UP).get_grid(1, width * line_density + 1, buff=SMALL_BUFF),
            Line(LEFT, RIGHT).get_grid(height * line_density + 1, 1, buff=SMALL_BUFF),
        )
        for group in rect_lines:
            group.replace(rect, stretch=True)
            group.set_stroke(WHITE, 1, 0.5)
            for line in group:
                line.insert_n_curves(20)

        return VGroup(rect, rect_lines)


class WriteFPrimeExists(InteractiveScene):
    def construct(self):
        words = TexText("$f'(z)$ Exists")
        self.play(Write(words))
        self.wait()


class ZetaFunctionPlot(InteractiveScene):
    # resolution = (51, 51)
    resolution = (1001, 1001)  # Probably takes like an hour to compute

    def construct(self):
        # Planes
        x_max = 25
        s_plane = ComplexPlane((-x_max, x_max), (-x_max, x_max), faded_line_ratio=5)
        s_plane.set_height(40)
        s_plane.add_coordinate_labels(font_size=16)

        partial_plane = ComplexPlane((1, x_max), (-x_max, x_max))
        partial_plane.shift(s_plane.n2p(0) - partial_plane.n2p(0))

        self.add(s_plane)

        # True function
        import mpmath as mp

        def zeta_log_deriv(s):
            epsilon = 1e-4
            if s == 1:
                return 1 / epsilon
            out = mp.zeta(s)
            if abs(out) < 1e-3:
                return 1 / epsilon
            out_prime = (mp.zeta(s + epsilon) - out) / epsilon
            # return mp.zeta(s, derivative=1) / out
            return out_prime / out

        graph = get_complex_graph(
            s_plane,
            zeta_log_deriv,
            resolution=self.resolution,
        )
        graph.set_clip_plane(RIGHT, -1)
        graph.set_opacity(0.6)

        self.add(graph)

        # Panning
        frame = self.frame
        frame.reorient(24, 82, 0, (0.39, 0.58, 0.49), 4.02)
        self.play(
            frame.animate.reorient(-14, 79, 0, (-0.62, -0.08, 1.41), 8.12),
            run_time=8
        )
        self.play(
            graph.animate.set_clip_plane(RIGHT, x_max),
            frame.animate.reorient(-13, 80, 0, (-0.28, -0.01, 2.17), 12.57),
            run_time=5
        )
        self.play(
            frame.animate.reorient(32, 81, 0, (-0.98, -2.23, 5.19), 32.31),
            run_time=20
        )
        self.play(
            frame.animate.reorient(91, 84, 0, (-0.25, -1.54, 7.01), 32.31),
            run_time=15,
        )


class WriteZetaPrimeFact(InteractiveScene):
    def construct(self):
        # Test
        formula = Tex(
            R"\\frac{\\zeta'({s})}{\\zeta({s})} = \\sum_{spacer} \\sum_{k=1}^\\infty \\frac{1}{k} \\frac{1}{p^{s}}",
            t2c={"{s}": YELLOW}
        )
        spacer = formula["spacer"][0]
        p_prime = TexText("$p$ prime")
        p_prime.replace(spacer)
        p_prime.scale(0.8, about_edge=UP)
        formula.remove(*spacer)
        formula.add(*p_prime)
        formula.sort()

        formula.to_edge(UP, buff=MED_SMALL_BUFF)
        self.play(Write(formula))
        self.wait()


class SimpleExpToPole(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{t}": BLUE, "{s}": YELLOW}
        kw = dict(t2c=t2c, font_size=72)
        lhs, arrow, rhs = group = VGroup(
            Tex(R"e^{a{t}}", **kw),
            Vector(1.5 * RIGHT, thickness=5),
            Tex(R"{1 \\over {s} - a}", **kw)
        )
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.to_edge(UP, MED_LARGE_BUFF)
        fancy_L = Tex(R"\\mathcal{L}")
        fancy_L.next_to(arrow, UP, SMALL_BUFF)

        self.play(FadeIn(lhs))
        self.play(
            GrowArrow(arrow),
            Write(fancy_L),
            run_time=1,
        )
        self.play(
            Write(rhs[:-1]),
            TransformFromCopy(lhs["a"], rhs["a"])
        )
        self.wait()


class Linearity(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"f": GREEN, "g": BLUE, R"\\mathcal{L}": GREY_A, R"\\big\\{": WHITE, R"\\big\\}": WHITE, "a": GREEN_A, "b": BLUE_A}
        lhs = Tex(R"\\mathcal{L}\\big\\{a \\cdot f(t) + b \\cdot g(t) \\big\\}", t2c=t2c)
        rhs = Tex(R"a \\cdot \\mathcal{L} \\big\\{ f(t) \\big\\} + b \\cdot \\mathcal{L} \\big\\{ g(t) \\big\\}", t2c=t2c)
        arrow = Vector(RIGHT, thickness=4)
        group = VGroup(lhs, arrow, rhs)
        group.arrange(RIGHT)

        self.add(lhs)
        self.wait()
        self.play(
            GrowArrow(arrow),
            LaggedStart(
                AnimationGroup(*(
                    TransformFromCopy(lhs[tex][0], rhs[tex][0], path_arc=45 * DEG)
                    for tex in [R"\\mathcal{L}", R"\\big\\{", R"a \\cdot", "f(t)", R"\\big\\}"]
                )),
                AnimationGroup(*(
                    TransformFromCopy(lhs[tex][0], rhs[tex][-1], path_arc=-45 * DEG)
                    for tex in ["+", R"\\mathcal{L}", R"\\big\\{", R"b \\cdot", "g(t)", R"\\big\\}"]
                )),
                lag_ratio=0.75,
                run_time=3
            ),
        )
        self.wait()


class LaplaceTransformOfCosineSymbolically(InteractiveScene):
    def construct(self):
        # Add defining integral
        frame = self.frame
        t2c = {
            "{t}": BLUE,
            "{s}": YELLOW,
            R"\\omega": PINK,
            R"int^\\infty_0": WHITE,
        }
        key_strings = [
            R"int^\\infty_0",
            R"e^{\\minus{s}{t}} d{t}",
            "+",
            R"\\frac{1}{2}",
            R"e^{i{t}}",
            R"e^{\\minus i{t}}",
        ]
        kw = dict(isolate=key_strings, t2c=t2c)

        cos_t = Tex(R"\\cos({t})", **kw)
        cos_t.to_corner(UL, buff=LARGE_BUFF)
        arrow = Vector(1.5 * RIGHT)
        arrow.next_to(cos_t)
        fancy_L = Tex(R"\\mathcal{L}")
        fancy_L.next_to(arrow, UP, SMALL_BUFF)

        def lt_string(interior):
            return Rf"\\int^\\infty_0 " + interior + R"e^{\\minus{s}{t}} d{t}"

        lt_def = Tex(lt_string(R"\\cos({t})"), **kw)
        lt_def.next_to(arrow, RIGHT)

        self.add(cos_t)
        self.play(LaggedStart(
            GrowArrow(arrow),
            Write(fancy_L),
            Write(lt_def[R"\\int^\\infty_0"]),
            TransformFromCopy(cos_t, lt_def[R"\\cos({t})"][0], path_arc=45 * DEG),
            Write(lt_def[R"e^{\\minus{s}{t}} d{t}"]),
            lag_ratio=0.2,
        ))

        # Split up into exponential parts
        spilt_cos_str = R"\\left( \\frac{1}{2} e^{i{t}} + \\frac{1}{2} e^{\\minus i{t}} \\right)"
        split_inside = Tex("=" + lt_string(spilt_cos_str), **kw)
        split_inside.next_to(lt_def, RIGHT)

        cos_rect = SurroundingRectangle(lt_def[R"\\cos({t})"])
        cos_rect.set_stroke(TEAL, 2)

        self.play(ShowCreation(cos_rect))
        self.play(
            TransformMatchingTex(
                lt_def.copy(),
                split_inside,
                key_map={R"\\cos({t})": spilt_cos_str},
                path_arc=30 * DEG,
                mismatch_animation=FadeTransform,
            ),
            cos_rect.animate.surround(split_inside[spilt_cos_str]).set_anim_args(path_arc=30 * DEG),
            run_time=1.5
        )
        self.play(FadeOut(cos_rect))
        self.wait()
        self.add(split_inside)

        # Rect growth
        self.play(cos_rect.animate.surround(split_inside[1:]).set_stroke(width=5))
        self.wait()
        self.play(FadeOut(cos_rect))

        # Linearity
        split_tex = " ".join([
            R"\\frac{1}{2}", lt_string(R"e^{i{t}}"), R"\\, + \\,",
            R"\\frac{1}{2}", lt_string(R"e^{\\minus i{t}}"),
        ])
        split_outside = Tex(split_tex, **kw)
        side_eq = Tex(R"=", font_size=72).rotate(90 * DEG)
        side_eq.next_to(split_inside, DOWN, MED_LARGE_BUFF)
        split_outside.next_to(side_eq, DOWN, MED_LARGE_BUFF)
        split_outside.shift_onto_screen()

        srcs = VGroup()
        trgs = VGroup()
        for tex in key_strings:
            src = split_inside[tex]
            trg = split_outside[tex]
            if tex is key_strings[0]:
                src = VGroup(part[:3] for part in src)
                trg = VGroup(part[:3] for part in trg)
            srcs.add(src)
            trgs.add(trg)

        self.play(
            Write(side_eq),
            LaggedStart(
                (TransformFromCopy(*pair)
                for pair in zip(srcs[:3], trgs[:3])),
                lag_ratio=0.01,
                run_time=2
            ),
        )
        self.wait()
        self.play(
            TransformFromCopy(srcs[3][0], trgs[3][0]),
            TransformFromCopy(srcs[4][0], trgs[4][0])
        )
        self.wait()
        self.play(
            TransformFromCopy(srcs[3][1], trgs[3][1]),
            TransformFromCopy(srcs[5][0], trgs[5][0])
        )
        self.wait()

        # Collapse to poles
        exp_transform_parts = VGroup(
            split_outside[lt_string(R"e^{i{t}}")],
            split_outside[lt_string(R"e^{\\minus i{t}}")],
        )
        pole_strings = [R"\\frac{1}{{s} - i}", R"\\frac{1}{{s} \\, + \\, i}"]
        half_string = R"\\frac{1}{2}"
        pole_sum = Tex(
            R" \\, ".join([half_string, pole_strings[0], "+", half_string, pole_strings[1]]),
            **kw
        )
        pole_sum.scale(1.25)
        pole_sum.move_to(split_outside).shift(0.2 * LEFT)

        split_inside_rect = SurroundingRectangle(split_inside[spilt_cos_str])
        exp_transform_rects = VGroup(
            SurroundingRectangle(part, buff=SMALL_BUFF)
            for part in exp_transform_parts
        )
        pole_rects = VGroup(
            SurroundingRectangle(pole_sum[tex], buff=SMALL_BUFF)
            for tex in pole_strings
        )

        VGroup(split_inside_rect, exp_transform_rects, pole_rects).set_stroke(TEAL, 2)

        self.play(ShowCreation(split_inside_rect))
        self.wait()
        self.play(LaggedStart(*(
            TransformFromCopy(split_inside_rect, rect)
            for rect in exp_transform_rects
        )))
        self.play(FadeOut(split_inside_rect))
        self.wait()
        for i, tex in enumerate([R"e^{i{t}}", R"e^{\\minus i{t}}"]):
            self.play(
                ReplacementTransform(exp_transform_rects[i], pole_rects[i]),
                ReplacementTransform(split_outside[half_string][i], pole_sum[half_string][i]),
                FadeTransform(split_outside[lt_string(tex)], pole_sum[pole_strings[i]]),
                Transform(split_outside["+"][0], pole_sum["+"][0])
            )
            self.play(FadeOut(pole_rects[i]))
        self.remove(split_outside)
        self.add(pole_sum)
        self.play(pole_sum.animate.match_x(side_eq))

        # Read it as "pole at i", etc.
        pole_rects = VGroup(
            SurroundingRectangle(pole_sum[tex], buff=SMALL_BUFF)
            for tex in pole_strings
        )
        pole_rects.set_stroke(YELLOW, 2)
        pole_words = VGroup(
            TexText(Rf"Pole at \\\\ $s = {value}$", font_size=60, t2c={"Pole at": YELLOW, "s": YELLOW})
            for value in ["i", "-i"]
        )

        last_group = VGroup()
        for word, rect in zip(pole_words, pole_rects):
            word.next_to(rect, DOWN, MED_LARGE_BUFF)
            self.play(
                FadeIn(word, lag_ratio=0.1),
                ShowCreation(rect),
                FadeOut(last_group)
            )
            self.wait()
            last_group = VGroup(word, rect)

        self.play(FadeOut(last_group))

        # Add an omega
        old_group = VGroup(cos_t, lt_def, split_inside, pole_sum)
        new_group = VGroup(
            Tex(R"\\cos(\\omega{t})", **kw),
            Tex(lt_string(R"\\cos(\\omega{t})"), **kw),
            Tex("=" + lt_string(R"\\left(\\frac{1}{2} e^{i\\omega{t}} + \\frac{1}{2}e^{\\minus i \\omega {t}} \\right)"), **kw),
            Tex(R" \\, ".join([
                half_string, R"\\frac{1}{{s} - \\omega i}", "+",
                half_string, R"\\frac{1}{{s} \\, + \\, \\omega i}",
            ]), **kw)
        )
        for new, old in zip(new_group, old_group):
            new.match_width(old)
            new.move_to(old)

        omegas = VGroup()
        for new in new_group:
            omegas.add(*new[R"\\omega"])

        omega_copies = omegas.copy()
        omegas.set_fill(opacity=0)
        omegas[0].set_fill(opacity=1)

        cos_omega = new_group[0]
        cos_omega.scale(1.25, about_edge=RIGHT)
        cos_omega_rect = SurroundingRectangle(cos_omega)
        cos_omega_rect.set_stroke(PINK, 2)

        self.play(
            ShowCreation(cos_omega_rect),
            TransformMatchingTex(cos_t, cos_omega),
            run_time=1
        )
        self.wait()
        self.play(
            LaggedStart(
                (TransformMatchingTex(old, new)
                for new, old in zip(new_group[1:], old_group[1:])),
                lag_ratio=0.05,
                run_time=1
            ),
            TransformFromCopy(
                omegas[0].replicate(len(omega_copies) - 1),
                omega_copies[1:],
                path_arc=30 * DEG,
                lag_ratio=0.1,
                run_time=2
            ),
        )
        self.remove(omega_copies)
        omegas.set_fill(opacity=1)
        self.add(new_group)
        self.play(FadeOut(cos_omega_rect))
        self.wait()

        # Simplify fraction
        lower_arrow = Tex(R"\\longleftarrow", font_size=60)
        lower_arrow.next_to(pole_sum, LEFT)

        transform_kw = dict(
            matched_keys=[
                R"{s}^2 \\,+\\, \\omega^2",
                R"{s} \\,+\\, \\omega i",
                R"{s} - \\omega i",
                R"\\over",
            ],
            key_map={
                R"({s} - \\omega i)({s} + \\omega i)": R"{s}^2 \\,+\\, \\omega^2"
            }
        )

        steps = VGroup(
            Tex(R"""
                \\frac{1}{2}\\left(
                {{s} \\,+\\, \\omega i \\over ({s} - \\omega i)({s} + \\omega i)} +
                {{s} - \\omega i \\over ({s} - \\omega i)({s} + \\omega i)}
                \\right)
            """, **kw),
            Tex(R"""
                \\frac{1}{2}\\left(
                {{s} \\,+\\, \\omega i \\over {s}^2 \\,+\\, \\omega^2} +
                {{s} - \\omega i \\over {s}^2 \\,+\\, \\omega^2}
                \\right)
            """, **kw),
            Tex(R"""
                \\frac{1}{2} {{s} \\,+\\, \\omega i \\,+\\, {s} - \\omega i \\over {s}^2 \\,+\\, \\omega^2}
            """, **kw),
            Tex(R"""
                \\frac{1}{2} {2{s} \\over {s}^2 \\,+\\, \\omega^2}
            """, **kw),
            Tex(R"{{s} \\over {s}^2 \\,+\\, \\omega^2}", **kw),
        )
        for step in steps:
            step.next_to(lower_arrow, LEFT)

        self.play(
            Write(lower_arrow),
            FadeTransform(pole_sum.copy(), steps[0]),
            frame.animate.set_height(8.5, about_edge=DR),
            run_time=2
        )
        for step1, step2 in zip(steps, steps[1:]):
            self.play(
                TransformMatchingTex(step1, step2, **transform_kw)
            )
            self.wait()

        # Circle answer
        answer = steps[-1]
        answer.target = answer.generate_target()
        answer.target.scale(1.5, about_edge=RIGHT)
        answer_rect = SurroundingRectangle(answer.target)
        answer_rect.set_stroke(TEAL, 3)
        self.play(
            ShowCreation(answer_rect),
            MoveToTarget(answer)
        )
        self.wait()

        # Highlight direct equality
        lt_def, int_of_expanded, imag_result = new_group[-3:]

        direct_equals = Tex(R"=", font_size=90)
        direct_equals.rotate(90 * DEG)
        direct_equals.next_to(lt_def, DOWN, MED_LARGE_BUFF)

        to_fade = VGroup(int_of_expanded, side_eq, imag_result)

        self.play(
            lower_arrow.animate.set_fill(opacity=0),
            to_fade.animate.set_fill(opacity=0.2),
            answer_rect.animate.surround(VGroup(lt_def, answer)),
            Write(direct_equals),
        )
        self.wait()
        self.play(
            lower_arrow.animate.set_fill(opacity=1).rotate(PI).set_anim_args(path_arc=PI),
            answer_rect.animate.surround(VGroup(answer, imag_result)),
            imag_result.animate.set_fill(opacity=1),
        )
        self.wait()


class SimplePolesOverImaginaryLine(InteractiveScene):
    def construct(self):
        s_plane = ComplexPlane((-5, 5), (-5, 5))
        s_plane.add_coordinate_labels()
        omega = 2
        graph = get_complex_graph(
            s_plane,
            lambda s: (s**2) / (s**2 + omega**2 + 1e-6),
            resolution=(301, 301)
        )
        graph[0].sort_faces_back_to_front(DOWN)
        graph[1].set_clip_plane(OUT, 0)
        self.add(s_plane, graph)

        # Pan
        frame = self.frame
        frame.reorient(-57, 78, 0, (-1.28, 0.48, 0.92), 9.65)
        self.play(frame.animate.reorient(59, 78, 0, (-0.31, -0.1, 0.87), 10.53), run_time=12)


class IntegrationByParts(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{s}": YELLOW, "{t}": BLUE, R"\\omega": PINK}
        steps = VGroup(
            Tex(R"X = \\int^\\infty_0 \\cos(\\omega{t}) e^{\\minus {s}{t}}d{t}", t2c=t2c),
            Tex(R"X = \\left[\\frac{1}{\\omega} \\sin(\\omega{t}) e^{\\minus {s}{t}} \\right]_0^\\infty - \\int^\\infty_0 \\frac{1}{\\omega} \\sin(\\omega {t}) \\left(\\minus {s} e^{\\minus{s}{t}} \\right) d{t}", t2c=t2c),
            Tex(R"X = \\frac{s}{\\omega} \\int^\\infty_0 \\sin(\\omega{t}) e^{\\minus{s}{t}} d{t}", t2c=t2c),
            Tex(R"X = \\frac{s}{\\omega} \\left(\\left[\\frac{\\minus 1}{\\omega} \\cos(\\omega{t}) e^{\\minus {s}{t}} \\right]_0^\\infty - \\int^\\infty_0 \\frac{\\minus 1}{\\omega} \\cos(\\omega{t}) \\left(\\minus {s} e^{\\minus{s}{t}} \\right) d{t} \\right)", t2c=t2c),
            Tex(R"X = \\frac{s}{\\omega} \\left(\\frac{1}{\\omega} - \\frac{s}{\\omega} \\int^\\infty_0 \\cos(\\omega{t}) e^{\\minus{s}{t}} d{t} \\right)", t2c=t2c),
            Tex(R"X = \\frac{s}{\\omega^2} \\left(1 - {s} X \\right)", t2c=t2c),
            Tex(R"X\\left(\\omega^2 + {s}^2 \\right) = {s}", t2c=t2c),
            Tex(R"X = \\frac{s}{\\omega^2 + {s}^2}", t2c=t2c),
        )
        steps.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        steps.to_edge(LEFT, buff=LARGE_BUFF)

        randy = Randolph(mode="raise_left_hand")
        randy.to_edge(DOWN)
        steps[0].save_state()
        steps[0].next_to(randy, UL, MED_LARGE_BUFF)
        randy.look_at(steps[0])

        ibp = Tex(R"\\int u \\, dv = uv - \\int v \\, du", t2c={"u": RED, "v": PINK})
        ibp.next_to(randy, UR, MED_LARGE_BUFF)

        self.add(randy, steps[0])
        self.play(Blink(randy))
        self.play(
            randy.change("raise_right_hand", ibp),
            FadeIn(ibp, UP)
        )
        self.wait()
        self.play(
            self.frame.animate.set_height(steps.get_height() + 3, about_edge=LEFT),
            Restore(steps[0]),
            Write(steps[1:], run_time=2),
            randy.change("pondering", 5 * UL).shift(6 * RIGHT + 2 * DOWN),
            ibp.animate.shift(6 * RIGHT + 3 * DOWN),
        )
        self.play(randy.animate.look_at(steps[-1]))
        self.wait()


class AlternateBreakDown(TranslateToNewLanguage):
    def construct(self):
        # Set up
        axes, graph, graph_label = self.get_graph_group(
            lambda t: 0.35 * t**2,
            func_tex=R"f({t}) = {t}^2"
        )
        graph_label.set_backstroke(BLACK, 8)
        s_samples = self.get_s_samples()
        s_plane, exp_pieces, s_plane_name = self.get_s_plane_and_exp_pieces(s_samples)
        arrow, fancy_L, Fs_label = self.get_arrow_to_Fs(graph_label)

        self.add(axes, graph, graph_label)
        self.add(exp_pieces)

        # Note equal to a sum
        ne = Tex(R"\\ne", font_size=96)
        ne.next_to(graph_label, RIGHT, MED_LARGE_BUFF)
        ne.set_color(RED)
        sum_tex = Tex(
            R"\\sum_{n=1}^N c_n e^{s_n t}",
            t2c={"s_n": YELLOW, "c_n": GREY_A},
            font_size=72,
        )
        sum_tex.next_to(ne, RIGHT)
        ne_rhs = VGroup(ne, sum_tex)

        self.play(LaggedStart(
            FadeIn(ne, scale=2),
            Write(sum_tex),
            exp_pieces.animate.scale(0.7, about_edge=DR),
            lag_ratio=0.2
        ))
        self.wait()

        # Show transform
        self.play(
            LaggedStart(
                FadeOut(graph_label[4:]),
                graph_label[:4].animate.next_to(arrow, LEFT),
                GrowArrow(arrow),
                Write(fancy_L),
                TransformFromCopy(graph_label, Fs_label, path_arc=20 * DEG),
                lag_ratio=0.05
            ),
            FadeOut(ne_rhs, DOWN, scale=0.5),
        )
        self.wait()

        # Show integral
        inv_lt = Tex(
            R"f({t}) = \\frac{1}{2\\pi i} \\int_\\gamma F({s}) e^{{s}{t}} d{s}",
            t2c=self.label_config["t2c"]
        )
        inv_lt.next_to(arrow, DOWN, LARGE_BUFF)

        s_plane.replace(exp_pieces)
        s_plane.add_coordinate_labels(font_size=12)
        line = Line(s_plane.get_bottom(), s_plane.get_top())
        line.shift(RIGHT)
        line.set_stroke(YELLOW, 2)
        line.insert_n_curves(20)

        self.play(
            TransformFromCopy(graph_label[:4], inv_lt[:4]),
            TransformFromCopy(Fs_label, inv_lt["F({s})"][0]),
            Write(inv_lt[4:]),
        )
        self.play(
            FadeIn(s_plane),
            LaggedStartMap(FadeOut, exp_pieces, scale=0.1),
        )
        self.play(
            VShowPassingFlash(line.copy()),
            ShowCreation(line),
            run_time=5
        )
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports get_complex_graph from the _2025.laplace.integration module within the 3b1b videos codebase.",
      3: "Imports SPlane from the _2025.laplace.exponentials module within the 3b1b videos codebase.",
      4: "Imports get_exp_graph_icon from the _2025.laplace.exponentials module within the 3b1b videos codebase.",
      7: "DrivenHarmonicOscillatorEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      8: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      9: "Creates a list of colors smoothly interpolated between the given endpoints.",
      10: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      22: "SimpleCosGraph extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      25: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      31: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      37: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      42: "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion.",
      49: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      51: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      52: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      58: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      62: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      63: "FadeOut transitions a mobject from opaque to transparent.",
      64: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      74: "BreakUpCosineTex extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      75: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      77: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      80: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      87: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      88: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      90: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      92: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      100: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      101: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      102: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      103: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      104: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      105: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      108: "TranslateToNewLanguage extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      116: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      130: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      142: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      143: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      144: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      145: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      146: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      147: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      149: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      152: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      162: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      378: "TranslateDifferentialEquationAndInvert extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      379: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      488: "DesiredMachine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      491: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      632: "ExpDeriv extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      633: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      672: "Class IntroduceTransform inherits from SPlane.",
      675: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1276: "SimpleToComplex extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1277: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1376: "SetSToMinus1 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1377: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1384: "RealExtension extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1385: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1497: "ComplexExtension extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1498: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1660: "WriteFPrimeExists extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1661: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1667: "ZetaFunctionPlot extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1671: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1729: "WriteZetaPrimeFact extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1730: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1749: "SimpleExpToPole extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1750: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1777: "Linearity extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1778: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1807: "LaplaceTransformOfCosineSymbolically extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1808: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2135: "SimplePolesOverImaginaryLine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2136: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2155: "IntegrationByParts extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2156: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2199: "Class AlternateBreakDown inherits from TranslateToNewLanguage.",
      2200: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/laplace/main_supplements.py"] = {
    description: "Supplementary material for the main Laplace transform video, including alternative derivations and extended examples.",
    code: `from manim_imports_ext import *
from _2025.laplace.integration import get_complex_graph


class WriteLaplace(InteractiveScene):
    def construct(self):
        title = Text("Laplace Transform", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class TwoLevels(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Solve equations"),
            Text("See what it’s doing"),
        )
        words.scale(1.5)
        for word, y in zip(words, (2, -2)):
            word.set_y(y)
            word.to_edge(LEFT, buff=LARGE_BUFF)
            self.play(Write(word))
            self.wait()


class TwoKeyIdeas(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"{s}": YELLOW, "{s_n}": YELLOW, "{t}": BLUE}
        title = Text("Key ideas from the last chapter", font_size=72)
        title.to_edge(UP)
        underline = Underline(title, buff=-0.05)

        self.add(title, underline)

        ideas = VGroup(
            TexText(R"1) How to think about $e^{{s}{t}}$", t2c=t2c),
            TexText(R"2) Often, for functions in physics, $\\displaystyle f({t}) = \\sum_{n=1}^N c_n e^{{s_n} {t}}$", t2c=t2c),
        )
        for idea in ideas:
            idea[:2].scale(1.25, about_edge=RIGHT)
        ideas[0][R"$e^{{s}{t}}$"].scale(1.5, about_edge=DL)
        ideas.arrange(DOWN, aligned_edge=LEFT, buff=LARGE_BUFF)
        ideas.to_edge(LEFT)

        numbers = VGroup(idea[:2] for idea in ideas)
        words = VGroup(idea[2:] for idea in ideas)

        self.play(LaggedStartMap(FadeIn, numbers, shift=LEFT, lag_ratio=0.5))
        self.wait()
        for word in words:
            self.play(Write(word))
            self.wait()


class LevelsOfUnderstanding(InteractiveScene):
    def construct(self):
        items = BulletedList(
            "1) Use",
            "2) Dissect",
            "3) Reinvent",
            font_size=72,
            buff=2
        )
        numbers = VGroup()
        words = VGroup()
        for item in items:
            item[0].scale(0).set_opacity(0)
            numbers.add(item[1:3])
            words.add(item[3:])
        items.to_edge(LEFT)

        self.play(LaggedStartMap(FadeIn, numbers, shift=DOWN, lag_ratio=0.35))
        self.wait()
        for word in words:
            self.play(Write(word))
            self.wait()


class DrivingACar(InteractiveScene):
    def construct(self):
        # Test
        car = Car()
        car.move_to(4 * LEFT)
        car[0][2].set_fill(BLUE).insert_n_curves(100)
        self.add(car)
        self.play(MoveCar(4 * RIGHT), run_time=10)
        self.wait()


class WhatIsItTryingToDo2(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.change("raise_right_hand", look_at=3 * UP),
            self.change_students("thinking", "confused", "pondering", look_at=3 * UR)
        )
        self.wait(2)
        self.play(LaggedStart(
            stds[0].change("pondering", morty.eyes),
            stds[1].says("What does it\\nactually do?", mode="raise_left_hand", look_at=morty.eyes),
            stds[2].change("hesitant", morty.eyes),
            morty.change("tease"),
            lag_ratio=0.3
        ))
        self.wait(4)

        # Test2
        self.play(
            stds[0].change("pondering", 3 * UP),
            stds[1].debubble(look_at=3 * UP),
            stds[2].change("pondering", 3 * UP),
            morty.change("raise_right_hand"),
        )
        self.wait(3)


class ReferenceComplexExponents(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students('pondering', 'thinking', 'tease', look_at=self.screen)
        )
        self.wait(3)
        self.play(self.change_students('tease', 'tease', 'confused', look_at=self.screen))
        self.wait(2)
        self.play(self.change_students('thinking', 'tease', 'erm', look_at=self.screen))
        self.wait(2)
        self.play(
            morty.change('hesitant', stds[2].eyes),
            stds[2].says("Why?", mode='maybe', bubble_direction=LEFT),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            stds[2].debubble(mode="pondering", look_at=3 * UR),
            stds[0].change("pondering", look_at=3 * UR),
            stds[1].change("hesitant", look_at=3 * UR),
        )
        self.wait(3)


class ReferenceWorkedExample(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        rect = Rectangle(8, 1.5)
        rect.next_to(morty, UP).to_edge(RIGHT).shift(UP)
        rect.set_stroke(YELLOW, 2)
        self.play(
            morty.change("raise_left_hand", rect),
            self.change_students("pondering", "thinking", "erm", look_at=rect),
        )
        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            morty.change("raise_right_hand", self.screen),
            FadeOut(rect),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(7)


class ButWhatIsIt(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            stds[0].change("erm", self.screen),
            stds[1].says("But, what is it?", mode="maybe", bubble_direction=LEFT),
            stds[2].change("sassy", self.screen),
            morty.change("guilty"),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", 2 * UR),
            stds[0].change("pondering", 2 * UR),
            stds[1].debubble(mode="pondering", look_at=2 * UR),
            stds[2].change("pondering", 2 * UR),
        )
        self.wait(3)


class MoreInAMoment(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.body.insert_n_curves(1000)
        self.play(morty.says("Much, much more on\\nthis in a moment"))
        self.play(morty.change("tease"))
        self.play(Blink(morty))
        self.wait()


class YouAsAMathematician(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=2.5)
        randy.flip()
        randy.to_edge(DOWN, buff=MED_SMALL_BUFF)

        you = Text("You")
        you.next_to(randy, UR, LARGE_BUFF)
        mathematician = Text("(Mathematician)")
        mathematician.next_to(you, DOWN, aligned_edge=LEFT)
        arrow = Arrow(you.get_corner(DL), randy.body.get_corner(UR) + 0.7 * LEFT, buff=0.1, thickness=5)

        self.play(
            randy.change('pondering', 3 * UL),
            GrowArrow(arrow),
            Write(you)
        )
        self.play(FadeIn(mathematician, 0.25 * DOWN))
        self.play(Blink(randy))
        self.wait()

        # Reference machine
        label = VGroup(you, mathematician)
        self.play(
            randy.change("raise_left_hand", 3 * UR),
            label.animate.next_to(randy, RIGHT).set_opacity(0.5),
            FadeOut(arrow, scale=0.5, shift=DL),
        )
        self.play(Blink(randy))
        self.wait(2)
        self.play(
            randy.change("hesitant", 3 * UL)
        )
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)
        self.play(randy.change("pondering", 3 * UL))
        for _ in range(2):
            self.play(Blink(randy))


class FullCosInsideSum(InteractiveScene):
    def construct(self):
        tex = Tex(R"\\frac{1}{2} \\left( e^{(i - {s}){t}} + e^{(\\minus i - {s}){t}} \\right)", t2c={"{t}": BLUE, "{s}": YELLOW})
        self.add(tex)


class ReferenceTheIntegral(TeacherStudentsScene):
    def construct(self):
        # React to the preview
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("guilty"),
            self.change_students("confused", "concentrating", "pleading", look_at=self.screen)
        )
        self.wait(4)

        # Show expression
        t2c = {"{s}": YELLOW, "{t}": BLUE}
        lt_def = Tex(R"F({s}) = \\int^\\infty_0 f({t}) e^{\\minus {s} {t}} d{t}", t2c=t2c)
        lt_def.move_to(self.hold_up_spot, DOWN)
        lt_def.shift(0.5 * DOWN)

        self.play(
            morty.change("raise_right_hand", look_at=lt_def),
            FadeIn(lt_def, UP),
            self.change_students("confused", "tease", "happy", look_at=lt_def)
        )
        self.wait(2)

        # Highlight integral
        int_rect = SurroundingRectangle(lt_def[R"\\int^\\infty_0"])
        body_rect = SurroundingRectangle(lt_def[R"f({t}) e^{\\minus {s} {t}}"])
        VGroup(int_rect, body_rect).set_stroke(TEAL, 2)
        arrow = Vector(0.75 * DOWN, thickness=6)
        arrow.set_fill(TEAL)
        arrow.next_to(int_rect, UP)

        self.play(
            morty.change("hesitant", int_rect),
            ShowCreation(int_rect),
            GrowArrow(arrow),
            self.change_students("pondering", "happy", "tease"),
        )
        self.wait(2)

        # Show complex plane
        s_plane = ComplexPlane((-2, 2), (-2, 2))
        s_plane.add_coordinate_labels(font_size=16)
        s_plane.set_width(2.5)
        s_plane.next_to(body_rect, UP, LARGE_BUFF)
        s_plane.shift_onto_screen()
        s_plane.to_edge(UP, SMALL_BUFF)

        s = complex(-0.1, 1.5)
        t_max = 20
        func_path = ParametricCurve(
            lambda t: s_plane.n2p(2 * math.cos(t) * np.exp(s * t)),
            t_range=(1e-3, t_max, 0.1),
        )
        func_path.set_stroke(YELLOW, 2)
        dot = GlowDot()
        dot.move_to(func_path.get_start())
        tail = TracedPath(dot.get_center, stroke_color=YELLOW, time_traced=5)
        tail.add_updater(lambda m: m.set_stroke(YELLOW, width=(0, 3)))
        for t in range(30):
            tail.update(1 / 30)

        self.play(
            morty.change("raise_left_hand", s_plane),
            self.change_students("erm", "confused", "sassy", s_plane),
            arrow.animate.rotate(PI).scale(0.75).next_to(body_rect, UP).set_anim_args(path_arc=90 * DEG),
            ReplacementTransform(int_rect, body_rect),
        )
        self.add(tail)
        self.play(
            FadeIn(s_plane),
            MoveAlongPath(dot, func_path, run_time=10, rate_func=linear),
        )

        # Clear the board
        self.play(
            VFadeOut(tail),
            FadeOut(body_rect),
            FadeOut(arrow),
            FadeOut(s_plane),
            FadeOut(dot),
            lt_def.animate.move_to(2 * UP),
            morty.change("tease", look_at=2 * UP),
            self.change_students("pondering", "pondering", "pondering", look_at=2 * UP)
        )
        self.wait()

        # Remove f(t)
        lt_def.save_state()
        ft = lt_def[R"f({t})"][0]
        ft_rect = SurroundingRectangle(ft, buff=0.05)
        ft_rect.set_stroke(RED, 2)
        self.play(
            ShowCreation(ft_rect)
        )
        self.play(
            VGroup(ft_rect, ft).animate.to_corner(UL).set_stroke(width=0).fade(0.5),
            lt_def[R"e^{\\minus {s} {t}} d{t}"].animate.align_to(ft, LEFT).shift(SMALL_BUFF * LEFT),
            lt_def["F({s}) = "].animate.set_fill(opacity=0),
        )
        self.wait(5)

        # Bring back the definition
        exp_int = VGroup(*lt_def[R"\\int^\\infty_0"][0], *lt_def[R"e^{\\minus {s} {t}} d{t}"]).copy()
        exp_int_rhs = Tex(R"= \\frac{1}{{s}}", t2c=t2c)
        exp_int_rhs.next_to(exp_int, RIGHT)
        exp_int_rhs.save_state()
        exp_int_rhs.move_to(morty.get_corner(UL)).set_fill(opacity=0)
        morty.body.insert_n_curves(500)
        exp_int.add(*exp_int_rhs)

        ft2, arrow, fancy_L = new_def_lhs = VGroup(
            Tex(R"f({t})", t2c=t2c),
            Vector(1.5 * RIGHT),
            Tex(R"\\mathcal{L}"),
        )
        arrow.next_to(lt_def[R"\\int"], LEFT)
        fancy_L.next_to(arrow, UP, SMALL_BUFF)
        ft2.next_to(arrow, LEFT)

        lt_def.saved_state[:5].set_opacity(0)

        self.play(
            morty.change("raise_right_hand", exp_int_rhs),
            Restore(exp_int_rhs)
        )
        self.wait(2)
        self.play(
            exp_int.animate.scale(0.5).fade(0.5).to_corner(UR),
            Restore(lt_def),
            self.change_students("confused", "erm", "sassy", look_at=exp_int),
            morty.change("guilty", stds[2].eyes),
            Write(new_def_lhs),
        )
        self.wait()
        self.play(
            TransformFromCopy(ft2, ft, path_arc=PI / 2)
        )
        self.wait(3)

        # Substitute in an exponential
        exp1, exp2 = exp_examples = Tex(R"e^{1.5{t}}", t2c=t2c).replicate(2)
        ft.refresh_bounding_box()
        exp1.move_to(ft2)
        exp2.move_to(ft)
        exp_examples.align_to(lt_def["e"], DOWN)

        self.play(
            LaggedStart(
                FadeOut(ft2, 0.5 * UP),
                FadeIn(exp1, 0.5 * UP),
                FadeOut(ft, 0.5 * UP),
                FadeIn(exp2, 0.5 * UP),
                lag_ratio=0.25
            ),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "tease")
        )
        self.wait(4)

        # Collapse the inside
        lt_def["f({t})"].set_fill(opacity=0)
        new_int = Tex(R"\\int^\\infty_0 e^{(1.5 - {s}){t}} d{t}", t2c=t2c)
        new_int.move_to(lt_def[R"\\int"], LEFT).shift(1.0 * DOWN)
        new_int2 = Tex(R"\\int^\\infty_0 e^{\\minus({s} - 1.5){t}} d{t}", t2c=t2c)
        new_int2.move_to(new_int)

        self.play(
            VGroup(exp_examples, arrow, fancy_L, lt_def).animate.to_edge(UP),
            FadeIn(new_int, DOWN),
            self.change_students("pondering", "pondering", "pondering", new_int),
            morty.change("tease", new_int),
        )
        self.wait()
        self.play(TransformMatchingTex(new_int, new_int2, path_arc=90 * DEG))
        self.wait()

        # Show the answer
        exp_int.target = exp_int.generate_target()
        exp_int.target.scale(1.5, about_edge=UR)
        exp_int.target.set_opacity(1)
        exp_int_rect = SurroundingRectangle(exp_int.target, buff=MED_SMALL_BUFF)
        exp_int_rect.set_stroke(YELLOW, 2)

        example_rhs = Tex(R"= {1 \\over {s} - 1.5}", t2c=t2c)
        example_rhs.next_to(new_int2[-1], RIGHT)

        self.play(
            morty.change('raise_left_hand', exp_int),
            MoveToTarget(exp_int),
            ShowCreation(exp_int_rect),
            self.change_students("pondering", "thinking", "happy", look_at=exp_int),
        )
        self.wait()
        self.play(
            TransformMatchingShapes(exp_int[-4:], example_rhs),
            FadeOut(exp_int[:-4]),
            exp_int_rect.animate.surround(example_rhs, buff=SMALL_BUFF),
            morty.change("raise_right_hand"),
            run_time=1.5,
        )
        self.wait()

        # Label it as pole
        words = Text("Pole at 1.5")
        words.next_to(exp_int_rect, UP)
        words.match_x(example_rhs[1:])
        words.set_color(YELLOW)

        self.play(
            Write(words),
            exp_int_rect.animate.surround(example_rhs[1:], buff=SMALL_BUFF),
            morty.change("tease", words),
        )
        self.wait(3)
        self.play(
            FadeOut(exp_int_rect),
            FadeOut(words),
        )

        # Make it general
        srcs = VGroup(exp1, exp2, new_int2, example_rhs)
        trgs = VGroup(
            Tex(R"e^{a{t}}", t2c=t2c),
            Tex(R"e^{a{t}}", t2c=t2c),
            Tex(R"\\int^\\infty_0 e^{\\minus(s - a){t}} d{t}", t2c=t2c),
            Tex(R"= {1 \\over {s} - a}", t2c=t2c),
        )
        example_rects = VGroup(
            SurroundingRectangle(src["1.5"], buff=0.05)
            for src in srcs
        )
        example_rects.set_stroke(TEAL, 2)
        for src, trg in zip(srcs, trgs):
            trg.move_to(src)

        self.play(Write(example_rects))
        self.wait()
        self.play(
            FadeOut(example_rects),
            LaggedStart(
                (TransformMatchingTex(src, trg, key_map={"1.5": "a"})
                for src, trg in zip(srcs, trgs)),
                lag_ratio=0.2,
                run_time=1.5,
            )
        )
        self.wait()

        # Write conclusion
        new_rhs = trgs[3]

        self.play(
            FadeOut(trgs[2]),
            new_rhs.animate.next_to(lt_def, RIGHT),
            self.change_students("thinking", "happy", "tease", look_at=new_rhs),
            morty.change("raise_right_hand", new_rhs),
        )

        big_arrow = Arrow(trgs[0].get_bottom(), new_rhs.get_bottom(), path_arc=90 * DEG, thickness=5)
        words = TexText(R"Exponentials $\\longrightarrow$ Poles")
        words.next_to(big_arrow, DOWN)

        self.play(
            Write(big_arrow),
            FadeIn(words, lag_ratio=0.1),
        )
        self.look_at(words)
        self.wait(3)

        # Clear
        self.remove(self.background)
        self.remove(self.pi_creatures)


class TryADifferentInterpretation(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        self.play(
            morty.says("Let’s try an\\nalternate\\ninterpretation"),
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.wait(2)
        self.play(self.change_students("maybe", "pondering", "thinking", look_at=self.screen))
        self.wait(4)


class AskAboutZero(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=2)
        self.play(randy.says(Tex(R"s = 0 ?"), mode="erm", look_at=3 * UP + 0.5 * RIGHT))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class SeemsDumbAndPointless(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        self.play(LaggedStart(
            stds[0].says("Seems dumb", mode="sassy", look_at=self.screen),
            Animation(Point()),
            stds[2].animate.look_at(self.screen),
            morty.change('guilty'),
            stds[1].says(
                "And pointless!",
                mode="maybe",
                look_at=self.screen,
                bubble_direction=LEFT,
                bubble_creation_class=FadeIn
            ),
            # stds[1].animate.look_at(self.screen),
            lag_ratio=0.25
        ))
        self.wait(7)

        # Show Analytic Continuation
        words = Text("Analytic\\nContinuation", font_size=60)
        words.next_to(morty, UP, LARGE_BUFF)
        words.match_x(morty.get_left())

        self.wait(2)
        self.play(
            stds[0].debubble("pondering", look_at=words),
            stds[1].debubble("pondering", look_at=words),
            stds[2].change("pondering", words),
            morty.change("raise_right_hand"),
            Write(words),
        )
        self.wait(7)


class ReferenceAnalyticContinuation(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)
        words = Text("Analytic\\nContinuation", font_size=60)
        words.next_to(morty, UP, LARGE_BUFF)
        words.match_x(morty.get_left())

        self.play(
            self.change_students("pondering", "pondering", 'pondering', look_at=self.screen),
            morty.change("raise_right_hand"),
            Write(words),
        )
        self.wait(4)
        self.play(self.change_students("erm", "thinking", "pondering", look_at=self.screen))
        self.wait(4)
        self.play(self.change_students("pondering", "tease", "thinking", look_at=self.screen))
        self.wait(4)


class PreviewAnalyticContinuation(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        plane = ComplexPlane((-3, 3), (-3, 3))
        plane.add_coordinate_labels(font_size=16)

        pole_values = [1 - 2.0j, -1 - 2.0j, 2 + 1j, -2 + 1j, 2.5j]
        graph = get_complex_graph(
            plane,
            lambda s: -1 * sum((np.divide(1, (s - p)) for p in pole_values))
        )
        graph.rotate(90 * DEG, about_point=plane.n2p(0))
        graph.sort_faces_back_to_front(DOWN)
        graph.set_opacity(0.8)

        frame.reorient(-1, 75, 0, (0.35, 0.49, 1.27), 7.85)
        self.add(plane, graph)
        self.play(
            ShowCreation(graph),
            frame.animate.reorient(-24, 72, 0, (0.97, 0.24, 0.97), 9.65),
            run_time=12
        )


class SimpleRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(4.5, 2)
        rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rect))
        self.wait()


class ReactingToCosineMachine(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_edge(RIGHT)
        randy.shift(DOWN)

        for mode in ["erm", "confused"]:
            self.play(randy.change(mode))
            self.play(Blink(randy))
            self.wait(2)

        # Consider expression
        randy.body.insert_n_curves(500)
        self.play(randy.change("thinking", look_at=3 * UR))
        self.play(Blink(randy))
        self.wait()
        self.play(randy.change("tease", look_at=randy.get_top() + 2 * UP))
        self.play(Blink(randy))
        self.wait(5)


class PonderingCosineMachine(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=1.5)
        morty.flip()
        morty.to_corner(UL)

        for mode in ["tease", "pondering"]:
            self.play(morty.change(mode, look_at=ORIGIN))
            self.play(Blink(morty))
            self.wait(2)


class LaplaceFourierContrast(InteractiveScene):
    def construct(self):
        # Titles and definitions
        self.add(FullScreenRectangle())

        titles = VGroup(
            Text("Laplace Transform"),
            Text("Fourier Transform"),
        )
        t2c = {"{t}": BLUE, "{s}": YELLOW, R"\\xi": RED}
        formulas = VGroup(
            Tex(R"F({s}) = \\int^\\infty_0 f({t}) e^{\\minus {s}{t}} d{t}", t2c=t2c),
            Tex(R"\\hat f(\\xi) = \\int^\\infty_{\\minus \\infty} f({t}) e^{\\minus 2\\pi i \\xi {t}} d{t}", t2c=t2c),
        )
        formulas.scale(0.75)

        for title, formula, sign in zip(titles, formulas, [-1, 1]):
            for mob in [title, formula]:
                mob.next_to(1.5 * UP, UP)
                mob.set_x(sign * FRAME_WIDTH / 4)
        titles[0].align_to(titles[1], UP)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.5 * UP, lag_ratio=0.5)
        )
        self.wait()
        self.play(
            titles.animate.to_edge(UP),
            FadeIn(formulas, 0.5 * UP)
        )
        self.wait()


class NotWhatYouWouldSee(InteractiveScene):
    def construct(self):
        morty = Mortimer().flip()
        morty.to_corner(DL)
        self.play(
            morty.says("Not the\\nstandard form", mode="hesitant")
        )
        self.play(Blink(morty))
        self.wait()
        self.play(morty.debubble(mode="pondering", look_at=5 * UL))
        self.play(Blink(morty))
        self.wait()


class CosLTLogicReversal(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        arrow = Tex(R"\\rightarrow", font_size=120)
        cos_lt = Tex(
            R"\\frac{s}{s^2 + \\omega^2}",
            font_size=90,
            t2c={"s": YELLOW, R"\\omega": PINK}
        )
        cos_lt.next_to(arrow, RIGHT, LARGE_BUFF)

        self.add(arrow, cos_lt)
        self.wait()
        self.play(cos_lt.animate.next_to(arrow, LEFT, LARGE_BUFF).set_anim_args(path_arc=120 * DEG), run_time=3)
        self.wait()


class CosineEqualsWhat(InteractiveScene):
    def construct(self):
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))
        cos = Tex(R"\\cos(t) = ???", font_size=90, t2c={"t": BLUE})
        q_marks = cos["???"][0]
        q_marks.scale(1.2, about_edge=UL)
        q_marks.space_out_submobjects(1.1)
        q_marks.shift(MED_SMALL_BUFF * RIGHT)
        self.add(cos)
        self.play(Write(q_marks))
        self.wait()


class OhLookAtTheTime(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        morty.body.insert_n_curves(500)
        # Test
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("dejected", "tired", "pondering", look_at=3 * UP)
        )
        self.wait(3)
        self.play(
            morty.change("hesitant", 5 * UR),
        )
        self.play(self.change_students("tease", "well", "happy"))
        self.wait(3)
        self.play(morty.change("raise_left_hand", 5 * UR))
        self.wait(3)


class TimePassing(InteractiveScene):
    def construct(self):
        clock = Clock()
        self.add(clock)
        self.play(ClockPassesTime(clock, hours_passed=2, run_time=10))


class DerivativeRule(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"t": BLUE, "s": YELLOW}
        kw = dict(t2c=t2c, font_size=96)
        in_texs = VGroup(
            Tex(R"x(t)", **kw),
            Tex(R"x'(t)", **kw),
        )
        out_texs = VGroup(
            Tex(R"X(s)", **kw),
            Tex(R"s X(s) - x(0)", **kw),
        )
        arrows = Vector(3 * RIGHT, thickness=7).replicate(2)

        in_texs.arrange(DOWN, buff=2)
        in_texs.to_edge(LEFT, buff=2)

        for in_tex, arrow, output in zip(in_texs, arrows, out_texs):
            arrow.next_to(in_tex, RIGHT)
            output.next_to(arrow, RIGHT)
            fancy_L = Tex(R"\\mathcal{L}")
            fancy_L.next_to(arrow, UP, SMALL_BUFF)
            arrow.add(fancy_L)

        self.add(in_texs[0], arrows[0], out_texs[0])
        self.wait()
        self.play(
            TransformMatchingTex(in_texs[0].copy(), in_texs[1]),
            TransformMatchingTex(out_texs[0].copy(), out_texs[1]),
            TransformFromCopy(*arrows),
            run_time=1.5
        )
        self.wait()
        SpeechBubble


class EndScreen(SideScrollEndScreen):
    pass`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports get_complex_graph from the _2025.laplace.integration module within the 3b1b videos codebase.",
      5: "WriteLaplace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      6: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      7: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      13: "TwoLevels extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      14: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      28: "TwoKeyIdeas extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      29: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      58: "LevelsOfUnderstanding extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      59: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      82: "DrivingACar extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      83: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      93: "Class WhatIsItTryingToDo2 inherits from TeacherStudentsScene.",
      94: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      125: "Class ReferenceComplexExponents inherits from TeacherStudentsScene.",
      126: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      157: "Class ReferenceWorkedExample inherits from TeacherStudentsScene.",
      158: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      180: "Class ButWhatIsIt inherits from TeacherStudentsScene.",
      181: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      202: "MoreInAMoment extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      203: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      212: "YouAsAMathematician extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      213: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      254: "FullCosInsideSum extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      255: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      260: "Class ReferenceTheIntegral inherits from TeacherStudentsScene.",
      261: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      536: "Class TryADifferentInterpretation inherits from TeacherStudentsScene.",
      537: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      548: "AskAboutZero extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      549: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      558: "Class SeemsDumbAndPointless inherits from TeacherStudentsScene.",
      559: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      596: "Class ReferenceAnalyticContinuation inherits from TeacherStudentsScene.",
      597: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      618: "PreviewAnalyticContinuation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      619: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      643: "SimpleRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      644: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      652: "ReactingToCosineMachine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      653: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      675: "PonderingCosineMachine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      676: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      687: "LaplaceFourierContrast extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      688: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      720: "NotWhatYouWouldSee extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      721: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      734: "CosLTLogicReversal extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      735: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      752: "CosineEqualsWhat extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      753: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      765: "Class OhLookAtTheTime inherits from TeacherStudentsScene.",
      766: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      786: "TimePassing extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      787: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      793: "DerivativeRule extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      794: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      830: "Class EndScreen inherits from SideScrollEndScreen.",
    }
  };

  files["_2025/laplace/prequel_equations.py"] = {
    description: "Equation scenes for the Laplace transform prequel, introducing the motivation and historical context for Laplace transforms.",
    code: `from manim_imports_ext import *


class IntroduceTrilogy(InteractiveScene):
    def construct(self):
        # Add definition
        frame = self.frame
        laplace = Tex(R"F(s) = \\int_0^\\infty f({t}) e^{-s{t}} dt", font_size=72, t2c={"s": YELLOW, R"{t}": WHITE})
        name = Text("Laplace Transform", font_size=72)
        name.next_to(laplace, UP, LARGE_BUFF)

        frame.move_to(laplace["f({t})"])
        self.play(
            Write(laplace["f({t})"]),
        )
        self.play(
            Write(laplace["e^{-s"]),
            TransformFromCopy(*laplace[R"{t}"][0:2]),
            frame.animate.move_to(laplace[R"f({t}) e^{-s{t}}"])
        )
        self.play(
            FadeIn(laplace[R"\\int_0^\\infty"], shift=0.25 * RIGHT, scale=1.5),
            FadeIn(laplace[R"dt"], shift=0.25 * LEFT, scale=1.5),
        )
        self.play(
            FadeTransform(laplace["f("].copy(), laplace["F("], path_arc=-PI / 2),
            TransformFromCopy(laplace[")"][1], laplace[")"][0], path_arc=-PI / 2),
            TransformFromCopy(laplace["s"][1], laplace["s"][0], path_arc=-PI / 4),
            Write(laplace["="]),
            frame.animate.center(),
        )
        self.play(Write(name))
        self.wait()

        # Show trilogy
        background = FullScreenRectangle().set_fill(GREY_E, 1)
        screens = ScreenRectangle().replicate(3)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        screens[1].save_state()
        screens[1].replace(background)
        screens[1].set_stroke(width=0)
        screens.set_stroke(behind=True)

        terms = VGroup(name, laplace)

        self.add(background, screens, terms)
        self.play(
            FadeIn(background),
            Restore(screens[1]),
            terms.animate.scale(0.4).move_to(screens[1].saved_state),
        )
        self.wait()

        # Inverse
        ilp = VGroup(
            Text("Inverse Laplace Transform"),
            Tex(R"f({t}) = \\frac{1}{2\\pi i} \\int_{a - i \\infty}^{a + i \\infty} F(s) e^{s{t}} ds", t2c={"s": YELLOW})
        )
        for mob1, mob2 in zip(ilp, terms):
            mob1.replace(mob2, dim_to_match=1)

        ilp.move_to(screens[2])

        self.play(
            TransformMatchingStrings(name.copy(), ilp[0], lag_ratio=1e-2, path_arc=-20 * DEG),
            TransformMatchingTex(
                laplace.copy(),
                ilp[1],
                lag_ratio=1e-2,
                path_arc=-20 * DEG,
                matched_keys=["f({t})", "F(s)", "e^{s{t}}", R"\\int"],
                key_map={"dt": "ds"},
            ),
        )
        self.wait()


class DiscussTrilogy(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        screens = ScreenRectangle().replicate(3)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        # Reference last two
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        morty.change_mode("tease")
        brace1 = Brace(screens[0], DOWN)
        brace2 = Brace(screens[1:3], DOWN)

        self.wait(2)
        self.play(
            morty.change("raise_left_hand", look_at=brace2),
            self.change_students("pondering", "erm", "sassy", look_at=brace2),
            GrowFromCenter(brace2),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("thinking", "pondering", "pondering", look_at=brace1),
            ReplacementTransform(brace2, brace1, path_arc=-30 * DEG),
        )
        self.wait(6)


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says("Who cares?", mode="angry", look_at=3 * UP),
            morty.change("guilty", stds[2].eyes),
            stds[1].change("hesitant", 3 * UP),
            stds[0].change("erm", stds[2].eyes),
        )
        self.wait(3)


class MiniLessonTitle(InteractiveScene):
    def construct(self):
        title = Text("Visualizing complex exponents", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class WeGotThis(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            self.change_students("coin_flip_2", "tease", "hooray", look_at=3 * UP),
            morty.change("tease")
        )
        self.wait()
        self.play(
            self.change_students("tease", "happy", "well", look_at=morty.eyes)
        )
        self.wait(3)


class ConfusionAndWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        stds = self.students

        q_marks = Tex(R"???").replicate(3)
        q_marks.space_out_submobjects(1.5)
        for mark, student in zip(q_marks, stds):
            mark.next_to(student, UP, MED_SMALL_BUFF)
        self.play(
            self.change_students("confused", "pondering", "pleading", look_at=self.screen),
            FadeIn(q_marks, 0.2 * UP, lag_ratio=0.05),
            morty.change("raise_right_hand")
        )
        self.wait(3)
        self.play(morty.change("raise_left_hand", look_at=3 * UR))
        self.play(
            self.change_students("erm", "thinking", "hesitant", look_at=morty.get_top() + 2 * UP),
            FadeOut(q_marks)
        )
        self.wait(4)
        self.play(self.change_students("pondering"))
        self.wait(3)


class ArrowBetweenScreens(InteractiveScene):
    def construct(self):
        # Test
        screens = ScreenRectangle().replicate(2)
        screens.arrange(RIGHT, buff=MED_LARGE_BUFF)
        screens.set_width(FRAME_WIDTH - 2)
        screens.move_to(DOWN)
        arrow = Arrow(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, thickness=6, buff=0.25)
        line = Line(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, stroke_width=8, buff=0.25)
        VGroup(arrow, line).set_color(TEAL)
        self.play(
            ShowCreation(line),
            FadeIn(arrow, time_span=(0.75, 1))
        )
        self.wait()



class WhatAndWhy(InteractiveScene):
    def construct(self):
        words = VGroup(
            Tex(R"\\text{1) Understanding } e^{i {t}} \\\\ \\text{ intuitively}", t2c={R"{t}": GREY_B}),
            TexText(R"2) How they \\\\ \\quad \\quad naturally arise"),
        )
        words[0][R"intuitively"].align_to(words[0]["Understanding"], LEFT)
        words[1][R"naturally arise"].align_to(words[1]["How"], LEFT)
        words.refresh_bounding_box()
        words.scale(1.25)
        self.add(words)
        words.arrange(DOWN, aligned_edge=LEFT, buff=2.5)
        words.next_to(ORIGIN, RIGHT)
        words.set_opacity(0)
        for word, u in zip(words, [1, -1]):
            word.set_opacity(1)
            self.play(Write(word))
            self.wait()
        # Test


class PrequelToLaplace(InteractiveScene):
    def construct(self):
        # False goal of motivating the i
        pass

        # Swap out i and π for s and t


class OtherExponentialDerivatives(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"t": GREY_B})
        derivs = VGroup(
            Tex(R"\\frac{d}{dt} 2^t = (0.693...)2^t", **kw),
            Tex(R"\\frac{d}{dt} 3^t = (1.098...)3^t", **kw),
            Tex(R"\\frac{d}{dt} 4^t = (1.386...)4^t", **kw),
            Tex(R"\\frac{d}{dt} 5^t = (1.609...)5^t", **kw),
            Tex(R"\\frac{d}{dt} 6^t = (1.791...)6^t", **kw),
        )
        derivs.scale(0.75)
        derivs.arrange(DOWN, buff=0.7)
        derivs.to_corner(UL)

        self.play(LaggedStartMap(FadeIn, derivs, shift=UP, lag_ratio=0.5, run_time=5))
        self.wait()


class VariousExponentials(InteractiveScene):
    def construct(self):
        # Test
        exp_st = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=90)
        gen_exp = Tex(R"e^{+0.50 t}", t2c={"+0.50": YELLOW, "t": BLUE}, font_size=90)
        exp_st.to_edge(UP, buff=MED_LARGE_BUFF)
        gen_exp.move_to(exp_st)

        num = gen_exp["+0.50"]
        num.set_opacity(0)
        gen_exp["t"].scale(1.25, about_edge=UL)

        s_num = DecimalNumber(-1.00, edge_to_fix=ORIGIN, include_sign=True)
        s_num.set_color(YELLOW)
        s_num.replace(num, dim_to_match=1)

        self.add(gen_exp, s_num)
        self.play(ChangeDecimalToValue(s_num, 0.5, run_time=4))
        self.wait()
        self.play(LaggedStart(
            ReplacementTransform(gen_exp["e"][0], exp_st["e"][0]),
            ReplacementTransform(s_num, exp_st["s"]),
            ReplacementTransform(gen_exp["t"][0], exp_st["t"][0]),
        ))
        self.wait()


class WhyToWhat(InteractiveScene):
    def construct(self):
        # Title text
        why = Text("Why", font_size=90)
        what = Text("Wait, what does this even mean?", font_size=72)
        VGroup(why, what).to_edge(UP)

        what_word = what["what"][0].copy()
        what["what"][0].set_opacity(0)

        arrow = Arrow(
            what["this"].get_bottom(),
            (2.5, 2, 0),
            thickness=5,
            fill_color=YELLOW
        )

        self.play(FadeIn(why, UP))
        self.wait()
        self.play(
            # FadeOut(why, UP),
            ReplacementTransform(why, what_word),
            FadeIn(what, lag_ratio=0.1),
        )
        self.play(
            GrowArrow(arrow),
            what["this"].animate.set_color(YELLOW)
        )
        self.wait()


class DerivativeOfExp(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        tex_kw = dict(t2c={"t": GREY_B, "s": YELLOW})
        equation = Tex(R"\\frac{d}{dt} e^{st} = s \\cdot e^{st}", font_size=90, **tex_kw)
        deriv_part = equation[R"\\frac{d}{dt}"][0]
        exp_parts = equation[R"e^{st}"]
        equals = equation[R"="][0]
        s_dot = equation[R"s \\cdot"][0]

        v_box = SurroundingRectangle(VGroup(deriv_part, exp_parts[0]))
        p_box = SurroundingRectangle(exp_parts[1])
        s_box = SurroundingRectangle(s_dot)
        s_box.match_height(p_box, stretch=True).match_y(p_box)
        boxes = VGroup(v_box, p_box, s_box)
        boxes.set_stroke(width=2)
        boxes.set_submobject_colors_by_gradient(GREEN, BLUE, YELLOW)

        v_label = Text("Velocity", font_size=48).match_color(v_box)
        p_label = Text("Position", font_size=48).match_color(p_box)
        s_label = Text("Modifier", font_size=48).match_color(s_box)
        v_label.next_to(v_box, UP, MED_SMALL_BUFF)
        p_label.next_to(p_box, UP, MED_SMALL_BUFF, aligned_edge=LEFT)
        s_label.next_to(s_box, DOWN, MED_SMALL_BUFF)
        labels = VGroup(v_label, p_label, s_label)

        frame.move_to(exp_parts[0])

        self.add(exp_parts[0])
        self.wait()
        self.play(Write(deriv_part))
        self.play(
            TransformFromCopy(*exp_parts, path_arc=90 * DEG),
            Write(equals),
            frame.animate.center(),
        )
        self.play(
            TransformFromCopy(exp_parts[1][1], s_dot[0], path_arc=90 * DEG),
            Write(s_dot[1]),
        )
        self.wait()

        # Show labels
        for box, label in zip(boxes, labels):
            self.play(ShowCreation(box), FadeIn(label))

        self.wait()
        full_group = VGroup(equation, boxes, labels)

        # Set s equal to 1
        s_eq_1 = Tex(R"s = 1", font_size=72, **tex_kw)
        simple_equation = Tex(R"\\frac{d}{dt} e^{t} = e^{t}", font_size=72, **tex_kw)
        simple_equation.to_edge(UP).shift(2 * LEFT)
        s_eq_1.next_to(simple_equation, RIGHT, buff=2.5)
        arrow = Arrow(s_eq_1, simple_equation, thickness=5, buff=0.35).shift(0.05 * DOWN)

        self.play(
            Write(s_eq_1),
            GrowArrow(arrow),
            TransformMatchingTex(equation.copy(), simple_equation, run_time=1.5, lag_ratio=0.02),
            full_group.animate.shift(DOWN).scale(0.75).fade(0.15)
        )
        self.wait()


class HighlightRect(InteractiveScene):
    def construct(self):
        img = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/exponentials/DynamicExpIntuitionStill.png')
        img.set_height(FRAME_HEIGHT)
        self.add(img)

        # Rects
        rects = VGroup(
            Rectangle(2.25, 1).move_to((2.18, 2.74, 0)),
            Rectangle(2, 0.85).move_to((-5.88, -2.2, 0.0)),
        )
        rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rects[0]))
        self.play(TransformFromCopy(*rects))
        self.play(FadeOut(rects))


class DefineI(InteractiveScene):
    def construct(self):
        eq = Tex(R"i = \\sqrt{-1}", t2c={"i": YELLOW}, font_size=90)
        self.play(Write(eq))
        self.wait()


class WaitWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].change("erm", self.screen),
            self.students[1].change("tease", self.screen),
            self.students[2].says("Wait, why?", "confused", look_at=self.screen, bubble_direction=LEFT),
        )
        self.wait(4)


class MultiplicationByI(InteractiveScene):
    def construct(self):
        # Example number
        plane = ComplexPlane(
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            # faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.add_coordinate_labels(font_size=24)

        z = 3 + 2j
        tex_kw = dict(t2c={"a": YELLOW, "b": PINK})

        vect = Vector(plane.n2p(z), fill_color=WHITE, thickness=4)
        vect_label = Tex(R"a + bi", **tex_kw)
        vect_label.next_to(vect.get_end(), UR, SMALL_BUFF)
        vect_label.set_backstroke(BLACK, 5)

        lines = VGroup(
            Line(ORIGIN, plane.n2p(z.real)).set_color(YELLOW),
            Line(plane.n2p(z.real), plane.n2p(z)).set_color(PINK),
        )
        a_label, b_label = line_labels = VGroup(
            Tex(R"a", font_size=36, **tex_kw).next_to(lines[0], UP, SMALL_BUFF),
            Tex(R"bi", font_size=36, **tex_kw).next_to(lines[1], RIGHT, SMALL_BUFF),
        )
        line_labels.set_backstroke(BLACK, 5)

        self.add(plane, Point(), plane.coordinate_labels)
        self.add(vect)
        self.add(vect_label)
        for line, label in zip(lines, line_labels):
            self.play(
                ShowCreation(line),
                FadeIn(label, 0.25 * line.get_vector())
            )
        self.wait()

        # Multiply components by i
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG, about_point=ORIGIN)
        new_lines[1].move_to(ORIGIN, RIGHT)

        new_a_label = Tex(R"ai", font_size=36, **tex_kw).next_to(new_lines[0], RIGHT, SMALL_BUFF)
        new_b_label = Tex(R"bi \\cdot i", font_size=36, **tex_kw).next_to(new_lines[1], UP, SMALL_BUFF)
        neg_b_label = Tex(R"=-b", font_size=36, **tex_kw)
        neg_b_label.move_to(new_b_label.get_right())

        mult_i_label = Tex(R"\\times i", font_size=90)
        mult_i_label.set_backstroke(BLACK, 5)
        mult_i_label.to_corner(UR, buff=MED_LARGE_BUFF).shift(0.2 * UP)

        self.play(Write(mult_i_label))
        self.wait()
        self.play(
            TransformFromCopy(lines[0], new_lines[0], path_arc=90 * DEG),
            TransformFromCopy(a_label[0], new_a_label[0], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_a_label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(lines[1], new_lines[1], path_arc=90 * DEG),
            TransformFromCopy(b_label[0], new_b_label[:-1], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_b_label[-1]),
        )
        self.wait()
        self.play(
            FlashAround(VGroup(new_b_label, new_lines[1]), color=PINK, time_width=1.5, run_time=2),
            new_b_label.animate.next_to(neg_b_label, LEFT, SMALL_BUFF),
            FadeIn(neg_b_label, SMALL_BUFF * RIGHT),
        )
        self.wait()
        self.play(VGroup(new_lines[1], new_b_label, neg_b_label).animate.shift(new_lines[0].get_vector()))

        # New vector
        vect_copy = vect.copy()
        elbow = Elbow().rotate(vect.get_angle(), about_point=ORIGIN)
        self.play(
            Rotate(vect_copy, 90 * DEG, run_time=2, about_point=ORIGIN),
        )
        self.play(
            ShowCreation(elbow)
        )
        self.wait()

    def old_material(self):
        # Show the algebra
        algebra = VGroup(
            Tex(R"i \\cdot (a + bi)", **tex_kw),
            Tex(R"ai + bi^2", **tex_kw),
            Tex(R"-b + ai", **tex_kw),
        )
        algebra.set_backstroke(BLACK, 8)
        algebra.arrange(DOWN, buff=0.35)
        algebra.to_corner(UL)

        self.play(
            TransformFromCopy(vect_label, algebra[0]["a + bi"][0]),
            FadeIn(algebra[0]),
        )
        self.play(LaggedStart(
            TransformFromCopy(algebra[0]["a"], algebra[1]["a"]),
            TransformFromCopy(algebra[0]["+ bi"], algebra[1]["+ bi"]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["i"][0]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["2"]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(algebra[1]["bi^2"], algebra[2]["-b"]),
            TransformFromCopy(algebra[1]["ai"], algebra[2]["ai"]),
            TransformFromCopy(algebra[1]["+"], algebra[2]["+"]),
            lag_ratio=0.25
        ))
        self.wait()

        # New lines
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG)
        new_lines.refresh_bounding_box()
        new_lines[1].move_to(ORIGIN, RIGHT)
        new_lines[0].move_to(new_lines[1].get_left(), DOWN)

        neg_b_label = Tex(R"-b", fill_color=PINK, font_size=36).next_to(new_lines[1], UP, SMALL_BUFF)
        new_a_label = Tex(R"a", fill_color=YELLOW, font_size=36).next_to(new_lines[0], LEFT, SMALL_BUFF)

        self.play(
            TransformFromCopy(lines[1], new_lines[1]),
            FadeTransform(algebra[2]["-b"].copy(), neg_b_label),
        )
        self.play(
            TransformFromCopy(lines[0], new_lines[0]),
            FadeTransform(algebra[2]["a"].copy(), new_a_label),
        )
        self.wait()


class UnitArcLengthsOnCircle(InteractiveScene):
    def construct(self):
        # Moving sectors
        arc = Arc(0, 1, radius=2.5, stroke_color=GREEN, stroke_width=8)
        sector = Sector(angle=1, radius=2.5).set_fill(GREEN, 0.25)
        v_line = Line(ORIGIN, 2.5 * UP)
        v_line.match_style(arc)
        v_line.move_to(arc.get_start(), DOWN)

        self.add(v_line)
        self.play(
            FadeIn(sector),
            ReplacementTransform(v_line, arc),
        )

        group = VGroup(sector, arc)
        self.add(group)

        for n in range(5):
            self.wait(2)
            group.rotate(1, about_point=ORIGIN)

        return

        # Previous
        colors = [RED, BLUE]
        arcs = VGroup(
            Arc(n, 1, radius=2.5, stroke_color=colors[n % 2], stroke_width=8)
            for n in range(6)
        )
        for arc in arcs:
            one = Integer(1, font_size=24).move_to(1.0 * arc.get_center())
            self.play(ShowCreation(arc, rate_func=linear, run_time=2))
        self.wait()


class SimpleIndicationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(3, 2)
        # Test
        self.play(FlashAround(rect, time_width=2.0, run_time=2, color=WHITE))


class WriteSPlane(InteractiveScene):
    def construct(self):
        title = Text("S-plane", font_size=72)
        title.set_color(YELLOW)
        self.play(Write(title))
        self.wait()


class ODEStoExp(InteractiveScene):
    def construct(self):
        # Test
        odes, exp = words = VGroup(
            Text("Differential\\nEquations"),
            Tex("e^{st}", t2c={"s": YELLOW}, font_size=72),
        )
        exp.match_height(odes)
        words.arrange(RIGHT, buff=3.0)
        words.to_edge(UP, buff=1.25)

        top_arrow, low_arrow = arrows = VGroup(
            Arrow(odes.get_corner(UR), exp.get_corner(UL), path_arc=-60 * DEG, thickness=5),
            Arrow(exp.get_corner(DL), odes.get_corner(DR), path_arc=-60 * DEG, thickness=5),
        )
        arrows.set_fill(TEAL)

        top_words = Tex(R"Explain", font_size=36).next_to(top_arrow, UP, SMALL_BUFF)
        low_words = Tex(R"Solves", font_size=36).next_to(low_arrow, DOWN, SMALL_BUFF)

        exp.shift(0.25 * UP + 0.05 * LEFT)

        self.add(words)
        self.wait()
        self.play(
            Write(top_arrow),
            Write(top_words),
        )
        self.wait()
        self.play(
            # Write(low_arrow),
            TransformFromCopy(top_arrow, low_arrow, path_arc=-PI),
            Write(low_words),
        )
        self.wait()


class GenLinearEquationToOscillator(InteractiveScene):
    def construct(self):
        # General equation
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = dict()
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \\cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c, font_size=60)
        ode.move_to(DOWN)
        ode_2nd = ode["a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0"]

        self.play(Write(ode))
        self.wait()
        self.play(
            FadeOut(ode[R"a_n x^{n}(t) + \\cdots + "]),
            ode_2nd.animate.move_to(UP),
            self.frame.animate.set_height(7)
        )

        # Transition
        alt_consts = VGroup(Tex(R"m"), Tex(R"\\mu"), Tex(R"k"))
        alt_consts.scale(60 / 48)
        a_parts = VGroup(ode[tex][0] for tex in a_texs[1:])
        for const, a_part in zip(alt_consts, a_parts):
            const.move_to(a_part, RIGHT)
            const.align_to(ode[-1], DOWN)
            if const is alt_consts[1]:
                const.shift(0.1 * DOWN)
            self.play(
                FadeOut(a_part, 0.25 * UP),
                FadeIn(const, 0.25 * UP),
            )
        self.wait()


class VLineOverZero(InteractiveScene):
    def construct(self):
        # Test
        rect = Square(0.25)
        rect.move_to(2.5 * DOWN)
        v_line = Line(rect.get_top(), 4 * UP, buff=0.1)
        v_line.set_stroke(YELLOW, 2)
        rect.match_style(v_line)

        self.play(
            ShowCreationThenFadeOut(rect),
            ShowCreationThenFadeOut(v_line),
        )
        self.wait()


class KIsSomeConstant(InteractiveScene):
    def construct(self):
        rect = SurroundingRectangle(Text("k"), buff=0.05)
        rect.set_stroke(YELLOW, 2)
        words = Text("Some constant", font_size=24)
        words.next_to(rect, UP, SMALL_BUFF)
        words.match_color(rect)

        self.play(ShowCreation(rect), FadeIn(words))
        self.wait()


class WriteMu(InteractiveScene):
    def construct(self):
        sym = Tex(R"\\mu")
        rect = SurroundingRectangle(sym, buff=0.05)
        rect.set_stroke(YELLOW, 2)
        mu = TexText("\`\`Mu''")
        mu.set_color(YELLOW)
        mu.next_to(rect, DOWN)
        self.play(
            Write(mu),
            ShowCreation(rect)
        )
        self.wait()


class ReferenceGuessingExp(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Student asks
        question = Tex(R"x(t) = ???")
        lhs = question["x(t)"][0]
        rhs = question["= ???"][0]
        bubble = stds[2].get_bubble(question, bubble_type=SpeechBubble, direction=LEFT)
        lhs.save_state()
        lhs.scale(0.25).move_to([-6.24, 2.38, 0])

        self.play(
            morty.change("hesitant", look_at=stds[2].eyes),
            self.change_students("erm", "confused", "maybe", look_at=self.screen)
        )
        self.wait()
        self.play(
            stds[2].change("raise_left_hand", morty.eyes),
            Write(bubble[0]),
            Write(rhs, time_span=(0.5, 1.0)),
            Restore(lhs),
        )
        self.wait()
        self.add(Point())
        self.play(
            morty.says("Here's a trick:", mode="tease", bubble_creation_class=FadeIn),
            self.change_students("pondering", "thinking", "hesitant", look_at=UL),
        )
        self.wait(2)

        # Teacher gestures to upper right, students look confused and hesitant
        eq_point = 5 * RIGHT + 3 * UP
        self.play(
            morty.change("raise_right_hand", look_at=eq_point),
            FadeOut(bubble),
            FadeOut(morty.bubble),
            self.change_students("confused", "thinking", "hesitant", look_at=eq_point),
        )
        self.wait()
        self.play(self.change_students("confused", "hesitant", "confused", look_at=eq_point, lag_ratio=0.1))
        self.wait()
        self.play(
            morty.change("shruggie", look_at=eq_point),
        )
        self.wait(2)
        self.play(
            self.change_students("angry", "hesitant", "erm", look_at=morty.eyes),
            morty.animate.look_at(stds)
        )
        self.wait(2)

        # Transition: flip and reposition morty to where stds are
        new_teacher_pos = stds[2].get_bottom()
        new_teacher = morty.copy()
        new_teacher.change_mode("raise_left_hand")
        new_teacher.look_at(3 * UR)
        new_teacher.body.set_color(GREY_C)

        self.play(
            morty.animate.scale(0.8).flip().change_mode("confused").look_at(5 * UR).move_to(new_teacher_pos, DOWN),
            LaggedStartMap(FadeOut, stds, shift=DOWN, lag_ratio=0.2, run_time=1),
            FadeIn(new_teacher, time_span=(0.5, 1.5)),
        )
        self.play(morty.change("pleading", 3 * UR))
        self.play(Blink(new_teacher))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("erm", new_teacher.eyes),
            new_teacher.change("guilty", look_at=morty.eyes),
            lag_ratio=0.5,
        ))
        self.wait(3)

        # Reference a graph
        self.play(
            morty.change("angry", 2 * UR),
            new_teacher.change("tease", 2 * UR)
        )
        self.play(Blink(morty))
        self.play(Blink(new_teacher))
        self.wait()


class FromGuessToLaplace(InteractiveScene):
    def construct(self):
        # Words
        strategy = VGroup(
            Text("“Strategy”", fill_color=GREY_A, font_size=72),
            TexText("Guess $x(t) = e^{{s}t}$", t2c={"{s}": YELLOW}, fill_color=WHITE, font_size=72),
        )
        strategy.arrange(DOWN)
        self.add(strategy)
        return

        # Comment on it
        exp_rect = SurroundingRectangle(strategy[1]["x(t) = e^{{s}t}"], buff=SMALL_BUFF)
        exp_words = Text("Why?", font_size=42)
        exp_words.next_to(exp_rect, RIGHT, SMALL_BUFF)
        VGroup(exp_rect, exp_words).set_color(PINK)

        guess_rect = SurroundingRectangle(strategy[1]["Guess"], buff=SMALL_BUFF)
        guess_rect.match_height(exp_rect, stretch=True).match_y(exp_rect)
        guess_words = Text("Seems dumb", font_size=36)
        guess_words.next_to(guess_rect, DOWN, SMALL_BUFF)
        VGroup(guess_rect, guess_words).set_color(RED)

        self.play(LaggedStart(
            ShowCreation(guess_rect),
            FadeIn(guess_words, lag_ratio=0.1),
            ShowCreation(exp_rect),
            FadeIn(exp_words, lag_ratio=0.1),
            lag_ratio=0.25
        ))
        self.wait()

        # Transition to Laplace
        laplace = Tex(R"\\int_0^\\infty x(t) e^{-{s}t} dt", t2c={"{s}": YELLOW}, font_size=72)
        laplace.move_to(strategy[1])

        self.play(LaggedStart(
            LaggedStartMap(FadeOut, VGroup(strategy[1]["Guess"], guess_rect, guess_words), shift=DOWN, lag_ratio=0.1),
            TransformFromCopy(strategy[0]["S"][0], laplace[R"\\int"][0]),
            TransformFromCopy(strategy[0]["e"][0], laplace[R"0"][0]),
            TransformFromCopy(strategy[0]["g"][0], laplace[R"\\infty"][0]),
            FadeOut(strategy[0], lag_ratio=0.1),
            # Break
            FadeOut(VGroup(exp_rect, exp_words), 0.5 * LEFT, lag_ratio=0.1),
            FadeTransform(strategy[1]["x(t)"][0], laplace["x(t)"][0]),
            FadeTransform(strategy[1]["="][0], laplace["-"][0]),
            FadeTransform(strategy[1]["e"][-1], laplace["e"][0]),
            FadeTransform(strategy[1]["{s}t"][0], laplace["{s}t"][0]),
            Write(laplace["dt"][0]),
            lag_ratio=0.15,
            run_time=3,
        ))
        self.wait()

        # Label laplace
        laplace_rect = SurroundingRectangle(laplace)
        laplace_rect.set_color(BLUE)
        laplace_label = Text("Laplace Transform", font_size=72)
        laplace_label.next_to(laplace_rect, UP)
        laplace_label.match_color(laplace_rect)

        self.play(
            Write(laplace_label),
            ShowCreation(laplace_rect),
        )
        self.wait()


class JustAlgebra(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="tease")
        morty.body.insert_n_curves(100)
        self.play(morty.says("Just algebra!", mode="hooray", look_at=2 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeOut(morty.bubble),
            morty.change("tease", look_at=2 * UL + UP)
        )
        self.play(Blink(morty))
        self.wait()


class BothPositiveNumbers(InteractiveScene):
    def construct(self):
        tex = Tex("k / m")
        self.add(tex)

        # Test
        rects = VGroup(SurroundingRectangle(tex[c], buff=0.05) for c in "km")
        rects.set_stroke(GREEN, 3)
        plusses = VGroup(Tex(R"+").next_to(rect, DOWN, SMALL_BUFF) for rect in rects)
        plusses.set_fill(GREEN)

        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.5),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * DOWN, lag_ratio=0.5)
        )
        self.wait()


class ButSpringsAreReal(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("maybe", self.screen),
            stds[1].says("But...springs are real", mode="confused", look_at=self.screen),
            stds[2].change("erm", self.screen),
            morty.change("tease", stds[2].eyes)
        )
        self.wait(4)


class ShowIncreaseToK(InteractiveScene):
    def construct(self):
        # Test
        k = Tex(R"k")

        box = SurroundingRectangle(k)
        box.set_stroke(GREEN, 5)
        arrow = Vector(UP, thickness=6)
        arrow.set_fill(GREEN)
        center = box.get_center()

        self.play(
            ShowCreation(box),
            UpdateFromAlphaFunc(
                arrow, lambda m, a: m.move_to(
                    center + interpolate(-1, 1, a) * UP
                ).set_fill(
                    opacity=there_and_back(a) * 0.7
                ),
                run_time=4
            ),
        )
        self.wait()


class PureMathEquation(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"x''(t)": RED, "x(t)": TEAL, R"\\omega": PINK}
        physics_eq = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c, font_size=72)
        math_eq = Tex(R"a_2 x''(t) + a_0 x(t) = 0", t2c=t2c, font_size=72)

        self.add(physics_eq)
        self.play(LaggedStart(
            *(
                ReplacementTransform(physics_eq[tex][0], math_eq[tex][0])
                for tex in ["x''(t) +", "x(t) = 0"]
            ),
            FadeOut(physics_eq["m"], 0.5 * UP),
            FadeIn(math_eq["a_2"], 0.5 * UP),
            FadeOut(physics_eq["k"], 0.5 * UP),
            FadeIn(math_eq["a_0"], 0.5 * UP),
            run_time=2,
            lag_ratio=0.15
        ))
        self.wait()

        # Show solution
        implies = Tex(R"\\Downarrow", font_size=72)
        answer = Tex(R"e^{\\pm i\\omega t}", font_size=90, t2c=t2c)
        answer.next_to(implies, DOWN, MED_LARGE_BUFF)
        omega_eq = Tex(R"\\text{Where } \\omega = \\sqrt{a_2 / a_0}", t2c=t2c)
        omega_eq.next_to(answer, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            math_eq.animate.next_to(implies, UP, MED_LARGE_BUFF),
            Write(implies),
            FadeIn(answer, DOWN),
            lag_ratio=0.25
        ))
        self.play(FadeIn(omega_eq))
        self.wait()


class LinearityDefinition(InteractiveScene):
    def construct(self):
        # Base differential equation string
        eq_str = R"m x''(t) + k x(t) = 0"
        t2c = {"x_1": TEAL, "x_2": RED, "0.0": YELLOW, "2.0": YELLOW}

        base_eq = Tex(eq_str)
        base_eq.to_edge(UP)

        eq1, eq2, eq3, eq4 = equations = VGroup(
            Tex(eq_str.replace("x", "x_1"), t2c=t2c),
            Tex(eq_str.replace("x", "x_2"), t2c=t2c),
            Tex(R"m\\Big(x_1''(t) + x_2''(t) \\Big) + k \\Big(x_1(t) + x_2(t)\\Big) = 0", t2c=t2c),
            Tex(R"m\\Big(0.0 x_1''(t) + 2.0 x_2''(t) \\Big) + k \\Big(0.0 x_1(t) + 2.0 x_2(t)\\Big) = 0", t2c=t2c),
        )
        for eq in equations:
            eq.set_max_width(7)
        equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        equations.to_edge(RIGHT)
        equations.shift(DOWN)

        phrase1, phrase2, phrase3, phrase4 = phrases = VGroup(
            TexText("If $x_1$ solves it:", t2c=t2c),
            TexText("and $x_2$ solves it:", t2c=t2c),
            TexText("Then $(x_1 + x_2)$ solves it:", t2c=t2c),
            TexText("Then $(0.0 x_1 + 2.0 x_2)$ solves it:", t2c=t2c),
        )

        for phrase, eq in zip(phrases, equations):
            phrase.set_max_width(5)
            phrase.next_to(eq, LEFT, LARGE_BUFF)

        eq4.move_to(eq3)
        phrase4.move_to(phrase3)

        kw = dict(edge_to_fix=RIGHT)
        c1_terms = VGroup(phrase4.make_number_changeable("0.0", **kw), *eq4.make_number_changeable("0.0", replace_all=True, **kw))
        c2_terms = VGroup(phrase4.make_number_changeable("2.0", **kw), *eq4.make_number_changeable("2.0", replace_all=True, **kw))

        # Show base equation
        self.play(Write(phrase1), FadeIn(eq1))
        self.wait()
        self.play(
            TransformMatchingTex(eq1.copy(), eq2, key_map={"x_1": "x_2"}, run_time=1, lag_ratio=0.01),
            FadeTransform(phrase1.copy(), phrase2)
        )
        self.wait()
        self.play(
            FadeIn(phrase3, DOWN),
            FadeIn(eq3, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(eq3, 0.5 * DOWN),
            FadeOut(phrase3, 0.5 * DOWN),
            FadeIn(eq4, 0.5 * DOWN),
            FadeIn(phrase4, 0.5 * DOWN),
        )
        for _ in range(8):
            new_c1 = random.random() * 10
            new_c2 = random.random() * 10
            self.play(*(
                ChangeDecimalToValue(c1, new_c1, run_time=1)
                for c1 in c1_terms
            ))
            self.wait(0.5)
            self.play(*(
                ChangeDecimalToValue(c2, new_c2, run_time=1)
                for c2 in c2_terms
            ))
            self.wait(0.5)


class ComplainAboutNeelessComplexity(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Complain
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].says("That’s needlessly\\ncomplicated!", mode="angry", look_at=self.screen),
            stds[2].change("maybe", self.screen),
            morty.change("guilty"),
        )
        self.wait(3)
        self.play(
            stds[0].change("erm", self.screen),
            stds[1].debubble(mode="raise_left_hand", look_at=self.screen),
            stds[2].change("sassy", self.screen),
            morty.change("tease"),
        )
        self.wait()
        self.play(
            stds[1].change("raise_right_hand", ORIGIN),
            stds[0].change("pondering", ORIGIN),
            stds[2].change("pondering", ORIGIN),
        )
        self.wait(5)


class LetsGeneralize(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            morty.says("Let’s\\n  generalize!", mode="hooray")
        )
        self.play(Blink(morty))
        self.wait(3)


class EquationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(5.25, 1)
        rect.set_stroke(YELLOW, 3)

        # Test
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.stretch(0.5, 0).shift(4 * RIGHT).set_opacity(0))
        self.wait()


class GeneralLinearEquation(InteractiveScene):
    def construct(self):
        # Set up equations
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = {"{s}": YELLOW}
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \\cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c)
        exp_version = Tex(
            R"a_n \\left({s}^n e^{{s}t}\\right) "
            R"+ \\cdots "
            R"+ a_2 \\left({s}^2 e^{{s}t}\\right) "
            R"+ a_1 \\left({s}e^{{s}t}\\right) "
            R"+ a_0 e^{{s}t} = 0",
            t2c=t2c
        )
        factored = Tex(R"e^{{s}t} \\left(a_n {s}^n + \\cdots + a_2 {s}^2 + a_1 {s} + a_0 \\right) = 0", t2c=t2c)

        ode.to_edge(UP)
        exp_version.next_to(ode, DOWN, MED_LARGE_BUFF)
        factored.move_to(exp_version)

        # Introduce ode
        index = ode.submobjects.index(ode["a_2"][0][0])

        right_part = ode[index:]
        left_part = ode[:index]
        right_part.save_state()
        right_part.set_x(0)

        self.play(FadeIn(right_part, UP))
        self.wait()
        self.play(LaggedStart(
            Restore(right_part),
            Write(left_part)
        ))
        self.add(ode)

        # Highlight equation parts
        x_arrows = VGroup(
            Arrow(UP, ode[x_tex].get_bottom(), fill_color=color)
            for x_tex, color in zip(x_texs, x_colors)
        )
        x_arrows.reverse_submobjects()

        x_rects = VGroup(SurroundingRectangle(ode[x_tex], buff=SMALL_BUFF) for x_tex in x_texs)
        a_rects = VGroup(SurroundingRectangle(ode[a_tex]) for a_tex in a_texs)
        full_rect = SurroundingRectangle(ode[:-2])
        zero_rect = SurroundingRectangle(ode[-2:])
        VGroup(x_rects, a_rects, full_rect, zero_rect).set_stroke(YELLOW, 2)

        self.play(LaggedStartMap(ShowCreation, x_rects))
        self.wait()
        self.play(ReplacementTransform(x_rects, a_rects, lag_ratio=0.2))
        self.wait()
        self.play(ReplacementTransform(a_rects, VGroup(full_rect)))
        self.wait()
        self.play(ReplacementTransform(full_rect, zero_rect))
        self.wait()
        self.play(FadeOut(zero_rect))

        # Plug in e^{st}
        key_map = {
            R"+ a_0 x(t) = 0": R"+ a_0 e^{{s}t} = 0",
            R"+ a_1 x'(t)": R"+ a_1 \\left({s}e^{{s}t}\\right)",
            R"+ a_2 x''(t)": R"+ a_2 \\left({s}^2 e^{{s}t}\\right)",
            R"+ \\cdots": R"+ \\cdots",
            R"a_n x^{n}(t)": R"a_n \\left({s}^n e^{{s}t}\\right)",
        }

        self.play(LaggedStart(*(
            FadeTransform(ode[k1].copy(), exp_version[k2])
            for k1, k2 in key_map.items()
        ), lag_ratio=0.6, run_time=4))
        self.wait()
        self.play(
            TransformMatchingTex(
                exp_version,
                factored,
                matched_keys=[R"e^{{s}t}", "{s}^n", "{s}^2", "{s}", "a_n", "a_2", "a_1", "a_0"],
                path_arc=45 * DEG
            )
        )
        self.wait()

        # Highlight the polynomail
        poly_rect = SurroundingRectangle(factored[R"a_n {s}^n + \\cdots + a_2 {s}^2 + a_1 {s} + a_0"])
        poly_rect.set_stroke(YELLOW, 1)

        self.play(
            ShowCreation(poly_rect),
            FadeOut(factored["e^{{s}t}"]),
            FadeOut(factored[R"\\left("]),
            FadeOut(factored[R"\\right)"]),
        )

        # Show factored expression
        linear_term_texs = [
            R"({s} - s_1)",
            R"({s} - s_2)",
            R"({s} - s_3)",
            R"\\cdots",
            R"({s} - s_n)",
        ]
        fully_factored = Tex(
            R"a_n" + " ".join(linear_term_texs),
            t2c=t2c,
            font_size=42,
            isolate=linear_term_texs
        )
        fully_factored.next_to(poly_rect, DOWN)
        linear_terms = VGroup(
            fully_factored[tex][0]
            for tex in linear_term_texs
        )

        self.play(
            Transform(factored["{s}"][1].copy().replicate(4), fully_factored["{s}"].copy(), remover=True),
            FadeIn(fully_factored, time_span=(0.25, 1)),
        )
        self.wait()

        # Plane
        plane = ComplexPlane((-3, 3), (-3, 3), width=6, height=6)
        plane.set_height(4.5)
        plane.next_to(poly_rect, DOWN, LARGE_BUFF)
        plane.set_x(0)
        plane.add_coordinate_labels(font_size=16)
        c_label = Tex(R"\\mathds{C}", font_size=90, fill_color=BLUE)
        c_label.next_to(plane, LEFT, aligned_edge=UP).shift(0.5 * DOWN)

        self.play(
            Write(plane, run_time=1, lag_ratio=2e-2),
            Write(c_label),
        )

        # Show some random root collections
        for n in range(4):
            roots = []
            n_roots = random.randint(3, 7)
            for _ in range(n_roots):
                root = complex(random.uniform(-3, 3), random.uniform(-3, 3))
                if random.random() < 0.25:
                    roots.append(root.real)
                else:
                    roots.extend([root, root.conjugate()])
            dots = Group(GlowDot(plane.n2p(z)) for z in roots)

            self.play(ShowIncreasingSubsets(dots))
            self.play(FadeOut(dots))

        # Turn linear terms into
        roots = [0.2 + 1j, 0.2 - 1j, -0.5 + 3j, -0.5 - 3j, -2]
        root_dots = Group(GlowDot(plane.n2p(root)) for root in roots)

        root_labels = VGroup(
            Tex(Rf"s_{{{n + 1}}}", font_size=36).next_to(dot.get_center(), UR, SMALL_BUFF)
            for n, dot in enumerate(root_dots)
        )
        root_labels.set_color(YELLOW)

        root_intro_kw = dict(lag_ratio=0.3, run_time=4)
        self.play(
            LaggedStart(*(
                FadeTransform(term, dot)
                for term, dot in zip(linear_terms, root_dots)
            ), **root_intro_kw),
            LaggedStart(*(
                TransformFromCopy(term[3:5], label)
                for term, label in zip(linear_terms, root_labels)
            ), **root_intro_kw),
            FadeOut(fully_factored["a_n"][0]),
        )
        self.wait()

        # Show the solutions
        frame = self.frame
        axes = VGroup(
            Axes((0, 10), (-y_max, y_max), width=5, height=1.25)
            for root in roots
            for y_max in [3 if root.real > 0 else 1]
        )
        axes.arrange(DOWN, buff=0.75)
        axes.next_to(plane, RIGHT, buff=6)

        c_trackers = Group(ComplexValueTracker(1) for root in roots)
        graphs = VGroup(
            self.get_graph(axes, root, c_tracker.get_value)
            for axes, root, c_tracker in zip(axes, roots, c_trackers)
        )

        axes_labels = VGroup(
            Tex(Rf"e^{{s_{{{n + 1}}} t}}", font_size=60)
            for n in range(len(axes))
        )
        for label, ax in zip(axes_labels, axes):
            label.next_to(ax, LEFT, aligned_edge=UP)
            label[1:3].set_color(YELLOW)

        self.play(
            FadeIn(axes, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (4.67, -0.94, 0.0), 10.96),
            LaggedStart(
                (FadeTransform(m1.copy(), m2) for m1, m2 in zip(root_labels, axes_labels)),
                lag_ratio=0.05,
                group_type=Group
            ),
            run_time=2
        )

        rect = Square(side_length=1e-3).move_to(plane.n2p(0))
        rect.set_stroke(TEAL, 3)
        for root_label, graph in zip(root_labels, graphs):
            self.play(
                ShowCreation(graph, time_span=(0.5, 2.0), suspend_mobject_updating=True),
                rect.animate.surround(root_label, buff=0.1),
            )
        self.play(FadeOut(rect))
        self.wait()

        # Add on constants
        constant_labels = VGroup(
            Tex(Rf"c_{{{n + 1}}}", font_size=60).next_to(label[0], LEFT, SMALL_BUFF, aligned_edge=UP)
            for n, label in enumerate(axes_labels)
        )
        constant_labels.set_color(BLUE_B)
        target_values = [0.5, 0.25, 1.5, -1.5, -1]

        solution_rect = SurroundingRectangle(VGroup(axes_labels, axes, constant_labels), buff=MED_SMALL_BUFF)
        solution_rect.set_stroke(WHITE, 1)
        solution_words = Text("All Solutions", font_size=60)
        solution_words.next_to(solution_rect, UP)
        solution_word = solution_words["Solutions"][0]
        solution_word.save_state(0)
        solution_word.match_x(solution_rect)

        const_rects = VGroup(SurroundingRectangle(c_label) for c_label in constant_labels)
        const_rects.set_stroke(BLUE, 3)

        plusses = Tex("+").replicate(4)
        for l1, l2, plus in zip(axes_labels, axes_labels[1:], plusses):
            plus.move_to(VGroup(l1, l2)).shift(SMALL_BUFF * LEFT)

        self.play(
            ShowCreation(solution_rect),
            Write(solution_word),
        )
        self.play(
            LaggedStartMap(Write, constant_labels, lag_ratio=0.5),
            LaggedStart(*(
                c_tracker.animate.set_value(value)
                for c_tracker, value in zip(c_trackers, target_values)
            ), lag_ratio=0.5),
            run_time=4
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, plusses),
            Write(solution_words["All"]),
            Restore(solution_word),
        )
        self.wait()

        # Play with constants
        self.play(LaggedStartMap(ShowCreation, const_rects, lag_ratio=0.15))
        value_sets = [
            [1, 1, 1, 1, 1],
            [1j, -1j, 1 + 1j, -1 + 1j, -0.5],
            [-0.5, 1j, 1j, 1 + 1j, -1],
        ]
        for values in value_sets:
            self.play(
                LaggedStart(*(
                    c_tracker.animate.set_value(value)
                    for c_tracker, value in zip(c_trackers, values)
                ), lag_ratio=0.25, run_time=3)
            )
            self.wait()
        self.play(LaggedStartMap(FadeOut, const_rects, lag_ratio=0.25))
        self.wait()

    def get_graph(self, axes, s, get_const):
        def func(t):
            return (get_const() * np.exp(s * t)).real

        graph = axes.get_graph(func, bind=True, stroke_color=TEAL, stroke_width=2)
        return graph


class HoldUpGeneralLinear(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "tease", look_at=3 * UR)
        )
        self.wait()
        self.play(
            morty.change("sassy", look_at=3 * UR),
            self.change_students("hesitant", "erm", "maybe")
        )
        self.wait(5)


class BigCross(InteractiveScene):
    def construct(self):
        cross = Cross(Rectangle(4, 1.5))
        cross.set_stroke(RED, width=(0, 8, 8, 8, 0))
        self.play(ShowCreation(cross))
        self.wait()


class DifferentialEquation(InteractiveScene):
    def construct(self):
        # ode to x
        x_term = Tex(R"x(t)", font_size=90)
        arrow = Vector(DOWN, thickness=5)
        arrow.move_to(ORIGIN, DOWN)
        words = Text("Differential Equation", font_size=72)
        words.next_to(arrow, UP)

        self.play(Write(x_term))
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(words),
            x_term.animate.next_to(arrow, DOWN),
        )
        self.wait()


class DumbTrickAlgebra(InteractiveScene):
    def construct(self):
        pass


class LaplaceTransformAlgebra(InteractiveScene):
    def construct(self):
        # Add equation
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            R"x(t)": colors[0],
            R"x'(t)": colors[1],
            R"x''(t)": colors[2],
            "{s}": YELLOW,
        }
        equation = Tex(
            R"{m} x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega_l t)",
            t2c=t2c
        )
        equation.to_edge(UP, buff=1.5)

        arrow = Vector(1.25 * DOWN, thickness=6)
        arrow.next_to(equation, DOWN)
        arrow_label = Tex(R"\\mathcal{L}", font_size=72)
        arrow_label.next_to(arrow, RIGHT, buff=SMALL_BUFF)

        self.add(equation)
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label, shift=0.5 * DOWN)
        )

        # Make transformed
        transformed_eq = Tex(
            R"{m} {s}^2 X({s}) + \\mu {s} X({s}) + k X({s}) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2}",
            t2c=t2c
        )
        transformed_eq.next_to(arrow, DOWN)

        xt_texs = ["x(t)", "x'(t)", "x''(t)"]
        Xs_texs = ["X({s})", "{s} X({s})", "{s}^2 X({s})"]

        rects = VGroup()
        srcs = VGroup()
        trgs = VGroup()
        for t1, t2, color in zip(xt_texs, Xs_texs, colors):
            src = equation[t1][0]
            trg = transformed_eq[t2][-1]
            rect = SurroundingRectangle(src, buff=0.05)
            rect.set_stroke(color, 2)
            rect.target = rect.generate_target()
            rect.target.surround(trg, buff=0.05)

            rects.add(rect)
            srcs.add(src.copy())
            trgs.add(trg)

        self.play(LaggedStartMap(ShowCreation, rects, lag_ratio=0.25, run_time=1.5))
        self.play(
            LaggedStart(
                *(FadeTransform(src, trg)
                for src, trg in zip(srcs, trgs)),
                lag_ratio=0.25,
                group_type=Group,
                run_time=1.5
            ),
            LaggedStartMap(MoveToTarget, rects, lag_ratio=0.25, run_time=1.5)
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation[tex], transformed_eq[tex][:2])
                for tex in ["{m}", "+", R"\\mu", "k", "="]
            )),
            TransformMatchingParts(
                equation[R"F_0 \\cos(\\omega_l t)"].copy(),
                transformed_eq[R"\\frac{F_0 {s}}{{s}^2 + \\omega_l^2}"]
            )
        )
        self.wait()

        # Factor it
        factored = Tex(
            R"X({s}) \\left({m} {s}^2+ \\mu {s} + k\\right) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2}",
            t2c=t2c
        )
        factored.move_to(transformed_eq)
        left_rect = SurroundingRectangle(factored["X({s})"], buff=0.05)
        left_rect.set_stroke(YELLOW, 2)

        self.play(
            TransformMatchingTex(
                transformed_eq,
                factored,
                matched_keys=["X({s})"],
                path_arc=30 * DEG
            ),
            ReplacementTransform(rects, VGroup(left_rect), path_arc=30 * DEG),
            run_time=2
        )
        self.play(FadeOut(left_rect))
        self.wait()

        # Rearrange
        rearranged = Tex(
            R"X({s}) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2} \\frac{1}{{m} {s}^2+ \\mu {s} + k}",
            t2c=t2c
        )
        rearranged.next_to(factored, DOWN, LARGE_BUFF)

        self.play(
            TransformMatchingTex(
                factored.copy(),
                rearranged,
                matched_keys=["X({s})"],
            )
        )


class ContrastDumbTrickAndLT(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(font_size=60)
        titles = VGroup(
            Text("Dumb trick", **kw),
            Text("Laplace Transform", **kw),
        )
        underlines = VGroup()
        for x, title in zip([-1, 1], titles):
            title.set_x(x * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=MED_SMALL_BUFF)
            underlines.add(Underline(title))
        underlines[0].scale(1.25)
        underlines[1].shift(SMALL_BUFF * UP)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)

        self.play(
            LaggedStartMap(FadeIn, titles[::-1], lag_ratio=0.5),
            LaggedStartMap(ShowCreation, underlines[::-1], lag_ratio=0.5),
            ShowCreation(v_line, time_span=(0.5, 2.0)),
            run_time=2
        )
        self.wait()


class DifferentialEquationToAlgebra(InteractiveScene):
    def construct(self):
        # Associations
        t2c = {"{s}": YELLOW}
        arrow = Vector(1.5 * RIGHT, thickness=5)
        words = VGroup(Text("ODE"), Text("Algebra"))
        symbols = VGroup(Tex(R"d / dt"), Tex(R"\\times {s}", t2c=t2c))
        for group in words, symbols:
            for mob, vect in zip(group, [LEFT, RIGHT]):
                mob.next_to(arrow, vect)

        deriv_eq = Tex(R"\\frac{d}{dt} e^{{s}t} = {s} e^{{s} t}", t2c=t2c, font_size=36)
        deriv_eq.next_to(arrow, UP, buff=0.25)

        self.play(LaggedStart(
            FadeIn(words[0], lag_ratio=0.1),
            GrowArrow(arrow),
            FadeIn(words[1], 0.25 * RIGHT),
        ))
        self.wait()
        self.play(
            FadeOut(words[0], 0.5 * UP),
            FadeIn(symbols[0], 0.5 * UP),
            FadeIn(deriv_eq)
        )
        self.play(
            FadeOut(words[1], 0.5 * UP),
            FadeIn(symbols[1], 0.5 * UP),
        )
        self.wait()


class SimpleExp(InteractiveScene):
    def construct(self):
        self.add(Tex(R"e^{st}", t2c={"s": YELLOW}, font_size=60))


class AtomsOfCalculus(InteractiveScene):
    def construct(self):
        # Test
        expr = Tex(R"e^{s_n t}", font_size=72)
        expr.to_edge(RIGHT, buff=LARGE_BUFF)
        rect = SurroundingRectangle(expr, buff=SMALL_BUFF)
        rect.set_stroke(YELLOW, 2)
        words = Text("Atoms of\\nCalculus", font_size=60)
        words.set_x(FRAME_WIDTH / 5)
        words.set_y(3)
        arrow = Arrow(words.get_right(), rect.get_top(), path_arc=-90 * DEG, thickness=4)
        arrow.set_fill(YELLOW)

        self.play(LaggedStart(
            Write(words),
            Write(arrow),
            ShowCreation(rect)
        ))
        self.wait()


class EndScreen(SideScrollEndScreen):
    pass`,
    annotations: {
      4: "IntroduceTrilogy extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      84: "Class DiscussTrilogy inherits from TeacherStudentsScene.",
      85: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      116: "Class WhoCares inherits from TeacherStudentsScene.",
      117: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      132: "MiniLessonTitle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      133: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      140: "Class WeGotThis inherits from TeacherStudentsScene.",
      141: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      155: "Class ConfusionAndWhy inherits from TeacherStudentsScene.",
      156: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      183: "ArrowBetweenScreens extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      184: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      201: "WhatAndWhy extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      202: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      222: "PrequelToLaplace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      223: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      230: "OtherExponentialDerivatives extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      231: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      249: "VariousExponentials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      250: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      276: "WhyToWhat extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      277: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      307: "DerivativeOfExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      308: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      373: "HighlightRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      374: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      391: "DefineI extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      392: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      398: "Class WaitWhy inherits from TeacherStudentsScene.",
      399: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      409: "MultiplicationByI extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      410: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      545: "UnitArcLengthsOnCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      546: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      581: "SimpleIndicationRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      582: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      588: "WriteSPlane extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      589: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      596: "ODEStoExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      597: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      633: "GenLinearEquationToOscillator extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      634: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      670: "VLineOverZero extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      671: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      686: "KIsSomeConstant extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      687: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      698: "WriteMu extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      699: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      713: "Class ReferenceGuessingExp inherits from TeacherStudentsScene.",
      714: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      801: "FromGuessToLaplace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      802: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      869: "JustAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      870: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      885: "BothPositiveNumbers extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      886: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      903: "Class ButSpringsAreReal inherits from TeacherStudentsScene.",
      904: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      917: "ShowIncreaseToK extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      918: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      942: "PureMathEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      943: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      981: "LinearityDefinition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      982: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1054: "Class ComplainAboutNeelessComplexity inherits from TeacherStudentsScene.",
      1055: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1082: "LetsGeneralize extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1083: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1093: "EquationRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1094: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1105: "GeneralLinearEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1106: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1396: "Class HoldUpGeneralLinear inherits from TeacherStudentsScene.",
      1397: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1414: "BigCross extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1415: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1422: "DifferentialEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1423: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1441: "DumbTrickAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1442: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1446: "LaplaceTransformAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1447: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1560: "ContrastDumbTrickAndLT extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1561: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1587: "DifferentialEquationToAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1588: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1619: "SimpleExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1620: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1624: "AtomsOfCalculus extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1625: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1645: "Class EndScreen inherits from SideScrollEndScreen.",
    }
  };

  files["_2025/laplace/shm.py"] = {
    description: "Spring-mass system for the Laplace transform series. The SrpingMassSystem class (typo in original) combines visual springs, mass blocks, and real-time physics ODE integration via updaters. Supports spring constant k, damping mu, external forcing, and force/velocity visualization.",
    code: `from manim_imports_ext import *


def get_coef_colors(n_coefs=3):
    return [
        interpolate_color_by_hsl(TEAL, RED, a)
        for a in np.linspace(0, 1, n_coefs)
    ]


class SrpingMassSystem(VGroup):
    def __init__(
        self,
        x0=0,
        v0=0,
        k=3,
        mu=0.1,
        equilibrium_length=7,
        equilibrium_position=ORIGIN,
        direction=RIGHT,
        spring_stroke_color=GREY_B,
        spring_stroke_width=2,
        spring_radius=0.25,
        n_spring_curls=8,
        mass_width=1.0,
        mass_color=BLUE_E,
        mass_label="m",
        external_force=None,
    ):
        super().__init__()
        self.equilibrium_position = equilibrium_position
        self.fixed_spring_point = equilibrium_position - (equilibrium_length - 0.5 * mass_width) * direction
        self.direction = direction
        self.rot_off_horizontal = angle_between_vectors(RIGHT, direction)
        self.mass = self.get_mass(mass_width, mass_color, mass_label)
        self.spring = self.get_spring(spring_stroke_color, spring_stroke_width, n_spring_curls, spring_radius)
        self.add(self.spring, self.mass)

        self.k = k
        self.mu = mu
        self.set_x(x0)
        self.velocity = v0

        self.external_force = external_force

        self._is_running = True
        self.add_updater(lambda m, dt: m.time_step(dt))

    def get_spring(self, stroke_color, stroke_width, n_curls, radius):
        spring = ParametricCurve(
            lambda t: [t, -radius * math.sin(TAU * t), radius * math.cos(TAU * t)],
            t_range=(0, n_curls, 1e-2),
            stroke_color=stroke_color,
            stroke_width=stroke_width,
        )
        spring.rotate(self.rot_off_horizontal)
        return spring

    def get_mass(self, mass_width, mass_color, mass_label):
        mass = Square(mass_width)
        mass.set_fill(mass_color, 1)
        mass.set_stroke(WHITE, 1)
        mass.set_shading(0.1, 0.1, 0.1)
        label = Tex(mass_label)
        label.set_max_width(0.5 * mass.get_width())
        label.move_to(mass)
        mass.add(label)
        mass.label = label
        return mass

    def set_x(self, x):
        self.mass.move_to(self.equilibrium_position + x * self.direction)
        spring_width = SMALL_BUFF + get_norm(self.mass.get_left() - self.fixed_spring_point)
        self.spring.rotate(-self.rot_off_horizontal)
        self.spring.set_width(spring_width, stretch=True)
        self.spring.rotate(self.rot_off_horizontal)
        self.spring.move_to(self.fixed_spring_point, -self.direction)

    def get_x(self):
        return (self.mass.get_center() - self.equilibrium_position)[0]

    def time_step(self, delta_t, dt_size=1e-3):
        if not self._is_running:
            return
        if delta_t == 0:
            return

        state = [self.get_x(), self.velocity]
        sub_steps = max(int(delta_t / dt_size), 1)
        true_dt = delta_t / sub_steps
        for _ in range(sub_steps):
            # ODE
            x, v = state
            state += np.array([v, self.get_force(x, v)]) * true_dt

        self.set_x(state[0])
        self.velocity = state[1]

    def pause(self):
        self._is_running = False

    def unpause(self):
        self._is_running = True

    def set_k(self, k):
        self.k = k
        return self

    def set_mu(self, mu):
        self.mu = mu
        return self

    def get_velocity(self):
        return self.velocity

    def set_velocity(self, velocity):
        self.velocity = velocity
        return self

    def get_velocity_vector(self, scale_factor=0.5, thickness=3.0, v_offset=-0.25, color=GREEN):
        """Get a vector showing the mass's velocity"""
        vector = Vector(RIGHT, fill_color=color)
        v_shift = v_offset * UP
        vector.add_updater(lambda m: m.put_start_and_end_on(
            self.mass.get_center() + v_shift,
            self.mass.get_center() + v_shift + scale_factor * self.velocity * RIGHT
        ))
        return vector

    def get_force_vector(self, scale_factor=0.5, thickness=3.0, v_offset=-0.25, color=RED):
        """Get a vector showing the mass's velocity"""
        vector = Vector(RIGHT, fill_color=color)
        v_shift = v_offset * UP
        vector.add_updater(lambda m: m.put_start_and_end_on(
            self.mass.get_center() + v_shift,
            self.mass.get_center() + v_shift + scale_factor * self.get_force(self.get_x(), self.velocity) * RIGHT
        ))
        return vector

    def add_external_force(self, func):
        self.external_force = func

    def get_force(self, x, v):
        force = -self.k * x - self.mu * v
        if self.external_force is not None:
            force += self.external_force()
        return force


class BasicSpringScene(InteractiveScene):
    def construct(self):
        # Add spring, give some initial oscillation
        spring = SrpingMassSystem(
            x0=2,
            mu=0.1,
            k=3,
            equilibrium_position=2 * LEFT,
            equilibrium_length=5,
        )
        self.add(spring)

        # Label on a number line
        number_line = NumberLine(x_range=(-4, 4, 1))
        number_line.next_to(spring.equilibrium_position, DOWN, buff=2.0)
        number_line.add_numbers(font_size=24)

        # Dashed line from mass to number line
        dashed_line = DashedLine(
            spring.mass.get_bottom(),
            number_line.n2p(spring.get_x()),
            stroke_color=GREY,
            stroke_width=2
        )
        dashed_line.always.match_x(spring.mass)

        # Arrow tip on number line
        arrow_tip = ArrowTip(length=0.2, width=0.1)
        arrow_tip.rotate(-90 * DEG)  # Point downward
        arrow_tip.set_fill(TEAL)
        arrow_tip.add_updater(lambda m: m.move_to(number_line.n2p(spring.get_x()), DOWN))

        x_label = Tex("x = 0.00", font_size=24)
        x_number = x_label.make_number_changeable("0.00")
        x_number.add_updater(lambda m: m.set_value(spring.get_x()))
        x_label.add_updater(lambda m: m.next_to(arrow_tip, UR, buff=0.1))

        # Ambient playing, fade in labels
        self.wait(2)
        self.play(
            VFadeIn(number_line),
            VFadeIn(dashed_line),
            VFadeIn(arrow_tip),
            VFadeIn(x_label),
        )
        self.wait(7)

        # (For an insertion)
        if False:
            x_label_arrow = Vector(1.5 * DL, thickness=8)
            x_label_arrow.set_fill(YELLOW)
            x_label_arrow.always.move_to(arrow_tip, DL).shift(2 * RIGHT + 0.75 * UP)
            self.play(
                VFadeIn(x_label_arrow, time_span=(1, 2)),
                x_label.animate.scale(2),
                run_time=2
            )
            self.wait(8)

        # Show velocity
        x_color, v_color, a_color = [interpolate_color_by_hsl(TEAL, RED, a) for a in np.linspace(0, 1, 3)]
        v_vect = spring.get_velocity_vector(color=v_color, scale_factor=0.25)
        a_vect = spring.get_force_vector(color=a_color, scale_factor=0.25)
        a_vect.add_updater(lambda m: m.shift(v_vect.get_end() - m.get_start()))

        self.play(VFadeIn(v_vect))
        self.wait(5)
        self.play(VFadeIn(a_vect))
        self.wait(8)
        self.wait_until(lambda: spring.velocity <= 0)

        # Show the force law
        self.remove(v_vect)
        a_vect.remove_updater(a_vect.get_updaters()[-1])
        spring.pause()

        self.wait()
        for x in range(2, 5):
            self.play(spring.animate.set_x(x))
            self.wait()

        # Back and forth
        t_tracker = ValueTracker(0)
        self.play(
            UpdateFromAlphaFunc(
                spring,
                lambda m, a: m.set_x(4 * math.cos(2 * TAU * a)),
                rate_func=linear,
                run_time=8,
            )
        )

        # Ambient springing
        spring.unpause()
        spring.set_mu(0.25)
        self.wait(15)

        # Show the solution graph
        frame = self.frame

        time_tracker = ValueTracker(0)
        time_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        axes = Axes(
            x_range=(0, 20, 1),
            y_range=(-2, 2, 1),
            width=12,
            height=3,
            axis_config={"stroke_color": GREY}
        )
        axes.next_to(spring, UP, LARGE_BUFF)
        axes.align_to(number_line, LEFT)

        x_axis_label = Text("Time", font_size=24).next_to(axes.x_axis, RIGHT, buff=0.1)
        y_axis_label = Tex("x(t)", font_size=24).next_to(axes.y_axis.get_top(), RIGHT, buff=0.1)
        axes.add(x_axis_label)
        axes.add(y_axis_label)

        tracking_point = Point()
        tracking_point.add_updater(lambda p: p.move_to(
            axes.c2p(time_tracker.get_value(), spring.get_x())
        ))

        position_graph = TracedPath(
            tracking_point.get_center,
            stroke_color=BLUE,
            stroke_width=3,
        )

        spring.pause()
        spring.set_velocity(0)
        self.play(
            frame.animate.reorient(0, 0, 0, (2.88, 1.88, 0.0), 12.48),
            FadeIn(axes),
            VFadeOut(a_vect),
            spring.animate.set_x(2),
        )
        self.add(tracking_point, position_graph, time_tracker)
        spring.unpause()
        self.wait(20)
        position_graph.clear_updaters()
        self.wait(20)


class DampingForceDemo(InteractiveScene):
    def construct(self):
        # Create spring-mass system with invisible spring and damping only
        spring_system = SrpingMassSystem(
            x0=-4,
            v0=2,
            k=0,
            mu=0.3,
            equilibrium_position=ORIGIN,
            equilibrium_length=6,
            mass_width=0.8,
            mass_color=BLUE_E,
            mass_label="m",
        )
        spring_system.spring.set_opacity(0)
        self.add(spring_system)

        # Create velocity vector
        v_color = interpolate_color_by_hsl(TEAL, RED, 0.5)
        velocity_vector = spring_system.get_velocity_vector(color=v_color, scale_factor=0.8)

        velocity_label = Tex(R'\\vec{\\textbf{v}}', font_size=24)
        velocity_label.set_color(v_color)
        velocity_label.always.next_to(velocity_vector, RIGHT, buff=SMALL_BUFF)
        velocity_label.add_updater(lambda m: m.set_max_width(0.5 * velocity_vector.get_width()))

        # Create damping force vector
        damping_vector = spring_system.get_velocity_vector(scale_factor=-0.5, color=RED, v_offset=-0.5)
        damping_label = Tex(R"-\\mu v", fill_color=RED, font_size=24)
        damping_label.always.next_to(damping_vector, DOWN, SMALL_BUFF)

        # Add vectors and labels
        self.add(velocity_vector, velocity_label)
        self.add(damping_vector, damping_label)

        # Let the system evolve
        self.wait(15)


class SolveDampedSpringEquation(InteractiveScene):
    def construct(self):
        # Show x and its derivatives
        pos, vel, acc = funcs = VGroup(
            Tex(R"x(t)"),
            Tex(R"x'(t)"),
            Tex(R"x''(t)"),
        )
        funcs.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)

        labels = VGroup(
            Text("Position").set_color(BLUE),
            Text("Velocity").set_color(RED),
            Text("Acceleration").set_color(YELLOW),
        )
        colors = get_coef_colors()
        for line, label, color in zip(funcs, labels, colors):
            label.set_color(color)
            label.next_to(line, RIGHT, MED_LARGE_BUFF)
            label.align_to(labels[0], LEFT)

        VGroup(funcs, labels).to_corner(UR)

        arrows = VGroup()
        for l1, l2 in zip(funcs, funcs[1:]):
            arrow = Line(l1.get_left(), l2.get_left(), path_arc=150 * DEG, buff=0.2)
            arrow.add_tip(width=0.2, length=0.2)
            arrow.set_color(GREY_B)
            ddt = Tex(R"\\frac{d}{dt}", font_size=30)
            ddt.set_color(GREY_B)
            ddt.next_to(arrow, LEFT, SMALL_BUFF)
            arrow.add(ddt)
            arrows.add(arrow)

        self.play(Write(funcs[0]), Write(labels[0]))
        self.wait()
        for func1, func2, label1, label2, arrow in zip(funcs, funcs[1:], labels, labels[1:], arrows):
            self.play(LaggedStart(
                GrowFromPoint(arrow, arrow.get_corner(UR), path_arc=30 * DEG),
                TransformFromCopy(func1, func2, path_arc=30 * DEG),
                FadeTransform(label1.copy(), label2),
                lag_ratio=0.1
            ))
            self.wait()

        deriv_group = VGroup(funcs, labels, arrows)

        # Show F=ma
        t2c = {
            "x(t)": colors[0],
            "x'(t)": colors[1],
            "x''(t)": colors[2],
        }
        equation1 = Tex(R"{m} x''(t) = -k x(t) - \\mu x'(t)", t2c=t2c)
        equation1.to_corner(UL)

        ma = equation1["{m} x''(t)"][0]
        kx = equation1["-k x(t)"][0]
        mu_v = equation1[R"- \\mu x'(t)"][0]
        rhs = VGroup(kx, mu_v)

        ma_brace, kx_brace, mu_v_brace = braces = VGroup(
            Brace(part, DOWN, buff=SMALL_BUFF)
            for part in [ma, kx, mu_v]
        )
        label_texs = [R"\\textbf{F}", R"\\text{Spring force}", R"\\text{Damping}"]
        for brace, label_tex in zip(braces, label_texs):
            brace.add(brace.get_tex(label_tex))

        self.play(TransformFromCopy(acc, ma[1:], path_arc=-45 * DEG))
        self.play(LaggedStart(
            GrowFromCenter(ma_brace),
            Write(ma[0]),
            run_time=1,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(LaggedStart(
            Write(equation1["= -k"][0]),
            FadeTransformPieces(ma_brace, kx_brace),
            TransformFromCopy(pos, equation1["x(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(LaggedStart(
            FadeTransformPieces(kx_brace, mu_v_brace),
            Write(equation1[R"- \\mu"][0]),
            TransformFromCopy(vel, equation1["x'(t)"][0], path_arc=-45 * DEG),
        ))
        self.wait()
        self.play(FadeOut(mu_v_brace))

        # Rearrange
        equation2 = Tex(R"{m} x''(t) + \\mu x'(t) + k x(t) = 0", t2c=t2c)
        equation2.move_to(equation1, UL)

        self.play(TransformMatchingTex(equation1, equation2, path_arc=45 * DEG))
        self.wait()

        # Hypothesis of e^st
        t2c = {"s": YELLOW, "x(t)": TEAL}
        hyp_word, hyp_tex = hypothesis = VGroup(
            Text("Hypothesis: "),
            Tex("x(t) = e^{st}", t2c=t2c),
        )
        hypothesis.arrange(RIGHT)
        hypothesis.to_corner(UR)
        sub_hyp_word = TexText(R"(For some $s$)", t2c={"$s$": YELLOW}, font_size=36, fill_color=GREY_B)
        sub_hyp_word.next_to(hyp_tex, DOWN)

        self.play(LaggedStart(
            FadeTransform(pos.copy(), hyp_tex[:4], path_arc=45 * DEG, remover=True),
            FadeOut(deriv_group),
            Write(hyp_word, run_time=1),
            Write(hyp_tex[4:], time_span=(0.5, 1.5)),
        ))
        self.add(hypothesis)
        self.wait()
        self.play(FadeIn(sub_hyp_word, 0.25 * DOWN))
        self.wait()

        # Plug it in
        t2c["s"] = YELLOW
        equation3 = Tex(R"{m} s^2 e^{st} + \\mu s e^{st} + k e^{st} = 0", t2c=t2c)
        equation3.next_to(equation2, DOWN, LARGE_BUFF)
        pos_parts = VGroup(equation2["x(t)"][0], equation3["e^{st}"][-1])
        vel_parts = VGroup(equation2["x'(t)"][0], equation3["s e^{st}"][0])
        acc_parts = VGroup(equation2["x''(t)"][0], equation3["s^2 e^{st}"][0])
        matched_parts = VGroup(pos_parts, vel_parts, acc_parts)

        pos_rect, vel_rect, acc_rect = rects = VGroup(
            SurroundingRectangle(group[0], buff=0.05).set_stroke(group[0][0].get_color(), 1)
            for group in matched_parts
        )

        pos_arrow, vel_arrow, acc_arrow = arrows = VGroup(
            Arrow(*pair, buff=0.1)
            for pair in matched_parts
        )

        for rect, arrow, pair in zip(rects, arrows, matched_parts):
            self.play(ShowCreation(rect))
            self.play(
                GrowArrow(arrow),
                FadeTransform(pair[0].copy(), pair[1]),
                rect.animate.surround(pair[1]),
            )
            self.wait()
        self.play(
            LaggedStart(
                (TransformFromCopy(equation2[tex], equation3[tex])
                for tex in ["{m}", "+", "k", R"\\mu", "=", "0"]),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(FadeOut(arrows, lag_ratio=0.1), FadeOut(rects, lag_ratio=0.1))

        # Solve for s
        key_syms = ["s", "m", R"\\mu", "k"]
        equation4, equation5, equation6 = new_equations = VGroup(
            Tex(R"e^{st} \\left( ms^2 + \\mu s + k \\right) = 0", t2c=t2c),
            Tex(R"ms^2 + \\mu s + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\\mu \\pm \\sqrt{\\mu^2 - 4mk}} \\over 2m}", isolate=key_syms)
        )
        rhs = equation6[2:]
        rhs.set_width(equation5.get_width() - equation6[:2].get_width(), about_edge=LEFT)
        equation6.refresh_bounding_box()
        equation6["{s}"].set_color(YELLOW)
        equation6.scale(1.25, about_edge=LEFT)

        new_equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        new_equations.move_to(equation3, UL)
        equation4 = new_equations[0]

        exp_rect = SurroundingRectangle(equation4[R"e^{st}"])
        exp_rect.set_stroke(YELLOW, 2)
        ne_0 = VGroup(Tex(R"\\ne").rotate(90 * DEG), Integer(0))
        ne_0.arrange(DOWN).next_to(exp_rect, DOWN)

        self.play(
            TransformMatchingTex(
                equation3,
                equation4,
                matched_keys=[R"e^{st}"],
                run_time=1.5,
                path_arc=30 * DEG
            )
        )
        self.wait(0.5)
        self.play(ShowCreation(exp_rect))
        self.wait()
        self.play(Write(ne_0))
        self.wait()
        self.play(FadeOut(ne_0))
        self.play(
            *(
                TransformFromCopy(equation4[key], equation5[key])
                for key in [R"ms^2 + \\mu s + k", "= 0"]
            ),
            FadeOut(exp_rect),
        )
        self.wait()

        # Show mirror image
        self.play(
            TransformMatchingTex(
                equation5.copy(), equation2.copy(),
                key_map={
                    "s^2": "x''(t)",
                    R"\\mu s": R"\\mu x(t)",
                    R"k": R"k x(t)",
                },
                # match_animation=FadeTransform,
                # mismatch_animation=FadeTransform,
                remover=True,
                rate_func=there_and_back_with_pause,
                run_time=6
            ),
            equation4.animate.set_fill(opacity=0.25),
        )
        self.play(equation4.animate.set_fill(opacity=1))
        self.wait()

        # Cover up mu terms
        boxes = VGroup(
            SurroundingRectangle(mob)
            for mob in [
                equation2[R"+ \\mu x'(t)"],
                equation4[R"+ \\mu s"],
                equation5[R"+ \\mu s"],
            ]
        )
        boxes.set_fill(BLACK, 0)
        boxes.set_stroke(colors[1], 2)

        self.add(Point())
        self.play(FadeIn(boxes, lag_ratio=0.1))
        self.play(boxes.animate.set_fill(BLACK, 0.8).set_stroke(width=1, opacity=0.5))
        self.wait()

        # Add simple answer
        simple_answer = Tex(R"s = \\pm i \\sqrt{k / m}", t2c=t2c)
        simple_answer.next_to(equation5, DOWN, LARGE_BUFF, aligned_edge=RIGHT)

        omega_brace = Brace(simple_answer[R"\\sqrt{k / m}"], DOWN, SMALL_BUFF)
        omega = omega_brace.get_tex(R"\\omega")
        omega.set_color(PINK)

        self.play(FadeIn(simple_answer))
        self.wait()
        self.play(GrowFromCenter(omega_brace), Write(omega))
        self.wait()

        simple_answer.add(omega_brace, omega)

        # Reminder of what s represents
        s_copy = simple_answer[0].copy()
        s_rect = SurroundingRectangle(s_copy)

        self.play(ShowCreation(s_rect))
        self.wait()
        self.play(
            s_rect.animate.surround(hyp_tex["e^{st}"]).set_anim_args(path_arc=-60 * DEG),
            FadeTransform(s_copy, hyp_tex["s"], path_arc=-60 * DEG),
            run_time=2
        )
        self.wait()
        self.play(FadeOut(s_rect))

        # Move hypothesis
        frame = self.frame
        self.play(
            frame.animate.scale(1.5, about_edge=LEFT),
            hypothesis.animate.next_to(equation2, UP, LARGE_BUFF, aligned_edge=LEFT),
            FadeOut(sub_hyp_word),
            run_time=1.5
        )
        self.wait()

        # Show quadratic formula
        qf_arrow = Arrow(
            equation5.get_right(),
            equation6.get_corner(UR) + 0.5 * LEFT,
            path_arc=-150 * DEG
        )
        qf_words = Text("Quadratic\\nFormula", font_size=30, fill_color=GREY_B)
        qf_words.next_to(qf_arrow.get_center(), UR)

        naked_equation = equation6.copy()
        for sym in key_syms:
            naked_equation[sym].scale(0).set_fill(opacity=0).move_to(10 * LEFT)

        qf_rect = SurroundingRectangle(equation6[2:])
        qf_rect.set_stroke(YELLOW, 1.5)

        self.play(
            FadeOut(simple_answer, DOWN),
            boxes.animate.set_fill(opacity=0).set_stroke(width=2, opacity=1)
        )
        self.play(FadeOut(boxes))
        self.wait()
        self.play(
            TransformFromCopy(equation5["s"], equation6["s"]),
            Write(equation6["="]),
            GrowFromPoint(qf_arrow, qf_arrow.get_corner(UL)),
            FadeIn(qf_words, shift=0.5 * DOWN),
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation5[sym], equation6[sym], time_span=(0.5, 1.5))
                for sym in key_syms[1:]
            ), lag_ratio=0.1),
            Write(naked_equation),
        )
        self.wait()
        self.remove(naked_equation)
        self.add(equation6)
        self.play(ShowCreation(qf_rect))
        self.wait()

    def old_material(self):
        # Show implied exponentials
        final_equation = new_equations[-1]
        consolidated_lines = VGroup(
            hypothesis,
            equation2,
            equation4,
            final_equation,
        )
        consolidated_lines.target = consolidated_lines.generate_target()
        consolidated_lines.target.scale(0.7)
        consolidated_lines.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        consolidated_lines.target.to_corner(UL)

        implies = Tex(R"\\Longrightarrow", font_size=60)
        implies.next_to(consolidated_lines.target[0], RIGHT, buff=0.75)

        t2c = {"x(t)": TEAL, R"\\omega": PINK}
        imag_exps = VGroup(
            Tex(R"x(t) = e^{+i \\omega t}", t2c=t2c),
            Tex(R"x(t) = e^{-i \\omega t}", t2c=t2c),
        )
        imag_exps.arrange(RIGHT, buff=2.0)
        imag_exps.next_to(implies, RIGHT, buff=0.75)

        self.remove(final_equation)
        self.play(LaggedStart(
            FadeOut(arrows),
            FadeOut(equation3, 0.5 * UP),
            FadeOut(sub_hyp_word),
            MoveToTarget(consolidated_lines),
            Write(implies),
        ))
        for imag_exp, sgn in zip(imag_exps, "+-"):
            self.play(
                TransformFromCopy(hyp_tex["x(t) ="][0], imag_exp["x(t) ="][0]),
                TransformFromCopy(hyp_tex["e"][0], imag_exp["e"][0]),
                TransformFromCopy(hyp_tex["t"][-1], imag_exp["t"][-1]),
                FadeTransform(final_equation[R"\\pm i"][0].copy(), imag_exp[Rf"{sgn}i"][0]),
                FadeTransform(final_equation[R"\\sqrt{k/m}"][0].copy(), imag_exp[R"\\omega"][0]),
            )

        omega_brace = Brace(final_equation[R"\\sqrt{k/m}"], DOWN, SMALL_BUFF)
        omega_label = omega_brace.get_tex(R"\\omega").set_color(PINK)
        self.play(GrowFromCenter(omega_brace), Write(omega_label))
        self.wait()

        # Combine two solutions
        cos_equation = Tex(R"e^{+i \\omega t} + e^{-i \\omega t} = 2\\cos(\\omega t)", t2c={R"\\omega": PINK})
        cos_equation.move_to(imag_exps)
        omega_brace2 = omega_brace.copy()
        omega_brace2.stretch(0.5, 0).match_width(cos_equation[R"\\omega"][-1])
        omega_brace2.next_to(cos_equation[R"\\omega"][-1], DOWN, SMALL_BUFF)
        omega_brace2_tex = omega_brace2.get_tex(R"\\sqrt{k / m}", buff=SMALL_BUFF, font_size=24)

        self.remove(imag_exps)
        self.play(
            TransformFromCopy(imag_exps[0][R"e^{+i \\omega t}"], cos_equation[R"e^{+i \\omega t}"]),
            TransformFromCopy(imag_exps[1][R"e^{-i \\omega t}"], cos_equation[R"e^{-i \\omega t}"]),
            FadeOut(imag_exps[0][R"x(t) ="]),
            FadeOut(imag_exps[1][R"x(t) ="]),
            Write(cos_equation["+"][1]),
        )
        self.wait()
        self.play(Write(cos_equation[R"= 2\\cos(\\omega t)"]))
        self.wait()
        self.play(GrowFromCenter(omega_brace2), Write(omega_brace2_tex))

        # Clear the board
        self.play(LaggedStart(
            FadeOut(implies),
            FadeOut(cos_equation),
            FadeOut(omega_brace2),
            FadeOut(omega_brace2_tex),
            FadeOut(consolidated_lines[2:]),
            FadeOut(omega_brace),
            FadeOut(omega_label),
            lag_ratio=0.1
        ))

        # Add damping term
        t2c = {"x''(t)": colors[2], "x'(t)": colors[1], "x(t)": colors[0], "{s}": YELLOW}
        new_lines = VGroup(
            Tex(R"m x''(t) + \\mu x'(t) + k x(t) = 0", t2c=t2c),
            Tex(R"m ({s}^2 e^{{s}t}) + \\mu ({s} e^{{s}t}) + k (e^{{s}t}) = 0", t2c=t2c),
            Tex(R"e^{{s}t}\\left(m {s}^2 + \\mu {s} + k \\right) = 0", t2c=t2c),
            Tex(R"m {s}^2 + \\mu {s} + k = 0", t2c=t2c),
            Tex(R"{s} = {{-\\mu \\pm \\sqrt{\\mu^2 - 4mk}} \\over 2m}", t2c=t2c),
        )
        new_lines.scale(0.7)
        new_lines.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        new_lines.move_to(equation2, UL)

        self.play(
            TransformMatchingTex(
                equation2,
                new_lines[0],
                matched_keys=t2c.keys(),
                run_time=1
            )
        )
        self.wait()
        for line1, line2 in zip(new_lines, new_lines[1:]):
            if line1 is new_lines[0]:
                key_map = {
                    "x''(t)": R"({s}^2 e^{{s}t})",
                    "x'(t)": R"({s} e^{{s}t})",
                    "x(t)": R"(e^{{s}t})",
                }
            else:
                key_map = dict()
            self.play(TransformMatchingTex(line1.copy(), line2, key_map=key_map, run_time=1, lag_ratio=0.01))
            self.wait()


class DampedSpringSolutionsOnSPlane(InteractiveScene):
    def construct(self):
        # Add the plane
        plane = ComplexPlane((-3, 2), (-2, 2))
        plane.set_height(5)
        plane.background_lines.set_stroke(BLUE, 1)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)
        plane.add_coordinate_labels(font_size=24)
        plane.move_to(DOWN)
        plane.to_edge(RIGHT, buff=1.0)
        self.add(plane)

        # Add the sliders
        colors = [interpolate_color_by_hsl(RED, TEAL, a) for a in np.linspace(0, 1, 3)]
        chars = ["m", R"\\mu", "k"]
        m_slider, mu_slider, k_slider = sliders = VGroup(
            self.get_slider(char, color)
            for char, color in zip(chars, colors)
        )
        m_tracker, mu_tracker, k_tracker = trackers = Group(
            slider.value_tracker for slider in sliders
        )
        sliders.arrange(RIGHT, buff=MED_LARGE_BUFF)
        sliders.next_to(plane, UP, aligned_edge=LEFT)

        for tracker, value in zip(trackers, [1, 0, 3]):
            tracker.set_value(value)

        self.add(trackers)
        self.add(sliders[0], sliders[2])

        # Add the dots
        def get_roots():
            a, b, c = [t.get_value() for t in trackers]
            m = -b / 2
            p = c / a
            disc = m**2 - p
            radical = math.sqrt(disc) if disc >= 0 else 1j * math.sqrt(-disc)
            return (m + radical, m - radical)

        def update_dots(dots):
            roots = get_roots()
            for dot, root in zip(dots, roots):
                dot.move_to(plane.n2p(root))

        root_dots = GlowDot().replicate(2)
        root_dots.add_updater(update_dots)

        s_rhs_point = Point((-4.09, -1.0, 0.0))
        rect_edge_point = (-3.33, -1.18, 0.0)

        def update_lines(lines):
            for line, dot in zip(lines, root_dots):
                line.put_start_and_end_on(
                    s_rhs_point.get_center(),
                    dot.get_center(),
                )

        lines = Line().replicate(2)
        lines.set_stroke(YELLOW, 2, 0.35)
        lines.add_updater(update_lines)

        self.add(root_dots)

        # Play with k
        self.play(ShowCreation(lines, lag_ratio=0, suspend_mobject_updating=True))
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(m_tracker.animate.set_value(4), run_time=2)
        self.wait()
        self.play(k_tracker.animate.set_value(3), run_time=2)
        self.play(m_tracker.animate.set_value(1), run_time=2)
        self.wait()

        # Play with mu
        self.play(
            s_rhs_point.animate.move_to(rect_edge_point),
            VFadeOut(lines),
            VFadeIn(sliders[1])
        )
        self.wait()
        self.play(mu_tracker.animate.set_value(3), run_time=5)
        self.wait()
        self.play(mu_tracker.animate.set_value(0.5), run_time=3)
        self.play(ShowCreation(lines, lag_ratio=0, suspend_mobject_updating=True))

        # Background
        self.add_background_image()

        # Zoom out and show graph
        frame = self.frame

        axes = Axes((0, 10, 1), (-1, 1, 1), width=10, height=3.5)
        axes.next_to(plane, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        def func(t):
            roots = get_roots()
            return 0.5 * (np.exp(roots[0] * t) + np.exp(roots[1] * t)).real

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 3)
        axes.bind_graph_to_func(graph, func)

        graph_label = Tex(R"\\text{Re}[e^{st}]", t2c={"s": YELLOW}, font_size=72)
        graph_label.next_to(axes.get_corner(UL), DL)

        self.play(
            frame.animate.set_height(12, about_point=4 * UP + 2 * LEFT),
            FadeIn(axes, time_span=(1.5, 3)),
            ShowCreation(graph, suspend_mobject_updating=True, time_span=(1.5, 3)),
            Write(graph_label),
            run_time=3
        )
        self.wait()

        # Show exponential decay
        exp_graph = axes.get_graph(lambda t: np.exp(get_roots()[0].real * t))
        exp_graph.set_stroke(WHITE, 1)

        self.play(ShowCreation(exp_graph))
        self.wait()

        # More play
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(k_tracker.animate.set_value(4), run_time=2)
        self.play(FadeOut(exp_graph))
        self.wait()
        self.play(mu_tracker.animate.set_value(2), run_time=3)
        self.play(k_tracker.animate.set_value(2), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(3.5), run_time=3)
        self.play(k_tracker.animate.set_value(5), run_time=2)
        self.wait()
        self.play(
            mu_tracker.animate.set_value(0.5),
            m_tracker.animate.set_value(3),
            run_time=3
        )
        self.wait()

        # Smooth all the way to end
        self.play(mu_tracker.animate.set_value(4.2), run_time=12)

    def add_background_image(self):
        image = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/shm/images/LaplaceFormulaStill.png')
        image.replace(self.frame)
        image.set_z_index(-1)
        self.background_image = image
        self.add(image)

    def get_slider(self, char_name, color=WHITE, x_range=(0, 5), height=1.5, font_size=36):
        tracker = ValueTracker(0)
        number_line = NumberLine(x_range, width=height, tick_size=0.05)
        number_line.rotate(90 * DEG)

        indicator = ArrowTip(width=0.1, length=0.2)
        indicator.rotate(PI)
        indicator.add_updater(lambda m: m.move_to(number_line.n2p(tracker.get_value()), LEFT))
        indicator.set_color(color)

        label = Tex(Rf"{char_name} = 0.00", font_size=font_size)
        label[char_name].set_color(color)
        label.rhs = label.make_number_changeable("0.00")
        label.always.next_to(indicator, RIGHT, SMALL_BUFF)
        label.rhs.f_always.set_value(tracker.get_value)

        slider = VGroup(number_line, indicator, label)
        slider.value_tracker = tracker
        return slider

    def insertion(self):
        # Insertion after "play with mu" above
        self.wait()
        self.play(mu_tracker.animate.set_value(3), run_time=3)
        self.play(k_tracker.animate.set_value(1), run_time=2)
        self.play(k_tracker.animate.set_value(4), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(4), run_time=2)
        self.play(k_tracker.animate.set_value(0.5), run_time=2)
        self.wait()
        self.play(mu_tracker.animate.set_value(0.5), run_time=4)
        self.play(k_tracker.animate.set_value(2), run_time=4)


class RotatingExponentials(InteractiveScene):
    def construct(self):
        # Create time tracker
        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))
        get_t = t_tracker.get_value
        omega = PI / 2

        def get_x():
            return math.cos(omega * get_t())

        self.add(t_tracker)

        # Create two complex planes side by side
        left_plane, right_plane = planes = VGroup(
            ComplexPlane(
                (-2, 2), (-2, 2),
                background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            )
            for _ in range(2)
        )
        for plane in planes:
            plane.axes.set_stroke(width=1)
            plane.set_height(3.5)
            plane.add_coordinate_labels(font_size=16)
        planes.arrange(RIGHT, buff=1.0)
        planes.to_edge(RIGHT)
        planes.to_edge(UP, buff=1.5)

        self.add(planes)

        # Add titles
        t2c = {R"\\omega": PINK}
        left_title, right_title = titles = VGroup(
            Tex(tex, t2c=t2c, font_size=48)
            for tex in [
                R"e^{+i \\omega t}",
                R"e^{-i \\omega t}",
            ]
        )
        for title, plane in zip(titles, planes):
            title.next_to(plane, UP)

        self.add(titles)

        # Create rotating vectors
        left_vector = self.get_rotating_vector(left_plane, 1j * omega, t_tracker, color=TEAL)
        right_vector = self.get_rotating_vector(right_plane, -1j * omega, t_tracker, color=RED)
        vectors = VGroup(left_vector, right_vector)

        left_tail, right_tail = tails = VGroup(
            TracingTail(vect.get_end, stroke_color=vect.get_color(), time_traced=2)
            for vect in vectors
        )

        self.add(Point())
        self.add(vectors, tails)

        # Add time display
        time_display = Tex("t = 0.00", font_size=36).to_corner(UR)
        time_label = time_display.make_number_changeable("0.00")
        time_label.add_updater(lambda m: m.set_value(t_tracker.get_value()))

        # Animate rotation
        self.wait(12)

        # Add spring
        spring = SrpingMassSystem(
            equilibrium_position=planes[0].get_bottom() + DOWN,
            equilibrium_length=3,
            n_spring_curls=8,
            mass_width=0.5,
            spring_radius=0.2,
        )
        spring.pause()
        unit_size = planes[0].x_axis.get_unit_size()
        spring.add_updater(lambda m: m.set_x(unit_size * get_x()))

        v_line = Line()
        v_line.set_stroke(BLUE_A, 2)
        v_line.f_always.put_start_and_end_on(spring.mass.get_top, left_vector.get_end)

        self.play(VFadeIn(spring), VFadeIn(v_line))
        self.wait(20)
        self.play(
            VFadeOut(spring),
            VFadeOut(v_line),
            VFadeOut(tails),
        )

        # Add them up
        new_plane_center = planes.get_center()
        shift_factor = ValueTracker(0)
        right_vector.add_updater(lambda m: m.shift(shift_factor.get_value() * left_vector.get_vector()))

        sum_expr = VGroup(titles[0], Tex(R"+"), titles[1])
        sum_expr.target = sum_expr.generate_target()
        sum_expr.target.arrange(RIGHT, buff=MED_SMALL_BUFF, aligned_edge=DOWN)
        sum_expr.target.next_to(planes, UP, MED_SMALL_BUFF)
        sum_expr[1].set_opacity(0).next_to(planes, UP)

        result_dot = GlowDot()
        result_dot.f_always.move_to(right_vector.get_end)

        self.play(
            planes[0].animate.move_to(new_plane_center),
            planes[1].animate.move_to(new_plane_center).set_opacity(0),
            MoveToTarget(sum_expr),
            run_time=2,
        )
        self.play(shift_factor.animate.set_value(1))
        self.play(FadeIn(result_dot))
        self.wait(4)

        # Add another spring
        spring = SrpingMassSystem(
            equilibrium_position=planes[0].get_bottom() + DOWN,
            equilibrium_length=5,
            n_spring_curls=8,
            mass_width=0.5,
            spring_radius=0.2,
        )
        spring.pause()
        unit_size = planes[0].x_axis.get_unit_size()
        spring.add_updater(lambda m: m.set_x(2 * unit_size * get_x()))

        v_line = Line()
        v_line.set_stroke(BLUE_A, 2)
        v_line.f_always.put_start_and_end_on(spring.mass.get_top, result_dot.get_center)

        self.play(VFadeIn(spring), VFadeIn(v_line))
        self.wait(2)

        # Right hand side
        rhs = Tex(R"= 2 \\cos(\\omega t)", t2c={R"\\omega": PINK})
        rhs.next_to(sum_expr, RIGHT, buff=MED_SMALL_BUFF).shift(SMALL_BUFF * DOWN)

        self.play(Write(rhs))
        self.wait(20)

    def get_rotating_vector(self, plane, s, t_tracker, color=TEAL, thickness=3):
        """Create a rotating vector for e^(st) on the given plane"""
        def update_vector(vector):
            t = t_tracker.get_value()
            c = vector.coef_tracker.get_value()
            z = c * np.exp(s * t)
            vector.put_start_and_end_on(plane.n2p(0), plane.n2p(z))

        vector = Arrow(LEFT, RIGHT, fill_color=color, thickness=thickness)
        vector.coef_tracker = ComplexValueTracker(1)
        vector.add_updater(update_vector)

        return vector


class SimpleSolutionSummary(InteractiveScene):
    def construct(self):
        # Summary of the "strategy" up top
        t2c = {"m": RED, "k": TEAL, "{s}": YELLOW, R"\\omega": PINK}
        kw = dict(t2c=t2c, font_size=36)
        arrow = Vector(1.5 * RIGHT)
        top_eq = VGroup(
            Tex(R"m x''(t) + k x(t) = 0", **kw),
            arrow,
            Tex(R"e^{{s}t}\\left(m{s}^2 + k\\right) = 0", **kw),
            Tex(R"\\Longrightarrow", **kw),
            Tex(R"{s} = \\pm i \\underbrace{\\sqrt{k / m}}_{\\omega}", **kw),
        )
        top_eq.arrange(RIGHT)
        top_eq[-1].align_to(top_eq[2], UP)
        guess = Tex(R"\\text{Guess } e^{{s}t}", **kw)
        guess.scale(0.75)
        guess.next_to(arrow, UP, buff=0)
        arrow.add(guess)

        top_eq.set_width(FRAME_WIDTH - 1)

        top_eq.center().to_edge(UP)
        self.add(top_eq)


class ShowFamilyOfComplexSolutions(RotatingExponentials):
    tex_to_color_map = {R"\\omega": PINK}
    plane_config = dict(
        background_line_style=dict(stroke_color=BLUE, stroke_width=1),
        faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.25),
    )
    vect_colors = [TEAL, RED]
    rotation_frequency = TAU / 4

    def construct(self):
        # Show the equation
        frame = self.frame
        frame.set_x(-10)

        colors = get_coef_colors()
        t2c = {"x''(t)": colors[2], "x'(t)": colors[1], "x(t)": colors[0]}
        equation = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c, font_size=42)
        equation.next_to(frame.get_left(), RIGHT, buff=1.0)

        arrow = Vector(3.0 * RIGHT, thickness=6, fill_color=GREY_B)
        arrow.next_to(equation, RIGHT, MED_LARGE_BUFF)

        strategy_words = VGroup(
            Text("“Strategy”"),
            TexText(R"Guess $e^{{s}t}$", t2c={R"{s}": YELLOW}, font_size=36, fill_color=GREY_A)
        )
        strategy_words.arrange(DOWN)
        strategy_words.next_to(arrow, UP, MED_SMALL_BUFF)

        self.add(equation)
        self.play(
            GrowArrow(arrow),
            FadeIn(strategy_words, lag_ratio=0.1)
        )
        self.wait()

        # Show two basis solutions on the left
        t2c = self.tex_to_color_map
        left_planes, left_plane_labels = self.get_left_planes(label_texs=[R"e^{+i\\omega t}", R"e^{-i\\omega t}"])
        rot_vects, tails, t_tracker = self.get_rot_vects(left_planes)
        left_planes_brace = Brace(left_planes, LEFT, MED_SMALL_BUFF)

        self.add(rot_vects, tails)
        self.add(t_tracker)
        self.play(
            GrowFromCenter(left_planes_brace),
            FadeIn(left_planes),
            FadeTransform(strategy_words[1][R"e^{{s}t}"].copy(), left_plane_labels[0]),
            FadeTransform(strategy_words[1][R"e^{{s}t}"].copy(), left_plane_labels[1]),
            VFadeIn(rot_vects),
        )
        self.wait(3)

        self.wait(8)

        # Show combination with tunable parameters
        right_plane = self.get_right_plane()
        right_plane.next_to(left_planes, RIGHT, buff=1.5)

        scaled_solution = Tex(
            R"c_1 e^{+i\\omega t} + c_2 e^{-i\\omega t}",
            t2c={R"\\omega": PINK, "c_1": BLUE, "c_2": BLUE}
        )
        scaled_solution.next_to(right_plane, UP)

        vect1, vect2 = right_rot_vects = self.get_rot_vect_sum(right_plane, t_tracker)
        c1_eq, c2_eq = coef_eqs = VGroup(
            VGroup(Tex(fR"c_{n} = "), DecimalNumber(1))
            for n in [1, 2]
        )
        coef_eqs.scale(0.85)
        for coef_eq in coef_eqs:
            coef_eq.arrange(RIGHT, buff=SMALL_BUFF)
            coef_eq[1].align_to(coef_eq[0][0], DOWN)
            coef_eq[0][:2].set_fill(BLUE)
        coef_eqs.arrange(DOWN, MED_LARGE_BUFF)
        coef_eqs.to_corner(UR)
        coef_eqs.shift(LEFT)

        c1_eq[1].add_updater(lambda m: m.set_value(vect1.coef_tracker.get_value()))
        c2_eq[1].add_updater(lambda m: m.set_value(vect2.coef_tracker.get_value()))

        self.play(
            FadeIn(right_plane),
            FadeOut(left_planes_brace),
            frame.animate.center(),
            run_time=2
        )
        self.play(LaggedStart(
            FadeTransform(left_plane_labels[0].copy(), scaled_solution[R"e^{+i\\omega t}"]),
            FadeIn(scaled_solution[R"c_1"]),
            TransformFromCopy(rot_vects[0], right_rot_vects[0], suspend_mobject_updating=True),
            FadeTransform(left_plane_labels[1].copy(), scaled_solution[R"e^{-i\\omega t}"]),
            FadeIn(scaled_solution[R"+"][1]),
            FadeIn(scaled_solution[R"c_2"]),
            TransformFromCopy(rot_vects[1], right_rot_vects[1], suspend_mobject_updating=True)
        ))
        self.play(LaggedStart(
            FadeTransformPieces(scaled_solution[R"c_1"].copy(), c1_eq),
            FadeTransformPieces(scaled_solution[R"c_2"].copy(), c2_eq)
        ))
        self.play(LaggedStart(
            vect1.coef_tracker.animate.set_value(2),
            vect2.coef_tracker.animate.set_value(0.5),
            lag_ratio=0.5
        ))

        comb_tail = TracingTail(vect2.get_end, stroke_color=YELLOW, time_traced=2)
        glow_dot = GlowDot()
        glow_dot.f_always.move_to(vect2.get_end)
        self.add(comb_tail)
        self.play(FadeIn(glow_dot))

        self.wait(6)
        self.play(LaggedStart(
            vect1.coef_tracker.animate.set_value(complex(1.5, 1)),
            vect2.coef_tracker.animate.set_value(complex(0.5, -1.25)),
        ))
        self.wait(7)

        # Change the coefficients
        t_tracker.suspend_updating()
        self.play(
            FadeOut(comb_tail, suspend_mobject_updating=True),
            LaggedStart(
                vect1.coef_tracker.animate.set_value(complex(0.31, -0.41)),
                vect2.coef_tracker.animate.set_value(complex(2.71, -0.82)),
            ),
        )
        self.wait()
        self.play(
            LaggedStart(
                vect1.coef_tracker.animate.set_value(complex(-1.03, 0.5)),
                vect2.coef_tracker.animate.set_value(complex(1.5, 0.35)),
            ),
        )
        self.add(comb_tail)
        self.wait(2)
        t_tracker.resume_updating()

        # Zoom out
        self.play(frame.animate.set_height(13.75, about_edge=RIGHT), run_time=2)
        self.wait(4)
        self.play(frame.animate.to_default_state(), run_time=2)

        # Go to real valued
        self.play(
            LaggedStart(
                vect1.coef_tracker.animate.set_value(1),
                vect2.coef_tracker.animate.set_value(1),
            ),
        )
        self.wait(6)

        # Show initial conditions
        initial_conditions = VGroup(
            Tex(R"x_0 = 0.00"),
            Tex(R"v_0 = 0.00"),
        )
        x0_value = initial_conditions[0].make_number_changeable("0.00")
        v0_value = initial_conditions[1].make_number_changeable("0.00")
        x0_value.set_value(2)
        initial_conditions.scale(0.85)
        initial_conditions.arrange(DOWN)
        initial_conditions.move_to(coef_eqs, LEFT)
        initial_conditions.to_edge(UP)
        implies = Tex(R"\\Downarrow", font_size=72)
        implies.next_to(initial_conditions, DOWN)

        t_tracker.suspend_updating()
        t_tracker.set_value((t_tracker.get_value() + 2) % 4 - 2)
        self.play(
            FadeIn(initial_conditions),
            Write(implies),
            coef_eqs.animate.next_to(implies, DOWN).align_to(initial_conditions, LEFT),
        )
        self.remove(comb_tail)
        self.play(
            vect1.coef_tracker.animate.set_value(1),
            vect2.coef_tracker.animate.set_value(1),
            t_tracker.animate.set_value(0),
        )
        self.wait()
        self.remove(comb_tail)

        # Highlight values, rise
        t_tracker.resume_updating()

        highlight_rect = SurroundingRectangle(initial_conditions[0])
        highlight_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(highlight_rect))
        self.wait()
        self.play(highlight_rect.animate.surround(initial_conditions[1]))
        self.wait(2)
        self.play(highlight_rect.animate.surround(coef_eqs))
        self.wait(4)

        self.play(
            vect1.coef_tracker.animate.set_value(1.5),
            vect2.coef_tracker.animate.set_value(1.5),
            ChangeDecimalToValue(x0_value, 3)
        )
        self.wait(12)

    def get_left_planes(self, label_texs: list[str]):
        planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1), **self.plane_config)
            for _ in range(2)
        )
        planes.arrange(DOWN, buff=1.0)
        planes.set_height(6.5)
        planes.to_corner(DL)
        planes.set_z_index(-1)

        labels = VGroup(Tex(tex, t2c=self.tex_to_color_map) for tex in label_texs)
        for label, plane in zip(labels, planes):
            label.next_to(plane, UP, SMALL_BUFF)

        return planes, labels

    def get_rot_vects(self, planes):
        t_tracker = ValueTracker(0)
        t_tracker.add_updater(lambda m, dt: m.increment_value(dt))

        rot_vects = VGroup(
            self.get_rotating_vector(plane, u * 1j * self.rotation_frequency, t_tracker, color)
            for plane, u, color in zip(planes, [+1, -1], self.vect_colors)
        )
        tails = VGroup(
            TracingTail(vect.get_end, stroke_color=vect.get_color(), time_traced=2)
            for vect in rot_vects
        )

        return Group(rot_vects, tails, t_tracker)

    def get_rot_vect_sum(self, plane, t_tracker):
        vect1, vect2 = vect_sum = VGroup(
            self.get_rotating_vector(
                plane,
                u * 1j * self.rotation_frequency,
                t_tracker,
                color,
            )
            for u, color in zip([+1, -1], self.vect_colors)
        )
        vect2.add_updater(lambda m: m.put_start_on(vect1.get_end()))
        return vect_sum

    def get_right_plane(self, x_range=(-3, 3), height=5.5):
        right_plane = ComplexPlane(x_range, x_range, **self.plane_config)
        right_plane.set_height(height)
        return right_plane

    def add_scale_tracker(vector, initial_value=1):
        """
        Assumes the vector has another updater constantly setting a location in the plane
        """
        vector.c_tracker = ComplexValueTracker(initial_value)

        def update_vector(vect):
            c = vect.c_tracker.get_value()
            vect.scale()
            pass


class GuessSine(InteractiveScene):
    func_name = R"\\sin"

    def construct(self):
        # Set up
        self.frame.set_height(9, about_edge=LEFT)
        func_name = self.func_name
        func_tex = Rf"{func_name}(\\omega t)"

        t2c = {R"\\omega": PINK, "x(t)": TEAL, "x''(t)": RED}
        equation = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c)
        equation.to_edge(LEFT)
        guess_words = TexText(Rf"Try $x(t) = {func_tex}$", t2c=t2c, font_size=36)

        arrow = Arrow(guess_words.get_left(), guess_words.get_right(), buff=-0.1, thickness=6)
        arrow.next_to(equation, RIGHT)
        guess_words.next_to(arrow, UP, SMALL_BUFF)

        self.add(equation)
        self.add(arrow)
        self.add(guess_words)

        # Sub
        sub = Tex(fR"-m \\omega^2 {func_tex}  + k {func_tex} = 0", t2c=t2c)
        sub.next_to(arrow, RIGHT)
        simple_sub = Tex(Rf"\\left(-m \\omega^2 + k\\right) {func_tex} = 0", t2c=t2c)
        simple_sub.next_to(arrow, RIGHT)
        implies = Tex(R"\\Rightarrow", font_size=72)
        implies.next_to(simple_sub, RIGHT)

        t2c[R"\\ding{51}"] = GREEN
        t2c["Valid"] = GREEN
        result = TexText(R"\\ding{51} Valid if $\\omega = \\sqrt{k / m}$", t2c=t2c, font_size=36)
        result.next_to(simple_sub, DOWN)

        simple_sub.shift(0.05 * UP)
        blank_sub = sub.copy()
        blank_sub[func_tex].set_opacity(0)
        func_parts = sub[func_tex]

        self.play(LaggedStart(
            TransformMatchingTex(equation.copy(), blank_sub, path_arc=30 * DEG, run_time=2),
            FadeTransform(guess_words[func_tex][0].copy(), func_parts[0]),
            FadeTransform(guess_words[func_tex][0].copy(), func_parts[1]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(
            TransformMatchingTex(blank_sub, simple_sub),
            Transform(func_parts[0], simple_sub[func_tex][0], path_arc=-30 * DEG),
            Transform(func_parts[1], simple_sub[func_tex][0], path_arc=-30 * DEG),
            run_time=1
        )
        self.wait()
        self.play(FadeIn(result, 0.5 * UP))
        self.wait()


class GuessCosine(GuessSine):
    func_name = R"\\cos"


class ShowFamilyOfRealSolutions(InteractiveScene):
    t2c = {R"\\omega": PINK}
    omega = PI
    x_max = 10

    def construct(self):
        # Add cosine and sine graphs up top
        cos_graph, sin_graph = small_graphs = VGroup(
            self.get_small_graph(math.cos, R"\\cos(\\omega t)", BLUE),
            self.get_small_graph(math.sin, R"\\sin(\\omega t)", RED),
        )
        small_graphs.arrange(RIGHT, buff=LARGE_BUFF)
        small_graphs.to_edge(UP, buff=MED_SMALL_BUFF)

        self.add(small_graphs)

        # Add master graph
        coef_trackers = ValueTracker(1).replicate(2)

        def func(t):
            c1 = coef_trackers[0].get_value()
            c2 = coef_trackers[1].get_value()
            return c1 * np.cos(self.omega * t) + c2 * np.sin(self.omega * t)

        axes = Axes((0, self.x_max), (-3, 3), width=self.x_max, height=4)
        axes.to_edge(DOWN)
        graph_label = Tex(R"+1.00 \\cos(\\omega t) +1.00 \\sin(\\omega t)", t2c=self.t2c)
        graph_label.next_to(axes.y_axis.get_top(), UR)
        coef_labels = graph_label.make_number_changeable("+1.00", replace_all=True, edge_to_fix=RIGHT, include_sign=True)
        coef_labels.set_color(YELLOW)
        coef_labels[0].add_updater(lambda m: m.set_value(coef_trackers[0].get_value()))
        coef_labels[1].add_updater(lambda m: m.set_value(coef_trackers[1].get_value()))

        graph = axes.get_graph(func)
        graph.set_stroke(TEAL, 5)
        axes.bind_graph_to_func(graph, func)

        self.add(axes)
        self.add(graph_label)
        self.add(graph)

        # Tweak the parameters
        for c1, c2 in [(-0.5, 2), (1.5, 0), (0.25, -2)]:
            self.play(LaggedStart(
                coef_trackers[0].animate.set_value(c1),
                coef_trackers[1].animate.set_value(c2),
                lag_ratio=0.5
            ))
            self.wait()

        # Tweak with spring
        for c1, c2 in [(1.6, 1.8), (-2.7, 0.18), (0.5, -2)]:
            self.play(LaggedStart(
                coef_trackers[0].animate.set_value(c1),
                coef_trackers[1].animate.set_value(c2),
                lag_ratio=0.5
            ))
            self.show_spring(axes, graph, func)

    def get_small_graph(self, func, func_name, color):
        axes = Axes((0, 6), (-2, 2), height=2, width=6)
        graph = axes.get_graph(lambda t: func(self.omega * t))
        graph.set_stroke(color, 3)
        label = Tex(func_name, t2c=self.t2c, font_size=36)
        label.move_to(axes, UP)

        return VGroup(axes, graph, label)

    def show_spring(self, axes, graph, func):
        graph_copy = graph.copy()
        graph_copy.clear_updaters()

        spring = SrpingMassSystem(
            equilibrium_length=4,
            equilibrium_position=axes.get_right() + RIGHT,
            direction=UP,
            mass_width=0.75,
            spring_radius=0.2,
        )
        spring.add_updater(
            lambda m: m.set_x(axes.y_axis.get_unit_size() * axes.y_axis.p2n(graph_copy.get_end()))
        )

        h_line = Line()
        h_line.set_stroke(WHITE, 1)
        h_line.f_always.put_start_and_end_on(
            spring.mass.get_left,
            graph_copy.get_end,
        )

        self.play(
            graph.animate.set_stroke(opacity=0.2),
            ShowCreation(graph_copy, rate_func=linear, run_time=self.x_max),
            VFadeIn(spring, run_time=1),
            VFadeIn(h_line, run_time=1),
        )
        self.play(FadeOut(spring), FadeOut(h_line))
        self.wait()

        self.remove(graph_copy, spring, h_line)
        graph.set_stroke(opacity=1)


class SetOfInitialConditions(InteractiveScene):
    graph_time = 8

    def construct(self):
        # Set up all boxes
        frame = self.frame
        box = Rectangle(width=3.5, height=2.0)
        box.set_stroke(WHITE, 1)
        v_line = DashedLine(box.get_top(), box.get_bottom())
        v_line.set_stroke(GREY_C, 1, 0.5)
        v_line.scale(0.9)
        box.add(v_line)

        n_rows = 5
        n_cols = 5
        box_row = box.get_grid(1, n_cols, buff=0)
        box_grid = box_row.get_grid(n_rows, 1, buff=0)
        for row, v0 in zip(box_grid, np.linspace(1, -1, n_rows)):
            for box, x0 in zip(row, np.linspace(-1, 1, n_cols)):
                box.spring = self.get_spring_in_a_box(box, x0=x0, v0=v0)

        # Show the first example
        mid_row = box_grid[n_rows // 2]
        x0_labels = VGroup(
            Tex(Rf"x_0 = {x0}", font_size=48).next_to(box, UP, SMALL_BUFF)
            for x0, box in zip(range(-2, 3), mid_row)
        )
        mid_row_springs = VGroup(box.spring for box in mid_row)

        mid_row_solutions = VGroup(
            self.get_solution_graph(box, x0=x0)
            for box, x0 in zip(mid_row, range(-2, 3))
        )
        last_solution = mid_row_solutions[-1]

        last_box = mid_row[-1]
        last_spring = last_box.spring
        top_line = Line(last_box.get_center(), last_spring.mass.get_center())
        top_line.set_y(last_spring.mass.get_y(UP))
        brace = LineBrace(Line(ORIGIN, 1.5 * RIGHT))
        brace.match_width(top_line)
        brace.next_to(top_line, UP, SMALL_BUFF)
        brace.stretch(0.5, 1, about_edge=DOWN)
        last_x = last_spring.get_x()
        last_spring.set_x(0)

        self.add(last_box, last_spring)
        frame.move_to(mid_row[-1]).set_height(5)

        self.play(
            GrowFromPoint(brace, brace.get_left()),
            last_spring.animate.set_x(last_x),
            Write(x0_labels[-1]),
        )
        self.wait()
        last_spring.unpause()
        last_spring.v_vect.set_opacity(0)
        self.play(
            frame.animate.reorient(0, 0, 0, (6.7, -0.61, 0.0), 6.16).set_anim_args(run_time=2),
            FadeIn(last_solution[0]),
            ShowCreation(last_solution[1], rate_func=linear, run_time=self.graph_time)
        )
        last_spring.pause()

        # Show the full middle row
        self.play(
            frame.animate.center().set_width(box_grid.get_width() + 2).set_anim_args(time_span=(0, 1.5)),
            last_spring.animate.set_x(1).set_anim_args(time_span=(0, 1)),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row_springs[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(x0_labels[:-1])), lag_ratio=0.75),
            LaggedStartMap(FadeIn, VGroup(*reversed(mid_row_solutions[:-1])), lag_ratio=0.75),
            run_time=3
        )
        self.wait()

        graphs = VGroup(solution[1] for solution in mid_row_solutions)
        faint_graphs = graphs.copy().set_stroke(width=1, opacity=0.25)
        for spring in mid_row_springs:
            spring.unpause()
            spring.v_vect.set_opacity(0)
            spring.x0 = spring.get_x()
        self.add(faint_graphs)
        self.play(
            ShowCreation(graphs, lag_ratio=0, run_time=self.graph_time, rate_func=linear),
        )
        self.wait()
        self.remove(faint_graphs)
        for spring in mid_row_springs:
            spring.pause()
            spring.v_vect.set_opacity(0)
            spring.set_velocity(0)
        self.play(
            FadeOut(mid_row_solutions),
            *(spring.animate.set_x(spring.x0) for spring in mid_row_springs)
        )

        # Show initial velocities
        v0_labels = VGroup(
            Tex(Rf"v_0 = {v0}", font_size=48).next_to(row, LEFT).set_fill(TEAL)
            for v0, row in zip(range(2, -3, -1), box_grid)
        )
        other_indices = [0, 1, 3, 4]
        row_springs = VGroup(VGroup(box.spring for box in row) for row in box_grid)

        self.play(
            frame.animate.set_width(box_grid.get_width() + 3, about_edge=DR),
            Write(v0_labels[2])
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, VGroup(box_grid[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            LaggedStartMap(FadeIn, VGroup(v0_labels[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            LaggedStartMap(FadeIn, VGroup(row_springs[i] for i in other_indices), lag_ratio=0.5, run_time=3),
            x0_labels.animate.next_to(box_grid, UP, SMALL_BUFF).set_anim_args(run_time=1),
            FadeOut(brace),
        )
        self.wait()

        # Let it play
        for row in box_grid:
            for box in row:
                box.spring.unpause()

        self.wait(20)

        # Add graphs
        all_solutions = VGroup(
            self.get_solution_graph(box, x0=x0, v0=v0, graph_color=YELLOW).move_to(box).scale(0.8)
            for row, v0 in zip(box_grid, range(2, -3, -1))
            for box, x0 in zip(row, range(-2, 3))
        )
        all_axes = VGroup(s[0] for s in all_solutions)
        all_graphs = VGroup(s[1] for s in all_solutions)

        self.remove(*(box.spring for row in box_grid for box in row))
        self.add(all_axes)
        self.play(ShowCreation(all_graphs, lag_ratio=1e-1, run_time=3))
        self.wait()

        # Highlight one
        highlight = all_solutions[14].copy()
        self.add(highlight)
        self.play(all_solutions.animate.fade(0.75))
        self.wait()

    def get_spring_in_a_box(self, box, x0=0, v0=0, k=9, mu=0.5):
        box_width = box.get_width()
        spring = SrpingMassSystem(
            x0=x0,
            v0=v0,
            k=k,
            mu=mu,
            mass_width=0.1 * box_width,
            equilibrium_length=0.5 * box_width,
            equilibrium_position=box.get_center(),
            spring_radius=0.035 * box_width,
        )
        v_vect = spring.get_velocity_vector(v_offset=-box_width * 0.1, scale_factor=0.5)
        spring.add(v_vect)
        spring.v_vect = v_vect
        spring.pause()
        return spring

    def get_solution_graph(self, box, x0=2, v0=0, k=9, mu=0.5, width_factor=0.8, graph_color=TEAL):
        axes = Axes(
            x_range=(0, self.graph_time, 1),
            y_range=(-2, 2),
            width=width_factor * box.get_width(),
            height=box.get_height(),
        )
        axes.set_stroke(GREY, 1)
        axes.next_to(box, DOWN)

        s = 0.5 * (-mu + 1j * math.sqrt(4 * k - mu**2))
        z0 = complex(
            x0,
            (s.real * x0 - v0) / s.imag
        )

        graph = axes.get_graph(
            lambda t: (z0 * np.exp(s * t)).real,
        )
        graph.set_stroke(graph_color, 2)
        return VGroup(axes, graph)


# For ODE video


class SpringInTheWind(InteractiveScene):
    F_0 = 1.0
    omega = 2
    k = 3
    mu = 0.1

    def setup(self):
        super().setup()
        # Set up wind

        plane = NumberPlane()
        wind = TimeVaryingVectorField(
            lambda p, t: np.tile(self.external_force(t) * RIGHT, (len(p), 1)),
            plane,
            density=1,
            color=WHITE,
            stroke_width=6,
            stroke_opacity=0.5,
        )

        # Set up spring and ODE
        spring = SrpingMassSystem(
            k=self.k,
            mu=self.mu,
            external_force=lambda: self.external_force(wind.time)
        )
        spring.add_external_force(lambda: self.F_0 * math.cos(self.omega * wind.time))

        self.add(wind)
        self.add(spring)

        self.spring = spring
        self.wind = wind

    def external_force(self, time):
        return self.F_0 * math.cos(self.omega * time)

    def construct(self):
        spring = self.spring
        wind = self.wind
        self.play(VFadeIn(wind))
        self.wait(60)


class ShowSpringInWindGraph(SpringInTheWind):
    mu = 0.25
    k = 3
    omega = 2.5

    def construct(self):
        spring = self.spring
        wind = self.wind

        # Add graph
        t_max = 40
        frame = self.frame
        frame.set_y(1)
        graph_block = Rectangle(width=FRAME_WIDTH, height=2.5)
        graph_block.move_to(frame, UP)
        graph_block.set_stroke(width=0)
        graph_block.set_fill(BLACK, 1)

        axes = Axes((0, t_max), (-0.5, 0.5, 0.25), width=FRAME_WIDTH - 1, height=2.0)
        axes.x_axis.ticks.stretch(0.5, 1)
        axes.move_to(graph_block)
        axis_label = Text("Time", font_size=24)
        axis_label.next_to(axes.x_axis.get_right(), DOWN)
        axes.add(axis_label)

        graph = TracedPath(
            lambda: axes.c2p(self.time, spring.get_x()),
            stroke_color=BLUE,
            stroke_width=3,
        )

        self.add(graph_block)
        self.add(axes)
        self.add(graph)

        # Play it out
        self.wait(t_max)
        graph.clear_updaters()
        self.play(
            VFadeOut(spring),
            VFadeOut(wind),
        )

        # Comment on the graph
        left_highlight = graph_block.copy()
        left_highlight.set_width(6.5, stretch=True, about_edge=LEFT)
        left_highlight.set_fill(YELLOW, 0.2)
        left_highlight.set_height(2, stretch=True)
        right_highlight = left_highlight.copy()
        right_highlight.set_width(10, stretch=True, about_edge=LEFT)
        right_highlight.next_to(left_highlight, RIGHT, buff=0)
        right_highlight.set_fill(GREEN, 0.2)

        self.play(FadeIn(left_highlight))
        self.wait()
        self.play(FadeIn(right_highlight))
        self.wait()

        # Show solution components
        axes1, axes2 = axes_copies = VGroup(axes.deepcopy() for _ in range(2))
        buff = 0.75
        axes_copies.arrange(DOWN, buff=buff)
        axes_copies.next_to(axes, DOWN, buff=buff)

        s_root = (-self.mu + 1j * math.sqrt(-(self.mu**2 - 4 * self.k))) / 2.0
        amp = 0.35
        shm_graph = axes1.get_graph(lambda t: amp * np.exp(s_root * t).real)
        shm_graph.set_stroke(GREEN, 4)
        cos_graph = axes2.get_graph(lambda t: -amp * math.cos(self.omega * t))
        cos_graph.set_stroke(YELLOW, 4)

        equals = Tex(R"=", font_size=72).rotate(90 * DEG)
        equals.move_to(VGroup(axes, axes1)).set_x(-5)
        equals.shift(0.25 * DOWN)
        plus = Tex(R"+", font_size=72)
        plus.move_to(VGroup(axes_copies)).match_x(equals)

        self.play(LaggedStart(
            Write(equals),
            FadeIn(axes1),
            FadeIn(shm_graph),
            Write(plus),
            FadeIn(axes2),
            FadeIn(cos_graph),
            lag_ratio=0.15
        ))
        self.wait()

        # First graph label
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        shm_eq = VGroup(
            Text("Solution to", font_size=36),
            Tex(
                R"mx''(t) + \\mu x'(t) + k x(t) = 0",
                t2c={
                    "x(t)": colors[0],
                    "x'(t)": colors[1],
                    "x''(t)": colors[2],
                },
                font_size=36,
                alignment="",
            ),
        )
        shm_eq.arrange(DOWN)
        shm_eq.next_to(axes1.x_axis, UP, buff=0.35)
        shm_eq.set_x(-2.5)

        self.add(shm_graph.copy().set_stroke(opacity=0.25))
        self.play(
            FadeIn(shm_eq, lag_ratio=0.1),
            ShowCreation(shm_graph, run_time=3, rate_func=linear),
        )
        self.wait()

        # Grow left highlight
        self.play(
            left_highlight.animate.set_height(8, stretch=True, about_edge=UP).shift(0.15 * UP),
            shm_eq.animate.shift(5 * RIGHT),
            run_time=2,
        )
        self.wait()

        # Draw all graphs
        self.add(graph.copy().set_stroke(opacity=0.25))
        self.add(cos_graph.copy().set_stroke(opacity=0.25))
        self.play(
            left_highlight.animate.set_fill(opacity=0.1),
            *(
                ShowCreation(mob, run_time=15, rate_func=linear)
                for mob in [graph, shm_graph, cos_graph]
            ),
        )`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      6: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      7: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      11: "SrpingMassSystem extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      34: "Computes the angle between two vectors using the dot product formula: cos(θ) = (a·b)/(|a||b|).",
      47: "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion.",
      50: "ParametricCurve traces a function f(t) → (x,y,z) over a parameter range, producing a smooth 3D curve.",
      63: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      64: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      73: "Returns the Euclidean length of a vector. ManimGL utility wrapping np.linalg.norm.",
      124: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      134: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      150: "BasicSpringScene extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      151: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      180: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      182: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      184: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      185: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      188: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      189: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      190: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      191: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      192: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      193: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      195: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      202: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      203: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      204: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      207: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      210: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      213: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      215: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      216: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      217: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      218: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      226: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      228: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      229: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      232: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      233: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      234: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      245: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      250: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      251: "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion.",
      253: "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
      263: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      264: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      269: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      270: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      281: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      282: "Smoothly animates the camera to a new orientation over the animation duration.",
      283: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      285: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      289: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      291: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      294: "DampingForceDemo extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      295: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      312: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      315: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      333: "SolveDampedSpringEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      334: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      768: "DampedSpringSolutionsOnSPlane extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      769: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      951: "RotatingExponentials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      952: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1106: "SimpleSolutionSummary extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1107: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1132: "Class ShowFamilyOfComplexSolutions inherits from RotatingExponentials.",
      1141: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1398: "GuessSine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1401: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1456: "Class GuessCosine inherits from GuessSine.",
      1460: "ShowFamilyOfRealSolutions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1465: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1563: "SetOfInitialConditions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1566: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1753: "SpringInTheWind extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1790: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1797: "Class ShowSpringInWindGraph inherits from SpringInTheWind.",
      1802: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/laplace/supplements.py"] = {
    description: "General supplementary scenes for the Laplace transform series: extra examples, edge cases, and bonus demonstrations.",
    code: `from manim_imports_ext import *


class IntroduceTrilogy(InteractiveScene):
    def construct(self):
        # Add definition
        self.add(FullScreenRectangle().fix_in_frame())
        frame = self.frame
        name = Text("Laplace Transform", font_size=60)
        name.to_edge(UP)
        t2c = {"s": YELLOW, R"{t}": BLUE}
        laplace = Tex(R"F(s) = \\int_0^\\infty f({t}) e^{\\minus s{t}} d{t}", font_size=36, t2c=t2c)
        laplace.next_to(name, DOWN)

        frames = Square().replicate(3)
        frames.set_stroke(WHITE, 1).set_fill(BLACK, 1)
        frames.set_width(0.3 * FRAME_WIDTH)
        frames.arrange(RIGHT, buff=MED_LARGE_BUFF)
        frames.set_y(-1.0)
        frames.fix_in_frame()
        name.fix_in_frame()

        frame.match_x(laplace["f({t})"])

        self.play(
            Write(name),
            FadeIn(frames, lag_ratio=0.25, run_time=2)
        )
        self.play(
            Write(laplace["f({t})"]),
        )
        self.play(
            Write(laplace[R"e^{\\minus s"]),
            TransformFromCopy(*laplace[R"{t}"][0:2]),
            frame.animate.match_x(laplace[R"f({t}) e^{\\minus s{t}}"])
        )
        self.play(
            FadeIn(laplace[R"\\int_0^\\infty"], shift=0.25 * RIGHT, scale=1.5),
            FadeIn(laplace[R"d{t}"], shift=0.25 * LEFT, scale=1.5),
        )
        self.play(
            FadeTransform(laplace["f("].copy(), laplace["F("], path_arc=-PI / 2),
            TransformFromCopy(laplace[")"][1], laplace[")"][0], path_arc=-PI / 2),
            TransformFromCopy(laplace["s"][1], laplace["s"][0], path_arc=-PI / 4),
            Write(laplace["="]),
            frame.animate.center(),
        )
        self.wait()

        # Contrast the pair
        brace = Brace(frames[1:], UP)
        brace_ghost = brace.copy().set_fill(GREY_D)

        ilp = Tex(R"f({t}) = \\frac{1}{2\\pi i} \\int_{a - i \\infty}^{a + i \\infty} F(s) e^{s{t}} d{s}", t2c=t2c, font_size=36)
        ilp.scale(0.75)
        ilp.next_to(frames[2], UP, MED_LARGE_BUFF)

        self.play(
            GrowFromCenter(brace),
            laplace.animate.scale(0.75).next_to(frames[1], UP, MED_LARGE_BUFF),
            name.animate.scale(0.75).next_to(frames[1:], UP, buff=1.75),
        )
        self.play(TransformMatchingTex(
            laplace.copy(), ilp,
            lag_ratio=1e-2,
            path_arc=-20 * DEG,
            matched_keys=["f({t})", "F(s)", "e^{s{t}}", R"\\int"],
            key_map={"d{t}": "d{s}"},
        ))
        self.wait()
        self.add(brace_ghost, brace)
        self.play(
            brace.animate.match_width(frames[0], stretch=True).next_to(frames[0], UP).set_anim_args(path_arc=15 * DEG)
        )
        self.wait()

        # Clear the board
        self.play(
            LaggedStartMap(FadeOut, VGroup(*frames[1:], brace, brace_ghost, name, laplace, ilp), shift=RIGHT, lag_ratio=0.15),
            frames[0].animate.set_shape(16, 9).set_width(0.45 * FRAME_WIDTH).move_to(FRAME_WIDTH * LEFT / 4 + 0.5 * DOWN),
            run_time=2
        )
        self.wait()

        # Add new frame
        new_frame = frames[0].copy()
        new_frame.set_x(FRAME_WIDTH / 4)
        arc = -120 * DEG
        arrow = Arrow(frames[0].get_top(), new_frame.get_top(), path_arc=arc, thickness=6)
        arrow_line = Line(frames[0].get_top(), new_frame.get_top(), path_arc=arc, buff=0.2)
        arrow_line.pointwise_become_partial(arrow_line, 0, 0.95)
        arrow.set_fill(TEAL)
        arrow_line.set_stroke(TEAL, 10)
        self.play(
            ShowCreation(arrow_line, time_span=(0, 0.8)),
            FadeIn(arrow, time_span=(0.7, 1)),
            FadeIn(new_frame)
        )

    def old_material(self):
        # Show trilogy
        background = FullScreenRectangle().set_fill(GREY_E, 1)
        screens = ScreenRectangle().replicate(3)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        screens[1].save_state()
        screens[1].replace(background)
        screens[1].set_stroke(width=0)
        screens.set_stroke(behind=True)

        terms = VGroup(name, laplace)

        self.add(background, screens, terms)
        self.play(
            FadeIn(background),
            Restore(screens[1]),
            terms.animate.scale(0.4).move_to(screens[1].saved_state),
        )
        self.wait()

        # Inverse
        ilp = VGroup(
            Text("Inverse Laplace Transform"),
            Tex(R"f({t}) = \\frac{1}{2\\pi i} \\int_{a - i \\infty}^{a + i \\infty} F(s) e^{s{t}} ds", t2c={"s": YELLOW})
        )
        for mob1, mob2 in zip(ilp, terms):
            mob1.replace(mob2, dim_to_match=1)

        ilp.move_to(screens[2])

        self.play(
            TransformMatchingStrings(name.copy(), ilp[0], lag_ratio=1e-2, path_arc=-20 * DEG),
            TransformMatchingTex(
                laplace.copy(),
                ilp[1],
                lag_ratio=1e-2,
                path_arc=-20 * DEG,
                matched_keys=["f({t})", "F(s)", "e^{s{t}}", R"\\int"],
                key_map={"dt": "ds"},
            ),
        )
        self.wait()


class DiscussTrilogy(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        screens = ScreenRectangle().replicate(3)
        screens.set_width(0.3 * FRAME_WIDTH)
        screens.arrange(RIGHT, buff=0.25 * (FRAME_WIDTH - 3 * screens[0].get_width()))
        screens.set_width(FRAME_WIDTH - 1)
        screens.to_edge(UP)

        # Reference last two
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        morty.change_mode("tease")
        brace1 = Brace(screens[0], DOWN)
        brace2 = Brace(screens[1:3], DOWN)

        self.wait(2)
        self.play(
            morty.change("raise_left_hand", look_at=brace2),
            self.change_students("pondering", "erm", "sassy", look_at=brace2),
            GrowFromCenter(brace2),
        )
        self.wait(3)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("thinking", "pondering", "pondering", look_at=brace1),
            ReplacementTransform(brace2, brace1, path_arc=-30 * DEG),
        )
        self.wait(6)


class WhoCares(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        stds = self.students
        morty = self.teacher

        self.play(
            stds[2].says("Who cares?", mode="angry", look_at=3 * UP),
            morty.change("guilty", stds[2].eyes),
            stds[1].change("hesitant", 3 * UP),
            stds[0].change("erm", stds[2].eyes),
        )
        self.wait(3)


class MiniLessonTitle(InteractiveScene):
    def construct(self):
        title = Text("Visualizing complex exponents", font_size=72)
        title.to_edge(UP)
        self.play(Write(title))
        self.wait()


class WeGotThis(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            self.change_students("coin_flip_2", "tease", "hooray", look_at=3 * UP),
            morty.change("tease")
        )
        self.wait()
        self.play(
            self.change_students("tease", "happy", "well", look_at=morty.eyes)
        )
        self.wait(3)


class ConfusionAndWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(200)
        morty = self.teacher
        stds = self.students

        q_marks = Tex(R"???").replicate(3)
        q_marks.space_out_submobjects(1.5)
        for mark, student in zip(q_marks, stds):
            mark.next_to(student, UP, MED_SMALL_BUFF)
        self.play(
            self.change_students("confused", "pondering", "pleading", look_at=self.screen),
            FadeIn(q_marks, 0.2 * UP, lag_ratio=0.05),
            morty.change("raise_right_hand")
        )
        self.wait(3)
        self.play(morty.change("raise_left_hand", look_at=3 * UR))
        self.play(
            self.change_students("erm", "thinking", "hesitant", look_at=morty.get_top() + 2 * UP),
            FadeOut(q_marks)
        )
        self.wait(4)
        self.play(self.change_students("pondering"))
        self.wait(3)


class ArrowBetweenScreens(InteractiveScene):
    def construct(self):
        # Test
        screens = ScreenRectangle().replicate(2)
        screens.arrange(RIGHT, buff=MED_LARGE_BUFF)
        screens.set_width(FRAME_WIDTH - 2)
        screens.move_to(DOWN)
        arrow = Arrow(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, thickness=6, buff=0.25)
        line = Line(screens[0].get_top(), screens[1].get_top(), path_arc=-120 * DEG, stroke_width=8, buff=0.25)
        VGroup(arrow, line).set_color(TEAL)
        self.play(
            ShowCreation(line),
            FadeIn(arrow, time_span=(0.75, 1))
        )
        self.wait()


class WhatAndWhy(InteractiveScene):
    def construct(self):
        words = VGroup(
            Tex(R"\\text{1) Understanding } e^{i {t}} \\\\ \\text{ intuitively}", t2c={R"{t}": GREY_B}),
            TexText(R"2) How they \\\\ \\quad \\quad naturally arise"),
        )
        words[0][R"intuitively"].align_to(words[0]["Understanding"], LEFT)
        words[1][R"naturally arise"].align_to(words[1]["How"], LEFT)
        words.refresh_bounding_box()
        words.scale(1.25)
        self.add(words)
        words.arrange(DOWN, aligned_edge=LEFT, buff=2.5)
        words.next_to(ORIGIN, RIGHT)
        words.set_opacity(0)
        for word, u in zip(words, [1, -1]):
            word.set_opacity(1)
            self.play(Write(word))
            self.wait()
        # Test


class PrequelToLaplace(InteractiveScene):
    def construct(self):
        # False goal of motivating the i
        pass

        # Swap out i and π for s and t


class OtherExponentialDerivatives(InteractiveScene):
    def construct(self):
        # Test
        kw = dict(t2c={"t": GREY_B})
        derivs = VGroup(
            Tex(R"\\frac{d}{dt} 2^t = (0.693...)2^t", **kw),
            Tex(R"\\frac{d}{dt} 3^t = (1.098...)3^t", **kw),
            Tex(R"\\frac{d}{dt} 4^t = (1.386...)4^t", **kw),
            Tex(R"\\frac{d}{dt} 5^t = (1.609...)5^t", **kw),
            Tex(R"\\frac{d}{dt} 6^t = (1.791...)6^t", **kw),
        )
        derivs.scale(0.75)
        derivs.arrange(DOWN, buff=0.7)
        derivs.to_corner(UL)

        self.play(LaggedStartMap(FadeIn, derivs, shift=UP, lag_ratio=0.5, run_time=5))
        self.wait()


class VariousExponentials(InteractiveScene):
    def construct(self):
        # Test
        exp_st = Tex(R"e^{st}", t2c={"s": YELLOW, "t": BLUE}, font_size=90)
        gen_exp = Tex(R"e^{+0.50 t}", t2c={"+0.50": YELLOW, "t": BLUE}, font_size=90)
        exp_st.to_edge(UP, buff=MED_LARGE_BUFF)
        gen_exp.move_to(exp_st)

        num = gen_exp["+0.50"]
        num.set_opacity(0)
        gen_exp["t"].scale(1.25, about_edge=UL)

        s_num = DecimalNumber(-1.00, edge_to_fix=ORIGIN, include_sign=True)
        s_num.set_color(YELLOW)
        s_num.replace(num, dim_to_match=1)

        self.add(gen_exp, s_num)
        self.play(ChangeDecimalToValue(s_num, 0.5, run_time=4))
        self.wait()
        self.play(LaggedStart(
            ReplacementTransform(gen_exp["e"][0], exp_st["e"][0]),
            ReplacementTransform(s_num, exp_st["s"]),
            ReplacementTransform(gen_exp["t"][0], exp_st["t"][0]),
        ))
        self.wait()


class WhyToWhat(InteractiveScene):
    def construct(self):
        # Title text
        why = Text("Why", font_size=90)
        what = Text("Wait, what does this even mean?", font_size=72)
        VGroup(why, what).to_edge(UP)

        what_word = what["what"][0].copy()
        what["what"][0].set_opacity(0)

        arrow = Arrow(
            what["this"].get_bottom(),
            (2.5, 2, 0),
            thickness=5,
            fill_color=YELLOW
        )

        self.play(FadeIn(why, UP))
        self.wait()
        self.play(
            # FadeOut(why, UP),
            ReplacementTransform(why, what_word),
            FadeIn(what, lag_ratio=0.1),
        )
        self.play(
            GrowArrow(arrow),
            what["this"].animate.set_color(YELLOW)
        )
        self.wait()


class DerivativeOfExp(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        tex_kw = dict(t2c={"t": GREY_B, "s": YELLOW})
        equation = Tex(R"\\frac{d}{dt} e^{st} = s \\cdot e^{st}", font_size=90, **tex_kw)
        deriv_part = equation[R"\\frac{d}{dt}"][0]
        exp_parts = equation[R"e^{st}"]
        equals = equation[R"="][0]
        s_dot = equation[R"s \\cdot"][0]

        v_box = SurroundingRectangle(VGroup(deriv_part, exp_parts[0]))
        p_box = SurroundingRectangle(exp_parts[1])
        s_box = SurroundingRectangle(s_dot)
        s_box.match_height(p_box, stretch=True).match_y(p_box)
        boxes = VGroup(v_box, p_box, s_box)
        boxes.set_stroke(width=2)
        boxes.set_submobject_colors_by_gradient(GREEN, BLUE, YELLOW)

        v_label = Text("Velocity", font_size=48).match_color(v_box)
        p_label = Text("Position", font_size=48).match_color(p_box)
        s_label = Text("Modifier", font_size=48).match_color(s_box)
        v_label.next_to(v_box, UP, MED_SMALL_BUFF)
        p_label.next_to(p_box, UP, MED_SMALL_BUFF, aligned_edge=LEFT)
        s_label.next_to(s_box, DOWN, MED_SMALL_BUFF)
        labels = VGroup(v_label, p_label, s_label)

        frame.move_to(exp_parts[0])

        self.add(exp_parts[0])
        self.wait()
        self.play(Write(deriv_part))
        self.play(
            TransformFromCopy(*exp_parts, path_arc=90 * DEG),
            Write(equals),
            frame.animate.center(),
        )
        self.play(
            TransformFromCopy(exp_parts[1][1], s_dot[0], path_arc=90 * DEG),
            Write(s_dot[1]),
        )
        self.wait()

        # Show labels
        for box, label in zip(boxes, labels):
            self.play(ShowCreation(box), FadeIn(label))

        self.wait()
        full_group = VGroup(equation, boxes, labels)

        # Set s equal to 1
        s_eq_1 = Tex(R"s = 1", font_size=72, **tex_kw)
        simple_equation = Tex(R"\\frac{d}{dt} e^{t} = e^{t}", font_size=72, **tex_kw)
        simple_equation.to_edge(UP).shift(2 * LEFT)
        s_eq_1.next_to(simple_equation, RIGHT, buff=2.5)
        arrow = Arrow(s_eq_1, simple_equation, thickness=5, buff=0.35).shift(0.05 * DOWN)

        self.play(
            Write(s_eq_1),
            GrowArrow(arrow),
            TransformMatchingTex(equation.copy(), simple_equation, run_time=1.5, lag_ratio=0.02),
            full_group.animate.shift(DOWN).scale(0.75).fade(0.15)
        )
        self.wait()


class HighlightRect(InteractiveScene):
    def construct(self):
        img = ImageMobject('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/laplace/exponentials/DynamicExpIntuitionStill.png')
        img.set_height(FRAME_HEIGHT)
        self.add(img)

        # Rects
        rects = VGroup(
            Rectangle(2.25, 1).move_to((2.18, 2.74, 0)),
            Rectangle(2, 0.85).move_to((-5.88, -2.2, 0.0)),
        )
        rects.set_stroke(YELLOW, 2)

        self.play(ShowCreation(rects[0]))
        self.play(TransformFromCopy(*rects))
        self.play(FadeOut(rects))


class DefineI(InteractiveScene):
    def construct(self):
        eq = Tex(R"i = \\sqrt{-1}", t2c={"i": YELLOW}, font_size=90)
        self.play(Write(eq))
        self.wait()


class WaitWhy(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].change("erm", self.screen),
            self.students[1].change("tease", self.screen),
            self.students[2].says("Wait, why?", "confused", look_at=self.screen, bubble_direction=LEFT),
        )
        self.wait(4)


class MultiplicationByI(InteractiveScene):
    def construct(self):
        # Example number
        plane = ComplexPlane(
            background_line_style=dict(stroke_color=BLUE, stroke_width=1),
            # faded_line_style=dict(stroke_color=BLUE, stroke_width=0.5, stroke_opacity=0.5),
        )
        plane.add_coordinate_labels(font_size=24)

        z = 3 + 2j
        tex_kw = dict(t2c={"a": YELLOW, "b": PINK})

        vect = Vector(plane.n2p(z), fill_color=WHITE, thickness=4)
        vect_label = Tex(R"a + bi", **tex_kw)
        vect_label.next_to(vect.get_end(), UR, SMALL_BUFF)
        vect_label.set_backstroke(BLACK, 5)

        lines = VGroup(
            Line(ORIGIN, plane.n2p(z.real)).set_color(YELLOW),
            Line(plane.n2p(z.real), plane.n2p(z)).set_color(PINK),
        )
        a_label, b_label = line_labels = VGroup(
            Tex(R"a", font_size=36, **tex_kw).next_to(lines[0], UP, SMALL_BUFF),
            Tex(R"bi", font_size=36, **tex_kw).next_to(lines[1], RIGHT, SMALL_BUFF),
        )
        line_labels.set_backstroke(BLACK, 5)

        self.add(plane, Point(), plane.coordinate_labels)
        self.add(vect)
        self.add(vect_label)
        for line, label in zip(lines, line_labels):
            self.play(
                ShowCreation(line),
                FadeIn(label, 0.25 * line.get_vector())
            )
        self.wait()

        # Multiply components by i
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG, about_point=ORIGIN)
        new_lines[1].move_to(ORIGIN, RIGHT)

        new_a_label = Tex(R"ai", font_size=36, **tex_kw).next_to(new_lines[0], RIGHT, SMALL_BUFF)
        new_b_label = Tex(R"bi \\cdot i", font_size=36, **tex_kw).next_to(new_lines[1], UP, SMALL_BUFF)
        neg_b_label = Tex(R"=-b", font_size=36, **tex_kw)
        neg_b_label.move_to(new_b_label.get_right())

        mult_i_label = Tex(R"\\times i", font_size=90)
        mult_i_label.set_backstroke(BLACK, 5)
        mult_i_label.to_corner(UR, buff=MED_LARGE_BUFF).shift(0.2 * UP)

        self.play(Write(mult_i_label))
        self.wait()
        self.play(
            TransformFromCopy(lines[0], new_lines[0], path_arc=90 * DEG),
            TransformFromCopy(a_label[0], new_a_label[0], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_a_label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(lines[1], new_lines[1], path_arc=90 * DEG),
            TransformFromCopy(b_label[0], new_b_label[:-1], path_arc=90 * DEG),
            TransformFromCopy(mult_i_label[1], new_b_label[-1]),
        )
        self.wait()
        self.play(
            FlashAround(VGroup(new_b_label, new_lines[1]), color=PINK, time_width=1.5, run_time=2),
            new_b_label.animate.next_to(neg_b_label, LEFT, SMALL_BUFF),
            FadeIn(neg_b_label, SMALL_BUFF * RIGHT),
        )
        self.wait()
        self.play(VGroup(new_lines[1], new_b_label, neg_b_label).animate.shift(new_lines[0].get_vector()))

        # New vector
        vect_copy = vect.copy()
        elbow = Elbow().rotate(vect.get_angle(), about_point=ORIGIN)
        self.play(
            Rotate(vect_copy, 90 * DEG, run_time=2, about_point=ORIGIN),
        )
        self.play(
            ShowCreation(elbow)
        )
        self.wait()

    def old_material(self):
        # Show the algebra
        algebra = VGroup(
            Tex(R"i \\cdot (a + bi)", **tex_kw),
            Tex(R"ai + bi^2", **tex_kw),
            Tex(R"-b + ai", **tex_kw),
        )
        algebra.set_backstroke(BLACK, 8)
        algebra.arrange(DOWN, buff=0.35)
        algebra.to_corner(UL)

        self.play(
            TransformFromCopy(vect_label, algebra[0]["a + bi"][0]),
            FadeIn(algebra[0]),
        )
        self.play(LaggedStart(
            TransformFromCopy(algebra[0]["a"], algebra[1]["a"]),
            TransformFromCopy(algebra[0]["+ bi"], algebra[1]["+ bi"]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["i"][0]),
            TransformFromCopy(algebra[0]["i"][0], algebra[1]["2"]),
            lag_ratio=0.25
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(algebra[1]["bi^2"], algebra[2]["-b"]),
            TransformFromCopy(algebra[1]["ai"], algebra[2]["ai"]),
            TransformFromCopy(algebra[1]["+"], algebra[2]["+"]),
            lag_ratio=0.25
        ))
        self.wait()

        # New lines
        new_lines = lines.copy()
        new_lines.rotate(90 * DEG)
        new_lines.refresh_bounding_box()
        new_lines[1].move_to(ORIGIN, RIGHT)
        new_lines[0].move_to(new_lines[1].get_left(), DOWN)

        neg_b_label = Tex(R"-b", fill_color=PINK, font_size=36).next_to(new_lines[1], UP, SMALL_BUFF)
        new_a_label = Tex(R"a", fill_color=YELLOW, font_size=36).next_to(new_lines[0], LEFT, SMALL_BUFF)

        self.play(
            TransformFromCopy(lines[1], new_lines[1]),
            FadeTransform(algebra[2]["-b"].copy(), neg_b_label),
        )
        self.play(
            TransformFromCopy(lines[0], new_lines[0]),
            FadeTransform(algebra[2]["a"].copy(), new_a_label),
        )
        self.wait()


class UnitArcLengthsOnCircle(InteractiveScene):
    def construct(self):
        # Moving sectors
        arc = Arc(0, 1, radius=2.5, stroke_color=GREEN, stroke_width=8)
        sector = Sector(angle=1, radius=2.5).set_fill(GREEN, 0.25)
        v_line = Line(ORIGIN, 2.5 * UP)
        v_line.match_style(arc)
        v_line.move_to(arc.get_start(), DOWN)

        self.add(v_line)
        self.play(
            FadeIn(sector),
            ReplacementTransform(v_line, arc),
        )

        group = VGroup(sector, arc)
        self.add(group)

        for n in range(5):
            self.wait(2)
            group.rotate(1, about_point=ORIGIN)

        return

        # Previous
        colors = [RED, BLUE]
        arcs = VGroup(
            Arc(n, 1, radius=2.5, stroke_color=colors[n % 2], stroke_width=8)
            for n in range(6)
        )
        for arc in arcs:
            one = Integer(1, font_size=24).move_to(1.0 * arc.get_center())
            self.play(ShowCreation(arc, rate_func=linear, run_time=2))
        self.wait()


class SimpleIndicationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(3, 2)
        # Test
        self.play(FlashAround(rect, time_width=2.0, run_time=2, color=WHITE))


class WriteSPlane(InteractiveScene):
    def construct(self):
        title = Text("S-plane", font_size=72)
        title.set_color(YELLOW)
        self.play(Write(title))
        self.wait()


class ODEStoExp(InteractiveScene):
    def construct(self):
        # Test
        odes, exp = words = VGroup(
            Text("Differential\\nEquations"),
            Tex("e^{st}", t2c={"s": YELLOW}, font_size=72),
        )
        exp.match_height(odes)
        words.arrange(RIGHT, buff=3.0)
        words.to_edge(UP, buff=1.25)

        top_arrow, low_arrow = arrows = VGroup(
            Arrow(odes.get_corner(UR), exp.get_corner(UL), path_arc=-60 * DEG, thickness=5),
            Arrow(exp.get_corner(DL), odes.get_corner(DR), path_arc=-60 * DEG, thickness=5),
        )
        arrows.set_fill(TEAL)

        top_words = Tex(R"Explain", font_size=36).next_to(top_arrow, UP, SMALL_BUFF)
        low_words = Tex(R"Solves", font_size=36).next_to(low_arrow, DOWN, SMALL_BUFF)

        exp.shift(0.25 * UP + 0.05 * LEFT)

        self.add(words)
        self.wait()
        self.play(
            Write(top_arrow),
            Write(top_words),
        )
        self.wait()
        self.play(
            # Write(low_arrow),
            TransformFromCopy(top_arrow, low_arrow, path_arc=-PI),
            Write(low_words),
        )
        self.wait()


class GenLinearEquationToOscillator(InteractiveScene):
    def construct(self):
        # General equation
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = dict()
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \\cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c, font_size=60)
        ode.move_to(DOWN)
        ode_2nd = ode["a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0"]

        self.play(Write(ode))
        self.wait()
        self.play(
            FadeOut(ode[R"a_n x^{n}(t) + \\cdots + "]),
            ode_2nd.animate.move_to(UP),
            self.frame.animate.set_height(7)
        )

        # Transition
        alt_consts = VGroup(Tex(R"m"), Tex(R"\\mu"), Tex(R"k"))
        alt_consts.scale(60 / 48)
        a_parts = VGroup(ode[tex][0] for tex in a_texs[1:])
        for const, a_part in zip(alt_consts, a_parts):
            const.move_to(a_part, RIGHT)
            const.align_to(ode[-1], DOWN)
            if const is alt_consts[1]:
                const.shift(0.1 * DOWN)
            self.play(
                FadeOut(a_part, 0.25 * UP),
                FadeIn(const, 0.25 * UP),
            )
        self.wait()


class VLineOverZero(InteractiveScene):
    def construct(self):
        # Test
        rect = Square(0.25)
        rect.move_to(2.5 * DOWN)
        v_line = Line(rect.get_top(), 4 * UP, buff=0.1)
        v_line.set_stroke(YELLOW, 2)
        rect.match_style(v_line)

        self.play(
            ShowCreationThenFadeOut(rect),
            ShowCreationThenFadeOut(v_line),
        )
        self.wait()


class KIsSomeConstant(InteractiveScene):
    def construct(self):
        rect = SurroundingRectangle(Text("k"), buff=0.05)
        rect.set_stroke(YELLOW, 2)
        words = Text("Some constant", font_size=24)
        words.next_to(rect, UP, SMALL_BUFF)
        words.match_color(rect)

        self.play(ShowCreation(rect), FadeIn(words))
        self.wait()


class WriteMu(InteractiveScene):
    def construct(self):
        sym = Tex(R"\\mu")
        rect = SurroundingRectangle(sym, buff=0.05)
        rect.set_stroke(YELLOW, 2)
        mu = TexText("\`\`Mu''")
        mu.set_color(YELLOW)
        mu.next_to(rect, DOWN)
        self.play(
            Write(mu),
            ShowCreation(rect)
        )
        self.wait()


class ReferenceGuessingExp(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.remove(self.background)
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        # Student asks
        question = Tex(R"x(t) = ???")
        lhs = question["x(t)"][0]
        rhs = question["= ???"][0]
        bubble = stds[2].get_bubble(question, bubble_type=SpeechBubble, direction=LEFT)
        lhs.save_state()
        lhs.scale(0.25).move_to([-6.24, 2.38, 0])

        self.play(
            morty.change("hesitant", look_at=stds[2].eyes),
            self.change_students("erm", "confused", "maybe", look_at=self.screen)
        )
        self.wait()
        self.play(
            stds[2].change("raise_left_hand", morty.eyes),
            Write(bubble[0]),
            Write(rhs, time_span=(0.5, 1.0)),
            Restore(lhs),
        )
        self.wait()
        self.add(Point())
        self.play(
            morty.says("Here's a trick:", mode="tease", bubble_creation_class=FadeIn),
            self.change_students("pondering", "thinking", "hesitant", look_at=UL),
        )
        self.wait(2)

        # Teacher gestures to upper right, students look confused and hesitant
        eq_point = 5 * RIGHT + 3 * UP
        self.play(
            morty.change("raise_right_hand", look_at=eq_point),
            FadeOut(bubble),
            FadeOut(morty.bubble),
            self.change_students("confused", "thinking", "hesitant", look_at=eq_point),
        )
        self.wait()
        self.play(self.change_students("confused", "hesitant", "confused", look_at=eq_point, lag_ratio=0.1))
        self.wait()
        self.play(
            morty.change("shruggie", look_at=eq_point),
        )
        self.wait(2)
        self.play(
            self.change_students("angry", "hesitant", "erm", look_at=morty.eyes),
            morty.animate.look_at(stds)
        )
        self.wait(2)

        # Transition: flip and reposition morty to where stds are
        new_teacher_pos = stds[2].get_bottom()
        new_teacher = morty.copy()
        new_teacher.change_mode("raise_left_hand")
        new_teacher.look_at(3 * UR)
        new_teacher.body.set_color(GREY_C)

        self.play(
            morty.animate.scale(0.8).flip().change_mode("confused").look_at(5 * UR).move_to(new_teacher_pos, DOWN),
            LaggedStartMap(FadeOut, stds, shift=DOWN, lag_ratio=0.2, run_time=1),
            FadeIn(new_teacher, time_span=(0.5, 1.5)),
        )
        self.play(morty.change("pleading", 3 * UR))
        self.play(Blink(new_teacher))
        self.wait(2)
        self.play(LaggedStart(
            morty.change("erm", new_teacher.eyes),
            new_teacher.change("guilty", look_at=morty.eyes),
            lag_ratio=0.5,
        ))
        self.wait(3)

        # Reference a graph
        self.play(
            morty.change("angry", 2 * UR),
            new_teacher.change("tease", 2 * UR)
        )
        self.play(Blink(morty))
        self.play(Blink(new_teacher))
        self.wait()


class FromGuessToLaplace(InteractiveScene):
    def construct(self):
        # Words
        strategy = VGroup(
            Text("“Strategy”", fill_color=GREY_A, font_size=72),
            TexText("Guess $x(t) = e^{{s}t}$", t2c={"{s}": YELLOW}, fill_color=WHITE, font_size=72),
        )
        strategy.arrange(DOWN)
        self.add(strategy)
        return

        # Comment on it
        exp_rect = SurroundingRectangle(strategy[1]["x(t) = e^{{s}t}"], buff=SMALL_BUFF)
        exp_words = Text("Why?", font_size=42)
        exp_words.next_to(exp_rect, RIGHT, SMALL_BUFF)
        VGroup(exp_rect, exp_words).set_color(PINK)

        guess_rect = SurroundingRectangle(strategy[1]["Guess"], buff=SMALL_BUFF)
        guess_rect.match_height(exp_rect, stretch=True).match_y(exp_rect)
        guess_words = Text("Seems dumb", font_size=36)
        guess_words.next_to(guess_rect, DOWN, SMALL_BUFF)
        VGroup(guess_rect, guess_words).set_color(RED)

        self.play(LaggedStart(
            ShowCreation(guess_rect),
            FadeIn(guess_words, lag_ratio=0.1),
            ShowCreation(exp_rect),
            FadeIn(exp_words, lag_ratio=0.1),
            lag_ratio=0.25
        ))
        self.wait()

        # Transition to Laplace
        laplace = Tex(R"\\int_0^\\infty x(t) e^{-{s}t} dt", t2c={"{s}": YELLOW}, font_size=72)
        laplace.move_to(strategy[1])

        self.play(LaggedStart(
            LaggedStartMap(FadeOut, VGroup(strategy[1]["Guess"], guess_rect, guess_words), shift=DOWN, lag_ratio=0.1),
            TransformFromCopy(strategy[0]["S"][0], laplace[R"\\int"][0]),
            TransformFromCopy(strategy[0]["e"][0], laplace[R"0"][0]),
            TransformFromCopy(strategy[0]["g"][0], laplace[R"\\infty"][0]),
            FadeOut(strategy[0], lag_ratio=0.1),
            # Break
            FadeOut(VGroup(exp_rect, exp_words), 0.5 * LEFT, lag_ratio=0.1),
            FadeTransform(strategy[1]["x(t)"][0], laplace["x(t)"][0]),
            FadeTransform(strategy[1]["="][0], laplace["-"][0]),
            FadeTransform(strategy[1]["e"][-1], laplace["e"][0]),
            FadeTransform(strategy[1]["{s}t"][0], laplace["{s}t"][0]),
            Write(laplace["dt"][0]),
            lag_ratio=0.15,
            run_time=3,
        ))
        self.wait()

        # Label laplace
        laplace_rect = SurroundingRectangle(laplace)
        laplace_rect.set_color(BLUE)
        laplace_label = Text("Laplace Transform", font_size=72)
        laplace_label.next_to(laplace_rect, UP)
        laplace_label.match_color(laplace_rect)

        self.play(
            Write(laplace_label),
            ShowCreation(laplace_rect),
        )
        self.wait()


class JustAlgebra(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer(mode="tease")
        morty.body.insert_n_curves(100)
        self.play(morty.says("Just algebra!", mode="hooray", look_at=2 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(
            FadeOut(morty.bubble),
            morty.change("tease", look_at=2 * UL + UP)
        )
        self.play(Blink(morty))
        self.wait()


class BothPositiveNumbers(InteractiveScene):
    def construct(self):
        tex = Tex("k / m")
        self.add(tex)

        # Test
        rects = VGroup(SurroundingRectangle(tex[c], buff=0.05) for c in "km")
        rects.set_stroke(GREEN, 3)
        plusses = VGroup(Tex(R"+").next_to(rect, DOWN, SMALL_BUFF) for rect in rects)
        plusses.set_fill(GREEN)

        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.5),
            LaggedStartMap(FadeIn, plusses, shift=0.25 * DOWN, lag_ratio=0.5)
        )
        self.wait()


class ButSpringsAreReal(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("maybe", self.screen),
            stds[1].says("But...springs are real", mode="confused", look_at=self.screen),
            stds[2].change("erm", self.screen),
            morty.change("tease", stds[2].eyes)
        )
        self.wait(4)


class ShowIncreaseToK(InteractiveScene):
    def construct(self):
        # Test
        k = Tex(R"k")

        box = SurroundingRectangle(k)
        box.set_stroke(GREEN, 5)
        arrow = Vector(UP, thickness=6)
        arrow.set_fill(GREEN)
        center = box.get_center()

        self.play(
            ShowCreation(box),
            UpdateFromAlphaFunc(
                arrow, lambda m, a: m.move_to(
                    center + interpolate(-1, 1, a) * UP
                ).set_fill(
                    opacity=there_and_back(a) * 0.7
                ),
                run_time=4
            ),
        )
        self.wait()


class PureMathEquation(InteractiveScene):
    def construct(self):
        # Test
        t2c = {"x''(t)": RED, "x(t)": TEAL, R"\\omega": PINK}
        physics_eq = Tex(R"m x''(t) + k x(t) = 0", t2c=t2c, font_size=72)
        math_eq = Tex(R"a_2 x''(t) + a_0 x(t) = 0", t2c=t2c, font_size=72)

        self.add(physics_eq)
        self.play(LaggedStart(
            *(
                ReplacementTransform(physics_eq[tex][0], math_eq[tex][0])
                for tex in ["x''(t) +", "x(t) = 0"]
            ),
            FadeOut(physics_eq["m"], 0.5 * UP),
            FadeIn(math_eq["a_2"], 0.5 * UP),
            FadeOut(physics_eq["k"], 0.5 * UP),
            FadeIn(math_eq["a_0"], 0.5 * UP),
            run_time=2,
            lag_ratio=0.15
        ))
        self.wait()

        # Show solution
        implies = Tex(R"\\Downarrow", font_size=72)
        answer = Tex(R"e^{\\pm i\\omega t}", font_size=90, t2c=t2c)
        answer.next_to(implies, DOWN, MED_LARGE_BUFF)
        omega_eq = Tex(R"\\text{Where } \\omega = \\sqrt{a_2 / a_0}", t2c=t2c)
        omega_eq.next_to(answer, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            math_eq.animate.next_to(implies, UP, MED_LARGE_BUFF),
            Write(implies),
            FadeIn(answer, DOWN),
            lag_ratio=0.25
        ))
        self.play(FadeIn(omega_eq))
        self.wait()


class LinearityDefinition(InteractiveScene):
    def construct(self):
        # Base differential equation string
        eq_str = R"m x''(t) + k x(t) = 0"
        t2c = {"x_1": TEAL, "x_2": RED, "0.0": YELLOW, "2.0": YELLOW}

        base_eq = Tex(eq_str)
        base_eq.to_edge(UP)

        eq1, eq2, eq3, eq4 = equations = VGroup(
            Tex(eq_str.replace("x", "x_1"), t2c=t2c),
            Tex(eq_str.replace("x", "x_2"), t2c=t2c),
            Tex(R"m\\Big(x_1''(t) + x_2''(t) \\Big) + k \\Big(x_1(t) + x_2(t)\\Big) = 0", t2c=t2c),
            Tex(R"m\\Big(0.0 x_1''(t) + 2.0 x_2''(t) \\Big) + k \\Big(0.0 x_1(t) + 2.0 x_2(t)\\Big) = 0", t2c=t2c),
        )
        for eq in equations:
            eq.set_max_width(7)
        equations.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        equations.to_edge(RIGHT)
        equations.shift(DOWN)

        phrase1, phrase2, phrase3, phrase4 = phrases = VGroup(
            TexText("If $x_1$ solves it:", t2c=t2c),
            TexText("and $x_2$ solves it:", t2c=t2c),
            TexText("Then $(x_1 + x_2)$ solves it:", t2c=t2c),
            TexText("Then $(0.0 x_1 + 2.0 x_2)$ solves it:", t2c=t2c),
        )

        for phrase, eq in zip(phrases, equations):
            phrase.set_max_width(5)
            phrase.next_to(eq, LEFT, LARGE_BUFF)

        eq4.move_to(eq3)
        phrase4.move_to(phrase3)

        kw = dict(edge_to_fix=RIGHT)
        c1_terms = VGroup(phrase4.make_number_changeable("0.0", **kw), *eq4.make_number_changeable("0.0", replace_all=True, **kw))
        c2_terms = VGroup(phrase4.make_number_changeable("2.0", **kw), *eq4.make_number_changeable("2.0", replace_all=True, **kw))

        # Show base equation
        self.play(Write(phrase1), FadeIn(eq1))
        self.wait()
        self.play(
            TransformMatchingTex(eq1.copy(), eq2, key_map={"x_1": "x_2"}, run_time=1, lag_ratio=0.01),
            FadeTransform(phrase1.copy(), phrase2)
        )
        self.wait()
        self.play(
            FadeIn(phrase3, DOWN),
            FadeIn(eq3, DOWN),
        )
        self.wait()
        self.play(
            FadeOut(eq3, 0.5 * DOWN),
            FadeOut(phrase3, 0.5 * DOWN),
            FadeIn(eq4, 0.5 * DOWN),
            FadeIn(phrase4, 0.5 * DOWN),
        )
        for _ in range(8):
            new_c1 = random.random() * 10
            new_c2 = random.random() * 10
            self.play(*(
                ChangeDecimalToValue(c1, new_c1, run_time=1)
                for c1 in c1_terms
            ))
            self.wait(0.5)
            self.play(*(
                ChangeDecimalToValue(c2, new_c2, run_time=1)
                for c2 in c2_terms
            ))
            self.wait(0.5)


class ComplainAboutNeelessComplexity(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Complain
        self.play(
            stds[0].change("confused", self.screen),
            stds[1].says("That’s needlessly\\ncomplicated!", mode="angry", look_at=self.screen),
            stds[2].change("maybe", self.screen),
            morty.change("guilty"),
        )
        self.wait(3)
        self.play(
            stds[0].change("erm", self.screen),
            stds[1].debubble(mode="raise_left_hand", look_at=self.screen),
            stds[2].change("sassy", self.screen),
            morty.change("tease"),
        )
        self.wait()
        self.play(
            stds[1].change("raise_right_hand", ORIGIN),
            stds[0].change("pondering", ORIGIN),
            stds[2].change("pondering", ORIGIN),
        )
        self.wait(5)


class LetsGeneralize(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            morty.says("Let’s\\ngenerlize!", mode="hooray")
        )
        self.play(Blink(morty))
        self.wait(3)


class EquationRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(5.25, 1)
        rect.set_stroke(YELLOW, 3)

        # Test
        self.play(ShowCreation(rect))
        self.wait()
        self.play(rect.animate.stretch(0.5, 0).shift(4 * RIGHT).set_opacity(0))
        self.wait()


class GeneralLinearEquation(InteractiveScene):
    def construct(self):
        # Set up equations
        a_texs = ["a_n", "a_2", "a_1", "a_0"]
        x_texs = ["x^{n}(t)", "x''(t)", "x'(t)", "x(t)"]
        x_colors = color_gradient([BLUE, TEAL], len(x_texs), interp_by_hsl=True)
        t2c = {"{s}": YELLOW}
        t2c.update({a: WHITE for a in a_texs})
        t2c.update({x: color for x, color in zip(x_texs, x_colors)})
        ode = Tex(R"a_n x^{n}(t) + \\cdots + a_2 x''(t) + a_1 x'(t) + a_0 x(t) = 0", t2c=t2c)
        exp_version = Tex(
            R"a_n \\left({s}^n e^{{s}t}\\right) "
            R"+ \\cdots "
            R"+ a_2 \\left({s}^2 e^{{s}t}\\right) "
            R"+ a_1 \\left({s}e^{{s}t}\\right) "
            R"+ a_0 e^{{s}t} = 0",
            t2c=t2c
        )
        factored = Tex(R"e^{{s}t} \\left(a_n {s}^n + \\cdots + a_2 {s}^2 + a_1 {s} + a_0 \\right) = 0", t2c=t2c)

        ode.to_edge(UP)
        exp_version.next_to(ode, DOWN, MED_LARGE_BUFF)
        factored.move_to(exp_version)

        # Introduce ode
        index = ode.submobjects.index(ode["a_2"][0][0])

        right_part = ode[index:]
        left_part = ode[:index]
        right_part.save_state()
        right_part.set_x(0)

        self.play(FadeIn(right_part, UP))
        self.wait()
        self.play(LaggedStart(
            Restore(right_part),
            Write(left_part)
        ))
        self.add(ode)

        # Highlight equation parts
        x_arrows = VGroup(
            Arrow(UP, ode[x_tex].get_bottom(), fill_color=color)
            for x_tex, color in zip(x_texs, x_colors)
        )
        x_arrows.reverse_submobjects()

        x_rects = VGroup(SurroundingRectangle(ode[x_tex], buff=SMALL_BUFF) for x_tex in x_texs)
        a_rects = VGroup(SurroundingRectangle(ode[a_tex]) for a_tex in a_texs)
        full_rect = SurroundingRectangle(ode[:-2])
        zero_rect = SurroundingRectangle(ode[-2:])
        VGroup(x_rects, a_rects, full_rect, zero_rect).set_stroke(YELLOW, 2)

        self.play(LaggedStartMap(ShowCreation, x_rects))
        self.wait()
        self.play(ReplacementTransform(x_rects, a_rects, lag_ratio=0.2))
        self.wait()
        self.play(ReplacementTransform(a_rects, VGroup(full_rect)))
        self.wait()
        self.play(ReplacementTransform(full_rect, zero_rect))
        self.wait()
        self.play(FadeOut(zero_rect))

        # Plug in e^{st}
        key_map = {
            R"+ a_0 x(t) = 0": R"+ a_0 e^{{s}t} = 0",
            R"+ a_1 x'(t)": R"+ a_1 \\left({s}e^{{s}t}\\right)",
            R"+ a_2 x''(t)": R"+ a_2 \\left({s}^2 e^{{s}t}\\right)",
            R"+ \\cdots": R"+ \\cdots",
            R"a_n x^{n}(t)": R"a_n \\left({s}^n e^{{s}t}\\right)",
        }

        self.play(LaggedStart(*(
            FadeTransform(ode[k1].copy(), exp_version[k2])
            for k1, k2 in key_map.items()
        ), lag_ratio=0.6, run_time=4))
        self.wait()
        self.play(
            TransformMatchingTex(
                exp_version,
                factored,
                matched_keys=[R"e^{{s}t}", "{s}^n", "{s}^2", "{s}", "a_n", "a_2", "a_1", "a_0"],
                path_arc=45 * DEG
            )
        )
        self.wait()

        # Highlight the polynomail
        poly_rect = SurroundingRectangle(factored[R"a_n {s}^n + \\cdots + a_2 {s}^2 + a_1 {s} + a_0"])
        poly_rect.set_stroke(YELLOW, 1)

        self.play(
            ShowCreation(poly_rect),
            FadeOut(factored["e^{{s}t}"]),
            FadeOut(factored[R"\\left("]),
            FadeOut(factored[R"\\right)"]),
        )

        # Show factored expression
        linear_term_texs = [
            R"({s} - s_1)",
            R"({s} - s_2)",
            R"({s} - s_3)",
            R"\\cdots",
            R"({s} - s_n)",
        ]
        fully_factored = Tex(
            R"a_n" + " ".join(linear_term_texs),
            t2c=t2c,
            font_size=42,
            isolate=linear_term_texs
        )
        fully_factored.next_to(poly_rect, DOWN)
        linear_terms = VGroup(
            fully_factored[tex][0]
            for tex in linear_term_texs
        )

        self.play(
            Transform(factored["{s}"][1].copy().replicate(4), fully_factored["{s}"].copy(), remover=True),
            FadeIn(fully_factored, time_span=(0.25, 1)),
        )
        self.wait()

        # Plane
        plane = ComplexPlane((-3, 3), (-3, 3), width=6, height=6)
        plane.set_height(4.5)
        plane.next_to(poly_rect, DOWN, LARGE_BUFF)
        plane.set_x(0)
        plane.add_coordinate_labels(font_size=16)
        c_label = Tex(R"\\mathds{C}", font_size=90, fill_color=BLUE)
        c_label.next_to(plane, LEFT, aligned_edge=UP).shift(0.5 * DOWN)

        self.play(
            Write(plane, run_time=1, lag_ratio=2e-2),
            Write(c_label),
        )

        # Show some random root collections
        for n in range(4):
            roots = []
            n_roots = random.randint(3, 7)
            for _ in range(n_roots):
                root = complex(random.uniform(-3, 3), random.uniform(-3, 3))
                if random.random() < 0.25:
                    roots.append(root.real)
                else:
                    roots.extend([root, root.conjugate()])
            dots = Group(GlowDot(plane.n2p(z)) for z in roots)

            self.play(ShowIncreasingSubsets(dots))
            self.play(FadeOut(dots))

        # Turn linear terms into
        roots = [0.2 + 1j, 0.2 - 1j, -0.5 + 3j, -0.5 - 3j, -2]
        root_dots = Group(GlowDot(plane.n2p(root)) for root in roots)

        root_labels = VGroup(
            Tex(Rf"s_{{{n + 1}}}", font_size=36).next_to(dot.get_center(), UR, SMALL_BUFF)
            for n, dot in enumerate(root_dots)
        )
        root_labels.set_color(YELLOW)

        root_intro_kw = dict(lag_ratio=0.3, run_time=4)
        self.play(
            LaggedStart(*(
                FadeTransform(term, dot)
                for term, dot in zip(linear_terms, root_dots)
            ), **root_intro_kw),
            LaggedStart(*(
                TransformFromCopy(term[3:5], label)
                for term, label in zip(linear_terms, root_labels)
            ), **root_intro_kw),
            FadeOut(fully_factored["a_n"][0]),
        )
        self.wait()

        # Show the solutions
        frame = self.frame
        axes = VGroup(
            Axes((0, 10), (-y_max, y_max), width=5, height=1.25)
            for root in roots
            for y_max in [3 if root.real > 0 else 1]
        )
        axes.arrange(DOWN, buff=0.75)
        axes.next_to(plane, RIGHT, buff=6)

        c_trackers = Group(ComplexValueTracker(1) for root in roots)
        graphs = VGroup(
            self.get_graph(axes, root, c_tracker.get_value)
            for axes, root, c_tracker in zip(axes, roots, c_trackers)
        )

        axes_labels = VGroup(
            Tex(Rf"e^{{s_{{{n + 1}}} t}}", font_size=60)
            for n in range(len(axes))
        )
        for label, ax in zip(axes_labels, axes):
            label.next_to(ax, LEFT, aligned_edge=UP)
            label[1:3].set_color(YELLOW)

        self.play(
            FadeIn(axes, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (4.67, -0.94, 0.0), 10.96),
            LaggedStart(
                (FadeTransform(m1.copy(), m2) for m1, m2 in zip(root_labels, axes_labels)),
                lag_ratio=0.05,
                group_type=Group
            ),
            run_time=2
        )

        rect = Square(side_length=1e-3).move_to(plane.n2p(0))
        rect.set_stroke(TEAL, 3)
        for root_label, graph in zip(root_labels, graphs):
            self.play(
                ShowCreation(graph, time_span=(0.5, 2.0), suspend_mobject_updating=True),
                rect.animate.surround(root_label, buff=0.1),
            )
        self.play(FadeOut(rect))
        self.wait()

        # Add on constants
        constant_labels = VGroup(
            Tex(Rf"c_{{{n + 1}}}", font_size=60).next_to(label[0], LEFT, SMALL_BUFF, aligned_edge=UP)
            for n, label in enumerate(axes_labels)
        )
        constant_labels.set_color(BLUE_B)
        target_values = [0.5, 0.25, 1.5, -1.5, -1]

        solution_rect = SurroundingRectangle(VGroup(axes_labels, axes, constant_labels), buff=MED_SMALL_BUFF)
        solution_rect.set_stroke(WHITE, 1)
        solution_words = Text("All Solutions", font_size=60)
        solution_words.next_to(solution_rect, UP)
        solution_word = solution_words["Solutions"][0]
        solution_word.save_state(0)
        solution_word.match_x(solution_rect)

        const_rects = VGroup(SurroundingRectangle(c_label) for c_label in constant_labels)
        const_rects.set_stroke(BLUE, 3)

        plusses = Tex("+").replicate(4)
        for l1, l2, plus in zip(axes_labels, axes_labels[1:], plusses):
            plus.move_to(VGroup(l1, l2)).shift(SMALL_BUFF * LEFT)

        self.play(
            ShowCreation(solution_rect),
            Write(solution_word),
        )
        self.play(
            LaggedStartMap(Write, constant_labels, lag_ratio=0.5),
            LaggedStart(*(
                c_tracker.animate.set_value(value)
                for c_tracker, value in zip(c_trackers, target_values)
            ), lag_ratio=0.5),
            run_time=4
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, plusses),
            Write(solution_words["All"]),
            Restore(solution_word),
        )
        self.wait()

        # Play with constants
        self.play(LaggedStartMap(ShowCreation, const_rects, lag_ratio=0.15))
        value_sets = [
            [1, 1, 1, 1, 1],
            [1j, -1j, 1 + 1j, -1 + 1j, -0.5],
            [-0.5, 1j, 1j, 1 + 1j, -1],
        ]
        for values in value_sets:
            self.play(
                LaggedStart(*(
                    c_tracker.animate.set_value(value)
                    for c_tracker, value in zip(c_trackers, values)
                ), lag_ratio=0.25, run_time=3)
            )
            self.wait()
        self.play(LaggedStartMap(FadeOut, const_rects, lag_ratio=0.25))
        self.wait()

    def get_graph(self, axes, s, get_const):
        def func(t):
            return (get_const() * np.exp(s * t)).real

        graph = axes.get_graph(func, bind=True, stroke_color=TEAL, stroke_width=2)
        return graph


class HoldUpGeneralLinear(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(500)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("pondering", "thinking", "tease", look_at=3 * UR)
        )
        self.wait()
        self.play(
            morty.change("sassy", look_at=3 * UR),
            self.change_students("hesitant", "erm", "maybe")
        )
        self.wait(5)


class BigCross(InteractiveScene):
    def construct(self):
        cross = Cross(Rectangle(4, 1.5))
        cross.set_stroke(RED, width=(0, 8, 8, 8, 0))
        self.play(ShowCreation(cross))
        self.wait()


class DifferentialEquation(InteractiveScene):
    def construct(self):
        # ode to x
        x_term = Tex(R"x(t)", font_size=90)
        arrow = Vector(DOWN, thickness=5)
        arrow.move_to(ORIGIN, DOWN)
        words = Text("Differential Equation", font_size=72)
        words.next_to(arrow, UP)

        self.play(Write(x_term))
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(words),
            x_term.animate.next_to(arrow, DOWN),
        )
        self.wait()


class DumbTrickAlgebra(InteractiveScene):
    def construct(self):
        pass


class LaplaceTransformAlgebra(InteractiveScene):
    def construct(self):
        # Add equation
        colors = color_gradient([TEAL, RED], 3, interp_by_hsl=True)
        t2c = {
            R"x(t)": colors[0],
            R"x'(t)": colors[1],
            R"x''(t)": colors[2],
            "{s}": YELLOW,
        }
        equation = Tex(
            R"{m} x''(t) + \\mu x'(t) + k x(t) = F_0 \\cos(\\omega_l t)",
            t2c=t2c
        )
        equation.to_edge(UP, buff=1.5)

        arrow = Vector(1.25 * DOWN, thickness=6)
        arrow.next_to(equation, DOWN)
        arrow_label = Tex(R"\\mathcal{L}", font_size=72)
        arrow_label.next_to(arrow, RIGHT, buff=SMALL_BUFF)

        self.add(equation)
        self.wait()
        self.play(
            GrowArrow(arrow),
            FadeIn(arrow_label, shift=0.5 * DOWN)
        )

        # Make transformed
        transformed_eq = Tex(
            R"{m} {s}^2 X({s}) + \\mu {s} X({s}) + k X({s}) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2}",
            t2c=t2c
        )
        transformed_eq.next_to(arrow, DOWN)

        xt_texs = ["x(t)", "x'(t)", "x''(t)"]
        Xs_texs = ["X({s})", "{s} X({s})", "{s}^2 X({s})"]

        rects = VGroup()
        srcs = VGroup()
        trgs = VGroup()
        for t1, t2, color in zip(xt_texs, Xs_texs, colors):
            src = equation[t1][0]
            trg = transformed_eq[t2][-1]
            rect = SurroundingRectangle(src, buff=0.05)
            rect.set_stroke(color, 2)
            rect.target = rect.generate_target()
            rect.target.surround(trg, buff=0.05)

            rects.add(rect)
            srcs.add(src.copy())
            trgs.add(trg)

        self.play(LaggedStartMap(ShowCreation, rects, lag_ratio=0.25, run_time=1.5))
        self.play(
            LaggedStart(
                *(FadeTransform(src, trg)
                for src, trg in zip(srcs, trgs)),
                lag_ratio=0.25,
                group_type=Group,
                run_time=1.5
            ),
            LaggedStartMap(MoveToTarget, rects, lag_ratio=0.25, run_time=1.5)
        )
        self.play(
            LaggedStart(*(
                TransformFromCopy(equation[tex], transformed_eq[tex][:2])
                for tex in ["{m}", "+", R"\\mu", "k", "="]
            )),
            TransformMatchingParts(
                equation[R"F_0 \\cos(\\omega_l t)"].copy(),
                transformed_eq[R"\\frac{F_0 {s}}{{s}^2 + \\omega_l^2}"]
            )
        )
        self.wait()

        # Factor it
        factored = Tex(
            R"X({s}) \\left({m} {s}^2+ \\mu {s} + k\\right) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2}",
            t2c=t2c
        )
        factored.move_to(transformed_eq)
        left_rect = SurroundingRectangle(factored["X({s})"], buff=0.05)
        left_rect.set_stroke(YELLOW, 2)

        self.play(
            TransformMatchingTex(
                transformed_eq,
                factored,
                matched_keys=["X({s})"],
                path_arc=30 * DEG
            ),
            ReplacementTransform(rects, VGroup(left_rect), path_arc=30 * DEG),
            run_time=2
        )
        self.play(FadeOut(left_rect))
        self.wait()

        # Rearrange
        rearranged = Tex(
            R"X({s}) = \\frac{F_0 {s}}{{s}^2 + \\omega_l^2} \\frac{1}{{m} {s}^2+ \\mu {s} + k}",
            t2c=t2c
        )
        rearranged.next_to(factored, DOWN, LARGE_BUFF)

        self.play(
            TransformMatchingTex(
                factored.copy(),
                rearranged,
                matched_keys=["X({s})"],
            )
        )`,
    annotations: {
      4: "IntroduceTrilogy extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      150: "Class DiscussTrilogy inherits from TeacherStudentsScene.",
      151: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      182: "Class WhoCares inherits from TeacherStudentsScene.",
      183: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      198: "MiniLessonTitle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      199: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      206: "Class WeGotThis inherits from TeacherStudentsScene.",
      207: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      221: "Class ConfusionAndWhy inherits from TeacherStudentsScene.",
      222: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      249: "ArrowBetweenScreens extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      250: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      266: "WhatAndWhy extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      267: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      287: "PrequelToLaplace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      288: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      295: "OtherExponentialDerivatives extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      296: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      314: "VariousExponentials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      315: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      341: "WhyToWhat extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      342: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      372: "DerivativeOfExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      373: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      438: "HighlightRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      439: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      456: "DefineI extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      457: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      463: "Class WaitWhy inherits from TeacherStudentsScene.",
      464: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      474: "MultiplicationByI extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      475: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      610: "UnitArcLengthsOnCircle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      611: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      646: "SimpleIndicationRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      647: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      653: "WriteSPlane extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      654: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      661: "ODEStoExp extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      662: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      698: "GenLinearEquationToOscillator extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      699: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      735: "VLineOverZero extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      736: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      751: "KIsSomeConstant extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      752: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      763: "WriteMu extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      764: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      778: "Class ReferenceGuessingExp inherits from TeacherStudentsScene.",
      779: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      866: "FromGuessToLaplace extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      867: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      934: "JustAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      935: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      950: "BothPositiveNumbers extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      951: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      968: "Class ButSpringsAreReal inherits from TeacherStudentsScene.",
      969: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      982: "ShowIncreaseToK extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      983: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1007: "PureMathEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1008: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1046: "LinearityDefinition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1047: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1119: "Class ComplainAboutNeelessComplexity inherits from TeacherStudentsScene.",
      1120: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1147: "LetsGeneralize extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1148: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1158: "EquationRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1159: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1170: "GeneralLinearEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1171: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1461: "Class HoldUpGeneralLinear inherits from TeacherStudentsScene.",
      1462: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1479: "BigCross extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1480: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1487: "DifferentialEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1488: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1506: "DumbTrickAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1507: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1511: "LaplaceTransformAlgebra extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1512: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();