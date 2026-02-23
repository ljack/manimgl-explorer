(function() {
  const files = window.MANIM_DATA.files;

  files["_2025/grover/clarification.py"] = {
    description: "Clarification scenes for the Grover's algorithm video: addressing common misconceptions about quantum speedup and measurement.",
    code: `from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *


default_sudoku_values = [  # TODO
    [7, 4, 3, 6, 5, 1, 9, 8, 2],
    [1, 8, 9, 2, 3, 4, 7, 6, 5],
    [6, 2, 5, 9, 8, 7, 4, 3, 1],
    [3, 2, 5, 8, 6, 9, 4, 1, 7],
    [9, 7, 6, 3, 4, 1, 5, 2, 8],
    [1, 4, 8, 7, 5, 2, 3, 6, 9],
    [2, 3, 8, 5, 7, 6, 1, 9, 4],
    [4, 1, 7, 8, 9, 3, 6, 5, 2],
    [5, 9, 6, 2, 1, 4, 8, 7, 3],
]

default_sudoku_locked_cells = [
    [3, 7, 8],
    [1, 6, 7, 8],
    [3, 5],
    [1, 3, 6, 8],
    [5],
    [1, 3, 4, 8],
    [5, 6],
    [5],
    [4, 9],
]


class Sudoku(VGroup):
    def __init__(
        self,
        values=default_sudoku_values,
        locked_cells=default_sudoku_locked_cells,
        height=4,
        big_square_stroke_width=3,
        little_square_stroke_width=0.5,
        locked_number_color=BLUE_B,
        num_to_square_height_ratio=0.5
    ):
        self.big_grid = self.get_square_grid(height, big_square_stroke_width)
        self.little_grids = VGroup(
            self.get_square_grid(height / 3, little_square_stroke_width).move_to(square)
            for square in self.big_grid
        )
        self.numbers = VGroup(
            VGroup(
                Integer(num).replace(square, 1).scale(num_to_square_height_ratio)
                for square, num in zip(grid, arr)
            )
            for grid, arr in zip(self.little_grids, values)
        )
        self.locked_cells = locked_cells
        self.locked_numbers = VGroup()
        self.unlocked_numbers = VGroup()

        for coords, group in zip(locked_cells, self.numbers):
            for x, num in enumerate(group):
                if x in coords:
                    self.locked_numbers.add(num)
                else:
                    self.unlocked_numbers.add(num)
        self.locked_numbers.set_fill(locked_number_color, border_width=2)

        super().__init__(self.big_grid, self.little_grids, self.numbers)

    def get_rows(self):
        grids = self.numbers
        slices = [slice(0, 3), slice(3, 6), slice(6, 9)]
        rows = VGroup()
        for slc1 in slices:
            for slc2 in slices:
                row = VGroup()
                for grid in grids[slc1]:
                    for num in grid[slc2]:
                        row.add(num)
                rows.add(row)
        return rows

    def get_columns(self):
        grids = self.numbers
        slices = [slice(0, 9, 3), slice(1, 9, 3), slice(2, 9, 3)]
        cols = VGroup()
        for slc1 in slices:
            for slc2 in slices:
                col = VGroup()
                for grid in grids[slc1]:
                    for num in grid[slc2]:
                        col.add(num)
                cols.add(col)
        return cols

    def get_number_squares(self):
        return self.numbers

    def get_square_grid(
        self,
        height,
        stroke_width,
        stroke_color=WHITE,
        fill_color=GREY_E,
    ):
        square = Square()
        square.set_fill(fill_color, 1)
        square.set_stroke(stroke_color, stroke_width)
        grid = square.get_grid(3, 3, buff=0)
        grid.set_height(height)
        return grid


class Intro(InteractiveScene):
    random_seed = 1

    def construct(self):
        # Test
        background = FullScreenRectangle().set_fill(GREY_E)
        background.set_fill(GREY_E, 0.5)
        self.add(background)

        icon = get_quantum_computer_symbol(height=3)
        icon.center()
        self.play(Write(icon, run_time=3, lag_ratio=1e-2, stroke_color=TEAL))
        self.wait()
        self.play(icon.animate.to_edge(LEFT))
        self.wait()

        # Comments
        folder = Path('/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2025/blocks_and_grover/Comments')
        comments = Group(
            Group(ImageMobject(folder / name))
            for name in os.listdir(folder)
        )
        for comment in comments:
            comment.add_to_back(SurroundingRectangle(comment, buff=0).set_stroke(WHITE, 1))
            comment.set_width(4)
            comment.move_to(4 * LEFT)
            comment.shift(np.random.uniform(-1, 1) * RIGHT + np.random.uniform(-3, 3) * UP)

        self.play(
            FadeOut(icon, time_span=(0, 2)),
            LaggedStartMap(FadeIn, comments, lag_ratio=0.6, shift=0.25 * UP, run_time=8)
        )
        self.wait()


class HowDoYouKnowWhichAxis(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("guilty"),
            stds[2].says("Doesn't this assumes\\nknowing the key value?", mode="angry", bubble_direction=LEFT),
            stds[0].change("confused", look_at=self.screen),
            stds[1].change("pleading", look_at=self.screen),
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(5)
        self.play(self.change_students("erm", "confused", "sassy"))
        self.wait(5)


class SudokuChecker(InteractiveScene):
    def construct(self):
        # Introduce, and add solution
        sudoku = Sudoku(height=6)
        sudoku.unlocked_numbers.set_opacity(0)

        self.play(Write(sudoku, run_time=3, lag_ratio=1e-2))
        sudoku.unlocked_numbers.shuffle()
        self.play(sudoku.unlocked_numbers.animate.set_opacity(1).set_anim_args(lag_ratio=0.1), run_time=3)
        self.wait()

        # Check rows, columns, squares
        row_rects, col_rects, square_rects = rect_groups = VGroup(
            VGroup(
                VGroup(SurroundingRectangle(num) for num in sg)
                for sg in group
            )
            for group in [sudoku.get_rows(), sudoku.get_columns(), sudoku.get_number_squares()]
        )
        rect_groups.set_stroke(YELLOW, 3)

        for group in rect_groups:
            self.play(
                LaggedStart(*(
                    VFadeInThenOut(sg, lag_ratio=0.025)
                    for sg in group
                ), lag_ratio=0.5, run_time=5)
            )

        # Replace with question marks
        unlocked_numbers = sudoku.unlocked_numbers
        q_marks = VGroup(
            Text("?").replace(num, 1)
            for num in sudoku.unlocked_numbers
        )
        q_marks.set_color(RED)
        unlocked_numbers.save_state()
        self.play(
            Transform(unlocked_numbers, q_marks, lag_ratio=1e-2, run_time=2)
        )
        self.wait()
        self.play(Restore(unlocked_numbers))

        # Show randomization
        def randomize_numbers(numbers):
            for number in numbers:
                number.set_value(random.uniform(1, 9))

        self.play(UpdateFromFunc(sudoku.unlocked_numbers, randomize_numbers, run_time=10))


class SudokuCheckingCode(InteractiveScene):
    def construct(self):
        # Thanks Claude!
        code = Code("""
            def is_valid_sudoku(board):
                \\"\\"\\"
                Check if a completed Sudoku board is valid.

                Args:
                    board: A 9x9 list of lists where each
                    cell contains an integer from 1 to 9

                Returns:
                    bool: True if the solution is valid, False otherwise
                \\"\\"\\"
                # Check rows
                for row in board:
                    if set(row) != set(range(1, 10)):
                        return False

                # Check columns
                for col in range(9):
                    column = [board[row][col] for row in range(9)]
                    if set(column) != set(range(1, 10)):
                        return False

                # Check 3x3 sub-boxes
                for box_row in range(0, 9, 3):
                    for box_col in range(0, 9, 3):
                        # Get all numbers in the current 3x3 box
                        box = []
                        for i in range(3):
                            for j in range(3):
                                box.append(board[box_row + i][box_col + j])
                        if set(box) != set(range(1, 10)):
                            return False

                # If all checks pass, the solution is valid
                return True
        """, alignment="LEFT")
        code.set_height(7)
        self.play(ShowIncreasingSubsets(code, run_time=8, rate_func=linear))
        self.wait()


class ArrowToQC(InteractiveScene):
    def construct(self):
        # Test
        icon = get_quantum_computer_symbol(height=2.5)
        icon.center().to_edge(DOWN)
        arrow = Vector(1.5 * DOWN, thickness=6)
        arrow.next_to(icon, UP)

        self.play(
            GrowArrow(arrow),
            FadeIn(icon, DOWN)
        )
        self.wait()


class CompiledSudokuVerifier(InteractiveScene):
    def construct(self):
        # Set up
        sudoku = Sudoku(height=5)
        sudoku.to_edge(LEFT)

        machine = get_blackbox_machine(height=3, label_tex=R"\\text{Verifier}")
        machine.next_to(sudoku, RIGHT, LARGE_BUFF)
        machine[-1].scale(0.65).set_color(YELLOW)

        self.add(sudoku)
        self.add(machine)

        # Pile logic gates into the machine
        gates = VGroup(
            SVGMobject("and_gate"),
            SVGMobject("or_gate"),
            SVGMobject("not_gate"),
        )
        names = VGroup(Text(text) for text in ["AND", "OR", "NOT"])
        names.scale(0.5)
        names.set_fill(GREY_C)

        gates.set_height(0.65)
        gates.set_fill(GREY_B)
        gates.arrange(RIGHT, buff=MED_LARGE_BUFF)
        gates.next_to(machine, UP, MED_LARGE_BUFF)
        for name, gate in zip(names, gates):
            name.next_to(gate, UP)

        pile_of_gates = VGroup(*it.chain(*(g.replicate(200) for g in gates)))
        pile_of_gates.shuffle()
        pile_of_gates.set_fill(opacity=0.25)
        pile_of_gates.generate_target()
        for gate in pile_of_gates.target:
            gate.scale(0.25)
            shift = np.random.uniform(-1.5, 1.5, 3)
            shift[2] = 0
            gate.move_to(machine.get_center() + shift)
            gate.set_fill(opacity=0.1)

        self.add(gates, names)
        self.play(
            MoveToTarget(pile_of_gates, lag_ratio=3.0 / len(pile_of_gates), run_time=8)
        )

        # Turn sudoku into binary
        all_numbers = VGroup(num for grid in sudoku.numbers for num in grid)
        bit_groups = VGroup(
            BitString(num.get_value()).replace(num, 1).scale(0.35)
            for num in all_numbers
        )

        self.play(
            ReplacementTransform(all_numbers, bit_groups, lag_ratio=1e-3),
            sudoku.big_grid.animate.set_fill(opacity=0.1).set_stroke(opacity=0.25),
            sudoku.little_grids.animate.set_fill(opacity=0.1).set_stroke(opacity=0.25),
        )
        self.wait()

        target_bits = bit_groups.copy()
        target_bits.arrange(DOWN, buff=0.025)
        target_bits.set_height(machine.get_height() * 0.5)
        target_bits.move_to(machine, LEFT)
        target_bits.set_opacity(0.1)
        self.play(TransformFromCopy(bit_groups, target_bits, lag_ratio=1e-3, run_time=2))
        self.wait()

        # Show outputs
        outputs = VGroup(Integer(1), Integer(0))
        outputs.set_height(0.75)
        outputs.next_to(machine, RIGHT, LARGE_BUFF)
        marks = VGroup(Checkmark().set_color(GREEN), Exmark().set_color(RED))
        for mark, output in zip(marks, outputs):
            mark.match_height(output)
            mark.next_to(output, RIGHT)
            self.play(FadeIn(output, RIGHT))
            self.play(Write(mark))
            self.wait()
            self.play(FadeOut(mark), FadeOut(output))


class StateVectorsAsABasis(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2, 1)
        axes = ThreeDAxes(x_range, y_range, z_range)
        axes.set_height(4)

        basis_vectors = VGroup(
            Vector(2 * vect, thickness=4)
            for vect in np.identity(3)
        )
        basis_vectors.set_submobject_colors_by_gradient(BLUE_D, BLUE_B)
        for vect in basis_vectors:
            vect.rotate(90 * DEG, axis=vect.get_vector())

        frame.reorient(-23, 81, 0, (-1.0, 0, 0.5), 4)
        frame.add_ambient_rotation()
        self.add(axes)

        # Bit strings
        two_qubits = VGroup(
            KetGroup(BitString(n, length=2))
            for n in range(4)
        )
        four_qubits = VGroup(
            KetGroup(BitString(n, length=4))
            for n in range(16)
        )
        for group in [two_qubits, four_qubits]:
            group.fix_in_frame()
            group.arrange(DOWN)
            group.set_max_height(7)
            group.to_edge(LEFT, buff=LARGE_BUFF)
        two_qubits.scale(1.5, about_edge=LEFT).space_out_submobjects(1.25)

        basis_labels = VGroup(
            two_qubits[0].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[0].get_end(), OUT, SMALL_BUFF),
            two_qubits[1].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[1].get_end(), OUT, SMALL_BUFF),
            two_qubits[2].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[2].get_end(), RIGHT, SMALL_BUFF),
        )
        basis_labels.unfix_from_frame()

        self.play(LaggedStartMap(FadeIn, two_qubits, shift=0.5 * UP, lag_ratio=0.5))

        for src, trg, vect in zip(two_qubits, basis_labels, basis_vectors):
            self.play(
                TransformFromCopy(src, trg),
                GrowArrow(vect)
            )
        self.wait(2)

        # Name basis vectors
        basis_name = TexText(R"\`\`Basis vectors''")
        basis_name.fix_in_frame()
        basis_name.set_color(BLUE)
        basis_name.to_corner(UR, buff=MED_SMALL_BUFF)
        self.play(Write(basis_name))
        self.wait(10)
        self.play(FadeOut(basis_name))

        # Replace with larger vectors
        new_basis_labels = VGroup(
            four_qubits[0].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[0].get_end(), OUT, SMALL_BUFF),
            four_qubits[1].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[1].get_end(), OUT, SMALL_BUFF),
            four_qubits[2].copy().scale(0.3).rotate(90 * DEG, RIGHT).next_to(basis_vectors[2].get_end(), RIGHT, SMALL_BUFF),
        )
        new_basis_labels.unfix_from_frame()
        self.play(
            FadeOut(two_qubits),
            FadeOut(basis_labels),
            LaggedStartMap(FadeIn, four_qubits, lag_ratio=0.25)
        )

        for src, trg in zip(four_qubits, new_basis_labels):
            self.play(TransformFromCopy(src, trg))
        self.wait(20)


class OperationsOnQC(InteractiveScene):
    def construct(self):
        # Bad output
        icon = get_quantum_computer_symbol(height=3)
        icon.center()
        in_ket = KetGroup(BitString(12).scale(2))
        in_ket.next_to(icon, LEFT)
        arrows = Vector(RIGHT, thickness=4).replicate(2)
        arrows[0].next_to(icon, LEFT)
        arrows[1].next_to(icon, RIGHT)
        in_ket.next_to(arrows, LEFT)

        bad_output = VGroup(Text("True"), Text("or"), Text("False"))
        bad_output.scale(1.5)
        bad_output.arrange(DOWN)
        bad_output.next_to(arrows, RIGHT)
        big_cross = Cross(bad_output)
        big_cross.scale(1.25)
        big_cross.set_stroke(RED, [0, 12, 12, 12, 0])

        self.add(icon, arrows, in_ket)
        self.play(LaggedStart(
            FadeOut(in_ket.copy(), shift=2 * RIGHT, scale=0.5),
            FadeIn(bad_output, shift=2 * RIGHT, scale=2),
            lag_ratio=0.5
        ))
        self.play(ShowCreation(big_cross))
        self.wait()
        self.play(FadeOut(in_ket), FadeOut(bad_output), FadeOut(big_cross))
        self.wait()


class TwoByTwoGrid(InteractiveScene):
    random_seed = 1

    def construct(self):
        # Set up
        rects = ScreenRectangle().set_height(FRAME_HEIGHT / 2).get_grid(2, 2, buff=0)
        h_line = Line(LEFT, RIGHT).replace(rects, 0)
        v_line = Line(UP, DOWN).replace(rects, 1)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(WHITE, 2)

        self.add(lines)

        # Add classical verifiers
        verifiers = get_blackbox_machine(label_tex="").replicate(4)
        for n, verifier, rect in zip(it.count(), verifiers, rects):
            verifier.set_height(1.0)
            verifier.move_to(rect)
            if n < 2:
                label = Text("Verifier", font_size=24)
                label.set_color(YELLOW)
            else:
                verifier.rotate(90 * DEG)
                label = get_quantum_computer_symbol(height=0.75)

            label.move_to(verifier)
            verifier.add(label)

        verifiers[:2].shift(0.75 * RIGHT)

        good_sudoku, bad_sudoku = sudokus = VGroup(
            Sudoku(big_square_stroke_width=2, little_square_stroke_width=0.25)
            for x in range(2)
        )
        sudokus.set_height(2.5)

        for number in bad_sudoku.unlocked_numbers:
            number.set_value(random.randint(1, 9))

        classical_outputs = VGroup(Integer(1), Integer(0))
        marks = VGroup(Checkmark().set_color(GREEN), Exmark().set_color(RED))

        for sudoku, verifier, output, mark in zip(sudokus, verifiers, classical_outputs, marks):
            sudoku.next_to(verifier, LEFT, MED_LARGE_BUFF)
            output.next_to(verifier, RIGHT, MED_LARGE_BUFF)
            mark.match_height(output)
            mark.next_to(output, RIGHT, SMALL_BUFF)

        self.add(verifiers[:2])
        self.add(sudokus)

        for sudoku, verifier, output, mark in zip(sudokus, verifiers, classical_outputs, marks):
            self.play(LaggedStart(
                FadeOutToPoint(sudoku.numbers.copy(), verifier.get_center(), lag_ratio=1e-3),
                FadeInFromPoint(output, verifier.get_center()),
                lag_ratio=0.5,
                run_time=2
            ))
            self.play(Write(mark))

        self.wait()

        # Map to quantum verifiers
        self.play(TransformFromCopy(verifiers[:2], verifiers[2:], run_time=2))
        self.wait()

        # Translate True behavior
        good_input = self.turn_into_bits(good_sudoku)
        good_ket = KetGroup(good_input.copy(), height_scale_factor=1.5)
        good_ket.next_to(verifiers[2], UP)
        good_ket_out = good_ket.copy()
        neg = Tex("-", font_size=24)
        neg.set_fill(GREEN, border_width=3)
        neg.next_to(good_ket_out, LEFT, SMALL_BUFF)
        good_ket_out.add_to_back(neg)
        good_ket_out.next_to(verifiers[2], DOWN, buff=0.2)

        mult_neg_1_words = TexText(R"Multiply\\\\by $-1$", font_size=36)
        mult_neg_1_words.set_fill(TEAL_A)
        mult_neg_1_words.next_to(verifiers[2], RIGHT, MED_LARGE_BUFF)

        self.play(FadeTransform(good_input.copy(), good_ket))
        self.wait()
        self.play(
            LaggedStart(
                FadeOutToPoint(good_ket.copy(), verifiers[2].get_center(), lag_ratio=0.01),
                FadeInFromPoint(good_ket_out, verifiers[2].get_center(), lag_ratio=0.01),
                lag_ratio=0.5
            ),
            FadeIn(mult_neg_1_words, RIGHT),
        )
        self.wait()

        # Translate False behavior (a lot of coying, but I'm in a rush)
        bad_input = self.turn_into_bits(bad_sudoku)
        bad_ket = KetGroup(bad_input.copy(), height_scale_factor=1.5)
        bad_ket.next_to(verifiers[3], UP)
        bad_ket_out = bad_ket.copy()
        bad_ket_out.next_to(verifiers[3], DOWN)

        mult_pos_1_words = TexText(R"Multiply\\\\by $+1$", font_size=36)
        mult_pos_1_words.set_fill(TEAL_A)
        mult_pos_1_words.next_to(verifiers[3], RIGHT, MED_LARGE_BUFF)

        self.play(FadeTransform(bad_input.copy(), bad_ket))
        self.wait()
        self.play(
            LaggedStart(
                FadeOutToPoint(bad_ket.copy(), verifiers[3].get_center(), lag_ratio=0.01),
                FadeInFromPoint(bad_ket_out, verifiers[3].get_center(), lag_ratio=0.01),
                lag_ratio=0.5
            ),
            FadeIn(mult_pos_1_words, RIGHT),
        )
        self.wait()

        # Show logic gates
        gate_groups = VGroup(
            SVGMobject("and_gate").replicate(2),
            SVGMobject("not_gate").replicate(2),
            SVGMobject("or_gate").replicate(2),
        )
        for group in gate_groups:
            group.set_fill(BLUE)
            group.target = group.generate_target()
            for mob, box in zip(group, verifiers):
                mob.match_height(box)
                mob.scale(0.8)
                mob.next_to(box, UP)
            for mob, box in zip(group.target, verifiers[2:]):
                mob.match_width(box)
                mob.scale(0.7)
                mob.move_to(box)
                mob.set_fill(TEAL, 0.5)

        for group in gate_groups:
            self.play(LaggedStartMap(FadeIn, group, shift=UP, lag_ratio=0.5))
            self.wait()
            self.play(TransformFromCopy(group, group.target))
            self.play(FadeOut(group.target))
            self.play(FadeOut(group))

    def turn_into_bits(self, sudoku):
        # Test
        bits = VGroup(
            BitString(num.get_value()).replace(num, 1).scale(0.35)
            for grid in sudoku.numbers for num in grid
        )
        sudoku.save_state()

        in_group = VGroup(
            bits[0].copy().set_width(0.25),
            Tex(R"\\cdots", font_size=20),
            bits[-1].copy().set_width(0.25),
        )
        for piece in in_group:
            piece.space_out_submobjects(0.85)
        in_group[1].scale(0.7)
        in_group.arrange(RIGHT, buff=0.025)
        in_group.replace(sudoku, 0)
        in_group.set_fill(GREY_A)

        sudoku.saved_state.scale(0.5)
        sudoku.saved_state.fade(0.5)
        sudoku.saved_state.to_edge(UP, buff=MED_SMALL_BUFF)

        self.play(
            sudoku.animate.fade(0.9),
            FadeIn(bits, lag_ratio=1e-3)
        )
        self.play(LaggedStart(
            ReplacementTransform(bits[0], in_group[0]),
            *(ReplacementTransform(bs, in_group[1]) for bs in bits[1:-1]),
            ReplacementTransform(bits[-1], in_group[-1]),
            Restore(sudoku),
            lag_ratio=1e-2,
        ))
        return in_group


class AskWhyThatsTrue(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            stds[1].says("How does that work?", mode="maybe"),
            stds[0].change('erm', look_at=self.screen),
            stds[2].change("confused", look_at=self.screen),
            morty.change('hesitant'),
        )
        self.look_at(self.screen)
        self.wait(5)

        # Mapping
        mapping = VGroup(
            VGroup(Text("True").set_color(GREEN), Vector(DOWN), Tex(R"\\times -1")).arrange(DOWN),
            VGroup(Text("False").set_color(RED), Vector(DOWN), Tex(R"\\times +1")).arrange(DOWN),
        )
        for part in mapping:
            tex = part[2]
            tex[1].scale(0.75, about_point=tex[2].get_left())
            tex[0].scale(1.5)
        mapping.arrange(RIGHT, buff=2.0)
        mapping.move_to(self.hold_up_spot, DOWN).shift(LEFT)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(mapping, UP),
            stds[1].debubble(mode="pondering"),
            stds[0].change("sassy", mapping),
            stds[2].change("hesitant", mapping),
        )
        self.wait()
        self.play(morty.change("hesitant"))
        self.wait(3)
        self.play(morty.change("pondering", mapping))
        self.wait(3)


class ListOfConfusions(InteractiveScene):
    def construct(self):
        # Test
        items = BulletedList(
            "Insufficient detail",
            "Bad framing",
            "Glossing over linearity",
            buff=1.0
        )
        rects = VGroup(
            SurroundingRectangle(item[1:])
            for item in items
        )
        rects.set_stroke(width=0)
        rects.set_fill(GREY_D, 1)

        self.add(items)
        self.add(rects[1:])
        self.wait()
        self.play(
            items[0].animate.fade(0.5),
            rects[1].animate.stretch(0, 0, about_edge=RIGHT),
        )
        self.wait()
        self.play(
            items[1].animate.fade(0.5),
            rects[2].animate.stretch(0, 0, about_edge=RIGHT),
        )
        self.wait()


class SolveSHAWord(InteractiveScene):
    def construct(self):
        words = VGroup(
            TexText(R"Solve for \${x}$"),
            Tex(R"\\text{SHA256}({x}) = 0"),
        )
        words.set_color(GREY_B)
        for word in words:
            word["{x}"].set_color(YELLOW)
        words.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        self.play(Write(words))
        self.wait()


class ThatsOnMe(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.says("That's on me", mode="guilty"),
            self.change_students("pondering", "well", "hesitant", look_at=self.screen),
        )
        self.wait(3)


class ShowSuperposition(InteractiveScene):
    def construct(self):
        # Axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2, 1)
        axes = ThreeDAxes(x_range, y_range, z_range)
        unit_size = 3
        axes.set_height(2 * unit_size)

        basis_vectors = VGroup(
            Vector(unit_size * vect, thickness=5)
            for vect in np.identity(3)
        )
        basis_vectors.set_submobject_colors_by_gradient(BLUE_D, BLUE_B)
        for n, vect in enumerate(basis_vectors):
            if n < 2:
                vect.always.set_perpendicular_to_camera(frame)
            else:
                vect.rotate(90 * DEG, OUT)

        frame.reorient(-33, 74, 0)
        frame.clear_updaters()
        frame.add_ambient_rotation(0.5 * DEG)
        self.add(axes)
        self.add(basis_vectors)

        # Label bases
        basis_labels = VGroup(
            KetGroup(BitString(n, length=2)).set_height(0.35)
            for n in range(3)
        )
        basis_labels.rotate(90 * DEG, RIGHT)
        for label, vect, direction in zip(basis_labels, basis_vectors, [RIGHT, UP + OUT, RIGHT]):
            label.next_to(vect.get_end(), direction, MED_SMALL_BUFF)
            label.set_fill(vect.get_color(), 1)

        self.add(basis_labels, basis_vectors)
        self.wait(4)

        # Show a general vector
        vect_coords = normalize([np.sqrt(2) / 2, 0.5, 0.5])
        vector = Vector(unit_size * vect_coords, thickness=5, fill_color=TEAL)
        vector.set_perpendicular_to_camera(frame)
        dec_kw = dict(include_sign=True, font_size=36)
        vect_label = VGroup(
            DecimalNumber(vect_coords[0], **dec_kw),
            KetGroup(BitString(0, 2)),
            DecimalNumber(vect_coords[1], **dec_kw),
            KetGroup(BitString(1, 2)),
            DecimalNumber(vect_coords[2], **dec_kw),
            KetGroup(BitString(2, 2)),
            DecimalNumber(0, **dec_kw),
            KetGroup(BitString(3, 2)),
        )
        vect_label[1::2].set_submobject_colors_by_gradient(BLUE, BLUE_E)
        vect_label.arrange(RIGHT, buff=SMALL_BUFF)
        vect_label.rotate(90 * DEG, RIGHT)
        vect_label.next_to(vector.get_end(), RIGHT)

        self.play(
            GrowArrow(vector),
            frame.animate.set_x(1),
        )
        self.play(FadeIn(vect_label))
        self.wait(5)

        # Show column vector
        col = DecimalMatrix(np.array([*vect_coords, 0]).reshape(-1, 1), decimal_config=dec_kw)
        col.scale(0.75)
        col.rotate(90 * DEG, RIGHT)
        col.next_to(vector.get_end(), RIGHT)
        eq = Tex("=")
        eq.rotate(90 * DEG, RIGHT)
        eq.next_to(col, RIGHT, SMALL_BUFF)

        self.play(
            FadeIn(col.get_brackets()),
            TransformFromCopy(vect_label[0::2], col.get_entries(), run_time=2),
            FadeIn(eq),
            vect_label.animate.next_to(eq, RIGHT, SMALL_BUFF)
        )

        # Surrounding rectangles
        rects = VGroup(
            SurroundingRectangle(mob.copy().rotate(90 * DEG, LEFT)).rotate(90 * DEG, RIGHT)
            for mob in [col, vect_label]
        )
        rects.set_stroke(YELLOW, 2)
        self.play(ShowCreation(rects[0]))
        self.wait(5)
        self.play(Transform(*rects))
        self.wait(7)

        # Write superposition
        word = TexText("\`\`Superposition''", font_size=72)
        word.rotate(90 * DEG, RIGHT)
        word.set_color(YELLOW)
        word.next_to(rects[0], OUT)

        self.play(Write(word), frame.animate.set_x(2), run_time=3)
        self.wait(8)


class NorthEastTraveler(InteractiveScene):
    def construct(self):
        # Add compass
        compass = self.get_compass()
        compass.move_to(5 * RIGHT + 2.5 * UP)
        self.add(compass)

        # Show travler
        randy = Randolph(height=1, mode="tease")
        vel_vect = Vector(2 * RIGHT, thickness=4, color=YELLOW)
        vel_vect.move_to(randy.get_bottom(), LEFT)
        travler = VGroup(randy, vel_vect)
        travler.rotate(45 * DEG) 
        travler.move_to(4 * LEFT + 2 * DOWN)

        self.add(travler)
        self.play(travler.animate.shift(3 * UR), run_time=5, rate_func=linear)

        # Show components
        components = VGroup(
            Vector(math.sqrt(2) * UP).shift(math.sqrt(2) * RIGHT).set_fill(GREEN),
            Vector(math.sqrt(2) * RIGHT).set_fill(RED)
        )
        components.shift(vel_vect.get_start() - components[1].get_start())
        labels = VGroup(
            Text("North").next_to(components[0], RIGHT, buff=-0.05),
            Text("East").next_to(components[1], DOWN, SMALL_BUFF),
        )

        for component, label in zip(components, labels):
            label.scale(0.75)
            label.match_color(component)
            self.play(
                GrowArrow(component),
                FadeIn(label)
            )
        self.wait()

        # Show sum
        sum_expr = VGroup(labels[0].copy(), Tex(R"+", font_size=24), labels[1].copy())
        sum_expr.arrange(RIGHT, buff=SMALL_BUFF)
        sum_expr.next_to(vel_vect.get_end(), UP)
        self.play(
            TransformFromCopy(labels, sum_expr[0::2]),
            Write(sum_expr[1])
        )
        self.wait()

        # Do a 90 degree rotation
        north_group = VGroup(components[0], labels[0])
        east_group = VGroup(components[1], labels[1])
        self.play(
            FadeOut(sum_expr),
            north_group.animate.shift(DR).fade(0.5),
            east_group.animate.shift(DR).fade(0.5),
        )
        self.wait()

        self.add(travler.copy().fade(0.75))
        t_rot_marks = self.show_90_degree_rotation(travler, 45 * DEG, about_point=vel_vect.get_start())
        self.wait()

        self.play(
            north_group.animate.set_fill(opacity=1).shift(UR),
            FadeOut(t_rot_marks)
        )
        n_rot_marks = self.show_90_degree_rotation(components[0], 90 * DEG, about_point=components[0].get_start())
        self.wait()
        self.play(
            east_group.animate.set_fill(opacity=1).shift(DR + DOWN)
        )
        e_rot_marks = self.show_90_degree_rotation(components[1], 0, about_point=components[1].get_start())

        self.add(components.copy().set_opacity(0.5))
        self.play(components[1].animate.shift(vel_vect.get_start() - components[1].get_start()))
        self.play(components[0].animate.shift(components[1].get_end() - components[0].get_start()))
        self.wait()
        self.play(LaggedStart(*(
            Rotate(mob, -90 * DEG, about_point=vel_vect.get_start(), run_time=6, rate_func=there_and_back_with_pause)
            for mob in [travler, *components]
        ), lag_ratio=0.05))

    def get_compass(self):
        spike = Triangle()
        spike.set_shape(0.25, 1)
        spike.move_to(ORIGIN, DOWN)
        spikes = VGroup(
            spike.copy().rotate(x * TAU / 4, about_point=ORIGIN)
            for x in range(4)
        )
        lil_spikes = spikes.copy().rotate(45 * DEG).scale(0.75)
        dot = Circle(radius=0.25)
        compass = VGroup(spikes, lil_spikes, dot)
        compass.set_stroke(width=0)
        compass.set_fill(GREY_D, 1)
        compass.set_shading(0.5, 0.5, 1)

        labels = VGroup(map(Text, "NWSE"))
        labels.scale(0.5)
        for label, spike, vect in zip(labels, spikes, compass_directions(4, start_vect=UP)):
            label.next_to(spike, np.round(vect), SMALL_BUFF)
        compass.add(labels)

        return compass

    def show_90_degree_rotation(self, mobject, start_angle, about_point, radius=1.5):
        arc = Arc(start_angle, 90 * DEG, arc_center=about_point, radius=radius)
        arc.add_tip(width=0.2, length=0.2)
        arc.set_color(YELLOW)
        label = Tex(R"90^\\circ")
        midpoint = arc.pfp(0.5)
        label.next_to(midpoint, normalize(midpoint - about_point))

        self.play(LaggedStart(
            ShowCreation(arc),
            FadeIn(label),
            Rotate(mobject, 90 * DEG, about_point=about_point),
            lag_ratio=0.5
        ))

        return VGroup(arc, label)


class SimpleTwobitKet(InteractiveScene):
    def construct(self):
        group = KetGroup(BitString(2, 2))
        group.scale(2)
        self.add(group)


class ShowLinearityExample(InteractiveScene):
    def construct(self):
        # Add machine
        machine = get_blackbox_machine()
        machine[-1].set_color(TEAL)
        icon = get_quantum_computer_symbol(height=1)
        icon.next_to(machine, DOWN)

        self.add(machine, icon)
        self.wait()

        # Show a weighted sum
        kets = VGroup(
            KetGroup(BitString(n, 2))
            for n in range(4)
        )
        kets.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        kets.set_submobject_colors_by_gradient(BLUE, BLUE_E)
        kets.next_to(machine, LEFT, MED_LARGE_BUFF)

        components = VGroup(
            DecimalNumber(n, include_sign=True, font_size=42)
            for n in normalize([1, -2, 3, -4])
        )
        for component, ket in zip(components, kets):
            component.next_to(ket, LEFT, SMALL_BUFF)

        in_group = VGroup(components, kets)
        self.play(FadeIn(in_group, RIGHT))
        self.wait()
        self.play(
            FadeOutToPoint(in_group.copy(), machine.get_left() + RIGHT, lag_ratio=0.01, run_time=2)
        )

        # Show output
        out_kets = VGroup()
        for ket in kets:
            f_group = Tex(R"f()", font_size=60)
            f_group.set_color(TEAL)
            f_group[:2].next_to(ket, LEFT, SMALL_BUFF)
            f_group[2].next_to(ket, RIGHT, SMALL_BUFF)
            out_ket = VGroup(f_group, ket.copy())
            out_kets.add(out_ket)

        out_components = components.copy()
        out_kets.next_to(machine, RIGHT, buff=1.5)
        for out_ket, component in zip(out_kets, out_components):
            component.next_to(out_ket, LEFT, SMALL_BUFF)

        out_group = VGroup(out_components, out_kets)

        self.play(
            FadeInFromPoint(out_group, machine.get_right() + LEFT, lag_ratio=0.01, run_time=2)
        )
        self.wait()

        # Highlight components
        in_rects = VGroup(map(SurroundingRectangle, components))
        out_rects = VGroup(map(SurroundingRectangle, out_components))

        self.play(
            LaggedStartMap(VFadeInThenOut, in_rects, lag_ratio=0.5),
            LaggedStartMap(VFadeInThenOut, out_rects, lag_ratio=0.5),
            run_time=5,
        )
        self.wait()


class ZGateExample(InteractiveScene):
    def construct(self):
        # Test
        z_gates = VGroup(get_blackbox_machine(label_tex="Z") for n in range(3))
        z_gates.scale(0.35)
        z_gates.arrange(DOWN, buff=LARGE_BUFF)
        z_gates.move_to(4 * LEFT + UP)
        for gate in z_gates:
            gate[-1].set_color(TEAL)
            gate[-1].scale(1.5)

        zero, one = kets = VGroup(
            KetGroup(Integer(0)),
            KetGroup(Integer(1)),
        )
        gen_input = VGroup(Tex(R"x"), zero.copy(), Tex(R"+"), Tex(R"y"), one.copy())
        gen_input.arrange(RIGHT, buff=SMALL_BUFF)
        inputs = VGroup(zero, one, gen_input)
        for in_group, gate in zip(inputs, z_gates):
            in_group.next_to(gate, LEFT)

        # Act on zero, then one
        outputs = VGroup(
            zero.copy().next_to(z_gates[0], RIGHT),
            VGroup(Tex("-"), one.copy()).arrange(RIGHT, buff=SMALL_BUFF).next_to(z_gates[1], RIGHT)
        )

        self.play(FadeIn(z_gates[0]))
        self.wait()
        self.play(FadeIn(zero, RIGHT))
        self.play(LaggedStart(
            FadeOutToPoint(zero.copy(), z_gates[0].get_center(), lag_ratio=0.05),
            FadeInFromPoint(outputs[0], z_gates[0].get_center(), lag_ratio=0.05),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(
            TransformFromCopy(*z_gates[:2]),
            TransformFromCopy(*kets),
        )
        self.play(LaggedStart(
            FadeOutToPoint(one.copy(), z_gates[1].get_center(), lag_ratio=0.05),
            FadeInFromPoint(outputs[1], z_gates[1].get_center(), lag_ratio=0.05),
            lag_ratio=0.5
        ))
        self.wait()

        # General input
        self.play(
            TransformFromCopy(*z_gates[1:3]),
            TransformFromCopy(kets, gen_input[1::3]),
            *(FadeIn(gen_input[i]) for i in [0, 2, 3])
        )
        self.wait()

        rhss = VGroup(
            Tex(R"xZ|0\\rangle + y Z|1\\rangle ", t2c={"Z": TEAL}),
            Tex(R"x |0\\rangle - y |1\\rangle"),
        )
        rhss[0].next_to(z_gates[2], RIGHT)
        rhss[1].next_to(rhss[0], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            FadeOutToPoint(gen_input.copy(), z_gates[2].get_center(), lag_ratio=0.01),
            FadeInFromPoint(rhss[0], z_gates[2].get_center(), lag_ratio=0.01),
            lag_ratio=0.5
        ))
        self.wait()
        self.play(FadeIn(rhss[1], DOWN))
        self.wait()


class PassSuperpositionIntoVerifier(InteractiveScene):
    def construct(self):
        # Set up machine
        frame = self.frame
        frame.shift(1.5 * LEFT)
        machine = get_blackbox_machine(label_tex="V")
        machine[-1].set_color(TEAL)
        self.add(machine)

        # Bit string to general input
        bit_ket = KetGroup(Tex("0111...0011"))
        bit_ket.next_to(machine, LEFT, MED_LARGE_BUFF)

        basis_kets = VGroup(
            KetGroup(Tex(R"b_0")),
            KetGroup(Tex(R"b_1")),
            Tex(R"\\vdots"),
            KetGroup(Tex(R"b_k")),
            Tex(R"\\vdots"),
            KetGroup(Tex(R"b_{n-2}")),
            KetGroup(Tex(R"b_{n-1}")),
        )
        basis_kets.set_submobject_colors_by_gradient(BLUE, BLUE_E)
        basis_kets.arrange(DOWN)
        basis_kets.next_to(machine, LEFT, MED_LARGE_BUFF)
        components = VGroup(
            Tex(R"x_0"), Tex(R"+x_1"), VectorizedPoint(), Tex("+x_k"), VectorizedPoint(), Tex("+x_{n-2}"), Tex(R"+x_{n-1}")
        )
        for comp, ket in zip(components, basis_kets):
            comp.next_to(ket, LEFT, SMALL_BUFF)

        basis_label = Text("Basis")
        basis_label.next_to(bit_ket, LEFT)
        brace = Brace(components, LEFT)
        superposition_word = brace.get_text("Superposition")

        self.add(bit_ket)
        self.play(FadeIn(basis_label))
        self.wait()
        self.play(
            Transform(bit_ket, basis_kets[3]),
            basis_label.animate.next_to(basis_kets[3], LEFT),
        )
        self.wait()
        self.play(
            FadeTransform(basis_label, superposition_word),
            GrowFromCenter(brace),
            FadeIn(basis_kets, lag_ratio=0.1, run_time=2),
            FadeIn(components, lag_ratio=0.1, run_time=2),
        )
        self.remove(bit_ket)
        self.wait()

        # Show the output
        out_kets = basis_kets.copy()
        out_Vs = VGroup(
            Tex(R"V").set_color(TEAL).next_to(ket, LEFT, SMALL_BUFF)
            for ket in out_kets
        )
        VGroup(out_Vs[2], out_Vs[4]).set_opacity(0)
        out_comps = components.copy()
        out_comps.shift(out_Vs[0].get_width() * LEFT)
        out_group = VGroup(out_comps, out_Vs, out_kets)
        out_group.next_to(machine, RIGHT)

        in_group = VGroup(components, basis_kets)

        self.play(
            LaggedStart(
                FadeOutToPoint(in_group.copy(), machine.get_left() + RIGHT, lag_ratio=0.01),
                FadeInFromPoint(out_group, machine.get_right() + LEFT, lag_ratio=0.01),
                lag_ratio=0.5,
                run_time=2
            ),
            frame.animate.center(),
            FadeOut(brace),
            FadeOut(superposition_word),
        )
        self.wait()

        # Show unchanged parts
        last_annotation = VGroup()
        for n in [0, 1, -2, -1]:
            comp = out_comps[n]
            V = out_Vs[n]
            ket = out_kets[n]

            rect = SurroundingRectangle(VGroup(comp, V, ket))
            rect.set_stroke(YELLOW)
            label = Text("Unchanged", font_size=36)
            label.set_color(YELLOW)
            label.next_to(rect, RIGHT)
            annotation = VGroup(rect, label)
            self.play(LaggedStart(
                FadeOut(last_annotation),
                FadeIn(annotation),
                FadeOut(V),
                comp.animate.shift(V.get_width() * RIGHT),
            ))
            last_annotation = annotation
        self.play(FadeOut(annotation))

        # Show key solution
        sol_index = 3
        solution_label = Text("Sudoku\\nSolution")
        solution_label.next_to(components[sol_index], LEFT, LARGE_BUFF)
        solution_arrow = Arrow(solution_label, components[sol_index])

        out_rect = SurroundingRectangle(
            VGroup(out_comps[sol_index], out_Vs[sol_index], out_kets[sol_index])
        )
        out_rect.set_stroke(YELLOW)
        flip_word = Text("Flip!")
        flip_word.set_color(YELLOW)
        flip_word.next_to(out_rect, RIGHT)
        new_out_comp = Tex(R"-x_k")
        new_out_comp.next_to(out_kets[sol_index], LEFT, SMALL_BUFF)

        self.play(
            FadeIn(out_rect),
            FadeIn(flip_word),
            Transform(out_comps[sol_index], new_out_comp),
            FadeOut(out_Vs[sol_index], scale=0.25),
        )
        self.play(
            Write(solution_label),
            GrowArrow(solution_arrow)
        )
        self.wait()

        # Parallelization
        lines = VGroup(
            Line(basis_kets[n].get_right(), out_comps[n].get_left(), buff=SMALL_BUFF)
            for n in range(7)
        )

        new_machines = VGroup(machine.copy().scale(0.25).move_to(line) for line in lines)

        pre_machine = VGroup(machine.copy())
        self.remove(machine)
        self.add(lines, Point(), pre_machine)
        self.play(
            Transform(pre_machine, new_machines),
            LaggedStartMap(ShowCreation, lines)
        )
        self.wait()
        self.play(
            FadeOut(lines),
            FadeOut(pre_machine),
            FadeIn(machine),
        )
        self.wait()

        # Column vector
        in_col = TexMatrix(np.array(["x_0", "x_1", R"\\vdots", "x_k", R"\\vdots", R"x_{n-2}", R"x_{n-1}"]).reshape(-1, 1))
        out_col = TexMatrix(np.array(["x_0", "x_1", R"\\vdots", "-x_k", R"\\vdots", R"x_{n-2}", R"x_{n-1}"]).reshape(-1, 1))

        for col, vect in zip([in_col, out_col], [LEFT, RIGHT]):
            col.match_height(basis_kets)
            col.next_to(machine, vect)

        self.play(
            FadeOut(solution_label),
            FadeOut(solution_arrow),
            FadeOut(basis_kets),
            FadeOut(out_kets),
            out_rect.animate.surround(out_col.get_entries()[3]).set_stroke(width=1),
            FadeOut(flip_word),
            ReplacementTransform(components, in_col.get_entries()),
            ReplacementTransform(out_comps, out_col.get_entries()),
            Write(in_col.get_brackets()),
            Write(out_col.get_brackets()),
        )
        self.wait()


class OverOrUnderExplain(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("shruggie"),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(LaggedStart(
            stds[2].change("erm", look_at=morty.eyes),
            morty.change("hesitant", look_at=self.students),
        ))
        self.wait(2)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("hesitant", "well", "confused", look_at=3 * UR)
        )
        self.wait(5)


class IsThisUseful(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.change("hesitant", self.screen),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(4)
        self.play(
            stds[1].says("Is this...useful?", mode="confused"),
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[2].change("well", look_at=stds[1].eyes),
            morty.change("guilty", look_at=stds[1].eyes),
        )
        self.wait(4)


class SudokuBruteForce(InteractiveScene):
    def construct(self):
        # Test
        words = Text("Comically inefficient\\nbrute force approach:")
        steps = TexText("$9^{60}$ Steps", isolate=["60"])
        steps.scale(1.5)
        group = VGroup(words, steps)
        group.arrange(RIGHT, buff=MED_LARGE_BUFF)
        group.to_edge(UP)

        brace = Brace(steps[:3], DOWN)
        number = Tex("{:,d}".format(9**60).replace(",", "{,}"))
        number.scale(0.75)
        number.set_color(RED)
        number.next_to(brace, DOWN).shift_onto_screen()

        self.add(group)
        self.play(
            GrowFromCenter(brace),
            FadeIn(number, lag_ratio=0.1)
        )
        self.wait()

        # Show Grover
        grover_words = TexText("Using Grover's Algorithm:")
        grover_words.move_to(words, RIGHT)
        grover_steps = TexText(R"$\\displaystyle \\left\\lceil\\frac{\\pi}{4} 9^{30}\\right\\rceil$ Steps", isolate=["30"])
        grover_steps.move_to(steps, LEFT)
        new_brace = Brace(grover_steps[:8], DOWN)
        new_number = Tex("{:,d}".format(int(np.ceil(9**30 * PI / 4))).replace(",", "{,}"))
        new_number.scale(0.75)
        new_number.set_color(RED)
        new_number.next_to(new_brace, DOWN)

        self.play(
            FadeTransformPieces(words, grover_words),
            FadeTransform(steps, grover_steps),
            Transform(brace, new_brace),
            FadeTransformPieces(number, new_number),
        )
        self.wait()


class ShaInversionCounts(InteractiveScene):
    def construct(self):
        # Test
        classical = VGroup(
            get_classical_computer_symbol(height=1),
            TexText("$2^{256}$ Steps"),
        )
        quantum = VGroup(
            get_quantum_computer_symbol(height=1),
            TexText(R"$\\displaystyle \\left\\lceil \\frac{\\pi}{4} 2^{128} \\right\\rceil$ Steps"),
        )
        group = VGroup(classical, quantum)
        for elem in group:
            elem.arrange(RIGHT)
        group.arrange(DOWN, buff=2.0, aligned_edge=LEFT)
        group.to_corner(UR)

        self.play(FadeIn(classical, UP))
        self.wait()
        self.play(FadeTransformPieces(classical.copy(), quantum, lag_ratio=1e-4))
        self.wait()


class SkepticalPiCreature(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph().to_edge(DOWN).shift(3 * LEFT)
        randy.body.set_color(MAROON_E)
        morty = Mortimer().to_edge(DOWN).shift(3 * RIGHT)
        morty.make_eye_contact(randy)
        for pi in [randy, morty]:
            pi.change_mode("tease")
            pi.body.insert_n_curves(100)

        self.play(LaggedStart(
            randy.says("Quantum Computing will\\nchange everything!", mode="surprised"),
            morty.change("hesitant")
        ))
        self.play(Blink(morty))
        self.wait(2)
        self.play(Blink(randy))
        self.add(Point())
        self.play(
            morty.says("...will it?", mode="sassy"),
            randy.change("guilty"),
        )
        self.play(Blink(morty))
        self.wait()


class FactoringNumbers(InteractiveScene):
    def construct(self):
        # Show factoring number
        factor_values = [314159265359, 1618033988749]
        icon = get_quantum_computer_symbol(height=2).move_to(RIGHT)
        factors = VGroup(Integer(value) for value in factor_values)
        factors.arrange(RIGHT, buff=LARGE_BUFF)
        product_value = int(factor_values[0] * factor_values[1])
        product = Tex("{:,d}".format(product_value).replace(",", "{,}"))
        product.next_to(icon, LEFT)

        product.set_color(TEAL)
        factors.set_submobject_colors_by_gradient(BLUE, GREEN)

        times = Tex(R"\\times")
        times.move_to(factors)

        factor_group = VGroup(factors[0], times, factors[1])
        factor_group.arrange(DOWN, SMALL_BUFF)
        factor_group.next_to(icon, RIGHT, MED_LARGE_BUFF)

        self.add(icon)
        self.play(FadeIn(product, lag_ratio=0.1))
        self.wait()
        self.play(LaggedStart(
            FadeOutToPoint(product.copy(), icon.get_center(), lag_ratio=0.02, path_arc=45 * DEG),
            FadeInFromPoint(factor_group, icon.get_center(), lag_ratio=0.02, path_arc=45 * DEG),
            lag_ratio=0.3
        ))
        self.wait()


class FourBitAdder(InteractiveScene):
    def construct(self):
        # Test
        circuit = SVGMobject("Four_bit_adder_with_carry_lookahead")
        circuit.set_height(7.0)
        circuit.set_stroke(WHITE, 1)
        circuit.set_fill(BLACK, 0)
        circuit.sort(lambda p: np.dot(p, DR))

        self.play(Write(circuit, lag_ratio=1e-2, run_time=3))
        self.wait()


class PatronScroll(PatreonEndScreen):
    pass`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2025.blocks_and_grover.qc_supplements module within the 3b1b videos codebase.",
      30: "Sudoku extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      48: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      111: "Intro extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      114: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      122: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      123: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      124: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      125: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      139: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      140: "FadeOut transitions a mobject from opaque to transparent.",
      141: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      143: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      146: "Class HowDoYouKnowWhichAxis inherits from TeacherStudentsScene.",
      147: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      151: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      157: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      159: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      161: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      164: "SudokuChecker extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      165: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      170: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      172: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      173: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      186: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      215: "SudokuCheckingCode extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      216: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      260: "ArrowToQC extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      261: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      275: "CompiledSudokuVerifier extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      276: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      357: "StateVectorsAsABasis extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      358: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      436: "OperationsOnQC extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      437: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      468: "TwoByTwoGrid extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      471: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      648: "Class AskWhyThatsTrue inherits from TeacherStudentsScene.",
      649: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      688: "ListOfConfusions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      689: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      719: "SolveSHAWord extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      720: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      733: "Class ThatsOnMe inherits from TeacherStudentsScene.",
      734: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      745: "ShowSuperposition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      746: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      848: "NorthEastTraveler extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      849: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      971: "SimpleTwobitKet extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      972: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      978: "ShowLinearityExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      979: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1046: "ZGateExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1047: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1118: "PassSuperpositionIntoVerifier extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1119: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1294: "Class OverOrUnderExplain inherits from TeacherStudentsScene.",
      1295: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1317: "Class IsThisUseful inherits from TeacherStudentsScene.",
      1318: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1337: "SudokuBruteForce extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1338: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1380: "ShaInversionCounts extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1381: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1403: "SkepticalPiCreature extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1404: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1430: "FactoringNumbers extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1431: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1462: "FourBitAdder extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1463: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1475: "Class PatronScroll inherits from PatreonEndScreen.",
    }
  };

  files["_2025/grover/polarization.py"] = {
    description: "Polarization analogy for quantum mechanics: uses light polarization to build intuition for qubit states and measurement before introducing Grover's algorithm.",
    code: `from manim_imports_ext import *


class BeamSplitter(InteractiveScene):
    def construct(self):
        # Add laser device
        frame = self.frame
        light_source = self.camera.light_source
        light_source.move_to([-8, 5, 1])
        pointer = self.get_laser_pointer()
        pointer.to_edge(LEFT)
        beam = self.get_beam(pointer.get_right(), pointer.get_right() + 1000 * RIGHT)
        pointer.set_z_index(2)

        theta_tracker = ValueTracker(90 * DEG)
        pointer.curr_angle = 90 * DEG

        def set_theta(target_angle, run_time=2):
            curr_angle = theta_tracker.get_value()
            return AnimationGroup(
                theta_tracker.animate.set_value(target_angle),
                Rotate(pointer, curr_angle - target_angle, axis=RIGHT),
                run_time=run_time
            )

        frame.reorient(-121, 76, 0, (-3.29, -0.25, -0.34), 4.80)
        self.add(pointer)
        self.play(ShowCreation(beam, rate_func=lambda t: t**10))

        # Set up linear vector field
        wave = self.get_wave(theta_tracker, start_point=pointer.get_right(), max_x=200)

        self.add(theta_tracker)
        self.play(VFadeIn(wave))
        self.play(
            frame.animate.reorient(-57, 74, 0, (-3.29, -0.18, -0.18), 4.80),
            run_time=8
        )

        # Add sample vector
        amplitude = wave.amplitude
        sample_point = pointer.get_right() + 3 * RIGHT

        corner_plane, h_plane, v_plane, plane_in_situ = planes = VGroup(
            NumberPlane((-1, 1), (-1, 1))
            for _ in range(4)
        )
        for plane in planes:
            plane.axes.set_stroke(WHITE, 2, 0.5)
            plane.background_lines.set_stroke(opacity=0.5)
            plane.faded_lines.set_stroke(opacity=0.25)

        plane_in_situ.scale(amplitude)
        plane_in_situ.rotate(90 * DEG, RIGHT).rotate(90 * DEG, IN)
        plane_in_situ.move_to(sample_point)

        fixed_planes = planes[:3]
        fixed_planes.fix_in_frame()
        fixed_planes.scale(1.25)
        fixed_planes.arrange(RIGHT, buff=1.0)
        fixed_planes.to_corner(UL)

        corner_vector = Vector(RIGHT, thickness=3, fill_color=BLUE)
        corner_vector.plane = corner_plane
        corner_vector.wave = wave
        corner_vector.force_unit = False
        corner_vector.fix_in_frame()

        def update_corner_vect(vect, vertical=False, horizontal=False):
            coords = vect.wave.axes.p2c(sample_point)
            output = vect.wave.func(np.array([coords]))[0]
            x = np.dot(output, DOWN) / amplitude if not vertical else 0
            y = np.dot(output, OUT) / amplitude if not horizontal else 0
            if vect.force_unit:
                theta = theta_tracker.get_value()
                x /= math.cos(theta) or 1
                y /= math.sin(theta) or 1
            vect.put_start_and_end_on(vect.plane.c2p(0, 0), vect.plane.c2p(x, y))
            return vect

        corner_vector.add_updater(update_corner_vect)

        self.play(
            FadeIn(plane_in_situ),
            TransformFromCopy(plane_in_situ, corner_plane, run_time=2)
        )
        self.play(VFadeIn(corner_vector))
        self.play(frame.animate.reorient(-80, 79, 0, (-3.36, 0.1, -0.46), 4.80), run_time=12)

        # Add beam splitter
        split_point_dist = 6
        split_point = pointer.get_right() + split_point_dist * RIGHT
        splitter = Cube()
        splitter.set_color(WHITE)
        splitter.set_opacity(0.25)
        splitter.rotate(45 * DEG)
        splitter.set_height(0.5)
        splitter.move_to(split_point)

        top_axes, low_axies = split_axes = VGroup(ThreeDAxes(), ThreeDAxes())
        split_axes.move_to(split_point)
        for axes, sgn in zip(split_axes, [1, -1]):
            axes.rotate(sgn * 45 * DEG)

        short_wave = self.get_wave(theta_tracker, pointer.get_right(), max_x=split_point_dist, stroke_opacity=0.5)
        short_wave.time = wave.time
        corner_vector.wave = short_wave
        top_wave = self.get_wave(theta_tracker, split_point, refraction_angle=45 * DEG, project_horizontal=True, stroke_opacity=0.25)
        low_wave = self.get_wave(theta_tracker, split_point, refraction_angle=-45 * DEG, project_vertical=True, stroke_opacity=0.25)

        short_beam = self.get_beam(pointer.get_right(), split_point)
        top_beam = self.get_beam(split_point, split_point + 20 * UR / math.sqrt(2))
        low_beam = self.get_beam(split_point, split_point + 20 * DR / math.sqrt(2))
        top_beam.f_always.set_stroke(opacity=lambda: math.cos(theta_tracker.get_value()))
        low_beam.f_always.set_stroke(opacity=lambda: math.sin(theta_tracker.get_value()))
        top_beam.suspend_updating()
        low_beam.suspend_updating()

        self.play(
            FadeIn(splitter),
            FadeOut(wave),
            FadeIn(short_wave),
            FadeIn(top_wave),
            FadeIn(low_wave),
            FadeOut(beam),
            FadeIn(short_beam),
            FadeIn(top_beam),
            FadeIn(low_beam),
            plane_in_situ.animate.fade(0.5),
            frame.animate.reorient(-76, 62, 0, (-2.7, 0.08, -0.8), 6.22).set_anim_args(run_time=4)
        )
        self.wait(3)

        # Show rotation of the beam
        top_beam.resume_updating()
        low_beam.resume_updating()

        polarization_line = DashedLine(LEFT, RIGHT)
        polarization_line.set_stroke(WHITE, 1)
        polarization_line.fix_in_frame()
        polarization_line.add_updater(lambda m: m.set_angle(theta_tracker.get_value()))
        polarization_line.add_updater(lambda m: m.move_to(corner_plane))

        rot_arrows = VGroup(
            Arrow(RIGHT, LEFT, path_arc=PI),
            Arrow(LEFT, RIGHT, path_arc=PI),
        )
        rot_arrows.scale(0.5)
        rot_arrows.rotate(90 * DEG, RIGHT)
        rot_arrows.rotate(90 * DEG, OUT)
        rot_arrows.move_to(pointer.get_right() + 0.5 * RIGHT)

        self.add(polarization_line, corner_vector)
        self.play(Write(rot_arrows, lag_ratio=0, run_time=1))
        self.play(set_theta(0, run_time=3))
        self.play(FadeOut(rot_arrows))
        self.play(frame.animate.reorient(-90, 82, 0, (-1.73, 0.07, 1.0), 8.00), run_time=6)

        self.play(set_theta(60 * DEG, run_time=4))
        self.wait(4)

        # Express sample vector as a sum
        eq, plus = signs = VGroup(Tex(R"="), Tex(R"+"))
        signs.scale(1.5)
        signs.fix_in_frame()
        for sign, plane1, plane2 in zip(signs, fixed_planes, fixed_planes[1:]):
            sign.move_to(midpoint(plane1.get_right(), plane2.get_left()))

        coords = VGroup(DecimalNumber(0, unit=R"\\times"), DecimalNumber(0, unit=R"\\times"))  # Stopped using these, maybe later?
        coords.fix_in_frame()
        coords.scale(0.75)
        for coord, plane in zip(coords, fixed_planes[1:]):
            coord.next_to(plane, LEFT, SMALL_BUFF)
        coords[0].add_updater(lambda m: m.set_value(math.sin(theta_tracker.get_value())))
        coords[1].add_updater(lambda m: m.set_value(math.cos(theta_tracker.get_value())))

        h_vect, soft_h_vect, v_vect, soft_v_vect = corner_vector.replicate(4).clear_updaters()

        h_vect.plane = h_plane
        v_vect.plane = v_plane
        soft_h_vect.plane = corner_plane
        soft_v_vect.plane = corner_plane

        VGroup(soft_h_vect, soft_v_vect).set_fill(opacity=0.5)
        h_vect.add_updater(lambda m: update_corner_vect(m, horizontal=True))
        soft_h_vect.add_updater(lambda m: update_corner_vect(m, horizontal=True))
        v_vect.add_updater(lambda m: update_corner_vect(m, vertical=True))
        soft_v_vect.add_updater(lambda m: update_corner_vect(m, vertical=True))

        for plane in h_plane, v_plane:
            plane.save_state()
            plane.move_to(corner_plane)
            plane.set_stroke(opacity=0)

        self.play(
            VFadeIn(h_plane),
            VFadeIn(v_plane),
            VFadeIn(soft_h_vect),
            VFadeIn(soft_v_vect),
        )
        self.wait()
        self.play(
            Restore(h_plane, path_arc=30 * DEG),
            VFadeIn(h_vect),
            Write(eq),
        )
        self.play(
            Restore(v_plane, path_arc=30 * DEG),
            VFadeIn(v_vect),
            Write(plus),
        )
        self.wait(5)

        # Do some rotations
        curr_angle = 60 * DEG
        for target_angle in [90 * DEG, 0, 60 * DEG]:
            self.play(set_theta(target_angle))
            self.wait(3)

        # Add the angle, and sine/cosine terms
        arc = always_redraw(lambda: Arc(
            0, theta_tracker.get_value(), radius=0.5, arc_center=corner_plane.get_center(),
        ).fix_in_frame())
        theta_label = Tex(R"\\theta")
        theta_label.fix_in_frame()
        theta_label_height = theta_label.get_height()

        def update_theta_label(theta_label):
            point = arc.pfp(0.25)
            direction = rotate_vector(RIGHT, 0.5 * theta_tracker.get_value())
            height = min(arc.get_height(), theta_label_height)
            theta_label.set_height(height)
            theta_label.next_to(point, direction, SMALL_BUFF)

        theta_label.add_updater(update_theta_label)

        movers = VGroup(h_plane, plus, v_plane)
        for mob in movers:
            mob.generate_target()

        cos_term = Tex(R"\\cos(\\theta) \\, \\cdot ").fix_in_frame()
        sin_term = Tex(R"\\sin(\\theta) \\, \\cdot ").fix_in_frame()
        rhs = VGroup(cos_term, h_plane.target, plus.target, sin_term, v_plane.target)
        rhs.arrange(RIGHT, buff=0.25)
        rhs.next_to(eq, RIGHT, 0.25)

        self.play(
            VFadeIn(arc),
            Write(theta_label),
        )
        h_vect.force_unit = True
        v_vect.force_unit = True
        self.play(
            LaggedStartMap(MoveToTarget, movers),
            LaggedStartMap(FadeIn, VGroup(cos_term, sin_term)),
            FadeTransform(theta_label.copy().clear_updaters(), cos_term[R"\\theta"][0], time_span=(0.25, 1.25)),
            FadeTransform(theta_label.copy().clear_updaters(), sin_term[R"\\theta"][0], time_span=(0.5, 1.5)),
            run_time=1.5
        )
        self.wait(4)

        # Put each part in context
        sin_part = VGroup(sin_term, v_plane)
        cos_part = VGroup(cos_term, h_plane)

        self.play(
            sin_part.animate.scale(0.5).rotate(5 * DEG).rotate(45 * DEGREES, UP).shift(3 * DOWN),
            rate_func=there_and_back_with_pause,
            run_time=4
        )
        self.play(
            cos_part.animate.scale(0.5).rotate(-5 * DEG).rotate(45 * DEGREES, DOWN).shift(3 * DOWN + 2 * LEFT),
            rate_func=there_and_back_with_pause,
            run_time=4
        )
        self.wait(3)

        # More rotation!
        for target_angle in [80 * DEG, 10 * DEG, 60 * DEG]:
            self.play(set_theta(target_angle))
            self.wait()

        # Energy is proportional to amlpitude squared
        e_expr = Tex(R"E = k \\cdot (\\text{Amplitude})^2", font_size=36)
        e_expr.fix_in_frame()
        e_expr.next_to(corner_plane, DOWN, aligned_edge=LEFT)
        amp_brace = LineBrace(polarization_line.copy().scale(0.5, about_edge=UR), buff=SMALL_BUFF)
        amp_brace.fix_in_frame()
        one_label = amp_brace.get_tex("1").fix_in_frame()
        VGroup(amp_brace, one_label).set_fill(WHITE)

        self.play(
            Write(e_expr),
            frame.animate.reorient(-77, 80, 0, (-1.83, 2.76, 0.89), 8.00),
        )
        self.wait(5)
        self.play(
            GrowFromCenter(amp_brace),
            Write(one_label),
            VFadeOut(soft_h_vect),
            VFadeOut(soft_v_vect),
        )
        self.wait()

        # Numbers of the 60 degree example
        eq_60 = Tex(R"= 60^\\circ")

        # A lot of lingering

        # Add photo sensors

        # Turn down power

    def get_laser_pointer(self):
        box = Prism(0.75, 0.25, 0.25)
        box.set_color(GREY_D)
        box.set_shading(0.5, 0.5, 0)

        cone = ParametricSurface(
            lambda u, v: np.array([
                u,
                u * math.cos(-TAU * v),
                u * math.sin(TAU * v),
            ])
        )
        cone.stretch(5, 0)
        cone.set_width(0.25)
        cone.move_to(box.get_right())
        cone.set_color(GREY)

        return Group(box, cone)

    def get_beam(
        self,
        start,
        end,
        color=GREEN_SCREEN,
        stroke_width=2,
        opacity=1.0,
        anti_alias_width=25,
    ):
        beam = Line(start, end)
        beam.set_stroke(color, stroke_width, opacity)
        beam.set_anti_alias_width(anti_alias_width)
        return beam

    def get_wave(
        self,
        theta_tracker,
        start_point=ORIGIN,
        max_x=20,
        refraction_angle=0 * DEG,
        wave_number=4.0,
        freq=0.5,
        amplitude=0.25,  # Maybe replace with an amplitude tracker?
        color=BLUE,
        stroke_opacity=0.5,
        vector_density=0.1,
        max_vect_len=1.0,
        project_vertical=False,
        project_horizontal=False,
    ):
        axes = ThreeDAxes()
        axes.rotate(refraction_angle)
        axes.move_to(start_point)

        def field_func(points, time):
            theta = theta_tracker.get_value()
            magnitudes = amplitude * np.cos(wave_number * points[:, 0] - TAU * freq * time)
            result = np.zeros_like(points)
            if not project_vertical:
                result[:, 1] = -math.cos(theta) * magnitudes
            if not project_horizontal:
                result[:, 2] = math.sin(theta) * magnitudes
            return result

        density = 0.1
        sample_coords = np.arange(0, max_x, density)[:, np.newaxis] * RIGHT
        wave = TimeVaryingVectorField(
            field_func,
            axes,
            sample_coords=sample_coords,
            max_vect_len=max_vect_len,
            color=color,
            stroke_opacity=stroke_opacity
        )
        wave.amplitude = amplitude
        wave.axes = axes
        return wave`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "BeamSplitter extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      5: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      15: "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
      20: "AnimationGroup plays multiple animations together with individual timing control.",
      21: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      26: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      28: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      34: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      35: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      36: "Smoothly animates the camera to a new orientation over the animation duration.",
      45: "NumberPlane creates an infinite-looking 2D coordinate grid with major and minor gridlines.",
      58: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      67: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      70: "p2c (point to coords) converts scene positions back to mathematical coordinates.",
      72: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      73: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      78: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      83: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      84: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      85: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      87: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      88: "Smoothly animates the camera to a new orientation over the animation duration.",
      100: "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
      119: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      120: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      121: "FadeOut transitions a mobject from opaque to transparent.",
      122: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      123: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      124: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      125: "FadeOut transitions a mobject from opaque to transparent.",
      126: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      127: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      128: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      129: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      130: "Smoothly animates the camera to a new orientation over the animation duration.",
      132: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      140: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      141: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
    }
  };

  files["_2025/grover/qc_supplements.py"] = {
    description: "Quantum computing supplementary scenes: additional qubit visualizations, gate operations, and quantum circuit diagrams.",
    code: `from manim_imports_ext import *
from _2025.blocks_and_grover.state_vectors import *


def get_quantum_computer_symbol(height=2, color=GREY_B, symbol_tex=R"|Q\\rangle", symbol_color=TEAL):
    chip = SVGMobject("computer_chip")
    chip.set_height(height)
    chip.to_edge(RIGHT)
    chip.set_fill(GREY_C)
    chip.set_shading(0.7, 0, 0)
    symbol = Tex(symbol_tex)
    symbol.set_fill(symbol_color)
    symbol.set_stroke(symbol_color, 1)
    symbol.set_height(0.4 * chip.get_height())
    symbol.move_to(chip)

    result = VGroup(chip, symbol)
    return result


def get_classical_computer_symbol(height=2, color=GREY_B, symbol_tex=R"\\mathcal{C}", symbol_color=YELLOW):
    return get_quantum_computer_symbol(height, color, symbol_tex, symbol_color)


def get_blackbox_machine(height=2, color=GREY_D, label_tex="f(n)", label_height_ratio=0.25):
    square = Square(height)
    in_tri = ArrowTip().set_height(0.5 * height)
    out_tri = in_tri.copy().rotate(PI)
    in_tri.move_to(square.get_left())
    out_tri.move_to(square.get_right())
    machine = Union(square, in_tri, out_tri)
    machine.set_fill(color, 1)
    machine.set_stroke(WHITE, 2)

    label = Tex(label_tex)
    label.set_height(label_height_ratio * height)
    label.move_to(machine)
    machine.add(label)

    machine.output_group = VGroup()

    return machine


def get_bit_circuit(n_bits=4):
    circuit = SVGMobject("BitCircuit")
    circuit.set_stroke(WHITE, 2)
    circuit[-2:].set_fill(WHITE, 1)
    result = circuit.get_grid(1, n_bits, buff=0)
    return result


def get_magnifying_glass(height=3.0, color=GREY_D):
    glass = SVGMobject("magnifying_glass2")
    glass.set_fill(GREY_D)
    glass.set_shading(0.35, 0.15, 0.5)
    circle = VMobject().set_points(glass[0].get_subpaths()[1])
    circle.set_fill(GREEN_SCREEN, 1, border_width=3)
    circle.set_stroke(width=0)
    glass.add_to_back(circle)
    glass.set_height(height)
    return glass


def get_key_icon(height=0.5):
    key_icon = SVGMobject("key").rotate(135 * DEG)
    key_icon.set_fill(YELLOW)
    key_icon.set_height(height)
    return key_icon


class Superposition(Group):
    def __init__(
        self,
        pieces,
        offset_multiple=0.2,
        max_rot_vel=3,
        glow_color=TEAL,
        glow_stroke_range=(1, 22, 4),
        glow_stroke_opacity=0.05
    ):
        self.pieces = pieces
        self.center_points = Group(
            Point(piece.get_center())
            for piece in pieces
        )
        self.offset_multipler = ValueTracker(offset_multiple)

        for piece, point_mob in zip(pieces, self.center_points):
            piece.center_point = point_mob

            piece.offset_vect = rotate_vector(RIGHT, np.random.uniform(0, TAU))
            piece.offset_vect_rot_vel = np.random.uniform(-max_rot_vel, max_rot_vel)

        glow_strokes = np.arange(*glow_stroke_range)
        glows = pieces.replicate(len(glow_strokes))
        glows.set_fill(opacity=0)
        glows.set_joint_type('no_joint')
        for glow, sw in zip(glows, glow_strokes):
            glow.set_stroke(glow_color, width=float(sw), opacity=glow_stroke_opacity)

        self.glows = glows

        super().__init__(glows, pieces, self.center_points, self.offset_multipler)
        self.add_updater(lambda m, dt: m.update_piece_positions(dt))

    def update_piece_positions(self, dt):
        offset_multiple = self.offset_multipler.get_value()

        for piece in self.pieces:
            piece.offset_vect = rotate_vector(piece.offset_vect, dt * piece.offset_vect_rot_vel)
            piece.offset_radius = offset_multiple
            piece.move_to(piece.center_point.get_center() + piece.offset_radius * piece.offset_vect)

        for glow in self.glows:
            for sm1, sm2 in zip(glow.family_members_with_points(), self.pieces.family_members_with_points()):
                sm1.match_points(sm2)

    def set_offset_multiple(self, value):
        self.offset_multipler.set_value(value)

    def set_glow_opacity(self, opacity=0.1):
        self.glows.set_stroke(opacity=opacity)


###


class ReferenceSummary(InteractiveScene):
    def construct(self):
        # Laptop
        laptop = Laptop()
        self.frame.reorient(85, 79, 0, (-1.04, -1.42, 0.47), 6.97)
        self.add(laptop)

        # Randy
        randy = Randolph(mode="thinking", height=4)
        randy.body.insert_n_curves(100)
        randy.fix_in_frame()
        randy.next_to(ORIGIN, LEFT)
        randy.look_at(2 * RIGHT)

        self.add(randy)
        self.play(Blink(randy))
        self.wait()
        self.play(Blink(randy))
        self.play(randy.change("confused").fix_in_frame())
        self.wait(2)
        self.play(Blink(randy))
        self.wait(2)


class WriteQuantumComputingTitle(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Quantum Computing", font_size=120)
        title.move_to(UP)
        subtitle = Tex(R"\\frac{1}{\\sqrt{2}}\\big(|0\\rangle + |1\\rangle \\big)", font_size=90)
        subtitle.next_to(title, DOWN, LARGE_BUFF)
        title.set_color(TEAL)

        self.add(title, subtitle)
        self.play(LaggedStart(
            Flash(letter, color=TEAL, flash_radius=1.0, line_stroke_width=2)
            for letter in title
        ))


class MentionQuiz(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher

        to_confusion = VGroup(
            Vector(1.5 * RIGHT, thickness=6),
            Text("???", font_size=90).space_out_submobjects(1.5)
        )
        to_confusion.arrange(RIGHT)
        to_confusion.next_to(self.screen, RIGHT)
        to_confusion.shift(0.5 * UP + LEFT)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.change("angry", self.screen),
            self.change_students("thinking", "tease", "confused", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.change("raise_right_hand", to_confusion),
            self.change_students("erm", "guilty", "hesitant"),
            FadeInFromPoint(to_confusion, morty.get_corner(UL), lag_ratio=0.1),
        )
        self.wait()
        self.play(morty.says("Quiz time!"))
        self.wait(3)


class QuizMarks(InteractiveScene):
    def construct(self):
        marks = VGroup(
            Exmark().set_color(RED),
            Exmark().set_color(RED),
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        )
        marks.arrange(DOWN).scale(2)
        self.add(marks)


class SimpleScreenReference(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher

        self.play(
            morty.change("hesitant"),
            self.change_students("erm", "concentrating", "sassy", look_at=self.screen)
        )
        self.wait(5)


class StudentsCommentOnQuiz(TeacherStudentsScene):
    def construct(self):
        # Ask about quantum computer
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        qc = get_quantum_computer_symbol(height=1.5)
        qc.move_to(self.hold_up_spot, DOWN)
        qc_outline = qc.copy()
        qc_outline.set_fill(opacity=0).set_stroke(TEAL, 2)

        q_marks = VGroup(
            Text("?!?", font_size=72).next_to(std, UP)
            for std in self.students
        )

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(qc, UP),
            self.change_students("confused", "horrified", "angry"),
            LaggedStartMap(FadeIn, q_marks, shift=0.25 * UP),
        )
        self.play(
            VShowPassingFlash(qc_outline, time_width=2, run_time=3),
            Blink(morty),
        )
        self.wait(4)

        # Reference alternate version of blackbox function
        std = self.students[2]
        machine = get_blackbox_machine(height=1.0)
        machine.next_to(std.get_corner(UR), UP)
        arrow = Tex(R"\\updownarrow").set_height(1)
        arrow.next_to(machine, UP)

        self.play(LaggedStart(
            FadeOut(q_marks[:2], lag_ratio=0.1),
            self.change_students("pondering", "erm", "raise_right_hand"),
            morty.change("tease"),
            qc.animate.next_to(arrow, UP),
            FadeIn(machine),
            q_marks[2].animate.next_to(arrow, LEFT),
            Write(arrow),
        ))
        self.wait(5)

        # Thought bubble
        bubble = ThoughtBubble().pin_to(std)
        self.play(
            self.change_students("pondering", "pondering", "thinking", look_at=self.screen, lag_ratio=0.1),
            FadeIn(bubble, lag_ratio=0.2),
            LaggedStartMap(FadeOut, VGroup(machine, arrow, q_marks[2], qc), shift=LEFT, lag_ratio=0.1, run_time=1)
        )
        self.wait(3)


class Wrong(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        # morty.body.insert_n_curves(10000)
        self.play(morty.says("Wrong!", mode="surprised"))
        self.play(Blink(morty))
        self.wait()

        old_bubble = morty.bubble
        new_bubble = morty.get_bubble(Text("Also wrong!"), bubble_type=SpeechBubble)
        self.play(
            TransformMatchingStrings(
                old_bubble.content,
                new_bubble.content,
                key_map={"Wrong": "wrong"},
                run_time=1
            ),
            Transform(old_bubble[0], new_bubble[0]),
            morty.change("tease")
        )
        self.wait()


class WriteCheckMark(InteractiveScene):
    def construct(self):
        # Test
        check = Checkmark()
        check.set_color(GREEN)
        check.set_height(0.75)
        self.play(
            Write(check),
            Flash(check.get_left() + 1.0 * LEFT, flash_radius=0.8, line_length=0.2, color=GREEN)
        )
        self.wait()


class ExpressSkepticism(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph(height=3.5)
        randy.flip()
        randy.to_corner(DR)
        for mood in ["sassy", "pondering", "confused"]:
            self.play(randy.change(mood, ORIGIN))
            for _ in range(2):
                self.play(Blink(randy))
                self.wait(2)


class WriteGroversAlgorithm(InteractiveScene):
    def construct(self):
        text = TexText("Grover's Algorithm", font_size=60)
        self.play(Write(text))
        self.wait()


class TwoThirdsDivision(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 0.5))

        rects = Rectangle(3, 0.5).replicate(2)
        rects.arrange(RIGHT, buff=0)
        rects.set_fill(BLUE_E, 1)
        rects.set_submobject_colors_by_gradient(BLUE_E, BLUE_D)
        rects.set_stroke(WHITE, 1)
        rects.stretch_to_fit_width(12)
        rects.move_to(2 * DOWN)

        self.add(rects)
        self.wait()
        self.play(
            rects[0].animate.stretch_to_fit_width(8, about_edge=LEFT),
            rects[1].animate.stretch_to_fit_width(4, about_edge=RIGHT),
            run_time=2
        )
        self.wait()


class ReactToStrangeness(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.flip()
        randy.to_corner(UR)

        self.play(randy.change('confused', ORIGIN))
        self.play(Blink(randy))
        self.play(randy.change('erm', ORIGIN))
        for _ in range(2):
            self.play(Blink(randy))
            self.wait(2)


class DotsAndArrow(InteractiveScene):
    def construct(self):
        # Test
        dots = Tex(R"\\cdots", font_size=160)
        dots.space_out_submobjects(0.8)
        arrow = Vector(2.0 * RIGHT, thickness=8)
        arrow.next_to(dots, RIGHT, MED_LARGE_BUFF)
        self.play(LaggedStart(
            FadeIn(dots, lag_ratio=0.5),
            GrowArrow(arrow),
            lag_ratio=0.7
        ))
        self.wait()


class BigCross(InteractiveScene):
    def construct(self):
        # Test
        cross = Cross(Rectangle(4, 7))
        max_width = 15
        cross.set_stroke(RED, width=[0, max_width, max_width, max_width, 1])
        self.play(ShowCreation(cross))
        self.wait()


class VectSize16(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(Line(3 * DOWN, 3 * UP), RIGHT)
        tex = brace.get_tex("2^{4} = 16")
        tex.shift(SMALL_BUFF * UR)
        self.play(GrowFromCenter(brace), Write(tex))
        self.wait()


class MoreDimensionsNote(InteractiveScene):
    def construct(self):
        # Test
        words = TexText(R"(Except $2^k$ dimensions instead of 3)")
        words.set_color(GREY_B)
        self.play(FadeIn(words, lag_ratio=1))
        self.wait()


class QuestionsOnTheStateVector(TeacherStudentsScene):
    def construct(self):
        # React
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)  # Make it 10k?
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change("hesitant"),
            self.change_students("pleading", "confused", "erm", look_at=self.screen)
        )
        self.wait(3)

        # Questions
        self.play(
            stds[0].says("But...\\nwhat is it?", mode="maybe", look_at=morty.eyes)
        )
        self.wait()
        self.add(Point())
        self.play(
            stds[1].says("Why square things?", mode="raise_right_hand", look_at=morty.eyes)
        )
        self.wait(3)

        # Reference Complex
        expr = Tex(R"\\mathds{R}^n \\text{ vs. } \\mathds{C}^n")
        expr.move_to(self.hold_up_spot, DOWN)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(expr, UP),
            stds[0].debubble(mode="sassy"),
            stds[1].debubble(mode="erm"),
            stds[2].change("confused", look_at=expr)
        )
        self.wait(3)
        self.play(self.change_students("pondering", "pondering", "pondering", look_at=self.screen))
        self.wait(3)


class MagnifyingGlassOverComputer(InteractiveScene):
    def construct(self):
        # Test
        comp = get_quantum_computer_symbol(height=2.5)
        comp.center()
        glass = get_magnifying_glass()
        glass.next_to(comp, UL, buff=2).shift_onto_screen()

        self.add(comp, glass)
        self.play(
            glass.animate.shift(-glass[0].get_center()).set_anim_args(path_arc=-60 * DEG),
            rate_func=there_and_back_with_pause,
            run_time=6
        )
        self.wait()


class SimpleSampleValue(InteractiveScene):
    def construct(self):
        value = Tex(R"|0011\\rangle", font_size=60)
        value.set_fill(border_width=2)
        self.add(value)


class BitVsQubitMatrix(InteractiveScene):
    def construct(self):
        # Set up frame
        boxes = Rectangle(5, 3).get_grid(2, 2, buff=0)
        boxes.to_edge(DOWN, buff=MED_LARGE_BUFF)
        boxes.shift(RIGHT)
        boxes.set_stroke(WHITE, 1)

        top_titles = VGroup(
            Text("Bit", font_size=72),
            Text("Qubit", font_size=72),
        )
        for title, box in zip(top_titles, boxes):
            title.next_to(box, UP)
            title.align_to(top_titles[0], UP)

        side_titles = VGroup(
            Text("State", font_size=48),
            Text("What you\\nobserve", font_size=48),
        )
        for title, box in zip(side_titles, boxes[::2]):
            title.next_to(box, LEFT)
        side_titles[0].match_x(side_titles[1])

        for i in [0, 2, 3]:
            content = Text(R"0 or 1", font_size=72)
            content.move_to(boxes[i])
            boxes[i].add(content)

        self.add(boxes)
        self.add(top_titles)
        self.add(side_titles)


class ReferenceQubitComplexity(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)
        expr = Tex(R"\\mathds{R}^2 \\text{ vs. } \\mathds{C}^2", font_size=60)
        expr.move_to(self.hold_up_spot, DOWN)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("erm", "confused", "tease", look_at=expr),
            FadeIn(expr, UP)
        )
        self.wait(2)
        self.play(
            self.teacher.change("tease"),
            FadeOut(expr, shift=3 * RIGHT, path_arc=-90 * DEG),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen)
        )
        self.wait(4)


class ConfusionAtPresmises(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        self.play(
            morty.change('guilty'),
            stds[2].says("Okay...\\nbut why?", mode="sassy", bubble_direction=LEFT),
            stds[1].change("confused", self.screen),
            stds[0].change("maybe", self.screen),
        )
        self.wait(2)
        self.play(self.change_students("confused", "erm", "maybe", look_at=self.screen))
        self.wait(3)


class BitExamples(InteractiveScene):
    def construct(self):
        # All bits
        circuit = get_bit_circuit(1)
        circuit.to_edge(LEFT, buff=0.75).to_edge(DOWN, buff=0.25)

        switch = SVGMobject("light_switch")
        switch.set_height(1.5)
        switch.to_corner(UL)
        switch.match_x(circuit)
        switch.set_fill(GREY_B)
        switch.flip(RIGHT)

        coins = Circle(radius=0.5).replicate(2)
        for coin, color, letter in zip(coins, [RED_E, BLUE_E], "TH"):
            coin.set_fill(color, 1)
            coin.set_stroke(WHITE, 2)
            coin.add(Text(letter).move_to(coin))
        coins.set_y(0).match_x(circuit)
        coins[1].flip(RIGHT)
        coins.apply_depth_test()
        op_tracker = ValueTracker(0)

        def update_coins(coins):
            index = int(op_tracker.get_value() > 0.5)
            coins[index].set_opacity(1)
            coins[1 - index].set_opacity(0)

        coins.add_updater(update_coins)

        self.play(
            Write(circuit, lag_ratio=1e-2),
            Write(switch, lag_ratio=1e-2),
            FadeIn(coins),
        )
        switch.flip(RIGHT)
        self.play(
            Rotate(coins, PI, axis=RIGHT),
            op_tracker.animate.set_value(1),
        )
        self.wait()


class KetDefinition(InteractiveScene):
    def construct(self):
        # Test
        square = Rectangle(1, 0.75)
        dots = Tex(R"\\cdots")
        dots.space_out_submobjects(0.5)
        dots.replace(square, 0).scale(0.75)
        ket = VGroup(Ket(square), dots)
        ket.set_stroke(WHITE, 3)

        ket_name = TexText(R"\`\`ket''", font_size=60)
        ket_name.next_to(ket, UP, buff=MED_LARGE_BUFF)

        self.add(ket)
        self.play(Write(ket_name))
        self.wait()

        # Show unit vector
        arrow = Tex(R"\\longrightarrow", font_size=90)

        vector = Vector(2 * UP + RIGHT, thickness=6, fill_color=TEAL)
        vector.next_to(arrow, RIGHT, MED_LARGE_BUFF)
        brace = LineBrace(vector, DOWN, buff=SMALL_BUFF)
        brace_label = brace.get_tex("1")

        self.play(
            ket.animate.next_to(arrow, LEFT, MED_LARGE_BUFF),
            MaintainPositionRelativeTo(ket_name, ket),
            Write(arrow),
        )
        self.play(GrowArrow(vector))
        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_label)
        )
        self.wait()

        # Show examples of what goes inside it
        examples = VGroup(
            Tex(tex)
            for tex in [
                R"0",
                R"1",
                "+z",
                "-z",
                R"\\updownarrow",
                R"E",
                R"\\text{Dead}",
                R"\\text{Alive}",
                R"\\psi",
            ]
        )
        last = dots
        for example in examples:
            example.scale(2)
            example.set_max_width(dots.get_width())
            example.move_to(last)
            self.play(
                FadeOut(last, 0.5 * UP),
                FadeIn(example, 0.5 * UP),
                rate_func=linear,
                run_time=0.5
            )
            last = example


class ClassicalGates(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Logic Gates", font_size=60)
        title.to_edge(UP)
        self.add(title)

        # Basic gates
        gates = VGroup(
            SVGMobject("and_gate"),
            SVGMobject("or_gate"),
            SVGMobject("not_gate"),
        )
        gates.get_width()
        gates.set_width(2.0)
        gates.set_fill(GREY_B)
        gates.arrange(RIGHT, buff=2)
        gates.set_y(0)

        gate_names = VGroup(map(Text, ["AND", "OR", "NOT"]))
        for name, gate in zip(gate_names, gates):
            name.set_color(GREY_B)
            name.next_to(gate, DOWN)

        self.play(
            LaggedStartMap(Write, gates, lag_ratio=0.5),
            LaggedStartMap(Write, gate_names, lag_ratio=0.5),
        )

        # Bit examples
        inputs = VGroup(Text("01"), Text("01"), Text("0"))
        outputs = VGroup(Text("0"), Text("1"), Text("1"))
        for in_bits, out_bit, gate in zip(inputs, outputs, gates):
            in_bits.arrange(DOWN)
            in_bits.arrange_to_fit_height(1)
            in_bits.next_to(gate, LEFT, buff=SMALL_BUFF)
            out_bit.next_to(gate, RIGHT, buff=SMALL_BUFF)

        self.play(LaggedStartMap(FadeIn, inputs, shift=0.5 * RIGHT, lag_ratio=0.2, run_time=1.0)),
        self.play(LaggedStart(
            *(FadeOut(in_bit.copy(), RIGHT) for in_bit in inputs),
            *(FadeIn(out_bit, 0.5 * RIGHT) for out_bit in outputs),
            lag_ratio=0.2
        ))
        self.add(inputs, outputs)

        # Show full circuit
        gate_groups = VGroup(
            VGroup(gate, name, in_bits, out_bits)
            for gate, name, in_bits, out_bits in zip(gates, gate_names, inputs, outputs)
        )

        circuit = SVGMobject("Four_bit_adder_with_carry_lookahead")
        circuit.set_height(5)
        circuit.to_edge(DOWN, buff=0.25)
        circuit.set_fill(opacity=0)
        circuit.set_stroke(WHITE, 1)

        self.play(
            gate_groups.animate.scale(0.35).space_out_submobjects(0.9).next_to(title, DOWN, MED_LARGE_BUFF),
            Write(circuit, run_time=3)
        )
        self.wait()


class HLine(InteractiveScene):
    def construct(self):
        line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        line.set_stroke(GREY_B, 2)
        self.play(ShowCreation(line))
        self.wait()


class ShowQubitThroughTwoHadamardGates(InteractiveScene):
    def construct(self):
        # Add the line
        line = Line(4 * LEFT, 4 * RIGHT)
        h_box = Square(1)
        h_box.set_fill(BLACK, 1).set_stroke(WHITE)
        h_box.add(Text("H", font_size=60))
        h_boxes = h_box.replicate(2)
        h_boxes.arrange(RIGHT, buff=3)

        self.add(line, Point(), h_boxes)

        # Symbols
        symbols = VGroup(
            Tex(R"|0\\rangle", font_size=60),
            Tex(R"\\frac{1}{\\sqrt{2}}\\left(|0\\rangle + |1\\rangle\\right)"),
            Tex(R"|0\\rangle", font_size=60),
        )
        symbols[0].next_to(line, LEFT, MED_SMALL_BUFF)
        symbols[1].next_to(line, UP, LARGE_BUFF)
        symbols[2].next_to(line, RIGHT, MED_SMALL_BUFF)

        mid_sym_rect = SurroundingRectangle(symbols[1], buff=0.2)
        mid_sym_lines = VGroup(
            Line(line.pfp(0.5 + 0.01 * u), mid_sym_rect.get_corner(DOWN + u * RIGHT))
            for u in [-1, 1]
        )
        VGroup(mid_sym_lines, mid_sym_rect).set_stroke(BLUE, 2)

        # Show progression
        dot = GlowDot(color=BLUE, radius=0.5)
        dot.move_to(line.get_left())
        dot.set_opacity(0)

        cover = FullScreenFadeRectangle()
        cover.set_fill(BLACK, 1)
        cover.stretch(0.45, 0, about_edge=RIGHT)

        self.add(line, dot, h_boxes, cover)

        self.play(FadeIn(symbols[0], 0.5 * UR, run_time=1))
        self.play(dot.animate.set_opacity(1))
        self.wait()
        self.play(dot.animate.move_to(line.get_center()))
        self.play(
            ShowCreation(mid_sym_lines, lag_ratio=0),
            GrowFromPoint(mid_sym_rect, dot.get_center()),
            GrowFromPoint(symbols[1], dot.get_center()),
        )
        self.wait()
        self.play(FadeOut(cover))
        self.play(dot.animate.move_to(line.get_end()))
        self.play(FadeIn(symbols[2], 0.25 * RIGHT, rate_func=rush_from))
        self.wait()


class ReferencePreview(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(1000)

        self.play(
            morty.says("First, a preview"),
            self.change_students("happy", "tease", "happy", look_at=morty.eyes)
        )
        self.wait(4)
        self.play(
            morty.debubble(mode="raise_right_hand"),
            self.change_students("pondering", "thinking", "pondering", look_at=2 * UR)
        )
        self.look_at(3 * UR)
        self.wait(4)

        # Ask about sign
        stds = self.students
        self.play(
            stds[2].says("Why can you flip the\\nkey sign like that?", mode="raise_left_hand", look_at=self.teacher.eyes),
            stds[1].change("sassy", look_at=3 * UR),
            stds[0].change("maybe", look_at=3 * UR),
            morty.change('tease')
        )
        self.wait(5)


class GroverPreviewBox(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        rect = ScreenRectangle(height=4.0)
        rect.next_to(morty, UL)
        rect.set_fill(BLACK, 1).set_stroke(WHITE, 2)

        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        for pi in self.students:
            pi.change_mode("confused").look_at(rect)

        self.add(rect)
        self.play(
            morty.change("raise_right_hand", rect),
            self.change_students("pondering", "pondering", "pondering", look_at=rect)
        )
        self.wait(4)
        self.play(self.change_students("tease", "thinking", "pondering", look_at=rect))
        self.wait(5)


class SimpleMagnifyingGlass(InteractiveScene):
    def construct(self):
        # Test
        glass = get_magnifying_glass(height=3)
        glass.to_corner(UL)
        glass.target = glass.generate_target()
        glass.target.scale(1.75)
        glass.target.shift(1.5 * LEFT - glass.target[0].get_center())

        self.add(glass)
        self.wait()
        self.play(
            MoveToTarget(glass, path_arc=-45 * DEG),
            run_time=6,
            rate_func=there_and_back_with_pause,
        )
        self.wait()


class ShowAbstractionArrows(InteractiveScene):
    def construct(self):
        arrows = VGroup(
            Arrow(point + 4 * LEFT, ORIGIN + 0.35 * point, buff=1.0, thickness=5)
            for point in np.linspace(2.5 * DOWN, 2.5 * UP, 3)
        )
        arrows.set_fill(border_width=2)
        self.play(LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1))


class WriteGroversAlgorithm2(InteractiveScene):
    def construct(self):
        text = TexText("Grover's Algorithm", font_size=72)
        text.to_edge(UP)
        self.play(Write(text))
        self.wait()


class StareAtPicture(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        self.play(
            morty.change("speaking", look_at=self.students),
            self.change_students("tease", "well", "coin_flip_2")
        )
        self.wait(5)


class TryingToDescribeComputing(InteractiveScene):
    def construct(self):
        # Characters
        randy = Randolph()
        randy.move_to(2 * DOWN + 3 * LEFT)
        buddy = PiCreature(color=MAROON_E).flip()
        buddy.next_to(randy, RIGHT, buff=3)
        randy.make_eye_contact(buddy)

        for pi in [randy, buddy]:
            pi.body.insert_n_curves(100)

        self.add(randy, buddy)

        # Objects
        laptop = Laptop()
        laptop.scale(0.75)
        laptop.rotate(70 * DEG, LEFT)
        laptop.rotate(45 * DEG, UP)
        laptop.move_to(UP + 0.5 * LEFT)

        chip = get_classical_computer_symbol(height=1)
        chip.next_to(randy, UR, MED_LARGE_BUFF)

        # Show objects
        self.play(LaggedStart(
            randy.change("raise_right_hand", laptop),
            FadeIn(laptop, UP),
            buddy.change("erm", laptop),
            lag_ratio=0.5,
        ))
        self.play(Blink(buddy))
        self.wait()
        self.play(LaggedStart(
            randy.change("well", buddy.eyes),
            FadeIn(chip, UP),
            laptop.animate.shift(1.5 * UP),
            buddy.change("confused"),
            lag_ratio=0.25
        ))
        self.play(Blink(randy))

        # Show factoring numbers
        factors = Tex(R"91 = 7 \\times 13")
        factors.next_to(randy, UL, MED_LARGE_BUFF)
        factors.shift_onto_screen(buff=LARGE_BUFF)
        self.play(
            randy.change("raise_left_hand", factors),
            FadeInFromPoint(factors, chip.get_center(), lag_ratio=0.05),
        )
        self.play(Blink(buddy))
        self.wait(2)

        # Put numbers in chip
        c_label = chip[1]
        chip.remove(c_label)

        seven = factors["7"][0].copy()
        product = factors[R"7 \\times 13"][0].copy()

        self.play(
            randy.change('raise_right_hand', chip),
            chip.animate.scale(2, about_edge=DOWN),
            FadeOut(c_label, 0.5 * UP),
            FadeOut(laptop, UP),
            buddy.change("pondering", chip),
        )
        self.play(
            seven.animate.move_to(chip).scale(1.5),
        )
        self.play(Blink(buddy))
        self.wait()

        product.move_to(chip)
        self.play(
            ReplacementTransform(seven, product[0]),
            Write(product[1:]),
            buddy.change('hesitant', chip),
        )
        result = Tex(R"91")
        result.scale(1.5)
        result.move_to(chip)
        self.play(
            TransformFromCopy(product, result, lag_ratio=0.2),
            product.animate.set_opacity(0.25),
        )
        self.play(Blink(randy))
        self.wait()

        # Logic gates
        gates = SVGMobject("Four_bit_adder_with_carry_lookahead")
        gates.set_height(4)
        gates.to_edge(UP, MED_SMALL_BUFF)
        gates.to_edge(LEFT)
        gates.set_fill(opacity=0)
        gates.set_stroke(WHITE, 1)

        self.play(
            randy.change("dance_3", gates),
            Write(gates, run_time=2),
            FadeOut(factors, DOWN),
            buddy.change("awe", gates)
        )
        self.play(Blink(buddy))
        self.play(Blink(randy))
        self.wait()
        self.play(
            randy.change("tease"),
            FadeOut(gates, 3 * LEFT, rate_func=running_start, path_arc=30 * DEG)
        )
        self.wait()


class ProbForMillionDim(InteractiveScene):
    def construct(self):
        # Context
        context = VGroup(
            Tex(R"N = 2^{20}"),
            TexText(R"\\# Reps: 804"),
        )
        context.arrange(DOWN, aligned_edge=LEFT)
        context.to_corner(UL)

        # Prob
        theta = math.asin(2**(-10))
        prob = math.sin((2 * 804 + 1) * theta)**2

        chance_lhs = Tex(R"P(k) = ", t2c={"k": YELLOW})
        chance_rhs = DecimalNumber(100 * prob, num_decimal_places=7, unit="%")
        chance_rhs.next_to(chance_lhs, RIGHT, SMALL_BUFF).shift(0.05 * UR)
        chance = VGroup(chance_lhs, chance_rhs)
        chance.scale(1.25)
        chance.to_edge(RIGHT)

        brace = Brace(chance_rhs, DOWN)
        equation = brace.get_tex(R"\\sin((2 \\cdot 804 + 1) \\theta)^2")
        equation.set_color(GREY_C)
        equation.scale(0.7, about_edge=UP)

        self.play(Write(chance))
        self.play(
            GrowFromCenter(brace),
            FadeIn(equation),
        )
        self.wait()


class AskAreYouSure(InteractiveScene):
    def construct(self):
        # Test
        randy = Randolph()
        randy.to_edge(LEFT).shift(2 * DOWN)

        number = Integer(random.randint(0, int(1e6)))
        answer = KetGroup(number)
        answer.set_width(2)
        answer.move_to(4 * RIGHT)

        self.add(answer)
        self.add(randy)
        self.play(randy.change('pondering', answer))
        self.wait()
        self.play(randy.says("Are we...sure?", "hesitant"))
        self.play(Blink(randy))
        self.wait()

        # Show the machine
        machine = get_blackbox_machine()
        machine.to_edge(UP)
        output = Text("True", font_size=60)
        output.set_color(GREEN)
        output.next_to(machine, RIGHT, MED_LARGE_BUFF)

        self.play(
            randy.debubble("pondering", machine),
            FadeOut(answer[0]),
            number.animate.scale(1.5).next_to(machine, LEFT).set_anim_args(path_arc=-45 * DEG),
            FadeIn(machine),
        )

        self.add(number.copy().set_opacity(0.5))
        self.play(
            FadeOutToPoint(number, machine.get_center(), path_arc=-45 * DEG, lag_ratio=0.01)
        )
        self.play(
            FadeIn(output, 2 * RIGHT),
            randy.change("well", output),
        )
        self.play(Blink(randy))
        self.wait(2)


class WrapUpList(InteractiveScene):
    def construct(self):
        # List
        morty = Mortimer()
        morty.to_corner(DR)

        items = BulletedList(
            "The Lie",
            R"Why $\\sqrt{\\quad}$",
            "A suprising analogy",
            buff=0.75
        )
        items[1][-2:].shift(SMALL_BUFF * UP)
        items[1][1:].shift(0.5 * SMALL_BUFF * DOWN)
        items.scale(1.25)
        items.to_edge(UP, LARGE_BUFF)

        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand", items),
            FadeInFromPoint(items[0], morty.get_corner(UL), lag_ratio=0.01)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("pondering", items[1]),
            Write(items[1]),
        )
        self.wait()
        self.play(
            morty.change("tease", 5 * UR),
            FadeIn(items[2], DOWN),
        )
        self.play(Blink(morty))
        self.wait()

        for n, mode in enumerate(["guilty", "tease", "surprised"]):
            self.play(
                items.animate.fade_all_but(n, scale_factor=0.5),
                morty.change(mode, items[n]),
            )
            self.play(Blink(morty))
            self.wait()


class WriteShorName(InteractiveScene):
    def construct(self):
        name = Text("Peter Shor", font_size=72)
        name.to_corner(UR)
        self.play(Write(name, run_time=3))
        self.wait()


class OneWordSummary(InteractiveScene):
    def construct(self):
        # Replace "Parallelism" with "Pythagoras"
        prompt = Text("Source of the speed-up", font_size=72)
        prompt.move_to(2 * UP)

        words = VGroup(
            Text("Parallelism").set_color(TEAL),
            Text("Pythagoras").set_color(YELLOW),
        )
        words.scale(2)
        words.next_to(prompt, DOWN, LARGE_BUFF)
        strike = Line(words[0].get_left(), words[0].get_right())
        strike.set_stroke(RED, 10)

        self.add(prompt)
        self.add(words[0])
        self.wait()
        self.play(ShowCreation(strike))
        self.play(
            FadeIn(words[1], 0.5 * DOWN),
            words[0].animate.scale(0.75).shift(2 * DOWN).set_opacity(0.25),
            strike.animate.scale(0.75).shift(2 * DOWN),
        )
        self.wait()


class PythagoreanIntuition(InteractiveScene):
    def construct(self):
        # Set up axes
        x_range = y_range = z_range = (-2, 2)
        axes = ThreeDAxes(x_range, y_range, z_range)
        plane = NumberPlane(x_range, y_range)
        plane.fade(0.5)
        axes_group = VGroup(plane, axes)
        axes_group.scale(2)

        # Trace square
        frame = self.frame
        square = Square(2)
        square.move_to(ORIGIN, DL)
        square.set_stroke(WHITE, 2)

        side_lines = VGroup(
            Line(axes.get_origin(), axes.c2p(1, 0, 0)),
            Line(axes.c2p(1, 0, 0), axes.c2p(1, 1, 0)),
            Line(axes.c2p(1, 1, 0), axes.c2p(1, 1, 1)),
        )
        side_lines.set_stroke(RED, 4)
        ones = VGroup(
            Tex(R"1", font_size=36).next_to(line, vect, SMALL_BUFF)
            for line, vect in zip(side_lines, [DOWN, RIGHT, RIGHT])
        )
        ones[2].rotate(90 * DEG, RIGHT)

        dot = GlowDot(color=RED)
        dot.move_to(ORIGIN)

        frame.set_height(5).move_to(square)
        self.add(square, dot)
        for line, one in zip(side_lines[:2], ones):
            self.play(
                ShowCreation(line),
                FadeIn(one, 0.5 * line.get_vector()),
                dot.animate.move_to(line.get_end())
            )
        self.wait()
        self.play(
            MoveAlongPath(dot, square, rate_func=lambda t: 1 - 0.5 * smooth(t))
        )

        # Show diagonal
        diag = Line(square.get_corner(DL), square.get_corner(UR))
        diag.set_stroke(PINK, 3)

        sqrts = VGroup(
            Tex(R"\\sqrt{1^2 + 1^2}", font_size=24),
            Tex(R"\\sqrt{2}", font_size=36),
        )
        for sqrt in sqrts:
            sqrt.next_to(diag.pfp(0.5), UL, buff=0.05)

        self.play(
            ShowCreation(diag),
            dot.animate.move_to(square.get_corner(UR)),
            TransformFromCopy(ones[:2], sqrts[0]["1"], time_span=(1, 2)),
            *(
                Write(sqrts[0][tex], time_span=(1, 2))
                for tex in [R"\\sqrt", "+", "2"]
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformMatchingTex(*sqrts, key_map={"1^2 + 1^2": "2"}, run_time=1)
        )
        self.wait()

        # Bring it up to a cube
        axes_group.set_z_index(-1)
        cube = VCube(2)
        cube.move_to(ORIGIN, DL + IN)
        cube.set_stroke(WHITE, 2)
        cube.set_fill(opacity=0)

        self.add(cube, side_lines[:2])
        self.play(
            FadeIn(axes_group),
            ShowCreation(cube, lag_ratio=0.1, time_span=(0.5, 2.0)),
            frame.animate.reorient(-16, 68, 0, (0.45, 0.98, 1.05), 4.36),
            run_time=2
        )
        frame.add_ambient_rotation(DEG)
        line = side_lines[2]
        self.play(
            ShowCreation(line),
            FadeIn(ones[2], 0.5 * line.get_vector()),
            dot.animate.move_to(line.get_end())
        )
        self.wait(2)

        # Show three dimensional diagonal
        diag3 = Line(axes.c2p(0, 0, 0), axes.c2p(1, 1, 1))
        diag3.set_stroke(YELLOW, 3)

        new_sqrts = VGroup(
            Tex(R"\\sqrt{\\sqrt{2}^2 + 1^2}", font_size=24),
            Tex(R"\\sqrt{3}", font_size=36),
        )
        for sqrt in new_sqrts:
            sqrt.rotate(90 * DEG, RIGHT)
            sqrt.next_to(diag3.get_center(), LEFT + OUT, SMALL_BUFF)

        self.play(ShowCreation(diag3, run_time=2))
        self.play(
            TransformFromCopy(ones[2], new_sqrts[0]["1"][0], time_span=(1, 2)),
            TransformFromCopy(sqrts[1], new_sqrts[0][R"\\sqrt{2}"][0], time_span=(1, 2)),
            *(
                Write(new_sqrts[0][tex], time_span=(1, 2))
                for tex in [R"\\sqrt", "+", "2"]
            ),
            run_time=2,
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                *new_sqrts,
                key_map={R"\\sqrt{2}^2 + 1^2": "3"},
                match_animation=FadeTransform,
                run_time=1,
            )
        )
        self.wait(6)

        # Show observables
        symbols = VGroup(ones, sqrts[1], new_sqrts[1])
        wireframe = VGroup(cube, side_lines, diag, diag3)

        basis_vectors = VGroup(
            Vector(2 * v, thickness=4, fill_color=color)
            for v, color in zip(np.identity(3), [BLUE_E, BLUE_D, BLUE_C])
        )
        basis_vectors.set_z_index(1)
        for vector in basis_vectors:
            vector.always.set_perpendicular_to_camera(frame)

        obs_labels = VGroup(
            KetGroup(Text(f"Obs {n}", font_size=30), height_scale_factor=1.5, buff=0.05)
            for n in range(1, 4)
        )
        obs_labels[2].rotate(90 * DEG, RIGHT)
        for vector, label, nudge in zip(basis_vectors, obs_labels, [UP, RIGHT, RIGHT]):
            label.next_to(vector.get_end(), vector.get_vector() + nudge, buff=0.05)

        self.add(Point(), basis_vectors)
        self.play(
            LaggedStartMap(GrowArrow, basis_vectors, lag_ratio=0.25),
            FadeOut(symbols),
            FadeOut(dot),
            wireframe.animate.set_stroke(opacity=0.2),
            frame.animate.reorient(13, 67, 0, (-0.04, 0.76, 0.87), 4.84),
        )
        self.play(LaggedStartMap(FadeIn, obs_labels, lag_ratio=0.25))
        self.wait(4)

        new_cube = cube.copy()
        new_cube.deactivate_depth_test()
        new_cube.set_z_index(0)
        new_cube.set_stroke(WHITE, 3, 1)
        self.play(
            Write(new_cube, stroke_width=5, lag_ratio=0.1, run_time=3),
        )
        self.play(FadeOut(new_cube))
        self.wait(4)

        # Show many diagonal directions
        diag_vects = VGroup(
            Vector(2 * normalize(np.array(tup)))
            for tup in it.product(* 3 * [[-1, 0, 1]])
            if get_norm(tup) > 0
        )
        for vect in diag_vects:
            vect.set_perpendicular_to_camera(frame)
            color = random_bright_color(
                hue_range=(0.4, 0.5),
                saturation_range=(0.5, 0.7),
                luminance_range=(0.5, 0.6)
            )
            vect.set_color(color)

        self.play(
            FadeOut(obs_labels),
            LaggedStartMap(GrowArrow, diag_vects),
            frame.animate.reorient(-19, 61, 0, (-0.17, 0.22, -0.25), 6.96),
            run_time=3
        )
        self.wait(10)


class GrainOfSalt(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("coin_flip_2"),
            self.change_students("hesitant", "sassy", "tease", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            morty.says("Take this with\\na grain of salt", mode="hesitant"),
            self.change_students("pondering", "pondering", "pondering", look_at=morty.eyes),
        )
        self.wait(3)
        self.play(
            morty.debubble(mode="raise_right_hand", look_at=3 * UP),
            self.change_students("hesitant", "erm", "hesitant", look_at=2 * UP)
        )
        self.wait(4)

        # More pondering
        self.play(
            morty.change("tease"),
            self.change_students("pondering", "thinking", "thinking", look_at=2 * UP)
        )
        self.wait(5)


class PatronScroll(PatreonEndScreen):
    def construct(self):
        # Title
        title = Text("Special thanks to\\nthese supporters")
        title.to_corner(UR).shift(LEFT)
        title.set_color(BLUE)
        underline = Underline(title)
        rect = BackgroundRectangle(VGroup(title, underline))
        rect.set_fill(BLACK, 1)
        rect.scale(2, about_edge=DOWN)

        morty = Mortimer(height=1)
        morty.next_to(title, LEFT)
        morty.flip()

        for mob in rect, title, underline, morty:
            mob.fix_in_frame()
            self.add(mob)

        # Names
        names = self.get_names()
        name_mobs = VGroup(Text(name) for name in names)
        name_mobs.scale(0.5)
        for mob in name_mobs:
            mob.set_max_width(4)
        name_mobs.arrange(DOWN, aligned_edge=LEFT)
        name_mobs.next_to(underline.get_left(), DR).shift(0.5 * RIGHT)
        name_mobs.set_z_index(-1)

        self.add(name_mobs)

        # Scroll
        frame = self.frame
        dist = - name_mobs.get_y(DOWN) - 1
        total_time = 37
        velocity = dist / total_time
        frame.clear_updaters()
        frame.add_updater(lambda m, dt: m.shift(velocity * dt * DOWN))

        self.play(morty.change("gracious", name_mobs).fix_in_frame())
        for x in range(total_time):
            if random.random() < 0.15:
                self.play(Blink(morty))
            else:
                self.wait()


class ConstructQRCode2(InteractiveScene):
    def construct(self):
        # Test
        code = SVGMobject("channel_support_QR_code")
        code.set_fill(BLACK, 1)
        code.set_height(4)
        background = SurroundingRectangle(code, buff=0.25)
        background.set_fill(GREY_A, 1)
        background.set_stroke(width=0)
        background.set_z_index(-1)

        squares = code[:-6]
        corner_pieces = code[-6:]

        squares.shuffle()
        squares.sort(get_norm)
        squares.set_fill(interpolate_color(BLUE_E, BLACK, 0.5), 1)

        union = Union(*squares.copy().space_out_submobjects(0.99)).scale(1 / 0.99)
        union.set_stroke(WHITE, 2)
        union_pieces = VGroup(
            VMobject().set_points(path)
            for path in union.get_subpaths()
        )
        union_pieces.submobjects.sort(key=lambda m: -len(m.get_points()))
        union_pieces.note_changed_family()
        union_pieces.set_stroke(WHITE, 1)
        union_pieces.set_anti_alias_width(3)

        # New
        frame = self.frame
        frame.set_height(3)
        self.add(background, union_pieces, squares, corner_pieces)
        self.play(
            frame.animate.to_default_state(),
            ShowCreation(
                union_pieces,
                lag_ratio=0,
            ),
            # Write(squares, lag_ratio=0.1, time_span=(10, 20)),
            FadeIn(background, time_span=(20, 25)),
            Write(squares, stroke_color=BLUE, stroke_width=1, time_span=(12, 25)),
            Write(corner_pieces, time_span=(20, 25)),
            run_time=25,
        )
        self.play(
            FadeOut(union_pieces),
            squares.animate.set_fill(BLACK, 1),
        )
        squares.shuffle()
        self.play(LaggedStart(
            *(
                Rotate(square, 90 * DEG)
                for square in squares
            ),
            lag_ratio=0.02,
            run_time=10
        ))

        # Old
        return

        squares.save_state()
        squares.arrange_in_grid(buff=0)
        squares.move_to(background)
        squares.shuffle()
        for square in squares:
            dot = Dot()
            dot.set_fill(BLACK)
            dot.replace(square)
            square.set_points(dot.get_points())
        squares.set_stroke(WHITE, 1)
        squares.set_fill(opacity=0)

        background.save_state()
        background.set_fill(GREY_D)
        background.match_height(squares.saved_state)

        self.play(
            Restore(background),
            Restore(squares, lag_ratio=0.1),
            Write(corner_pieces, time_span=(18, 23), lag_ratio=0.25),
            run_time=35,
        )`,
    annotations: {
      72: "Superposition extends Group. Group holds heterogeneous mobjects (mix of VMobjects, Surfaces, Images) and transforms them together.",
      129: "ReferenceSummary extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      130: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      153: "WriteQuantumComputingTitle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      154: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      169: "Class MentionQuiz inherits from TeacherStudentsScene.",
      170: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      200: "QuizMarks extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      201: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      212: "Class SimpleScreenReference inherits from TeacherStudentsScene.",
      213: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      223: "Class StudentsCommentOnQuiz inherits from TeacherStudentsScene.",
      224: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      280: "Wrong extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      281: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      305: "WriteCheckMark extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      306: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      318: "ExpressSkepticism extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      319: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      331: "WriteGroversAlgorithm extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      332: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      338: "TwoThirdsDivision extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      339: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      361: "ReactToStrangeness extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      362: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      376: "DotsAndArrow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      377: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      391: "BigCross extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      392: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      401: "VectSize16 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      402: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      411: "MoreDimensionsNote extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      412: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      420: "Class QuestionsOnTheStateVector inherits from TeacherStudentsScene.",
      421: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      460: "MagnifyingGlassOverComputer extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      461: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      477: "SimpleSampleValue extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      478: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      484: "BitVsQubitMatrix extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      485: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      518: "Class ReferenceQubitComplexity inherits from TeacherStudentsScene.",
      519: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      538: "Class ConfusionAtPresmises inherits from TeacherStudentsScene.",
      539: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      554: "BitExamples extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      555: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      597: "KetDefinition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      598: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      663: "ClassicalGates extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      664: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      728: "HLine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      729: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      736: "ShowQubitThroughTwoHadamardGates extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      737: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      792: "Class ReferencePreview inherits from TeacherStudentsScene.",
      793: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      822: "Class GroverPreviewBox inherits from TeacherStudentsScene.",
      823: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      845: "SimpleMagnifyingGlass extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      846: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      864: "ShowAbstractionArrows extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      865: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      874: "WriteGroversAlgorithm2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      875: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      882: "Class StareAtPicture inherits from TeacherStudentsScene.",
      883: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      901: "TryingToDescribeComputing extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      902: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1014: "ProbForMillionDim extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1015: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1048: "AskAreYouSure extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1049: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1093: "WrapUpList extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1094: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1139: "WriteShorName extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1140: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1147: "OneWordSummary extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1148: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1174: "PythagoreanIntuition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1175: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1366: "Class GrainOfSalt inherits from TeacherStudentsScene.",
      1367: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1397: "Class PatronScroll inherits from PatreonEndScreen.",
      1398: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1444: "ConstructQRCode2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1445: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/grover/runtime.py"] = {
    description: "Runtime analysis of Grover's algorithm: visualizes the O(√N) speedup over classical search and why you can't do better.",
    code: `from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *
from _2025.blocks_and_grover.state_vectors import DisectAQuantumComputer


class Quiz(InteractiveScene):
    def construct(self):
        # Set up terms
        choices = VGroup(
            TexText(R"A) $\\mathcal{O}\\big(\\sqrt{N}\\big)$"),
            TexText(R"B) $\\mathcal{O}\\big(\\log(N)\\big)$"),
            TexText(R"C) $\\mathcal{O}\\big(\\log(\\log(N))\\big)$"),
            TexText(R"D) $\\mathcal{O}(1)$"),
        )
        choices.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        choices.to_edge(LEFT, buff=1.0)

        covers = VGroup(
            SurroundingRectangle(choice[2:], buff=SMALL_BUFF)
            for choice in choices
        )
        covers.set_fill(GREY_D, 1)
        covers.set_stroke(WHITE, 1)
        for cover, choice in zip(covers, choices):
            choice[2:].set_fill(opacity=0)
            cover.set_width(choices.get_width(), about_edge=LEFT, stretch=True)
            cover.align_to(covers, LEFT)
            cover.save_state()
            cover.stretch(0, 0, about_edge=LEFT)

        self.play(
            LaggedStartMap(Restore, covers, lag_ratio=0.25),
            LaggedStartMap(FadeIn, choices, lag_ratio=0.25),
        )
        self.wait()

        # Reference mostly wrong answers
        pis = Randolph().get_grid(6, 6, buff=2.0)
        pis.set_height(7)
        pis.to_edge(RIGHT)
        pis.sort(lambda p: np.dot(p, DR))

        symbol_height = 0.35
        symbols = [
            Exmark().set_color(RED).set_height(symbol_height),
            Checkmark().set_color(GREEN).set_height(symbol_height),
        ]
        all_symbols = VGroup()
        for pi in pis:
            pi.body.set_color(interpolate_color(BLUE_E, BLUE_C, random.random()))
            pi.change_mode("pondering")
            correct = random.random() < 0.2
            symbol = symbols[correct].copy()
            symbol.next_to(pi, UR, buff=0)
            all_symbols.add(symbol)
            pi.generate_target()
            pi.target.change_mode(["sad", "hooray"][correct])

        self.play(LaggedStartMap(FadeIn, pis, run_time=2))
        self.play(
            LaggedStartMap(MoveToTarget, pis, lag_ratio=0.01),
            LaggedStartMap(Write, all_symbols, lag_ratio=0.01),
        )
        pis.shuffle()
        self.play(LaggedStartMap(Blink, pis[::4]))
        self.wait()
        self.play(
            FadeOut(pis, lag_ratio=1e-3),
            FadeOut(all_symbols, lag_ratio=1e-3),
        )

        # Show question
        frame = self.frame

        question = VGroup(
            get_quantum_computer_symbol(),
            Clock(),
            Tex(R"?"),
        )
        for mob in question:
            mob.set_height(1.5)
        question.arrange(RIGHT, buff=0.5)
        question.set_width(4)
        question[2].scale(0.7)
        question.next_to(choices, UP, aligned_edge=LEFT)

        clock = question[1]
        cycle_animation(ClockPassesTime(clock, hours_passed=12, run_time=24))
        self.play(
            VFadeIn(question, suspend_mobject_updating=True, lag_ratio=0.01),
            VGroup(choices, covers).animate.shift(DOWN).set_anim_args(run_time=2),
            frame.animate.match_x(choices).set_anim_args(run_time=2),
        )
        self.add(choices, Point(), covers)
        choices.set_opacity(1)
        self.play(
            LaggedStart(
                (cover.animate.stretch(0, 0, about_edge=RIGHT).set_opacity(0)
                for cover in covers),
                lag_ratio=0.25,
            ),
        )
        self.wait(16)

        # Show distribution
        question.clear_updaters()
        dists = [
            np.array([18, 20, 8, 54], dtype=float),
            np.array([51, 55, 37, 65], dtype=float),  # Stanford
            np.array([17, 25, 5, 39], dtype=float),   # IMO, IIRC
        ]
        for dist in dists:
            dist[:] = dist / dist.sum()

        max_bar_width = 5.0
        prob_bar_group = VGroup()
        for dist in dists:
            prob_bars = VGroup()
            for choice, prob in zip(choices, dist):
                bar = Rectangle(width=prob * max_bar_width, height=0.35)
                bar.next_to(choice, LEFT)
                bar.set_fill(interpolate_color(BLUE_D, GREEN, prob * 1.5), 1)
                bar.set_stroke(WHITE, 1)
                prob_bars.add(bar)
            prob_bar_group.add(prob_bars)

        prob_bars = prob_bar_group[0].copy()

        prob_labels = VGroup()
        for bar in prob_bars:
            label = Integer(100, font_size=36, unit=R"\\%")
            label.bar = bar
            label.add_updater(lambda m: m.set_value(np.round(100 * m.bar.get_width() / max_bar_width)))
            label.add_updater(lambda m: m.next_to(m.bar, LEFT))
            prob_labels.add(label)

        self.play(
            LaggedStart(
                (GrowFromPoint(bar, bar.get_right())
                for bar in prob_bars),
                lag_ratio=0.2,
            ),
            VFadeIn(prob_labels)
        )
        self.wait()
        for index in [1, 2, 0]:
            self.play(Transform(prob_bars, prob_bar_group[index]))
            self.wait()

        # Go through each answer
        covers = VGroup(
            SurroundingRectangle(VGroup(bar, label, choice))
            for bar, label, choice in zip(prob_bars, prob_labels, choices)
        )
        covers.set_stroke(width=0)
        covers.set_fill(BLACK, 0.8)

        self.add(Point())
        self.play(FadeIn(covers[:3]))
        self.wait()
        self.play(
            FadeOut(covers[1]),
            FadeIn(covers[3])
        )
        self.wait()
        self.play(
            FadeOut(covers[0]),
            FadeIn(covers[1])
        )
        self.wait()

        # Add two additional answers


class ShowOptionGraphs(InteractiveScene):
    def construct(self):
        # Axes
        x_max = 15
        axes = Axes((-1, x_max), (-1, x_max))
        axes.set_height(7)
        self.add(axes)

        # Add graphs
        graphs = VGroup(
            axes.get_graph(func, x_range=(0.01, x_max))
            for func in [
                lambda n: n,
                lambda n: math.sqrt(n),
                lambda n: 0.8 * math.log(n + 1),
                lambda n: math.log(math.log(n + 1) + 1),
                lambda n: 1,
            ]
        )
        graphs.set_submobject_colors_by_gradient(YELLOW, ORANGE, RED, RED_E, BLUE)
        labels = VGroup(
            Tex(sym, font_size=30).match_color(graph).next_to(graph.get_end(), RIGHT, SMALL_BUFF)
            for graph, sym in zip(graphs, [
                R"\\mathcal{O}\\left(N\\right)",
                R"\\mathcal{O}\\left(\\sqrt{N}\\right)",
                R"\\mathcal{O}\\left(\\log(N)\\right)",
                R"\\mathcal{O}\\left(\\log(\\log(N))\\right)",
                R"\\mathcal{O}\\left(1\\right)",
            ])
        )
        labels[-1].shift(2 * SMALL_BUFF * DOWN)

        for graph, label in zip(graphs, labels):
            vect = label.get_center() - graph.get_end()
            self.play(
                ShowCreation(graph),
                VFadeIn(label),
                UpdateFromFunc(label, lambda m: m.move_to(graph.get_end() + vect)),
            )
        self.wait()


class NeedleInAHaystackProblem(InteractiveScene):
    def construct(self):
        # Set up terms
        shown_numbers = list(range(20))
        number_strs = list(map(str, shown_numbers))
        number_set = Tex("".join([
            R"\\{",
            *[str(n) + "," for n in shown_numbers],
            R"\\dots N - 1"
            R"\\}",
        ]), isolate=number_strs)
        number_mobs = VGroup(number_set[n_str][0] for n_str in number_strs)
        number_set.set_width(FRAME_WIDTH - 1)
        number_set.to_edge(UP)

        machine = get_blackbox_machine()
        machine.set_z_index(2)

        self.play(FadeIn(number_set, lag_ratio=0.01))
        self.wait()

        # Show mystery machine
        q_marks = Tex(R"???", font_size=90)
        q_marks.space_out_submobjects(1.2)
        q_marks.next_to(machine, UP)

        self.play(
            FadeIn(machine, scale=2),
            Write(q_marks)
        )
        self.play(LaggedStartMap(FadeOut, q_marks, shift=0.25 * DOWN, lag_ratio=0.1))
        self.wait()

        # Plug in key value
        key_number = 12
        key_input = number_mobs[key_number]
        key_icon = SVGMobject("key").rotate(135 * DEG)
        key_icon.set_fill(YELLOW)
        key_icon.match_width(key_input)
        key_icon.next_to(key_input, DOWN, SMALL_BUFF)

        self.play(
            FlashAround(key_input),
            key_input.animate.set_color(YELLOW),
            FadeIn(key_icon, 0.25 * DOWN)
        )
        self.wait()

        in_mob = key_input.copy().set_color(YELLOW)
        self.play(in_mob.animate.scale(1.5).next_to(machine, LEFT, MED_LARGE_BUFF))
        self.play(self.evaluation_animation(in_mob, machine, True))
        self.wait()

        # Plug in other values
        other_inputs = number_mobs.copy()
        other_inputs.remove(other_inputs[key_number])
        other_inputs.add(number_set["N - 1"][0].copy())
        other_inputs.generate_target()
        other_inputs.target.arrange_in_grid(n_cols=3, buff=MED_SMALL_BUFF)
        other_inputs.target.next_to(machine, LEFT, LARGE_BUFF)

        self.play(
            FadeOut(in_mob, DOWN),
            FadeOut(machine.output_group, DOWN),
            MoveToTarget(other_inputs, lag_ratio=0.01),
        )
        machine.output_group.clear()
        self.play(LaggedStart(
            (self.evaluation_animation(mob, machine)
            for mob in other_inputs),
            lag_ratio=0.2,
        ))
        self.wait()
        self.play(
            FadeOut(other_inputs, shift=0.25 * DOWN, lag_ratio=0.01),
            FadeOut(machine.output_group, 0.25 * DOWN),
        )
        machine.output_group.clear()
        self.wait()

        # Show innards
        innards = Code("""
            def f(n):
                return (n == 12)
        """, font_size=16)
        innards[8:].shift(0.5 * RIGHT)
        innards.move_to(machine).shift(0.25 * LEFT)

        self.play(
            machine.animate.set_fill(opacity=0),
            FadeIn(innards)
        )
        self.wait()
        self.play(
            FadeOut(innards),
            machine.animate.set_fill(opacity=1),
            FadeIn(q_marks, shift=0.25 * UP, lag_ratio=0.25)
        )
        self.play(FadeOut(q_marks))
        self.wait()

        # Guess and check
        last_group = VGroup()
        for n, in_mob in enumerate(number_mobs[:key_number + 1].copy()):
            self.play(
                FadeOut(last_group),
                in_mob.animate.scale(1.5).next_to(machine, LEFT, MED_LARGE_BUFF)
            )
            output = (n == key_number)
            self.play(self.evaluation_animation(in_mob, machine, output))
            last_group = VGroup(in_mob, machine.output_group[0])
            machine.output_group.clear()

        self.wait()
        self.play(FadeOut(last_group))

        # Put into a superposition
        pile = number_mobs.copy()
        for mob in pile:
            mob.scale(0.5)
        superposition = Superposition(pile)
        superposition.set_offset_multiple(0)
        superposition.set_glow_opacity(0)
        superposition.update()

        superposition.generate_target()
        for piece in superposition.pieces:
            piece.scale(2)

        for point in superposition.target[2]:
            point.next_to(machine, LEFT, buff=2.0)
            point.scale(0.5)
            point.shift(np.random.normal(0, 0.5, 3))

        superposition.target[2].arrange(DOWN, buff=0.25).next_to(machine, LEFT, buff=1.5)

        superposition.target.set_offset_multiple(0.1)
        superposition.target.set_glow_opacity(0.1)

        self.play(
            MoveToTarget(superposition, run_time=2),
        )

        # Pass superposition through the function
        answers = VGroup(
            Text("True").set_color(GREEN) if n == key_number else Text("False").set_color(RED)
            for n, piece in enumerate(superposition.pieces)
        )
        answers.match_height(superposition.pieces[0])
        answers.arrange_to_fit_height(superposition.get_height())
        answers.next_to(machine, RIGHT, buff=1.5)
        answers.shuffle()
        answer_superposition = Superposition(answers, glow_color=RED)
        answer_superposition.set_offset_multiple(0)
        answer_superposition.set_glow_opacity(0)
        answer_superposition.update()

        superposition.set_z_index(2)
        self.play(LaggedStart(
            LaggedStart(
                (FadeOutToPoint(glow.copy(), machine.get_left() + 0.5 * RIGHT)
                for glow in superposition.glows),
                lag_ratio=0.1,
            ),
            LaggedStart(
                (FadeInFromPoint(answer, machine.get_right() + 0.5 * LEFT)
                for answer in answer_superposition.pieces),
                lag_ratio=0.05,
            ),
            lag_ratio=0.5
        ))
        self.play(answer_superposition.animate.set_offset_multiple(0.025).set_glow_opacity(1e-2))
        self.wait(10)

    def evaluation_animation(self, input_mob, machine, output=False, run_time=1.0):
        if output:
            out_mob = Text("True").set_color(GREEN)
        else:
            out_mob = Text("False").set_color(RED)
        out_mob.scale(1.25)
        out_mob.next_to(machine, RIGHT, MED_LARGE_BUFF)

        moving_input = input_mob.copy()
        input_mob.set_opacity(0.25)

        machine.output_group.add(out_mob)
        in_point = interpolate(machine.get_left(), machine.get_center(), 0.5)

        return AnimationGroup(
            FadeOutToPoint(moving_input, in_point, time_span=(0, 0.75 * run_time)),
            FadeInFromPoint(out_mob, machine.get_left(), time_span=(0.25 * run_time, run_time)),
        )


class LargeGuessAndCheck(InteractiveScene):
    key_value = 42
    wait_time_per_mob = 0.1
    row_size = 10

    def construct(self):
        # Create grid of values and machine
        N = self.row_size
        grid = VGroup(Integer(n) for n in range(int(self.row_size**2)))
        grid.arrange_in_grid(buff=0.75, fill_rows_first=False)
        grid.set_height(FRAME_HEIGHT - 1)

        output = Text("False").set_color(RED)
        output.match_height(grid[0])

        machine = get_blackbox_machine(height=1.5 * grid[0].get_height())
        machine.next_to(output, LEFT, SMALL_BUFF)
        machine_group = VGroup(machine, output)
        extra_width = machine_group.get_width()
        grid.shift(0.5 * extra_width * RIGHT)

        self.add(grid)

        # Sweep through
        self.add(machine_group)
        for n, mob in enumerate(grid):
            if n % self.row_size == 0:
                grid[n:n + self.row_size].shift(extra_width * LEFT)
            machine_group.next_to(mob, RIGHT, buff=0.5 * mob.get_width())
            if n != self.key_value:
                self.play(grid[n].animate.set_opacity(0.5), run_time=self.wait_time_per_mob)
                continue

            new_output = Text("True").set_color(GREEN)
            new_output.replace(machine_group[1])
            machine_group.replace_submobject(1, new_output)
            break

        rect = SurroundingRectangle(grid[n])
        self.play(ShowCreation(rect), grid[n].animate.set_color(YELLOW))
        self.wait()


class GuessAndCheckEarlyGet(LargeGuessAndCheck):
    key_value = 12


class GuessAndCheckLateGet(LargeGuessAndCheck):
    key_value = 92


class GuessAndCheckMidGet(LargeGuessAndCheck):
    key_value = 53


class BigGuessAndCheck(LargeGuessAndCheck):
    key_value = 573
    wait_time_per_mob = 0.01
    row_size = 30


class WriteClassicalBigO(InteractiveScene):
    def construct(self):
        # Background
        self.add(FullScreenRectangle().set_fill(GREY_E))

        # Terms
        avg = TexText(R"Avg: $\\displaystyle \\frac{1}{2} N$")
        arrow = Tex(R"\\longrightarrow")
        big_o = Tex(R"\\mathcal{O}(N)")
        group = VGroup(avg, arrow, big_o)
        group.arrange(RIGHT, SMALL_BUFF)
        group.scale(1.25)
        group.to_edge(UP, buff=MED_SMALL_BUFF)

        avg.save_state()
        avg.set_x(0)

        self.play(Write(avg))
        self.wait()
        self.play(Restore(avg))
        self.play(
            Write(arrow),
            TransformFromCopy(avg[-1], big_o[2], path_arc=45 * DEG),
            Write(big_o[:2]),
            Write(big_o[3:]),
        )
        self.wait()


class ReferenceNeedleInAHaystack(InteractiveScene):
    key = 61

    def construct(self):
        # Show that grid of 100 values, with arrows to exes or checks
        N = 10
        grid = VGroup(Integer(n) for n in range(int(N * N)))
        grid.arrange_in_grid(fill_rows_first=False, h_buff=1.35, v_buff=0.75)
        grid.set_height(FRAME_HEIGHT - 1)

        self.add(grid)

        # Show marks
        key = self.key
        symbols = VGroup()
        for n in range(N * N):
            if n == key:
                symbol = Checkmark().set_color(GREEN)
            else:
                symbol = Exmark().set_color(RED)
            symbol.set_height(grid[0].get_height())
            symbol.next_to(grid[n], RIGHT, SMALL_BUFF)
            symbols.add(symbol)

        key_group = VGroup(grid[key], symbols[key])
        symbols.shuffle()

        self.play(LaggedStartMap(FadeIn, symbols, shift=0.25 * RIGHT, lag_ratio=0.05))
        self.wait()

        # Show key
        rect = SurroundingRectangle(key_group)
        rect.set_stroke(GREEN, 2)
        fader = FullScreenFadeRectangle(fill_opacity=0.5)
        self.add(fader, key_group, rect)
        self.play(
            FadeIn(fader),
            ShowCreation(rect),
        )


class ReferenceNeedleInAHaystack2(ReferenceNeedleInAHaystack):
    key = 31


class SuperpositionAsParallelization(InteractiveScene):
    def construct(self):
        # Set up
        classical, quantum = symbols = VGroup(
            get_classical_computer_symbol(),
            get_quantum_computer_symbol(),
        )
        for symbol, vect in zip(symbols, [LEFT, RIGHT]):
            symbol.set_height(1)
            symbol.move_to(vect * FRAME_WIDTH / 4)
            symbol.to_edge(UP, buff=MED_SMALL_BUFF)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 1)

        self.add(symbols)
        self.add(v_line)

        # Bit string
        boxes = Square().get_grid(1, 4, buff=0)
        boxes.set_height(0.5)
        boxes.match_x(classical)
        boxes.set_stroke(WHITE, 2)

        bit_mobs = VGroup(Integer(0).move_to(box) for box in boxes)

        def match_value(bit_mobs, value):
            bit_str = bin(int(value))[2:].zfill(4)
            for bit, mob in zip(bit_str, bit_mobs):
                mob.set_value(int(bit))
            return bit_mobs

        self.play(
            Write(boxes),
            Write(bit_mobs)
        )
        value_tracker = ValueTracker(0)
        self.play(
            value_tracker.animate.set_value(12),
            UpdateFromFunc(bit_mobs, lambda m: match_value(m, value_tracker.get_value())),
            run_time=3,
            rate_func=linear
        )
        self.wait()

        # Superposition
        bit_strings = VGroup()
        for bits in it.product(* 4 * [[0, 1]]):
            bit_string = VGroup(Integer(b) for b in bits)
            bit_string.arrange(RIGHT, buff=SMALL_BUFF)
            bit_strings.add(bit_string)
        bit_strings.arrange(DOWN)
        bit_strings.set_height(5.5)
        bit_strings.next_to(quantum, DOWN, MED_LARGE_BUFF)
        bit_strings.set_fill(opacity=0.75)

        superposition = Superposition(bit_strings)
        superposition.set_offset_multiple(0)
        superposition.set_glow_opacity(0)
        superposition.update()

        superposition_name = TexText(R"\`\`Superposition''")
        superposition_name.set_color(TEAL)
        superposition_name.next_to(superposition, RIGHT, aligned_edge=UP).shift(LEFT)

        self.play(
            LaggedStart(
                (TransformFromCopy(bit_mobs, bit_string)
                for bit_string in bit_strings),
                lag_ratio=0.05,
            )
        )
        self.play(
            superposition.animate.set_offset_multiple(0.1).set_glow_opacity(0.05).shift(1.5 * LEFT),
            Write(superposition_name, run_time=1)
        )
        self.wait(15)

        # Show parallelization lines
        mini_classical = VGroup(
            classical.copy().set_height(0.25).move_to(point).match_x(quantum)
            for point in superposition.center_points
        )
        lines = VGroup(
            VGroup(
                Line(mc.get_left() + 1.1 * LEFT, mc.get_left(), buff=0.1),
                Line(mc.get_right(), mc.get_right() + 1.1 * RIGHT, buff=0.1),
            )
            for mc in mini_classical
        )
        lines.set_stroke(GREY, 2)

        outputs = VGroup(
            Integer(int(n == 12), font_size=24).next_to(line, RIGHT)
            for n, line in enumerate(lines)
        )

        self.play(
            superposition.animate.set_offset_multiple(0.025),
            FadeOut(superposition_name),
            LaggedStart(
                (TransformFromCopy(classical, mc)
                for mc in mini_classical),
                lag_ratio=0.05,
            ),
            LaggedStartMap(ShowCreation, lines),
        )
        self.wait()
        self.play(LaggedStart(
            (TransformFromCopy(piece, output)
            for piece, output in zip(superposition.pieces, outputs)),
            lag_ratio=0.01,
            run_time=3
        ))
        self.wait(15)


class ListTwoMisconceptions(TeacherStudentsScene):
    def construct(self):
        # Add title and two misconceptions
        pass


class LogTable(InteractiveScene):
    def construct(self):
        # Set up table
        n_samples = 9
        line_width = 8
        line_buff = 0.75

        h_line = Line(LEFT, RIGHT).set_width(line_width)
        h_lines = h_line.get_grid(n_samples, 1, buff=line_buff)
        h_lines.set_stroke(WHITE, 1)
        h_lines.shift(0.25 * DOWN)

        v_line = Line(UP, DOWN).set_height(7)
        v_line.set_stroke(WHITE, 2)

        N_title, logN_title = titles = VGroup(
            Tex("N"),
            Tex(R"\\log_2(N)"),
        )
        titles.scale(1.25)
        for sign, title in zip([-1, 1], titles):
            title.set_x(sign * 2)
            title.to_edge(UP)

        self.add(h_lines, v_line)
        self.add(titles)

        # Fill with numbers
        N_values = VGroup()
        logN_values = VGroup()

        for n, line in enumerate(h_lines[1:]):
            N = 10**(n + 1)
            N_value = Integer(N)
            logN_value = DecimalNumber(np.log2(N))
            N_value.next_to(line, UP, SMALL_BUFF).match_x(N_title)
            logN_value.next_to(line, UP, SMALL_BUFF).match_x(logN_title)
            N_value.align_to(logN_value, UP)
            N_values.add(N_value)
            logN_values.add(logN_value)

        self.add(N_values[0], logN_values[0])
        for index in range(len(N_values) - 1):
            self.play(
                TransformMatchingShapes(N_values[index].copy(), N_values[index + 1]),
                FadeIn(logN_values[index + 1], shift=0.5 * DOWN),
                run_time=1
            )
        self.add(N_values)
        self.add(logN_values)

        # Show addition
        all_arrows = VGroup()
        for line in h_lines[1:-1]:
            arrow = Arrow(
                logN_values[0].get_right(),
                logN_values[1].get_right(),
                path_arc=-PI,
                buff=0
            )
            arrow.scale(0.8)
            arrow.next_to(line, RIGHT, SMALL_BUFF)
            plus_label = Tex(R"+\\log_2(10)", font_size=24)
            plus_label.set_color(BLUE)
            plus_label.next_to(arrow, RIGHT, SMALL_BUFF)

            all_arrows.add(VGroup(arrow, plus_label))

        self.play(LaggedStartMap(FadeIn, all_arrows, lag_ratio=0.25), run_time=3)


class SecondMisconception(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Misconception #2", font_size=90)
        title.to_edge(UP, buff=LARGE_BUFF)
        title.add(Underline(title))
        title.set_color(BLUE)

        words = Text(
            "Quantum Computers would make\\n" + \\
            "everything exponentially faster",
            font_size=60
        )
        words.next_to(title, DOWN, MED_LARGE_BUFF)
        red_cross = Cross(words["everything"])
        red_cross.set_stroke(RED, [0, 8, 8, 8, 0])
        new_words = Text("some very\\nspecial problems", alignment="LEFT")
        new_words.set_color(RED)
        new_words.next_to(red_cross, DOWN, aligned_edge=LEFT)

        self.add(title)
        self.play(Write(words, run_time=2))
        self.wait()
        self.play(LaggedStart(
            ShowCreation(red_cross),
            FadeIn(new_words, lag_ratio=0.1),
            FadeOut(title),
            lag_ratio=0.35
        ))
        self.wait()

        # Show factoring number
        factors = VGroup(Integer(314159), Integer(271829))
        factors.arrange(RIGHT, buff=LARGE_BUFF)
        product = Integer(factors[0].get_value() * factors[1].get_value())
        product.next_to(factors, UP, LARGE_BUFF)
        lines = VGroup(
            Line(product.get_bottom(), factor.get_top(), buff=0.2)
            for factor in factors
        )
        lines.set_submobject_colors_by_gradient(BLUE, GREEN)
        product.set_color(TEAL)
        factors.set_submobject_colors_by_gradient(BLUE, GREEN)
        times = Tex(R"\\times")
        times.move_to(factors)

        factor_group = VGroup(product, lines, factors, times)
        factor_group.next_to(words, DOWN, LARGE_BUFF, aligned_edge=RIGHT)

        self.play(FadeIn(product))
        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.25),
            LaggedStart(
                (TransformFromCopy(product, factor)
                for factor in factors),
                lag_ratio=0.25
            ),
            Write(times, time_span=(0.5, 1.5)),
        )
        self.wait()


class GroverTimeline(InteractiveScene):
    def construct(self):
        # Test
        timeline = NumberLine(
            (1990, 2025, 1),
            big_tick_spacing=5,
            width=FRAME_WIDTH - 1
        )
        timeline.set_y(-2)
        timeline.add_numbers(
            range(1990, 2030, 5),
            group_with_commas=False,
            font_size=24,
        )
        self.add(timeline)

        # BBBV
        bbbv_statement = VGroup(
            Text("Quantum Search", font_size=36),
            Tex(R"\\ge", font_size=42),
            Tex(R"\\mathcal{O}(\\sqrt{N})", font_size=36),
        )
        bbbv_statement.arrange(RIGHT, SMALL_BUFF)
        bbbv_statement.to_corner(UL)
        bbbv_statement.set_color(RED)

        bbbv_attribution = TexText("BBBV$^*$ Theorem (1994)", font_size=36)
        bbbv_attribution.next_to(bbbv_statement, DOWN, aligned_edge=LEFT)
        bbbv_attribution.set_color(RED_B)

        bbbv_attribution.to_corner(UL)
        bbbv_statement.next_to(bbbv_attribution, DOWN, MED_SMALL_BUFF)
        bbbv_statement.set_x(-3.5)

        footnote = Text("*Bennett, Bernstein, Brassard, Vazirani", font_size=24)
        footnote.set_color(GREY_C)
        footnote.to_corner(DL, buff=MED_SMALL_BUFF)

        bbbv_dots = VGroup(
            Dot(timeline.n2p(1994)),
            Dot().next_to(bbbv_attribution, LEFT, SMALL_BUFF),
        )
        bbbv_dots.set_color(RED_B)
        arc = -45 * DEG
        bbbv_line = Line(
            bbbv_dots[0].get_center(),
            bbbv_dots[1].get_center(),
            path_arc=arc,
        )
        bbbv_line.set_stroke(RED_B)

        self.play(
            GrowFromCenter(bbbv_dots[0]),
            TransformFromCopy(*bbbv_dots, path_arc=arc),
            ShowCreation(bbbv_line),
            FadeIn(bbbv_attribution, UP),
            FadeIn(footnote),
        )
        self.play(
            FadeIn(bbbv_statement, 0.5 * DOWN)
        )
        self.wait()

        # Lov Grover
        grover_name = TexText("Grover's Algorithm (1996)", font_size=36)
        grover_name.next_to(bbbv_statement, DOWN, buff=0.75)
        grover_name.set_color(BLUE)
        grover_name.shift(0.5 * LEFT)

        grover_statement = bbbv_statement.copy()
        eq = Tex(R"=").replace(grover_statement[1], dim_to_match=0)
        grover_statement[1].become(eq)
        grover_statement.set_color(BLUE_D)
        grover_statement.next_to(grover_name, DOWN)
        grover_statement.match_x(bbbv_statement)

        grover_dots = VGroup(
            Dot(timeline.n2p(1996)),
            Dot().next_to(grover_name, LEFT, SMALL_BUFF),
        )
        arc = -35 * DEG
        grover_line = Line(
            grover_dots[0].get_center(),
            grover_dots[1].get_center(),
            path_arc=arc,
        )
        VGroup(grover_dots, grover_line).set_color(BLUE_B)

        self.play(TransformFromCopy(bbbv_dots[0], grover_dots[0], path_arc=-PI))
        self.play(
            TransformFromCopy(*grover_dots, path_arc=arc),
            ShowCreation(grover_line),
            FadeIn(grover_name, UP),
        )
        self.wait()
        self.play(
            TransformFromCopy(bbbv_statement, grover_statement),
        )
        self.wait()

        # Show examples
        examples = VGroup()
        for n in [6, 12]:
            n_eq = VGroup(Tex(R"N = "), Integer(10**n))
            n_eq.arrange(RIGHT, SMALL_BUFF)
            n_eq.to_corner(UR, buff=MED_LARGE_BUFF)
            steps = VGroup(Tex(R"\\sim"), Integer(10**(n / 2)), Dot().set_fill(opacity=0), Text("Steps"))
            steps.arrange(RIGHT, buff=0.05)
            steps.next_to(n_eq, DOWN, LARGE_BUFF)

            arrow = Arrow(n_eq, steps, buff=0.15)

            examples.add(VGroup(n_eq, arrow, steps))

        sqrt_N = grover_statement[2][2:4]

        for n, example in enumerate(examples):
            if n == 0:
                self.play(
                    CountInFrom(example[0][1], 0),
                    VFadeIn(example[0]),
                )
            elif n == 1:
                self.play(
                    ReplacementTransform(examples[0][0], examples[1][0]),
                    FadeOut(examples[0][1:])
                )
            sqrt_rect = SurroundingRectangle(sqrt_N, buff=SMALL_BUFF)
            sqrt_rect.set_stroke(WHITE, 2)
            self.play(ShowCreation(sqrt_rect))
            self.play(
                GrowArrow(example[1]),
                sqrt_rect.animate.surround(example[2][1]).set_stroke(opacity=0),
                FadeTransform(sqrt_N.copy(), example[2][1]),
                FadeIn(example[2][0]),
                FadeIn(example[2][-1]),
            )
            self.add(example)
            self.wait()

        # Show the π / 4
        big_O = grover_statement[2]
        runtime = Tex(R"\\left\\lceil \\frac{\\pi}{4} \\right\\rceil \\sqrt{N}", font_size=36)
        runtime.move_to(big_O, LEFT)
        runtime.set_color(BLUE_B)

        rect = SurroundingRectangle(big_O)
        rect.set_stroke(WHITE, 2)

        self.play(ShowCreation(rect))
        self.wait()
        self.play(
            VGroup(rect, big_O).animate.shift(DOWN),
            FadeIn(runtime, scale=2)
        )
        self.wait()
        self.play(rect.animate.surround(runtime[1], buff=0.05))


class NPProblemExamples(InteractiveScene):
    def construct(self):
        # Examples
        example_images = Group(
            self.get_sudoku(),
            ImageMobject("US_color_graph"),
            Square().set_opacity(0),
        )
        for img in example_images:
            img.set_height(2)
        example_images.arrange_in_grid(2, 2, buff=1.5)
        example_images[2].match_x(example_images[:2])

        example_names = VGroup(
            Text("Sudoku"),
            Text("Graph Coloring"),
            Text("Reversing Cryptographic\\nHash Functions"),
        )
        examples = Group()
        for name, img in zip(example_names, example_images):
            name.scale(0.65)
            name.next_to(img, DOWN)
            examples.add(Group(img, name))
        examples.move_to(2 * LEFT)

        self.play(LaggedStartMap(FadeIn, examples, lag_ratio=0.5))

        # Name them
        big_rect = SurroundingRectangle(examples, buff=0.35)
        big_rect.set_stroke(BLUE, 3)
        big_rect.round_corners(radius=0.5)
        name = Text("NP Problems", font_size=60)
        name.next_to(big_rect, buff=MED_LARGE_BUFF)

        self.play(LaggedStart(
            ShowCreation(big_rect),
            Write(name),
            # self.frame.animate.set_x(2),
            lag_ratio=0.2
        ))
        self.wait()

    def get_sudoku(self):
        sudoku = SVGMobject("sudoku_example")
        small_width = sudoku[0].get_width()
        for part in sudoku.submobjects:
            if len(part.get_anchors()) == 5:
                part.set_fill(opacity=0)
                part.set_stroke(WHITE, 1, 0.5)
                if part.get_width() > 2 * small_width:
                    part.set_stroke(WHITE, 2, 1)
            else:
                part.set_fill(WHITE, 1)

        return sudoku


class ShowSha256(InteractiveScene):
    def construct(self):
        # Test
        import hashlib

        input_int = Integer(0, min_total_width=8, group_with_commas=False)
        output_text = Text("")
        lhs = VGroup(Text(R"SHA256("), input_int, Text(")"))
        lhs.arrange(RIGHT, buff=SMALL_BUFF),
        equation = VGroup(lhs, Tex("=").rotate(90 * DEG), output_text)
        equation.arrange(DOWN, buff=MED_SMALL_BUFF)

        def update_hash(text_mob):
            input_bytes = str(input_int.get_value()).encode()
            sha256_hash = hashlib.new('sha256')
            sha256_hash.update(input_bytes)
            hash_hex = sha256_hash.hexdigest()

            new_text = "\\n".join(
                "".join(row)
                for row in np.array(list(hash_hex)).reshape((4, 16))
            )

            new_text = Text(new_text, font="Consolas")
            new_text.move_to(text_mob, UP)
            text_mob.set_submobjects(new_text.submobjects)

        output_text.add_updater(update_hash)

        self.add(equation)
        self.play(
            ChangeDecimalToValue(input_int, 2400, rate_func=linear, run_time=24)
        )


class ContrastTwoAlgorithmsFrame(DisectAQuantumComputer):
    def construct(self):
        # Set up screens
        background = FullScreenFadeRectangle()
        background.set_fill(GREY_E, 1)
        screens = Rectangle(6, 5).get_grid(1, 2, buff=LARGE_BUFF)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.to_edge(DOWN, buff=LARGE_BUFF)

        self.add(background)
        self.add(screens)

        # Titles
        titles = VGroup(
            VGroup(get_classical_computer_symbol(), Tex(R"\\mathcal{O}(N)")),
            VGroup(get_quantum_computer_symbol(), Tex(R"\\mathcal{O}(\\sqrt{N})")),
        )
        for title, screen in zip(titles, screens):
            title[0].set_height(1.5)
            title[1].set_height(0.75)
            title.arrange(RIGHT, buff=0.5)
            title.next_to(screen, UP, aligned_edge=LEFT)

        self.add(titles)

        # Preview quantum search
        boxes = Square(0.1).get_grid(25, 4, fill_rows_first=False, v_buff=0.05, h_buff=1.1)
        boxes.set_height(screens[1].get_height() - MED_LARGE_BUFF)
        boxes.move_to(screens[1], UL).shift(MED_SMALL_BUFF * DR)

        values = VGroup(
            Integer(n, font_size=12).replace(box, dim_to_match=1)
            for n, box in enumerate(boxes)
        )

        dist = np.ones(100)
        width_ratio = 5
        bars = self.get_dist_bars(dist, boxes, width_ratio=width_ratio)

        q_dots = DotCloud().to_grid(4, 25).rotate(-90 * DEG)
        q_dots.replace(titles[1][0], dim_to_match=1)
        q_dots.stretch(0.5, 1)
        lines = VGroup(
            Line(point, box.get_center())
            for point, box in zip(q_dots.get_points(), boxes)
        )
        for line in lines:
            line.insert_n_curves(20)
            color = random_bright_color(hue_range=(0.3, 0.4))
            line.set_stroke(color, [0, 2, 2, 0], opacity=0.5)

        self.add(values)
        self.add(bars)

        for n in range(1, 10):
            dist[42] += n
            width_ratio *= 0.9
            lines.shuffle
            self.play(
                LaggedStartMap(VShowPassingFlash, lines, time_width=1.5, lag_ratio=2e-3),
                Transform(bars, self.get_dist_bars(dist, boxes, width_ratio=width_ratio), time_span=(0.5, 1))
            )


class QuantumCompilation(InteractiveScene):
    def construct(self):
        # Show circuitry
        machine = get_blackbox_machine()
        label = machine.submobjects[0]
        machine.remove(label)
        circuit = SVGMobject("BinaryFunctionCircuit")
        circuit.flip(RIGHT)
        circuit.set_stroke(width=0)
        circuit.set_fill(BLUE_B, 1)
        circuit.set_height(machine.get_height() * 0.8)
        circuit.move_to(machine).shift(0.25 * RIGHT)
        circuit.scale(2, about_point=ORIGIN)
        circuit.sort(lambda p: np.dot(p, DR))

        self.add(machine, label)

        self.wait()
        self.play(
            machine.animate.scale(2, about_point=ORIGIN).set_fill(GREY_E),
            FadeOut(label, scale=2),
        )
        self.play(Write(circuit, lag_ratio=0.05))
        self.wait()

        # Show binary input
        number = Integer(13, font_size=72, edge_to_fix=ORIGIN)
        bit_string = BitString(number.get_value())
        bit_string.next_to(machine, LEFT)
        number.next_to(machine, LEFT, MED_LARGE_BUFF)

        bit_string.set_z_index(-1)
        output = BitString(0, length=1).scale(1.5)
        output.set_z_index(-1)
        output.next_to(machine, RIGHT, MED_LARGE_BUFF)

        self.play(FadeIn(number, RIGHT))
        self.play(
            number.animate.next_to(bit_string, UP, MED_LARGE_BUFF),
            TransformFromCopy(number.replicate(5), bit_string, lag_ratio=0.01),
        )
        self.wait()

        self.play(
            FadeOut(bit_string.copy(), 2 * RIGHT, lag_ratio=0.05, path_arc=45 * DEG),
            FadeIn(output, RIGHT, time_span=(0.75, 1.5))
        )
        self.play(
            ChangeDecimalToValue(number, 5),
            UpdateFromFunc(bit_string, lambda m: m.set_value(number.get_value())),
            run_time=1
        )
        output.set_value(1)
        self.wait()

        # Show quantum case
        c_machine = VGroup(machine, circuit)
        c_machine.target = c_machine.generate_target()
        c_machine.target.scale(0.5).to_edge(UP)

        q_machine = Square().match_style(machine).set_height(0.5 * machine.get_height())
        lines = Line(ORIGIN, 0.75 * RIGHT).get_grid(4, 1, v_buff=0.25)
        lines.next_to(q_machine, LEFT, buff=0)
        q_machine.add(lines)
        q_machine.add(lines.copy().next_to(q_machine, RIGHT, buff=0))
        q_machine.to_edge(DOWN, buff=1.5)

        q_label = Text("Quantum\\nGates")  # If I were ambitious, I'd show the proper quantum circuit here
        q_label.set_color(TEAL)
        q_label.set_height(q_machine.get_height() * 0.4)
        q_label.move_to(q_machine)

        arrow = Arrow(c_machine.target, q_machine, thickness=5)

        self.play(
            MoveToTarget(c_machine),
            bit_string.animate.next_to(c_machine.target, LEFT),
            output.animate.next_to(c_machine.target, RIGHT, MED_LARGE_BUFF),
            FadeOut(number, UP),
        )
        self.play(GrowArrow(arrow))
        self.play(
            FadeTransform(c_machine[0].copy(), q_machine),
            TransformFromCopy(c_machine[1], q_label, lag_ratio=0.01, run_time=2),
        )
        self.wait()

        # Map to quantum input
        q_input = KetGroup(bit_string.copy())
        q_input.next_to(q_machine, LEFT)
        q_output = q_input.copy()
        neg = Tex(R"-").next_to(q_output, LEFT, SMALL_BUFF)
        q_output.add(neg)
        q_output.next_to(q_machine, RIGHT)

        input_rect = SurroundingRectangle(bit_string)
        input_rect.set_stroke(YELLOW, 2)
        output_rect = SurroundingRectangle(output)
        output_rect.set_stroke(GREEN, 2)
        check = Checkmark()
        check.match_height(output)
        check.set_color(GREEN)
        check.next_to(output, RIGHT)

        self.play(ShowCreation(input_rect))
        self.play(TransformFromCopy(input_rect, output_rect, path_arc=-45 * DEG))
        self.play(
            FadeOut(output_rect),
            Write(check[0], run_time=1)
        )
        self.wait()
        self.play(
            input_rect.animate.surround(q_input),
            TransformFromCopy(VGroup(VectorizedPoint(bit_string.get_center()), bit_string), q_input)
        )
        self.play(
            FadeOut(input_rect),
            FadeOut(q_input.copy(), 3 * RIGHT),
            FadeIn(q_output, 3 * RIGHT, time_span=(0.5, 1.5))
        )
        self.wait()

        # Show False inputs
        flipped_input = q_input.copy()
        flipped_output = q_output.copy()

        input_value_tracker = ValueTracker(number.get_value())
        ex = Exmark()
        ex.set_color(RED)
        ex.replace(check, 1)

        self.remove(q_output, check)
        self.add(ex)
        output.set_value(0)

        input_value_tracker.increment_value(1)
        self.play(
            input_value_tracker.animate.set_value(13).set_anim_args(rate_func=linear),
            UpdateFromFunc(bit_string, lambda m: m.set_value(int(input_value_tracker.get_value()))),
            UpdateFromFunc(q_input[1], lambda m: m.set_value(int(input_value_tracker.get_value()))),
            run_time=2
        )
        self.wait()

        q_output2 = q_input.copy()
        q_output2.next_to(q_machine, RIGHT, MED_LARGE_BUFF)
        self.play(TransformFromCopy(q_input, q_output2, path_arc=45 * DEG))
        self.wait()

        # Show combination
        combined_input = VGroup(q_input.copy(), Tex(R"+"), flipped_input)
        combined_input.arrange(DOWN, buff=SMALL_BUFF)
        combined_input.next_to(q_machine, LEFT)
        key_icon = get_key_icon()
        key_icon.match_height(q_input)
        key_icon.next_to(combined_input[2], LEFT, SMALL_BUFF)

        combined_output = VGroup(q_output2.copy(), Tex(R"+"), flipped_output)
        combined_output.arrange(DOWN, buff=SMALL_BUFF)
        combined_output.next_to(q_machine, RIGHT)

        self.play(
            ReplacementTransform(q_input, combined_input[0]),
            Write(combined_input[1:]),
            ReplacementTransform(q_output2, combined_output[0]),
            Write(combined_output[1:]),
            FadeIn(key_icon)
        )
        self.play(
            input_value_tracker.animate.set_value(5).set_anim_args(rate_func=linear),
            UpdateFromFunc(bit_string, lambda m: m.set_value(int(input_value_tracker.get_value()))),
        )
        output.set_value(1)
        self.remove(ex)
        self.add(check)
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2025.blocks_and_grover.qc_supplements module within the 3b1b videos codebase.",
      3: "Imports DisectAQuantumComputer from the _2025.blocks_and_grover.state_vectors module within the 3b1b videos codebase.",
      6: "Quiz extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      7: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      28: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      31: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      32: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      33: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      35: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      41: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      50: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      59: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      60: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      61: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      62: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      65: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      66: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      67: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      68: "FadeOut transitions a mobject from opaque to transparent.",
      69: "FadeOut transitions a mobject from opaque to transparent.",
      78: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      88: "Wraps an animation to loop indefinitely. The animation restarts seamlessly when it completes.",
      89: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      90: "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
      91: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      92: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      96: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      97: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      98: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      103: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      175: "ShowOptionGraphs extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      176: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      217: "NeedleInAHaystackProblem extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      218: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      411: "LargeGuessAndCheck extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      416: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      454: "Class GuessAndCheckEarlyGet inherits from LargeGuessAndCheck.",
      458: "Class GuessAndCheckLateGet inherits from LargeGuessAndCheck.",
      462: "Class GuessAndCheckMidGet inherits from LargeGuessAndCheck.",
      466: "Class BigGuessAndCheck inherits from LargeGuessAndCheck.",
      472: "WriteClassicalBigO extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      473: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      501: "ReferenceNeedleInAHaystack extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      504: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      542: "Class ReferenceNeedleInAHaystack2 inherits from ReferenceNeedleInAHaystack.",
      546: "SuperpositionAsParallelization extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      547: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      663: "Class ListTwoMisconceptions inherits from TeacherStudentsScene.",
      664: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      669: "LogTable extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      670: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      740: "SecondMisconception extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      741: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      802: "GroverTimeline extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      803: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      961: "NPProblemExamples extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      962: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1018: "ShowSha256 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1019: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1053: "Class ContrastTwoAlgorithmsFrame inherits from DisectAQuantumComputer.",
      1054: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1118: "QuantumCompilation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1119: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2025/grover/state_vectors.py"] = {
    description: "Quantum state vector visualizations for Grover's algorithm. Includes custom Ket notation, amplitude bar charts, and the geometric interpretation of Grover iterations as rotations.",
    code: `from manim_imports_ext import *
from _2025.blocks_and_grover.qc_supplements import *


class BitString(VGroup):
    def __init__(self, value, length=4, buff=SMALL_BUFF):
        self.length = length
        bit_mob = Integer(0)
        super().__init__(bit_mob.copy() for n in range(length))
        self.arrange(RIGHT, buff=buff)
        self.set_value(value)

    def set_value(self, value):
        bits = bin(value)[2:].zfill(self.length)
        for mob, bit in zip(self, bits):
            mob.set_value(int(bit))


class Ket(Tex):
    def __init__(self, mobject, height_scale_factor=1.25, buff=SMALL_BUFF):
        super().__init__(R"| \\rangle")
        self.set_height(height_scale_factor * mobject.get_height())
        self[0].next_to(mobject, LEFT, buff)
        self[1].next_to(mobject, RIGHT, buff)


class KetGroup(VGroup):
    def __init__(self, mobject, **kwargs):
        ket = Ket(mobject, **kwargs)
        super().__init__(ket, mobject)


class RandomSampling(Animation):
    def __init__(
        self,
        mobject: Mobject,
        samples: list,
        weights: list[float] | None = None,
        **kwargs
    ):
        self.samples = samples
        self.weights = weights
        super().__init__(mobject, **kwargs)

    def interpolate(self, alpha: float) -> None:
        if self.weights is None:
            target = random.choice(self.samples)
        else:
            target = random.choices(self.samples, self.weights)[0]
        self.mobject.set_submobjects(target.submobjects)


class ContrstClassicalAndQuantum(InteractiveScene):
    def construct(self):
        # Titles
        classical, quantum = symbols = VGroup(
            get_classical_computer_symbol(),
            get_quantum_computer_symbol(),
        )
        for symbol, vect in zip(symbols, [LEFT, RIGHT]):
            symbol.set_height(1)
            symbol.move_to(vect * FRAME_WIDTH / 4)
            symbol.to_edge(UP, buff=MED_SMALL_BUFF)

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 1)

        self.add(symbols)
        self.add(v_line)

        # Bits
        frame = self.frame
        value = ord('C')
        short_boxed_bits = self.get_boxed_bits(12, 4)
        boxed_bits = self.get_boxed_bits(value, 8)
        for group in short_boxed_bits, boxed_bits:
            group.match_x(classical)
        boxes, bits = boxed_bits

        self.add(short_boxed_bits)
        self.wait()
        self.play(
            FadeOut(v_line, shift=2 * RIGHT),
            FadeOut(quantum, shift=RIGHT),
            ReplacementTransform(short_boxed_bits, boxed_bits),
            frame.animate.match_x(classical),
        )
        self.wait()

        # Draw layers of abstraction
        layers = Rectangle(8.0, 1.5).replicate(3)
        layers.arrange(UP, buff=0)
        layers.set_stroke(width=0)
        layers.set_fill(opacity=0.5)
        layers.set_submobject_colors_by_gradient(BLUE_E, BLUE_D, BLUE_C)
        layers.set_z_index(-1)
        layers.move_to(boxes)

        layers_name = Text("Layers\\nof\\nAbstraction", alignment="LEFT")
        layers_name.next_to(layers, RIGHT)

        layer_names = VGroup(
            Text("Hardware"),
            Text("Bits"),
            Text("Data types"),
        )
        layer_names.set_fill(GREY_B)
        layer_names.scale(0.6)
        for name, layer in zip(layer_names, layers):
            name.next_to(layer, LEFT, MED_SMALL_BUFF)

        num_mob = Integer(value)
        num_mob.move_to(layers[2])
        character = Text(f"'{chr(value)}'")
        character.move_to(layers[2]).shift(0.75 * RIGHT)

        circuitry = get_bit_circuit(4)
        circuitry.set_height(layers[2].get_height() * 0.7)
        circuitry.move_to(layers[0])

        self.play(
            LaggedStartMap(FadeIn, layers, lag_ratio=0.25, run_time=1),
            FadeIn(layers_name, lag_ratio=1e-2),
            Write(layer_names[1]),
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(bit, num_mob)
                for bit in bits),
                lag_ratio=0.02,
            ),
            FadeIn(layer_names[2], UP),
        )
        self.wait()
        self.play(
            num_mob.animate.shift(0.75 * LEFT),
            FadeIn(character, 0.5 * RIGHT)
        )
        self.wait()
        self.play(
            Write(circuitry),
            FadeIn(layer_names[0], DOWN)
        )
        self.wait()

        # Extend layers to the quantum case
        new_layers_name = Text("Layers of Abstraction")
        new_layers_name.next_to(layers, DOWN)
        new_layers_name.match_x(quantum)

        layers.target = layers.generate_target()
        layers.target.set_width(FRAME_WIDTH, stretch=True)
        layers.target.set_x(0)

        layer_names.generate_target()
        for name, layer in zip(layer_names.target, layers.target):
            name.next_to(layer.get_left(), RIGHT, MED_SMALL_BUFF)

        self.play(
            LaggedStart(
                frame.animate.set_x(0),
                MoveToTarget(layer_names),
                FadeIn(quantum, RIGHT),
                ShowCreation(v_line),
                Transform(layers_name, new_layers_name),
                Group(num_mob, character, boxed_bits).animate.shift(RIGHT),
                circuitry.animate.scale(0.8).shift(RIGHT)
            ),
            MoveToTarget(layers, lag_ratio=0.01),
        )
        self.wait()

        # Show quantum material
        qubit_string = BitString(0, length=8)
        qubit_string.set_value(ord("Q"))
        qubit_ket = Ket(qubit_string)
        qubits = VGroup(qubit_ket, qubit_string)

        qunit_num = Integer(ord("Q"))
        qunit_ket = Ket(qunit_num)
        qunit = VGroup(qunit_ket, qunit_num)

        ion = Group(
            GlowDot(color=RED, radius=0.5),
            Dot(radius=0.1).set_fill(RED, 0.5),
            Tex(R"+", font_size=14).set_fill(border_width=1)
        )
        trapped_ions = Group(ion.copy().shift(x * RIGHT) for x in np.linspace(0, 4, 8))

        for mob, layer in zip([trapped_ions, qubits, qunit], layers):
            mob.move_to(layer).match_x(quantum)

        for ion, bit in zip(trapped_ions, qubit_string):
            if bit.get_value() == 1:
                ion[0].set_opacity(0)

        self.play(LaggedStartMap(FadeIn, trapped_ions))
        self.wait()
        self.play(FadeIn(qubits, UP))
        self.wait()
        self.play(
            TransformFromCopy(qubits, qunit)
        )
        self.wait()
        value_tracker = ValueTracker(ord("Q"))
        for value in [ord('C'), ord('Q')]:
            self.play(
                value_tracker.animate.set_value(value),
                UpdateFromFunc(qunit_num, lambda m: m.set_value(int(value_tracker.get_value()))),
                UpdateFromFunc(qubit_string, lambda m: m.set_value(int(value_tracker.get_value()))),
                rate_func=linear,
                run_time=1.0
            )
            self.wait(0.25)

        # Show some measurements
        lasers = VGroup()
        for ion in trapped_ions:
            point = ion.get_center()
            laser = Line(point + 0.5 * DL, point)
            laser.insert_n_curves(20)
            laser.set_stroke(RED, [1, 3, 3, 3, 1])
            lasers.add(laser)

        for value in [*np.random.randint(0, 2**8, 4), ord("Q")]:
            qubit_string.generate_target()
            qunit_num.generate_target()
            trapped_ions.generate_target()
            qunit_num.target.set_value(value)
            qubit_string.target.set_value(value)
            for ion, bit in zip(trapped_ions.target, qubit_string.target):
                ion[0].set_opacity(1.0 - bit.get_value())
            self.play(
                LaggedStartMap(VShowPassingFlash, lasers, lag_ratio=0.1, time_width=2.0, run_time=2),
                MoveToTarget(trapped_ions, lag_ratio=0.1, time_span=(0.5, 2.0)),
                MoveToTarget(qubit_string, lag_ratio=0.1, time_span=(0.5, 2.0)),
                MoveToTarget(qunit_num, time_span=(1.0, 1.25)),
                Transform(qunit_ket, Ket(qunit_num.target), time_span=(1.0, 1.5)),
            )

        # Describe a ket
        morty = Mortimer(height=5)
        morty.move_to(np.array([13., -6., 0.]))
        big_ket = Ket(Square(1))
        big_ket.set_fill(border_width=3)
        big_ket.next_to(morty.get_corner(UL), UP, MED_LARGE_BUFF)
        big_ket_name = TexText("\`\`ket''", font_size=96)
        big_ket_name.next_to(big_ket, UP, MED_LARGE_BUFF)

        self.play(
            frame.animate.reorient(0, 0, 0, (4.66, -2.55, 0.0), 13.19),
            morty.change("raise_right_hand", big_ket),
            VFadeIn(morty),
            *(
                TransformFromCopy(src, big_ket)
                for src in [qubit_ket, qunit_ket]
            ),
        )
        self.play(
            Write(big_ket_name, time_span=(0.75, 2.0)),
            FlashAround(big_ket, time_width=1.5, run_time=2)
        )
        self.wait()

        # Refocus
        self.remove(big_ket)
        self.play(LaggedStart(
            FadeOut(VGroup(morty, big_ket, big_ket_name)),
            TransformFromCopy(big_ket, qubit_ket),
            TransformFromCopy(big_ket, qunit_ket),
            frame.animate.to_default_state(),
        ))

        # Expand mid layer
        mid_layer = layers[1]
        mid_layer.set_z_index(-2)
        mid_layer.generate_target()
        mid_layer.target.set_height(7, stretch=True)
        mid_layer.target.move_to(layers, UP)
        mid_layer.target.set_fill(opacity=0.25)

        target_y = -1.0

        self.play(
            FadeOut(
                VGroup(layers[2], num_mob, character, qunit, layer_names[2]),
                UP,
            ),
            FadeOut(
                Group(layers[0], circuitry, trapped_ions, layer_names[0]),
                DOWN,
            ),
            FadeOut(layers_name, DOWN),
            qubits.animate.set_y(target_y),
            boxed_bits.animate.match_x(classical).set_y(target_y),
            layer_names[1].animate.set_y(target_y),
            MoveToTarget(mid_layer, time_span=(0.5, 2.0)),
            run_time=2
        )
        self.play(
            FadeOut(layers[1]),
            FadeOut(layer_names[1]),
            run_time=3
        )

        # Show state vs. what you read, classical
        contrast = VGroup(
            Text("State"),
            Tex(R"=", font_size=72),
            Text("What you see"),
        )
        contrast.arrange(RIGHT)
        contrast[2].align_to(contrast[0], UP)
        contrast.match_x(classical)
        contrast.set_y(0.5)
        contrast.shift(0.5 * RIGHT)

        boxed_bits_copy = boxed_bits.copy()
        boxed_bits_copy.scale(0.7)
        boxed_bits_copy.stretch(0.8, 0)
        for bit in boxed_bits_copy[1]:
            bit.stretch(1 / 0.8, 0)
        boxed_bits_copy.next_to(contrast[2], DOWN, buff=0.75)
        boxed_bits_copy[0].set_stroke(WHITE, 1)

        boxed_bits.target = boxed_bits_copy.copy()
        boxed_bits.target.match_x(contrast[0])

        self.play(
            FadeIn(contrast[::2]),
            MoveToTarget(boxed_bits),
        )
        self.play(
            Write(contrast[1]),
            TransformFromCopy(boxed_bits, boxed_bits_copy, path_arc=30 * DEG),
        )
        self.wait()
        self.play(*(
            LaggedStart(
                (bit.animate.set_stroke(YELLOW, 3).set_anim_args(rate_func=there_and_back)
                for bit in group[1]),
                lag_ratio=0.25,
                run_time=4
            )
            for group in [boxed_bits, boxed_bits_copy]
        ))
        self.wait()

        # Show state vs. what you read, quantum
        q_contrast = contrast.copy()
        q_contrast.match_x(quantum)
        ne = Tex(R"\\ne", font_size=72)
        ne.move_to(q_contrast[1])
        ne.set_color(RED)
        q_contrast[1].become(ne)

        state_vector = Vector(UR, thickness=4)
        state_vector.set_color(TEAL)
        state_vector.next_to(q_contrast[0], DOWN, MED_LARGE_BUFF)
        state_vector.set_opacity(0)  # Going to overlap something else instead

        state_vector_outline = state_vector.copy().set_fill(opacity=0)
        state_vector_outline.set_stroke(BLUE_A, 3)
        state_vector_outline.insert_n_curves(100)

        qubits.generate_target()
        qubits.target[1].space_out_submobjects(0.8)
        qubits.target[0].become(Ket(qubits.target[1]))
        qubits.target.match_x(q_contrast[2]).match_y(state_vector)

        moving_rect = SurroundingRectangle(state_vector)
        moving_rect.set_stroke(YELLOW, 3, 0)

        self.play(LaggedStart(
            TransformFromCopy(contrast, q_contrast, path_arc=-45 * DEG),
            MoveToTarget(qubits),
            GrowArrow(state_vector),
        ))
        self.wait()
        self.play(moving_rect.animate.surround(qubits).set_stroke(YELLOW, 3, 1))
        self.play(FadeOut(moving_rect))
        self.play(
            value_tracker.animate.set_value(0).set_anim_args(rate_func=there_and_back, run_time=4),
            UpdateFromFunc(qubit_string, lambda m: m.set_value(int(value_tracker.get_value()))),
        )
        self.wait()

        # Show randomness
        qubit_samples = list()
        for n in range(2**8):
            sample = qubits.copy()
            sample[1].set_value(n)
            sample.shift(np.random.uniform(-0.05, 0.05, 3))
            sample.set_stroke(TEAL, 1)
            qubit_samples.append(sample)

        labels = VGroup(Text("Random"), Text("Deterministic"))
        for label, mob, color in zip(labels, [qubits, boxed_bits_copy], [TEAL, YELLOW]):
            label.scale(0.75)
            label.next_to(mob, DOWN, buff=MED_LARGE_BUFF)
            label.set_color(color)

        self.play(
            FadeIn(labels),
            RandomSampling(qubits, qubit_samples),
        )
        self.wait()
        for _ in range(8):
            self.play(RandomSampling(qubits, qubit_samples))
            self.wait()

    def get_boxed_bits(self, value, length, height=0.5):
        boxes = Square().get_grid(1, length, buff=0)
        boxes.set_height(height)
        boxes.set_stroke(WHITE, 2)
        bits = BitString(value, length)
        for bit, box in zip(bits, boxes):
            bit.move_to(box)
        return VGroup(boxes, bits)


class AmbientStateVector(InteractiveScene):
    moving = False

    def construct(self):
        plane, axes = self.get_plane_and_axes()

        frame.reorient(14, 76, 0)
        frame.add_ambient_rotation(3 * DEG)
        self.add(plane, axes)

        # Vector
        vector = Vector(2 * normalize([1, 1, 2]), thickness=5)
        vector.set_fill(border_width=2)
        vector.set_color(TEAL)
        vector.always.set_perpendicular_to_camera(frame)

        self.play(GrowArrow(vector))

        if not self.moving:
            self.wait(36)
        else:
            for n in range(16):
                axis = normalize(np.random.uniform(-1, 1, 3))
                angle = np.random.uniform(0, PI)
                self.play(Rotate(vector, angle, axis=axis, about_point=ORIGIN))
                self.wait()

        # Show the sphere
        frame.reorient(16, 77, 0)

        sphere = Sphere(radius=2)
        sphere.always_sort_to_camera(self.camera)
        sphere.set_color(BLUE, 0.25)
        sphere_mesh = SurfaceMesh(sphere, resolution=(41, 21))
        sphere_mesh.set_stroke(WHITE, 0.5, 0.5)

        self.play(
            ShowCreation(sphere),
            Write(sphere_mesh, lag_ratio=1e-3),
        )
        self.wait(10)

    def get_plane_and_axes(self, scale_factor=2.0):
        # Add axes
        frame = self.frame
        axes = ThreeDAxes((-1, 1), (-1, 1), (-1, 1))
        plane = NumberPlane(
            (-1, 1 - 1e-5),
            (-1, 1 - 1e-5),
            faded_line_ratio=5
        )
        plane.background_lines.set_stroke(opacity=0.5)
        plane.faded_lines.set_stroke(opacity=0.25)
        plane.axes.set_stroke(opacity=0.25)
        result = VGroup(plane, axes)
        result.scale(scale_factor)

        return result


class RotatingStateVector(AmbientStateVector):
    moving = True


class FlipsToCertainDirection(AmbientStateVector):
    def construct(self):
        # Test
        frame = self.frame
        plane, axes = axes_group = self.get_plane_and_axes(scale_factor=3)

        vector = Vector(axes.c2p(1, 0, 0), fill_color=TEAL, thickness=5)
        vector.always.set_perpendicular_to_camera(frame)

        frame.reorient(-31, 72, 0)
        frame.add_ambient_rotation(DEG)
        self.add(frame)
        self.add(plane, axes)
        self.add(vector)

        # Show some flips
        theta = 15 * DEG
        h_plane = Square3D()
        h_plane.replace(plane)
        h_plane.set_color(WHITE, 0.15)
        diag_plane = h_plane.copy().rotate(theta, axis=DOWN)
        ghosts = VGroup()

        for n in range(1, 14):
            axis = [UP, DOWN][n % 2]
            shown_plane = [h_plane, diag_plane][n % 2]
            if n == 1:
                shown_plane = VectorizedPoint()
            ghosts.add(vector.copy())
            ghosts.generate_target()
            for n, vect in enumerate(ghosts.target[::-1]):
                vect.set_opacity(0.5 / (n + 1))
            self.play(
                MoveToTarget(ghosts),
                Rotate(vector, n * theta, axis=axis, about_point=ORIGIN),
            )
            shown_plane.set_opacity(0.15)

        self.wait(6)


class DisectAQuantumComputer(InteractiveScene):
    def construct(self):
        # Set up the machine with a random pile of gates
        wires = Line(4 * LEFT, 4 * RIGHT).replicate(4)
        wires.arrange(DOWN, buff=0.5)
        wires.shift(LEFT)

        gates = VGroup(
            self.get_labeled_box(wires[0], 0.1),
            self.get_cnot(wires[1], wires[0], 0.2),
            self.get_cnot(wires[3], wires[1], 0.3),
            self.get_cnot(wires[2], wires[3], 0.4),
            self.get_labeled_box(wires[3], 0.5, "X"),
            self.get_labeled_box(wires[1], 0.5, "Z"),
            self.get_cnot(wires[2], wires[1], 0.6),
            self.get_labeled_box(wires[1], 0.7),
            self.get_cnot(wires[3], wires[1], 0.8),
            self.get_cnot(wires[0], wires[1], 0.9),
        )
        mes_gates = VGroup(
            self.get_measurement(wires[0], 1),
            self.get_measurement(wires[1], 1),
            self.get_measurement(wires[2], 1),
            self.get_measurement(wires[3], 1),
        )

        circuit = Group(wires, Point(), gates, mes_gates)
        machine_rect = SurroundingRectangle(circuit)
        machine_rect.set_stroke(TEAL, 2)
        qc_label = get_quantum_computer_symbol(height=1)
        qc_label.next_to(machine_rect, UP)

        self.add(circuit, machine_rect, qc_label)

        # Show a program running through quantum wires (TODO< show the results)
        n_repetitions = 25

        wire_glows = Group(
            GlowDot(wire.get_start(), color=TEAL, radius=0.25)
            for wire in wires
        )
        wire_glows.set_z_index(-1)
        gate_glows = gates.copy()
        gate_glows.set_stroke(TEAL, 1)
        gate_glows.set_fill(opacity=0)
        for glow in gate_glows:
            glow.add_updater(lambda m: m.set_stroke(width=4 * np.exp(
                -(0.5 * (m.get_x() - wire_glows.get_x()))**2
            )))

        output = self.get_random_qubits()
        output.next_to(machine_rect, RIGHT, buff=LARGE_BUFF)

        for n in range(n_repetitions):
            self.add(gate_glows)
            wire_glows.set_x(wires.get_x(LEFT))
            wire_glows.set_opacity(1)
            self.play(
                wire_glows.animate.match_x(mes_gates),
                rate_func=linear,
                run_time=1
            )
            output_value = random.randint(0, 15)
            output[0].set_value(output_value)
            for mes_gate, bit in zip(mes_gates, output[0]):
                mes_gate[-1].set_stroke(opacity=0)
                mes_gate[-1][bit.get_value()].set_stroke(opacity=1)
            self.add(output)
            wire_glows.set_opacity(0)
            self.play(wire_glows.animate.shift(6 * RIGHT), rate_func=linear)

        self.remove(wire_glows, gate_glows)

        # Setup all 16 outputs
        machine = Group(circuit, machine_rect, qc_label)
        all_qubits = output.replicate(16)
        for n, qubits in enumerate(all_qubits):
            qubits[0].set_value(n)
        all_qubits.arrange(DOWN)
        all_qubits.set_height(FRAME_HEIGHT - 1)
        all_qubits.move_to(machine, RIGHT)
        all_qubits.set_y(0)

        all_qubit_rects = VGroup(
            SurroundingRectangle(qubits, buff=0.05).set_stroke(YELLOW, 1, 0.5)
            for qubits in all_qubits
        )
        output_rect = SurroundingRectangle(output)

        brace = Brace(all_qubit_rects, RIGHT, SMALL_BUFF)
        brace_label = brace.get_tex("2^4 = 16")
        brace_label.shift(0.2 * RIGHT + 0.05 * UP)

        self.play(ShowCreation(output_rect))
        self.wait()
        self.remove(output, output_rect)
        self.play(
            machine.animate.to_edge(LEFT),
            TransformFromCopy(VGroup(output), all_qubits),
            TransformFromCopy(VGroup(output_rect), all_qubit_rects),
        )
        self.play(
            GrowFromCenter(brace),
            Write(brace_label),
        )
        self.wait()

        # Show probability distribution
        qc_label.generate_target()
        qc_label.target.set_y(0).scale(1.5)
        comp_to_dist_arrow = Arrow(qc_label.target, all_qubits, buff=1.0, thickness=6)

        dists = [
            np.random.randint(1, 8, 16).astype(float),
            np.random.random(16) + 10 * np.eye(16)[10],
            np.random.random(16) + 5,
            np.random.randint(1, 8, 16).astype(float),
        ]
        for dist in dists:
            dist /= dist.sum()

        all_dist_rects = VGroup(
            self.get_dist_bars(dist, all_qubits, width_ratio=6)
            for dist in dists
        )
        dist_rects = all_dist_rects[0]

        dist_rect_labels = VGroup(
            Integer(100 * prob, unit=R"\\%", font_size=24, num_decimal_places=1).next_to(rect, RIGHT, SMALL_BUFF)
            for prob, rect in zip(dists[0], dist_rects)
        )

        self.play(
            FadeOut(machine_rect, DOWN),
            FadeOut(circuit, DOWN),
            MoveToTarget(qc_label)
        )
        self.play(
            GrowArrow(comp_to_dist_arrow),
            ReplacementTransform(all_qubit_rects, dist_rects, lag_ratio=0.1),
            FadeOut(brace, RIGHT),
            FadeOut(brace_label, RIGHT),
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, dist_rect_labels, shift=0.25 * RIGHT, lag_ratio=0.1, run_time=2))
        self.wait(2)
        self.play(FadeOut(dist_rect_labels))
        self.wait()
        for new_rects in all_dist_rects[1:]:
            self.play(Transform(dist_rects, new_rects, run_time=2))
            self.wait()

        # Show magnifying glass
        if False:
            # This is just ment for an insertion
            dist = VGroup(all_qubits, dist_rects)
            dist.shift(0.25 * LEFT)
            qc_label.scale(1.5).next_to(comp_to_dist_arrow, LEFT, MED_LARGE_BUFF)
            glass = get_magnifying_glass()
            glass.next_to(qc_label, UL).to_edge(UP)
            glass.save_state()
            dist_rects.save_state()

            wigglers = Superposition(
                VGroup(VGroup(qb, bar) for qb, bar in zip(all_qubits, dist_rects)).copy(),
                max_rot_vel=8,
                glow_stroke_opacity=0
            )

            index = 7
            new_dist = np.zeros(16)
            new_dist[index] = 7
            choice_rect = SurroundingRectangle(all_qubits[index])

            new_bars = self.get_dist_bars(new_dist, all_qubits, width_ratio=3)
            new_bars.set_stroke(width=0)
            new_bars[index].set_stroke(WHITE, 1)

            self.play(FadeIn(glass))
            for _ in range(2):
                self.play(
                    glass.animate.shift(qc_label.get_center() - glass[0].get_center()).set_anim_args(path_arc=-45 * DEG),
                    Transform(dist_rects, new_bars, time_span=(1.25, 1.5)),
                    FadeIn(choice_rect, time_span=(1.25, 1.5)),
                    run_time=2
                )
                self.wait()
                self.play(
                    Restore(glass, path_arc=45 * DEG),
                    FadeOut(choice_rect, time_span=(0.25, 0.5)),
                    run_time=1.5
                )
                self.wait()

            # Delicate state
            wigglers.set_offset_multiple(0)
            self.play(FadeOut(dist), FadeIn(wigglers))
            self.play(wigglers.animate.set_offset_multiple(0.05))
            self.wait(8)

            # For the chroma key
            qubits = all_qubits[index].copy()
            qubits.set_width(0.7 * qc_label.get_width()).move_to(qc_label)
            qubits.set_fill(border_width=3)

            self.clear()
            self.add(qubits)

        # Label it as 4 qubit
        title = Text("4 qubit quantum computer")
        title.next_to(qc_label, UP, buff=LARGE_BUFF)
        title.set_backstroke(BLACK, 3)
        underline = Underline(title, buff=-0.05)

        self.play(
            ShowCreation(underline, time_span=(1, 2)),
            Write(title, run_time=2),
        )
        self.wait()

        # k qubit case
        last_sym = title["4"][0]
        last_qubits = all_qubits
        last_dist_rects = dist_rects
        for k in [5, 6, "k"]:
            k_sym = Tex(str(k))
            k_sym.move_to(last_sym)
            k_sym.set_color(YELLOW)

            n_bits = k if isinstance(k, int) else 7

            multi_bitstrings = VGroup(
                BitString(n, length=n_bits)
                for n in range(2**n_bits)
            )
            new_qubits = VGroup(
                VGroup(bits, Ket(bits))
                for bits in multi_bitstrings
            )
            new_qubits.arrange(DOWN)
            new_qubits.set_height(FRAME_HEIGHT - 1)
            new_qubits.move_to(all_qubits)
            big_dist_rects = self.get_dist_bars(np.random.random(2**n_bits)**2, new_qubits)

            self.play(
                FadeOut(last_sym, 0.25 * UP),
                FadeIn(k_sym, 0.25 * UP),
                FadeOut(last_qubits),
                FadeOut(last_dist_rects),
                FadeIn(new_qubits),
                FadeIn(big_dist_rects),
            )
            self.wait()

            last_sym = k_sym
            last_qubits = new_qubits
            last_dist_rects = big_dist_rects

        # Count all outputs
        brace = Brace(last_dist_rects, RIGHT, SMALL_BUFF)
        brace_label = Tex(R"2^k", t2c={"k": YELLOW})
        brace_label.next_to(brace, RIGHT)

        bits = BitString(70, length=7)
        longer_example = last_qubits[100].copy().set_width(2.0)
        longer_example.next_to(brace_label, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)

        down_brace = Brace(longer_example[0], DOWN, SMALL_BUFF)
        down_brace_label = down_brace.get_tex("k").set_color(YELLOW)

        self.play(
            GrowFromCenter(brace),
            TransformFromCopy(last_sym, brace_label),
        )
        self.play(
            TransformFromCopy(last_qubits[70], longer_example),
        )
        self.play(
            GrowFromCenter(down_brace),
            TransformFromCopy(brace_label[1], down_brace_label)
        )
        self.wait()

        # Highlight the word qubit
        word_rect = SurroundingRectangle(title["qubit"], buff=0.1)
        word_rect.set_stroke(YELLOW, 2)
        self.play(ReplacementTransform(underline, word_rect))
        self.wait()
        title[0].set_opacity(0)
        self.play(
            FadeOut(title),
            FadeOut(word_rect),
            FadeOut(last_sym),
            LaggedStartMap(FadeOut, VGroup(brace, brace_label, longer_example, down_brace, down_brace_label)),
            FadeOut(last_qubits),
            FadeOut(last_dist_rects),
            FadeIn(all_qubits),
            FadeIn(dist_rects),
        )
        self.wait()

        # Emphasize you see only one
        frame = self.frame

        dist_rect = SurroundingRectangle(VGroup(comp_to_dist_arrow, all_qubits, dist_rects), buff=0.25)
        dist_rect.set_stroke(BLUE, 2)
        dist_rect.stretch(1.2, 0)
        output.set_width(2)
        output.next_to(dist_rect, RIGHT, buff=2.5)
        output.align_to(all_qubits, UP)

        dist_words = Text("Implicit", font_size=72)
        dist_words.next_to(dist_rect, UP)
        output_words = Text("What you see", font_size=72)
        output_words.next_to(output, UP).match_y(dist_words)

        sample_rects = VGroup(
            SurroundingRectangle(qb, buff=0.05)
            for qb, bar in zip(all_qubits, dist_rects)
        )
        sample_rects.set_stroke(YELLOW, 1)
        sample = 6
        output[0].set_value(sample)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (4.2, 0.7, 0.0), 9.86),
            FadeIn(dist_rect),
            FadeIn(dist_words, lag_ratio=0.1),
            FadeIn(output),
            FadeIn(output_words, lag_ratio=0.1),
            FadeIn(sample_rects[sample]),
            lag_ratio=0.2
        ))

        # Don't see multiple at once
        superposition_pieces = all_qubits.copy()
        superposition_pieces.set_width(1.5)
        superposition_pieces.space_out_submobjects(0.5)
        superposition_pieces.move_to(output, UP)
        superposition = Superposition(superposition_pieces)
        big_cross = Cross(superposition, stroke_width=[0, 15, 15, 15, 0])

        self.play(
            FadeIn(superposition),
            FadeOut(output),
            ShowCreation(big_cross)
        )
        self.wait(6)
        self.play(
            FadeOut(superposition),
            FadeOut(big_cross),
        )

        # Show some random samples
        n_samples = 8
        output_choices = output.replicate(16)
        for n, choice, sample_rect in zip(it.count(), output_choices, sample_rects):
            choice[0].set_value(n)
            choice.shift(np.random.uniform(-0.1, 0.1, 3))
            choice.set_backstroke(TEAL, 2)
            choice.add(sample_rect)

        self.remove(output, sample_rects)
        selection = VGroup()
        for _ in range(n_samples):
            self.play(RandomSampling(selection, list(output_choices), weights=dists[-1]))
            self.wait()

        # Concentrate probability to one value
        conc_dist = np.zeros(16)
        num = 7
        conc_dist[num] = 1
        new_rects = self.get_dist_bars(conc_dist, all_qubits, width_ratio=2.0)
        self.play(
            FadeOut(selection),
            Transform(dist_rects, new_rects),
            dist_rect.animate.stretch(1.2, 0, about_edge=LEFT),
            UpdateFromFunc(dist_words, lambda m: m.match_x(dist_rect)),
            output_words.animate.shift(RIGHT),
        )
        self.wait()
        self.play(ShowCreation(sample_rects[num]))
        output[0].set_value(num)
        output.match_x(output_words)
        self.play(TransformFromCopy(all_qubits[num], output))
        self.wait()

        # Back to another distribution
        point1_index = 2
        point5_index = 3
        new_dist = np.random.random(16)
        new_dist[point1_index] = new_dist[point5_index] = 0
        new_dist *= (1.0 - 0.25 - 0.01) / new_dist.sum()
        new_dist[point1_index] = 0.01
        new_dist[point5_index] = 0.25

        self.play(Transform(dist_rects, self.get_dist_bars(new_dist, all_qubits)))
        self.remove(output, selection, sample_rects)
        VGroup(choice[:-1] for choice in output_choices).match_x(output_words)
        self.play(RandomSampling(selection, list(output_choices), weights=new_dist))
        self.remove(selection)
        self.add(output, sample_rects[num])
        self.wait()

        # Ask where the distribution comes from
        q_marks = Tex(R"???", font_size=96)
        dist_group = VGroup(all_qubits, dist_rects)
        q_marks.next_to(dist_group, UP, MED_LARGE_BUFF)

        self.play(
            dist_rect.animate.surround(dist_group),
            FadeTransformPieces(dist_words, q_marks),
        )
        self.wait()

        dist_group.add(dist_rect)

        # Introduce the state vector
        state = np.array([random.choice([-1, 1]) * np.sqrt(p) for p in new_dist])
        state[point5_index] = 0.5
        state[point1_index] = -0.1
        state_vector = DecimalMatrix(np.zeros((16, 1)), decimal_config=dict(include_sign=True, edge_to_fix=LEFT))
        for value, elem in zip(state, state_vector.elements):
            elem.set_value(value)
        state_vector.match_height(all_qubits)
        state_vector.next_to(qc_label, RIGHT, LARGE_BUFF)

        vector_title = Text("State Vector", font_size=60)
        vector_title.set_color(TEAL)
        vector_title.next_to(state_vector, UP, MED_LARGE_BUFF)

        comp_to_dist_arrow.target = comp_to_dist_arrow.generate_target()
        comp_to_dist_arrow.target.next_to(state_vector, RIGHT)

        vect_lines = self.get_connecting_lines(qc_label, state_vector, from_buff=-0.1, to_buff=0.1)

        self.play(LaggedStart(
            ShowCreation(vect_lines, lag_ratio=0),
            FadeInFromPoint(state_vector, qc_label.get_right()),
            MoveToTarget(comp_to_dist_arrow),
            dist_group.animate.next_to(comp_to_dist_arrow.target, RIGHT),
            MaintainPositionRelativeTo(sample_rects[num], dist_group),
            MaintainPositionRelativeTo(q_marks, dist_group),
            VGroup(output, output_words).animate.shift(0.5 * RIGHT),
        ))
        self.play(Write(vector_title))
        self.wait()

        # Label the positions of the vector
        pre_indices = VGroup(qb[0] for qb in all_qubits)
        indices = pre_indices.copy()
        indices.scale(0.75)
        indices.set_color(GREY_C)
        for bits, entry in zip(indices, state_vector.get_entries()):
            bits.next_to(state_vector, LEFT, buff=0.2)
            bits.match_y(entry)

        self.play(
            VGroup(qc_label, vect_lines).animate.shift(0.75 * LEFT),
            LaggedStart(
                (TransformFromCopy(pre_bits, bits, path_arc=30 * DEG)
                for pre_bits, bits in zip(pre_indices, indices)),
                lag_ratio=0.25,
                run_time=6
            ),
        )
        self.wait()

        # Fundamental rule
        ne_sign = Tex(R"\\ne", font_size=120)
        ne_sign.move_to(comp_to_dist_arrow)
        ne_sign.set_color(RED)

        rule = Tex(R"x \\rightarrow |x|^2", font_size=42)
        rule.next_to(comp_to_dist_arrow, UP, buff=SMALL_BUFF)

        template_eq = Tex(R"(+0.50)^2 = 0.25", font_size=20)
        entry_template = template_eq.make_number_changeable("+0.50", include_sign=True)
        percent_template = template_eq.make_number_changeable("0.25")

        vector_entries = state_vector.get_entries()

        bar_values = VGroup()
        for bar, entry in zip(dist_rects, vector_entries):
            entry_template.set_value(entry.get_value())
            percent_template.set_value(entry.get_value()**2)
            template_eq.next_to(bar, RIGHT, SMALL_BUFF)
            bar_values.add(template_eq.copy())

        self.play(
            FadeOut(comp_to_dist_arrow),
            FadeIn(ne_sign),
        )
        self.wait()
        self.play(
            FadeIn(comp_to_dist_arrow),
            FadeOut(ne_sign),
        )
        self.play(
            ReplacementTransform(q_marks, rule, path_arc=90 * DEG, run_time=1.5),
        )
        self.wait()
        self.play(
            dist_rect.animate.stretch(1.5, 0, about_edge=LEFT).set_anim_args(time_span=(1, 2)),
            FadeOut(sample_rects[num]),
            LaggedStart(
                (TransformFromCopy(entry, value[1], path_arc=-30 * DEG)
                for entry, value in zip(vector_entries, bar_values)),
                lag_ratio=0.5,
            ),
            LaggedStart(
                (FadeIn(value[0])
                for value in bar_values),
                lag_ratio=0.5,
                time_span=(2, 6)
            ),
            LaggedStart(
                (FadeIn(value[2:])
                for value in bar_values),
                lag_ratio=0.5,
                time_span=(2, 6)
            ),
            frame.animate.reorient(0, 0, 0, (2.15, 0.04, 0.0), 7.64),
            run_time=6,
        )
        self.remove(vector_title, output_words, output)
        self.wait()

        # Process the vector
        alt_states = [normalize(np.random.uniform(-1, 1, 16)) for x in range(2)]
        label_line = Line(UP, DOWN).set_height(1.0).next_to(qc_label.get_right(), LEFT, SMALL_BUFF)
        entries = state_vector.get_entries()
        comp_lines = VGroup(
            Line(
                label_line.pfp(alpha),
                entry.get_center(),
            ).insert_n_curves(20).set_stroke(
                color=random_bright_color(hue_range=(0.3, 0.5)),
                width=(0, 3, 3, 3, 0),
            )
            for alpha, entry in zip(np.linspace(0, 1, len(entries)), entries)
        )
        comp_lines.shuffle()

        self.play(FlashAround(rule, time_width=1.5, run_time=2))
        self.play(FadeOut(bar_values))
        for new_state in [*alt_states, state]:
            comp_lines.shuffle()
            self.play(
                LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.02),
                Transform(dist_rects, self.get_dist_bars(new_state**2, all_qubits), time_span=(0.75, 1.75)),
                *(
                    ChangeDecimalToValue(entry, value, time_span=(0.75, 1.75))
                    for entry, value in zip(entries, new_state)
                ),
                run_time=2
            )

        # Highlight a few examples
        highlight_rect = SurroundingRectangle(
            VGroup(indices[point5_index], bar_values[point5_index])
        )
        highlight_rect.set_stroke(YELLOW, 2, 0)
        groups = [vector_entries, all_qubits, dist_rects, bar_values]
        bar_values.set_opacity(0)

        for index in [point5_index, point1_index]:
            self.play(
                frame.animate.reorient(0, 0, 0, (3.36, 0.87, 0.0), 5.60),
                dist_rect.animate.set_stroke(opacity=0),
                highlight_rect.animate.set_stroke(opacity=1).match_y(all_qubits[index]),
                *(
                    group[slc].animate.set_opacity(0.25)
                    for group in groups
                    for slc in [slice(0, index), slice(index + 1, None)]
                ),
                *(
                    group[index].animate.set_opacity(1)
                    for group in groups
                ),
                run_time=2,
            )
            self.play(
                FlashAround(indices[index], buff=SMALL_BUFF, color=RED, time_width=1.5),
                WiggleOutThenIn(indices[index]),
                run_time=2
            )
            self.wait()
            self.play(TransformFromCopy(vector_entries[index], bar_values[index][1], run_time=2, path_arc=-45 * DEG))
            self.play(FlashAround(bar_values[index][-1], buff=SMALL_BUFF, color=RED, time_width=1.5))
            self.wait()

        self.play(
            frame.animate.reorient(0, 0, 0, (2.13, 1.33, 0.0), 10.48),
            FadeOut(highlight_rect, time_span=(0, 1)),
            FadeIn(vector_title),
            *(
                group.animate.set_opacity(1)
                for group in groups
            ),
            FadeOut(bar_values, lag_ratio=0.01),
            run_time=4
        )

        # Concentrate value
        if False:  # Only used for an insertion
            # Concentrate onto one value
            og_state = state
            frame.reorient(0, 0, 0, (0.94, 0.34, 0.0), 9.20)
            target_state = np.zeros(16)
            key = 5
            target_state[key] = 1
            for alpha in [0.1, 0.1, 0.2, 0.25, 0.25, 0.25, 0.25, 1.0]:
                new_state = normalize(interpolate(state, target_state, alpha))

                self.play(
                    Transform(dist_rects, self.get_dist_bars(new_state**2, all_qubits, width_ratio=4), time_span=(0.5, 1.5)),
                    LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.01, run_time=2.0),
                    *(
                        ChangeDecimalToValue(entry, value, time_span=(0.5, 1.5))
                        for entry, value in zip(state_vector.elements, new_state)
                    ),
                )

                state = new_state

            # Comment on the value
            rect = SurroundingRectangle(all_qubits[key])
            self.play(ShowCreation(rect, run_time=2))

        # Show the squares of all the values
        sum_expr = Tex(
            R"x_{0}^2 + x_{1}^2 + x_{2}^2 + x_{3}^2 + \\cdots + x_{14}^2 + x_{15}^2 = 1",
            font_size=60
        )
        sum_expr.next_to(vector_title, UP, LARGE_BUFF)
        sum_expr.match_x(frame)

        self.play(
            LaggedStart(
                (FadeTransform(vector_entries[n].copy(), sum_expr[fR"x_{{{n}}}^2"])
                for n in [0, 1, 2, 3, 14, 15]),
                lag_ratio=0.05,
                run_time=2
            ),
            FadeTransformPieces(vector_entries[4:14].copy(), sum_expr[R"\\cdots"][0], time_span=(0.5, 2.0)),
            Write(sum_expr["+"], time_span=(0.5, 1.5)),
            Write(sum_expr["= 1"], time_span=(1, 2)),
        )
        self.wait()
        self.play(FadeOut(sum_expr))

        # Flip some signs
        top_eq = Tex(R"(+0.50)^2 = 0.25", font_size=60)
        top_eq.move_to(sum_expr)
        top_value = top_eq.make_number_changeable("+0.50", include_sign=True)

        entry = vector_entries[point5_index]
        entry_rect = SurroundingRectangle(entry, buff=0.1)
        entry_rect.set_stroke(YELLOW, 1)
        value_rect = SurroundingRectangle(top_value)
        value_rect.set_stroke(YELLOW, 2)

        lines = VGroup(
            Line(entry_rect.get_corner(UL), value_rect.get_corner(DL)),
            Line(entry_rect.get_corner(UR), value_rect.get_corner(DR)),
        )
        lines.set_stroke(YELLOW, 1)

        self.play(ShowCreation(entry_rect))
        self.play(
            TransformFromCopy(entry_rect, value_rect),
            TransformFromCopy(entry, top_value),
            ShowCreation(lines, lag_ratio=0),
            FadeOut(vector_title),
        )
        self.wait()
        self.play(
            entry.animate.set_value(-0.5),
            top_value.animate.set_value(-0.5),
        )
        self.wait()
        self.play(
            Write(top_eq[0]),
            Write(top_eq[2:]),
            FadeOut(value_rect)
        )
        self.wait()
        self.play(
            entry.animate.set_value(0.5),
            top_value.animate.set_value(0.5),
        )
        self.wait()
        self.play(
            Uncreate(lines, lag_ratio=0),
            ReplacementTransform(top_value, entry),
            FadeOut(top_eq[0]),
            FadeOut(top_eq[2:]),
            FadeOut(entry_rect),
            FadeIn(vector_title),
            frame.animate.reorient(0, 0, 0, (0.45, 0.46, 0.0), 8.70),
        )

        # Flip some more signs
        signs = VGroup(entry[0] for entry in vector_entries)
        vector_entries.save_state()
        sign_choice = VGroup(Tex("+").set_color(BLUE), Tex("-").set_color(RED))
        sign_choice.set_width(1.2 * signs[0].get_width())
        sign_choices = VGroup(
            sign_choice.copy().move_to(sign, RIGHT)
            for sign in signs
        )

        for n in range(3):
            self.play(LaggedStart(*(
                Transform(sign, random.choice(choice), path_arc=PI)
                for sign, choice in zip(signs, sign_choices)),
                lag_ratio=0.15,
                run_time=2
            ))
        self.play(
            Restore(vector_entries),
        )

        # Go down to two dimensions
        small_state = normalize([1, 2])
        small_state_vector = DecimalMatrix(small_state.reshape([2, 1]), v_buff=1.0)
        small_state_vector.set_height(2.0)
        small_state_vector.move_to(state_vector, RIGHT)

        bits = VGroup(Integer(0), Integer(0))
        bits.arrange(DOWN, MED_LARGE_BUFF)
        qubits = VGroup(
            VGroup(bit, Ket(bit))
            for bit in bits
        )
        qubits.scale(1.5)
        qubits.arrange_to_fit_height(small_state_vector.get_entries().get_height())
        qubits.next_to(comp_to_dist_arrow, RIGHT, MED_LARGE_BUFF)
        qubits.reverse_submobjects()
        qubits[1][0].set_value(1)

        small_dist_bars = self.get_dist_bars(small_state**2, qubits, width_ratio=2)

        small_vect_lines = self.get_connecting_lines(qc_label, small_state_vector, from_buff=-0.1, to_buff=0.1)

        self.remove(state_vector, all_qubits, dist_rects, vect_lines)
        self.play(LaggedStart(
            FadeOut(indices, scale=0.25),
            TransformFromCopy(vect_lines, small_vect_lines),
            TransformFromCopy(state_vector.get_brackets(), small_state_vector.get_brackets()),
            TransformFromCopy(state_vector.get_entries(), small_state_vector.get_entries()),
            vector_title.animate.next_to(small_state_vector, UP, MED_LARGE_BUFF),
            FadeTransformPieces(all_qubits.copy(), qubits),
            TransformFromCopy(dist_rects, small_dist_bars),
            lag_ratio=0.1,
            run_time=3
        ))
        self.add(small_state_vector)
        self.wait()

        self.play(
            FadeOut(VGroup(small_vect_lines, small_state_vector, qubits, small_dist_bars)),
            FadeIn(VGroup(vect_lines, indices, state_vector, all_qubits, dist_rects)),
            vector_title.animate.next_to(state_vector, UP, MED_LARGE_BUFF),
        )

        # Apply Grover's
        self.play(
            frame.animate.reorient(0, 0, 0, (2, 0.45, 0.0), 9),
            qc_label[1].animate.set_color(BLUE),
        )
        key = 12
        state0 = np.sqrt(1.0 / 16) * np.ones(16)
        new_states = [state0]
        for n in range(3):
            state = new_states[-1].copy()
            state[key] *= -1
            new_states.append(state)
            new_states.append(2 * np.dot(state0, state) * state0 - state)

        key_icon = get_key_icon()
        key_entry_rect = SurroundingRectangle(VGroup(indices[key], vector_entries[key]))
        key_entry_rect.stretch(1.25, 0, about_edge=RIGHT)
        key_entry_rect.set_stroke(YELLOW, 2)

        for n, state in enumerate(new_states):
            anims = [Transform(
                dist_rects, self.get_dist_bars(state**2, all_qubits, width_ratio=4.0),
                time_span=(1, 2)
            )]
            if n == 1:
                # Highlight the outcomes
                entry_rects = VGroup(SurroundingRectangle(e, buff=0.05) for e in vector_entries)
                qubit_rects = VGroup(
                    SurroundingRectangle(VGroup(*pair), buff=0.05)
                    for pair in zip(all_qubits, dist_rects)
                )
                VGroup(entry_rects, qubit_rects).set_stroke(YELLOW, 2)
                self.play(
                    LaggedStartMap(VFadeInThenOut, entry_rects, lag_ratio=0.1),
                    LaggedStartMap(VFadeInThenOut, qubit_rects, lag_ratio=0.1),
                    run_time=6
                )
                self.wait()

                # Show the key
                bits = indices[key]
                bits.save_state()
                self.play(bits.animate.scale(2).set_color(WHITE).move_to(8 * RIGHT))

                key_icon.set_height(1.5 * bits.get_height())
                key_icon.next_to(bits, LEFT, SMALL_BUFF)
                self.play(Write(key_icon))
                self.wait()

                self.play(
                    Restore(bits),
                    key_icon.animate.scale(0.5).next_to(bits.saved_state, LEFT, buff=SMALL_BUFF),
                )

            if n % 2 == 1:
                neg1 = Tex(R"\\times -1")
                neg1.next_to(key_entry_rect, RIGHT)
                neg1.set_color(YELLOW)
                anims.append(VFadeInThenOut(neg1, run_time=2))
                anims.append(VFadeInThenOut(key_entry_rect, run_time=2))
                anims.append(ChangeDecimalToValue(vector_entries[key], state[key], time_span=(0.5, 1.5)))
            else:
                anims.append(
                    LaggedStartMap(VShowPassingFlash, comp_lines, time_width=2.0, lag_ratio=0.005, run_time=2)
                )
                anims.extend([
                    ChangeDecimalToValue(entry, value, time_span=(1, 2))
                    for entry, value in zip(vector_entries, state)
                ])
            self.play(*anims)
            self.wait()

        # Read out from memory
        glass = get_magnifying_glass()
        glass.next_to(qc_label, UP, MED_LARGE_BUFF)
        glass.scale(0.75)
        glass.to_edge(LEFT, buff=0).shift(frame.get_x() * RIGHT)

        self.play(FadeIn(glass))
        self.play(
            glass.animate.shift(qc_label.get_center() - glass[0].get_center()).set_anim_args(path_arc=-45 * DEG),
            rate_func=there_and_back_with_pause,
            run_time=6
        )
        self.wait()

        qubits = all_qubits[key].copy()
        qubits.move_to(qc_label)
        qubits.scale(1.25)
        black_rect = FullScreenRectangle().set_fill(BLACK, 1)
        black_rect.fix_in_frame()
        self.add(black_rect, qubits)
        self.wait()

    def get_labeled_box(self, wire, alpha, label="H", size=0.5):
        box = Square(size)
        box.set_stroke(WHITE, 1)
        box.set_fill(BLACK, 1)
        box.move_to(wire.pfp(alpha))

        vect = rotate_vector(UP, PI / 8)
        arrow = VGroup(
            Vector(size * vect, thickness=1).center(),
            Vector(-size * vect, thickness=1).center(),
        )
        arrow.move_to(box)

        label = Tex(label)
        label.set_height(0.5 * size)
        label.move_to(box)

        return VGroup(box, label)

    def get_cnot(self, wire1, wire2, alpha):
        oplus = Tex(R"\\oplus")
        dot = Dot()
        oplus.move_to(wire1.pfp(alpha))
        dot.move_to(wire2.pfp(alpha))
        connector = Line(dot, oplus, buff=0)
        connector.set_stroke(WHITE, 1)
        return VGroup(oplus, connector, dot)

    def get_measurement(self, wire, alpha, size=0.5):
        box = Square(size)
        box.set_stroke(WHITE, 1)
        box.set_fill(BLACK, 1)
        box.move_to(wire.pfp(alpha))
        arc = Arc(PI / 4, PI / 2)
        arc.set_width(0.7 * box.get_width())
        arc.move_to(box)
        arc.set_stroke(WHITE, 1)
        lines = VGroup(Line(ORIGIN, 0.2 * vect) for vect in [UL, UR])
        lines.move_to(box)
        lines.set_stroke(WHITE, 1)
        lines[1].set_stroke(opacity=0)
        return VGroup(box, arc, lines)

    def get_random_qubits(self):
        bits = BitString(random.randint(0, 15))
        ket = Ket(bits)
        return VGroup(bits, ket)

    def get_connecting_lines(self, from_mob, to_mob, from_buff=0, to_buff=0, stroke_color=TEAL_A, stroke_width=2):
        l_ur = from_mob.get_corner(UR) + from_buff * UR
        l_dr = from_mob.get_corner(DR) + from_buff * DR
        v_ul = to_mob.get_corner(UL) + to_buff * UL
        v_dl = to_mob.get_corner(DL) + to_buff * DL

        lines = VGroup(
            CubicBezier(l_ur, l_ur + RIGHT, v_ul + LEFT, v_ul),
            CubicBezier(l_dr, l_dr + RIGHT, v_dl + LEFT, v_dl),
        )
        lines.set_stroke(TEAL_A, 2)
        return lines

    def get_dist_bars(
        self,
        dist,
        objs,
        height_ratio=0.8,
        width_ratio=8,
        fill_colors=(BLUE_D, GREEN),
        stroke_color=WHITE,
        stroke_width=1,
    ):
        normalized_dist = np.array(dist) / sum(dist)
        height = objs[0].get_height() * height_ratio
        rects = VGroup(
            Rectangle(width_ratio * p, height).next_to(obj)
            for p, obj in zip(normalized_dist, objs)
        )
        rects.set_fill(opacity=1)
        rects.set_submobject_colors_by_gradient(*fill_colors)
        rects.set_stroke(stroke_color, stroke_width)
        return rects


class Qubit(DisectAQuantumComputer):
    def construct(self):
        # Set up plane
        frame = self.frame
        plane = self.get_plane()
        zero_label, one_label = qubit_labels = self.get_qubit_labels(plane)

        frame.move_to(plane)
        self.add(plane)
        self.add(qubit_labels)

        # Add vector
        vector = self.get_vector(plane)
        vector_label = DecimalMatrix([[1.0], [0.0]], bracket_h_buff=0.1, decimal_config=dict(include_sign=True))
        vector_label.add_background_rectangle()
        vector_label.scale(0.5)
        vector_label.set_backstroke(BLACK, 5)
        theta_tracker = ValueTracker(0)
        vector.add_updater(lambda m: m.set_angle(theta_tracker.get_value()))
        vector.add_updater(lambda m: m.shift(plane.c2p(0, 0) - m.get_start()))

        def get_state():
            theta = theta_tracker.get_value()
            return np.array([math.cos(theta), math.sin(theta)])

        def position_label(vector_label):
            x, y = get_state()
            buff = SMALL_BUFF + 0.5 * interpolate(vector_label.get_width(), vector_label.get_height(), x**2)

            vect = normalize(vector.get_vector())
            vector_label.move_to(vector.get_end() + buff * vect)

        def update_coordinates(vector_label):
            for element, value in zip(vector_label.elements, get_state()):
                element.set_value(value)

        vector_label.add_updater(position_label)
        vector_label.add_updater(update_coordinates)

        self.add(vector, vector_label)
        self.play(theta_tracker.animate.set_value(240 * DEG), run_time=5)
        self.play(theta_tracker.animate.set_value(120 * DEG), run_time=4)
        self.wait()

        # Add rule
        var_vect = TexMatrix([["x"], ["y"]], bracket_h_buff=0.1)
        var_vect.next_to(plane, RIGHT, buff=1.0)
        var_vect.to_edge(UP)
        var_vect.add_background_rectangle()

        coord_vect = DecimalMatrix([[0], [1]], v_buff=0.5, bracket_h_buff=0.1, decimal_config=dict(include_sign=True))
        coord_vect.scale(0.75)
        coord_vect.next_to(var_vect, DOWN, buff=2.25, aligned_edge=RIGHT)
        coord_vect.clear_updaters()
        coord_vect.add_background_rectangle()
        coord_vect.add_updater(update_coordinates)

        prob_rule = VGroup(
            Tex(R"P(0) = x^2"),
            Tex(R"P(1) = y^2"),
        )
        prob_rule.arrange(DOWN)
        prob_rule.next_to(var_vect, RIGHT, buff=1.5)
        var_arrow = Arrow(var_vect, prob_rule, buff=0.25)

        qubits = qubit_labels.copy()
        qubits.scale(2)
        qubits.arrange(DOWN)
        qubits.match_y(coord_vect)
        qubits.align_to(prob_rule, LEFT)
        dist_bars = always_redraw(lambda: self.get_dist_bars(get_state()**2, qubits, width_ratio=1.5))
        dist_bars.suspend_updating()

        dist_arrow = Arrow(coord_vect, qubits, buff=0.25)
        dist_arrow_label = Tex(R"c \\rightarrow c^2", font_size=24)
        dist_arrow_label.next_to(dist_arrow, UP, SMALL_BUFF)

        bar_labels = VGroup(
            Integer(25, unit=R"\\%", font_size=24),
            Integer(75, unit=R"\\%", font_size=24),
        )

        def update_bar_labels(bar_labels):
            for label, bar, value in zip(bar_labels, dist_bars, get_state()):
                label.set_value(np.round(100 * value**2, 0))
                label.next_to(bar, RIGHT, SMALL_BUFF)

        bar_labels.add_updater(update_bar_labels)

        top_rule_rect = SurroundingRectangle(prob_rule[0])
        bar_rect = SurroundingRectangle(VGroup(qubits[0], bar_labels[0]))
        bar_rect.get_width()
        bar_rect.set_width(3, stretch=True, about_edge=LEFT)
        VGroup(top_rule_rect, bar_rect).set_stroke(YELLOW, 1.5)

        dist_group = VGroup(coord_vect, dist_arrow, dist_arrow_label, qubits, dist_bars, bar_labels)

        self.play(LaggedStart(
            TransformFromCopy(vector_label.copy().clear_updaters(), var_vect),
            frame.animate.center(),
            GrowArrow(var_arrow),
            FadeIn(prob_rule),
            run_time=2,
            lag_ratio=0.1
        ))
        self.play(ShowCreation(top_rule_rect))
        self.wait()
        self.play(LaggedStart(
            Transform(var_vect.copy(), coord_vect.copy().clear_updaters(), remover=True),
            TransformFromCopy(var_arrow, dist_arrow),
            FadeIn(dist_arrow_label, DOWN),
            TransformFromCopy(VGroup(pr[1:4] for pr in prob_rule).copy(), qubits),
            FadeTransformPieces(VGroup(pr[5:] for pr in prob_rule).copy(), dist_bars),
            TransformFromCopy(top_rule_rect, bar_rect),
            lag_ratio=1e-2
        ))
        self.add(coord_vect)
        self.play(FadeIn(bar_labels))
        self.wait()
        dist_bars.resume_updating()
        self.play(theta_tracker.animate.set_value(10 * DEG), run_time=8)
        self.wait()
        self.play(
            top_rule_rect.animate.match_y(prob_rule[1]),
            bar_rect.animate.match_y(qubits[1]),
        )
        self.play(theta_tracker.animate.set_value(90 * DEG), run_time=8)
        self.wait()
        self.play(FadeOut(top_rule_rect), FadeOut(bar_rect))
        self.wait()
        self.play(theta_tracker.animate.set_value(60 * DEG), run_time=2)

        # Note x^2 + y^2
        var_group = VGroup(var_vect, var_arrow, prob_rule)
        pythag = Tex(R"x^2 + y^2 = 1")
        pythag.match_x(var_group)
        pythag.to_edge(UP)

        self.play(LaggedStart(
            var_group.animate.shift(1.5 * DOWN),
            TransformFromCopy(prob_rule[0]["x^2"][0], pythag["x^2"][0]),
            Write(pythag["+"][0]),
            TransformFromCopy(prob_rule[1]["y^2"][0], pythag["y^2"][0]),
        ))
        self.play(Write(pythag["= 1"][0]), run_time=1)
        self.add(pythag)
        self.wait()

        # Show vector length
        brace = LineBrace(vector, DOWN, buff=0)
        brace_label = brace.get_tex(R"\\sqrt{x^2 + y^2} = 1", font_size=36)
        brace_label.shift(MED_SMALL_BUFF * UP)

        circle = Circle(radius=vector.get_length())
        circle.move_to(plane)
        circle.set_stroke(YELLOW, 2)
        circle.rotate(vector.get_angle(), about_point=plane.get_center())

        vector_ghost = vector.copy()
        vector_ghost.clear_updaters()
        vector_ghost.set_opacity(0.25)

        self.play(
            GrowFromCenter(brace),
            TransformMatchingTex(pythag.copy(), brace_label, run_time=1)
        )
        self.wait()
        self.add(vector_ghost)
        self.play(
            ShowCreation(circle),
            theta_tracker.animate.set_value(theta_tracker.get_value() + TAU),
            run_time=6,
        )
        self.remove(vector_ghost)
        self.wait()
        self.play(
            FadeOut(brace),
            FadeOut(brace_label),
            circle.animate.set_stroke(width=1, opacity=0.5)
        )

        # Name this as a qubit
        title = Text("Qubit", font_size=90)
        title.next_to(plane.get_corner(UL), DR, MED_SMALL_BUFF)
        title.set_backstroke(BLACK, 4)

        self.play(Write(title))
        self.wait()

        # Illustrate collpase
        if False:
            # Just for an insertion
            small_plane = self.small_plane(plane)
            small_plane.set_z_index(-1)
            self.remove(plane, vector_label, pythag)
            self.add(small_plane)
            self.remove(title)
            theta_tracker.set_value(45 * DEG)

            self.wait()
            self.play(theta_tracker.animate.set_value(90 * DEG), run_time=0.15)
            self.wait()
            self.play(theta_tracker.animate.set_value(45 * DEG))
            self.wait()
            self.play(theta_tracker.animate.set_value(0), run_time=0.15)
            self.wait()

            # Put in the qubits
            one = one_label.copy()
            zero = zero_label.copy()
            for mob in [zero, one]:
                mob.set_height(1)
                mob.set_fill(WHITE, 1, border_width=3)
                mob.move_to(plane)

            self.clear()
            self.add(zero)

        # Ambient change
        theta_tracker.set_value(60 * DEG)
        for value in [180 * DEG, 90 * DEG, 0, 120 * DEG]:
            self.play(theta_tracker.animate.set_value(value), run_time=6)

        # Show kets
        self.play(FadeOut(vector_label))
        self.play(theta_tracker.animate.set_value(0), run_time=2)

        zero_vect = vector.copy().clear_updaters().set_fill(BLUE, 0.5)
        self.play(zero_label.animate.scale(2).next_to(zero_vect, DOWN, buff=0))
        self.play(
            FlashAround(zero_label, run_time=2, time_width=1.5, color=BLUE),
            FadeIn(zero_vect),
        )
        self.wait()
        self.play(theta_tracker.animate.set_value(90 * DEG), run_time=2)

        one_vect = vector.copy().clear_updaters().set_fill(GREEN, 0.5)
        self.play(
            one_label.animate.scale(2).next_to(one_vect, LEFT, buff=0)
        )
        self.play(
            FlashAround(one_label, run_time=2, time_width=1.5, color=GREEN),
            FadeIn(one_vect),
        )
        self.wait()

        # General unit vector
        var_vect_copy = var_vect.copy()
        self.play(theta_tracker.animate.set_value(55 * DEG), run_time=2)
        self.play(var_vect_copy.animate.scale(0.75).next_to(vector.get_end(), UR, buff=0))

        weighted_sum = Tex(R"x|0\\rangle + y|1\\rangle")
        weighted_sum.next_to(vector.get_end(), RIGHT)
        weighted_sum.shift(SMALL_BUFF * UL)
        weighted_sum.set_backstroke(BLACK, 5)
        red_cross = Cross(var_vect_copy)
        red_cross.scale(1.5)

        self.play(ShowCreation(red_cross))
        self.wait()
        self.play(
            FadeOut(red_cross),
            FadeTransform(var_vect_copy.get_entries()[0].copy(), weighted_sum["x"]),
            FadeTransform(var_vect_copy.get_entries()[1].copy(), weighted_sum["y"]),
            FadeOut(var_vect_copy),
            FadeTransform(zero_label.copy(), weighted_sum[R"|0\\rangle"]),
            FadeTransform(one_label.copy(), weighted_sum[R"|1\\rangle"]),
            FadeIn(weighted_sum["+"]),
        )
        self.wait()

        # Prepare to show a gate
        faders = VGroup(
            pythag, var_group, dist_group
        )
        faders.clear_updaters()

        plane2 = plane.copy()
        plane2.next_to(plane, RIGHT, buff=5)
        planes = VGroup(plane, plane2)

        arrow = Arrow(plane, plane2, thickness=8)
        arrow_label = Text("Hadamard gate", font_size=60)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        matrix_tex = Tex(R"\\frac{1}{\\sqrt{2}} \\left[\\begin{array}{cc} 1 & 1 \\\\ 1 & -1 \\end{array}\\right]", font_size=24)
        matrix_tex.set_fill(GREY_B)
        matrix_tex.next_to(arrow, DOWN, SMALL_BUFF)

        self.play(
            frame.animate.set_width(planes.get_width() + LARGE_BUFF).move_to(planes),
            LaggedStartMap(FadeOut, faders, shift=0.5 * DOWN, lag_ratio=0.25),
            zero_vect.animate.set_fill(opacity=1),
            one_vect.animate.set_fill(opacity=1),
            FadeIn(plane2, shift=RIGHT),
            GrowArrow(arrow),
            FadeIn(arrow_label, lag_ratio=0.1),
            FadeIn(matrix_tex),
            run_time=2
        )

        # Hadamard gate
        vector.clear_updaters()
        movers = VGroup(circle, zero_vect, one_vect, vector)
        movers_image = movers.copy()
        movers_image.flip(axis=rotate_vector(RIGHT, PI / 8), about_point=plane.get_center())
        movers_image.move_to(plane2)

        labels = VGroup(zero_label, one_label, weighted_sum)
        labels.set_backstroke(BLACK, 3)
        labels_image = VGroup(
            Tex(R"\\text{H}|0\\rangle").scale(0.75).rotate(45 * DEG).next_to(movers_image[1].get_center(), UL, buff=-0.05),
            Tex(R"\\text{H}|1\\rangle").scale(0.75).rotate(-45 * DEG).next_to(movers_image[2].get_center(), DL, buff=-0.05),
            Tex(R"\\text{H}\\big(x|0\\rangle + y|1\\rangle\\big)").scale(0.75).next_to(movers_image[3].get_end(), RIGHT, SMALL_BUFF),
        )
        labels_image.set_backstroke(BLACK, 7)

        self.play(
            TransformFromCopy(movers, movers_image, path_arc=-30 * DEG),
            TransformFromCopy(labels, labels_image, path_arc=-30 * DEG),
            run_time=2
        )
        self.wait()

        # Go through each part
        faders = VGroup(movers[1:], movers_image[1:], labels, labels_image)
        for index in range(2):
            faders.generate_target()
            for mob in faders.target:
                for j, part in enumerate(mob):
                    if j == index:
                        part.set_opacity(1)
                    else:
                        part.set_opacity(0.25)

            rect = SurroundingRectangle(labels[index])
            rect.set_stroke(YELLOW, 2)
            self.play(
                MoveToTarget(faders),
                ShowCreation(rect),
            )
            self.play(
                TransformFromCopy(movers[index + 1], movers_image[index + 1], path_arc=-30 * DEG),
                TransformFromCopy(labels[index], labels_image[index], path_arc=-30 * DEG),
                rect.animate.surround(labels_image[index]).set_anim_args(path_arc=-30 * DEG),
                run_time=2,
            )
            self.play(FadeOut(rect))
            self.wait()
        self.play(faders.animate.set_fill(opacity=1))

    def get_plane(self):
        plane = NumberPlane((-2, 2), (-2, 2), faded_line_ratio=5)
        plane.set_height(7.5)
        plane.to_edge(LEFT, buff=MED_SMALL_BUFF)
        return plane

    def get_small_plane(self, plane):
        x_range = (-1, 1 - 1e-5)
        small_plane = NumberPlane(x_range, x_range, faded_line_ratio=5)
        small_plane.set_height(0.5 * plane.get_height())
        small_plane.move_to(plane)
        return small_plane

    def get_qubit_labels(self, plane):
        zero, one = bits = VGroup(Integer(0), Integer(1))
        zero_label, one_label = qubit_labels = VGroup(
            VGroup(Ket(bit), bit)
            for bit in bits
        )
        qubit_labels.scale(0.5)
        zero_label.next_to(plane.c2p(1, 0), DR, SMALL_BUFF)
        one_label.next_to(plane.c2p(0, 1), DR, SMALL_BUFF)
        return qubit_labels

    def get_vector(self, plane, x=1, y=0, fill_color=TEAL, thickness=6):
        return Arrow(
            plane.c2p(0, 0),
            plane.c2p(x, y),
            buff=0,
            thickness=thickness,
            fill_color=fill_color
        )

    def thumbnail_insert(self):
        # To be put above the "note x^2 + y^2" above
        self.remove(vector_label)
        self.remove(plane)
        small_plane = self.get_small_plane(plane)
        small_plane.set_z_index(-1)
        small_plane.axes.set_stroke(WHITE, 4)
        small_plane.background_lines.set_stroke(BLUE, 3)
        small_plane.faded_lines.set_stroke(BLUE, 2, 0.8)
        qubit_labels.set_fill(border_width=2)
        vector.set_color(YELLOW)
        theta_tracker.set_value(45 * DEG)
        self.add(small_plane)

        # Add glass
        glass = get_magnifying_glass()
        glass.set_height(5)
        glass[0].set_fill(BLACK)
        glass.shift(vector.get_center() - glass[0].get_center())
        one = KetGroup(Integer(1))
        one.set_height(1.5)
        one.move_to(glass[0])
        self.add(glass, one)

    def z_filp_insert(self):
        # For the clarification supplement
        angle = theta_tracker.get_value()
        self.play(theta_tracker.animate.set_value(angle + TAU), run_time=5)
        theta_tracker.set_value(angle)

        # Flips
        flipper = VGroup(vector, circle, zero_vect, one_vect)
        flipper.clear_updaters()
        self.play(
            Rotate(flipper, PI, axis=RIGHT, about_point=plane.c2p(0, 0), run_time=6, rate_func=there_and_back_with_pause)
        )


class ShowAFewFlips(Qubit):
    def construct(self):
        # Set up
        title = Text("Quantum Gates", font_size=60)
        title.to_edge(UP)

        plane = self.get_plane()
        plane.center()
        small_plane = self.get_small_plane(plane)
        qubit_labels = self.get_qubit_labels(plane)
        vector = self.get_vector(plane)

        self.add(title)
        self.add(small_plane, qubit_labels, vector)

        # Show H, Z and X gates
        lines = DashedLine(2 * LEFT, 2 * RIGHT).replicate(3)
        for line, angle in zip(lines, [0, PI / 8, PI / 4]):
            line.rotate(angle)
        lines.set_stroke(YELLOW, 2)

        gate_labels = VGroup(Text(c) for c in "ZHX")
        gate_labels.next_to(plane.c2p(1, 1), DR)

        for i in [1, 0, 2, 1, 2, 1, 0, 1]:
            self.play(LaggedStart(
                AnimationGroup(
                    FadeIn(lines[i]),
                    FadeIn(gate_labels[i])
                ),
                Rotate(vector, PI, axis=lines[i].get_vector(), about_point=ORIGIN),
                lag_ratio=0.5
            ))
            self.play(
                FadeOut(lines[i]),
                FadeOut(gate_labels[i]),
            )


class ExponentiallyGrowingState(InteractiveScene):
    def construct(self):
        # Initialize vector
        label = TexText(R"State of a\\\\ 1 qubit computer")
        n_label = label.make_number_changeable("1", edge_to_fix=RIGHT)
        n_label.set_color(YELLOW)
        label.move_to(3.5 * LEFT)
        vect = self.get_state_vector(1)
        vect.move_to(1.5 * RIGHT)

        brace = self.get_brace_group(vect, 1)
        brace.set_opacity(0)

        self.add(label)
        self.add(vect)

        # Grow the vector
        for n in range(2, 9):
            new_vect = self.get_state_vector(n)
            new_vect.move_to(vect)
            new_brace = self.get_brace_group(new_vect, n)

            n_label.set_value(n)
            self.play(
                ReplacementTransform(vect[0], new_vect[0]),
                FadeTransform(vect[1], new_vect[1]),
                FadeTransformPieces(vect[2], new_vect[2]),
                FadeTransform(brace[0], new_brace[0]),
                FadeTransform(brace[1], new_brace[1]),
            )
            self.wait()

            vect = new_vect
            brace = new_brace

        # Change to 100
        frame = self.frame

        new_brace_label = Tex(R"2^{100}")
        new_brace_label.set_color(YELLOW)
        new_brace_label.next_to(brace[0], RIGHT).shift(SMALL_BUFF * UR)

        top_eq = Tex(R"2^{100} = " + "{:,}".format(2**100))
        top_eq.next_to(vect, UP, MED_LARGE_BUFF).set_x(0)

        self.play(
            FadeTransform(brace[1], new_brace_label),
            ChangeDecimalToValue(n_label, 100),
            frame.animate.set_height(9, about_edge=DOWN),
            FadeIn(top_eq),
            FadeTransform(vect, self.get_state_vector(n + 2).move_to(vect, RIGHT)),
        )
        self.wait()

    def get_state_vector(self, n):
        # Actualy function
        values = normalize(np.random.uniform(-1, 1, 2**n))
        array = DecimalMatrix(
            values.reshape((2**n, 1)),
            decimal_config=dict(include_sign=True)
        )
        array.set_max_height(7)

        bit_strings = VGroup(
            BitString(k, length=n)
            for k in range(2**n)
        )
        bit_strings.set_color(GREY)
        for bits, entry in zip(bit_strings, array.get_entries()):
            bits.set_max_height(entry.get_height())
            bits.next_to(array, LEFT, buff=0.2)
            bits.match_y(entry)

        return VGroup(bit_strings, array.get_brackets(), array.get_entries())

    def get_brace_group(self, vect, n):
        brace = Brace(vect, RIGHT, buff=0.25)
        label = brace.get_tex(Rf"2^{{{n}}} = {2**n}")
        label.shift(SMALL_BUFF * UR)
        return VGroup(brace, label)


class InvisibleStateValues(InteractiveScene):
    def construct(self):
        # Test
        frame = self.frame
        n = 5
        values = np.random.uniform(-1, 1, 2**n)
        vect = DecimalMatrix(values.reshape(-1, 1), decimal_config=dict(include_sign=True))
        vect.set_height(FRAME_HEIGHT - 1)
        indices = VGroup(BitString(k, 5) for k in range(2**n))
        for index, elem in zip(indices, vect.elements):
            index.match_height(elem)
            index.set_color(GREY_C)
            index.next_to(vect, LEFT, SMALL_BUFF)
            index.match_y(elem)

        rects = VGroup(SurroundingRectangle(elem, buff=0.05) for elem in vect.elements)
        rects.set_stroke(YELLOW, 1)

        frame.set_height(2, about_edge=UP)
        self.add(vect, indices)
        self.play(
            LaggedStartMap(ShowCreation, rects, lag_ratio=0.25),
            frame.animate.to_default_state(),
            run_time=5,
        )

        rects.target = rects.generate_target()
        rects.target.set_stroke(WHITE, 1)
        rects.target.set_fill(GREY_D, 1)
        self.play(MoveToTarget(rects, lag_ratio=0.1, run_time=3))
        self.wait()

        # Shrink down
        group = Group(indices, vect, Point(), rects)
        self.play(group.animate.set_height(4).move_to(4 * RIGHT), run_time=2)
        self.wait()

        # Revealed value
        value = KetGroup(BitString(13, 5))
        value.move_to(group)

        self.clear()
        self.add(value)
        self.wait()


class ThreeDSample(InteractiveScene):
    def construct(self):
        # Set up axes
        frame = self.frame
        x_range = y_range = z_range = (-2, 2)
        axes = ThreeDAxes(x_range, y_range, z_range, axis_config=dict(tick_size=0.05))
        plane = NumberPlane(x_range, y_range, faded_line_ratio=5)
        plane.axes.set_opacity(0)
        plane.background_lines.set_stroke(BLUE, 1, 0.5)
        plane.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        rot_vel_tracker = ValueTracker(DEG)

        frame.reorient(-31, 71, 0, (0.22, 0.17, 0.13), 2.88)
        frame.add_updater(lambda m, dt: m.increment_theta(dt * rot_vel_tracker.get_value()))
        self.add(axes, plane, Point())

        # Add vector
        vector = Vector(normalize([1, -1, 2]), thickness=2, fill_color=TEAL)
        vector.always.set_perpendicular_to_camera(self.frame)
        vector.set_z_index(2)

        coord_array = DecimalMatrix(
            np.zeros((3, 1)),
            decimal_config=dict(include_sign=True)
        )
        coord_array.set_height(1.25)
        coord_array.to_corner(UL)
        coord_array.fix_in_frame()

        def update_coord_array(coord_array):
            for elem, coord in zip(coord_array.elements, vector.get_end()):
                elem.set_value(coord)
            coord_array.set_fill(GREY_B, border_width=1)

        coord_array.add_updater(update_coord_array)

        self.play(
            GrowArrow(vector),
            VFadeIn(coord_array),
        )
        self.play(Rotate(vector, TAU, axis=OUT, about_point=ORIGIN, run_time=6))
        self.wait(5)

        # Show 0, 1, and 2 directions
        symbols = VGroup(KetGroup(Integer(n)) for n in range(3))
        symbols.scale(0.5)
        symbols.set_backstroke(BLACK, 5)
        symbols.rotate(90 * DEG, RIGHT)
        directions = [UP + 0.5 * OUT, LEFT, LEFT]

        for symbol, direction, trg_coords in zip(symbols, directions, np.identity(3)):
            symbol.next_to(0.5 * trg_coords, direction, SMALL_BUFF)
            self.play(
                self.set_vect_anim(vector, trg_coords),
                FadeIn(symbol, 0.25 * OUT)
            )
            self.play(symbol.animate.scale(0.5).next_to(trg_coords, direction, SMALL_BUFF))
            self.wait(0.5)
        self.wait(2)

        # Highlight a secret key
        key_icon = get_key_icon()
        key_icon.rotate(90 * DEG, RIGHT)
        key_icon.match_depth(symbols[2])
        key_icon.next_to(symbols[2], LEFT, buff=0.05)

        self.play(
            FadeIn(key_icon, 0.25 * LEFT),
            symbols[2].animate.set_fill(YELLOW),
        )
        self.wait(3)

        # Go to the balanced state
        coord_rects = VGroup(
            SurroundingRectangle(elem)
            for elem in coord_array.elements
        )
        coord_rects.set_stroke(TEAL, 3)
        coord_rects.fix_in_frame()

        balanced_state = normalize([1, 1, 1])
        balance_name = Text("balanced state", font_size=16)
        balance_ket = KetGroup(Text("b", font_size=16), buff=0.035)
        for mob in [balance_name, balance_ket]:
            mob.rotate(90 * DEG, RIGHT)
            mob.next_to(balanced_state, OUT + RIGHT, buff=0.025)

        self.play(
            self.set_vect_anim(vector, balanced_state, run_time=4),
            FadeIn(balance_name, lag_ratio=0.1, time_span=(3, 4)),
        )
        self.play(LaggedStartMap(ShowCreation, coord_rects, lag_ratio=0.25))
        self.play(LaggedStartMap(FadeOut, coord_rects, lag_ratio=0.25))
        self.wait()
        self.play(
            FadeTransformPieces(balance_name, balance_ket[1]),
            Write(balance_ket[0])
        )
        self.play(rot_vel_tracker.animate.set_value(-DEG), run_time=3)
        self.wait(7)

        # Show the goal
        z_vect = vector.copy()
        z_vect.set_fill(YELLOW)

        tail = TracingTail(z_vect.get_end, stroke_color=YELLOW, time_traced=3)
        self.add(tail)
        self.wait(2)
        self.play(FadeIn(z_vect))
        self.play(
            self.set_vect_anim(z_vect, OUT, run_time=2)
        )
        self.wait(3)
        self.remove(tail)

        # Show 2d slice
        v_slice = NumberPlane(x_range, y_range, faded_line_ratio=5)
        v_slice.rotate(90 * DEG, RIGHT)
        v_slice.rotate(45 * DEG, OUT)
        v_slice.axes.set_stroke(WHITE, 1, 0.5)
        v_slice.background_lines.set_stroke(BLUE, 1, 1)
        v_slice.faded_lines.set_stroke(BLUE, 0.5, 0.25)

        b_vect_ghost = vector.copy()
        b_vect_ghost.set_fill(opacity=0.5)
        tail = TracingTail(vector.get_end, stroke_color=TEAL, time_traced=8)
        symbols[2].set_z_index(1)

        self.add(tail)
        self.play(
            frame.animate.reorient(43, 79, 0, (-0.26, 0.37, 0.15), 4.97),
            rot_vel_tracker.animate.set_value(-2 * DEG),
            FadeIn(v_slice),
            plane.animate.fade(0.75),
            axes.animate.set_stroke(opacity=0.5),
            run_time=3
        )
        self.add(b_vect_ghost)
        self.play(
            Rotate(
                vector,
                TAU,
                axis=np.cross(vector.get_end(), OUT),
                run_time=8,
                about_point=ORIGIN,
            )
        )
        self.play(
            frame.animate.reorient(-1, 79, 0, (-0.13, 0.55, 1.08), 7.67),
            coord_array.animate.set_x(0),
            run_time=2,
        )
        tail.clear_updaters()
        self.play(FadeOut(tail))
        self.wait(20)

        # Show xy line
        xy_line = v_slice.x_axis.copy()
        xy_line.set_stroke(WHITE, 3, 1)
        self.play(
            frame.animate.reorient(67, 77, 0, (-0.57, 0.12, 0.86), 6.62),
            run_time=2
        )
        self.wait(4)
        self.play(
            GrowFromCenter(xy_line),
            self.set_vect_anim(vector, normalize(UR)),
        )
        self.wait(10)
        self.play(self.set_vect_anim(vector, balanced_state))
        self.wait(20)

    def set_vect_anim(self, vector, trg_coords, run_time=1, **kwargs):
        return Rotate(
            vector,
            angle_between_vectors(vector.get_end(), trg_coords),
            axis=np.cross(vector.get_end(), trg_coords),
            about_point=ORIGIN,
            run_time=run_time,
            **kwargs
        )

    def flip_along_key_axis(self):
        # To be inserted after highlighting the key state above
        sphere = Sphere(radius=vector.get_length())
        mesh = SurfaceMesh(sphere, resolution=(51, 101))
        mesh.set_stroke(WHITE, 2, 0.1)

        key_vect = vector.copy()
        key_vect.clear_updaters()
        key_vect.set_fill(YELLOW, 0.5)

        self.add(key_vect)
        self.play(
            self.set_vect_anim(vector, normalize([1, 1, 1])),
            FadeIn(mesh)
        )
        vector.clear_updaters()
        for _ in range(4):
            self.play(
                Group(mesh, vector, key_vect).animate.stretch(-1, 2, about_point=ORIGIN),
                run_time=2
            )
            self.wait()


class GroversAlgorithm(InteractiveScene):
    def construct(self):
        # Set up plane
        x_range = y_range = (-1, 1 - 1e-6)
        plane = NumberPlane(x_range, y_range, faded_line_ratio=5)
        plane.set_height(6)
        plane.background_lines.set_stroke(BLUE, 1, 1)
        plane.faded_lines.set_stroke(BLUE, 1, 0.25)

        self.add(plane)

        # Add key and balance directions
        key_vect = Vector(plane.c2p(0, 1), thickness=5, fill_color=YELLOW)
        b_vect = key_vect.copy().rotate(- np.arccos(1 / math.sqrt(3)), about_point=ORIGIN)
        b_vect.set_fill(TEAL)

        key_label, b_label = labels = VGroup(
            KetGroup(Tex(char))
            for char in "kb"
        )
        labels.set_submobject_colors_by_gradient(YELLOW, TEAL)
        labels.set_backstroke(BLACK, 3)
        for label in labels:
            label[1].shift(0.02 * UR)
        key_label.next_to(key_vect.get_end(), UP, SMALL_BUFF)
        b_label.next_to(b_vect.get_end(), UR, SMALL_BUFF)
        key_icon = get_key_icon()
        key_icon.set_height(0.75 * key_label.get_height())
        key_icon.next_to(key_label, LEFT, SMALL_BUFF)

        self.add(key_vect, b_vect)
        self.add(key_label, b_label, key_icon)
        self.wait()

        # Highlight key
        self.play(GrowArrow(key_vect))
        self.play(FlashAround(key_label, time_width=1.5, run_time=2))

        # Label the x-direction
        x_vect = Vector(plane.c2p(1, 0), thickness=5)
        x_vect.set_fill(WHITE)
        x_label_example, x_label_general = x_labels = VGroup(
            Tex(R"\\frac{1}{\\sqrt{2}}\\big(|0\\rangle + |1\\rangle \\big)", font_size=36),
            Tex(R"\\frac{1}{\\sqrt{N - 1}} \\sum_{n \\ne k} |n \\rangle", font_size=24),
        )
        for label in x_labels:
            label.next_to(x_vect.get_end(), RIGHT)

        x_label_general.shift(SMALL_BUFF * DL)

        self.play(
            TransformFromCopy(key_vect, x_vect, path_arc=-90 * DEG),
            FadeIn(x_label_general, time_span=(0.5, 1.5)),
        )
        self.wait()

        x_label_general.save_state()
        self.play(
            FadeIn(x_label_example, DOWN),
            x_label_general.animate.fade(0.5).shift(1.25 * DOWN),
        )
        self.wait()

        # Show component of b in the direction of key
        rhs = Tex(R"= \\frac{1}{\\sqrt{3}}\\big(|0\\rangle + |1\\rangle + |2\\rangle \\big)", font_size=36)
        rhs.next_to(b_label, RIGHT)
        rhs.set_color(TEAL)

        b_vect_proj = Vector(plane.c2p(0, plane.p2c(b_vect.get_end())[1]), thickness=5)
        b_vect_proj.set_fill(TEAL_E, 1)
        dashed_line = DashedLine(b_vect.get_end(), b_vect_proj.get_end())

        self.play(FadeIn(rhs, lag_ratio=0.1))
        self.wait()
        self.play(
            TransformFromCopy(b_vect, b_vect_proj),
            ShowCreation(dashed_line),
        )
        self.wait()
        self.play(
            FlashAround(rhs[R"|2\\rangle"], time_width=1.5, run_time=3),
            rhs[R"|2\\rangle"].animate.set_color(YELLOW),
        )
        self.wait()

        # Set up N
        N_eq = Tex(R"N = 3")
        dim = N_eq.make_number_changeable("3", edge_to_fix=LEFT)
        N_eq.next_to(plane, UR, LARGE_BUFF).shift_onto_screen()

        self.play(
            LaggedStart(
                FadeOut(rhs),
                Restore(x_label_general),
                FadeOut(x_label_example, 0.5 * UP),
            ),
            Write(N_eq)
        )

        # Increase N
        N_tracker = ValueTracker(3)
        get_N = N_tracker.get_value

        def update_b_vects(vects):
            N = int(get_N())
            x = math.sqrt(1 - 1.0 / N)
            y = 1 / math.sqrt(N)
            vects[0].put_start_and_end_on(plane.c2p(0, 0), plane.c2p(x, y))
            vects[1].put_start_and_end_on(plane.c2p(0, 0), plane.c2p(0, y))

        self.play(
            N_tracker.animate.set_value(100).set_anim_args(rate_func=rush_into),
            UpdateFromFunc(dim, lambda m: m.set_value(int(get_N()))),
            UpdateFromFunc(VGroup(b_vect, b_vect_proj), update_b_vects),
            UpdateFromFunc(dashed_line, lambda m: m.become(DashedLine(b_vect.get_end(), b_vect_proj.get_end()))),
            UpdateFromFunc(b_label, lambda m: m.next_to(b_vect.get_end(), UR, SMALL_BUFF)),
            run_time=12
        )
        self.wait()

        # Reference the angle
        alpha = math.acos(1 / math.sqrt(N_tracker.get_value()))
        arc = Arc(90 * DEG, -alpha, radius=0.5)
        alpha_label = Tex(R"\\alpha")
        alpha_label.next_to(arc.pfp(0.5), UR, SMALL_BUFF)
        lt_90 = Tex(R"< 90^\\circ")
        lt_90.next_to(alpha_label, RIGHT).shift(0.05 * UL)

        self.play(
            FadeOut(b_vect_proj),
            FadeOut(dashed_line),
            ShowCreation(arc),
            FadeIn(alpha_label),
        )
        self.play(Write(lt_90))
        self.wait()
        self.play(FadeIn(VGroup(b_vect_proj, dashed_line), run_time=3, rate_func=there_and_back_with_pause, remover=True))
        self.wait()

        # Show the dot product
        frame = self.frame

        b_comp_tex = R"1 / \\sqrt{N}"
        b_array = TexMatrix(np.array([
            *2 * [b_comp_tex],
            R"\\vdots",
            *3 * [b_comp_tex],
            R"\\vdots",
            *2 * [b_comp_tex],
        ]).reshape(-1, 1))
        k_array = TexMatrix(np.array([
            *2 * ["0"],
            R"\\vdots",
            "0", "1", "0",
            R"\\vdots",
            *2 * ["0"],
        ]).reshape(-1, 1))
        arrays = VGroup(k_array, b_array)

        for array in arrays:
            array.set_height(6)

        arrays.arrange(LEFT, buff=MED_LARGE_BUFF)
        arrays.next_to(plane, RIGHT, buff=2.5).to_edge(DOWN, buff=MED_LARGE_BUFF)
        dot = Tex(R"\\cdot", font_size=72)
        dot.move_to(midpoint(b_array.get_right(), k_array.get_left()))

        k_array_label, b_array_label = array_labels = labels.copy()

        for label, arr in zip(array_labels, arrays):
            arr.set_fill(interpolate_color(label.get_fill_color(), WHITE, 0.2), 1)
            label.next_to(arr, UP, MED_LARGE_BUFF)

        self.play(
            GrowFromPoint(b_array, b_label.get_center()),
            TransformFromCopy(b_label, b_array_label),
            N_eq.animate.to_edge(UP, MED_SMALL_BUFF).set_x(2),
            frame.animate.set_x(4),
            FadeOut(x_label_general),
            FadeOut(x_vect),
            run_time=2
        )
        self.play(
            GrowFromPoint(k_array, key_label.get_center()),
            TransformFromCopy(key_label, k_array_label),
            Write(dot),
            run_time=2
        )
        self.wait()

        # Evaluate the dot product
        elem_rects = VGroup(
            SurroundingRectangle(VGroup(*elems), buff=0.05).set_width(2.5, stretch=True)
            for elems in zip(b_array.elements, k_array.elements)
        )
        for rect in elem_rects:
            rect.set_stroke(WHITE, 2)
            rect.align_to(elem_rects[0], RIGHT)

        equals = Tex(R"=").next_to(arrays, RIGHT)
        rhs = Tex(R"1 / \\sqrt{N}")
        rhs.next_to(equals, RIGHT)

        self.play(Write(equals))
        self.play(
            LaggedStartMap(VFadeInThenOut, elem_rects, lag_ratio=0.2, run_time=5),
            FadeIn(rhs, time_span=(1.75, 2.25)),
        )
        self.wait()

        # Show cosine expression
        cos_expr = Tex(R"\\cos(\\alpha) = 1 / \\sqrt{N}", font_size=36)
        cos_expr.next_to(alpha_label, UP, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            FadeOut(lt_90),
            Write(cos_expr[R"\\cos("]),
            Write(cos_expr[R") ="]),
            TransformFromCopy(alpha_label, cos_expr[R"\\alpha"][0]),
            TransformFromCopy(rhs, cos_expr[R"1 / \\sqrt{N}"][0]),
            lag_ratio=0.1,
        ))
        self.wait()

        # Show sine of smaller angle
        target_thickness = 3

        for vect in [b_vect, key_vect]:
            vect.target = Arrow(ORIGIN, vect.get_end(), thickness=target_thickness, buff=0)
            vect.target.match_style(vect)

        theta = 90 * DEG - alpha
        theta_arc = Arc(0, theta, radius=2.0)
        theta_label = Tex(R"\\theta", font_size=24)
        theta_label.next_to(theta_arc, RIGHT, buff=0.1)
        alpha_label.set_backstroke(BLACK, 5)
        theta_label.set_backstroke(BLACK, 5)

        sin_expr = Tex(R"\\sin(\\theta) = 1 / \\sqrt{N}", font_size=36)
        sin_expr.move_to(cos_expr).set_y(-0.5)

        theta_approx = Tex(R"\\theta \\approx 1 / \\sqrt{N}", font_size=36)
        theta_approx.next_to(sin_expr, DOWN, aligned_edge=RIGHT)

        cos_group = VGroup(arc, alpha_label, cos_expr)

        self.play(
            MoveToTarget(b_vect),
            MoveToTarget(key_vect),
            LaggedStart(
                TransformFromCopy(arc, theta_arc),
                TransformFromCopy(alpha_label, theta_label),
                TransformFromCopy(cos_expr, sin_expr),
                lag_ratio=0.25
            ),
            cos_group.animate.fade(0.5)
        )
        self.wait()
        self.play(
            TransformMatchingTex(
                sin_expr.copy(),
                theta_approx,
                matched_keys=[R"1 / \\sqrt{N}"],
                key_map={"=": R"\\approx"}
            )
        )
        self.wait()

        # Put it in the corner
        self.play(
            LaggedStartMap(FadeOut, VGroup(array_labels, arrays, dot, equals, rhs, cos_group, sin_expr), lag_ratio=0.25),
            frame.animate.set_x(-2),
            N_eq.animate.set_x(-2),
            theta_approx.animate.to_corner(UR, buff=MED_SMALL_BUFF).shift(2 * LEFT),
            run_time=2
        )
        self.wait()

        # Add vector components
        vector = b_vect.copy()

        initial_coords = 0.1 * np.ones(11)
        vect_coords = DecimalMatrix(initial_coords.reshape(-1, 1), num_decimal_places=3, decimal_config=dict(include_sign=True))
        vect_coords.set_height(6)
        vect_coords.next_to(frame.get_left(), RIGHT, MED_LARGE_BUFF)
        mid_index = len(initial_coords) // 2
        dot_indices = [mid_index - 2, mid_index + 2]
        arr_dots = VGroup()
        for index in dot_indices:
            element = vect_coords.elements[index]
            dots = Tex(R"\\vdots")
            dots.move_to(element)
            element.become(dots)
            arr_dots.add(element)

        vect_coords.set_fill(GREY_B)

        def update_vect_coords(vect_coords):
            x, y = plane.p2c(vector.get_end())
            x /= math.sqrt(99)
            for n, elem in enumerate(vect_coords.elements):
                if n in dot_indices:
                    continue
                elif n == mid_index:
                    elem.set_value(y)
                else:
                    elem.set_value(x)

        vect_coords.add_updater(update_vect_coords)

        key_icon2 = get_key_icon(height=0.25)
        key_icon2.next_to(vect_coords, LEFT, SMALL_BUFF)

        self.add(vect_coords)
        self.add(key_icon2)

        # Add bars
        min_bar_width = 0.125
        bar_height = 0.25

        dec_elements = VGroup(
            elem for n, elem in enumerate(vect_coords.elements) if n not in dot_indices
        )
        bars = Rectangle(min_bar_width, bar_height).replicate(len(dec_elements))
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(BLUE, GREEN)
        bars.set_stroke(WHITE, 0.5)

        for bar, elem in zip(bars, dec_elements):
            bar.next_to(vect_coords, RIGHT)
            bar.match_y(elem)
            bar.elem = elem

        def update_bars(bars):
            x, y = plane.p2c(vector.get_end())
            x /= math.sqrt(99)
            mid_index = len(bars) // 2
            for n, bar in enumerate(bars):
                if n == mid_index:
                    prob = 1 - 99 * (x**2)
                else:
                    prob = x**2
                width = min_bar_width * np.sqrt(prob / 0.01)
                bar.set_width(width, about_edge=LEFT, stretch=True)

        self.add(bars)

        # Show the flips
        circle = Circle(radius=0.5 * plane.get_width())
        circle.set_stroke(WHITE, 1)
        b_vect.set_fill(opacity=0.5)

        frame.set_field_of_view(20 * DEG)

        diag_line = DashedLine(-b_vect.get_end(), b_vect.get_end())
        h_line = DashedLine(plane.get_left(), plane.get_right())
        VGroup(h_line, diag_line).set_stroke(WHITE, 2)

        flip_line = h_line.copy()

        flipper = VGroup(circle, vector)

        vect_ghosts = VGroup()

        def right_filp(run_time=2):
            self.play(
                Rotate(flipper, PI, RIGHT, about_point=ORIGIN),
                vect_ghosts.animate.set_fill(opacity=0.25),
                run_time=run_time
            )

        def diag_flip(run_time=2, draw_time=0.5):
            self.play(ShowCreation(diag_line, run_time=draw_time))
            self.play(
                flipper.animate.flip(axis=diag_line.get_vector(), about_point=ORIGIN),
                vect_ghosts.animate.set_fill(opacity=0.25),
                UpdateFromFunc(bars, update_bars),
                run_time=run_time
            )
            self.play(FadeOut(diag_line, run_time=2 * draw_time))

        self.add(vect_ghosts, vector)
        self.play(FadeIn(circle))
        for n in range(4):
            vect_ghosts.add(vector.copy())
            right_filp()
            self.wait()
            vect_ghosts.add(vector.copy())
            diag_flip(draw_time=(0.5 if n < 2 else 1 / 30))
            self.wait()

        vect_ghosts.add(vect.copy()).set_fill(opacity=0.25)

        # Show vertical component
        if False:  # For an insertion
            # Show vert component
            self.add(diag_line)

            x, y = plane.p2c(vector.get_end())
            v_part = Arrow(plane.get_origin(), plane.c2p(0, y), thickness=4, buff=0, max_width_to_length_ratio=0.25)
            v_part.set_fill(GREEN)
            h_line = DashedLine(vector.get_end(), v_part.get_end())
            brace = Brace(v_part, LEFT, buff=0.05)

            self.play(
                FadeOut(diag_line),
                TransformFromCopy(vector, v_part),
                ShowCreation(h_line),
            )
            self.wait()
            self.play(GrowFromCenter(brace))
            self.wait()

        # Show steps of 2 * theta
        bars.add_updater(update_bars)

        arcs = VGroup(
            Arc(n * theta, 2 * theta, radius=circle.get_radius())
            for n in range(1, len(vect_ghosts), 2)
        )
        for arc, color in zip(arcs, it.cycle([BLUE, RED])):
            arc.set_stroke(color, 3)

        arc_labels = VGroup(
            Tex(R"2\\theta", font_size=24).next_to(arc.pfp(0.5), normalize(arc.pfp(0.5)), SMALL_BUFF)
            for arc in arcs
        )

        self.play(
            Rotate(
                vector,
                -angle_between_vectors(vector.get_vector(), b_vect.get_vector()),
                about_point=ORIGIN
            ),
            FadeOut(b_label)
        )
        self.wait()
        right_filp()
        diag_flip(draw_time=0.25)
        self.wait()
        self.play(
            TransformFromCopy(vect_ghosts[0], vector, path_arc=theta),
            ShowCreation(arcs[0])
        )
        self.play(TransformFromCopy(theta_label, arc_labels[0]))
        self.wait()
        for arc, label in zip(arcs[1:], arc_labels[1:]):
            self.play(
                Rotate(vector, 2 * theta, about_point=ORIGIN),
                ShowCreation(arc),
                FadeIn(label, shift=0.5 * (arc.get_end() - arc.get_start()))
            )
            self.wait()

        # Show full angle
        quarter_arc = Arc(0, 90 * DEG)
        ninety_label = Tex(R"90^\\circ")
        pi_halves_label = Tex(R"\\pi / 2")
        for label in [ninety_label, pi_halves_label]:
            label.set_backstroke(BLACK, 5)
            label.next_to(quarter_arc.pfp(0.5), UR, buff=0.05)

        self.play(
            ShowCreation(quarter_arc),
            Write(ninety_label),
        )
        self.wait()
        self.play(FadeTransform(ninety_label, pi_halves_label))
        self.wait()

        # Calculate n steps
        lhs = Text("# Repetitions")
        lhs.next_to(plane, RIGHT, buff=1.0).set_y(2)
        rhs_terms = VGroup(
            Tex(R"\\approx {\\pi / 2 \\over 2 \\theta}"),
            Tex(R"= {\\pi \\over 4} \\cdot {1 \\over \\theta}"),
            Tex(R"= {\\pi \\over 4} \\sqrt{N}"),
        )
        rhs_terms.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        rhs_terms.shift(lhs.get_right() + 0.2 * RIGHT + 0.05 * UP - rhs_terms[0].get_left())

        self.play(
            frame.animate.set_x(3),
            FadeOut(vect_coords),
            FadeOut(bars),
            FadeOut(key_icon2),
            Write(lhs),
            Write(rhs_terms[0]),
        )
        self.wait()
        self.play(LaggedStartMap(FadeIn, rhs_terms[1:], shift=0.5 * DOWN, lag_ratio=0.5))
        self.wait()

        # Collapse
        self.play(LaggedStart(
            Rotate(
                vector,
                -angle_between_vectors(vector.get_vector(), b_vect.get_vector()),
                about_point=ORIGIN
            ),
            FadeOut(arcs),
            FadeOut(arc_labels),
            FadeOut(vect_ghosts),
            FadeOut(b_vect),
            FadeOut(theta_label),
            FadeOut(theta_arc),
            frame.animate.set_y(0.5),
            N_eq.animate.shift(0.5 * UP),
            theta_approx.animate.align_to(plane, RIGHT),
            VGroup(lhs, rhs_terms).animate.shift(1.5 *  UP)
        ))

        # Increment to 2^20
        new_theta = math.asin(2**(-10))
        dim.set_value(100)
        step_count = Tex(R"\\frac{\\pi}{4}\\sqrt{2^{20}} = 804.248...")
        step_count.next_to(rhs_terms, DOWN, LARGE_BUFF)
        step_count.to_edge(RIGHT).shift(frame.get_x() * RIGHT)

        eq_two_twenty = Tex(R"=2^{20}")
        eq_two_twenty.next_to(N_eq["="], DOWN, aligned_edge=LEFT)

        self.play(
            ChangeDecimalToValue(dim, int(2**20)),
            Rotate(vector, new_theta - theta, about_point=ORIGIN),
            FadeIn(eq_two_twenty, time_span=(0, 1)),
            run_time=3
        )
        self.wait()
        self.play(
            TransformFromCopy(rhs_terms[-1][1:6], step_count[:5]),
            TransformFromCopy(eq_two_twenty[1:], step_count["2^{20}"][0]),
            run_time=2
        )
        self.play(Write(step_count["= 804.248..."][0]))
        self.wait()

        # Change vector
        step_tracker = ValueTracker(0)
        radius = 0.5 * plane.get_width()

        step_label = Tex(R"\\#\\text{Reps} = 0", font_size=36)
        step_label.next_to(plane.c2p(-0.6, 0), UP, SMALL_BUFF)
        step_count = step_label.make_number_changeable(0, edge_to_fix=UL)
        step_count.f_always.set_value(lambda: 0.5 * step_tracker.get_value())

        shadows = VectorizedPoint().replicate(300)

        def update_vector(vector):
            steps = int(step_tracker.get_value())
            if steps % 2 == 0:
                angle = new_theta * (steps + 1)
            else:
                angle = -new_theta * steps
            point = rotate_vector(radius * RIGHT, angle)
            vector.put_start_and_end_on(ORIGIN, point)
            shadows.remove(shadows[0])
            shadows.add(vector.copy())
            shadows.clear_updaters()
            for n, shadow in enumerate(shadows[::-1]):
                shadow.set_fill(opacity=0.75 / (n + 1))

        self.play(FadeIn(step_label))
        self.add(shadows)
        self.add(vect_coords, bars)
        self.play(
            step_tracker.animate.set_value(2 * 804),
            UpdateFromFunc(vector, update_vector),
            frame.animate.scale(1.4, about_edge=RIGHT).set_anim_args(time_span=(0, 8)),
            run_time=20,
            rate_func=linear,
        )
        self.play(FadeOut(shadows, lag_ratio=0.1, run_time=1))
        self.wait()

    def key_flip_insertion(self):
        # Test
        self.clear()
        key_ghost = key_vect.copy().set_fill(opacity=0.5)
        self.add(key_ghost)
        self.play(
            key_vect.animate.flip(axis=RIGHT, about_edge=DOWN),
            run_time=2
        )
        self.wait()

    def thumbnail_insertion(self):
        # Test
        plane.background_lines.set_stroke(BLUE, 8, 1)
        plane.faded_lines.set_stroke(BLUE, 5, 0.25)
        circle.set_stroke(WHITE, 3)
        self.remove(N_eq)
        self.remove(theta_label)
        self.remove(theta_arc)
        self.remove(theta_approx)


class TwoFlipsEqualsRotation(InteractiveScene):
    def construct(self):
        # Set up planes
        plane = NumberPlane((-2, 2), (-2, 2))
        plane.background_lines.set_stroke(BLUE, 1, 1)
        plane.faded_lines.set_stroke(BLUE, 1, 0.25)
        plane.axes.set_opacity(0.5)

        randy = Randolph(mode="pondering", height=3)

        ghost_plane = plane.copy()
        ghost_plane.fade(0.5)
        self.add(ghost_plane)

        plane2 = plane.copy()
        ghost_plane2 = ghost_plane.copy()
        randy2 = randy.copy()
        VGroup(plane2, ghost_plane2, randy2).next_to(plane, RIGHT, buff=3)

        self.add(plane, randy)

        # Show flips
        theta = 15 * DEG
        h_flip_line = DashedLine(plane.get_left(), plane.get_right())
        diag_flip_line = h_flip_line.copy().rotate(theta)
        flip_lines = VGroup(h_flip_line, diag_flip_line)
        flip_lines.set_stroke(WHITE, 4)

        for line in flip_lines:
            self.play(ShowCreation(line, run_time=0.5))
            self.play(
                randy.animate.flip(axis=line.get_vector(), about_point=plane.get_origin()),
                plane.animate.flip(axis=line.get_vector(), about_point=plane.get_origin()),
            )
        self.wait()

        # Show rotation
        arcs = VGroup(
            Arc(0, 90 * DEG, radius=1),
            Arc(180 * DEG, 90 * DEG, radius=1),
        )
        for arc, vect in zip(arcs, [UR, DL]):
            arc.set_stroke(WHITE, 5)
            arc.move_to(plane2.get_corner(vect))
            arc.add_tip()

        self.play(
            self.frame.animate.move_to(midpoint(plane.get_center(), plane2.get_center())),
            FadeIn(ghost_plane2),
            FadeIn(plane2),
            FadeIn(randy2),
        )
        self.play(
            Rotate(Group(plane2, Point(), randy2), 2 * theta, about_point=plane2.get_center()),
            *map(FadeIn, arcs)
        )
        self.wait()

        # Show angle
        arc = Arc(0, theta, radius=1.0)
        theta_label = Tex(R"\\theta", font_size=30)
        theta_label.set_backstroke(BLACK, 3)
        theta_label.next_to(arc, RIGHT, SMALL_BUFF)
        theta_label.shift(0.025 * UP)

        rot_arc = Arc(0, 2 * theta, arc_center=plane2.get_center())
        two_theta_label = Tex(R"2 \\theta")
        two_theta_label.set_backstroke(BLACK, 3)
        two_theta_label.next_to(rot_arc, RIGHT, SMALL_BUFF)

        self.play(ShowCreation(arc), Write(theta_label), run_time=1)
        self.play(
            TransformFromCopy(arc, rot_arc),
            TransformFromCopy(theta_label, two_theta_label),
        )
        self.wait()


class ComplexComponents(InteractiveScene):
    def construct(self):
        # Set up vectors
        state = normalize(np.random.uniform(-1, 1, 4))
        dec_vect = DecimalMatrix(state.reshape(-1, 1), decimal_config=dict(include_sign=True))
        dec_vect.move_to(3 * LEFT)

        indices = VGroup(BitString(n, 2) for n in range(4))
        for index, elem in zip(indices, dec_vect.elements):
            index.set_color(GREY_B)
            index.match_height(elem)
            index.next_to(dec_vect, LEFT)
            index.match_y(elem)

        x_vect, z_vect = var_vects = [
            TexMatrix(np.array([f"{char}_{n}" for n in range(4)]).reshape(-1, 1))
            for char in "xz"
        ]
        for vect in var_vects:
            vect.match_height(dec_vect)
            vect.move_to(dec_vect, LEFT)

        self.add(dec_vect, indices)
        self.wait()

        # Real number lines and complex planes
        number_lines = VGroup(
            NumberLine(
                (-1, 1, 0.1),
                big_tick_spacing=1,
                width=3,
                tick_size=0.05,
                stroke_color=GREY_C,
            )
            for n in range(4)
        )
        number_lines.arrange(DOWN, buff=1.0)
        number_lines.next_to(dec_vect, RIGHT, buff=1.0)

        complex_planes = VGroup(
            ComplexPlane((-1, 1), (-1, 1)).replace(number_line, 0)
            for number_line in number_lines
        )

        for number_line in number_lines:
            number_line.add_numbers([-1, 0, 1], font_size=24, buff=0.15)
        for plane in complex_planes:
            plane.add_coordinate_labels(font_size=16)

        complex_planes.generate_target()
        complex_planes.target.arrange(DOWN, buff=1.0)
        complex_planes.target.set_height(7)
        complex_planes.target.next_to(x_vect, RIGHT, buff=2.0)
        complex_planes.set_opacity(0)

        R_labels, C_labels = [
            VGroup(
                Tex(Rf"\\mathds{{{char}}}", font_size=36).next_to(mob.get_right(), RIGHT)
                for mob in group
            )
            for char, group in zip("RC", [number_lines, complex_planes.target])
        ]

        # Set up dots
        state_tracker = ComplexValueTracker(state)

        dots = GlowDot(color=YELLOW).replicate(len(state))

        def update_dots(dots):
            for dot, value, plane in zip(dots, state_tracker.get_value(), complex_planes):
                dot.move_to(plane.n2p(value))

        dots.add_updater(update_dots)

        dot_lines = VGroup(Line() for dot in dots)
        dot_lines.set_stroke(YELLOW, 2, 0.5)

        def update_dot_lines(lines):
            for line, x, dot in zip(lines, x_vect.elements, dots):
                line.put_start_and_end_on(
                    x.get_right() + 0.05 * RIGHT,
                    dot.get_center()
                )

        update_dot_lines(dot_lines)

        self.play(
            LaggedStartMap(FadeIn, number_lines, lag_ratio=0.25),
            LaggedStartMap(FadeIn, R_labels, lag_ratio=0.25),
            LaggedStart(
                # (FadeInFromPoint(dot, elem.get_center())
                (FadeTransform(elem, dot)
                for elem, dot in zip(dec_vect.elements, dots)),
                lag_ratio=0.05,
                group_type=Group
            ),
            LaggedStart(
                (FadeTransform(elem.copy(), x)
                for elem, x in zip(dec_vect.elements, x_vect.elements)),
                lag_ratio=0.05,
            ),
            LaggedStartMap(ShowCreation, dot_lines),
            ReplacementTransform(dec_vect.get_brackets(), x_vect.get_brackets(), time_span=(1, 2)),
            run_time=2
        )
        self.add(dots, dot_lines)
        dot_lines.add_updater(update_dot_lines)

        self.play(
            state_tracker.animate.set_value(normalize(np.random.uniform(-1, 1, 4))),
            run_time=4
        )

        # Transition to complex plane
        number_lines.generate_target()
        for line, plane in zip(number_lines.target, complex_planes.target):
            line.replace(plane, 0)
            line.set_opacity(0)

        self.play(
            MoveToTarget(complex_planes),
            MoveToTarget(number_lines, remover=True),
            *(FadeTransform(R, C) for R, C in zip(R_labels, C_labels)),
            ReplacementTransform(x_vect, z_vect)
        )
        self.wait()

        for n in range(4):
            self.random_state_change(state_tracker, pump_index=(0 if n == 3 else None))

        # Zoom in on one value
        frame = self.frame

        plane = complex_planes[0]
        c_dot = GlowDot(color=YELLOW, radius=0.05)
        c_dot.move_to(dots[0])

        self.play(
            frame.animate.set_height(1.6).move_to(plane),
            FadeIn(c_dot),
            FadeOut(dots),
            FadeOut(dot_lines),
            run_time=2
        )

        # Show magnitude
        c_line = Line(plane.c2p(0), c_dot.get_center())
        c_line.set_stroke(YELLOW, 2)
        big_brace_width = 3
        brace = Brace(Line(LEFT, RIGHT).set_width(big_brace_width), DOWN)
        brace.scale(c_line.get_length() / big_brace_width, about_point=ORIGIN)
        brace.rotate(c_line.get_angle() + PI, about_point=ORIGIN)
        brace.shift(c_line.get_center())

        mag_label = Tex(R"|z_0|", font_size=12)
        mag_label.next_to(brace.get_center(), DR, buff=0.05),

        self.play(
            ShowCreation(c_line),
            GrowFromCenter(brace),
            FadeIn(mag_label),
        )

        # Show phase
        arc = always_redraw(lambda: Arc(
            0, c_line.get_angle() % TAU, radius=0.1, arc_center=plane.c2p(0),
            stroke_color=MAROON_B
        ))
        arc.update()
        arc.suspend_updating()

        phi_label = Tex(R"\\varphi", font_size=12)
        phi_label.set_color(MAROON_B)
        phi_label.add_updater(
            lambda m: m.move_to(arc.pfp(0.5)).shift(0.5 * (arc.pfp(0.5) - plane.c2p(0)))
        )

        self.play(ShowCreation(arc), FadeIn(phi_label))
        self.wait()

        # Show prob
        prob_eq = Tex(R"\\text{Prob} = |z_0|^2", font_size=12)
        prob_eq.move_to(mag_label)
        prob_eq.align_to(C_labels[0], RIGHT).shift(0.2 * RIGHT)

        self.play(
            TransformFromCopy(mag_label, prob_eq["|z_0|"][0], path_arc=45 * DEG),
            Write(prob_eq[R"\\text{Prob} ="][0]),
            Write(prob_eq["2"][0]),
        )
        self.wait()

        # Change phase
        c_dot.add_updater(lambda m: m.move_to(c_line.get_end()))
        ghost_line = c_line.copy().set_stroke(opacity=0.5)

        self.add(ghost_line)
        arc.resume_updating()
        phi = c_line.get_angle() % TAU
        for angle in [-(1 - 1e-5) * phi, PI, phi - PI]:
            self.play(
                Rotate(c_line, angle, about_point=plane.n2p(0), run_time=6),
            )
            self.wait()

        # Zoom back out
        fader = Group(prob_eq, mag_label, brace, c_line, c_dot, arc, phi_label, ghost_line)
        fader.clear_updaters()
        dot_lines.set_stroke(opacity=0.25)
        self.play(
            frame.animate.to_default_state(),
            FadeOut(fader),
            FadeIn(dots),
            FadeIn(dot_lines),
            run_time=4
        )

        # More random motion
        self.random_state_change(state_tracker)

        dots.suspend_updating()
        self.play(
            *(
                Rotate(
                    dot,
                    random.random() * 2 * TAU,
                    about_point=plane.c2p(0),
                    run_time=10,
                    rate_func=there_and_back,
                )
                for dot, plane in zip(dots, complex_planes)
            )
        )
        dots.resume_updating()

        for n in range(4):
            self.random_state_change(state_tracker)

    def random_state_change(self, state_tracker, pump_index=None, run_time=2):
        rands = np.random.uniform(-1, 1, 4)
        if pump_index is not None:
            rands[pump_index] = 4
        mags = normalize(rands)
        phases = np.exp(TAU * np.random.random(4) * 1j)
        new_state = mags * phases
        self.play(state_tracker.animate.set_value(new_state), run_time=2)`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2025.blocks_and_grover.qc_supplements module within the 3b1b videos codebase.",
      5: "BitString extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      8: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      19: "Ket extends Tex. Custom Tex subclass for specialized LaTeX rendering with additional formatting.",
      27: "KetGroup extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      33: "RandomSampling extends Animation. Custom Animation subclass. Override interpolate(alpha) to define how the animation progresses from 0 to 1.",
      45: "interpolate(alpha) is called each frame with alpha from 0→1. Defines the custom animation's behavior over its duration.",
      53: "ContrstClassicalAndQuantum extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      54: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      81: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      82: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      83: "FadeOut transitions a mobject from opaque to transparent.",
      84: "FadeOut transitions a mobject from opaque to transparent.",
      85: "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
      86: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      88: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      99: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      103: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      104: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      105: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      112: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      114: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      121: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      122: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      123: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      124: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      126: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      127: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      128: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      132: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      134: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      135: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      136: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      137: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      139: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      140: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      141: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      142: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      144: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      147: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      159: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      160: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      161: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      163: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      164: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      165: "Transform smoothly morphs one mobject into another by interpolating their points.",
      166: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      167: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      171: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      179: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      184: "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
      186: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      188: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      197: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      198: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      199: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      422: "AmbientStateVector extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      425: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      482: "Class RotatingStateVector inherits from AmbientStateVector.",
      486: "Class FlipsToCertainDirection inherits from AmbientStateVector.",
      487: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      527: "DisectAQuantumComputer extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      528: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1471: "Class Qubit inherits from DisectAQuantumComputer.",
      1472: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1892: "Class ShowAFewFlips inherits from Qubit.",
      1893: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1931: "ExponentiallyGrowingState extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1932: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2013: "InvisibleStateValues extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2014: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2059: "ThreeDSample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2060: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2266: "GroversAlgorithm extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2267: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2863: "TwoFlipsEqualsRotation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2864: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2941: "ComplexComponents extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2942: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();