(function() {
  const files = window.MANIM_DATA.files;

  files["_2024/transformers/almost_orthogonal.py"] = {
    description: "Explores the geometry of high-dimensional embeddings: why random vectors in high dimensions are nearly orthogonal, and implications for transformer representations.",
    code: `import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

# List of vectors in some dimension, with many
# more vectors than there are dimensions
num_vectors = 10000
vector_len = 100
big_matrix = torch.randn(num_vectors, vector_len)
big_matrix /= big_matrix.norm(p=2, dim=1, keepdim=True)  # Normalize
big_matrix.requires_grad_(True)

# Set up an Optimization loop to create nearly-perpendicular vectors
optimizer = torch.optim.Adam([big_matrix], lr=0.01)
num_steps = 250

losses = []

dot_diff_cutoff = 0.01
big_id = torch.eye(num_vectors, num_vectors)

for step_num in tqdm(range(num_steps)):
    optimizer.zero_grad()

    dot_products = big_matrix @ big_matrix.T
    # Punish deviation from orthogonal
    diff = dot_products - big_id
    loss = (diff.abs() - dot_diff_cutoff).relu().sum()

    # Extra incentive to keep rows normalized
    loss += num_vectors * diff.diag().pow(2).sum()

    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# Loss curve
plt.plot(losses)
plt.grid(1)
plt.show()

# Angle distribution
dot_products = big_matrix @ big_matrix.T
norms = torch.sqrt(torch.diag(dot_products))
normed_dot_products = dot_products / torch.outer(norms, norms)
angles_degrees = torch.rad2deg(torch.acos(normed_dot_products.detach()))
# Use this to ignore self-orthogonality.
self_orthogonality_mask = ~(torch.eye(num_vectors, num_vectors).bool())
plt.hist(angles_degrees[self_orthogonality_mask].numpy().ravel(), bins=1000, range=(80, 100))
plt.grid(1)
plt.show()`,
    annotations: {}
  };

  files["_2024/transformers/attention.py"] = {
    description: "Core attention mechanism scenes for the transformer explainer series. Demonstrates how attention patterns work: queries, keys, values, softmax scoring, and multi-head attention. Includes visual demonstrations of word embeddings interacting through attention weights.",
    code: `from __future__ import annotations

from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import break_into_words
from _2024.transformers.embedding import break_into_tokens
from _2024.transformers.embedding import get_piece_rectangles


class AttentionPatterns(InteractiveScene):
    def construct(self):
        # Add sentence
        phrase = " a fluffy blue creature roamed the verdant forest"
        phrase_mob = Text(phrase)
        phrase_mob.move_to(2 * UP)
        words = list(filter(lambda s: s.strip(), phrase.split(" ")))
        word2mob: Dict[str, VMobject] = {
            word: phrase_mob[" " + word][0]
            for word in words
        }
        word_mobs = VGroup(*word2mob.values())

        self.play(
            LaggedStartMap(FadeIn, word_mobs, shift=0.5 * UP, lag_ratio=0.25)
        )
        self.wait()

        # Create word rects
        word2rect: Dict[str, VMobject] = dict()
        for word in words:
            rect = SurroundingRectangle(word2mob[word])
            rect.set_height(phrase_mob.get_height() + SMALL_BUFF, stretch=True)
            rect.match_y(phrase_mob)
            rect.set_stroke(GREY, 2)
            rect.set_fill(GREY, 0.2)
            word2rect[word] = rect

        # Adjectives updating noun
        adjs = ["fluffy", "blue", "verdant"]
        nouns = ["creature", "forest"]
        others = ["a", "roamed", "the"]
        adj_mobs, noun_mobs, other_mobs = [
            VGroup(word2mob[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects, noun_rects, other_rects = [
            VGroup(word2rect[substr] for substr in group)
            for group in [adjs, nouns, others]
        ]
        adj_rects.set_submobject_colors_by_gradient(BLUE_C, BLUE_D, GREEN)
        noun_rects.set_color(GREY_BROWN).set_stroke(width=3)
        kw = dict()
        adj_arrows = VGroup(
            Arrow(
                adj_mobs[i].get_top(), noun_mobs[j].get_top(),
                path_arc=-150 * DEGREES, buff=0.1, stroke_color=GREY_B
            )
            for i, j in [(0, 0), (1, 0), (2, 1)]
        )

        self.play(
            LaggedStartMap(DrawBorderThenFill, adj_rects),
            Animation(adj_mobs),
        )
        self.wait()
        self.play(
            LaggedStartMap(DrawBorderThenFill, noun_rects),
            Animation(noun_mobs),
            LaggedStartMap(ShowCreation, adj_arrows, lag_ratio=0.2, run_time=1.5),
        )
        kw = dict(time_width=2, max_stroke_width=10, lag_ratio=0.2, path_arc=150 * DEGREES)
        self.play(
            ContextAnimation(noun_mobs[0], adj_mobs[:2], strengths=[1, 1], **kw),
            ContextAnimation(noun_mobs[1], adj_mobs[2:], strengths=[1], **kw),
        )
        self.wait()

        # Show embeddings
        all_rects = VGroup(*adj_rects, *noun_rects, *other_rects)
        all_rects.sort(lambda p: p[0])
        embeddings = VGroup(
            NumericEmbedding(length=10).set_width(0.5).next_to(rect, DOWN, buff=1.5)
            for rect in all_rects
        )
        emb_arrows = VGroup(
            Arrow(all_rects[0].get_bottom(), embeddings[0].get_top()).match_x(rect)
            for rect in all_rects
        )
        for index, vect in [(5, LEFT), (6, RIGHT)]:
            embeddings[index].shift(0.1 * vect)
            emb_arrows[index].shift(0.05 * vect)

        self.play(
            FadeIn(other_rects),
            Animation(word_mobs),
            LaggedStartMap(GrowArrow, emb_arrows),
            LaggedStartMap(FadeIn, embeddings, shift=0.5 * DOWN),
            FadeOut(adj_arrows)
        )
        self.wait()

        # Mention dimension of embedding
        frame = self.frame
        brace = Brace(embeddings[0], LEFT, buff=SMALL_BUFF)
        dim_value = Integer(12288)
        dim_value.next_to(brace, LEFT)
        dim_value.set_color(YELLOW)

        self.play(
            GrowFromCenter(brace),
            CountInFrom(dim_value, 0),
            frame.animate.move_to(LEFT)
        )
        self.wait()

        # Ingest meaning and and position
        images = Group(
            ImageMobject(f"Dalle3_{word}").set_height(1.1).next_to(word2rect[word], UP)
            for word in ["fluffy", "blue", "creature", "verdant", "forest"]
        )
        image_vects = VGroup(embeddings[i] for i in [1, 2, 3, 6, 7])

        self.play(
            LaggedStartMap(FadeIn, images, scale=2, lag_ratio=0.05)
        )
        self.play(
            LaggedStart(
                (self.bake_mobject_into_vector_entries(image, vect, group_type=Group)
                for image, vect in zip(images, image_vects)),
                group_type=Group,
                lag_ratio=0.2,
                run_time=4,
                remover=True
            ),
        )
        self.wait()
        self.add(embeddings, images)

        # Show positions
        pos_labels = VGroup(
            Integer(n, font_size=36).next_to(rect, DOWN, buff=0.1)
            for n, rect in enumerate(all_rects, start=1)
        )
        pos_labels.set_color(TEAL)

        self.play(
            LaggedStart(
                (arrow.animate.scale(0.7, about_edge=DOWN)
                for arrow in emb_arrows),
                lag_ratio=0.1,
            ),
            LaggedStartMap(FadeIn, pos_labels, shift=0.25 * DOWN, lag_ratio=0.1)
        )
        self.play(
            LaggedStart(
                (self.bake_mobject_into_vector_entries(pos, vect)
                for pos, vect in zip(pos_labels, embeddings)),
                lag_ratio=0.2,
                run_time=4,
                remover=True
            ),
        )
        self.wait()

        # Collapse vectors
        template = Tex(R"\\vec{\\textbf{E}}_{0}")
        template[0].scale(1.5, about_edge=DOWN)
        dec = template.make_number_changeable(0)
        emb_syms = VGroup()
        for n, rect in enumerate(all_rects, start=1):
            dec.set_value(n)
            sym = template.copy()
            sym.next_to(rect, DOWN, buff=0.75)
            sym.set_color(GREY_A)
            emb_syms.add(sym)
        for subgroup in [emb_syms[:4], emb_syms[4:]]:
            subgroup.arrange_to_fit_width(subgroup.get_width())

        emb_arrows.target = emb_arrows.generate_target()

        for rect, arrow, sym in zip(all_rects, emb_arrows.target, emb_syms):
            x_min = rect.get_x(LEFT)
            x_max = rect.get_x(RIGHT)
            low_point = sym[0].get_top()
            if x_min < low_point[0] < x_max:
                top_point = np.array([low_point[0], rect.get_y(DOWN), 0])
            else:
                top_point = rect.get_bottom()
            arrow.become(Arrow(top_point, low_point, buff=SMALL_BUFF))

        all_brackets = VGroup(emb.get_brackets() for emb in embeddings)
        for brackets in all_brackets:
            brackets.target = brackets.generate_target()
            brackets.target.stretch(0, 1, about_edge=UP)
            brackets.target.set_fill(opacity=0)

        ghost_syms = emb_syms.copy()
        ghost_syms.set_opacity(0)

        self.play(
            frame.animate.set_x(0).set_anim_args(run_time=2),
            LaggedStart(
                (AnimationGroup(
                    LaggedStart(
                        (FadeTransform(entry, sym)
                        for entry in embedding.get_columns()[0]),
                        lag_ratio=0.01,
                        group_type=Group
                    ),
                    MoveToTarget(brackets),
                    group_type=Group,
                )
                for sym, embedding, brackets in zip(ghost_syms, embeddings, all_brackets)),
                group_type=Group
            ),
            LaggedStartMap(FadeIn, emb_syms, shift=UP),
            brace.animate.stretch(0.25, 1, about_edge=UP).set_opacity(0),
            FadeOut(dim_value, 0.25 * UP),
            MoveToTarget(emb_arrows, lag_ratio=0.1, run_time=2),
            LaggedStartMap(FadeOut, pos_labels, shift=UP),
        )
        emb_arrows.refresh_bounding_box(recurse_down=True)  # Why?
        self.clear()
        self.add(emb_arrows, all_rects, word_mobs, images, emb_syms)
        self.wait()

        # Preview desired updates
        emb_sym_primes = VGroup(
            sym.copy().add(Tex("'").move_to(sym.get_corner(UR) + 0.05 * DL))
            for sym in emb_syms
        )
        emb_sym_primes.shift(2 * DOWN)
        emb_sym_primes.set_color(TEAL)

        full_connections = VGroup()
        for i, sym1 in enumerate(emb_syms, start=1):
            for j, sym2 in enumerate(emb_sym_primes, start=1):
                line = Line(sym1.get_bottom(), sym2.get_top(), buff=SMALL_BUFF)
                line.set_stroke(GREY_B, width=random.random()**2, opacity=random.random()**0.25)
                if (i, j) in [(2, 4), (3, 4), (4, 4), (7, 8), (8, 8)]:
                    line.set_stroke(WHITE, width=2 + random.random(), opacity=1)
                full_connections.add(line)

        blue_fluff = ImageMobject("BlueFluff")
        verdant_forest = ImageMobject("VerdantForest")
        for n, image in [(3, blue_fluff), (7, verdant_forest)]:
            image.match_height(images)
            image.scale(1.2)
            image.next_to(emb_sym_primes[n], DOWN, buff=MED_SMALL_BUFF)

        self.play(
            ShowCreation(full_connections, lag_ratio=0.01, run_time=2),
            LaggedStart(
                (TransformFromCopy(sym1, sym2)
                for sym1, sym2 in zip(emb_syms, emb_sym_primes)),
                lag_ratio=0.05,
            ),
        )
        self.wait()
        self.play(LaggedStart(
            LaggedStart(
                (FadeTransform(im.copy(), blue_fluff, remover=True)
                for im in images[:3]),
                lag_ratio=0.02,
                group_type=Group
            ),
            LaggedStart(
                (FadeTransform(im.copy(), verdant_forest, remover=True)
                for im in images[3:]),
                lag_ratio=0.02,
                group_type=Group
            ),
            lag_ratio=0.5,
            run_time=2
        ))
        self.add(blue_fluff, verdant_forest)
        self.wait()

        # Show black box that matrix multiples can be added to
        in_arrows = VGroup(
            Vector(0.25 * DOWN, max_width_to_length_ratio=12.0).next_to(sym, DOWN, SMALL_BUFF)
            for sym in emb_syms
        )
        box = Rectangle(15.0, 3.0)
        box.set_fill(GREY_E, 1)
        box.set_stroke(WHITE, 1)
        box.next_to(in_arrows, DOWN, SMALL_BUFF)
        out_arrows = in_arrows.copy()
        out_arrows.next_to(box, DOWN)

        self.play(
            FadeIn(box, 0.25 * DOWN),
            LaggedStartMap(FadeIn, in_arrows, shift=0.25 * DOWN, lag_ratio=0.025),
            LaggedStartMap(FadeIn, out_arrows, shift=0.25 * DOWN, lag_ratio=0.025),
            FadeOut(full_connections),
            emb_sym_primes.animate.next_to(out_arrows, DOWN, SMALL_BUFF),
            MaintainPositionRelativeTo(blue_fluff, emb_sym_primes),
            MaintainPositionRelativeTo(verdant_forest, emb_sym_primes),
            frame.animate.set_height(10).move_to(4 * UP, UP),
        )
        self.wait()

        # Clear the board
        self.play(
            frame.animate.set_height(8).move_to(2 * UP).set_anim_args(run_time=1.5),
            LaggedStartMap(FadeOut, Group(
                *images, in_arrows, box, out_arrows, emb_sym_primes,
                blue_fluff, verdant_forest,
            ), lag_ratio=0.1)
        )

        # Ask questions
        word_groups = VGroup(VGroup(*pair) for pair in zip(all_rects, word_mobs))
        for group in word_groups:
            group.save_state()
        q_bubble = SpeechBubble("Any adjectives\\nin front of me?")
        q_bubble.move_tip_to(word2rect["creature"].get_top())

        a_bubbles = SpeechBubble("I am!", direction=RIGHT).replicate(2)
        a_bubbles[1].flip()
        a_bubbles[0].move_tip_to(word2rect["fluffy"].get_top())
        a_bubbles[1].move_tip_to(word2rect["blue"].get_top())

        self.play(
            FadeIn(q_bubble),
            word_groups[:3].animate.fade(0.75),
            word_groups[4:].animate.fade(0.75),
        )
        self.wait()
        self.play(LaggedStart(
            Restore(word_groups[1]),
            Restore(word_groups[2]),
            *map(Write, a_bubbles),
            lag_ratio=0.5
        ))
        self.wait()

        # Associate questions with vectors
        a_bubbles.save_state()
        q_arrows = VGroup(
            Vector(0.75 * DOWN).next_to(sym, DOWN, SMALL_BUFF)
            for sym in emb_syms
        )
        q_vects = VGroup(
            NumericEmbedding(length=7).set_height(2).next_to(arrow, DOWN)
            for arrow in q_arrows
        )
        question = q_bubble.content


        index = words.index("creature")
        q_vect = q_vects[index]
        q_arrow = q_arrows[index]
        self.play(LaggedStart(
            FadeOut(q_bubble.body, DOWN),
            question.animate.scale(0.75).next_to(q_vect, RIGHT),
            FadeIn(q_vect, DOWN),
            GrowArrow(q_arrow),
            frame.animate.move_to(ORIGIN),
            a_bubbles.animate.fade(0.5),
        ))
        self.play(
            self.bake_mobject_into_vector_entries(question, q_vect)
        )
        self.wait()

        # Label query vector
        brace = Brace(q_vect, LEFT, SMALL_BUFF)
        query_word = Text("Query")
        query_word.set_color(YELLOW)
        query_word.next_to(brace, LEFT, SMALL_BUFF)
        dim_text = Text("128-dimensional", font_size=36)
        dim_text.set_color(YELLOW)
        dim_text.next_to(brace, LEFT, SMALL_BUFF)
        dim_text.set_y(query_word.get_y(DOWN))

        self.play(
            GrowFromCenter(brace),
            FadeIn(query_word, 0.25 * LEFT),
        )
        self.wait()
        self.play(
            query_word.animate.next_to(dim_text, UP, SMALL_BUFF),
            FadeIn(dim_text, 0.1 * DOWN),
        )
        self.wait()

        # Show individual matrix product
        e_vect = NumericEmbedding(length=12)
        e_vect.match_width(q_vect)
        e_vect.next_to(q_vect, DR, buff=1.5)
        matrix = WeightMatrix(shape=(7, 12))
        matrix.match_height(q_vect)
        matrix.next_to(e_vect, LEFT)
        e_label_copy = emb_syms[index].copy()
        e_label_copy.next_to(e_vect, UP)
        q_vect.save_state()
        ghost_q_vect = NumericEmbedding(length=7).match_height(q_vect)
        ghost_q_vect.get_columns().set_opacity(0)
        ghost_q_vect.get_brackets().space_out_submobjects(1.75)
        ghost_q_vect.next_to(e_vect, RIGHT, buff=0.7)

        mat_brace = Brace(matrix, UP)
        mat_label = Tex("W_Q")
        mat_label.next_to(mat_brace, UP, SMALL_BUFF)
        mat_label.set_color(YELLOW)

        self.play(
            frame.animate.set_height(11).move_to(all_rects, UP).shift(0.35 * UP),
            FadeOut(a_bubbles),
            FadeInFromPoint(e_vect, emb_syms[index].get_center()),
            FadeInFromPoint(matrix, q_arrow.get_center()),
            TransformFromCopy(emb_syms[index], e_label_copy),
            FadeOut(q_vect),
            TransformFromCopy(q_vect, ghost_q_vect),
            MaintainPositionRelativeTo(question, q_vect),
        )
        self.play(
            GrowFromCenter(mat_brace),
            FadeIn(mat_label, 0.1 * UP),
        )
        self.remove(ghost_q_vect)
        eq, rhs = show_matrix_vector_product(self, matrix, e_vect)

        new_q_vect = rhs.deepcopy()
        new_q_vect.move_to(q_vect, LEFT)

        self.play(
            TransformFromCopy(rhs, new_q_vect, path_arc=PI / 2),
            question.animate.next_to(new_q_vect, RIGHT)
        )
        self.wait()

        # Collapse query vector
        q_sym_template = Tex(R"\\vec{\\textbf{Q}}_0", font_size=48)
        q_sym_template[0].scale(1.5, about_edge=DOWN)
        q_sym_template.set_color(YELLOW)
        subscript = q_sym_template.make_number_changeable(0)
        q_syms = VGroup()
        for n, arrow in enumerate(q_arrows, start=1):
            subscript.set_value(n)
            sym = q_sym_template.copy()
            sym.next_to(arrow, DOWN, SMALL_BUFF)
            q_syms.add(sym)

        mat_label2 = mat_label.copy()

        q_sym = q_syms[index]
        low_q_sym = q_sym.copy()
        low_q_sym.next_to(rhs, UP)

        self.play(LaggedStart(
            LaggedStart(
                (FadeTransform(entry, q_sym, remover=True)
                for entry in new_q_vect.get_columns()[0]),
                lag_ratio=0.01,
                group_type=Group,
            ),
            new_q_vect.get_brackets().animate.stretch(0, 1, about_edge=UP).set_opacity(0),
            FadeOutToPoint(query_word, q_sym.get_center()),
            FadeOutToPoint(dim_text, q_sym.get_center()),
            FadeOut(brace),
            question.animate.next_to(q_sym, DOWN),
            FadeIn(low_q_sym, UP),
            lag_ratio=0.1,
        ))
        self.remove(new_q_vect)
        self.add(q_sym)
        self.play(
            mat_label2.animate.scale(0.9).next_to(q_arrow, RIGHT, buff=0.15),
        )
        self.wait()

        # E to Q rects
        e_rects = VGroup(map(SurroundingRectangle, [emb_syms[index], e_vect]))
        q_rects = VGroup(map(SurroundingRectangle, [q_sym, rhs]))
        e_rects.set_stroke(TEAL, 3)
        q_rects.set_stroke(YELLOW, 3)
        self.play(ShowCreation(e_rects, lag_ratio=0.2))
        self.wait()
        self.play(Transform(e_rects, q_rects))
        self.wait()
        self.play(FadeOut(e_rects))

        # Add other query vectors
        remaining_q_arrows = VGroup(*q_arrows[:index], *q_arrows[index + 1:])
        remaining_q_syms = VGroup(*q_syms[:index], *q_syms[index + 1:])
        wq_syms = VGroup(
            Tex(R"W_Q", font_size=30).next_to(arrow, RIGHT, buff=0.1)
            for arrow in q_arrows
        )
        wq_syms.set_color(YELLOW)
        subscripts = VGroup(e_label_copy[-1], low_q_sym[-1][0])
        for subscript in subscripts:
            i_sym = Tex("i")
            i_sym.replace(subscript)
            i_sym.scale(0.75)
            i_sym.match_style(subscript)
            subscript.target = i_sym

        self.play(
            LaggedStartMap(GrowArrow, remaining_q_arrows),
            LaggedStartMap(FadeIn, remaining_q_syms, shift=0.1 * DOWN),
            ReplacementTransform(VGroup(mat_label2), wq_syms, lag_ratio=0.01, run_time=2),
            question.animate.shift(0.25 * DOWN),
            *map(Restore, word_groups),
            *map(MoveToTarget, subscripts),
        )
        self.wait()

        # Emphasize model weights
        self.play(
            LaggedStartMap(FlashAround, matrix.get_entries(), lag_ratio=1e-2),
            RandomizeMatrixEntries(matrix),
        )
        data_modifying_matrix(self, matrix, word_shape=(3, 8))
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                matrix, mat_brace, mat_label,
                e_vect, e_label_copy, eq, rhs,
                low_q_sym
            ), shift=0.2 * DR)
        )
        self.wait()

        # Move question
        noun_q_syms = VGroup(q_syms[words.index(word)] for word in ["creature", "forest"])

        self.play(
            question.animate.shift(0.25 * DOWN).match_x(noun_q_syms)
        )

        noun_q_lines = VGroup(
            Line(question.get_corner(v), sym.get_corner(-v) + 0.25 * v)
            for sym, v in zip(noun_q_syms, [UL, UR])
        )
        noun_q_lines.set_stroke(GREY, 1)
        self.play(ShowCreation(noun_q_lines, lag_ratio=0))
        self.wait()

        # Set up keys
        key_word_groups = word_groups.copy()
        key_word_groups.arrange(DOWN, buff=0.75, aligned_edge=RIGHT)
        key_word_groups.next_to(q_syms, DL, buff=LARGE_BUFF)
        key_word_groups.shift(3.0 * LEFT)
        key_emb_syms = emb_syms.copy()

        k_sym_template = Tex(R"\\vec{\\textbf{K}}_0", font_size=48)
        k_sym_template[0].scale(1.5, about_edge=DOWN)
        k_sym_template.set_color(TEAL)
        subscript = k_sym_template.make_number_changeable(0)

        k_syms = VGroup()
        key_emb_arrows = VGroup()
        wk_arrows = VGroup()
        wk_syms = VGroup()
        for group, emb_sym, n in zip(key_word_groups, key_emb_syms, it.count(1)):
            emb_arrow = Vector(0.5 * RIGHT)
            emb_arrow.next_to(group, RIGHT, SMALL_BUFF)
            emb_sym.next_to(emb_arrow, RIGHT, SMALL_BUFF)
            wk_arrow = Vector(0.75 * RIGHT)
            wk_arrow.next_to(emb_sym, RIGHT)
            wk_sym = Tex("W_k", font_size=30)
            wk_sym.set_fill(TEAL, border_width=1)
            wk_sym.next_to(wk_arrow, UP)
            subscript.set_value(n)
            k_sym = k_sym_template.copy()
            k_sym.next_to(wk_arrow, RIGHT, buff=MED_SMALL_BUFF)

            key_emb_arrows.add(emb_arrow)
            wk_arrows.add(wk_arrow)
            wk_syms.add(wk_sym)
            k_syms.add(k_sym)

        self.remove(question, noun_q_lines)
        self.play(
            frame.animate.move_to(2.5 * LEFT + 2.75 * DOWN),
            TransformFromCopy(word_groups, key_word_groups),
            TransformFromCopy(emb_arrows, key_emb_arrows),
            TransformFromCopy(emb_syms, key_emb_syms),
            run_time=2,
        )
        self.play(
            LaggedStartMap(GrowArrow, wk_arrows),
            LaggedStartMap(FadeIn, wk_syms, shift=0.1 * UP),
        )
        self.play(LaggedStart(
            (TransformFromCopy(e_sym, k_sym)
            for e_sym, k_sym in zip(key_emb_syms, k_syms)),
            lag_ratio=0.05,
        ))
        self.wait()

        # Isolate examples
        fade_rects = VGroup(
            BackgroundRectangle(VGroup(key_word_groups[0], wk_syms[0], k_syms[0])),
            BackgroundRectangle(VGroup(key_word_groups[3:], wk_syms[3:], k_syms[3:])),
            BackgroundRectangle(wq_syms[2]),
            BackgroundRectangle(VGroup(word_groups[:3], q_syms[:3])),
            BackgroundRectangle(VGroup(word_groups[4:], q_syms[4:])),
        )
        fade_rects.set_fill(BLACK, 0.75)
        fade_rects.set_stroke(BLACK, 3, 1)
        q_bubble = SpeechBubble("Any adjectives\\nin front of me?")
        q_bubble.flip(RIGHT)
        q_bubble.next_to(q_syms[3][-1], DOWN, SMALL_BUFF, LEFT)
        a_bubbles = SpeechBubble("I'm an adjective!\\nI'm there!").replicate(2)
        a_bubbles[0].pin_to(k_syms[1])
        a_bubbles[1].pin_to(k_syms[2])
        a_bubbles[1].flip(RIGHT, about_edge=DOWN)
        a_bubbles[1].shift(0.5 * DOWN)

        self.add(fade_rects, word_groups[3])
        self.play(FadeIn(fade_rects))
        self.play(FadeIn(q_bubble, lag_ratio=0.1))
        self.play(FadeIn(a_bubbles, lag_ratio=0.05))
        self.wait()

        # Show example key matrix
        matrix = WeightMatrix(shape=(7, 12))
        matrix.set_width(5)
        matrix.next_to(k_syms, UP, buff=2.0, aligned_edge=RIGHT)
        mat_rect = SurroundingRectangle(matrix, buff=MED_SMALL_BUFF)
        lil_rect = SurroundingRectangle(wk_syms[1])
        lines = VGroup(
            Line(lil_rect.get_corner(v + UP), mat_rect.get_corner(v + DOWN))
            for v in [LEFT, RIGHT]
        )
        VGroup(mat_rect, lil_rect, *lines).set_stroke(GREY_A, 1)

        self.play(ShowCreation(lil_rect))
        self.play(
            ShowCreation(lines, lag_ratio=0),
            TransformFromCopy(lil_rect, mat_rect),
            FadeInFromPoint(matrix, lil_rect.get_center()),
        )
        self.wait()
        data_modifying_matrix(self, matrix, word_shape=(3, 8))
        self.play(
            LaggedStartMap(FadeOut, VGroup(matrix, mat_rect, lines, lil_rect), run_time=1)
        )
        self.play(
            LaggedStartMap(FadeOut, VGroup(q_bubble, *a_bubbles), lag_ratio=0.25)
        )
        self.wait()

        # Draw grid
        emb_arrows.refresh_bounding_box(recurse_down=True)
        q_groups = VGroup(
            VGroup(group[i] for group in [
                emb_arrows, emb_syms, wq_syms, q_arrows, q_syms
            ])
            for i in range(len(emb_arrows))
        )
        q_groups.target = q_groups.generate_target()
        q_groups.target.arrange_to_fit_width(12, about_edge=LEFT)
        q_groups.target.shift(0.25 * DOWN)

        word_groups.target = word_groups.generate_target()
        for word_group, q_group in zip(word_groups.target, q_groups.target):
            word_group.scale(0.7)
            word_group.next_to(q_group[0], UP, SMALL_BUFF)

        h_lines = VGroup()
        v_buff = 0.5 * (key_word_groups[0].get_y(DOWN) - key_word_groups[1].get_y(UP))
        for kwg in key_word_groups:
            h_line = Line(LEFT, RIGHT).set_width(20)
            h_line.next_to(kwg, UP, buff=v_buff)
            h_line.align_to(key_word_groups, LEFT)
            h_lines.add(h_line)

        v_lines = VGroup()
        h_buff = 0.5
        for q_group in q_groups.target:
            v_line = Line(UP, DOWN).set_height(14)
            v_line.next_to(q_group, LEFT, buff=h_buff, aligned_edge=UP)
            v_lines.add(v_line)
        v_lines.add(v_lines[-1].copy().next_to(q_groups.target, RIGHT, 0.5, UP))

        grid_lines = VGroup(*h_lines, *v_lines)
        grid_lines.set_stroke(GREY_A, 1)

        self.play(
            frame.animate.set_height(15, about_edge=UP).set_x(-2).set_anim_args(run_time=3),
            MoveToTarget(q_groups),
            MoveToTarget(word_groups),
            ShowCreation(h_lines, lag_ratio=0.2),
            ShowCreation(v_lines, lag_ratio=0.2),
            FadeOut(fade_rects),
        )

        # Take all dot products
        dot_prods = VGroup()
        for k_sym in k_syms:
            for q_sym in q_syms:
                square_center = np.array([q_sym.get_x(), k_sym.get_y(), 0])
                dot = Tex(R".", font_size=72)
                dot.move_to(square_center)
                dot.set_fill(opacity=0)
                dot_prod = VGroup(k_sym.copy(), dot, q_sym.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=0.15)
                dot_prod.target.scale(0.65)
                dot_prod.target.move_to(square_center)
                dot_prod.target.set_fill(opacity=1)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.025, run_time=4)
        )
        self.wait()

        # Show grid of dots
        dots = VGroup(
            VGroup(Dot().match_x(q_sym).match_y(k_sym) for q_sym in q_syms)
            for k_sym in k_syms
        )
        for n, row in enumerate(dots, start=1):
            for k, dot in enumerate(row, start=1):
                dot.set_fill(GREY_C, 0.8)
                dot.set_width(random.random())
                dot.target = dot.generate_target()
                dot.target.set_width(0.1 + 0.2 * random.random())
                if (n, k) in [(2, 4), (3, 4), (7, 8)]:
                    dot.target.set_width(0.8 + 0.2 * random.random())
        flat_dots = VGroup(*it.chain(*dots))

        self.play(
            dot_prods.animate.set_fill(opacity=0.75),
            LaggedStartMap(GrowFromCenter, flat_dots)
        )
        self.wait()
        self.play(LaggedStartMap(MoveToTarget, flat_dots, lag_ratio=0.01))
        self.wait()

        # Resize to reflect true pattern
        k_groups = VGroup(
            VGroup(group[i] for group in [
                key_word_groups, key_emb_arrows,
                key_emb_syms, wk_syms, wk_arrows, k_syms
            ])
            for i in range(len(emb_arrows))
        )
        for q_group, word_group in zip(q_groups, word_groups):
            q_group.add_to_back(word_group)
        self.add(k_groups, q_groups, Point())

        k_fade_rects = VGroup(map(BackgroundRectangle, k_groups))
        q_fade_rects = VGroup(map(BackgroundRectangle, q_groups))
        for rect in (*k_fade_rects, *q_fade_rects):
            rect.scale(1.05)
            rect.set_fill(BLACK, 0.8)

        self.play(
            frame.animate.move_to([-4.33, -2.4, 0.0]).set_height(9.52),
            FadeIn(k_fade_rects[:1]),
            FadeIn(k_fade_rects[3:]),
            FadeIn(q_fade_rects[:3]),
            FadeIn(q_fade_rects[4:]),
            run_time=2
        )
        self.wait()

        k_rects = VGroup(map(SurroundingRectangle, k_groups[1:3]))
        k_rects.set_stroke(TEAL, 2)
        q_rects = VGroup(SurroundingRectangle(q_groups[3]))
        q_rects.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(k_rects, lag_ratio=0.5, run_time=2),
            LaggedStartMap(
                FlashAround, k_groups[1:3],
                color=TEAL,
                time_width=2,
                lag_ratio=0.25,
                run_time=3
            ),
        )
        self.wait()
        self.play(TransformFromCopy(k_rects, q_rects))
        self.wait()

        # Show numerical dot product
        high_dot_prods = VGroup(dot_prods[8 + 3], dot_prods[2 * 8 + 3])
        dots_to_grow = VGroup(dots[1][3], dots[2][3])
        numerical_dot_prods = VGroup(
            VGroup(
                DecimalNumber(
                    np.random.uniform(-100, 10),
                    include_sign=True,
                    font_size=42,
                    num_decimal_places=1,
                    edge_to_fix=ORIGIN,
                ).move_to(dot)
                for dot in row
            )
            for row in dots
        )
        for n, row in enumerate(numerical_dot_prods):
            row[n].set_value(5 * random.random())  # Add some self relevance
        flat_numerical_dot_prods = VGroup(*it.chain(*numerical_dot_prods))
        for ndp in flat_numerical_dot_prods:
            ndp.set_fill(interpolate_color(RED_E, GREY_C, random.random()))
        high_numerical_dot_prods = VGroup(
            numerical_dot_prods[1][3],
            numerical_dot_prods[2][3],
            numerical_dot_prods[6][7],
        )
        for hdp in high_numerical_dot_prods:
            hdp.set_value(92 + 2 * random.random())
            hdp.set_color(WHITE)
        low_numerical_dot_prod = numerical_dot_prods[5][3]
        low_numerical_dot_prod.set_value(-31.4)
        low_numerical_dot_prod.set_fill(RED_D)

        self.play(
            *(dtg.animate.scale(1.25) for dtg in dots_to_grow),
            *(CountInFrom(ndp, run_time=1) for ndp in high_numerical_dot_prods[:2]),
            *(VFadeIn(ndp) for ndp in high_numerical_dot_prods[:2]),
            *(FadeOut(dot_prod, run_time=0.5) for dot_prod in dot_prods),
        )
        self.wait()

        # Show "attends to"
        att_arrow = Arrow(k_rects.get_top(), q_rects.get_left(), path_arc=-90 * DEGREES)
        att_words = TexText("\`\`Attends to''", font_size=72)
        att_words.next_to(att_arrow.pfp(0.4), UL)

        self.play(
            ShowCreation(att_arrow),
            Write(att_words),
        )
        self.wait()
        self.play(FadeOut(att_words), FadeOut(att_arrow))

        # Contrast with "the" and "creature"
        self.play(
            frame.animate.move_to([-2.79, -3.66, 0.0]).set_height(12.29),
            *(k_rect.animate.surround(k_groups[5]) for k_rect in k_rects),
            FadeIn(k_fade_rects[1:3]),
            FadeOut(k_fade_rects[5]),
            run_time=2,
        )
        self.play(
            CountInFrom(low_numerical_dot_prod),
            VFadeIn(low_numerical_dot_prod),
            FadeOut(dots[5][3]),
        )
        self.wait()

        # Zoom out on full grid
        self.play(
            frame.animate.move_to([-1.5, -4.8, 0.0]).set_height(15).set_anim_args(run_time=3),
            LaggedStart(
                FadeOut(k_rects),
                FadeOut(q_rects),
                FadeOut(k_fade_rects[:5]),
                FadeOut(k_fade_rects[6:]),
                FadeOut(q_fade_rects[:3]),
                FadeOut(q_fade_rects[4:]),
                FadeOut(dots),
                LaggedStartMap(FadeIn, numerical_dot_prods),
                Animation(high_numerical_dot_prods.copy(), remover=True),
                Animation(low_numerical_dot_prod.copy(), remover=True),
            )
        )
        self.wait()

        # Focus on one column
        ndp_columns = VGroup(
            VGroup(row[i] for row in numerical_dot_prods)
            for i in range(len(numerical_dot_prods[0]))
        )
        col_rect = SurroundingRectangle(ndp_columns[3], buff=0.25)
        col_rect.set_stroke(YELLOW, 2)
        weight_words = Text("We want these to\\nact like weights", font_size=96)
        weight_words.set_backstroke(BLACK, 8)
        weight_words.next_to(col_rect, RIGHT, buff=MED_LARGE_BUFF)
        weight_words.match_y(h_lines[2])

        index = words.index("creature")
        self.play(
            ShowCreation(col_rect),
            grid_lines.animate.set_stroke(opacity=0.5),
            ndp_columns[:index].animate.set_opacity(0.35),
            ndp_columns[index + 1:].animate.set_opacity(0.35),
            FadeIn(weight_words, lag_ratio=0.1)
        )
        self.wait()

        # Show softmax of each columns
        self.set_floor_plane("xz")
        col_arrays = [np.array([num.get_value() for num in col]) for col in ndp_columns]
        softmax_arrays = list(map(softmax, col_arrays))
        softmax_cols = VGroup(
            VGroup(DecimalNumber(v) for v in softmax_array)
            for softmax_array in softmax_arrays
        )
        sm_arrows = VGroup()
        sm_labels = VGroup()
        sm_rects = VGroup()
        for sm_col, col in zip(softmax_cols, ndp_columns):
            for sm_val, val in zip(sm_col, col):
                sm_val.move_to(val)
            sm_col.save_state()
            sm_col.shift(6 * OUT)
            sm_rect = SurroundingRectangle(sm_col)
            sm_rect.match_style(col_rect)
            VGroup(sm_col, sm_rect).rotate(30 * DEGREES, DOWN)
            arrow = Arrow(col, sm_col.get_center() + SMALL_BUFF * RIGHT + IN)
            label = Text("softmax", font_size=72)
            label.set_backstroke(BLACK, 5)
            label.rotate(90 * DEGREES, DOWN)
            label.next_to(arrow, UP)
            sm_arrows.add(arrow)
            sm_labels.add(label)
            sm_rects.add(sm_rect)

        index = words.index("creature")
        self.play(
            frame.animate.reorient(-47, -7, 0, (-2.48, -5.84, -1.09), 20),
            GrowArrow(sm_arrows[index], time_span=(1, 2)),
            FadeIn(sm_labels[index], lag_ratio=0.1, time_span=(1, 2)),
            TransformFromCopy(ndp_columns[index], softmax_cols[index], time_span=(1.5, 3)),
            TransformFromCopy(col_rect, sm_rects[index], time_span=(1.5, 3)),
            FadeOut(weight_words),
            run_time=3
        )
        self.wait()

        remaining_indices = [*range(index), *range(index + 1, len(ndp_columns))]
        last_index = index
        for index in remaining_indices:
            self.play(
                ndp_columns[last_index].animate.set_opacity(0.35),
                ndp_columns[index].animate.set_opacity(1),
                col_rect.animate.move_to(ndp_columns[index]),
                softmax_cols[last_index].animate.set_opacity(0.25),
                *map(FadeOut, [sm_rects[last_index], sm_arrows[last_index], sm_labels[last_index]]),
            )
            self.play(
                GrowArrow(sm_arrows[index]),
                FadeIn(sm_labels[index], lag_ratio=0.1),
                TransformFromCopy(ndp_columns[index], softmax_cols[index]),
                TransformFromCopy(col_rect, sm_rects[index]),
            )
            last_index = index

        self.play(
            FadeOut(col_rect),
            *map(FadeOut, [sm_rects[last_index], sm_arrows[last_index], sm_labels[last_index]]),
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (-2.64, -4.8, 0.0), 14.54),
            LaggedStartMap(Restore, softmax_cols, lag_ratio=0.1),
            FadeOut(ndp_columns, time_span=(0, 1.5)),
            run_time=3,
        )
        self.wait()

        # Label attention pattern
        for n, row in enumerate(dots):
            if n not in [3, 7]:
                row[n].set_width(0.7 + 0.2 * random.random())
        dots[1][3].set_width(0.6 + 0.1 * random.random())
        dots[2][3].set_width(0.6 + 0.1 * random.random())
        dots[6][7].set_width(0.9 + 0.1 * random.random())

        pattern_words = Text("Attention\\nPattern", font_size=120)
        pattern_words.move_to(grid_lines, UL).shift(LEFT)

        self.play(
            FadeOut(softmax_cols, lag_ratio=0.001),
            FadeIn(dots, lag_ratio=0.001),
            Write(pattern_words),
            run_time=2
        )
        self.wait()

        # Preview masking
        masked_dots = VGroup()
        for n, row in enumerate(dots):
            masked_dots.add(*row[:n])
        mask_rects = VGroup()
        for dot in masked_dots:
            mask_rect = Square(0.5)
            mask_rect.set_stroke(RED, 2)
            mask_rect.move_to(dot)
            mask_rects.add(mask_rect)

        lag_ratio=1.0 / len(mask_rects)
        self.play(ShowCreation(mask_rects, lag_ratio=lag_ratio))
        self.play(
            LaggedStart(
                (dot.animate.scale(0) for dot in masked_dots),
                lag_ratio=lag_ratio
            )
        )
        self.play(
            FadeOut(mask_rects, lag_ratio=lag_ratio)
        )
        self.wait()

        # Set aside keys and queries
        pattern = VGroup(grid_lines, dots)
        for group in q_groups:
            group.sort(lambda p: -p[1])
            group.target = group.generate_target()
            m3 = len(group) - 3
            group.target[m3:].scale(0, about_edge=DOWN)
            group.target[:m3].move_to(group, DOWN)

        self.play(
            frame.animate.move_to((-2.09, -5.59, 0.0)).set_height(12.95).set_anim_args(run_time=3),
            LaggedStartMap(MoveToTarget, q_groups),
            FadeOut(pattern_words),
            v_lines.animate.stretch(0.95, 1, about_edge=DOWN),
        )
        self.play(
            LaggedStartMap(FadeOut, k_syms, shift=0.5 * DOWN, lag_ratio=0.1),
            LaggedStartMap(FadeOut, wk_syms, shift=0.5 * DOWN, lag_ratio=0.1),
        )
        self.wait()

        # Add values
        value_color = RED
        big_wv_sym = Tex(R"W_V", font_size=90)
        big_wv_sym.set_color(value_color)
        big_wv_sym.next_to(h_lines, UP, MED_LARGE_BUFF, LEFT)
        wv_word = Text("Value matrix", font_size=90)
        wv_word.next_to(big_wv_sym, UP, MED_LARGE_BUFF)
        wv_word.set_color(value_color)

        wv_arrows = wk_arrows
        v_sym_template = Tex(R"\\vec{\\textbf{V}}_{0}")
        v_sym_template[0].scale(1.5, about_edge=DOWN)
        v_sym_template.set_fill(value_color, border_width=1)
        subscript = v_sym_template.make_number_changeable("0")

        wv_syms = VGroup()
        v_syms = VGroup()
        for n, arrow in enumerate(wv_arrows, start=1):
            wv_sym = Tex("W_V", font_size=36)
            wv_sym.set_fill(value_color, border_width=1)
            wv_sym.next_to(arrow, UP, buff=0.2, aligned_edge=LEFT)
            subscript.set_value(n)
            v_sym = v_sym_template.copy()
            v_sym.next_to(arrow, RIGHT, MED_SMALL_BUFF)

            v_syms.add(v_sym)
            wv_syms.add(wv_sym)

        self.play(
            FadeIn(big_wv_sym, 0.5 * DOWN),
            FadeIn(wv_word, lag_ratio=0.1),
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(big_wv_sym, wv_sym)
                for wv_sym in wv_syms),
                lag_ratio=0.15,
            ),
            run_time=3
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(e_sym, v_sym)
                for e_sym, v_sym in zip(key_emb_syms, v_syms)),
                lag_ratio=0.15,
            ),
        )
        self.wait()
        self.play(
            FadeTransform(v_syms, k_syms),
            FadeTransform(wv_syms, wk_syms),
            rate_func=there_and_back_with_pause,
            run_time=3,
        )
        self.remove(k_syms, wk_syms)
        self.add(v_syms, wv_syms)
        self.wait()

        # Show column of weights
        index = words.index("creature")
        weighted_sum_cols = VGroup()
        for sm_col in softmax_cols:
            weighted_sum_col = VGroup()
            for weight, v_sym in zip(sm_col, v_syms):
                product = VGroup(weight, v_sym.copy())
                product.target = product.generate_target()
                product.target.arrange(RIGHT)
                product.target[1].shift(UP * (
                    product.target[0].get_y(DOWN) -
                    product.target[1][1].get_y(DOWN)
                ))
                product.target.scale(0.75)
                product.target.move_to(weight)
                product.target.set_fill(
                    opacity=clip(0.6 + weight.get_value(), 0, 1)
                )
                weighted_sum_col.add(product)
            weighted_sum_cols.add(weighted_sum_col)

        self.play(
            FadeOut(dots, lag_ratio=0.1),
            FadeIn(q_fade_rects[:index]),
            FadeIn(q_fade_rects[index + 1:]),
            FadeIn(softmax_cols[index]),
        )
        self.wait()
        self.play(
            LaggedStartMap(MoveToTarget, weighted_sum_cols[index])
        )
        self.wait()

        # Emphasize fluffy and blue weights
        rects = VGroup(
            key_word_groups[i][0].copy()
            for i in [1, 2]
        )
        alt_rects = VGroup(
            SurroundingRectangle(value, buff=SMALL_BUFF)
            for value in (* softmax_cols[index][:1], *softmax_cols[index][3:])
        )
        alt_rects.set_stroke(RED, 1)
        self.play(
            LaggedStart(
                (rect.animate.surround(value)
                for rect, value in zip(rects, softmax_cols[index][1:3])),
                lag_ratio=0.2,
            )
        )
        self.wait()
        self.play(Transform(rects, alt_rects))
        self.wait()
        self.play(FadeOut(rects, lag_ratio=0.1))

        # Show sum (Start re-rendering here, 151)
        emb_sym = emb_syms[index]
        ws_col = weighted_sum_cols[index]
        creature = images[2]
        creature.set_height(1.5)
        creature.next_to(word_groups[index], UP)

        emb_sym.target = emb_sym.generate_target()
        emb_sym.target.scale(1.25, about_edge=UP)
        sum_rect = SurroundingRectangle(emb_sym.target)
        sum_rect.set_stroke(YELLOW, 2)
        sum_rect.target = sum_rect.generate_target()
        sum_rect.target.surround(ws_col, buff=MED_SMALL_BUFF)
        plusses = VGroup()
        for m1, m2 in zip(ws_col, ws_col[1:]):
            plus = Tex(R"+", font_size=72)
            plus.move_to(midpoint(m1.get_bottom(), m2.get_top()))
            plusses.add(plus)

        self.play(
            frame.animate.reorient(0, 0, 0, (-2.6, -4.79, 0.0), 15.07).set_anim_args(run_time=2),
            MoveToTarget(emb_sym),
            ShowCreation(sum_rect),
            FadeIn(creature, UP),
            FadeOut(wv_word),
            FadeOut(big_wv_sym),
        )
        self.add(Point(), q_fade_rects[index + 1:])  # Hack
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (-2.9, -6.5, 0.0), 19).set_anim_args(run_time=2),
            MoveToTarget(sum_rect, run_time=2),
            Write(plusses),
        )
        self.wait()

        # Show Delta E
        low_eqs = VGroup(
            Tex("=", font_size=72).rotate(PI / 2).next_to(wsc[-1].target, DOWN, buff=0.5)
            for wsc in weighted_sum_cols
        )
        low_eqs.set_color(YELLOW)
        delta_Es = VGroup()
        for emb_sym, eq in zip(emb_syms, low_eqs):
            delta = Tex(R"\\Delta")
            delta.match_height(emb_sym[1])
            delta.next_to(emb_sym[1], LEFT, buff=0, aligned_edge=DOWN)
            delta_E = VGroup(delta, emb_sym.copy())
            delta_E.set_color(YELLOW)
            delta_E.set_height(0.8)
            delta_E.next_to(eq, DOWN)
            delta_Es.add(delta_E)

        self.play(
            LaggedStart(
                (FadeTransform(term.copy(), delta_Es[index])
                for term in weighted_sum_cols[index]),
                lag_ratio=0.05,
                group_type=Group
            ),
            Write(low_eqs[index])
        )
        self.wait()

        # Add Delta E
        creature_group = Group(creature, q_groups[index]).copy()
        creature_group.target = creature_group.generate_target()
        creature_group.target.scale(1.5)
        creature_group.target.next_to(h_lines, RIGHT, buff=4.0)
        creature_group.target.align_to(creature, UP)
        right_plus = Tex("+", font_size=96)
        right_eq = Tex("=", font_size=120).rotate(PI / 2)
        right_plus.next_to(creature_group.target, DOWN)
        creature_delta_E = delta_Es[index].copy()
        creature_delta_E.target = creature_delta_E.generate_target()
        creature_delta_E.target.set_height(1.0)
        creature_delta_E.target.next_to(right_plus, DOWN)
        right_eq.next_to(creature_delta_E.target, DOWN, MED_LARGE_BUFF)
        E_prime = emb_sym_primes[index].copy()
        E_prime.set_height(1.25)
        E_prime.next_to(right_eq, DOWN, MED_LARGE_BUFF)
        blue_fluff.set_height(2.5)
        blue_fluff.next_to(E_prime, DOWN, MED_LARGE_BUFF)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (4.96, -5.61, 0.0), 19.00),
            MoveToTarget(creature_group),
            FadeTransform(sum_rect.copy(), right_plus),
            MoveToTarget(creature_delta_E),
            run_time=2,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(
            FadeTransform(creature_group[1][-4].copy(), E_prime),
            FadeTransform(creature_delta_E.copy(), E_prime),
            Write(right_eq),
            FadeTransform(creature_group[0].copy(), blue_fluff, path_arc=-PI / 2, run_time=2)
        )
        self.wait()

        right_sum_group = Group(
            creature_group, right_plus, creature_delta_E,
            right_eq, E_prime, blue_fluff
        )

        # Show all column sums
        plus_groups = VGroup(
            plusses.copy().match_x(col[0].target)
            for col in weighted_sum_cols
        )
        plus_groups.set_fill(GREY_C, 1)

        for col in softmax_cols:
            for value in col:
                value.set_fill(
                    opacity=clip(0.6 + value.get_value(), 0, 1)
                )

        self.play(LaggedStart(
            right_sum_group.animate.fade(0.75),
            FadeOut(sum_rect),
            FadeOut(creature),
            FadeOut(q_fade_rects[:index]),
            FadeOut(q_fade_rects[index + 1:]),
            FadeIn(softmax_cols[:index]),
            FadeIn(softmax_cols[index + 1:]),
            plusses.animate.set_fill(GREY_C, 1),
            run_time=2,
        ))
        self.play(
            LaggedStart(
                (LaggedStartMap(MoveToTarget, col)
                for col in weighted_sum_cols),
                lag_ratio=0.1
            ),
            v_lines.animate.set_stroke(GREY_A, 4, 1),
            *(
                e_sym.animate.scale(1.25, about_edge=UP)
                for e_sym in (*emb_syms[:index], *emb_syms[index + 1:])
            ),
        )

        other_indices = [*range(index), *range(index + 1, len(plus_groups))]
        self.play(LaggedStart(
            (LaggedStart(
                FadeIn(plus_groups[j], lag_ratio=0.1),
                Write(low_eqs[j]),
                LaggedStart(
                    (FadeTransform(ws.copy(), delta_Es[j])
                    for ws in weighted_sum_cols[j]),
                    lag_ratio=0.05,
                    group_type=Group
                ),
                lag_ratio=0.05,
            )
            for j in other_indices),
            lag_ratio=0.1,
            group_type=Group
        ))
        self.wait()

        # Add all deltas to embeddings
        equations = VGroup()
        equation_targets = VGroup()
        for E, dE, Ep in zip(emb_syms.copy(), delta_Es.copy(), emb_sym_primes):
            Ep.match_height(E)
            plus = Tex("+", font_size=96)
            eq = Tex("=", font_size=96).rotate(PI / 2)
            equation = VGroup(E, plus, dE, eq, Ep)
            equation.target = equation.generate_target()
            for mob in equation.target[::2]:
                mob.set_height(0.8)
            equation.target.arrange(DOWN)
            for mob in [Ep, plus, eq]:
                mob.set_opacity(0)
                mob.move_to(dE)
            equations.add(equation)
            equation_targets.add(equation.target)

        equation_targets.scale(1.25)
        equation_targets.arrange(RIGHT, buff=0.75)
        equation_targets.next_to(h_lines, RIGHT, buff=1.5)
        equation_targets.match_y(h_lines)

        self.play(
            frame.animate.reorient(0, 0, 0, (9.5, -7.17, 0.0), 20.33),
            LaggedStartMap(MoveToTarget, equations, lag_ratio=0.05),
            FadeTransform(right_sum_group, equation_targets[index]),
            run_time=2.0
        )
        self.wait()

        result_rect = SurroundingRectangle(
            VGroup(eq[-1] for eq in equations),
            buff=0.25
        )
        result_rect.set_stroke(TEAL, 3)
        self.play(
            ShowCreation(result_rect)
        )
        self.wait()

    def bake_mobject_into_vector_entries(self, mob, vector, path_arc=30 * DEGREES, group_type=None):
        entries = vector.get_entries()
        mob_copies = mob.replicate(len(entries))
        return AnimationGroup(
            LaggedStart(
                (FadeOutToPoint(mc, entry.get_center(), path_arc=path_arc)
                for mc, entry in zip(mob_copies, entries)),
                lag_ratio=0.05,
                group_type=group_type,
                run_time=2,
                remover=True
            ),
            RandomizeMatrixEntries(
                vector,
                rate_func=lambda t: clip(smooth(2 * t - 1), 0, 1),
                run_time=2
            ),
        )

    def scrap():
        # To be inserted after Show grid of dots sections
        self.remove(dot_prods)
        np.random.seed(time.gmtime().tm_sec)
        pattern = np.random.normal(0, 1, (8, 8))
        for n in range(len(pattern[0])):
            pattern[:, n][n + 1:] = -np.inf
            pattern[:, n] = softmax(pattern[:, n])
        for row, arr in zip(dots, pattern):
            for dot, value in zip(row, arr):
                dot.set_width(value**0.5)
        dots.set_fill(GREY_B, 1)
        return

        ### To be inserted in "Show softmax" section
        np.random.seed(time.gmtime().tm_sec)
        softmax_arrays = np.random.normal(0, 1, (8, 8))
        for n in range(len(softmax_arrays[0])):
            softmax_arrays[:, n][n + 1:] = -np.inf
            softmax_arrays[:, n] = softmax(softmax_arrays[:, n])
        softmax_arrays = softmax_arrays.T
        ###

    def thumbnail():
        ### Thumbnail design, insert in the middle of softmax show columns ###
        self.remove(q_groups)
        self.add(q_syms)
        out_dots = VGroup()
        for col in softmax_cols:
            for value in col:
                dot = Dot(radius=0.35)
                dot.move_to(value)
                dot.set_fill(WHITE, opacity=interpolate(0.1, 0.9, value.get_value()))
                out_dots.add(dot)
        out_dots.shift(2 * OUT)
        out_dots.set_stroke(WHITE, 2, 0.25)
        self.remove(softmax_cols)
        self.remove(sm_rects[last_index])
        self.add(out_dots)
        index = 3
        ndp_columns[-1].set_opacity(0.25)
        ndp_columns[index].set_opacity(1)
        sm_label_group = VGroup(sm_arrows[last_index], sm_labels[last_index])
        sm_label_group.match_x(ndp_columns[index])
        sm_label_group[1].scale(1.5, about_edge=DOWN)
        sm_label_group[1].set_fill(border_width=0)
        col_rect.match_x(ndp_columns[index])
        col_rect.set_flat_stroke(False)
        sm_col = col_rect.copy()
        # sm_col.set_width(out_dots[0].get_width() + 0.2)
        sm_col.match_z(out_dots)
        sm_col.set_flat_stroke(False)
        self.add(sm_col)
        self.remove(sm_labels[last_index])
        sm_arrows[last_index].set_stroke(width=10)
        sm_arrows[last_index].shift(OUT)

        grid_lines.set_stroke(WHITE, 2)
        v_lines.set_height(12, about_edge=DOWN, stretch=True)

        frame.set_field_of_view(35 * DEGREES)
        frame.reorient(-52, -2, 0, (-1.74, -7.1, -0.03), 14.72)
        ###

        ### To be inserted before Set aside keys and queries
        frame.move_to([-4.62, -5.04, 0.0]).set_height(14.5)
        self.remove(pattern_words)

        for dot in dots.family_members_with_points():
            value = dot.get_radius() / 0.5
            dot.set_fill(WHITE, opacity=value**0.75)
            dot.set_width(1)

        title = Text("Attention", font_size=250)
        title.set_fill(border_width=2)
        title.next_to(q_syms, LEFT, LARGE_BUFF, DOWN)
        title.shift(0.5 * UP)
        # self.add(title)

        q_syms.set_fill(border_width=1.5)
        k_syms.set_fill(border_width=1.5)
        for q in q_syms:
            q.scale(1.5, about_edge=DOWN)
        for k in k_syms:
            k.scale(1.5, about_edge=RIGHT)

        self.remove(word_groups, q_arrows, emb_arrows, emb_syms, wq_syms)
        VGroup(key_word_groups, key_emb_syms, key_emb_arrows, wk_arrows, wk_syms).shift(0.25 * LEFT)
        ###


class MyseteryNovel(InteractiveScene):
    def construct(self):
        # Create paragraphs
        text = Path(DATA_DIR, "murder_story.txt").read_text()
        paragraphs = VGroup(
            get_paragraph(para.split(" "), line_len=40)
            for para in text.split("\\n\\n")
        )
        dots = Tex(R"\\vdots", font_size=200)
        paragraphs.replace_submobject(4, dots)
        paragraphs.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        dots.match_x(paragraphs)
        self.add(paragraphs)

        # Mark last word
        last_word = paragraphs[-1]["Derek!\\""][0]
        rect = SurroundingRectangle(last_word)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.25)
        q_marks = Tex("???")
        q_marks.move_to(rect)
        rect.add(q_marks)
        rect.shift(0.05 * DR)

        last_word.scale(0).set_fill(BLACK)
        self.add(rect)

        # Show the first line
        frame = self.frame
        frame.set_y(15)
        paragraphs.set_fill(opacity=0.25)
        opening = paragraphs[0]["It was a dark and stormy night."][0]
        self.play(opening.animate.set_fill(opacity=1).set_anim_args(lag_ratio=0.1))
        self.wait()

        # Scroll down
        penultimate_words = paragraphs[-1]["therefore, the murderer was"][0]
        self.play(
            frame.animate.set_y(-15.4),
            paragraphs.animate.set_fill(opacity=1).set_anim_args(lag_ratio=0.01),
            run_time=5,
        )
        self.wait()
        self.add(penultimate_words.copy())
        self.play(paragraphs.animate.set_opacity(0.25))
        self.wait()

        # Show the final vector
        was = penultimate_words[-3:]
        arrow = FillArrow(ORIGIN, DOWN, buff=0, thickness=0.07)
        arrow.next_to(was, DOWN, MED_SMALL_BUFF)
        vect = NumericEmbedding(length=12)
        vect.set_height(5)
        vect.next_to(arrow, DOWN)

        self.play(LaggedStart(
            frame.animate.set_y(-17.5).set_height(12.5),
            FadeIn(arrow, scale=3, shift=DOWN),
            FadeIn(vect, DOWN),
            run_time=2
        ))
        self.context_anim(paragraphs[-1], vect)
        self.wait()

        # Zoom out more
        vect_group = VGroup(arrow, vect)
        vect_group.target = vect_group.generate_target()
        vect_group.target.scale(2.35, about_edge=UP)
        self.play(
            paragraphs.animate.set_fill(opacity=0.8),
            frame.animate.set_height(37).set_y(-14),
            MoveToTarget(vect_group),
            run_time=2
        )
        self.context_anim(paragraphs[-4:], vect)

    def context_anim(self, source, vect):
        flat_source = VGroup(*source.family_members_with_points())
        vect_len = len(vect.get_entries())
        self.play(
            LaggedStart(
                (ContextAnimation(
                    entry, flat_source[n::vect_len],
                    path_arc=-PI / 2,
                    run_time=5,
                    lag_ratio=1e-3,
                    max_stroke_width=2
                )
                for n, entry in enumerate(vect.get_entries())),
                lag_ratio=0.1,
            ),
            RandomizeMatrixEntries(vect, run_time=5),
        )


class RoadNotTaken(InteractiveScene):
    def construct(self):
        # Add poem
        stanza_strs = [
            """
                Two roads diverged in a yellow wood,
                And sorry I could not travel both
                And be one traveler, long I stood
                And looked down one as far as I could
                To where it bent in the undergrowth;
            """,
            """
                Then took the other, as just as fair,
                And having perhaps the better claim,
                Because it was grassy and wanted wear;
                Though as for that the passing there
                Had worn them really about the same,
            """,
            """
                And both that morning equally lay
                In leaves no step had trodden black.
                Oh, I kept the first for another day!
                Yet knowing how way leads on to way,
                I doubted if I should ever come back.
            """,
            """
                I shall be telling this with a sigh
                Somewhere ages and ages hence:
                Two roads diverged in a wood, and I—
                I took the one less traveled by,
                And that has made all the difference.
            """,
        ]
        poem = Text("\\n\\n".join(stanza_strs), alignment="LEFT")
        stanzas = VGroup(poem[stanza_str][0] for stanza_str in stanza_strs)
        stanzas.arrange_in_grid(h_buff=1.5, v_buff=1.0, fill_rows_first=False)
        stanzas.set_width(FRAME_WIDTH - 1)
        stanzas.move_to(0.5 * UP)
        poem.refresh_bounding_box(recurse_down=True)

        self.play(FadeIn(poem, lag_ratio=0.01, run_time=4))
        self.wait()

        # Note all text until "one"
        rect = SurroundingRectangle(poem)
        less = poem["less"][-1]
        one = poem["one"][-1]
        diff_rects = VGroup(
            SurroundingRectangle(mob).scale(10, about_edge=UL)
            for mob in [less, poem["And"][-1]]
        )
        for diff_rect in diff_rects:
            rect = Difference(rect, diff_rect)
        rect.set_stroke(TEAL, 3)

        less_index = poem.submobjects.index(less[0])
        faded_portion = poem[less_index:]
        active_portion = poem[:less_index]
        less_rect = SurroundingRectangle(less)
        less_rect.set_stroke(YELLOW, 3)
        one_rect = SurroundingRectangle(one)
        one_rect.become(Difference(one_rect, less_rect))
        one_rect.match_height(less_rect, about_edge=DOWN, stretch=True)
        one_rect.set_stroke(BLUE, 3)
        arrow = Vector(0.75 * UP)
        arrow.next_to(one, DOWN, SMALL_BUFF)
        arrow.set_stroke(YELLOW)
        active_portion_copy = active_portion.copy()
        active_portion_copy.set_color(TEAL_B)

        self.play(
            FadeIn(rect),
            Write(active_portion_copy, run_time=2, stroke_color=TEAL, lag_ratio=0.01),
            faded_portion.animate.set_fill(opacity=0.5),
        )
        self.play(FadeOut(active_portion_copy))
        self.wait()
        self.play(GrowArrow(arrow))
        self.wait()
        self.play(
            ShowCreation(less_rect),
            less.animate.set_fill(opacity=1),
            arrow.animate.match_x(less),
        )
        self.wait()
        self.remove(less_rect)
        self.play(
            arrow.animate.match_x(one),
            TransformFromCopy(less_rect, one_rect),
        )
        self.wait()

        # Highlight "two roads"
        one = one.copy()
        less = less.copy()
        two_roads = poem["Two roads"][-1].copy()
        took_the = poem["I took the"][-1].copy()

        self.play(
            FadeIn(two_roads, lag_ratio=0.1),
            FadeIn(took_the, lag_ratio=0.1),
            FadeIn(one),
            arrow.animate.rotate(-PI / 2).next_to(two_roads, LEFT, SMALL_BUFF),
            poem.animate.set_fill(opacity=0.5),
            run_time=1.5
        )
        self.wait()

        # Highlight "took the other" and "grassy and wanted wear"
        top_two_roads = poem["Two roads diverged"][0].copy()
        took_other = poem["Then took the other"][0].copy()
        wanted_wear = poem["it was grassy and wanted wear"][0].copy()
        for phrase in [top_two_roads, took_other, wanted_wear]:
            phrase.set_fill(WHITE, 1)

        self.play(
            arrow.animate.rotate(PI / 2).next_to(top_two_roads, DOWN, SMALL_BUFF),
            FadeIn(top_two_roads),
        )
        self.wait()
        self.play(
            arrow.animate.rotate(3 * PI / 4).next_to(took_other, UP, SMALL_BUFF),
            FadeIn(took_other)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(-PI / 2).next_to(wanted_wear, DOWN, SMALL_BUFF),
            FadeIn(wanted_wear)
        )
        self.wait()

        # Higlight words throughout
        active_portion_copy.set_fill(YELLOW_A, 1)

        self.play(
            LaggedStart(
                (FadeIn(char, rate_func=there_and_back_with_pause)
                for char in active_portion_copy),
                lag_ratio=0.005,
                run_time=6
            )
        )
        self.wait()

        # Show less again
        self.play(
            arrow.animate.rotate(-PI / 4).next_to(less, DOWN, SMALL_BUFF),
            ShowCreation(less_rect),
            less.animate.set_fill(WHITE, 1)
        )
        self.wait()

        # Show final embedding
        frame = self.frame
        embedding = NumericEmbedding(length=10)
        embedding.set_height(3)
        embedding.next_to(one, DOWN, buff=arrow.get_length() + 2 * SMALL_BUFF)

        self.play(
            arrow.animate.rotate(PI).next_to(one, DOWN, SMALL_BUFF).set_anim_args(path_arc=PI),
            frame.animate.set_height(9).move_to(DOWN)
        )
        self.play(TransformFromCopy(one, embedding))
        self.play(RandomizeMatrixEntries(embedding))
        self.wait()


class QueryMap(InteractiveScene):
    map_tex = "W_Q"
    map_color = YELLOW
    src_name = "Creature"
    pos_word = "position 4"
    trg_name = "Any adjectives\\nbefore position 4?"
    in_vect_color = BLUE_B
    in_vect_coords = (3, 2, -2)
    out_vect_coords = (-2, -1)

    def construct(self):
        # Setup 3d axes
        axes_3d = ThreeDAxes((-4, 4), (-3, 3), (-4, 4))
        xz_plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_color=GREY,
                stroke_width=1,
            ),
            faded_line_ratio=0
        )
        xz_plane.rotate(90 * DEGREES, RIGHT)
        xz_plane.move_to(axes_3d)
        xz_plane.axes.set_opacity(0)
        axes_3d.add(xz_plane)
        axes_3d.set_height(2.0)

        self.set_floor_plane("xz")
        frame = self.frame
        frame.set_field_of_view(30 * DEGREES)
        frame.reorient(-32, 0, 0, (2.13, 1.11, 0.27), 4.50)
        frame.add_ambient_rotation(1 * DEGREES)

        self.add(axes_3d)

        # Set up target plane
        plane = NumberPlane(
            (-3, 3), (-3, 3),
            faded_line_ratio=1,
            background_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.75
            ),
            faded_line_style=dict(
                stroke_color=BLUE,
                stroke_width=1,
                stroke_opacity=0.25,
            )
        )
        plane.set_height(3.5)
        plane.to_corner(DR)

        arrow = Tex(R"\\longrightarrow")
        arrow.set_width(2)
        arrow.stretch(0.75, 1)
        arrow.next_to(plane, LEFT, buff=1.0)
        arrow.set_color(self.map_color)

        map_name = Tex(self.map_tex, font_size=72)
        map_name.set_color(self.map_color)
        map_name.next_to(arrow.get_left(), UR, SMALL_BUFF).shift(0.25 * RIGHT)

        for mob in [plane, arrow, map_name]:
            mob.fix_in_frame()

        self.add(plane)
        self.add(arrow)
        self.add(map_name)

        # Add titles
        titles = VGroup(
            Text("Embedding space"),
            Text("Query/Key space"),
        )
        subtitles = VGroup(
            Text("12,288-dimensional"),
            Text("128-dimensional"),
        )
        subtitles.scale(0.75)
        subtitles.set_fill(GREY_B)
        x_values = [-frame.get_x() * FRAME_HEIGHT / frame.get_height(), plane.get_x()]
        for title, subtitle, x_value in zip(titles, subtitles, x_values):
            subtitle.next_to(title, DOWN, SMALL_BUFF)
            title.add(subtitle)
            title.next_to(plane, UP, MED_LARGE_BUFF)
            title.set_x(x_value)
            title.fix_in_frame()

        self.add(titles)

        # Show vector transformation
        in_vect = Arrow(axes_3d.get_origin(), axes_3d.c2p(*self.in_vect_coords), buff=0)
        in_vect.set_stroke(self.in_vect_color)
        in_vect_label = TexText("\`\`" + self.src_name + "''", font_size=24)
        pos_label = Text(self.pos_word, font_size=16)
        pos_label.next_to(in_vect_label, DOWN, SMALL_BUFF)
        pos_label.set_opacity(0.75)
        in_vect_label.add(pos_label)
        in_vect_label.set_color(self.in_vect_color)
        in_vect_label.next_to(in_vect.get_end(), UP, SMALL_BUFF)

        out_vect = Arrow(plane.get_origin(), plane.c2p(*self.out_vect_coords), buff=0)
        out_vect.set_stroke(self.map_color)
        out_vect_label = Text(self.trg_name, font_size=30)
        out_vect_label.next_to(out_vect.get_end(), DOWN, buff=0.2)
        out_vect_label.set_backstroke(BLACK, 5)
        VGroup(out_vect, out_vect_label).fix_in_frame()

        self.play(
            GrowArrow(in_vect),
            FadeInFromPoint(in_vect_label, axes_3d.get_origin()),
        )
        self.wait(2)
        self.play(
            TransformFromCopy(in_vect, out_vect),
            FadeTransform(in_vect_label.copy(), out_vect_label),
            run_time=2,
        )
        self.wait(20)
        self.play(FadeOut(out_vect_label))
        self.wait(5)


class KeyMap(QueryMap):
    map_tex = "W_K"
    map_color = TEAL
    src_name = "Fluffy"
    pos_word = "position 2"
    trg_name = "Adjective at\\nposition 2"
    in_vect_color = BLUE_B
    in_vect_coords = (-3, 1, 2)
    out_vect_coords = (-1.75, -1)


class DescribeAttentionEquation(InteractiveScene):
    def construct(self):
        # Stage image
        image = ImageMobject("AttentionPaperStill")
        image.set_height(FRAME_HEIGHT)
        self.add(image)

        # Add equation
        equation = Tex(R"\\text{Attention}(Q, K, V) = \\text{softmax}\\left({K^T Q \\over \\sqrt{d_k}}\\right) V")
        equation.set_height(1.06929)
        equation.move_to([-0.41406, 1.177, 0])

        self.play(
            FadeIn(equation),
            FadeOut(image),
        )
        self.wait()

        # Show Q and K arrays
        syms = ["Q", "K"]
        colors = [YELLOW, TEAL]
        q_array, k_array = arrays = VGroup(
            self.get_array_representation(sym, color)
            for sym, color in zip(syms, colors)
        )
        arrays.arrange(RIGHT, buff=1.5)
        arrays.next_to(equation, DOWN, buff=1.0)

        lil_rects = VGroup()
        rect_lines = VGroup()
        big_rects = VGroup()
        for arr, sym, color in zip(arrays, syms, colors):
            lil_rect = SurroundingRectangle(equation["Q"][0])
            lil_rect.match_x(equation[sym][0])
            big_rect = SurroundingRectangle(arr)
            lines = VGroup(
                Line(lil_rect.get_corner(DOWN + v), big_rect.get_corner(UP + v))
                for v in [LEFT, RIGHT]
            )
            VGroup(lil_rect, big_rect, lines).set_stroke(color, 2)
            lil_rects.add(lil_rect)
            rect_lines.add(lines)
            big_rects.add(big_rect)

            self.play(
                ShowCreation(lil_rect),
                equation[sym].animate.set_color(color),
            )
            self.play(
                TransformFromCopy(lil_rect, big_rect),
                FadeInFromPoint(arr, lil_rect.get_center()),
                ShowCreation(lines, lag_ratio=0)
            )
        self.wait()

        # Highlight numerator
        num_rect = SurroundingRectangle(equation["K^T Q"])
        num_rect.set_stroke(BLUE, 2)

        self.play(
            ReplacementTransform(lil_rects[0], num_rect),
            ReplacementTransform(lil_rects[1], num_rect),
            FadeOut(rect_lines)
        )
        self.wait()

        # Arrange for grid
        frame = self.frame
        qs = q_array[1]
        ks = k_array[1]
        q_array.remove(qs)
        k_array.remove(ks)

        h_buff = 0.8
        v_buff = 0.6

        qs.target = qs.generate_target()
        qs.target.scale(0.75)
        qs.target.arrange(RIGHT, buff=h_buff)
        qs.target.next_to(equation, DOWN, buff=0.75)

        ks.target = ks.generate_target()
        ks.target.scale(0.75)
        ks.target.arrange(DOWN, buff=MED_LARGE_BUFF)
        ks.target[-2].rotate(PI / 2)
        ks.target.next_to(qs.target, DL, buff=v_buff)

        self.play(
            frame.animate.move_to(1.5 * DOWN),
            FadeOut(q_array),
            FadeOut(k_array),
            MoveToTarget(qs),
            MoveToTarget(ks),
            big_rects[0].animate.surround(qs.target).set_stroke(opacity=0),
            big_rects[1].animate.surround(ks.target).set_stroke(opacity=0),
            run_time=2
        )

        # Add grid lines
        grid = VGroup(qs, ks)

        v_lines = Line(UP, DOWN).match_height(grid).scale(1.1).replicate(len(qs) + 1)
        for v_line, mob in zip(v_lines, (ks, *qs)):
            v_line.next_to(mob, RIGHT, buff=h_buff / 2)
            v_line.align_to(qs, UP)

        h_lines = Line(LEFT, RIGHT).match_width(grid).scale(1.1).replicate(len(ks) + 1)
        for h_line, mob in zip(h_lines, (qs, *ks)):
            h_line.next_to(mob, DOWN, buff=v_buff / 2)
            h_line.align_to(ks, LEFT)

        VGroup(v_lines, h_lines).set_stroke(GREY_B, 1)

        grid.add(v_lines, h_lines)

        self.play(
            FadeIn(h_lines, lag_ratio=0.1),
            FadeIn(v_lines, lag_ratio=0.1),
            ks[-2].animate.match_y(h_lines[-3:-1]),
        )

        # Dot products
        dot_prods = VGroup()
        for q in qs:
            for k in ks:
                dot = Tex(".")
                dot.match_x(q)
                dot.match_y(k)
                dot_prod = VGroup(q.copy(), dot, k.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=SMALL_BUFF)
                dot_prod.target.scale(0.7)
                dot_prod.target.move_to(dot)
                if len(q) == 3:
                    dot_prod.target[1:].scale(0)
                    for mob in dot_prod.target:
                        mob.move_to(dot)
                elif len(k) == 3:
                    dot_prod.target[:-1].scale(0)
                    for mob in dot_prod.target:
                        mob.move_to(dot)
                dot.set_opacity(0)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.01),
            run_time=3
        )
        self.wait()

        # Add sqrt to denominator
        sqrt_part = equation[R"\\over \\sqrt{d_k}"][0]

        denoms = VGroup()
        for dot_prod in dot_prods:
            dot_prod.target = dot_prod.generate_target()
            if 3 in [len(dot_prod[0]), len(dot_prod[2])]:
                continue
            denom = sqrt_part.copy()
            denom.set_fill(opacity=0.9)
            denom.match_width(dot_prod)
            denom.move_to(dot_prod.get_center(), UP)
            dot_prod.target.next_to(denom, UP, buff=SMALL_BUFF)
            VGroup(dot_prod.target, denom).scale(0.75)
            denoms.add(denom)

        self.play(num_rect.animate.surround(equation[R"K^T Q \\over \\sqrt{d_k}"]))
        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.05, time_span=(1, 3)),
            LaggedStart(
                (TransformFromCopy(sqrt_part, denom)
                for denom in denoms),
                lag_ratio=0.01,
            ),
            run_time=3
        )
        self.wait()

        # Highlight softmax
        self.play(
            num_rect.animate.surround(equation[R"\\text{softmax}\\left({K^T Q \\over \\sqrt{d_k}}\\right)"])
        )
        self.wait()

        # Mention V
        v_parts = equation["V"]
        v_rects = VGroup(map(SurroundingRectangle, v_parts))
        v_rects.set_stroke(RED, 3)

        self.play(
            ReplacementTransform(VGroup(num_rect), v_rects),
            v_parts.animate.set_color(RED),
        )
        self.wait()

    def get_array_representation(self, sym, color=WHITE, length=7):
        template = Tex(f"{sym}_0")
        template.set_fill(color)
        substr = template.make_number_changeable(0)
        terms = VGroup()
        term_lines = VGroup()
        term_groups = VGroup()
        for n in range(1, length + 1):
            if n == length:
                substr.become(Tex("n").replace(substr))
            else:
                substr.set_value(n)
            substr.set_color(color)
            term = template.copy()
            lines = Line(ORIGIN, 0.5 * UP).replicate(2)
            lines.arrange(DOWN, buff=term.get_height() + 2 * SMALL_BUFF)
            lines.move_to(term)
            term_lines.add(lines)
            terms.add(term)
            term_groups.add(VGroup(term, lines))
        term_groups.arrange(RIGHT, buff=MED_SMALL_BUFF)

        dots = Tex(R"\\dots")
        dots.replace(terms[-2], dim_to_match=0)
        terms.replace_submobject(length - 2, dots)
        term_groups.remove(term_groups[-2])

        brackets = Tex("[]")
        brackets.stretch(1.5, 1)
        brackets.set_height(term_groups.get_height() + MED_SMALL_BUFF)
        for bracket, vect in zip(brackets, [LEFT, RIGHT]):
            bracket.next_to(terms, vect, SMALL_BUFF)

        result = VGroup(brackets, terms, term_lines)

        return result


class ShowAllPossibleNextTokenPredictions(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("the fluffy blue creature roamed the verdant forest despite")
        plain_words = break_into_words(phrase)
        rects = get_piece_rectangles(plain_words)
        words = VGroup(VGroup(*pair) for pair in zip(rects, plain_words))
        words = words[:-1]
        words.to_edge(LEFT, buff=MED_LARGE_BUFF)

        next_token_box = rects[-1].copy()
        next_token_box.set_color(YELLOW)
        next_token_box.set_stroke(YELLOW, 3)
        next_token_box.next_to(words, RIGHT, buff=LARGE_BUFF)
        q_marks = Tex("???")
        q_marks.move_to(next_token_box)
        next_token_box.add(q_marks)

        arrow = Arrow(words, next_token_box, buff=SMALL_BUFF)

        self.add(words)
        self.play(
            GrowArrow(arrow),
            FadeIn(next_token_box, RIGHT)
        )
        self.wait()

        # Set up subphrases
        scale_factor = 0.75
        v_buff = 0.4
        subphrases = VGroup(
            words[:n].copy().scale(scale_factor)
            for n in range(1, len(words) + 1)
        )
        subphrases.arrange(DOWN, buff=v_buff, aligned_edge=LEFT)
        subphrases.to_corner(UL)

        rhs = VGroup(arrow, next_token_box)
        alt_rhss = VGroup(
            rhs.copy().scale(scale_factor).next_to(subphrase, RIGHT, SMALL_BUFF)
            for subphrase in subphrases
        )

        self.play(
            Transform(words, subphrases[-1]),
            Transform(rhs, alt_rhss[-1]),
        )
        for n in range(len(subphrases) - 1, 0, -1):
            sp1 = subphrases[n]
            sp2 = subphrases[n - 1]
            rhs1 = alt_rhss[n]
            rhs2 = alt_rhss[n - 1]
            self.play(
                TransformFromCopy(sp1[:len(sp2)], sp2),
                TransformFromCopy(rhs1, rhs2),
                rate_func=linear,
                run_time=0.5
            )
        self.wait()

        # Highlight two examples
        for phrase, alt_rhs in zip(subphrases, alt_rhss):
            arrow = alt_rhs[0]
            alt_rhs.remove(arrow)
            phrase.add(arrow)
            phrase.save_state()
        index = 3
        self.play(LaggedStart(
            FadeOut(alt_rhss),
            FadeOut(rhs),
            words.animate.fade(0.75),
            subphrases[:index].animate.fade(0.75),
            subphrases[index + 1:].animate.fade(0.75),
            subphrases[index].animate.align_to(3 * RIGHT, RIGHT),
        ))
        self.wait()
        self.play(
            subphrases[index].animate.align_to(subphrases, LEFT).fade(0.75),
            subphrases[5].animate.restore().align_to(3 * RIGHT, RIGHT),
        )
        self.wait()

    def get_next_word_distribution():
        pass


class ShowMasking(InteractiveScene):
    def construct(self):
        # Set up two patterns
        shape = (6, 6)
        left_grid = Square().get_grid(*shape, buff=0)
        left_grid.set_shape(5.5, 5)
        left_grid.to_edge(LEFT)
        left_grid.set_y(-0.5)
        left_grid.set_stroke(GREY_B, 1)

        right_grid = left_grid.copy()
        right_grid.to_edge(RIGHT)

        grids = VGroup(left_grid, right_grid)
        arrow = Arrow(left_grid, right_grid)
        sm_label = Text("softmax")
        sm_label.next_to(arrow, UP)

        titles = VGroup(
            Text("Unnormalized\\nAttention Pattern"),
            Text("Normalized\\nAttention Pattern"),
        )
        for title, grid in zip(titles, grids):
            title.next_to(grid, UP, buff=MED_LARGE_BUFF)

        values_array = np.random.normal(0, 2, shape)
        font_size = 30
        raw_values = VGroup(
            DecimalNumber(
                value,
                include_sign=True,
                font_size=font_size,
            ).move_to(square)
            for square, value in zip(left_grid, values_array.flatten())
        )

        self.add(left_grid)
        self.add(right_grid)
        self.add(titles)
        self.add(arrow)
        self.add(sm_label)
        self.add(raw_values)

        # Highlight lower lefts
        changers = VGroup()
        for n, dec in enumerate(raw_values):
            i = n // shape[1]
            j = n % shape[1]
            if i > j:
                changers.add(dec)
                neg_inf = Tex(R"-\\infty", font_size=36)
                neg_inf.move_to(dec)
                neg_inf.set_fill(RED, border_width=1.5)
                dec.target = neg_inf
                values_array[i, j] = -np.inf
        rects = VGroup(map(SurroundingRectangle, changers))
        rects.set_stroke(RED, 3)

        self.play(LaggedStartMap(ShowCreation, rects))
        self.play(
            LaggedStartMap(FadeOut, rects),
            LaggedStartMap(MoveToTarget, changers)
        )
        self.wait()

        # Normalized values
        normalized_array = np.array([
            softmax(col)
            for col in values_array.T
        ]).T
        normalized_values = VGroup(
            DecimalNumber(value, font_size=font_size).move_to(square)
            for square, value in zip(right_grid, normalized_array.flatten())
        )
        for n, value in enumerate(normalized_values):
            value.set_fill(opacity=interpolate(0.5, 1, rush_from(value.get_value())))
            if (n // shape[1]) > (n % shape[1]):
                value.set_fill(RED, 0.75)

        self.play(
            LaggedStart(
                (FadeTransform(v1.copy(), v2)
                for v1, v2 in zip(raw_values, normalized_values)),
                lag_ratio=0.05,
                group_type=Group
            )
        )
        self.wait()


class ScalingAPattern(InteractiveScene):
    def construct(self):
        # Position grid
        N = 50
        grid = Square(side_length=1.0).get_grid(N, N, buff=0)
        grid.set_stroke(GREY_A, 1)
        grid.stretch(0.89, 0)
        grid.stretch(0.70, 1)
        # grid.move_to(1.67 * LEFT + 1.596 * UP, UL)
        grid.move_to(5.0 * LEFT + 2.5 * UP, UL)
        self.add(grid)

        # Dots
        values = np.random.normal(0, 1, (N, N))
        dots = VGroup()
        for n, row in enumerate(values):
            row[:n] = -np.inf
        for k, col in enumerate(values.T):
            for n, value in enumerate(softmax(col)):
                dot = Dot(radius=0.3 * value**0.75)
                dot.move_to(grid[n * N + k])
                dots.add(dot)
        dots.set_fill(GREY_C, 1)
        self.add(dots)

        # Add symbols
        q_template = Tex(R"\\vec{\\textbf{Q}}_0").set_color(YELLOW)
        k_template = Tex(R"\\vec{\\textbf{K}}_0").set_color(TEAL)
        for template in [q_template, k_template]:
            template.scale(0.75)
            template.substr = template.make_number_changeable("0")

        qs = VGroup()
        ks = VGroup()
        for n, square in enumerate(grid[:N], start=1):
            q_template.substr.set_value(n)
            q_template.next_to(square, UP, buff=SMALL_BUFF)
            qs.add(q_template.copy())
        for k, square in enumerate(grid[::N], start=1):
            k_template.substr.set_value(k)
            k_template.next_to(square, LEFT, buff=2 * SMALL_BUFF)
            ks.add(k_template.copy())
        self.add(qs, ks)

        # Slowly zoom out
        self.play(
            self.frame.animate.reorient(0, 0, 0, (14.72, -14.71, 0.0), 38.06),
            grid.animate.set_stroke(width=1, opacity=0.25),
            dots.animate.set_fill(GREY_B, 1).set_stroke(GREY_B, 1),
            run_time=20,
        )
        self.wait()


class IntroduceValueMatrix(InteractiveScene):
    def setup(self):
        self.fix_new_entries_in_frame = False
        super().setup()

    def construct(self):
        # Initialized axes
        frame = self.frame
        self.set_floor_plane("xz")
        axes = ThreeDAxes((-4, 4), (-4, 4), (-4, 4))
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_color=GREY,
                stroke_width=1,
                stroke_opacity=0.5,
            )
        )
        plane.axes.set_opacity(0)
        plane.rotate(PI / 2, RIGHT)
        axes.add(plane)

        frame.reorient(5, -4, 0, (-4.66, 2.07, 0.04), 12.48)
        # frame.add_ambient_rotation()
        self.add(axes)

        # Add word pair
        words = VGroup(Text("blue"), Text("fluffy"), Text("creature"))
        words.scale(1.5)
        words.arrange(RIGHT, aligned_edge=UP)
        words.to_edge(UP)
        words.to_edge(LEFT, buff=0)
        rects = get_piece_rectangles(words, h_buff=0.1)
        rects[0].set_color(BLUE)
        rects[1].set_color(TEAL)
        rects[2].set_color(ORANGE)
        arrows = VGroup(Vector(DOWN).next_to(rect, DOWN) for rect in rects)
        embs = VGroup(
            NumericEmbedding(length=8).set_height(4.0).next_to(arrow, DOWN)
            for arrow in arrows
        )

        blue_group = VGroup(rects[0], words[0], arrows[0], embs[0])
        blue_group.set_opacity(0)

        self.fix_new_entries_in_frame = True
        self.add(rects)
        self.add(words)
        self.add(arrows)
        self.add(embs)

        # Add word vectors
        creature_vect = self.get_labeled_vector(axes, (-2, 3, 1), ORANGE, "Dalle3_creature")
        with_fluffy_vect = self.get_labeled_vector(axes, (2, 3, 1), GREY_BROWN, "Dalle3_creature_2")
        with_blue_vect = self.get_labeled_vector(axes, (1, 2, 4), BLUE, "BlueFluff")

        self.wait()
        self.fix_new_entries_in_frame = False
        self.play(
            FadeTransform(words[1].copy(), creature_vect[1]),
            TransformFromCopy(
                Arrow(embs[1].get_bottom(), embs[1].get_top(), buff=0).fix_in_frame().set_stroke(width=10, opacity=0.25),
                creature_vect[0],
            )
        )
        self.add(creature_vect)

        # Show influence
        diff_vect = Arrow(
            creature_vect[0].get_end(),
            with_fluffy_vect[0].get_end(),
            buff=0
        )
        diff_vect.scale(0.95)
        self.fix_new_entries_in_frame = False
        self.play(
            FadeTransform(creature_vect[1].copy(), with_fluffy_vect[1]),
            TransformFromCopy(creature_vect[0], with_fluffy_vect[0]),
            run_time=3,
        )
        self.add(with_fluffy_vect)
        self.play(GrowArrow(diff_vect, run_time=2))

        self.fix_new_entries_in_frame = True
        self.play(
            RandomizeMatrixEntries(embs[2], time_span=(1, 5)),
            LaggedStart(
                (ContextAnimation(entry, embs[1].get_entries(), path_arc=10 * DEGREES, lag_ratio=0.1)
                for entry in embs[2].get_entries()),
                lag_ratio=0.01,
                run_time=5,
            ),
        )
        self.wait()

        # Make room
        corner_group = VGroup(rects, words, arrows, embs)
        self.play(
            frame.animate.reorient(10, -7, 0, (-8.33, -0.79, 0.37), 16.82),
            corner_group.animate.set_height(3).to_edge(UP, buff=0.25).set_x(-2),
            run_time=2
        )

        # Show value matrix
        matrix = WeightMatrix(shape=(8, 8))
        matrix.set_height(2.75)
        matrix.to_corner(DL)
        matrix_brace = Brace(matrix, UP)
        matrix_label = Tex("W_V")
        matrix_label.next_to(matrix_brace, UP)
        matrix_label.set_color(RED)

        fluff_emb = embs[1]
        in_vect_rect = SurroundingRectangle(fluff_emb)
        in_vect_rect.set_stroke(TEAL, 2)
        in_vect = fluff_emb.copy()
        in_vect.match_height(matrix)
        in_vect.next_to(matrix, RIGHT, SMALL_BUFF)
        in_vect_path = self.get_top_vect_to_low_vect_path(fluff_emb, in_vect, TEAL)

        self.fix_new_entries_in_frame = True
        self.play(
            FadeIn(matrix, lag_ratio=1e-3),
            GrowFromCenter(matrix_brace),
            FadeIn(matrix_label, shift=0.25 * UP)
        )
        self.play(ShowCreation(in_vect_rect))
        self.play(
            ShowCreation(in_vect_path),
            TransformFromCopy(fluff_emb, in_vect, path_arc=-20 * DEGREES),
            run_time=2
        )

        # Show matrix product
        eq, rhs = show_matrix_vector_product(self, matrix, in_vect)
        self.wait()

        # Position value vect
        value_rect = SurroundingRectangle(rhs)
        value_rect.set_stroke(RED, 2)
        value_label = Text("Value")
        value_label.next_to(value_rect, RIGHT)
        value_label.set_color(RED)
        value_label.set_backstroke()
        self.fix_new_entries_in_frame = True
        self.play(
            ShowCreation(value_rect),
            FadeIn(value_label, lag_ratio=0.1)
        )
        self.wait()

        value_label2 = value_label.copy()
        value_label2.set_backstroke(BLACK, 5)
        value_label2.scale(1.5)
        value_label2.next_to(diff_vect, UP, MED_SMALL_BUFF)
        value_label2.unfix_from_frame()

        self.fix_new_entries_in_frame = False
        self.play(
            frame.animate.reorient(29, -2, 0, (-7.48, 1.91, 1.21), 11.89),
            FadeInFromPoint(value_label2, np.array([-4, -5, 0])),
            TransformFromCopy(value_rect, diff_vect),
            run_time=2
        )
        self.wait()

        # Show blue
        blue_group.target = blue_group.generate_target()
        blue_group.target[0].set_stroke(opacity=1)
        blue_group.target[0].set_fill(opacity=0.2)
        blue_group.target[1:].set_opacity(1)
        blue_group.target.shift(0.2 * LEFT)

        blue_path = self.get_top_vect_to_low_vect_path(blue_group.target, in_vect, BLUE)
        blue_emb = blue_group[3]
        blue_in_vect = blue_emb.copy().set_opacity(1)
        blue_in_vect.replace(in_vect)

        self.fix_new_entries_in_frame = True
        self.play(
            MoveToTarget(blue_group),
            LaggedStartMap(FadeOut, VGroup(
                in_vect_path, in_vect_rect,
                rhs, value_rect, value_label,
                value_label2,
            )),
            run_time=1
        )
        self.play(
            TransformFromCopy(blue_emb, blue_in_vect),
            ShowCreation(blue_path),
            FadeOut(in_vect, 3 * DOWN),
            run_time=1.5
        )
        eq, rhs2 = show_matrix_vector_product(self, matrix, blue_in_vect)

        # Show in diagram
        diff2 = Arrow(
            with_fluffy_vect[0].get_end(),
            with_blue_vect[0].get_end(),
            buff=0.05
        )
        diff2.set_flat_stroke(False)
        rhs_rect = SurroundingRectangle(rhs2)
        rhs_rect.set_stroke(RED, 2)

        self.fix_new_entries_in_frame = True
        self.play(ShowCreation(rhs_rect))
        self.fix_new_entries_in_frame = False
        self.add(diff2)
        self.play(
            TransformFromCopy(rhs_rect, diff2),
            FadeIn(diff2),
            frame.animate.reorient(-16, -3, 0, (-6.41, 2.78, 1.37), 13.21),
            TransformFromCopy(with_fluffy_vect[0], with_blue_vect[0]),
            FadeTransform(with_fluffy_vect[1].copy(), with_blue_vect[1]),
            run_time=2,
        )
        frame.add_ambient_rotation(2 * DEGREES)
        self.wait(8)


    def get_top_vect_to_low_vect_path(self, top_vect, low_vect, color, top_buff=0.1, low_buff=0.2, bezier_factor=1.5):
        result = CubicBezier(
            top_vect.get_bottom() + top_buff * DOWN,
            top_vect.get_bottom() + bezier_factor * DOWN,
            low_vect.get_top() + bezier_factor * UP,
            low_vect.get_top() + low_buff * UP,
        )
        result.set_stroke(color, 3)
        return result

    def get_labeled_vector(self, axes, coords, color, image_name, image_height=1.0):
        vect = Arrow(axes.get_origin(), axes.c2p(*coords), buff=0)
        vect.set_color(color)
        image = ImageMobject(image_name)
        image.set_height(image_height)
        image.next_to(vect.get_end(), UP, MED_SMALL_BUFF)

        return Group(vect, image)

    def add(self, *mobjects):
        if self.fix_new_entries_in_frame:
            for mob in mobjects:
                mob.fix_in_frame()
        super().add(*mobjects)


class CountMatrixParameters(InteractiveScene):
    count_font_size = 36

    def construct(self):
        # Add three matrices
        d_embed = 12_288
        d_key = 128
        key_mat_shape = (5, 10)

        que_mat = WeightMatrix(shape=key_mat_shape)
        key_mat = WeightMatrix(shape=key_mat_shape)
        val_mat = WeightMatrix(shape=(key_mat_shape[1], key_mat_shape[1]))
        matrices = VGroup(que_mat, key_mat, val_mat)
        for matrix in matrices:
            matrix.set_max_width(4)

        matrices.arrange(DOWN, buff=0.75)

        colors = [YELLOW, TEAL, RED]

        titles = VGroup(Text("Query"), Text("Key"), Text("Value"))
        que_title, key_title, val_title = titles
        titles.arrange(DOWN, aligned_edge=LEFT)
        titles.next_to(matrices, LEFT, LARGE_BUFF)
        for title, matrix, color in zip(titles, matrices, colors):
            title.match_y(matrix)
            title.set_color(color)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.25 * LEFT, lag_ratio=0.5),
            LaggedStart(
                (FadeIn(matrix, lag_ratio=1e-2)
                for matrix in matrices),
                lag_ratio=0.5,
            )
        )
        self.wait()

        # Data animations
        change_anims = [RandomizeMatrixEntries(mat) for mat in matrices]
        highlight_anims = [
            LaggedStartMap(FlashUnder, mat.get_entries(), lag_ratio=5e-3, stroke_width=1)
            for mat in matrices
        ]

        self.play(
            LaggedStart(highlight_anims, lag_ratio=0.2),
            LaggedStart(change_anims, lag_ratio=0.2),
            run_time=3
        )

        # Ask about total number of parameters
        rects = VGroup(
            SurroundingRectangle(entry, buff=0.025)
            for matrix in matrices
            for entry in matrix.get_entries()
        )
        rects.set_stroke(WHITE, 1)
        question = Text("How many\\nparameters?")
        question.next_to(matrices, RIGHT, LARGE_BUFF)

        self.play(
            ShowCreation(rects, lag_ratio=5e-3, run_time=2),
            Write(question)
        )
        self.play(FadeOut(rects))
        self.wait()

        # Make room to count query/key
        value_group = VGroup(val_title, val_mat)
        value_group.save_state()
        qk_mats = matrices[:2]
        qk_mats.target = qk_mats.generate_target()
        qk_mats.target.arrange(RIGHT, buff=3.0)
        qk_mats.target.move_to(DR)

        self.play(
            FadeOut(question, DR),
            value_group.animate.scale(0.25).to_corner(DR).fade(0.25),
            MoveToTarget(qk_mats),
            que_title.animate.next_to(qk_mats.target[0], UP, buff=2.0),
            key_title.animate.next_to(qk_mats.target[1], UP, buff=2.0),
        )

        # Count up query and key
        que_col_count = self.show_column_count(que_mat, d_embed)
        key_col_count = self.show_column_count(key_mat, d_embed)
        self.wait()
        que_row_count = self.show_row_count(que_mat, d_key)
        key_row_count = self.show_row_count(key_mat, d_key)
        self.wait()

        que_product = self.show_product(
            que_col_count, que_row_count,
            added_anims=[que_title.animate.shift(UP)]
        )
        key_product = self.show_product(
            key_col_count, key_row_count,
            added_anims=[key_title.animate.shift(UP)]
        )
        self.wait()

        # Pull up the value matrix
        qk_titles = titles[:2]
        qk_titles.target = qk_titles.generate_target()
        qk_titles.target.arrange(DOWN, buff=2.0, aligned_edge=LEFT)
        qk_titles.target.to_corner(UL)
        qk_titles.target.scale(0.5, about_edge=UL)

        qk_mats.target = qk_mats.generate_target()

        qk_rhss = VGroup(que_product[-1], key_product[-1]).copy()
        qk_rhss.target = qk_rhss.generate_target()

        for mat, title, rhs in zip(qk_mats.target, qk_titles.target, qk_rhss.target):
            rhs.scale(0.5)
            mat.scale(0.5)
            rhs.next_to(title, DOWN, SMALL_BUFF, aligned_edge=LEFT)
            mat.next_to(VGroup(title, rhs), RIGHT, buff=MED_LARGE_BUFF)

        self.play(
            MoveToTarget(qk_titles),
            MoveToTarget(qk_mats),
            MoveToTarget(qk_rhss),
            FadeOut(VGroup(
                que_product, key_product,
                que_col_count, que_row_count,
                key_col_count, key_row_count,
            ), shift=0.5 * UL, lag_ratio=1e-3, time_span=(0, 1.0)),
            value_group.animate.restore().arrange(DOWN, buff=1.0).move_to(2.0 * RIGHT + 0.5 * DOWN),
            run_time=2,
        )
        self.wait()

        # Count up current value
        in_vect = NumericEmbedding(length=key_mat_shape[1])
        in_vect.match_height(val_mat)
        in_vect.next_to(val_mat, RIGHT, SMALL_BUFF)

        val_col_count = self.show_column_count(
            val_mat, d_embed,
            added_anims=[val_title.animate.shift(UP)]
        )
        self.play(FadeIn(in_vect))
        eq, rhs = show_matrix_vector_product(self, val_mat, in_vect)
        val_row_count = self.show_row_count(val_mat, d_embed)
        self.wait()
        val_product = self.show_product(
            val_col_count, val_row_count,
            added_anims=[val_title.animate.shift(UP)]
        )
        self.wait()

        # Compare the two
        frame = self.frame
        q_group, k_group = qk_groups = VGroup(
            VGroup(*trip)
            for trip in zip(qk_mats, qk_titles, qk_rhss)
        )
        for group, y in zip(qk_groups, [+1.25, -1.25]):
            group.save_state()
            group.target = group.generate_target()
            group.target.scale(2)
            group.target.next_to(val_mat, LEFT, buff=2.5)
            group.target.set_y(y)

        self.play(
            frame.animate.reorient(0, 0, 0, (-1.58, 0.02, 0.0), 9.22),
            LaggedStartMap(MoveToTarget, qk_groups),
        )
        self.wait()

        # Circle both
        val_rhs_rect = SurroundingRectangle(val_product[-1])
        val_rhs_rect.set_stroke(RED_B, 3)
        qk_rhs_rects = VGroup(
            SurroundingRectangle(rhs) for rhs in qk_rhss
        )
        qk_rhs_rects[0].set_stroke(YELLOW, 3)
        qk_rhs_rects[1].set_stroke(TEAL, 3)

        big_rect = FullScreenFadeRectangle()
        big_rect.scale(2)
        big_rect.set_fill(opacity=0.5)
        val_rhs_copy = val_product[-1].copy()
        qk_rhs_copies = qk_rhss.copy()

        self.add(big_rect, val_rhs_copy)
        self.play(
            FadeIn(big_rect),
            ShowCreation(val_rhs_rect)
        )
        self.wait()
        self.play(
            TransformFromCopy(VGroup(val_rhs_rect), qk_rhs_rects),
            FadeIn(qk_rhs_copies)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, VGroup(
                big_rect, qk_rhs_copies, val_rhs_copy,
                qk_rhs_rects, val_rhs_rect
            ))
        )

        # Cross out
        cross = Cross(val_product, stroke_width=[0, 12, 0]).scale(1.1)
        self.play(LaggedStart(
            FadeOut(qk_groups, 2 * UR, scale=0.5),
            ShowCreation(cross),
            frame.animate.set_height(FRAME_HEIGHT).move_to(RIGHT),
            run_time=2,
            lag_ratio=0.1
        ))
        self.wait()
        self.play(FadeOut(val_product), FadeOut(cross))

        # Factor out
        val_down_mat = WeightMatrix(shape=key_mat_shape)
        val_up_mat = WeightMatrix(shape=(key_mat_shape[1], 4))
        val_down_mat.match_width(val_mat)
        val_up_mat.match_height(in_vect)

        val_down_mat.move_to(val_mat, RIGHT)
        val_up_mat.next_to(val_down_mat, LEFT, SMALL_BUFF)

        self.remove(val_mat)
        self.play(
            TransformFromCopy(val_mat.get_brackets(), val_down_mat.get_brackets()),
            TransformFromCopy(val_mat.get_columns(), val_down_mat.get_columns()),
            TransformFromCopy(val_mat.get_brackets(), val_up_mat.get_brackets()),
            TransformFromCopy(val_mat.get_rows(), val_up_mat.get_rows()),
            val_col_count.animate.next_to(val_down_mat, UP, SMALL_BUFF),
            val_row_count.animate.next_to(val_up_mat, LEFT, SMALL_BUFF),
        )
        self.add(val_down_mat)
        self.wait()

        # Circle the full linear map
        big_rect = SurroundingRectangle(VGroup(val_row_count, val_col_count))
        big_rect.round_corners(radius=0.25)
        big_rect.set_stroke(RED_B, 2)
        linear_map_words = Text("Linear map")
        linear_map_words.next_to(big_rect, UP)
        linear_map_words.set_color(RED_B)

        in_label, out_label = [
            VGroup(Text(text), Integer(d_embed))
            for text in ["d_input", "d_output"]
        ]
        for label, array, shift in [(in_label, in_vect, LEFT), (out_label, rhs, RIGHT)]:
            label.arrange(DOWN)
            label.scale(0.65)
            label.next_to(array, UP, buff=LARGE_BUFF)
            label.shift(0.25 * shift)
            arrow = Arrow(label, array)
            label.add(arrow)

        self.play(
            FadeIn(big_rect),
            FadeTransform(val_title, linear_map_words),
        )
        self.wait()
        self.play(FadeIn(in_label, lag_ratio=0.1))
        self.play(FadeIn(out_label, lag_ratio=0.1))
        self.wait(2)

        # Show the value_down map
        val_down_group = VGroup(val_down_mat, val_col_count)
        val_up_group = VGroup(val_up_mat, val_row_count)
        val_down_group.save_state()
        val_up_group.save_state()

        small_row_count = self.show_row_count(
            val_down_mat, d_key,
            added_anims=[val_up_group.animate.scale(0.5).to_edge(LEFT, buff=1.25).fade(0.5)]
        )
        self.wait()
        self.play(frame.animate.set_y(0.5))
        self.wait()

        value_down_rect = SurroundingRectangle(
            VGroup(small_row_count, val_down_mat, val_col_count)
        )
        value_down_rect.round_corners(radius=0.25)
        value_down_rect.set_stroke(RED_B, 2)
        value_down_title = TexText(R"Value$_\\downarrow$")
        value_down_title.set_fill(RED_B)
        value_down_title.next_to(val_down_mat, DOWN)

        self.remove(big_rect)
        self.play(
            TransformFromCopy(big_rect, value_down_rect),
            FadeOut(linear_map_words),
            FadeIn(value_down_title, DOWN)
        )
        self.wait()

        # Show value_up map
        small_row_count.target = small_row_count.generate_target()
        small_row_count.target.rotate(-PI / 2)
        small_row_count.target[1].rotate(PI / 2)
        small_row_count.target[0].stretch_to_fit_width(val_up_group.saved_state[0].get_width())
        small_row_count.target[1].next_to(small_row_count.target[0], UP, SMALL_BUFF)
        small_row_count.target.next_to(val_up_group.saved_state[0], UP, SMALL_BUFF)
        big_rect.set_height(3.9, stretch=True)
        big_rect.align_to(VGroup(val_down_mat, val_up_group.saved_state), DR)
        big_rect.shift(0.8 * DOWN + 0.05 * RIGHT)
        linear_map_words.next_to(big_rect, UP)

        value_up_title = TexText(R"Value$_\\uparrow$")
        value_up_title.set_fill(RED_B)
        value_up_title.next_to(val_up_group.saved_state[0], DOWN)

        self.play(LaggedStart(
            val_down_group.animate.fade(0.5),
            value_down_title.animate.fade(0.5),
            ReplacementTransform(value_down_rect, big_rect),
            Restore(val_up_group),
            MoveToTarget(small_row_count),
            FadeIn(linear_map_words, shift=0.5 * UP),
            run_time=2,
        ))
        val_up_group.add(small_row_count)
        self.wait()
        self.play(TransformFromCopy(value_down_title, value_up_title))
        self.wait()

        # Low rank label
        low_rank_words = TexText("\`\`Low rank'' transformation")
        low_rank_words.next_to(big_rect, UP)
        low_rank_words.shift(0.5 * LEFT)
        self.play(
            val_down_group.animate.set_fill(opacity=1),
            value_down_title.animate.set_fill(opacity=1),
            FadeTransform(linear_map_words, low_rank_words)
        )
        self.wait()

    def scrap(self):
        # Label the value matrix
        tiny_buff = 0.025
        value_rect = SurroundingRectangle(val_down_group, buff=tiny_buff)
        value_rect.stretch(1.2, 1)
        value_rect.round_corners(0.1)
        value_rect.set_stroke(RED, 3)
        value_arrow = Vector(DOWN)
        value_arrow.match_color(value_rect)
        value_arrow.next_to(value_rect, UP, SMALL_BUFF)

        val_up_group.save_state()
        out_rect = SurroundingRectangle(val_up_group, buff=tiny_buff)
        out_rect.set_height(big_rect.get_height() - SMALL_BUFF, stretch=True)
        out_rect.match_y(big_rect)
        out_rect.round_corners(0.1)
        out_rect.set_stroke(PINK, 3)
        out_arrow = Vector(0.5 * DOWN)
        out_arrow.next_to(out_rect, UP, SMALL_BUFF)
        out_arrow.match_color(out_rect)
        output_title = TexText("Output$^{*}$")
        output_title.match_color(out_rect)
        output_title.next_to(out_arrow, UP, SMALL_BUFF)


        self.play(LaggedStart(
            Restore(val_down_group),
            LaggedStartMap(FadeOut, VGroup(in_label, out_label)),
            TransformFromCopy(big_rect, value_rect),
            FadeOut(linear_map_words),
            val_title.animate.next_to(value_arrow, UP, SMALL_BUFF),
            FadeIn(value_arrow, shift=DOWN),
            val_up_group.animate.fade(0.5),
        ))
        self.wait()
        self.play(LaggedStart(
            TransformFromCopy(big_rect, out_rect),
            TransformFromCopy(value_arrow, out_arrow),
            FadeTransform(val_title.copy(), output_title),
            Restore(val_up_group),
        ))
        self.wait()

    def show_column_count(self, matrix, count, added_anims=[]):
        cols = matrix.get_columns()
        col_rects = VGroup(SurroundingRectangle(cols[0], buff=0).match_x(col) for col in cols)
        col_rects.set_stroke(WHITE, 1, 0.5)
        col_rects.set_fill(GREY_D, 0.5)
        top_brace = Brace(col_rects, UP, buff=SMALL_BUFF)
        count_mob = Integer(count, font_size=self.count_font_size)
        count_mob.next_to(top_brace, UP)

        self.play(
            GrowFromCenter(top_brace),
            CountInFrom(count_mob, 0),
            FadeIn(col_rects, lag_ratio=0.25),
            *added_anims,
        )
        self.play(FadeOut(col_rects))
        return VGroup(top_brace, count_mob)

    def show_row_count(self, matrix, count, added_anims=[]):
        rows = matrix.get_rows()
        row_rects = VGroup(SurroundingRectangle(rows[0], buff=0).match_y(row) for row in rows)
        row_rects.set_stroke(WHITE, 1, 0.5)
        row_rects.set_fill(GREY_D, 0.5)
        left_brace = Brace(matrix, LEFT, buff=SMALL_BUFF)
        count_mob = Integer(count, font_size=self.count_font_size)
        count_mob.next_to(left_brace, LEFT)

        self.play(
            GrowFromCenter(left_brace),
            CountInFrom(count_mob, 0),
            FadeIn(row_rects, lag_ratio=0.25),
            *added_anims,
        )
        self.play(FadeOut(row_rects))
        return VGroup(left_brace, count_mob)

    def show_product(self, col_count, row_count, added_anims=[]):
        col_dec = col_count[1]
        row_dec = row_count[1]
        prod_dec = Integer(
            col_dec.get_value() * row_dec.get_value(),
            font_size=self.count_font_size
        )

        equation = VGroup(
            row_dec.copy(),
            Tex(R"\\times", font_size=self.count_font_size),
            col_dec.copy(),
            Tex(R"=", font_size=self.count_font_size),
            prod_dec
        )
        equation.arrange(RIGHT,buff=SMALL_BUFF)
        for index in [0, 2]:
            equation[index].align_to(equation[4], UP)
        equation.next_to(col_dec, UP, buff=1.0)

        self.play(
            TransformFromCopy(row_dec, equation[0]),
            FadeIn(equation[1]),
            TransformFromCopy(col_dec, equation[2]),
            FadeIn(equation[3]),
            *added_anims
        )
        self.play(
            FadeTransform(equation[0].copy(), equation[4]),
            FadeTransform(equation[2].copy(), equation[4]),
        )
        self.add(equation)
        return equation


class LowRankTransformation(InteractiveScene):
    def construct(self):
        # Add three sets of axes
        frame = self.frame
        frame.set_field_of_view(10 * DEGREES)

        all_axes = VGroup(
            self.get_3d_axes(),
            self.get_2d_axes(),
            self.get_3d_axes(),
        )
        all_axes.arrange(RIGHT, buff=2.0)
        all_axes.set_width(FRAME_WIDTH - 2)
        all_axes.move_to(0.5 * DOWN)
        dim_labels = VGroup(
            Text("12,288 dims"),
            Text("128 dims"),
            Text("12,288 dims"),
        )
        dim_labels.scale(0.75)
        dim_labels.set_fill(GREY_A)
        for label, axes in zip(dim_labels, all_axes):
            label.next_to(axes, UP, buff=MED_LARGE_BUFF)

        map_arrows = Tex(R"\\rightarrow", font_size=96).replicate(2)
        map_arrows.set_color(YELLOW)
        for arrow, vect in zip(map_arrows, [LEFT, RIGHT]):
            arrow.next_to(all_axes[1], vect, buff=0.5)

        axes_group = VGroup(all_axes, dim_labels)
        self.add(axes_group)
        self.add(map_arrows)

        # Add vectors
        all_coords = [
            (4, 2, 1),
            (2, 3),
            (-3, 3, -2),
        ]
        colors = [BLUE, RED_B, RED_C]
        vects = VGroup(
            Arrow(axes.get_origin(), axes.c2p(*coords), buff=0, stroke_color=color)
            for axes, coords, color in zip(all_axes, all_coords, colors)
        )

        self.add(vects[0])
        for v1, v2 in zip(vects, vects[1:]):
            self.play(TransformFromCopy(v1, v2))

        for axes, vect in zip(all_axes, vects):
            axes.add(vect)
        for axes in all_axes[0::2]:
            axes.add_updater(lambda m, dt: m.rotate(2 * dt * DEGREES, axis=m.y_axis.get_vector()))
        self.wait(3)

        # Add title
        big_rect = SurroundingRectangle(axes_group, buff=0.5)
        big_rect.round_corners(radius=0.5)
        big_rect.set_stroke(RED_B, 2)
        title = Text("Low-rank transformation", font_size=72)
        title.next_to(big_rect, UP, buff=MED_LARGE_BUFF)

        self.play(
            ShowCreation(big_rect),
            FadeIn(title, shift=0.25 * UP)
        )
        self.wait(5)


    def get_3d_axes(self, height=3):
        result = ThreeDAxes((-4, 4), (-4, 4), (-4, 4))
        result.set_height(height)
        result.rotate(20 * DEGREES, DOWN)
        result.rotate(5 * DEGREES, RIGHT)
        return result

    def get_2d_axes(self, height=2):
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            faded_line_ratio=0,
            background_line_style=dict(
                stroke_color=GREY_B,
                stroke_width=1,
                stroke_opacity=0.5
            )
        )
        plane.set_height(height)
        return plane


class ThinkAboutOverallMap(InteractiveScene):
    def construct(self):
        # Test
        rect = Rectangle(6.5, 2.75)
        rect.round_corners(radius=0.5)
        rect.set_stroke(RED_B, 2)
        label = Text("Think about the\\noverall map")
        label.next_to(rect, UP, aligned_edge=LEFT)
        label.shift(0.5 * RIGHT)
        self.play(
            ShowCreation(rect),
            FadeIn(label, UP),
        )
        self.wait()


class CrossAttention(InteractiveScene):
    def construct(self):
        # Show both
        en_tokens = self.get_words("I do not want to pet it")
        fr_tokens = self.get_words("Je ne veux pas le caresser", hue_range=(0.2, 0.3))
        phrases = VGroup(en_tokens, fr_tokens)
        phrases.arrange(DOWN, buff=2.0)
        self.play(LaggedStartMap(FadeIn, en_tokens, scale=2, lag_ratio=0.25))
        self.wait()
        self.play(LaggedStartMap(FadeIn, fr_tokens, scale=2, lag_ratio=0.25))
        self.wait()

        # Create attention pattern
        unnormalized_pattern = [
            [3, 0, 0, 0, 0, 0],
            [0, 1, 1.3, 1, 0, 0],
            [0, 3, 0, 3, 0, 0],
            [0, 0, 3, 0, 0, 0],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 0, 3],
            [0, 0, 0, 0, 3, 0],
        ]
        attention_pattern = np.array([
            softmax(col) for col in unnormalized_pattern
        ]).T

        # Show connections
        lines = VGroup()
        for n, row in enumerate(attention_pattern.T):
            for k, value in enumerate(row):
                line = Line(en_tokens[n].get_bottom(), fr_tokens[k].get_top(), buff=0)
                line.set_stroke(
                    color=[
                        en_tokens[n][0].get_color(),
                        fr_tokens[k][0].get_color(),
                    ],
                    width=3,
                    opacity=value,
                )
                lines.add(line)

        self.play(ShowCreation(lines, lag_ratio=0.01, run_time=2))
        self.wait(2)
        self.play(FadeOut(lines))

        # Create grid
        grid = Square().get_grid(len(fr_tokens), len(en_tokens), buff=0)
        grid.stretch(1.2, 0)
        grid.set_stroke(GREY_B, 1)
        grid.set_height(5.0)
        grid.to_edge(DOWN, buff=SMALL_BUFF)
        grid.set_x(1)

        # Create qk symbols
        q_sym_generator = self.get_symbol_generator(R"\\vec{\\textbf{Q}}_0", color=YELLOW)
        k_sym_generator = self.get_symbol_generator(R"\\vec{\\textbf{K}}_0", color=TEAL)
        e_sym_generator = self.get_symbol_generator(R"\\vec{\\textbf{E}}_0", color=GREY_B)
        f_sym_generator = self.get_symbol_generator(R"\\vec{\\textbf{F}}_0", color=BLUE)

        q_syms = VGroup(q_sym_generator(n + 1) for n in range(len(en_tokens)))
        k_syms = VGroup(k_sym_generator(n + 1) for n in range(len(fr_tokens)))
        e_syms = VGroup(e_sym_generator(n + 1) for n in range(len(en_tokens)))
        f_syms = VGroup(f_sym_generator(n + 1) for n in range(len(fr_tokens)))
        VGroup(q_syms, k_syms, e_syms, f_syms).scale(0.65)

        for q_sym, e_sym, square in zip(q_syms, e_syms, grid):
            q_sym.next_to(square, UP, SMALL_BUFF)
            e_sym.next_to(q_sym, UP, buff=0.65)

        for k_sym, f_sym, square in zip(k_syms, f_syms, grid[::len(en_tokens)]):
            k_sym.next_to(square, LEFT, SMALL_BUFF)
            f_sym.next_to(k_sym, LEFT, buff=0.75)

        q_arrows = VGroup(Arrow(*pair, buff=0.1) for pair in zip(e_syms, q_syms))
        k_arrows = VGroup(Arrow(*pair, buff=0.1) for pair in zip(f_syms, k_syms))
        e_arrows = VGroup(Vector(0.4 * DOWN).next_to(e_sym, UP, SMALL_BUFF) for e_sym in e_syms)
        f_arrows = VGroup(Vector(0.5 * RIGHT).next_to(f_sym, LEFT, SMALL_BUFF) for f_sym in f_syms)
        arrows = VGroup(q_arrows, k_arrows, e_arrows, f_arrows)
        arrows.set_color(GREY_B)

        wq_syms = VGroup(
            Tex("W_Q", font_size=20, fill_color=YELLOW).next_to(arrow, RIGHT, buff=0.1)
            for arrow in q_arrows
        )
        wk_syms = VGroup(
            Tex("W_K", font_size=20, fill_color=TEAL).next_to(arrow, UP, buff=0.1)
            for arrow in k_arrows
        )

        # Move tokens into place
        en_tokens.target = en_tokens.generate_target()
        fr_tokens.target = fr_tokens.generate_target()
        for token, arrow in zip(en_tokens.target, e_arrows):
            token.next_to(arrow, UP, SMALL_BUFF)
        for token, arrow in zip(fr_tokens.target, f_arrows):
            token.next_to(arrow, LEFT, SMALL_BUFF)
        self.play(
            MoveToTarget(en_tokens),
            MoveToTarget(fr_tokens),
        )
        self.play(
            LaggedStartMap(GrowArrow, e_arrows),
            LaggedStartMap(GrowArrow, f_arrows),
            LaggedStartMap(FadeIn, e_syms, shift=0.25 * DOWN),
            LaggedStartMap(FadeIn, f_syms, shift=0.25 * RIGHT),
            lag_ratio=0.25,
            run_time=1.5,
        )
        self.play(
            LaggedStartMap(GrowArrow, q_arrows),
            LaggedStartMap(GrowArrow, k_arrows),
            LaggedStartMap(FadeIn, wq_syms, shift=0.25 * DOWN),
            LaggedStartMap(FadeIn, wk_syms, shift=0.25 * RIGHT),
            LaggedStartMap(FadeIn, q_syms, shift=0.5 * DOWN),
            LaggedStartMap(FadeIn, k_syms, shift=0.5 * RIGHT),
            lag_ratio=0.25,
            run_time=1.5,
        )
        self.play(FadeIn(grid, lag_ratio=1e-2), run_time=2)
        self.wait()

        # Show dot products
        dot_prods = VGroup()
        for q_sym in q_syms:
            for k_sym in k_syms:
                dot = Tex(".")
                dot.match_x(q_sym)
                dot.match_y(k_sym)
                dot_prod = VGroup(q_sym.copy(), dot, k_sym.copy())
                dot_prod.target = dot_prod.generate_target()
                dot_prod.target.arrange(RIGHT, buff=SMALL_BUFF)
                dot_prod.target.scale(0.7)
                dot_prod.target.move_to(dot)
                dot.set_opacity(0)
                dot_prods.add(dot_prod)

        self.play(
            LaggedStartMap(MoveToTarget, dot_prods, lag_ratio=0.01),
            run_time=3
        )
        self.wait()

        # Show dots
        dots = VGroup()
        for square, value in zip(grid, attention_pattern.flatten()):
            dot = Dot(radius=value * 0.4)
            dot.set_fill(GREY_B, 1)
            dot.move_to(square)
            dots.add(dot)


        self.play(
            LaggedStartMap(GrowFromCenter, dots, lag_ratio=1e-2),
            dot_prods.animate.set_fill(opacity=0.2).set_anim_args(lag_ratio=1e-3),
            run_time=4
        )
        self.wait()




        pass

    def get_words(self, text, hue_range=(0.5, 0.6)):
        sent = Text(text)
        tokens = break_into_words(sent)
        rects = get_piece_rectangles(
            tokens, hue_range=hue_range,
            # h_buff=0, leading_spaces=True
        )
        return VGroup(VGroup(*pair) for pair in zip(rects, tokens))

    def get_symbol_generator(self, raw_tex, subsrc="0", color=WHITE):
        template = Tex(raw_tex)
        template.set_color(color)
        subscr = template.make_number_changeable(subsrc)

        def get_sym(number):
            subscr.set_value(number)
            return template.copy()

        return get_sym


class CarCrashedExample(InteractiveScene):
    def construct(self):
        # Add sentence
        sentence = Text("... when suddenly they crashed the car into a tree ...")
        words = break_into_words(sentence)
        rects = get_piece_rectangles(words)
        word_groups = VGroup(VGroup(*pair) for pair in zip(rects, words))

        car = word_groups[6]
        crashed = VGroup(*it.chain(*(wg[1] for wg in word_groups[3:6])))
        arrow = Vector(UP).next_to(car, UP, SMALL_BUFF)

        self.play(LaggedStartMap(FadeIn, word_groups, shift=0.25 * UP, lag_ratio=0.25))
        self.play(
            word_groups[:3].animate.fade(0.5),
            word_groups[7:].animate.fade(0.5),
            FadeIn(arrow),
        )
        self.wait()

        # Influence
        self.play(ContextAnimation(car, crashed, direction=DOWN, run_time=5))
        self.wait()


class TwoHarrysExample(InteractiveScene):
    def construct(self):
        # Test
        s1, s2 = sentences = VGroup(
            break_into_words(Text("... " + " ... ".join(words)))
            for words in [
                ("wizard", "Hogwarts", "Hermione", "Harry"),
                ("Queen", "Sussex", "William", "Harry"),
            ]
        )
        sentences.arrange(DOWN, buff=2.0, aligned_edge=RIGHT)
        sentences.to_edge(LEFT)

        def context_anim(group):
            self.play(
                ContextAnimation(
                    group[-1],
                    VGroup(*it.chain(*group[1:-1:2])),
                    direction=DOWN,
                    path_arc=PI / 4,
                    run_time=5,
                    lag_ratio=0.025,
                )
            )

        self.add(s1)
        context_anim(s1)
        self.wait()
        self.play(FadeTransformPieces(s1.copy(), s2))
        context_anim(s2)


class ManyTypesOfUpdates(InteractiveScene):
    def construct(self):
        # Add matrices
        shapes = [(4, 8), (4, 8), (8, 4), (4, 8)]
        names = ["W_Q", "W_K", R"\\uparrow W_V", R"\\downarrow W_V"]
        colors = [YELLOW, TEAL, RED_B, RED_C]

        matrices = VGroup(
            WeightMatrix(shape=shape)
            for shape in shapes
        )
        buff_ratio = 0.35
        matrices.arrange(RIGHT, buff=matrices[0].get_width() * buff_ratio)
        matrices[-1].next_to(matrices[-2], RIGHT, buff=matrices[-2].get_width() * 0.1)
        matrices.center()
        matrices.set_width(FRAME_WIDTH - 2)
        matrices.to_edge(UP, buff=1.0)
        titles = VGroup(
            Tex(name).set_color(color).match_x(mat).to_edge(UP, buff=MED_SMALL_BUFF)
            for name, color, mat in zip(names, colors, matrices)
        )
        for title in titles[2:]:
            title[0].next_to(title[1], LEFT, buff=0.5 * SMALL_BUFF)

        self.add(matrices, titles)

        # Add phrase
        phrase = Text("John hit the brakes sharply, they screeched loudly, and he jolted forward.")
        raw_words = break_into_words(phrase)
        rects = get_piece_rectangles(raw_words)
        rects.fade(0.5)
        words = VGroup(VGroup(*pair) for pair in zip(rects, raw_words))
        words.set_width(FRAME_WIDTH - 1)
        words.center().set_y(-2)

        self.add(words)

        labels = index_labels(words)
        labels.shift(0.5 * DOWN)

        # Set up association types
        attention_types = [
            (
                "Adverb to verb",
                [
                    (1, 4, 1.0), 
                    (6, 7, 1.0), 
                ]
            ),
            (
                "Subject to verb",
                [
                    (0, 1, 1.0),
                    (3, 6, 0.5),
                    (5, 6, 0.5),
                    (0, 10, 0.5),
                    (9, 10, 0.5),
                ],
            ),
            (
                "Antecedent to pronoun",
                [
                    (0, 9, 1.0),
                    (3, 5, 1.0),
                ]
            ),
            (
                "Related to the subject",
                [
                    (0, 1, 0.25),
                    (0, 3, 0.25),
                    (0, 9, 0.2),
                    (0, 10, 0.2),
                    (0, 11, 0.2),
                ]
            ),
            (
                "Related to the object",
                [
                    (3, 4, 0.2),
                    (3, 5, 0.5),
                    (3, 6, 0.35),
                    (3, 7, 0.2),
                ]
            ),
        ]

        # Animate
        last_group = VGroup()
        for description, connections in attention_types:
            desc = Text(description)
            desc.center()
            connections = VGroup(
                Line(
                    words[i].get_top(),
                    words[j].get_top(),
                    path_arc=-PI / 2,
                    stroke_color=random_bright_color(
                        hue_range=(0.3, 0.5),
                        luminance_range=(0.5, 0.7),
                    ),
                    stroke_opacity=strength**0.5,
                )
                for (i, j, strength) in connections
            )
            connections.set_stroke(width=(0, 5, 5, 5, 0))
            connections.shuffle()
            self.play(
                FadeOut(last_group),
                # FadeIn(desc, shift=0.25 * UP),
                ShowCreation(connections, lag_ratio=0.25, run_time=0.5 * len(connections)),
                LaggedStart(
                    (self.get_matrix_update_anim(mat)
                    for mat in matrices),
                    lag_ratio=0.15,
                ),
                LaggedStart(
                    (VShowPassingFlash(
                        line.copy().insert_n_curves(100).set_stroke(width=10),
                        time_width=2.0,
                        run_time=2,
                    )
                    for line in connections),
                    lag_ratio=0.1,
                )
            )
            self.wait(2)
            # last_group = VGroup(desc, connections)
            last_group = VGroup(connections)

    def get_matrix_update_anim(self, matrix):
        rects = VGroup(
            Underline(entry, buff=0.05)
            for entry in matrix.get_entries()
        )
        rects.set_stroke(WHITE, 1)
        return AnimationGroup(
            LaggedStartMap(ShowCreationThenFadeOut, rects, lag_ratio=1e-2),
            RandomizeMatrixEntries(matrix)
        )


class MultiHeadedAttention(InteractiveScene):
    def construct(self):
        # Mention head
        background_rect = FullScreenRectangle()
        single_title = Text("Single head of attention")
        multiple_title = Text("Multi-headed attention")
        titles = VGroup(single_title, multiple_title)
        for title in titles:
            title.scale(1.25)
            title.to_edge(UP)

        screen_rect = ScreenRectangle(height=6)
        screen_rect.set_fill(BLACK, 1)
        screen_rect.set_stroke(WHITE, 3)
        screen_rect.next_to(titles, DOWN, buff=0.5)

        head = single_title["head"][0]

        self.add(background_rect)
        self.add(single_title)
        self.add(screen_rect)
        self.wait()
        self.play(
            FlashAround(head, run_time=2),
            head.animate.set_color(YELLOW),
        )
        self.wait()

        # Change title
        kw = dict(path_arc=45 * DEGREES)
        self.play(
            FadeTransform(single_title["Single"], multiple_title["Multi-"], **kw),
            FadeTransform(single_title["head"], multiple_title["head"], **kw),
            FadeIn(multiple_title["ed"], 0.25 * RIGHT),
            FadeTransform(single_title["attention"], multiple_title["attention"], **kw),
            FadeOut(single_title["of"])
        )
        self.add(multiple_title)

        # Set up images
        n_heads = 15
        directory = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/attention/images/"
        heads = Group()
        for n in range(n_heads):
            im = ImageMobject(os.path.join(directory, f"AttentionPattern{n % 4 + 1}"))
            im.set_opacity(1)
            im.shift(0.01 * OUT)
            rect = SurroundingRectangle(im, buff=0)
            rect.set_fill(BLACK, 0.75)
            rect.set_stroke(WHITE, 1, 1)
            heads.add(Group(rect, im))

        # Show many parallel layers
        self.set_floor_plane("xz")
        frame = self.frame
        multiple_title.fix_in_frame()
        background_rect.fix_in_frame()

        heads.set_height(4)
        heads.arrange(OUT, buff=1.0)
        heads.move_to(DOWN)
        pre_head = ImageMobject(os.path.join(directory, f"AttentionPattern0"))

        pre_head.replace(screen_rect)
        pre_head = Group(screen_rect, pre_head)

        self.add(pre_head)
        self.wait()
        self.play(
            frame.animate.reorient(41, -12, 0, (-1.0, -1.42, 1.09), 12.90).set_anim_args(run_time=2),
            background_rect.animate.set_fill(opacity=0.75),
            FadeTransform(pre_head, heads[-1], time_span=(1, 2)),
        )
        self.play(
            frame.animate.reorient(48, -11, 0, (-1.0, -1.42, 1.09), 12.90),
            LaggedStart(
                (FadeTransform(heads[-1].copy(), image)
                for image in heads),
                lag_ratio=0.1,
                group_type=Group,
            ),
            run_time=4,
        )
        self.add(heads)
        self.wait()

        # Show matrices
        colors = [YELLOW, TEAL, RED, PINK]
        texs = ["W_Q", "W_K", R"\\downarrow W_V", R"\\uparrow W_V"]
        n_shown = 9
        wq_syms, wk_syms, wv_down_syms, wv_up_syms = sym_groups = VGroup(
            VGroup(
                Tex(tex + f"^{{({n})}}", font_size=36).next_to(image, UP, MED_SMALL_BUFF)
                for n, image in enumerate(heads[:-n_shown - 1:-1], start=1)
            ).set_color(color).set_backstroke(BLACK, 5)
            for tex, color in zip(texs, colors)
        )
        for group in wv_down_syms, wv_up_syms:
            for sym in group:
                sym[0].next_to(sym[1], LEFT, buff=0.025)
        dots = Tex(R"\\dots", font_size=90)
        dots.rotate(PI / 2, UP)
        sym_rot_angle = 70 * DEGREES
        for syms in sym_groups:
            syms.align_to(heads, LEFT)
            for sym in syms:
                sym.rotate(sym_rot_angle, UP)
            dots.next_to(syms, IN, buff=0.5)
            dots.match_style(syms[0])
            syms.add(dots.copy())

        up_shift = 0.75 * UP
        self.play(
            LaggedStartMap(FadeIn, wq_syms, shift=0.2 * UP, lag_ratio=0.25),
            frame.animate.reorient(59, -7, 0, (-1.62, 0.25, 1.29), 14.18),
            run_time=2,
        )
        for n in range(1, len(sym_groups)):
            self.play(
                LaggedStartMap(FadeIn, sym_groups[n], shift=0.2 * UP, lag_ratio=0.1),
                sym_groups[:n].animate.shift(up_shift),
                run_time=1,
            )
        self.wait()

        # Count up 96 heads
        depth = heads.get_depth()
        brace = Brace(Line(LEFT, RIGHT).set_width(0.5 * depth), UP).scale(2)
        brace_label = brace.get_text("96", font_size=96, buff=MED_SMALL_BUFF)
        brace_group = VGroup(brace, brace_label)
        brace_group.rotate(PI / 2, UP)
        brace_group.next_to(heads, UP, buff=MED_LARGE_BUFF)

        self.add(brace, brace_label, sym_groups)
        self.play(
            frame.animate.reorient(62, -6, 0, (-0.92, -0.08, -0.51), 14.18).set_anim_args(run_time=5),
            GrowFromCenter(brace),
            sym_groups.animate.set_fill(opacity=0.5).set_stroke(width=0),
            FadeIn(brace_label, 0.5 * UP, time_span=(0.5, 1.5)),
        )

        # Set up pure attention patterns, flattened
        for head in heads:
            n_rows = 8
            grid = Square().get_grid(n_rows, 1, buff=0).get_grid(1, n_rows, buff=0)
            grid.set_stroke(WHITE, 1, 0.5)
            grid.set_height(0.9 * head.get_height())
            grid.move_to(head)

            pattern = np.random.normal(0, 1, (n_rows, n_rows))
            for n in range(len(pattern[0])):
                pattern[:, n][n + 1:] = -np.inf
                pattern[:, n] = softmax(pattern[:, n])
            pattern = pattern.T

            dots = VGroup()
            for col, values in zip(grid, pattern):
                for square, value in zip(col, values):
                    if value < 1e-3:
                        continue
                    dot = Dot(radius=0.4 * square.get_height() * value)
                    dot.move_to(square)
                    dots.add(dot)
            dots.set_fill(GREY_B, 1)
            grid.add(dots)

            head.add(grid)
            head.target = head.generate_target()
            grid.set_opacity(0)
            head.target[1].set_opacity(0)
            head.target[0].set_opacity(1)

        n_shown = 4
        heads_target = Group(h.target for h in heads)
        heads_target.arrange(LEFT, buff=MED_LARGE_BUFF)
        heads_target.set_height(1.5)
        heads_target.to_edge(LEFT)
        heads_target.shift(2 * UP)
        heads_target[:-n_shown].set_opacity(0)

        # Set up key/query targets
        for group in sym_groups:
            group.generate_target()
        group_targets = [group.target for group in sym_groups]

        for head, wq, wk, wv_down, wv_up in zip(heads_target[::-1], *group_targets):
            for sym in [wq, wk, wv_down, wv_up]:
                sym.set_fill(opacity=1)
                sym.set_height(0.35)
                sym.rotate(-sym_rot_angle, UP)
            wk.next_to(head, UP, aligned_edge=LEFT)
            wq.next_to(wk, RIGHT, buff=0.35)
            wv_up.next_to(head, UP, aligned_edge=LEFT)
            wv_down.next_to(wv_up, RIGHT, buff=0.35)

        for group in group_targets:
            group[n_shown:].set_opacity(0)

        # Animate the flattening
        right_dots = Tex(R"\\dots", font_size=96)
        right_dots.move_to(heads_target[-n_shown - 1], LEFT).shift(MED_SMALL_BUFF * RIGHT)

        brace_group.target = brace_group.generate_target()
        brace_group.target.shift(UP)
        brace_group.target.set_opacity(0)

        self.play(
            frame.animate.reorient(0, 0, 0, ORIGIN, FRAME_HEIGHT).set_anim_args(run_time=2),
            FadeOut(multiple_title, UP),
            MoveToTarget(brace_group, remover=True),
            MoveToTarget(wq_syms, time_span=(0.5, 2)),
            MoveToTarget(wk_syms, time_span=(0.5, 2)),
            FadeOut(wv_down_syms),
            FadeOut(wv_up_syms),
            LaggedStartMap(MoveToTarget, heads, lag_ratio=0.01),
            Write(right_dots, time_span=(1.5, 2.0)),
        )

        att_patterns = VGroup(
            VGroup(head[0], head[2])
            for head in heads[:len(heads) - n_shown - 1:-1]
        )
        self.remove(heads)
        self.add(att_patterns)

        # Show value maps
        for group in [wv_up_syms, wv_down_syms]:
            group.become(group.target)

        value_diagrams = VGroup()
        arrows = VGroup()
        all_v_stacks = VGroup()
        for pattern, wv_up, wv_down, idx in zip(att_patterns, wv_up_syms, wv_down_syms, it.count(1)):
            rect = pattern[0].copy()

            v_stack = VGroup(Tex(Rf"\\vec{{\\textbf{{v}}}}_{n}") for n in range(1, 4))
            v_stack.arrange(DOWN, buff=LARGE_BUFF)
            v_stack.set_color(RED)
            plusses = VGroup()
            coefs = VGroup()
            for n, v_term in enumerate(v_stack):
                coef = Tex(f"w_{n + 1}")
                coef.next_to(v_term, LEFT, SMALL_BUFF)
                coef.set_fill(GREY_B)
                plus = Tex("+")
                plus.next_to(VGroup(coef, v_term), DOWN)
                plusses.add(plus)
                coefs.add(coef)
            dots = Tex(R"\\vdots")
            dots.next_to(plusses, DOWN)
            v_stack.add(coefs, plusses, dots)

            v_stacks = v_stack.replicate(4)
            v_stacks.arrange(RIGHT, buff=LARGE_BUFF)
            v_stacks.set_height(rect.get_height() * 0.85)
            v_stacks.set_fill(border_width=1)

            v_terms = VGroup(
                *(Tex(Rf"\\vec{{\\textbf{{v}}}}_{n}^{{({idx})}}") for n in range(1, 4)),
                Tex(R"\\dots")
            )
            v_terms[:3].set_color(RED)
            v_terms.arrange(RIGHT)
            v_terms.set_width(0.8 * rect.get_width())
            v_terms.move_to(rect)

            diagram = VGroup(rect, v_terms)
            diagram.to_edge(DOWN, buff=1.5)

            v_stacks.move_to(rect)
            all_v_stacks.add(v_stacks)

            VGroup(wv_up, wv_down).next_to(diagram, UP, buff=SMALL_BUFF, aligned_edge=LEFT)

            arrow = Arrow(pattern, diagram, buff=0.5)
            arrow.shift(0.25 * UP)

            value_diagrams.add(diagram)
            arrows.add(arrow)

        right_dots2 = right_dots.copy()

        self.play(
            LaggedStart(
                (FadeTransform(m1.copy(), m2)
                for m1, m2 in zip(att_patterns, value_diagrams)),
                lag_ratio=0.25,
                group_type=Group,
            ),
            LaggedStartMap(FadeIn, wv_up_syms, shift=DOWN, lag_ratio=0.25),
            LaggedStartMap(FadeIn, wv_down_syms, shift=DOWN, lag_ratio=0.25),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.25),
            right_dots2.animate.match_y(value_diagrams).set_anim_args(time_span=(1.0, 1.75)),
        )
        self.wait()

        self.play(
            LaggedStart(
                (Transform(VGroup(diagram[1]), v_stacks)
                for diagram, v_stacks in zip(value_diagrams, all_v_stacks)),
                lag_ratio=0.25,
                run_time=2
            )
        )
        self.remove(value_diagrams)
        new_diagrams = VGroup(
            VGroup(vd[0], stacks)
            for vd, stacks in zip(value_diagrams, all_v_stacks)
        )
        value_diagrams = new_diagrams
        self.add(value_diagrams)

        # Show sums
        index = 2
        rects = VGroup()
        delta_Es = VGroup()
        arrows = VGroup()
        for n, diagram in enumerate(value_diagrams, start=1):
            diagram.target = diagram.generate_target()
            stacks = diagram.target[1]
            stacks.set_opacity(0.5)
            stacks[index].set_opacity(1, border_width=1)
            rect = SurroundingRectangle(stacks[index], buff=0.05)

            arrow = Vector(0.5 * DOWN)
            arrow.set_color(BLUE)
            arrow.next_to(rect, DOWN, SMALL_BUFF)

            delta_E = Tex(Rf"\\Delta \\vec{{\\textbf{{E}}}}^{{({n})}}_i", font_size=36)
            delta_E.set_color(BLUE)
            delta_E.next_to(arrow, DOWN, SMALL_BUFF)

            rects.add(rect)
            arrows.add(arrow)
            delta_Es.add(delta_E)

        rects.set_stroke(BLUE, 2)

        self.play(
            LaggedStartMap(MoveToTarget, value_diagrams),
            LaggedStartMap(ShowCreation, rects),
            LaggedStartMap(GrowArrow, arrows),
            LaggedStartMap(FadeIn, delta_Es, shift=0.5 * DOWN),
        )
        self.wait()

        # Add together all changes
        low_delta_Es = delta_Es.copy()
        low_delta_Es.scale(1.5)
        low_delta_Es.arrange(RIGHT, buff=0.75)
        low_delta_Es.next_to(delta_Es, DOWN, buff=1.0)
        plusses = VGroup(
            Tex("+", font_size=72).next_to(ldE, buff=0.1).shift(0.1 * DOWN)
            for ldE in low_delta_Es
        )
        dots = Tex(R"\\dots", font_size=72).next_to(plusses, RIGHT)

        self.play(
            TransformFromCopy(delta_Es, low_delta_Es),
            Write(plusses),
            Write(dots),
            frame.animate.reorient(0, 0, 0, (-0.99, -1.51, 0.0), 10.71),
        )
        self.wait()

        # Include original embedding
        og_emb = Tex(R"\\vec{\\textbf{E}}_i", font_size=72)
        og_emb_plus = Tex("+", font_size=72)
        og_emb_plus.next_to(low_delta_Es, LEFT, SMALL_BUFF)
        og_emb.next_to(og_emb_plus, LEFT, 2 * SMALL_BUFF)
        lil_rect = SurroundingRectangle(og_emb)
        big_rect = SurroundingRectangle(VGroup(og_emb, low_delta_Es, dots), buff=0.25)
        lil_rect.set_stroke(WHITE, 2)
        big_rect.set_stroke(TEAL, 3)
        og_label = Text("Original\\nembedding")
        new_label = Text("New\\nembedding")
        new_label.set_color(TEAL)
        for label in [og_label, new_label]:
            label.next_to(lil_rect, LEFT, buff=MED_LARGE_BUFF)

        self.play(
            FadeIn(og_emb, shift=RIGHT, scale=0.5),
            Write(og_emb_plus),
            FadeIn(og_label, shift=RIGHT),
        )
        self.play(ShowCreation(lil_rect))
        self.wait()
        self.play(
            ReplacementTransform(lil_rect, big_rect),
            FadeTransform(og_label, new_label)
        )
        self.wait()


class OutputMatrix(InteractiveScene):
    def construct(self):
        # Set up all heads
        matrix_pairs = VGroup(self.get_factored_value_map() for x in range(3))
        matrix_pairs.arrange(RIGHT, buff=LARGE_BUFF)
        matrix_pairs.to_edge(LEFT)
        matrix_pairs.set_y(1)
        dots = Tex(R"\\dots", font_size=120)
        dots.next_to(matrix_pairs, RIGHT, LARGE_BUFF)

        rects = VGroup(SurroundingRectangle(pair, buff=0.25) for pair in matrix_pairs)
        rects.set_stroke(RED, 2)
        labels = VGroup()
        for n, rect in enumerate(rects, start=1):
            rect.set_height(2.5, stretch=True, about_edge=UP)
            rect.round_corners(radius=0.1)
            label = Text(f"Head {n}\\nValue map", font_size=36)
            label.next_to(rect, UP)
            labels.add(label)

        up_labels = VGroup()
        down_labels = VGroup()
        for n, pair in enumerate(matrix_pairs, start=1):
            up_mat, down_mat = pair
            down_label = TexText(Rf"Value$^{{({n})}}_{{\\downarrow}}$", font_size=30)
            up_label = TexText(Rf"Value$^{{({n})}}_{{\\uparrow}}$", font_size=30)
            for label, mat, v in zip([up_label, down_label], pair, [ORIGIN, 0.25 * RIGHT]):
                label.next_to(pair, DOWN, buff=0.5)
                label[-1].scale(1.5, about_edge=UL)
                label.match_x(mat)
                label.shift(v)
                arrow = FillArrow(label[2], mat, thickness=0.025)
                arrow.scale(0.6)
                label.add(arrow)

            up_labels.add(up_label)
            down_labels.add(down_label)

        up_labels.set_fill(RED_B)
        down_labels.set_fill(RED_C)

        # Animate
        for pair, rect, label, up_label, down_label in zip(matrix_pairs, rects, labels, up_labels, down_labels):
            mat_labels =VGroup(up_label, down_label)
            self.play(
                FadeIn(label, 0.25 * UP),
                LaggedStartMap(FadeIn, pair, scale=1.25, lag_ratio=0.5),
                LaggedStartMap(FadeIn, mat_labels, lag_ratio=0.5),
                ShowCreation(rect),
            )
        self.play(Write(dots))
        self.wait()

        # Aggregate into the output matrix
        up_matrices = VGroup(pair[0] for pair in matrix_pairs)
        stapled_up_matrices = up_matrices.copy()
        for mat in stapled_up_matrices:
            brackets = mat[-2:]
            brackets[0].stretch(0, 0, about_edge=RIGHT)
            brackets[1].stretch(0, 0, about_edge=LEFT)
            brackets.set_opacity(0)
        stapled_up_matrices.arrange(RIGHT, buff=SMALL_BUFF)
        stapled_up_matrices.scale(2)
        stapled_up_matrices.next_to(rects, DOWN, buff=1.5)

        up_labels.target = up_labels.generate_target()
        lines = VGroup()
        for stum, up_label in zip(stapled_up_matrices, up_labels.target):
            line = Line(UP, DOWN).match_height(stum)
            line.set_stroke(WHITE, 1)
            line.next_to(stum, RIGHT, buff=SMALL_BUFF / 2)
            lines.add(line)
            up_label[-1].set_opacity(0)
            up_label[-1].scale(0, about_edge=DOWN)
            up_label.scale(0.75)
            up_label.next_to(stum, UP, buff=SMALL_BUFF)

        out_dots = dots.copy()
        out_dots.scale(0.5)
        out_dots.next_to(lines, RIGHT)
        out_brackets = up_matrices[0].get_brackets().copy()
        out_brackets.match_height(stapled_up_matrices)
        out_brackets[0].next_to(stapled_up_matrices, LEFT, SMALL_BUFF)
        out_brackets[1].next_to(out_dots, RIGHT, SMALL_BUFF)

        out_matrix = VGroup(stapled_up_matrices, lines, out_dots, out_brackets)

        self.play(
            self.frame.animate.reorient(0, 0, 0, (-0.88, -0.87, 0.0), 8.00),
            up_matrices.animate.set_opacity(0.5),
            TransformFromCopy(up_matrices, stapled_up_matrices, lag_ratio=1e-4),
            MoveToTarget(up_labels),
            TransformFromCopy(dots, out_dots),
            FadeIn(lines, lag_ratio=0.5),
            FadeIn(out_brackets, scale=1.25),
            run_time=2
        )
        self.wait()

        # Circle and label output
        out_rect = SurroundingRectangle(VGroup(out_matrix, up_labels), buff=MED_SMALL_BUFF)
        out_rect.round_corners(radius=0.1)
        out_rect.set_stroke(PINK, 3)
        out_label = Text("Output\\nmatrix")
        out_label.set_color(PINK)
        out_label.next_to(out_rect, LEFT)

        self.play(
            ShowCreation(out_rect),
            FadeIn(out_label, shift=0.25 * LEFT, scale=1.25),
        )
        self.wait()

        # Center the down matrices
        self.play(
            LaggedStart(
                (pair[1].animate.shift(0.5 * LEFT)
                for pair in matrix_pairs),
                lag_ratio=0.05,
            ),
            LaggedStart(
                (label.animate.shift(0.5 * LEFT)
                for label in down_labels),
                lag_ratio=0.05,
            ),
            LaggedStartMap(FadeOut, up_matrices)
        )
        self.wait()

    def get_factored_value_map(self, big_d=7, lil_d=4, height=1.0):
        matrices = VGroup(
            WeightMatrix(shape=(big_d, lil_d)),
            WeightMatrix(shape=(lil_d, big_d)),
        )
        matrices.arrange(RIGHT, buff=matrices[0].get_width() * 0.1)
        matrices.set_height(height)
        return matrices


class Parallelizability(InteractiveScene):
    def construct(self):
        # Set up curves
        n_instances = 20
        comp_syms = Tex(R"+\\,\\times").replicate(n_instances)
        comp_syms.arrange(DOWN)
        comp_syms.set_height(5.5)
        comp_syms.to_edge(DOWN)
        left_point = comp_syms.get_left() + 2 * LEFT
        right_point = comp_syms.get_right() + 2 * RIGHT
        curves = VGroup()
        for sym in comp_syms:
            curve = VMobject()
            curve.start_new_path(left_point)
            curve.add_cubic_bezier_curve_to(
                left_point + RIGHT,
                sym.get_left() + LEFT,
                sym.get_left()
            )
            curve.add_line_to(sym.get_right())
            curve.add_cubic_bezier_curve_to(
                sym.get_right() + RIGHT,
                right_point + LEFT,
                right_point,
            )
            curve.insert_n_curves(10)
            curves.add(curve)
        curves.set_stroke(width=(0, 2, 2, 2, 0))
        curves.set_submobject_colors_by_gradient(TEAL, BLUE)

        # Setup words
        in_word = Text("Input")
        out_word = Text("output")
        in_word.next_to(left_point, LEFT, SMALL_BUFF)
        out_word.next_to(right_point, RIGHT, SMALL_BUFF)
        self.add(comp_syms, in_word, out_word)

        # GPU symbol
        gpu = SVGMobject("gpu_large.svg")
        gpu.set_fill(GREY_B)
        gpu.set_width(1.5)
        gpu.next_to(comp_syms, UP)
        gpu_name = Text("GPU")
        gpu_name.next_to(gpu, UP)
        gpu_name.set_fill(GREY_B)
        self.add(gpu, gpu_name)

        # Animation
        for n in range(4):
            curves.shuffle()
            self.play(
                LaggedStartMap(
                    ShowPassingFlash, curves,
                    lag_ratio=5e-3,
                    time_width=1.5,
                    run_time=4
                )
            )`,
    annotations: {
      1: "Enables PEP 604 union types (X | Y) and postponed evaluation of annotations for cleaner type hints.",
      3: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      4: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      5: "Imports break_into_words from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      6: "Imports break_into_tokens from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      7: "Imports get_piece_rectangles from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      10: "AttentionPatterns extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      11: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      14: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      23: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      24: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      26: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      54: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      61: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      62: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      65: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      66: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      67: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      69: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      72: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      76: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      86: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      93: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      94: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      96: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      97: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      98: "FadeOut transitions a mobject from opaque to transparent.",
      100: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      105: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      109: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      111: "CountInFrom animates a number counting up from a starting value.",
      112: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      114: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      123: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      124: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      126: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      127: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      136: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      141: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      146: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      147: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      148: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      152: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      1453: "MyseteryNovel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1454: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1548: "RoadNotTaken extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1549: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1715: "QueryMap extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1725: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1839: "Class KeyMap inherits from QueryMap.",
      1850: "DescribeAttentionEquation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1851: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2083: "ShowAllPossibleNextTokenPredictions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2084: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2169: "ShowMasking extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2170: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2259: "ScalingAPattern extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2260: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2313: "IntroduceValueMatrix extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2318: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2561: "CountMatrixParameters extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2564: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3014: "LowRankTransformation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3015: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3104: "ThinkAboutOverallMap extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3105: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3120: "CrossAttention extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3121: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3304: "CarCrashedExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3305: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3329: "TwoHarrysExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3330: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3361: "ManyTypesOfUpdates extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3362: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3503: "MultiHeadedAttention extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3504: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      3897: "OutputMatrix extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      3898: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      4036: "Parallelizability extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      4037: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/auto_regression.py"] = {
    description: "Autoregressive text generation: demonstrates how transformers generate text one token at a time, with each prediction conditioned on all previous tokens.",
    code: `from manim_imports_ext import *
from _2024.transformers.helpers import *

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import PreTrainedModel
import torch
import openai
import tiktoken


@lru_cache(maxsize=1)
def get_gpt2_tokenizer(model_name='gpt2'):
    return GPT2Tokenizer.from_pretrained(model_name)


@lru_cache(maxsize=1)
def get_gpt2_model(model_name='gpt2'):
    return GPT2LMHeadModel.from_pretrained(model_name)


def gpt2_predict_next_token(text, n_shown=7):
    tokenizer = get_gpt2_tokenizer()
    model = get_gpt2_model()
    # Encode the input text
    indexed_tokens = tokenizer.encode(
        text, add_special_tokens=False, return_tensors='pt'
    )

    # Predict all tokens
    with torch.no_grad():
        outputs = model(indexed_tokens)
        # Pull out the first batch, and the last token prediction
        predictions = outputs[0][0, -1, :]

    # Get the predicted next token
    indices = torch.argsort(predictions)
    top_indices = reversed(indices[-n_shown:])
    tokens = list(map(tokenizer.decode, top_indices))
    probs = softmax(predictions)[top_indices]

    return tokens, probs


def gpt3_predict_next_token(text, n_shown=10, random_seed=0):
    openai.api_key = os.getenv('OPENAI_KEY')
    response = openai.Completion.create(
        # Or another model version, adjust as necessary
        engine="gpt-3.5-turbo-instruct",
        prompt=text,
        max_tokens=1,
        n=1,
        temperature=1.0,
        user=str(random_seed),
        # Retrieve more than are shown
        logprobs=50
    )
    top_logprob_dict = response.choices[0]["logprobs"]["top_logprobs"][0]
    tokens, logprobs = zip(*top_logprob_dict.items())
    probs = np.exp(logprobs)
    indices = np.argsort(probs)
    top_indices = indices[-1:-n_shown:-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    return top_tokens, top_probs


class SimpleAutogregression(InteractiveScene):
    text_corner = 3.5 * UP + 0.75 * RIGHT
    line_len = 31
    font_size = 35
    n_shown_predictions = 12
    seed_text = "Behold, a wild pi creature, foraging in its native"
    seed_text_color = BLUE_B
    machine_name = "Transformer"
    machine_phi = 10 * DEGREES
    machine_theta = 12 * DEGREES
    n_predictions = 120
    skip_through = False
    random_seed = 0
    model = "gpt2"

    def construct(self):
        # Repeatedly generate
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=(n > 10),
                skip_anims=self.skip_through,
            )

    def init_text_and_machine(self):
        # Set up active text
        self.cur_str = self.seed_text
        text_mob = self.string_to_mob(self.cur_str)
        text_mob.set_color(self.seed_text_color)
        next_word_line = self.get_next_word_line(text_mob)

        # Set up Transformer as some sort of machine
        machine = self.get_transformer_drawing()
        machine.set_y(0).to_edge(LEFT, buff=-0.6)

        self.add(text_mob)
        self.add(next_word_line)
        self.add(machine)

        return text_mob, next_word_line, machine

    def string_to_mob(self, text):
        text += " l"  # Dumb hack for alignment
        result = get_paragraph(
            text.replace("\\n", " ").split(" "),
            self.line_len,
            self.font_size
        )
        result.move_to(self.text_corner, UL)
        result[-1].set_fill(BLACK, 0)  # Continue dumb hack
        result[-1].stretch(0, 0, about_edge=LEFT)
        return result

    def get_next_word_line(self, text_mob, char_len=7):
        next_word_line = Underline(text_mob[:char_len])
        next_word_line.set_stroke(TEAL, 2)
        next_word_line.next_to(text_mob[-1], RIGHT, SMALL_BUFF, aligned_edge=DOWN)
        if self.skip_through:
            next_word_line.set_opacity(0)
        return next_word_line

    def get_transformer_drawing(self):
        self.camera.light_source.move_to([-5, 5, 10])
        self.frame.set_field_of_view(20 * DEGREES)
        blocks = VGroup(
            VPrism(3, 2, 0.2)
            for n in range(10)
        )
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(width=0)
        blocks.set_shading(0.25, 0.5, 0.2)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)
        blocks.rotate(self.machine_phi, RIGHT, about_edge=OUT)
        blocks.rotate(self.machine_theta, UP, about_edge=OUT)

        blocks.deactivate_depth_test()
        for block in blocks:
            block.sort(lambda p: p[2])

        word = Text(self.machine_name, alignment="LEFT")
        word.next_to(blocks[-1], UP)
        word.shift(0.1 * UP + 0.4 * LEFT)
        word.move_to(blocks[-1])
        word.set_backstroke(BLACK, 5)
        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow)
        return result

    def get_distribution(
        self, words, probs, machine,
        font_size=24,
        width_100p=1.8,
        bar_height=0.25,
        show_ellipses=True
    ):
        labels = VGroup(Text(word, font_size=font_size) for word in words)
        bars = VGroup(
            Rectangle(prob * width_100p, bar_height)
            for prob, label in zip(probs, labels)
        )
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.5 * bar_height)
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
        bars.set_stroke(WHITE, 1)

        bar_groups = VGroup()
        for label, bar, prob in zip(labels, bars, probs):
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
            prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
            label.next_to(bar, LEFT)
            bar_groups.add(VGroup(label, bar, prob_label))

        if show_ellipses:
            ellipses = Tex(R"\\vdots", font_size=font_size)
            ellipses.next_to(bar_groups[-1][0], DOWN)
            bar_groups.add(ellipses)

        arrow_point = machine[-1].get_right()
        bar_groups.shift(arrow_point - bars.get_left() + 1.5 * RIGHT)
        bar_groups.align_to(machine, UP)

        return bar_groups

    def animate_text_input(self, text_mob, machine, position_text_over_machine=True, added_anims=[], lag_ratio=0.02):
        blocks = machine[0]
        text_copy = text_mob.copy()
        if position_text_over_machine:
            text_copy.target = text_copy.generate_target()
            text_copy.target.set_max_width(4)
            text_copy.target.next_to(blocks[0], UP)
            text_copy.target.shift_onto_screen()
            self.play(MoveToTarget(text_copy, path_arc=-45 * DEGREES))
        self.play(LaggedStart(
            *added_anims,
            Transform(
                text_copy,
                VGroup(VectorizedPoint(machine.get_top())),
                lag_ratio=lag_ratio,
                run_time=1,
                path_arc=-45 * DEGREES,
                remover=True,
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(machine[1:]),
            lag_ratio=0.5
        ))

    def animate_prediction_ouptut(self, machine, cur_str):
        words, probs = self.predict_next_token(cur_str)
        bar_groups = self.get_distribution(words, probs, machine)
        self.play(
            LaggedStart(
                (FadeInFromPoint(bar_group, machine[0][-1].get_right())
                for bar_group in bar_groups),
                lag_ratio=0.025,
                group=bar_groups,
                run_time=1
            )
        )
        return bar_groups

    def animate_random_sample(self, bar_groups):
        widths = np.array([group[1].get_width() for group in bar_groups[:-1]])
        dist = widths / widths.sum()
        seed = random.randint(0, 1000)
        buff = 0.025
        highlight_rect = SurroundingRectangle(bar_groups[0], buff=buff)
        highlight_rect.set_stroke(YELLOW, 2)
        highlight_rect.set_fill(YELLOW, 0.25)

        def highlight_randomly(rect, dist, alpha):
            np.random.seed(seed + int(10 * alpha))
            index = np.random.choice(np.arange(len(dist)), p=dist)
            rect.surround(bar_groups[index], buff=buff)
            rect.stretch(1.1, 0)

        self.play(
            UpdateFromAlphaFunc(highlight_rect, lambda rect, a: highlight_randomly(rect, dist, a)),
            Animation(bar_groups)
        )

        bar_groups.add_to_back(highlight_rect)

    def animate_word_addition(self, bar_groups, text_mob, next_word_line, force_unskip=False):
        # Choose the highlighted_group
        bar_group = None
        if isinstance(bar_groups[0], Rectangle):
            # Use the highlight rect to find the group element
            bars = bar_groups[1:-1]
            diffs = [abs(bg.get_y() - bar_groups[0].get_y()) for bg in bars]
            bar_group = bar_groups[1:][np.argmin(diffs)]
        if bar_group is None:
            bar_group = bar_groups[0]

        # Animate selection
        word = bar_group[0].get_text()
        new_str = self.cur_str + word
        new_text_mob = self.string_to_mob(new_str)
        new_text_mob[:len(self.seed_text.replace(" ", ""))].set_color(self.seed_text_color)

        word_targets = new_text_mob[word.strip()]
        if len(word_targets) > 0:
            target = word_targets[-1]
        else:
            target = new_text_mob[-len(word) - 1:-1]

        # target = new_text_mob[-len(word):]

        self.add(bar_groups)
        self.play(
            FadeTransform(bar_group[0].copy(), target),
            Transform(
                next_word_line,
                self.get_next_word_line(new_text_mob),
            ),
        )
        if force_unskip:
            self.skip_animations = False
            target.save_state()
            target.set_fill(YELLOW)
            self.wait(0.5)
            target.restore()
            self.skip_animations = True
        self.play(
            FadeOut(bar_groups),
        )

        self.remove(text_mob)
        self.add(new_text_mob)

        self.cur_str = new_str

        return new_text_mob

    def new_selection_cycle(self, text_mob, next_word_line, machine, quick=False, skip_anims=False):
        if skip_anims:
            self.skip_animations = True

        if quick:
            words, probs = self.predict_next_token(self.cur_str)
            bar_groups = self.get_distribution(words, probs, machine)
            self.add(bar_groups)
        else:
            self.animate_text_input(text_mob, machine)
            bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
            force_unskip=skip_anims
        )
        return new_text_mob

    #

    def predict_next_token(self, text):
        result = None
        n_shown = self.n_shown_predictions
        if self.model == "gpt3":
            try:
                result = gpt3_predict_next_token(
                    text, n_shown, random_seed=self.random_seed
                )
            except Exception as e:
                pass
        if result is None:
            result = gpt2_predict_next_token(text, n_shown)
        return result


class AnnotateNextWord(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()
        self.add(machine, *machine[1:])
        words, probs = self.predict_next_token(self.cur_str)
        bar_groups = self.get_distribution(words, probs, machine)

        self.add(bar_groups)

        # Initial text
        from manimlib.mobject.boolean_ops import Union
        highlight = Union(
            SurroundingRectangle(text_mob["in its native"]),
            SurroundingRectangle(text_mob["Behold, a wild pi creature, foraging"]),
        )
        highlight.set_stroke(BLUE, 3)
        arrow = Vector(RIGHT, stroke_width=10)
        arrow.next_to(highlight, LEFT)

        dist_rect = SurroundingRectangle(bar_groups)
        dist_rect.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(highlight),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(-PI / 2).next_to(dist_rect, UP),
            ReplacementTransform(highlight, dist_rect),
        )
        self.wait()
        self.play(
            FadeOut(dist_rect),
            FadeOut(arrow),
        )


class QuickerRegression(SimpleAutogregression):
    skip_through = True


class AutoregressionGPT3(SimpleAutogregression):
    model = "gpt3"


class QuickRegressionGPT3(SimpleAutogregression):
    skip_through = True
    model = "gpt3"


class GPT3CleverestAutocomplete(QuickRegressionGPT3):
    seed_text = "To date, the cleverest thinker of all time was"
    n_predictions = 70

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 2),
            )


class GPT3OnLearningSimpler(QuickRegressionGPT3):
    seed_text = "The most effective way to learn computer science is"
    text_corner = 3.5 * UP + 3 * LEFT
    line_len = 35
    font_size = 35
    n_predictions = 300
    time_per_prediction = 0.2
    random_seed = 313

    def construct(self):
        # Test
        cur_str = self.seed_text
        text_mob = VGroup()
        for n in range(self.n_predictions):
            self.remove(text_mob)
            words, probs = self.predict_next_token(cur_str)
            probs = probs / probs.sum()
            index = np.random.choice(np.arange(len(words)), p=probs)
            new_word = words[index]
            cur_str += new_word
            text_mob = self.string_to_mob(cur_str)
            text_mob[:len(self.seed_text.replace(" ", ""))].set_color(BLUE)
            text_mob[new_word.strip()][-1].set_color(YELLOW)
            if text_mob.get_bottom()[1] < -3:
                text_mob.shift(5 * UP)
                self.text_corner += 5 * UP
            self.add(text_mob)
            self.wait(self.time_per_prediction)


class ModelTakingInTextWithSurroundingPieces(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()


class AthleteCompletion(SimpleAutogregression):
    seed_text = "Michael Jordan plays the sport of"
    text_corner = 3.5 * UP + 3.0 * LEFT
    machine_phi = 5 * DEGREES
    machine_theta = 12 * DEGREES
    model = "gpt3"

    def construct(self):
        # Initialize machine
        self.set_floor_plane("xz")
        frame = self.frame
        in_text, next_word_line, machine = self.init_text_and_machine()
        self.clear()
        machine = VGroup(*machine[0])
        machine.set_height(4)
        machine.next_to(in_text, DOWN, buff=LARGE_BUFF)

        dials = MachineWithDials(n_rows=10, n_cols=15).dials
        dials.set_stroke(opacity=0.25)
        dials.set_height(machine[-1].get_height() * 0.9)

        llm_title = Text("Large\\nLanguage\\nModel", alignment="LEFT", font_size=72)
        llm_title.set_backstroke(width=8)

        for mob in [dials, llm_title]:
            mob.rotate(self.machine_phi, RIGHT).rotate(self.machine_theta, UP)
            mob.move_to(machine[-1], OUT)

        last_block_copy = machine[-1].copy()
        self.add(last_block_copy)

        frame.reorient(-13, -6, 0)
        self.play(
            LaggedStart(
                (TransformFromCopy(last_block_copy.copy().set_opacity(0), block)
                for block in machine),
                lag_ratio=0.05,
            ),
            Write(dials),
            Write(llm_title),
            frame.animate.reorient(0, 0, 0),
            run_time=3
        )
        self.remove(last_block_copy)
        self.add(machine, dials, llm_title)

        # Feed in many facts
        facts = Path(DATA_DIR, "facts.txt").read_text().split("\\n")
        fact_mobs = VGroup(get_paragraph(fact.split(" "), line_len=20) for fact in facts)
        directions = compass_directions(12, start_vect=UR)
        for fact_mob, vect in zip(fact_mobs, it.cycle(directions)):
            fact_mob.set_max_width(2)
            fact_mob.move_to(5 * vect).shift_onto_screen(buff=0.25)

        self.play(
            LaggedStart(
                (Succession(
                    FadeIn(fact_mob),
                    fact_mob.animate.set_opacity(0).move_to(machine.get_center()),
                )
                for fact_mob in fact_mobs),
                lag_ratio=0.05,
                run_time=8
            )
        )
        self.remove(fact_mobs)
        self.wait()

        # Show MJ fact
        full_input = VGroup(in_text, next_word_line)
        full_input.set_height(0.4)
        full_input.to_edge(UP)

        in_arrow = Arrow(full_input, machine, buff=0.1)
        predictions, probs = self.predict_next_token(self.seed_text)

        bar_groups = self.get_distribution(predictions, probs, machine)
        bar_groups.next_to(machine[-1], RIGHT, buff=1.5)
        out_arrow = Arrow(machine[-1], bar_groups)

        top_rect = SurroundingRectangle(VGroup(bar_groups[0]))

        self.play(FadeIn(full_input, scale=2))
        self.play(
            GrowArrow(in_arrow),
            Transform(full_input.copy(), full_input.copy().scale(0.5).set_opacity(0).move_to(machine.get_top()))
        )
        self.play(
            frame.animate.reorient(-14, -2, 0, (1.83, 0.07, -0.38), 8.63),
            LaggedStart(
                (block.animate.set_color(TEAL).set_anim_args(rate_func=there_and_back)
                for block in machine[:-1]),
                lag_ratio=0.1,
                run_time=1
            ),
        )
        self.play(
            ShowCreation(out_arrow),
            FadeIn(bar_groups, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(top_rect))

        # Reshow parameters
        self.play(
            FadeOut(llm_title),
            dials.animate.set_stroke(opacity=1)
        )
        for _ in range(5):
            self.play(
                LaggedStart(
                    (dial.animate_set_value(dial.get_random_value())
                    for dial in dials),
                    lag_ratio=0.25 / len(dials),
                    run_time=1
                )
            )

        # Quetsions
        questions = VGroup(Text("How?"), Text("Where?"))
        questions.arrange(RIGHT, buff=1.0)
        questions.set_height(0.5)
        questions.next_to(machine[-1], DOWN)

        for question in questions:
            self.play(FadeIn(question, 0.5 * UP, scale=1.5))
        self.wait()


class ThatWhichDoesNotKillMe(SimpleAutogregression):
    text_corner = 3.5 * UP + 5.0 * LEFT
    line_len = 75
    # seed_text = "That which does not kill you only makes you"
    seed_text = "Down by the river bank"
    model = "gpt3"

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.set_x(0)
        text_mob = self.new_selection_cycle(
            text_mob, next_word_line, machine,
            quick=False,
            skip_anims=False,
        )`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      60: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      68: "SimpleAutogregression extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      83: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      139: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      149: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      172: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      184: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      190: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      209: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      211: "Transform smoothly morphs one mobject into another by interpolating their points.",
      219: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      221: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      236: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      237: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      262: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      263: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      295: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      297: "Transform smoothly morphs one mobject into another by interpolating their points.",
      304: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      306: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      309: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      310: "FadeOut transitions a mobject from opaque to transparent.",
      355: "Class AnnotateNextWord inherits from SimpleAutogregression.",
      356: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      377: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      378: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      379: "GrowArrow animates an arrow growing from its start point to full length.",
      381: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      382: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      383: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      384: "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
      386: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      387: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      388: "FadeOut transitions a mobject from opaque to transparent.",
      389: "FadeOut transitions a mobject from opaque to transparent.",
      393: "Class QuickerRegression inherits from SimpleAutogregression.",
      397: "Class AutoregressionGPT3 inherits from SimpleAutogregression.",
      401: "Class QuickRegressionGPT3 inherits from SimpleAutogregression.",
      406: "Class GPT3CleverestAutocomplete inherits from QuickRegressionGPT3.",
      410: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      420: "Class GPT3OnLearningSimpler inherits from QuickRegressionGPT3.",
      429: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      450: "Class ModelTakingInTextWithSurroundingPieces inherits from SimpleAutogregression.",
      451: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      455: "Class AthleteCompletion inherits from SimpleAutogregression.",
      462: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      476: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      486: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      487: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      488: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      489: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      493: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      494: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      495: "Smoothly animates the camera to a new orientation over the animation duration.",
      509: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      510: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      584: "Class ThatWhichDoesNotKillMe inherits from SimpleAutogregression.",
      591: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/chm.py"] = {
    description: "Chapter marker scenes and visual transitions for the transformer series. Contains title cards, section headers, and bridging animations between concepts.",
    code: `from manim_imports_ext import *
from _2024.transformers.generation import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *
from _2024.transformers.ml_basics import *


# Intro

class HoldUpThumbnail(TeacherStudentsScene):
    def construct(self):
        # Test
        im = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails/Chapter5_TN3.png")
        im_group = Group(
            SurroundingRectangle(im, buff=0).set_stroke(WHITE, 3),
            im
        )
        im_group.set_height(3)
        im_group.move_to(self.hold_up_spot, DOWN)

        morty = self.teacher
        stds = self.students

        self.play(
            FadeIn(im_group, UP),
            morty.change("raise_right_hand", look_at=im_group),
            self.change_students("tease", "happy", "tease", look_at=im_group),
        )
        self.wait(4)


class IsThisUsefulToShare(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        self.play(
            morty.says("Do you find\\nthis useful?"),
            self.change_students("pondering", "hesitant", "well", look_at=self.screen)
        )
        self.wait(3)
        self.play(self.change_students("thinking", "pondering", "tease"))
        self.wait(3)


class AskAboutAttention(TeacherStudentsScene):
    def construct(self):
        # Test
        stds = self.students
        morty = self.teacher
        self.play(
            morty.change("tease"),
            stds[2].says("Can you explain what\\nAttention does?", mode="raise_left_hand", bubble_direction=LEFT),
            stds[1].change("pondering", self.screen),
            stds[0].change("pondering", self.screen),
        )
        self.wait(4)


# Version 1

class PredictTheNextWord(SimpleAutogregression):
    text_corner = 3.5 * UP + 6.5 * LEFT
    machine_name = "Large\\nLanguage\\nModel"
    seed_text = "Paris is a city in"
    model = "gpt3"
    n_shown_predictions = 12
    random_seed = 2

    def construct(self):
        # Setup machine
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.move_to(ORIGIN)
        machine[1].set_backstroke(BLACK, 3)

        text_group = VGroup(text_mob, next_word_line)
        text_group.save_state()
        text_group.scale(1.5)
        text_group.match_x(machine[0]).to_edge(UP)

        # Introduce the machine
        in_arrow = Arrow(text_group, machine[0].get_top(), thickness=5)
        frame = self.frame
        self.set_floor_plane("xz")
        blocks = machine[0]
        llm_text = machine[1]
        block_outlines = blocks.copy()
        block_outlines.set_fill(opacity=0)
        block_outlines.set_stroke(GREY_B, 2)
        block_outlines.insert_n_curves(20)

        flat_dials, last_dials = self.get_machine_dials(blocks)

        self.clear()
        frame.reorient(-31, -4, -5, (-0.24, -0.26, -0.06), 3)
        self.play(
            FadeIn(blocks, shift=0.0, lag_ratio=0.01),
            LaggedStartMap(VShowPassingFlash, block_outlines.family_members_with_points(), time_width=2.0, lag_ratio=0.01, remover=True),
            LaggedStartMap(VFadeInThenOut, flat_dials, lag_ratio=0.001, remover=True),
            Write(llm_text, time_span=(2, 4), stroke_color=WHITE),
            FadeIn(last_dials, time_span=(4, 5)),
            frame.animate.reorient(0, 0, 0, (-0.17, -0.12, 0.0), 4.50),
            run_time=6,
        )
        blocks[-1].add(last_dials)
        self.play(
            frame.animate.to_default_state(),
            FadeIn(text_group, UP),
            GrowFromCenter(in_arrow),
            run_time=3
        )

        # Single word prediction
        out_arrow = Vector(1.5 * RIGHT, thickness=5)
        out_arrow.next_to(machine[0][-1], RIGHT)
        prediction = Text("France", font_size=72)
        prediction.next_to(out_arrow, RIGHT)

        self.animate_text_input(
            text_mob, machine,
            position_text_over_machine=False,
        )
        self.play(
            LaggedStart(
                (TransformFromCopy(VectorizedPoint(machine.get_right()), letter)
                for letter in prediction),
                lag_ratio=0.05,
            ),
            GrowArrow(out_arrow)
        )
        self.wait()
        machine.replace_submobject(2, out_arrow)

        # Probability distribution
        self.play(FadeOut(prediction, DOWN))
        bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.wait()

        # Show auto_regression
        self.play(
            Restore(text_group),
            FadeOut(in_arrow),
        )

        seed_label = Text("Seed text")
        seed_label.set_color(YELLOW)
        seed_label.next_to(text_mob, DOWN)

        self.play(
            FadeIn(seed_label, rate_func=there_and_back_with_pause),
            FlashAround(text_mob, time_width=2),
            frame.animate.reorient(0, 0, 0, (0.7, -0.01, 0.0), 8.52),
            run_time=2,
        )

        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
        )

        # More!
        for n in range(20):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=True,
                skip_anims=(n > 5),
            )
            self.wait(0.25)

    def get_machine_dials(self, blocks):
        dials = VGroup(
            Dial().get_grid(8, 12).set_width(0.9 * block.get_width()).move_to(block)
            for block in blocks
        )
        dials.set_stroke(opacity=0.5)
        for group in dials:
            for dial in group:
                dial.set_value(dial.get_random_value())
        flat_dials = VGroup(*it.chain(*dials))
        last_dials = dials[-1].copy()
        last_dials.set_stroke(opacity=0.1)

        return flat_dials, last_dials


class LotsOfTextIntoTheMachine(PredictTheNextWord):
    run_time = 25
    max_snippet_width = 3

    def construct(self):
        # Add machine
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.scale(1.5)
        self.clear()
        self.add(machine)

        blocks, title = machine[:2]
        dials = Dial().get_grid(8, 12).set_width(0.9 * blocks[-1].get_width()).move_to(blocks[-1])
        dials.set_stroke(opacity=0.1)
        blocks[-1].add(dials)

        machine.center()
        machine[1].set_stroke(BLACK, 3)

        # Feed in lots of text
        snippets = self.get_text_snippets()
        text_mobs = VGroup(get_paragraph(snippet.split(" "), line_len=25) for snippet in snippets)
        directions = compass_directions(12, start_vect=UR)
        for text_mob, vect in zip(text_mobs, it.cycle(directions)):
            text_mob.set_max_width(self.max_snippet_width)
            text_mob.move_to(5 * vect).shift_onto_screen(buff=0.25)

        self.play(
            LaggedStart(
                (Succession(
                    FadeIn(text_mob),
                    text_mob.animate.set_opacity(0).move_to(machine.get_center()),
                )
                for text_mob in text_mobs),
                lag_ratio=0.05,
                run_time=self.run_time
            )
        )
        self.remove(text_mobs)
        self.wait()

    def get_text_snippets(self):
        facts = Path(DATA_DIR, "pile_of_text.txt").read_text().split("\\n")
        random.shuffle(facts)
        return facts


class EvenMoreTextIntoMachine(LotsOfTextIntoTheMachine):
    run_time = 40
    max_snippet_width = 2.5
    n_examples = 300
    context_size = 25

    def get_text_snippets(self):
        book = Path(DATA_DIR, "tale_of_two_cities.txt").read_text()
        book = book.replace("\\n", " ")
        words = list(filter(lambda m: m, book.split(" ")))
        context_size = self.context_size
        result = []
        for n in range(self.n_examples):
            index = random.randint(0, len(words) - context_size - 1)
            result.append(" ".join(words[index:index + context_size]))

        return result


class WriteTransformer(InteractiveScene):
    def construct(self):
        text = Text("Transformer", font_size=120)
        self.play(Write(text))
        self.wait()


class LabelVector(InteractiveScene):
    def construct(self):
        brace = Brace(Line(UP, DOWN).set_height(4), RIGHT)
        name = Text("Vector", font_size=72)
        name.next_to(brace, RIGHT)
        name.set_backstroke(BLACK, 5)

        self.play(
            GrowFromCenter(brace),
            Write(name),
        )
        self.wait()


class AdjustingTheMachine(InteractiveScene):
    def construct(self):
        # Add a machine and repeatedly tweak it
        frame = self.frame
        self.set_floor_plane("xz")
        frame.reorient(-28, -17, 0, ORIGIN, 8.91)
        self.camera.light_source.move_to([-10, 10, 10])

        machine = MachineWithDials(n_rows=10, n_cols=12)
        machine.set_height(6)
        blocks = VCube().replicate(10)
        blocks.set_shape(machine.get_width(), machine.get_height(), 1.0)
        blocks.deactivate_depth_test()
        cam_loc = self.frame.get_implied_camera_location() 
        for block in blocks:
            block.sort(lambda p: -get_norm(p - cam_loc))
        blocks.set_fill(GREY_D, 1)
        blocks.set_shading(0.2, 0.5, 0.25)
        blocks.arrange(OUT, buff=0.5)
        blocks.move_to(machine, OUT)

        self.add(blocks)
        self.add(machine)

        frame.clear_updaters()
        frame.add_updater(lambda f: f.set_theta(-30 * DEGREES * math.cos(0.1 * self.time)))
        self.add(frame)
        for x in range(6):
            self.play(machine.random_change_animation(lag_factor=0.1))


class FirthQuote(InteractiveScene):
    def construct(self):
        # Show Quote
        quote = TexText(R"\`\`You shall know a word\\\\by the company it keeps!''", font_size=60)
        image = ImageMobject("JohnRFirth")  # From https://www.cambridge.org/core/journals/bulletin-of-the-school-of-oriental-and-african-studies/article/john-rupert-firth/D926AFCBF99AD17D5C7A7A9C0558DFDC
        image.set_height(6.5)
        image.to_corner(UL, buff=0.5)
        name = Text("John R. Firth")
        name.next_to(image, DOWN)
        quote.move_to(midpoint(image.get_right(), RIGHT_SIDE))
        quote.to_edge(UP)

        self.play(
            FadeIn(image, 0.25 * UP),
            FadeIn(name, lag_ratio=0.1)
        )
        self.play(Write(quote))
        self.wait()

        # Show two sentences
        phrases = VGroup(
            Text("Down by the river bank"),
            Text("Deposit a check at the bank"),
        )
        bank = Text("bank", font_size=90)
        bank.set_color(TEAL)
        bank.match_x(quote).match_y(image)
        for phrase in phrases:
            phrase["bank"].set_color(TEAL)

        phrases.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        phrases.next_to(quote, DOWN, buff=2.5)
        phrases[1].set_opacity(0.15)
        banks = VGroup(
            phrase["bank"][0]
            for phrase in phrases
        )

        self.play(
            FadeIn(bank, scale=2, lag_ratio=0.25),
            quote.animate.scale(0.7, about_edge=UP).set_opacity(0.75)
        )
        self.wait()
        self.remove(bank)
        self.play(
            FadeIn(phrases[0][:len("downbytheriver")], lag_ratio=0.1),
            FadeIn(phrases[1][:len("depositacheckatthe")], lag_ratio=0.1),
            *(TransformFromCopy(bank, bank2) for bank2 in banks)
        )
        self.wait()
        self.play(
            phrases[0].animate.set_opacity(0.5),
            phrases[1].animate.set_opacity(1),
        )
        self.wait()

        # Isolate both phrases
        self.play(LaggedStart(
            FadeOut(image, LEFT, scale=0.5),
            FadeOut(name, LEFT, scale=0.5),
            FadeOut(quote, LEFT, scale=0.5),
            phrases.animate.set_opacity(1).arrange(DOWN, buff=3.5, aligned_edge=LEFT).move_to(0.5 * UP),
        ))
        self.wait()

        # Recreate
        word = Text("bank", font_size=72)
        word.set_color(TEAL)
        self.clear()

        self.add(word)
        self.wait()
        self.remove(word)
        self.play(
            *(
                FadeIn(phrase[phrase.get_text().replace("bank", "")])
                for phrase in phrases
            ),
            *(
                TransformFromCopy(word, phrase["bank"][0])
                for phrase in phrases
            )
        )
        self.add(phrases)

        # Show influence
        query_rects = VGroup(
            SurroundingRectangle(bank)
            for bank in banks
        )
        query_rects.set_stroke(TEAL, 2)
        query_rects.set_fill(TEAL, 0.25)
        key_rects = VGroup(
            SurroundingRectangle(phrases[0]["river"]),
            SurroundingRectangle(phrases[1]["Deposit"]),
            SurroundingRectangle(phrases[1]["check"]),
        )
        key_rects.set_stroke(BLUE, 2)
        key_rects.set_fill(BLUE, 0.5)
        key_rects[2].match_height(key_rects[1], about_edge=UP, stretch=True)
        arrows = VGroup(
            Arrow(key_rects[0].get_top(), banks[0].get_top(), path_arc=-180 * DEGREES, buff=0.1),
            Arrow(key_rects[1].get_top(), banks[1].get_top(), path_arc=-90 * DEGREES),
            Arrow(key_rects[2].get_top(), banks[1].get_top(), path_arc=-90 * DEGREES),
        )
        arrows.set_color(BLUE)

        key_rects.save_state()
        key_rects[0].become(query_rects[0])
        key_rects[1].become(query_rects[1])
        key_rects[2].become(query_rects[1])
        key_rects.set_opacity(0)

        self.add(query_rects, phrases)
        self.play(FadeIn(query_rects, lag_ratio=0.25))
        self.wait()

        self.add(key_rects, phrases)
        self.play(Restore(key_rects, lag_ratio=0.1, path_arc=PI / 4, run_time=2))
        self.play(LaggedStartMap(Write, arrows, stroke_width=5, run_time=3))
        self.wait()

        # Show images
        images = Group(
            ImageMobject("RiverBank"),
            ImageMobject("FederalReserve"),
        )
        for image, bank in zip(images, banks):
            image.set_height(2.0)
            image.next_to(bank, DOWN, MED_SMALL_BUFF, aligned_edge=LEFT)

        self.play(
            LaggedStart(
                (FadeTransform(Group(word).copy(), image)
                for word, image in zip(banks, images)),
                lag_ratio=0.5,
                group_type=Group,
            )
        )
        self.wait(2)


class DownByTheRiverHeader(InteractiveScene):
    def construct(self):
        words = Text("Down by the river bank ...")
        rect = SurroundingRectangle(words["bank"])
        rect.set_fill(BLUE, 0.5)
        rect.set_stroke(BLUE, 3)
        brace = Brace(rect, DOWN, buff=SMALL_BUFF)
        self.add(rect, words, brace)


class RiverBankProbParts(SimpleAutogregression):
    seed_text = "Down by the river bank, "
    model = "gpt3"

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.set_x(0)
        words = [
            "water",
            "river",
            "lake",
            "grass",
            "waves",
            "shallows",
            "pool",
            "depths",
            "foam",
            "mist",
        ]
        probs = softmax([6, 5, 4, 4, 3.5, 3.25, 3, 3, 2.5, 2])
        bar_groups = self.get_distribution(words, probs, machine)

        self.clear()
        bar_groups.set_height(6).center()
        self.play(
            LaggedStartMap(FadeIn, bar_groups, shift=0.25 * DOWN, run_time=3)
        )
        self.wait()


class FourStepsWithParameters(InteractiveScene):
    def construct(self):
        # Add rectangles and titles
        self.add(FullScreenRectangle(fill_color=GREY_E))
        rects = Square().replicate(4)
        rects.arrange(RIGHT, buff=0.25 * rects[0].get_width())
        rects.set_width(FRAME_WIDTH - 1.0)
        rects.center().to_edge(UP, buff=0.5)
        rects.set_fill(BLACK, 1)
        rects.set_stroke(WHITE, 2)
        names = VGroup(*map(TexText, [
            R"Text snippets\\\\$\\downarrow$\\\\Vectors",
            R"Attention",
            R"Feedforward",
            R"Final prediction",
        ]))
        for name, rect in zip(names, rects):
            name.scale(0.8)
            name.next_to(rect, DOWN)

        self.add(rects)
        self.play(LaggedStartMap(FadeIn, names, shift=0.25 * DOWN, lag_ratio=0.25))
        self.wait()

        # Show many dials
        machines = VGroup(
            MachineWithDials(
                width=rect.get_width(),
                height=3.0,
                n_rows=9,
                n_cols=6,
            )
            for rect in rects
        )
        for machine, rect in zip(machines, rects):
            machine.next_to(rect, DOWN, buff=0)
            machine[0].set_opacity(0)
            machine.scale(rect.get_width() / machine.dials.get_width(), about_edge=UP)
            machine.dials.shift(0.25 * UP)
            for dial in machine.dials:
                dial.set_value(0)

        self.play(
            LaggedStart((
                LaggedStart(
                    (GrowFromPoint(dial, machine.get_top())
                    for dial in machine.dials),
                    lag_ratio=0.025,
                )
                for machine in machines
            ), lag_ratio=0.25),
            LaggedStartMap(FadeOut, names)
        )
        for _ in range(2):
            self.play(
                LaggedStart(
                    (machine.random_change_animation()
                    for machine in machines),
                    lag_ratio=0.2,
                )
            )


class ChatbotFeedback(InteractiveScene):
    random_seed = 404

    def construct(self):
        # Test
        self.frame.set_height(10).move_to(DOWN)
        user_prompt = "User: How and when was the internet invented?"

        prompt_mob = Text(user_prompt)
        prompt_mob.to_edge(UP)
        prompt_mob["User:"].set_color(BLUE)

        self.answer_mob = Text("AI Assistant:")
        self.answer_mob.next_to(prompt_mob, DOWN, buff=1.0, aligned_edge=LEFT)
        self.answer_mob.set_color(YELLOW)
        self.og_answer_mob = self.answer_mob

        self.add(prompt_mob, self.answer_mob)

        # Show multiple answer
        for n in range(8):
            self.give_answer(prompt_mob)
            mark = self.judge_answer()
            self.add(self.og_answer_mob)
            self.play(FadeOut(self.answer_mob), FadeOut(mark))
            self.answer_mob = self.og_answer_mob

    def display_answer(self, text):
        new_answer_mob = get_paragraph(text.replace("\\n", " ").split(" "))
        new_answer_mob[:len(self.og_answer_mob)].match_style(self.og_answer_mob)
        new_answer_mob.move_to(self.og_answer_mob, UL)
        self.remove(self.answer_mob)
        self.answer_mob = new_answer_mob
        self.add(self.answer_mob)

    def give_answer(self, prompt_mob, max_responses=100):
        answer = self.og_answer_mob.get_text()
        user_prompt = prompt_mob.get_text()
        for n in range(max_responses):
            answer, stop = self.add_to_answer(user_prompt, answer)
            if stop:
                break
            self.display_answer(answer)
            self.wait(2 / 30)

    def judge_answer(self):
        mark = random.choice([
            Checkmark().set_color(GREEN),
            Exmark().set_color(RED),
        ])
        mark.scale(5)
        mark.next_to(self.answer_mob, RIGHT, aligned_edge=UP)
        rect = SurroundingRectangle(self.answer_mob)
        rect.match_color(mark)
        self.play(FadeIn(mark, scale=2), FadeIn(rect, scale=1.05))
        self.wait()
        return VGroup(mark, rect)

    def add_to_answer(self, user_prompt: str, answer: str):
        try:
            tokens, probs = gpt3_predict_next_token("\\n\\n".join([user_prompt, answer]))
            token = random.choices(tokens, np.array(probs) / sum(probs))[0]
        except IndexError:
            return answer, True

        stop = False
        if token == '<|endoftext|>':
            stop = True
        else:
            answer += token
        return answer, stop


class ContrastWithEarlierFrame(InteractiveScene):
    def construct(self):
        # Test
        vline = Line(UP, DOWN)
        vline.set_height(FRAME_HEIGHT)
        self.add(vline)

        titles = VGroup(
            VGroup(
                Text("Most earlier models"),
                # Vector(0.75 * DOWN, thickness=4),
                # Text("One word at a time")
            ),
            VGroup(
                Text("Transformers"),
                # Vector(0.75 * DOWN, thickness=4),
                # Text("All words in parallel")
            ),
        )
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.arrange(DOWN, buff=0.2)
            title.scale(1.5)
            title.move_to(FRAME_WIDTH * vect / 4)
            title.to_edge(UP)

        self.add(titles)


class SequentialProcessing(InteractiveScene):
    def construct(self):
        # Add text
        text = Text("Down by the river bank, where I used to go fishing ...")
        text.move_to(1.0 * DOWN)
        words = break_into_words(text)
        rects = get_piece_rectangles(words)
        blocks = VGroup(VGroup(rect, word) for rect, word in zip(rects, words))
        blocks.save_state()
        self.add(blocks)

        # Vector wandering over
        vect = NumericEmbedding()
        vect.set_width(1.0)
        vect.next_to(rects[0], UP)

        for n in range(len(blocks) - 1):
            blocks.target = blocks.saved_state.copy()
            blocks.target[:n].fade(0.75)
            blocks.target[n + 1:].fade(0.75)
            self.play(
                vect.animate.next_to(blocks[n], UP),
                MoveToTarget(blocks)
            )
            self.play(
                LaggedStart(
                    (ContextAnimation(elem, blocks[n][1], lag_ratio=0.01)
                    for elem in vect.get_entries()),
                    lag_ratio=0.01,
                ),
                RandomizeMatrixEntries(vect),
                run_time=2
            )


# Version 2


class PartialScript(SimpleAutogregression):
    machine_name = "Magic next\\nword predictor"
    machine_phi = 5 * DEGREES
    machine_theta = 6 * DEGREES

    def construct(self):
        # Set frame
        frame = self.frame
        self.set_floor_plane("xz")

        # Unfurl script
        curled_script_img = ImageMobject("HumanAIScript")
        curled_script_img.set_height(7)

        curves = VGroup(SVGMobject("JaggedCurl1")[0], SVGMobject("JaggedCurl2")[0])
        for curve in curves:
            curve.make_smooth(approx=False)
            curve.insert_n_curves(100)
            curve.set_stroke(WHITE, 3)
            curve.set_fill(opacity=0)
            curve.set_height(5)
        curves[1].scale(curves[0].get_arc_length() / curves[1].get_arc_length())

        resolution = (2, 200)  # Change
        surface_kw = dict(u_range=(-6, 6), v_range=(0.05, 0.95), resolution=resolution)
        curled_script_templates = Group(
            ParametricSurface(
                lambda u, v: (*curve.pfp(v)[:2], u),
                **surface_kw
            )
            for curve in curves
        )
        curled_script_templates[1].rotate(PI / 2, UP)
        curled_script_templates[0].rotate(-PI / 2)
        flat_script_template = ParametricSurface(
            lambda u, v: (u, v, 0),
            **surface_kw
        )
        curled_script0 = TexturedSurface(curled_script_templates[0], "HumanAIScript")
        curled_script1 = TexturedSurface(curled_script_templates[1], "HumanAIScript")
        curled_script1_torn = TexturedSurface(curled_script_templates[1], "HumanAIScriptTorn")
        flat_script = TexturedSurface(flat_script_template, "HumanAIScriptTorn")
        flat_script.replace(curled_script_img, stretch=True)

        for script in [curled_script0, curled_script1]:
            script.set_shading(0.25, 0.25, 0.35)
        curled_script1_torn.set_shading(0, 0, 0)
        flat_script.set_shading(0, 0, 0)

        frame.reorient(0, -1, 0, (-0.28, 0.69, 0.0), 14.43)
        self.play(
            TransformFromCopy(curled_script0, curled_script1),
            frame.animate.reorient(56, -17, 0, (-0.2, -1.52, -2.39), 20.05),
            run_time=3
        )
        self.play(
            frame.animate.reorient(-6, -11, 0, (1.06, -1.22, -2.65), 20.05),
            run_time=8,
        )
        self.play(
            FadeOut(curled_script1, shift=1e-2 * IN),
            FadeIn(curled_script1_torn, shift=1e-2 * IN),
        )
        self.play(
            ReplacementTransform(curled_script1_torn, flat_script),
            frame.animate.to_default_state(),
            run_time=2
        )
        self.wait()

        # Show the machine
        machine = self.get_transformer_drawing()
        machine[1].set_height(0.7).set_stroke(width=2)
        machine[1].set_opacity(0)
        machine.remove(machine[-1])
        machine.set_height(3)
        machine.to_edge(RIGHT)

        self.play(
            flat_script.animate.set_height(5).to_edge(LEFT),
            FadeIn(machine, lag_ratio=0.01)
        )
        self.add(machine)
        self.wait()

        # Show example input and output
        out_arrow = Vector(DOWN, thickness=6)
        out_arrow.next_to(machine, DOWN)
        in_arrow = out_arrow.copy().next_to(machine, UP, SMALL_BUFF)
        in_text = Text("To be or not to _")
        in_text[-1].stretch(3, 0, about_edge=LEFT)
        in_text.next_to(in_arrow, UP)
        prediction = Text("be", font_size=72)
        prediction.next_to(out_arrow, DOWN)

        self.play(FadeIn(in_text), GrowArrow(in_arrow))
        self.animate_text_input(in_text, machine, position_text_over_machine=False)
        self.play(
            GrowArrow(out_arrow),
            FadeIn(prediction, DOWN),
        )
        self.wait()

        # Clear the board
        script_text = self.get_text()
        script_text.set_width(0.89 * flat_script.get_width())
        script_text.next_to(flat_script.get_top(), DOWN, buff=0.33)

        font_size = 48 * (script_text[0].get_height() / Text("H").get_height())
        completion = "A transistor is a semiconductor device used to amplify or switch electronic signals. It consists of three layers of semiconductor material, either p-type or n-type, forming a structure with terminals called the emitter, base, and collector."
        words = completion.split(" ")
        paragraph = get_paragraph(completion.split(" "), font_size=font_size)
        paragraph.next_to(script_text, DOWN, aligned_edge=LEFT)
        paragraph.set_color(YELLOW)

        self.play(
            FadeIn(script_text),
            FadeOut(flat_script),
            FadeOut(VGroup(in_text, in_arrow, prediction)),
        )

        # Repeatedly add predictions
        machine.scale(1.25, about_edge=RIGHT)
        out_arrow.next_to(machine, DOWN, buff=0.5)

        blocks = machine[0]
        dials = Dial().get_grid(11, 16)
        dials.set_width(blocks[-1].get_width() * 0.95)
        dials.rotate(5 * DEGREES, RIGHT).rotate(10 * DEGREES, UP)
        dials.move_to(blocks[-1])
        dials.set_stroke(opacity=0.5)
        for dial in dials:
            dial.set_value(dial.get_random_value())
        dials.set_z_index(2)
        self.add(dials)

        curr_answer = VGroup()
        curr_answer.next_to(script_text, DOWN)
        for n in range(6):
            word = words[n]
            prediction = Text(words[n], font_size=72)
            prediction.next_to(out_arrow, DOWN)
            word_in_answer = paragraph[len(curr_answer):len(curr_answer) + len(word)]
            word_in_answer.set_color(YELLOW)
            mover = VGroup(script_text, curr_answer).copy()

            if n > 2:
                self.skip_animations = True

            self.play(
                mover.animate.set_height(1.8).next_to(machine, UP, SMALL_BUFF).set_anim_args(path_arc=-30 * DEGREES),
            )
            self.animate_text_input(
                mover, machine,
                position_text_over_machine=False,
                lag_ratio=1e-3
            )
            self.play(FadeIn(prediction, DOWN, rate_func=rush_from, run_time=0.5))

            if n > 2:
                self.skip_animations = False
                self.wait(0.5)
                self.skip_animations = True

            self.play(
                curr_answer.animate.set_color(WHITE),
                Transform(prediction, word_in_answer),
                FadeOut(mover),
            )
            curr_answer.add(*word_in_answer)
            self.add(curr_answer)
            self.remove(prediction)

    def get_text(self):
        script_text = Text("""
            Human:
            Can you explain the history of
            transistors and how they're relevant
            to computers? What is a transistor,
            and how exactly is it used to
            perform computations?

            AI assistant:
        """, alignment="LEFT")
        script_text["Human"].set_color(BLUE)
        script_text["AI assistant"].set_color(TEAL)

        script_text.set_height(4).to_edge(UP)
        return script_text

    def create_image(self):
        # Create image
        script_text = self.get_text()
        script_text.set_fill(BLACK)
        script_text["Human"].set_fill(BLUE_D)
        script_text["AI assistant"].set_fill(TEAL_D)
        self.add(FullScreenRectangle(fill_color="#FCF5E5", fill_opacity=1))
        self.add(script_text)

        # Add off test
        tear_off = SVGMobject('TearOff')
        tear_off.set_stroke(width=0)
        tear_off.set_fill(BLACK, 1)
        tear_off.set_width(7.5)
        tear_off.next_to(script_text, DOWN, buff=-0.2)
        self.add(tear_off)


class ShowMachineWithDials(PredictTheNextWord):
    words = ['worst', 'age', 'worse', 'best', 'most', 'end', 'very', 'blur']
    logprobs = [4.0, 2.15, 1.89, 1.4, 0.1, -0.18, -0.23, -0.61]

    def construct(self):
        # Show machine (same position as in PredictTheNextWord)
        frame = self.frame
        self.set_floor_plane("xz")
        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()

        self.clear()
        self.add(frame)
        frame.reorient(0, 0, 0, (-0.17, -0.12, 0.0), 4.50)
        self.add(blocks, llm_text, last_dials)

        # Prepare dial highlight
        last_dials.target = last_dials.generate_target()
        self.fix_dials(last_dials.target)

        small_rect = SurroundingRectangle(last_dials[0], buff=0.025)
        small_rect.set_stroke(BLUE, 2)
        big_rect = small_rect.copy().scale(4)
        big_rect.next_to(blocks, UP, buff=SMALL_BUFF, aligned_edge=LEFT + OUT)
        big_rect.shift(1.5 * RIGHT)
        big_dial = last_dials[0].copy().scale(4).set_stroke(opacity=1)
        big_dial.move_to(big_rect)
        rect_lines = VGroup(
            Line(small_rect.get_corner(UL), big_rect.get_corner(DL)),
            Line(small_rect.get_corner(UR), big_rect.get_corner(DR)),
        )
        rect_lines.set_stroke(WHITE, width=(1, 3))
        highlighed_parameter_group = VGroup(small_rect, rect_lines, big_rect, big_dial)

        last_dials.set_stroke(width=1, opacity=1)
        self.play(
            MoveToTarget(last_dials),
            FadeOut(llm_text),
            FadeIn(small_rect),
        )

        # Show an example input and output
        example = self.get_example(blocks)
        in_text, in_arrow, out_arrow, bar_groups = example
        logprobs = example.logprobs
        true_probs = 100 * softmax(logprobs)
        bar_groups = self.get_output_distribution(self.words, 0.1 * logprobs, out_arrow)

        self.play(
            LaggedStart(
                ShowCreation(rect_lines, lag_ratio=0),
                TransformFromCopy(small_rect, big_rect),
                TransformFromCopy(last_dials[0], big_dial),
                FadeIn(in_text),
                GrowArrow(in_arrow),
                FadeIn(bar_groups),
                GrowArrow(out_arrow),
            ),
            frame.animate.reorient(0, 0, 0, (-0.43, 0.38, 0.0), 7.05),
            run_time=2
        )
        self.play(
            last_dials[0].animate_set_value(0.8),
            big_dial.animate_set_value(0.8),
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials[1:]),
                lag_ratio=1.0 / len(last_dials),
            ),
            *(
                self.bar_group_change_animation(bg, value)
                for bg, value in zip(bar_groups[:-1], true_probs)
            ),
            run_time=3
        )
        self.wait()

        # Play around tweaking the parameters, and seeing the output change
        self.play(
            LaggedStart(
                (dial.animate_set_value(0)
                for dial in last_dials[:12]),
                lag_ratio=0.01,
            ),
            big_dial.animate_set_value(0),
            self.bar_group_change_animation(bar_groups[0], 50),
            self.bar_group_change_animation(bar_groups[1], 34),
            self.bar_group_change_animation(bar_groups[2], 5),
            run_time=4,
        )
        self.play(
            LaggedStart(
                (dial.animate_set_value(1)
                for dial in last_dials[:12]),
                lag_ratio=0.01,
            ),
            big_dial.animate_set_value(1),
            self.bar_group_change_animation(bar_groups[0], 80),
            self.bar_group_change_animation(bar_groups[1], 5),
            self.bar_group_change_animation(bar_groups[2], 15),
            run_time=4,
        )
        self.wait()

        # Mention randomness
        random_words = Text("Initially random")
        random_words.next_to(blocks, UP)
        random_words.set_color(RED)
        out_dots = Tex(R"...", font_size=120)
        out_dots.next_to(out_arrow, RIGHT)

        self.play(
            FadeOut(big_rect),
            Uncreate(rect_lines, lag_ratio=0),
            FadeOut(small_rect),
            Transform(big_dial, last_dials[0])
        )
        self.play(
            Write(random_words),
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=0.5 / len(last_dials),
                run_time=2
            ),
            FadeOut(bar_groups),
        )
        self.play(Write(out_dots))
        self.wait()
        self.play(
            FadeOut(dots),
            FadeOut(random_words),
            FadeIn(bar_groups),
        )

        # Show many many parameters
        example.save_state()
        blocks.save_state()
        last_dials.save_state()
        all_dials = VGroup(*flat_dials, *last_dials)
        all_dials.generate_target()
        all_dials.target.space_out_submobjects(3)
        new_dials = VGroup(
            all_dials.target.copy().shift(3 * 2 * x * (flat_dials.get_center() - last_dials.get_center()))
            for x in range(1, 9)
        )

        self.play(
            FadeOut(example),
            FadeOut(blocks),
            FadeIn(flat_dials),
            FadeOut(bar_groups),
            FadeOut(out_arrow),
        )
        self.play(
            FadeOut(highlighed_parameter_group),
            MoveToTarget(all_dials),
            LaggedStart(
                (TransformFromCopy(all_dials.copy().set_opacity(0), nd)
                for nd in new_dials),
                lag_ratio=0.05,
            ),
            frame.animate.reorient(-9, 0, 0, (-0.71, -0.07, -0.06), 9.64),
            run_time=4
        )
        self.wait()

    def get_blocks_and_dials(self):
        machine = self.get_transformer_drawing()
        machine.move_to(ORIGIN)
        self.machine = machine

        blocks = machine[0]
        llm_text = machine[1]
        llm_text.set_backstroke(BLACK, 2)
        flat_dials, last_dials = self.get_machine_dials(blocks)
        return blocks, llm_text, flat_dials, last_dials

    def get_example(self, blocks):
        in_text = Text("It was the best\\nof times it was\\nthe _", alignment="LEFT")
        in_text[-1].stretch(4, 0, about_edge=LEFT)
        in_text.next_to(blocks, LEFT, LARGE_BUFF)
        in_arrow = Arrow(in_text, blocks)

        out_arrow = Vector(RIGHT)
        out_arrow.next_to(blocks[-1], RIGHT, buff=0.1)
        logprobs = np.array(self.logprobs)
        bar_groups = self.get_output_distribution(self.words, logprobs, out_arrow)
        example = VGroup(in_text, in_arrow, out_arrow, bar_groups)
        example.logprobs = logprobs
        return example

    def fix_dials(self, dials):
        for dial in dials:
            dial.set_stroke(width=1, opacity=1)
            dial.needle.set_stroke(width=(2, 0))
        return dials

    def bar_group_change_animation(self, bar_group, new_value):
        text, rect, value_mob = bar_group
        buff = value_mob.get_left() - rect.get_right()
        factor = new_value / value_mob.get_value()

        return AnimationGroup(
            rect.animate.stretch(factor, 0, about_edge=LEFT),
            ChangeDecimalToValue(value_mob, new_value),
            UpdateFromFunc(text, lambda m: value_mob.move_to(rect.get_right() + buff, LEFT)),
        )

    def get_output_distribution(self, words, logprobs, out_arrow):
        probs = softmax(logprobs)
        bar_groups = self.get_distribution(words, probs, self.machine, width_100p=1.0)
        bar_groups.next_to(out_arrow, RIGHT)
        return bar_groups


class ShowSingleTrainingExample(ShowMachineWithDials):
    logprobs = [4.0, 6.15, 1.89, 1.4, 0.1, -0.18, -0.23, -0.61]

    def construct(self):
        # Add state from before
        frame = self.frame
        self.set_floor_plane("xz")

        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()
        self.fix_dials(last_dials)
        example = self.get_example(blocks)
        in_text, in_arrow, out_arrow, bar_groups = example

        self.add(blocks, last_dials)

        # Show example up top
        parts = ("It was the best of times it was the", "worst")
        sentence = Text(" ".join(parts))
        start = sentence[parts[0]][0]
        end = sentence[parts[1]][0]
        sentence.set_width(10)
        sentence.next_to(blocks, UP, buff=1.5)

        start_rect = SurroundingRectangle(start)
        start_rect.set_stroke(BLUE, 2)
        start_rect.set_fill(BLUE, 0.2)
        end_rect = SurroundingRectangle(end)
        end_rect.match_height(start_rect, stretch=True).match_y(start_rect)
        end_rect.set_stroke(YELLOW, 2)
        end_rect.set_fill(YELLOW, 0.2)
        arrow = Arrow(start_rect.get_top(), end_rect.get_top(), path_arc=-90 * DEGREES, thickness=5)
        arrow.set_fill(border_width=1)

        frame.reorient(0, 0, 0, (-0.36, 0.97, 0.0), 7.52)
        self.play(FadeIn(sentence, UP))
        self.play(
            LaggedStartMap(DrawBorderThenFill, VGroup(start_rect, end_rect)),
            FadeIn(arrow),
        )
        self.remove(last_dials)
        self.play(LaggedStart(
            AnimationGroup(
                TransformFromCopy(start, in_text[:-1]),
                TransformFromCopy(end_rect, in_text[-1]),
                FadeIn(in_arrow)
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                group=blocks,
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(last_dials),
            GrowArrow(out_arrow),
            LaggedStartMap(GrowFromPoint, bar_groups, point=out_arrow.get_start()),
            lag_ratio=0.3
        ))
        self.wait()

        # Flag bad prediction
        out_rects = VGroup(
            SurroundingRectangle(bg)
            for bg in bar_groups[:2]
        )
        out_rects.set_stroke(RED, 3)
        annotations = VGroup(
            Tex(tex, font_size=60).next_to(rect, LEFT, buff=SMALL_BUFF)
            for rect, tex in zip(out_rects, [R"\\uparrow", R"\\downarrow"])
        )
        annotations.set_color(RED)

        self.play(
            FadeTransform(end_rect.copy(), out_rects[0]),
            Write(annotations[0]),
        )
        self.wait()
        self.play(
            FadeTransform(*out_rects),
            FadeTransform(*annotations),
        )
        self.wait()
        self.play(
            FadeOut(out_rects[1]),
            FadeOut(annotations[1]),
        )

        # Adjust
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=1.0 / len(last_dials),
            ),
            LaggedStart(
                (FlashAround(dial, stroke_width=2, color=YELLOW, time_width=1, buff=0.025) for dial in last_dials),
                lag_ratio=1.0 / len(last_dials),
            ),
            self.bar_group_change_animation(bar_groups[0], 70),
            self.bar_group_change_animation(bar_groups[1], 20),
            self.bar_group_change_animation(bar_groups[2], 8),
            run_time=6
        )


class ParameterWeight(InteractiveScene):
    def construct(self):
        # Test
        text = Text("Parameter / Weight", font_size=72)
        text.to_edge(UP)
        text.set_color(YELLOW)
        param = text["Parameter"][0]
        param.save_state()
        param.set_x(0)

        self.play(Write(param))
        self.wait()
        self.play(LaggedStart(
            Restore(param),
            FadeIn(text["/ Weight"]),
        ))
        self.wait()


class LargeInLargeLanguageModel(InteractiveScene):
    def construct(self):
        # Test
        text = Text("Large Language Model", font_size=72)
        text.to_edge(UP)
        large = text["Large"][0]
        large.save_state()
        large.set_x(0)

        self.add(large)
        self.play(FlashUnder(large), large.animate.set_color(YELLOW))
        self.play(
            Restore(large, path_arc=-30 * DEGREES),
            Write(text[len(large):], time_span=(0.5, 1.5))
        )
        self.wait()


class ThousandsOfWords(InteractiveScene):
    def construct(self):
        # Find passage
        file = Path("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/data/tale_of_two_cities.txt")
        novel = file.read_text()
        start_index = novel.index("It was the best of times")
        end_index = novel.index("There were a king with a large jaw")

        # Add text
        passage = novel[start_index:start_index + 5000].replace("\\n", " ")
        text = get_paragraph(passage.split(" "), line_len=150)
        text.set_width(14)
        text.to_edge(UP)
        self.add(text)


class EnormousAmountOfTrainingText(PremiseOfMLWithText):
    def construct(self):
        # Setup
        self.init_data()
        # n_rows = n_cols = 41
        n_rows = n_cols = 9
        screens = VGroup()
        for row in range(n_rows):
            for col in range(n_cols):
                screen = self.get_screen()
                screen.move_to(FRAME_WIDTH * row * RIGHT + FRAME_HEIGHT * col * DOWN)
                screens.add(screen)
        screens.center()
        screens.submobjects.sort(key=lambda sm: get_norm(sm.machine.get_center()))

        self.add(screens)

        # Add frame growth
        frame = self.frame
        frame.clear_updaters()
        frame.add_updater(lambda m: m.set_height(FRAME_HEIGHT * np.exp(0.2 * self.time)))

        # Show lots of new data
        inner_screens = screens[:25]
        n_examples = 20
        for n in range(n_examples):
            self.play(LaggedStart(
                *(self.change_example_animation(screen, show_dial_change=True)
                for screen in inner_screens),
                lag_ratio=0.1,
                run_time=0.5,
            ))

    def change_example_animation(self, screen, show_dial_change=True):
        new_example = VGroup(*self.new_input_output_example(*screen.arrows))
        time_span = (0, 0.35)
        anims = [
            FadeOut(screen.training_example, time_span=time_span),
            FadeIn(new_example, time_span=time_span),
        ]
        if show_dial_change:
            anims.append(screen.machine.random_change_animation(run_time=0.5))
        screen.training_example = new_example
        return AnimationGroup(*anims)

    def get_screen(self):
        border = FullScreenRectangle()
        border.set_fill(opacity=0)
        border.set_stroke(WHITE, 2)

        machine = MachineWithDials(width=3.5, height=2.5, n_rows=5, n_cols=7)
        machine.move_to(1.0 * RIGHT)
        in_arrow, out_arrow = arrows = Vector(RIGHT).replicate(2)
        in_arrow.next_to(machine, LEFT)
        out_arrow.next_to(machine, RIGHT)
        in_data, out_data = training_example = VGroup(
            *self.new_input_output_example(in_arrow, out_arrow)
        )

        screen = VGroup(
            border, machine,
            arrows, training_example
        )
        screen.border = border
        screen.machine = machine
        screen.arrows = arrows
        screen.training_example = training_example

        return screen

    def new_input_output_example(self, in_arrow, out_arrow):
        in_data, out_data = super().new_input_output_example(in_arrow, out_arrow)
        in_data.scale(0.8, about_edge=RIGHT)
        out_data.scale(0.8, about_edge=LEFT)
        return in_data, out_data


class BadChatBot(InteractiveScene):
    def construct(self):
        # Add bot
        bot = self.get_bot()
        bot.set_height(3)

        lines = Line(LEFT, RIGHT).get_grid(4, 1, buff=0.25)
        lines.set_stroke(WHITE, 1)
        lines[-1].stretch(0.5, 0, about_edge=LEFT)
        lines.set_width(3)
        bubble = SpeechBubble(lines, buff=MED_LARGE_BUFF)
        bubble.set_stroke(width=5)
        bubble.pin_to(bot).shift(DOWN)

        self.add(bot)
        self.play(Write(bubble, run_time=3))
        self.blink(bot)
        self.wait()

        # Make lines bad
        self.play(
            LaggedStart(
                (Transform(line, self.get_scribble(line))
                for line in lines),
                lag_ratio=0.1,
                run_time=2
            )
        )
        for _ in range(2):
            self.blink(bot)
            self.wait(2)

    def get_scribble(self, line):
        freqs = np.random.random(5)
        graph = FunctionGraph(
            lambda x: 0.05 * sum(math.sin(freq * TAU * x) for freq in freqs),
            x_range=(0, 5, 0.1)
        )
        graph.put_start_and_end_on(*line.get_start_and_end())
        graph.match_style(line)
        graph.set_stroke(color=RED)
        return graph

    def get_bot(self):
        bot = SVGMobject("Bot")
        subpaths = bot[0].get_subpaths()
        bot[0].set_points([*subpaths[0], subpaths[0][-1], *subpaths[1]])
        eyes = VGroup(Dot().replace(VMobject().set_points(subpath)) for subpath in subpaths[2:])
        bot.eyes = eyes
        bot.add(eyes)
        bot.set_stroke(width=0)

        bot.set_height(4)
        bot.set_fill(GREY_B)
        bot.set_shading(0.5, 0.5, 1)

        return bot

    def blink(self, bot):
        self.play(
            bot.eyes.animate.stretch(0, 1).set_anim_args(rate_func=squish_rate_func(there_and_back))
        )


class WriteRLHF(InteractiveScene):
    def construct(self):
        text = Text("Step 2: RLHF")
        full_text = Text("Reinforcement Learning\\nwith Human Feedback")
        full_text.next_to(text, UP, LARGE_BUFF)
        full_text.align_to(text, RIGHT).shift(RIGHT)
        initials = VGroup(full_text[letter[0]][0][0] for letter in "RLHF")
        full_text.remove(*initials)

        self.add(text)
        self.wait()
        self.play(
            TransformFromCopy(text["RLHF"][0], initials, lag_ratio=0.25),
            Write(full_text, time_span=(1.5, 3)),
            run_time=3
        )
        self.wait()


class RLHFWorker(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        # worker = SVGMobject("computer_stall")
        worker = SVGMobject("comp_worker")
        worker.set_height(4)
        worker.move_to(4 * LEFT)
        worker.set_fill(GREY_C, 1)

        rect = Rectangle(7, 5)
        rect.to_edge(RIGHT)
        rect.set_stroke(WHITE, 2)
        rect.set_fill(BLACK, 1)

        self.add(worker)
        self.add(rect)


class RLHFWorkers(ShowMachineWithDials):
    def construct(self):
        # Add workers
        self.add(FullScreenRectangle().set_fill(GREY_E, 1))
        workers = SVGMobject("comp_worker").get_grid(3, 2, buff=0.5)
        workers.set_height(7)
        workers.to_edge(LEFT)
        workers.set_fill(GREY_C, 1)

        self.add(workers)

        # Machine
        blocks, llm_text, flat_dials, last_dials = self.get_blocks_and_dials()
        machine = VGroup(blocks, last_dials)
        machine.set_height(4)
        machine.center().to_edge(RIGHT, buff=LARGE_BUFF)
        last_dials.set_stroke(opacity=1)

        self.add(machine)

        for _ in range(8):
            self.play(LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in last_dials),
                lag_ratio=0.5 / len(last_dials),
                run_time=2
            ))
            self.wait()


class SerialProcessing(InteractiveScene):
    phrase = "It was the best of times it was the worst of times"
    phrase_center = 2 * UP

    def construct(self):
        # Set up words
        words = self.get_words()
        rects = get_piece_rectangles(words)

        self.add(rects)
        self.add(words)

        # Animate in the vectors
        vectors = VGroup(
            self.get_abstract_vector().next_to(word, DOWN, LARGE_BUFF)
            for word in words
        )
        last_vect = VGroup(VectorizedPoint(rects[0].get_bottom()))

        for word, vect in zip(words, vectors):
            self.play(
                FadeIn(vect, run_time=2),
                LaggedStart(
                    (ContextAnimation(
                        square, VGroup(*word, *last_vect),
                        direction=DOWN,
                        lag_ratio=0.01,
                        path_arc=30 * DEGREES
                    )
                    for square in vect),
                    lag_ratio=0.05,
                    run_time=2
                ),
                last_vect.animate.set_opacity(0.2)
            )
            last_vect = vect

    def get_words(self):
        result = break_into_words(Text(self.phrase))
        result.move_to(self.phrase_center)
        return result

    def get_abstract_vector(self, values=None, default_length=10, elem_size=0.2):
        if values is None:
            values = np.random.uniform(-1, 1, default_length)
        result = Square().get_grid(len(values), 1, buff=0)
        result.set_width(elem_size)
        result.set_stroke(WHITE, 1)
        for square, value in zip(result, values):
            color = value_to_color(value, min_value=0, max_value=1)
            square.set_fill(color, opacity=1)
        return result


class ParallelProcessing(SerialProcessing):
    def construct(self):
        # Set up words
        words = self.get_words()
        rects = get_piece_rectangles(words)

        self.add(rects)
        self.add(words)

        # Animate in the vectors
        vectors = VGroup(
            self.get_abstract_vector().next_to(word, DOWN, buff=1.5)
            for word in words
        )

        lines = VGroup(
            Line(
                rect.get_bottom(), vect.get_top(),
                buff=0.05,
                stroke_color=WHITE,
                stroke_width=2 * random.random()**3
            )
            for rect in rects
            for vect in vectors
        )
        lines.shuffle()

        for vect, word in zip(vectors, words):
            vect.save_state()
            for square in vect:
                square.move_to(word)
                square.set_opacity(0)

        self.play(
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.01),
            LaggedStartMap(Restore, vectors, lag_ratio=0)
        )
        self.play(lines.animate.set_stroke(opacity=0.25))
        self.wait()


class ManyComputationsPerUnitTimeV2(InteractiveScene):
    def construct(self):
        # Add computations
        box = Rectangle(5, 5)
        label = Text("1 Billion computations per Second")
        label.next_to(box, UP)
        self.add(box)
        self.add(label)

        comps = self.get_computations(box)
        self.add(comps)
        self.wait(3)

        # Place box into minute interval
        width = FRAME_WIDTH - 1
        number_lines = VGroup(
            minute_line := NumberLine((0, 60, 1), width=width, big_tick_spacing=10),
            hour_line := NumberLine((0, 60, 1), width=width, big_tick_spacing=10),
            day_line := NumberLine((0, 24, 1), width=width, big_tick_spacing=6),
            month_line := NumberLine((0, 31, 1), width=width),
            year_line := NumberLine((0, 12, 1), width=width),
            y100_line := NumberLine((0, 100, 1), width=width),
            y10k_line := NumberLine((0, 100, 1), width=width),
            y1M_line := NumberLine((0, 100, 1), width=width),
            y100M_line := NumberLine((0, 100, 1), width=width),
        )
        number_lines.move_to(DOWN)

        first_ticks = minute_line.ticks[:2]
        sec_brace = Brace(first_ticks, DOWN, buff=0, tex_string=R"\\underbrace{\\qquad\\qquad}")
        sec_label = Text("Second", font_size=30).next_to(sec_brace, DOWN, SMALL_BUFF)

        self.play(
            ShowCreation(minute_line, lag_ratio=0.01),
            box.animate.match_width(first_ticks).move_to(first_ticks.get_center(), DOWN).set_stroke(width=1),
            TransformFromCopy(label["Second"][0], sec_label),
            GrowFromCenter(sec_brace),
            run_time=2
        )

        # Add other boxes
        minute_label = self.get_timeline_full_label(number_lines[1], "Minute")
        new_boxes = VGroup(
            box.copy().move_to(tick.get_center(), DL)
            for tick in minute_line.ticks[1:-1]
        )
        for new_box in new_boxes:
            new_box.save_state()
            new_box.move_to(box)
        computations = VGroup(
            self.get_computations(new_box, n_iterations=1)
            for new_box in new_boxes
        )
        # computations = VGroup()  # If needed

        self.add(computations)
        self.play(
            FadeIn(minute_label, DOWN),
            LaggedStartMap(Restore, new_boxes, lag_ratio=0.1),
            run_time=2
        )
        self.wait(2)

        # Add labels
        minute_line.add(minute_label)
        names = ["Hour", "Day", "Month", "Year", "100 Years", "10,000 Years", "1,000,000 Years", "100,000,000 Years"]
        for line, name in zip(number_lines[1:], names):
            line.label = self.get_timeline_full_label(line, name)
            line.add(line.label)

        # Arrange all lines
        number_lines[1:].arrange(DOWN, buff=2.0)
        number_lines[1:].next_to(minute_line, DOWN, buff=2.0)

        scale_lines = VGroup()
        for nl1, nl2 in zip(number_lines, number_lines[1:]):
            n = len(nl2.ticks) // 2
            mini_line = Line(nl2.ticks[n - 1].get_center(), nl2.ticks[n].get_center())
            pair = VGroup(
                DashedLine(nl1.get_start(), mini_line.get_start()),
                DashedLine(nl1.get_end(), mini_line.get_end()),
            )
            pair.set_stroke(WHITE, 2)
            nl1.target = nl1.copy()
            nl1.target.replace(mini_line, dim_to_match=0)
            nl1.target.shift(mini_line.pfp(0.5) - nl1.target.pfp(0.5))
            scale_lines.add(pair)

        # Start panning down
        lag_ratio = 1.5
        self.play(
            LaggedStart(
                *(AnimationGroup(*(ShowCreation(sl) for sl in pair)) for pair in scale_lines),
                lag_ratio=lag_ratio,
            ),
            LaggedStart(
                *(FadeIn(nl) for nl in number_lines[1:]),
                lag_ratio=lag_ratio,
            ),
            LaggedStart(
                *(TransformFromCopy(nl, nl.target) for nl in number_lines[:-1]),
                lag_ratio=lag_ratio,
            ),
            self.frame.animate.set_y(number_lines[-1].get_y() + 2).set_width(18).set_anim_args(
                rate_func=lambda t: interpolate(smooth(t), linear(t), there_and_back_with_pause(t, pause_ratio=0.8))
            ),
            run_time=30
        )
        self.play(self.frame.animate.reorient(0, 0, 0, (-0.03, -11.55, 0.0), 31.76), run_time=4)
        self.wait(4)

    def fade_in_bigger_interval(self, new_interval, prev_interval, fader, scale_factor, added_anims=[]):
        pivot = prev_interval.n2p(0)
        new_interval.save_state()
        new_interval.scale(scale_factor, about_point=pivot)
        new_interval[:-1].set_opacity(0)
        new_interval[-1].set_fill(BLACK)

        self.play(
            Restore(new_interval),
            prev_interval.animate.scale(1.0 / scale_factor, about_point=pivot).set_fill(border_width=0),
            fader.animate.scale(1.0 / scale_factor, about_point=pivot).set_opacity(0),
            *added_anims,
            run_time=4,
            rate_func=rush_from
        )
        self.remove(fader)

    def get_timeline_full_label(self, timeline, name):
        brace = Brace(Line().set_width(7), UP, buff=MED_SMALL_BUFF)
        brace.set_fill(border_width=5)
        brace.match_width(timeline)
        brace.next_to(timeline, UP, buff=MED_SMALL_BUFF)
        label = Text(name, font_size=72)
        label.next_to(brace, UP, MED_SMALL_BUFF)

        label.next_to(timeline, DOWN)
        return label

        return VGroup(brace, label)

    def get_computations(self, box, n_lines=10, n_iterations=3, n_digits=4, cycle_time=0.5):
        # Try adding lines
        lines = VGroup()
        for iteration in range(n_iterations):
            cluster = VGroup()
            for n in range(n_lines):
                x = random.uniform(0, 10**(n_digits))
                y = random.uniform(0, 10**(n_digits))
                if random.choice([True, False]):
                    comb = x * y
                    sym = Tex(R"\\times")
                else:
                    comb = x + y
                    sym = Tex(R"+")
                line = VGroup(
                    DecimalNumber(x, num_decimal_places=3), sym,
                    DecimalNumber(y, num_decimal_places=3), Tex("="),
                    DecimalNumber(comb, num_decimal_places=3)
                )
                line.arrange(RIGHT, buff=SMALL_BUFF)
                lines.add(line)
                cluster.add(line)
            cluster.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
            cluster.set_max_height(0.9 * box.get_height())
            cluster.set_max_width(0.9 * box.get_width())
            cluster.move_to(box)

        # Add updater
        def update_lines(lines):
            sigma = 0.12
            alpha = (self.time / (cycle_time * n_iterations)) % 1
            step = 1.0 / len(lines)
            for n, line in enumerate(lines):
                x = min((
                    abs(a - n * step)
                    for a in (alpha - 1, alpha, alpha + 1)
                ))
                y = np.exp(-x**2 / sigma**2)
                line.set_fill(opacity=y)

            lines.set_height(0.9 * box.get_height())
            lines.move_to(box)

        lines.clear_updaters()
        lines.add_updater(update_lines)

        return lines

    def old(self):
        # Repeatedly scale down
        to_fade = VGroup(sec_brace, sec_label, box, comps, new_boxes, computations)
        scale_factors = [60, 24, 365, 1000]
        for new_int, prev_int, scale_factor in zip(number_lines[1:], number_lines[0:], scale_factors):
            self.fade_in_bigger_interval(
                new_int, prev_int, to_fade, scale_factor,
                added_anims=[label.animate.set_opacity(0)],
            )
            self.wait(2)
            to_fade = prev_int

        # Multiply last line by 100
        self.fade_in_bigger_interval(
            y1M_line, millenium_line, year_line, 1000,
            added_anims=[self.frame.animate.reorient(0, 0, 0, (-3.51, -5.18, 0.0), 12.93)],
        )

        lines = Line(LEFT, RIGHT).replicate(100)
        lines.match_width(y1M_line)
        lines.arrange_to_fit_height(10)
        lines.sort(lambda p: -p[1])
        lines.set_stroke(WHITE, 1)
        lines.move_to(y1M_line[0].get_center(), UP)

        side_brace, label100M = self.get_timeline_full_label(y1M_line, "100,000,000 Years")
        side_brace.rotate(PI / 2)
        side_brace.match_height(lines)
        side_brace.next_to(lines, LEFT)
        label100M.next_to(side_brace, LEFT)

        self.play(
            LaggedStart(
                (TransformFromCopy(lines[0].copy().set_opacity(0), line)
                for line in lines),
                lag_ratio=0.03,
                run_time=2
            ),
            FadeIn(side_brace, scale=10, shift=2 * DOWN, time_span=(1, 2)),
            FadeIn(label100M, time_span=(1, 2)),
        )
        self.wait()


class VectorLabel(InteractiveScene):
    def construct(self):
        # Test
        brace = Brace(Line(4 * UP, ORIGIN), LEFT)
        brace.center()
        brace.set_stroke(WHITE, 3)
        text = Text("Vector", font_size=90)
        text.next_to(brace, LEFT, MED_SMALL_BUFF)
        text.shift(SMALL_BUFF * UP)

        self.play(
            GrowFromCenter(brace),
            Write(text)
        )
        self.play(
            FlashUnder(text, color=YELLOW)
        )
        self.wait()


class ParameterToVectorAnnotation(InteractiveScene):
    def construct(self):
        # Test
        dials = VGroup(Dial(value_range=(-10, 10, 1)) for _ in range(10))
        dials.arrange(DOWN)
        dials.set_height(5)

        values = [1, 4.3, 2, 0.9, -1.5, 2.9, -1.2, 7.8, 0, -2.3]
        arrows = VGroup(
            Vector(0.5 * RIGHT, thickness=2).next_to(dial, RIGHT, buff=SMALL_BUFF)
            for dial in dials
        )

        self.play(
            Write(dials, lag_ratio=0.01),
            LaggedStartMap(GrowArrow, arrows),
        )
        self.play(LaggedStart(
            (dial.animate_set_value(value)
            for dial, value in zip(dials, values)),
            lag_ratio=0.05,
        ))
        self.wait()


class ThreeWordsToOne(InteractiveScene):
    def construct(self):
        # Test
        image = ImageMobject("CHMTopText")
        image.set_height(FRAME_HEIGHT)
        # self.add(image)

        phrase = Text("Computer History Museum", font_size=61)
        words = VGroup(phrase[word][0] for word in phrase.get_text().split(" "))
        words.move_to([0, 2.627, 0])
        og_words = words.copy()
        og_words.shift(DOWN)
        words[0].shift(0.13 * LEFT)
        words[2].shift(0.4 * RIGHT)
        colors = ["#63DCF7", "#90C9FA", "#85D4FE"]
        for word, color in zip(words, colors):
            word.set_color(color)

        words.save_state()

        self.add(words)
        self.wait()

        # Back to unity
        rect = SurroundingRectangle(og_words)
        rect.set_color(RED)
        chm_image = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/chm/images/CHM_Exterior.jpeg")
        chm_image.match_width(rect)
        chm_image.next_to(rect, DOWN)

        self.play(Transform(words, og_words))
        self.play(
            ShowCreation(rect),
            FadeIn(chm_image, DOWN)
        )
        self.wait()

        # Three pieces
        rects = VGroup(
            SurroundingRectangle(word).set_fill(color, 0.2).set_stroke(color, 2)
            for word, color in zip(words.saved_state, colors)
        )
        words.set_z_index(1)

        icons = VGroup(
            SVGMobject("GenericComputer.svg"),
            SVGMobject("History.svg"),
            SVGMobject("Museum.svg"),
        )
        for word, icon in zip(words.saved_state, icons):
            icon.set_fill(word.get_color(), 1, border_width=1)
            icon.set_height(1)
            icon.next_to(word, DOWN)

        self.remove(chm_image)
        self.play(
            ReplacementTransform(VGroup(rect), rects),
            Restore(words),
            *(
                FadeTransform(chm_image.copy(), icon)
                for icon in icons
            )
        )
        self.wait()


class ExamplePhraseHeader(InteractiveScene):
    def construct(self):
        # Test
        phrase = Text("The Computer History Museum\\nis located in ?????")
        phrase.to_edge(UP)
        rect = SurroundingRectangle(phrase).set_stroke(WHITE, 2)

        q_marks = phrase["?????"][0]
        q_marks[::4].set_fill(opacity=0)
        q_rect = SurroundingRectangle(q_marks)
        q_rect.set_fill(YELLOW, 0.25)
        q_rect.set_stroke(YELLOW, 2)

        self.add(q_rect)
        self.add(phrase)


class TrainingDataCHM(InteractiveScene):
    def construct(self):
        # Test
        passages = [
            "The Computer History Museum (CHM) is a museum ... located in Mountain View...",
            "Computer History Museum ... 1401 N. Shoreline Blvd. Mountain View...",
            "Things to do in Mountain View ... the Computer History Museum ...",
            "While I was in Mountain View ... stopped by the Computer History Museum ...",
        ]
        items = VGroup(
            get_paragraph(passage.split(" "), line_len=35, font_size=30)
            for passage in passages
        )

        items.arrange(DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        items.to_corner(DL)
        items.shift(0.5 * UP)
        dots = Tex(R"\\vdots")
        dots.next_to(items, DOWN, MED_LARGE_BUFF)
        dots.shift_onto_screen(buff=MED_SMALL_BUFF)
        items.add(dots)

        title = Text("Training Data")
        title.next_to(items, UP, buff=LARGE_BUFF)
        title.shift_onto_screen(buff=MED_SMALL_BUFF)
        underline = Underline(title)

        chm_phrases = VGroup(item["Computer History Museum"] for item in items)
        mv_phrases = VGroup(item["Mountain View"] for item in items)

        self.play(
            FadeIn(title),
            ShowCreation(underline),
            LaggedStartMap(FadeIn, items, shift=DOWN, lag_ratio=0.15)
        )
        self.wait()
        self.play(chm_phrases.animate.set_color(RED).set_anim_args(lag_ratio=0.1))
        self.wait()
        self.play(mv_phrases.animate.set_color(PINK).set_anim_args(lag_ratio=0.1))
        self.wait()

        # Arrows to ffn
        ffn_point = 3 * RIGHT + DOWN
        arrows = VGroup(
            Arrow(
                item.get_right(),
                interpolate(item.get_right(), ffn_point, 0.6),
                path_arc=arc * DEGREES,
            )
            for item, arc in zip(items[:-1], range(-40, 40, 20))
        )
        arrows.set_fill(border_width=1)
        self.play(Write(arrows, lag_ratio=0.1), run_time=3)
        self.play(
            LaggedStart(
                *(
                    FadeOutToPoint(letter.copy(), ffn_point)
                    for letter in VGroup(chm_phrases, mv_phrases).family_members_with_points()
                ),
                lag_ratio=1e-2,
                run_time=3
            )
        )
        self.wait()


class DivyUpParameters(ShowMachineWithDials):
    def construct(self):
        # Show machine
        frame = self.frame
        self.set_floor_plane("xz")

        machine = VGroup(*self.get_blocks_and_dials())
        blocks, llm_text, flat_dials, last_dials = machine
        machine.set_height(3.0)
        machine.to_edge(DOWN, buff=LARGE_BUFF)

        block_outlines = blocks.copy()
        block_outlines.set_fill(opacity=0)
        block_outlines.set_stroke(WHITE, 2)
        block_outlines.insert_n_curves(20)

        # last_dials.set_submobjects(last_dials[:3])  # Remove
        last_dials.set_stroke(opacity=1)
        for dial in last_dials:
            dial[0].set_stroke(width=1)
            dial[1].set_stroke(width=1)
            dial[3].set_stroke(width=(3, 0))

        frame.reorient(-23, -13, 0, (-0.41, -1.71, -0.06), 4.95)
        self.play(
            FadeIn(blocks, shift=0.0, lag_ratio=0.01),
            LaggedStartMap(VShowPassingFlash, block_outlines.family_members_with_points(), time_width=2.0, lag_ratio=0.01, remover=True),
            LaggedStartMap(VFadeInThenOut, flat_dials, lag_ratio=0.001, remover=True),
            FadeIn(last_dials, time_span=(2, 3)),
            self.frame.animate.reorient(10, -2, 0, (-0.25, -1.58, -0.02), 4.61),
            run_time=3,
        )
        self.remove(flat_dials)

        # Show individual blocks
        top_blocks = blocks[:3].copy()
        all_dials = VGroup(*last_dials)
        for block in top_blocks:
            dials = last_dials.copy()
            dials.rotate(self.machine_phi, RIGHT)
            dials.rotate(self.machine_theta, UP)
            dials.move_to(block)
            dials.set_stroke(opacity=1)
            block.add(dials)
            block.target = block.generate_target()
            dials.set_opacity(0)
            all_dials.add(*dials)

        block_targets = Group(block.target for block in top_blocks)
        block_targets.rotate(-self.machine_theta, UP)
        block_targets.rotate(-self.machine_phi, RIGHT)
        block_targets.set_height(2)
        block_targets.arrange(RIGHT, buff=1.5)
        block_targets.to_edge(UP)
        block_targets.set_shading(0.1, 0.1, 0.1)

        labels = VGroup(
            TexText(R"Word $\\to$ Vector"),
            Text("Attention"),
            Text("Feedforward"),
        )
        for label, block in zip(labels, block_targets):
            label.next_to(block, DOWN)

        self.add(
            blocks[0], top_blocks[0],
            blocks[1], top_blocks[1],
            blocks[2], top_blocks[2],
            blocks[3:], last_dials
        )
        self.play(
            MoveToTarget(top_blocks[1], time_span=(0, 2)),
            MoveToTarget(top_blocks[2], time_span=(1, 3)),
            MoveToTarget(top_blocks[0], time_span=(2, 4)),
            Write(labels[1], time_span=(1.5, 2)),
            Write(labels[2], time_span=(2.5, 3)),
            Write(labels[0], time_span=(3.5, 4)),
            frame.animate.to_default_state(),
            run_time=4
        )
        self.wait()

        # Change all the parameters
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in all_dials),
                lag_ratio=1 / len(all_dials),
                run_time=6
            ),
            LaggedStart(
                (FlashAround(dial, buff=0, color=YELLOW)
                for dial in all_dials),
                lag_ratio=1 / len(all_dials),
                run_time=6
            ),
        )
        self.wait()


# End clips


class ShowPreviousVideos(InteractiveScene):
    def construct(self):
        # Backdrop
        background = FullScreenRectangle()
        self.add(background)

        line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        line.set_stroke(WHITE, 2)

        series_name = Text("Deep Learning Series", font_size=68)
        series_name.to_edge(UP, buff=0.35)
        self.add(series_name)

        # Show thumbnails
        thumbnails = Group(
            Group(
                Rectangle(16, 9).set_height(1).set_stroke(WHITE, 2),
                ImageMobject(f"https://img.youtube.com/vi/{slug}/maxresdefault.jpg", height=1)
            )
            for slug in [
                "aircAruvnKk",
                "IHZwWFHWa-w",
                "Ilg3gGewQ5U",
                "tIeHLnjs5U8",
                "wjZofJX0v4M",
                "eMlx5fFNoYc",
                "9-Jl0dxWQs8",
            ]
        )

        thumbnails.arrange_in_grid(n_cols=4, buff=0.2)
        thumbnails.set_width(FRAME_WIDTH - 1)
        thumbnails.next_to(series_name, DOWN, buff=1.0)
        thumbnails[-3:].set_x(0)

        self.play(LaggedStartMap(FadeIn, thumbnails, shift=0.3 * UP, lag_ratio=0.35, run_time=4))
        self.wait()

        # Rearrange
        left_x = -FRAME_WIDTH / 4
        self.play(
            series_name.animate.set_x(left_x),
            thumbnails.animate.arrange_in_grid(n_cols=2, buff=0.25).set_height(6).set_x(left_x).to_edge(DOWN),
            ShowCreation(line, time_span=(1, 2)),
            run_time=2,
        )
        self.wait()


class EndScreen(PatreonEndScreen):
    title_text = "Where to dig deeper"
    thanks_words = """
        Special thanks to these Patreon supporters
    """`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2024.transformers.generation module within the 3b1b videos codebase.",
      3: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      4: "Imports * from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      5: "Imports * from the _2024.transformers.ml_basics module within the 3b1b videos codebase.",
      10: "Class HoldUpThumbnail inherits from TeacherStudentsScene.",
      11: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      24: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      32: "Class IsThisUsefulToShare inherits from TeacherStudentsScene.",
      33: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      45: "Class AskAboutAttention inherits from TeacherStudentsScene.",
      46: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      61: "Class PredictTheNextWord inherits from SimpleAutogregression.",
      69: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      185: "Class LotsOfTextIntoTheMachine inherits from PredictTheNextWord.",
      189: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      232: "Class EvenMoreTextIntoMachine inherits from LotsOfTextIntoTheMachine.",
      251: "WriteTransformer extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      252: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      258: "LabelVector extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      259: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      272: "AdjustingTheMachine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      273: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      303: "FirthQuote extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      304: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      445: "DownByTheRiverHeader extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      446: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      455: "Class RiverBankProbParts inherits from SimpleAutogregression.",
      459: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      486: "FourStepsWithParameters extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      487: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      549: "ChatbotFeedback extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      552: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      622: "ContrastWithEarlierFrame extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      623: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      650: "SequentialProcessing extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      651: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      688: "Class PartialScript inherits from SimpleAutogregression.",
      693: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      896: "Class ShowMachineWithDials inherits from PredictTheNextWord.",
      900: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1111: "Class ShowSingleTrainingExample inherits from ShowMachineWithDials.",
      1114: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1220: "ParameterWeight extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1221: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1239: "LargeInLargeLanguageModel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1240: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1257: "ThousandsOfWords extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1258: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1273: "Class EnormousAmountOfTrainingText inherits from PremiseOfMLWithText.",
      1274: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1350: "BadChatBot extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1351: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1414: "WriteRLHF extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1415: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1433: "RLHFWorker extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1434: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1452: "Class RLHFWorkers inherits from ShowMachineWithDials.",
      1453: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1482: "SerialProcessing extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1486: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1536: "Class ParallelProcessing inherits from SerialProcessing.",
      1537: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1577: "ManyComputationsPerUnitTimeV2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1578: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1810: "VectorLabel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1811: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1830: "ParameterToVectorAnnotation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1831: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1855: "ThreeWordsToOne extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1856: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1921: "ExamplePhraseHeader extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1922: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1938: "TrainingDataCHM extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1939: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2004: "Class DivyUpParameters inherits from ShowMachineWithDials.",
      2005: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2107: "ShowPreviousVideos extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2108: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2156: "Class EndScreen inherits from PatreonEndScreen.",
    }
  };

  files["_2024/transformers/embedding.py"] = {
    description: "Word and token embedding visualizations. Shows how text is broken into tokens, mapped to high-dimensional vectors, and how positional encoding works. Foundational to the transformer series.",
    code: `import gensim
import tiktoken
from pathlib import Path

from manim_imports_ext import *
from _2024.transformers.helpers import *


def get_token_encoding():
    return tiktoken.encoding_for_model("davinci")


def get_principle_components(data, n_components=3):
    covariance_matrix = np.cov(data, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    order_of_importance = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, order_of_importance]  # sort the columns
    return sorted_eigenvectors[:, :n_components]


def find_nearest_words(model, vector, n=20):
    data = model.vectors
    indices = np.argsort(((data - vector)**2).sum(1))
    return [model.index_to_key[i] for i in indices[:n]]


def break_into_pieces(phrase_mob: Text, offsets: list[int]):
    phrase = phrase_mob.get_string()
    lhs = offsets
    rhs = [*offsets[1:], len(phrase)]
    result = []
    for lh, rh in zip(lhs, rhs):
        substr = phrase[lh:rh]
        start = phrase_mob.substr_to_path_count(phrase[:lh])
        end = start + phrase_mob.substr_to_path_count(substr)
        result.append(phrase_mob[start:end])
    return VGroup(*result)


def break_into_words(phrase_mob):
    offsets = [m.start() for m in re.finditer(" ", phrase_mob.get_string())]
    return break_into_pieces(phrase_mob, [0, *offsets])


def break_into_tokens(phrase_mob):
    tokenizer = get_token_encoding()
    tokens = tokenizer.encode(phrase_mob.get_string())
    _, offsets = tokenizer.decode_with_offsets(tokens)
    return break_into_pieces(phrase_mob, offsets)


def get_piece_rectangles(
    phrase_pieces,
    h_buff=0.05,
    v_buff=0.1,
    fill_opacity=0.15,
    fill_color=None,
    stroke_width=1,
    stroke_color=None,
    hue_range=(0.5, 0.6),
    leading_spaces=False,
):
    rects = VGroup()
    height = phrase_pieces.get_height() + 2 * v_buff
    last_right_x = phrase_pieces.get_x(LEFT)
    for piece in phrase_pieces:
        left_x = last_right_x if leading_spaces else piece.get_x(LEFT)
        right_x = piece.get_x(RIGHT)
        fill = random_bright_color(hue_range) if fill_color is None else fill_color
        stroke = fill if stroke_color is None else stroke_color
        rect = Rectangle(
            width=right_x - left_x + 2 * h_buff,
            height=height,
            fill_color=fill,
            fill_opacity=fill_opacity,
            stroke_color=stroke,
            stroke_width=stroke_width
        )
        if leading_spaces:
            rect.set_x(left_x, LEFT)
        else:
            rect.move_to(piece)
        rect.set_y(0)
        rects.add(rect)

        last_right_x = right_x

    rects.match_y(phrase_pieces)
    return rects


def get_word_to_vec_model(model_name="glove-wiki-gigaword-50"):
    filename = str(Path(DATA_DIR, model_name))
    if os.path.exists(filename):
        return gensim.models.keyedvectors.KeyedVectors.load(filename)
    model = gensim.downloader.load(model_name)
    model.save(filename)
    return model


def get_direction_lines(axes, direction, n_lines=500, color=YELLOW, line_length=1.0, stroke_width=3):
    line = Line(ORIGIN, line_length * normalize(direction))
    line.insert_n_curves(20).set_stroke(width=(0, stroke_width, stroke_width, stroke_width, 0))
    lines = line.replicate(n_lines)
    lines.set_color(color)
    for line in lines:
        line.move_to(axes.c2p(
            random.uniform(*axes.x_range[:2]),
            random.uniform(*axes.y_range[:2]),
            random.uniform(*axes.z_range[:2]),
        ))
    return lines


# For chapter 5


class LyingAboutTokens2(InteractiveScene):
    def construct(self):
        # Mention next word prediction task
        phrase = Text("The goal of our model is to predict the next word")

        words = break_into_tokens(phrase)
        rects = get_piece_rectangles(words, leading_spaces=True, h_buff=0)

        words.remove(words[-1])
        q_marks = Text("???")
        rects[-1].set_color(YELLOW)
        q_marks.next_to(rects[-1], DOWN)

        big_rect = Rectangle()
        big_rect.replace(rects[:-1], stretch=True)
        big_rect.set_stroke(GREY_B, 2)
        arrow = Arrow(big_rect.get_top(), rects[-1].get_top(), path_arc=-120 * DEGREES)
        arrow.scale(0.5, about_edge=DR)

        self.play(ShowIncreasingSubsets(words, run_time=1))
        self.add(rects[-1])
        self.play(LaggedStart(
            FadeIn(big_rect),
            ShowCreation(arrow),
            Write(q_marks),
            lag_ratio=0.3,
        ))
        self.wait()
        self.play(
            FadeOut(big_rect),
            LaggedStart(*(
                DrawBorderThenFill(rect)
                for rect in rects[:-1]
            ), lag_ratio=0.02),
            LaggedStart(*(
                word.animate.match_color(rect)
                for word, rect in zip(words, rects)
            )),
            FadeOut(arrow)
        )
        self.wait()

        # Show words into vectors
        vectors = VGroup(*(
            NumericEmbedding(length=8)
            for word in words
        ))
        vectors.arrange(RIGHT, buff=1.0 * vectors[0].get_width())
        vectors.set_width(12)
        vectors.to_edge(DOWN, buff=1.0)
        vectors.to_edge(LEFT, buff=0.5)
        for vector, word in zip(vectors, words):
            vector.get_brackets().match_color(word[0])

        blocks = VGroup(*(VGroup(rect, word) for rect, word in zip(rects, words)))
        q_group = VGroup(rects[-1], q_marks)
        blocks.target = blocks.generate_target()
        for block, vector in zip(blocks.target, vectors):
            block.next_to(vector, UP, buff=1.5)

        arrows = VGroup(*(
            Arrow(block, vect, stroke_width=3)
            for block, vect in zip(blocks.target, vectors)
        ))

        self.play(
            MoveToTarget(blocks),
            q_group.animate.next_to(blocks.target, RIGHT, aligned_edge=UP),
            LaggedStartMap(FadeIn, vectors, shift=0.5 * DOWN),
            LaggedStartMap(GrowFromCenter, arrows),
        )
        self.wait()

        # Setup titles
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        title1, title2 = titles = VGroup(
            Text("The Truth", font_size=72).to_edge(UP),
            Text("A Convenient Lie", font_size=72).next_to(h_line, DOWN),
        )
        h_line.set_stroke(WHITE, 2)
        h_line.next_to(titles[1], UP)
        for title in titles:
            title.add(Underline(title))

        # Show the lie
        phrase1, phrase2 = phrases = VGroup(
            Text("This process (known fancifully as tokenization) frequently subdivides words"),
            # Text("It's nice to sometimes pretend tokens are words"),
            Text("Let's pretend that tokens are always simply words"),
        )
        for phrase, title in zip(phrases, titles):
            phrase.set_width(FRAME_WIDTH - 1)
            phrase.next_to(title, DOWN, buff=1.0)

        tokens = break_into_tokens(phrase1)
        words = break_into_words(phrase2)
        token_rects = get_piece_rectangles(tokens, hue_range=(0.1, 0.2), leading_spaces=True, h_buff=0.0)
        word_rects = get_piece_rectangles(words, hue_range=(0.5, 0.6))

        self.play(
            FadeOut(blocks),
            FadeOut(q_group),
            FadeOut(arrows),
            FadeOut(vectors),
            ShowCreation(h_line),
            FadeIn(title1, lag_ratio=0.1),
            FadeIn(tokens),
        )
        self.add(token_rects, tokens)
        self.play(
            LaggedStartMap(FadeIn, token_rects),
            LaggedStart(*(
                token.animate.set_color(rect.get_color())
                for token, rect in zip(tokens, token_rects)
            ))
        )
        self.wait()
        self.play(
            FadeIn(title2, lag_ratio=0.1),
            FadeIn(words),
        )
        self.add(word_rects, words)
        self.play(
            LaggedStartMap(FadeIn, word_rects),
            LaggedStart(*(
                token.animate.set_color(rect.get_color())
                for token, rect in zip(words, word_rects)
            ))
        )
        self.wait()

        # Analyze tokenization
        brace = Brace(token_rects[8], buff=0.05)

        self.play(GrowFromCenter(brace))
        self.wait()
        for index in [2, 4, 5, 7, 8, 9, 11, 12]:
            self.play(brace.animate.become(Brace(token_rects[index], buff=0.05)))
            self.wait()


class DiscussTokenization(InteractiveScene):
    def construct(self):
        pass


class ImageTokens(InteractiveScene):
    n_divisions = 52

    def construct(self):
        # Add image
        image = ImageMobject("SmallFluffCreature")  # Change
        image.set_height(5)
        self.add(image)

        # Add pixels
        pixels = create_pixels(image, pixel_width=image.get_width() / self.n_divisions)
        big_pixels = create_pixels(image, pixel_width=image.get_width() / (self.n_divisions / 4))

        patches = big_pixels.copy().set_fill(opacity=0)
        p_points = np.array([p.get_center() for p in pixels])
        bp_points = np.array([bp.get_center() for bp in big_pixels])

        for pixel in pixels:
            dists = np.linalg.norm(bp_points - pixel.get_center(), axis=1)
            patches[np.argmin(dists)].add(pixel)

        # Anim test
        self.play(FadeIn(patches))
        self.remove(image)
        self.play(patches.animate.space_out_submobjects(2.0).scale(0.75))
        self.wait()
        self.play(LaggedStart(
            (patch.animate.set_stroke(TEAL, 3).set_anim_args(rate_func=there_and_back)
            for patch in patches),
            lag_ratio=5.0 / len(patches),
        ))
        self.wait()


class SoundTokens(InteractiveScene):
    def construct(self):
        # Add wave form
        n_lines = 100
        wave_form = Line(UP, DOWN).replicate(n_lines)
        wave_form.arrange(RIGHT)
        wave_form.arrange_to_fit_width(5)
        wave_form.next_to(ORIGIN, RIGHT)

        def func(x):
            x *= 1.7
            return sum([
                math.sin(x),
                0.5 * math.sin(2 * x),
                0.3 * math.sin(3 * x),
                0.2 * math.sin(4 * x),
                0.1 * math.sin(5 * x),
                0.15 * math.sin(6 * x),
            ])

        for line in wave_form:
            line.set_height(abs(func(line.get_x())))

        wave_form.center()
        self.add(wave_form)

        # Subdivide
        step = 5
        chunks = VGroup(wave_form[i:i + step] for i in range(0, len(wave_form), step))

        self.add(chunks)
        self.wait()
        self.play(chunks.animate.space_out_submobjects(2.0).scale(0.75))
        self.play(LaggedStart(
            (chunk.animate.set_stroke(TEAL, 3).scale(1.5).set_anim_args(rate_func=there_and_back)
            for chunk in chunks),
            lag_ratio=2.0 / len(chunks),
            run_time=2
        ))
        self.wait()


class IntroduceEmbeddingMatrix(InteractiveScene):
    def construct(self):
        # Load words
        words = [
            'aah',
            'aardvark',
            'aardwolf',
            'aargh',
            'ab',
            'aback',
            'abacterial',
            'abacus',
            'abalone',
            'abandon',
            'zygoid',
            'zygomatic',
            'zygomorphic',
            'zygosis',
            'zygote',
            'zygotic',
            'zyme',
            'zymogen',
            'zymosis',
            'zzz'
        ]

        # Get all words
        dots = Tex(R"\\vdots")
        shown_words = VGroup(
            *map(Text, words[:10]),
            dots,
            *map(Text, words[-10:]),
        )
        shown_words.arrange(DOWN, aligned_edge=LEFT)
        dots.match_x(shown_words[:5])
        shown_words.set_height(FRAME_HEIGHT - 1)
        shown_words.move_to(LEFT)
        shown_words.set_fill(border_width=0)

        brace = Brace(shown_words, RIGHT)
        brace_text = brace.get_tex(R"\\text{All words, } \\sim 50\\text{k}")

        self.play(
            LaggedStartMap(FadeIn, shown_words, shift=0.5 * LEFT, lag_ratio=0.1, run_time=2),
            GrowFromCenter(brace, time_span=(0.5, 2.0)),
            FadeIn(brace_text, time_span=(0.5, 1.5)),
        )
        self.wait()

        # Show embedding matrix
        dots_index = shown_words.submobjects.index(dots)
        matrix = WeightMatrix(
            shape=(10, len(shown_words)),
            ellipses_col=dots_index
        )
        matrix.set_width(13.5)
        matrix.center()
        columns = matrix.get_columns()

        matrix_name = Text("Embedding matrix", font_size=90)
        matrix_name.next_to(matrix, DOWN, buff=0.5)

        shown_words.target = shown_words.generate_target()
        shown_words.target.rotate(PI / 2)
        shown_words.target.next_to(matrix, UP)
        for word, column in zip(shown_words.target, columns):
            word.match_x(column)
            word.rotate(-45 * DEGREES, about_edge=DOWN)
        shown_words.target[dots_index].rotate(45 * DEGREES).move_to(
            shown_words.target[dots_index - 1:dots_index + 2]
        )
        new_brace = Brace(shown_words.target, UP, buff=0.0)
        column_rects = VGroup(*(
            SurroundingRectangle(column, buff=0.05)
            for column in columns
        ))
        column_rects.set_stroke(WHITE, 1)

        self.play(
            MoveToTarget(shown_words),
            brace.animate.become(new_brace),
            brace_text.animate.next_to(new_brace, UP, buff=0.1),
            LaggedStart(*(
                Write(column, lag_ratio=0.01, stroke_width=1)
                for column in columns
            ), lag_ratio=0.2, run_time=2),
            LaggedStartMap(FadeIn, matrix.get_brackets(), scale=0.5, lag_ratio=0)
        )
        self.play(Write(matrix_name, run_time=1))
        self.wait()

        # Show a few columns
        last_rect = VMobject()
        # for index in [9, -7, 7, -5, -6]:
        for index in range(len(columns)):
            for group in shown_words, columns:
                group.target = group.generate_target()
                group.target.set_opacity(0.2)
                group.target[index].set_opacity(1)
            rect = column_rects[index]
            self.play(
                *map(MoveToTarget, [shown_words, columns]),
                FadeIn(rect),
                FadeOut(last_rect),
            )
            last_rect = rect
            self.wait(0.5)
        self.play(
            FadeOut(last_rect),
            shown_words.animate.set_opacity(1),
            columns.animate.set_opacity(1),
        )

        # Label as W_E
        frame = self.frame
        lhs = Tex("W_E = ", font_size=90)
        lhs.next_to(matrix, LEFT)

        self.play(
            frame.animate.set_width(FRAME_WIDTH + 3, about_edge=RIGHT),
            Write(lhs)
        )
        self.wait()

        # Randomize entries
        rects = VGroup(*(
            SurroundingRectangle(entry).insert_n_curves(20)
            for entry in matrix.get_entries()
            if entry not in matrix.ellipses
        ))
        rects.set_stroke(WHITE, 1)
        for x in range(1):
            self.play(
                RandomizeMatrixEntries(matrix, lag_ratio=0.01),
                LaggedStartMap(VShowPassingFlash, rects, lag_ratio=0.01, time_width=1.5),
                run_time=2,
            )
        data_modifying_matrix(self, matrix)
        self.wait()

        # Highlight just one word
        matrix_group = VGroup(lhs, matrix, shown_words, matrix_name)
        index = words.index("aardvark")
        vector = VGroup(
            matrix.get_brackets()[0],
            matrix.get_columns()[index],
            matrix.get_brackets()[1],
        ).copy()
        vector.target = vector.generate_target()
        vector.target.arrange(RIGHT, buff=0.1)
        vector.target.set_height(4.5)
        vector.target.move_to(frame, DOWN).shift(0.5 * UP)
        vector.target.set_x(-3)

        word = shown_words[index].copy()
        word.target = word.generate_target()
        word.target.rotate(-45 * DEGREES)
        word.target.scale(3)
        word.target.next_to(vector.target, LEFT, buff=1.5)
        arrow = Arrow(word.target, vector.target)

        self.play(LaggedStart(
            matrix_group.animate.scale(0.5).next_to(frame.get_top(), DOWN, 0.5),
            FadeOut(brace, UP),
            FadeOut(brace_text, 0.5 * UP),
            MoveToTarget(word),
            MoveToTarget(vector),
            GrowFromPoint(arrow, word.get_center()),
        ), lag_ratio=0.15)
        self.wait()

        word_group = VGroup(word, arrow, vector)

        # Pull the matrix back up
        self.play(
            FadeOut(word_group, DOWN),
            matrix_group.animate.scale(2.0).move_to(frame)
        )

        # Have data fly across
        data_modifying_matrix(self, matrix, word_shape=(3, 10), alpha_maxes=(0.5, 0.9))
        self.wait()

        # Prep tokens
        encoding = get_token_encoding()
        n_vocab = encoding.n_vocab
        kw = dict(font_size=24)
        shown_tokens = VGroup(
            *(Text(encoding.decode([i]), **kw) for i in range(10)),
            shown_words[dots_index].copy().rotate(-45 * DEGREES),
            *(Text(encoding.decode([i]), **kw) for i in range(n_vocab - 10, n_vocab)),
        )
        for token, word in zip(shown_tokens, shown_words):
            token.rotate(45 * DEGREES)
            token.move_to(word, DL)
        shown_tokens[dots_index].move_to(
            shown_tokens[dots_index-1:dots_index + 2:2]
        )

        # Show dimensions
        top_brace = Brace(shown_words, UP)
        left_brace = Brace(matrix, LEFT, buff=SMALL_BUFF)
        vocab_count = Integer(50257)
        vocab_label = VGroup(vocab_count, Text("words"))
        vocab_label.arrange(RIGHT, aligned_edge=UP)
        vocab_label.next_to(top_brace, UP, SMALL_BUFF)
        token_label = Text("tokens", fill_color=YELLOW)
        token_label.move_to(vocab_label[1], LEFT)

        dim_count = Integer(12288)
        dim_count.next_to(left_brace, LEFT, SMALL_BUFF)

        self.play(
            GrowFromCenter(top_brace),
            CountInFrom(vocab_count, 0),
            FadeIn(vocab_label[1]),
        )
        self.wait()
        self.play(
            FadeOut(vocab_label[1], 0.5 * UP),
            FadeIn(token_label, 0.5 * UP),
            LaggedStartMap(FadeOut, shown_words, shift=0.25 * UP, lag_ratio=0.1),
            LaggedStartMap(FadeIn, shown_tokens, shift=0.25 * UP, lag_ratio=0.1),
        )
        self.wait()

        matrix_name.target = matrix_name.generate_target()
        matrix_name.target.shift(RIGHT)
        self.play(
            MoveToTarget(matrix_name),
            lhs.animate.next_to(matrix_name.target, LEFT),
            GrowFromCenter(left_brace),
            CountInFrom(dim_count, 0),
        )
        self.play(FlashAround(dim_count))
        self.wait()

        # Count total parameters
        matrix_group = VGroup(
            top_brace, vocab_count,
            left_brace, dim_count,
            matrix, shown_words, matrix_name, lhs
        )

        top_equation = VGroup(
            Text("Total parameters = "),
            dim_count.copy(),
            Tex(R"\\times"),
            vocab_count.copy(),
            Tex("="),
            Integer(vocab_count.get_value() * dim_count.get_value()).set_color(YELLOW),
        )
        top_equation.arrange(RIGHT)
        top_equation.set_height(0.5)
        top_equation.next_to(matrix_group, UP, buff=1.0)
        result_rect = SurroundingRectangle(top_equation[-1])
        result_rect.set_stroke(YELLOW, 2)

        self.play(
            LaggedStartMap(FadeIn, top_equation[::2], shift=0.25 * UP, lag_ratio=0.5),
            TransformFromCopy(dim_count, top_equation[1]),
            TransformFromCopy(vocab_count, top_equation[3]),
            frame.animate.set_height(11).move_to(matrix_group, DOWN).shift(DOWN),
        )
        self.play(FadeTransform(
            top_equation[1:5:2].copy(), top_equation[-1]
        ))
        self.play(ShowCreation(result_rect))
        self.wait()


class Word2VecScene(InteractiveScene):
    default_frame_orientation = (-30, 70)

    axes_config = dict(
        x_range=(-5, 5, 1),
        y_range=(-5, 5, 1),
        z_range=(-4, 4, 1),
        width=8,
        height=8,
        depth=6.4,
    )
    label_rotation = PI / 2
    # embedding_model = "word2vec-google-news-300"
    embedding_model = "glove-wiki-gigaword-50"

    def setup(self):
        super().setup()

        # Load model
        self.model = get_word_to_vec_model(self.embedding_model)

        # Decide on basis
        self.basis = self.get_basis(self.model)

        # Add axes
        self.axes = ThreeDAxes(**self.axes_config)
        self.add(self.axes)

    def get_basis(self, model):
        return get_principle_components(model.vectors, 3).T

    def add_plane(self, color=GREY, stroke_width=1.0):
        axes = self.axes
        plane = NumberPlane(
            axes.x_range, axes.y_range,
            width=axes.get_width(),
            height=axes.get_height(),
            background_line_style=dict(
                stroke_color=color,
                stroke_width=stroke_width,
            ),
            faded_line_style=dict(
                stroke_opacity=0.25,
                stroke_width=0.5 * stroke_width,
            ),
            faded_line_ratio=1,
        )
        self.plane = plane
        self.add(plane)
        return plane

    def get_labeled_vector(
        self,
        word,
        coords=None,
        thickness=5,
        color=YELLOW,
        func_name: str | None = "E",
        buff=0.05,
        direction=None,
        label_config: dict = dict()
    ):
        # Return an arrow with word label next to it
        axes = self.axes
        if coords is None:
            coords = self.basis @ self.model[word.lower()]
        point = axes.c2p(*coords)
        label_config.update(label_buff=buff)
        if "label_rotation" not in label_config:
            label_config.update(label_rotation=self.label_rotation)
        arrow = LabeledArrow(
            axes.get_origin(),
            point,
            thickness=thickness,
            fill_color=color,
            label_text=word if func_name is None else f"{func_name}({word})",
            buff=0,
            direction=direction,
            **label_config,
        )
        arrow.always.set_perpendicular_to_camera(self.frame)
        return arrow


class AmbientWordEmbedding(Word2VecScene):
    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(-30, 82, 0)
        frame.add_ambient_rotation(3 * DEGREES)

        axes = self.axes
        axes.set_stroke(width=2)
        axes.set_height(7)
        axes.move_to(0.2 * FRAME_WIDTH * RIGHT + 1.0 * IN)

        # Add titles
        titles = VGroup(Text("Words"), Text("Vectors"))
        colors = [YELLOW, BLUE]
        titles.set_height(0.5)
        xs = [-4.0, axes.get_x()]
        for title, x, color in zip(titles, xs, colors):
            title.move_to(x * RIGHT)
            title.to_edge(UP)
            title.add(Underline(title))
            title.fix_in_frame()
            title.set_color(color)

        arrow = Arrow(titles[0], titles[1], buff=0.5)
        arrow.fix_in_frame()

        arrow_label = TexText("\`\`Embedding''")
        arrow_label.set_submobject_colors_by_gradient(YELLOW, BLUE)
        arrow_label.next_to(arrow, UP, SMALL_BUFF)
        arrow_label.fix_in_frame()

        self.add(titles)
        self.add(arrow)

        # Add words
        words = "All data in deep learning must be represented as vectors".split(" ")
        pre_labels = VGroup(*(Text(word) for word in words))
        pre_labels.fix_in_frame()
        pre_labels.arrange(DOWN, aligned_edge=LEFT)
        pre_labels.next_to(titles[0], DOWN, buff=0.5)
        pre_labels.align_to(titles[0][0], LEFT)
        pre_labels.set_backstroke()

        coords = np.array([
            self.basis @ self.model[word.lower()]
            for word in words
        ])
        coords -= coords.mean(0)
        max_coord = max(coords.max(), -coords.min())
        coords *= 4.0 / max_coord

        embeddings = VGroup(*(
            self.get_labeled_vector(
                word,
                coord,
                stroke_width=2,
                color=interpolate_color(BLUE_D, BLUE_A, random.random()),
                func_name=None,
                label_config=dict(font_size=24)
            )
            for word, coord in zip(words, coords)
        ))

        self.play(LaggedStartMap(FadeIn, pre_labels, shift=0.2 * UP, lag_ratio=0.1, run_time=1))

        # Transition
        self.add(turn_animation_into_updater(
            Write(arrow_label, time_span=(1, 3))
        ))
        for label, vect in zip(pre_labels, embeddings):
            self.add(turn_animation_into_updater(
                TransformFromCopy(label, vect.label, run_time=2)
            ))
            self.add(turn_animation_into_updater(
                FadeIn(vect, run_time=1)
            ))
            self.wait(0.5)
        self.play(FlashAround(arrow_label, time_width=1.5, run_time=3))
        self.wait(15)


class ThreeDSpaceExample(InteractiveScene):
    def construct(self):
        # Set up axes
        frame = self.frame
        frame.reorient(-15, 78, 0, (1.07, 1.71, 1.41), 6.72)
        frame.add_ambient_rotation(1 * DEGREES)
        axes = ThreeDAxes((-5, 5), (-5, 5), (-4, 4))
        plane = NumberPlane((-5, 5), (-5, 5))
        plane.fade(0.5)

        self.add(plane)
        self.add(axes)

        # Show coordiantes creating directions
        x, y, z = coordinates = np.array([3, 1, 2])
        colors = [RED, GREEN, BLUE]

        coords = DecimalMatrix(np.zeros((3, 1)), num_decimal_places=1)
        coords.fix_in_frame()
        coords.to_corner(UR)
        coords.shift(1.5 * LEFT)
        coords.get_entries().set_submobject_colors_by_gradient(*colors)

        lines = VGroup(
            Line(axes.c2p(0, 0, 0), axes.c2p(x, 0, 0)),
            Line(axes.c2p(x, 0, 0), axes.c2p(x, y, 0)),
            Line(axes.c2p(x, y, 0), axes.c2p(x, y, z)),
        )
        lines.set_flat_stroke(False)
        lines.set_submobject_colors_by_gradient(*colors)
        labels = VGroup(*map(Tex, "xyz"))
        labels.rotate(89 * DEGREES, RIGHT)
        directions = [OUT, OUT + RIGHT, RIGHT]
        for label, line, direction in zip(labels, lines, directions):
            label.next_to(line, direction, buff=SMALL_BUFF)
            label.match_color(line)

        dot = GlowDot(color=WHITE)
        dot.move_to(axes.get_origin())

        vect = Arrow(axes.get_origin(), axes.c2p(x, y, z), buff=0)
        vect.set_flat_stroke(False)

        self.add(coords)
        for entry, line, label, value in zip(coords.get_entries(), lines, labels, coordinates):
            rect = SurroundingRectangle(entry)
            rect.set_fill(line.get_color(), 0.3)
            rect.set_stroke(line.get_color(), width=2)
            self.play(
                ShowCreation(line),
                FadeInFromPoint(label, line.get_start()),
                VFadeInThenOut(rect),
                ChangeDecimalToValue(entry, value),
                dot.animate.move_to(line.get_end()),
            )
            self.wait(0.5)
        self.play(ShowCreation(vect))

        # Wait for a bit
        self.wait(15)

        # Show many points
        points = GlowDots(np.random.uniform(-3, 3, size=(50, 3)), radius=0.1)
        frame.clear_updaters()
        self.play(
            FadeOut(coords),
            FadeOut(dot),
            FadeOut(plane),
            LaggedStartMap(FadeOut, VGroup(*lines, vect, *labels)),
            frame.animate.reorient(-81, 61, 0, (-0.82, 0.6, 0.36), 8.95),
            ShowCreation(points),
            run_time=2,
        )
        frame.add_ambient_rotation(5 * DEGREES)
        self.wait(2)

        # Take a 2d slice
        plane = Square3D()
        plane.set_height(10)
        plane.set_color([GREY_E, GREY_C])
        plane.set_opacity(0.25)
        grid = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_color=GREY_B, stroke_width=1),
            faded_line_ratio=0,
        )
        grid.axes.match_style(grid.background_lines)
        grid.match_height(plane)
        plane_group = Group(plane, grid)
        plane_group.rotate(60 * DEGREES, UR)

        bases = [
            normalize(point)
            for point in plane.get_points()[:2]
        ]

        def project(points):
            return np.array([
                sum(np.dot(point, b) * b for b in bases)
                for point in points
            ])

        projected_points = points.copy()
        projected_points.apply_points_function(project)

        projection_lines = VGroup(*(
            Line(p1, p2)
            for p1, p2 in zip(points.get_points(), projected_points.get_points())
        ))
        projection_lines.set_stroke()

        self.play(ShowCreation(plane), Write(grid, lag_ratio=0.01, stroke_width=1))
        self.wait(2)
        self.play(
            axes.animate.set_stroke(opacity=0.25),
            points.animate.set_opacity(0.5),
            TransformFromCopy(points, projected_points),
            ShowCreation(projection_lines, lag_ratio=0.05),
            run_time=3
        )
        self.play(
            FadeOut(points),
            FadeOut(projection_lines),
            FadeOut(axes)
        )
        self.wait(15)


class HighDimensionalSpaceCompanion(InteractiveScene):
    def construct(self):
        # Vector example
        word = Text("bank")
        vect = WeightMatrix(shape=(8, 1))
        vect.next_to(word, RIGHT, buff=LARGE_BUFF)
        vect.set_height(3)
        arrow = Arrow(word, vect)
        group = VGroup(word, arrow, vect)
        group.move_to(RIGHT)
        group.to_edge(UP, buff=0.1)
        self.add(group)

        # Draw vague embedding space
        bubble_center = np.array([0.5, -2.25, 0])
        base_bubble: VMobject = OldThoughtBubble()[-2][-1]
        base_bubble.set_shape(8, 7)
        base_bubble.rotate(PI)
        base_bubble.set_fill(GREY_D, opacity=[0.25, 1, 0.25])
        base_bubble.move_to(bubble_center)
        bubble_label = Text("Word vector space", font_size=60)
        bubble_label.move_to(base_bubble)
        bubble_label.shift(2.0 * UP)
        # bubble_label = Text("Embedding space", font_size=72)
        q_marks = Tex("???", font_size=120)
        q_marks.next_to(bubble_label, DOWN, buff=0.5)
        base_bubble.add(bubble_label, q_marks)

        def get_bubble():
            result = base_bubble.copy()
            result.apply_complex_function(
                lambda z: z * (1 + 0.025 * np.cos(5 * np.log(z).imag + self.time))
            )
            result.move_to(bubble_center)
            return result

        bubble = always_redraw(get_bubble)
        self.add(bubble)
        self.wait(10)

        # Show dimension
        brace = Brace(vect, RIGHT)
        label = VGroup(
            # Integer(12288),
            Integer(10000),
            Text("coordinates")
        )
        label.arrange(DOWN, aligned_edge=LEFT)
        label.set_height(1)
        label.set_color(YELLOW)
        label.next_to(brace, RIGHT)

        dimension_label = VGroup(
            label[0].copy(),
            Text("-dimensional")
        )
        dimension_label.arrange(RIGHT, buff=0.05, aligned_edge=UP)
        dimension_label.match_height(bubble_label).scale(0.8)
        dimension_label.set_color(YELLOW)
        dimension_label.next_to(q_marks, DOWN, buff=0.5)

        self.play(
            GrowFromCenter(brace),
            CountInFrom(label[0], 0),
            FadeIn(label[1]),
        )
        self.wait()
        self.play(
            TransformFromCopy(label[0], dimension_label[0]),
            FadeInFromPoint(dimension_label[1], label[0].get_center()),
        )
        self.remove(dimension_label)
        bubble_label.add(*dimension_label)

        self.wait(10)

        # Show 3d slice
        axes = ThreeDAxes()
        axes.rotate(20 * DEGREES, OUT)
        axes.rotate(80 * DEGREES, LEFT)
        axes.set_height(3)
        axes.move_to(bubble)
        axes.shift(0.5 * RIGHT)
        axes_label = TexText("3d \`\`slice''")
        axes_label.next_to(axes, RIGHT)
        axes_label.shift(0.35 * DOWN + 1.5 * LEFT)

        self.play(
            bubble_label.animate.scale(0.5).shift(1.2 * UP + 1.5 * LEFT),
            FadeOut(q_marks),
            Write(axes, lag_ratio=0.01),
            Write(axes_label)
        )
        self.wait(2)

        # Show some vector projections
        vectors = VGroup(*(
            Arrow(
                axes.get_origin(),
                axes.c2p(*np.random.uniform(-3, 3, 3)),
                buff=0,
                stroke_color=random_bright_color(hue_range=(0.55, 0.65))
            )
            for x in range(5)
        ))
        vectors.set_flat_stroke(False)

        self.play(LaggedStartMap(FadeIn, vectors, scale=0.5, lag_ratio=0.3))
        z_direction = axes.z_axis.get_vector()
        axes.add(vectors)
        self.play(Rotate(axes, -200 * DEGREES, axis=z_direction, run_time=10))


class LearningEmbeddings(Word2VecScene):
    def construct(self):
        # Setup
        self.add_plane()
        axes = self.axes
        plane = self.plane
        frame = self.frame
        frame.reorient(0, 90, 0)

        # Get sample words
        # phrase = "The first big idea is that as a model tweaks and tunes its weights"
        # phrase = "The big idea as a model tweaks and tunes its weights"
        phrase = "Features can be encoded with directions in a big space"
        words = [word.lower() for word in phrase.split(" ")]

        # Get initial and final states
        colors = [random_bright_color(hue_range=(0.5, 0.6)) for word in words]
        true_embeddings = np.array([
            self.basis @ self.model[word]
            for word in words
        ])
        true_embeddings -= true_embeddings.mean(0)
        true_embeddings *= 5 / np.abs(true_embeddings).max(0)

        np.random.seed(2)
        thetas = np.arange(0, TAU, TAU / len(words))
        thetas += np.random.uniform(-0.5, 0.5, thetas.size)
        amps = np.random.uniform(3, 5, thetas.size)
        initial_coords = [
            rotate_vector(amp * OUT, theta, axis=UP)
            for theta, amp in zip(thetas, amps)
        ]

        # Create word vectors
        word_vects = VGroup(*(
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
                func_name=None,
                label_config=dict(font_size=36)
            )
            for word, color, coords in zip(words, colors, initial_coords)
        ))
        labels = VGroup()
        for vect in word_vects:
            label = vect.label
            label.set_backstroke(BLACK, 3)
            label.vect = vect
            label.add_updater(lambda m: m.move_to(
                m.vect.get_end() + 0.25 * normalize(m.vect.get_vector())
            ))
            labels.add(label)

        self.play(
            LaggedStartMap(GrowArrow, word_vects, lag_ratio=0.2),
            LaggedStartMap(FadeIn, labels, lag_ratio=0.2),
            run_time=4
        )
        self.wait()

        # Tweak and tune weights
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, 4, 72, 0, (-0.04, -0.18, -0.5), 8.00),
            run_time=8
        )
        self.progressive_nudges(word_vects, true_embeddings, 8)
        frame.clear_updaters()
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, 38, 69, 0, (-0.32, 0.02, -0.54), 7.68),
            run_time=12
        )
        self.progressive_nudges(word_vects, true_embeddings, 12)

    def progressive_nudges(self, word_vects, true_embeddings, n_nudges, step_size=0.2):
        for x in range(n_nudges):
            anims = [
                vect.animate.put_start_and_end_on(
                    self.axes.get_origin(),
                    interpolate(vect.get_end(), self.axes.c2p(*embedding), step_size)
                )
                for vect, embedding in zip(word_vects, true_embeddings)
            ]
            self.play(*anims, run_time=0.5)
            self.wait(0.5)


class KingQueenExample(Word2VecScene):
    default_frame_orientation = (20, 70)

    def get_basis(self, model):
        basis = super().get_basis(model)
        basis[1] *= 2
        return basis

    def construct(self):
        # Axes and frame
        axes = self.axes
        frame = self.frame
        self.add_plane()
        self.plane.rotate(90 * DEGREES, LEFT)
        frame.reorient(-178, 9, 178, (2.15, 1.12, 0.56), 6.84)

        # Initial word vectors
        words = ["man", "woman", "king", "queen"]
        colors = [BLUE_B, RED_B, BLUE_D, RED_D]
        directions = [UR, RIGHT, UR, LEFT]
        all_coords = np.array([self.basis @ self.model[word] for word in words])
        all_coords[:2] += DOWN
        all_coords[2:] += 4 * LEFT + 1 * DOWN + IN

        label_config = dict(
            font_size=30,
            label_rotation=0,
        )
        man, woman, king, queen = word_vects = [
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
                direction=direction,
                label_config=label_config,
            )
            for word, color, direction, coords in zip(words, colors, directions, all_coords)
        ]
        woman.label.shift(SMALL_BUFF * DOWN)

        fake_queen_coords = all_coords[2] - all_coords[0] + all_coords[1]  # Tweak queen for demo purposes
        fake_queen = self.get_labeled_vector(
            "queen", fake_queen_coords,
            color=colors[3],
            label_config=label_config,
        )
        fake_queen.label.shift(0.1 * LEFT + 0.2 * DOWN)

        # Equation
        equation = self.get_equation1("queen", "king", "woman", "man")
        equation.set_x(0)
        eq, minus1, ek, approx, ew, minus2, em = equation
        top_rect = FullScreenFadeRectangle().set_fill(BLACK, 1)
        top_rect.set_height(1.5, about_edge=UP, stretch=True)
        top_rect.fix_in_frame()

        for part, vect in zip([em, ew, ek, eq], word_vects):
            part.set_fill(vect.get_color())

        # Show man and woman vectors
        diff = Arrow(man.get_end(), woman.get_end(), buff=0, stroke_color=YELLOW)
        diff.set_fill(YELLOW, opacity=0.8)
        diff.set_backstroke(BLACK, 3)
        self.play(
            LaggedStart(*map(Write, [ew, minus2, em])),
            GrowArrow(woman),
            FadeInFromPoint(woman.label, man.get_center()),
            GrowArrow(man),
            FadeInFromPoint(man.label, man.get_center()),
            frame.animate.reorient(0, 0, 0, (2.04, 2.06, 0.38), 4.76).set_anim_args(run_time=6)
        )
        self.play(
            GrowArrow(diff, time_span=(1, 3)),
            frame.animate.reorient(-179, 19, 179, (2.49, 1.96, 0.4), 4.76),
            run_time=5
        )

        # Show king and fake queen
        self.add(top_rect, *equation)
        new_diff = diff.copy()
        new_diff.shift(king.get_end() - man.get_end())

        self.play(
            FadeIn(top_rect),
            *map(Write, [eq, minus1, ek, approx]),
            LaggedStart(
                TransformFromCopy(man, king),
                TransformFromCopy(man.label, king.label),
                TransformFromCopy(woman, fake_queen),
                TransformFromCopy(woman.label, fake_queen.label),
            ),
            TransformFromCopy(diff, new_diff, time_span=(2, 3)),
            frame.animate.reorient(0, 2, 0, (0.04, 1.96, -0.13), 5.51).set_anim_args(run_time=3)
        )
        self.play(
            frame.animate.reorient(-110, 10, 110, (0.22, 1.6, -0.07), 6.72),
            run_time=10
        )

        # Rearrange the equation
        for mob in [ek, approx]:
            mob.target = mob.generate_target()
        approx.target.move_to(minus1, LEFT)
        ek.target.next_to(approx.target, RIGHT)
        minus1.target = Tex("+").next_to(ek.target, RIGHT, SMALL_BUFF)
        minus1.target.move_to(midpoint(ek.target.get_right(), ew.get_left()))
        minus1.target.fix_in_frame()

        self.play(
            FadeOut(fake_queen),
            FadeOut(fake_queen.label),
            FadeOut(new_diff),
        )
        self.play(
            LaggedStartMap(MoveToTarget, [minus1, ek, approx], path_arc=PI / 2)
        )
        self.play(FlashAround(VGroup(ek, em), run_time=3, time_width=1.5))
        self.play(TransformFromCopy(diff, new_diff))

        # Search near tip
        n_circs = 5
        src_circles = Circle(radius=1e-2).set_stroke(width=5, opacity=1).replicate(n_circs)
        trg_circles = Circle(radius=1).set_stroke(width=0, opacity=1).replicate(n_circs)
        circs = VGroup(src_circles, trg_circles)
        circs.set_stroke(WHITE)
        circs.move_to(new_diff.get_end())
        self.play(
            LaggedStart(*(
                Transform(src, trg)
                for src, trg in zip(src_circles, trg_circles)
            ), lag_ratio=0.15, run_time=3)
        )
        self.play(
            FadeIn(fake_queen),
            FadeIn(fake_queen.label),
        )
        self.wait()

        # Correct it
        self.play(
            TransformFromCopy(fake_queen, queen),
            TransformFromCopy(fake_queen.label, queen.label),
            VGroup(fake_queen, fake_queen.label).animate.set_opacity(0.2),
        )
        self.play(
            FadeOut(fake_queen),
            FadeOut(fake_queen.label),
            frame.animate.reorient(103, 9, -101, (0.01, 1.45, 0.07), 6.72),
            run_time=10
        )

        # Show a few other examples
        word_pairs = [
            ("uncle", "aunt"),
            ("brother", "sister"),
            ("nephew", "niece"),
            ("father", "mother"),
            ("son", "daughter"),
        ]
        turn_animation_into_updater(
            ApplyMethod(frame.reorient, -116, 21, 114, (0.37, 1.45, 0.23), 7.59, run_time=12)
        )

        last_group = VGroup(king, queen, king.label, queen.label, new_diff)
        last_equation = equation
        for word1, word2 in word_pairs:
            new_coords = np.array([self.basis @ self.model[w] for w in [word1, word2]])
            adj_point = np.array([
                np.random.uniform(-5, 0),
                np.random.uniform(2, 4),
                np.random.uniform(-3, 3),
            ])
            new_coords += (adj_point - new_coords[0])
            vect1 = self.get_labeled_vector(word1, color=colors[2], label_config=label_config, coords=new_coords[0])
            vect2 = self.get_labeled_vector(word2, color=colors[3], label_config=label_config, coords=new_coords[1])
            vect2.put_start_and_end_on(ORIGIN, vect1.get_end() + diff.get_vector() + np.random.uniform(-0.1, 0.1, 3))
            vect2.label.next_to(vect2.get_end(), LEFT)

            new_equation = self.get_equation1(word2, word1, "woman", "man")
            new_equation.move_to(equation, RIGHT)
            new_equation.match_style(equation)
            new_equation.set_fill(opacity=1)
            new_equation.fix_in_frame()

            diff_copy = diff.copy()
            diff_copy.shift(vect1.get_end() - diff_copy.get_start())

            self.play(
                LaggedStart(
                    FadeOut(last_group),
                    GrowArrow(vect1),
                    FadeIn(vect1.label),
                    GrowArrow(vect2),
                    FadeIn(vect2.label),
                ),
                *(
                    FadeTransform(sm1, sm2)
                    for sm1, sm2 in zip(last_equation, new_equation)
                ),
            )
            self.play(TransformFromCopy(diff, diff_copy))
            self.wait(2)

            last_equation = new_equation
            last_group = VGroup(vect1, vect2, vect1.label, vect2.label, diff_copy)
        self.wait(4)

        # Flash in direction
        vect = diff.get_vector()
        color = YELLOW
        lines = Line(ORIGIN, 2 * normalize(vect)).replicate(200)
        lines.insert_n_curves(20)
        lines.set_stroke(color, 3)
        for line in lines:
            line.move_to(np.random.uniform(-3, 3, 3))
        self.play(
            LaggedStartMap(
                VShowPassingFlash, lines,
                lag_ratio=1 / len(lines),
                run_time=4
            )
        )

    def get_labeled_vector(self, *args, **kwargs):
        kwargs.update(func_name = None)
        kwargs.update(thickness=3)
        return super().get_labeled_vector(*args, **kwargs)

    def get_equation1(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            # Rf"E({word1}) - E({word2}) $\\approx$ E({word3}) - E({word4})",
            Rf"{{{word1}}} - {{{word2}}} $\\approx$ {{{word3}}} - {{{word4}}}",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"{{{word1}}}"][0],
            equation["-"][0],
            equation[f"{{{word2}}}"][0],
            equation[R"$\\approx$"][0],
            equation[f"{{{word3}}}"][0],
            equation["-"][1],
            equation[f"{{{word4}}}"][0],
        )
        pieces.fix_in_frame(True)
        return pieces

    def get_equation2(self, word1, word2, word3, word4, colors=None):
        equation = TexText(
            # Rf"E({word1}) + E({word2}) - E({word3}) $\\approx$ E({word4})",
            Rf"{{{word1}}} + {{{word2}}} - {{{word3}}} $\\approx$ {{{word4}}}",
            font_size=48
        )
        equation.fix_in_frame(True)
        equation.to_corner(UR)
        if colors:
            words = [word1, word2, word3, word4]
            for word, color in zip(words, colors):
                equation[word].set_fill(color)
        pieces = VGroup(
            equation[f"{{{word1}}}"],
            equation["+"],
            equation[f"{{{word2}}}"],
            equation["-"],
            equation[f"{{{word3}}}"],
            equation[R"$\\approx$ "],
            equation[f"{{{word4}}}"],
        )
        pieces.fix_in_frame(True)
        return pieces


class HitlerMussoliniExample(KingQueenExample):
    words = ["Hitler", "Italy", "Germany", "Mussolini"]
    colors = [GREY_C, "#008C45", "#FFCC00", GREY_B]
    default_frame_orientation = (-17, 75, 0)
    second_frame_orientation = (-24, 66, 0)
    interpolation_factor = 0.2
    diff_color = RED_B

    def get_basis(self, model):
        v1, v2, v3, v4 = [model[word.lower()] for word in self.words]
        b1 = normalize(v2 - v3)
        b2 = normalize(v1 - v3)
        b3 = normalize(get_principle_components(model.vectors)[:, 0])
        return np.array([b1, b2, b3])

    def construct(self):
        # Set up
        frame = self.frame
        frame.move_to(1.0 * UP)
        frame.add_updater(lambda f, dt: f.increment_theta(dt * 1 * DEGREES))
        axes = self.axes

        # Add equation
        equation = self.get_equation2(*self.words, colors=self.colors)
        equation.center().to_edge(UP)
        self.add(equation[:-1])

        # Initialize vectors
        v1, v2, v3, v4 = vects = [
            self.get_labeled_vector(word, color=color)
            for word, color in zip(self.words, self.colors)
        ]
        fudged_v4 = self.get_labeled_vector(
            self.words[3],
            axes.p2c(interpolate(
                v1.get_end() + v2.get_end() - v3.get_end(),
                v4.get_end(),
                self.interpolation_factor,
            )),
            color=self.colors[3]
        )
        vects[3] = fudged_v4
        for vect in vects:
            vect.apply_depth_test()

        # Show (v3 - v2) difference
        diff = Arrow(
            v3.get_end(), v2.get_end(),
            buff=0,
            stroke_color=self.diff_color,
            stroke_width=2,
            flat_stroke=False,
        )
        diff.apply_depth_test()
        rect = SurroundingRectangle(equation[2:5])
        rect.set_stroke(diff.get_stroke_color(), 2)
        self.play(
            GrowArrow(v2),
            GrowArrow(v3),
            FadeIn(v2.label),
            FadeIn(v3.label),
        )
        self.play(
            ShowCreation(rect),
            Transform(v3.copy(), v2, remover=True),
            ShowCreation(diff)
        )
        self.wait(2)

        # Add to v1
        diff_copy = diff.copy()
        diff_copy.shift(v1.get_end() - diff.get_start())

        self.play(
            GrowArrow(v1),
            FadeIn(v1.label),
            frame.animate.reorient(*self.second_frame_orientation),
        )
        self.play(
            TransformFromCopy(diff, diff_copy),
            rect.animate.surround(equation[:5])
        )
        self.wait(2)
        self.play(
            rect.animate.surround(equation[-1]),
            FadeIn(equation[-1]),
            GrowArrow(fudged_v4),
            FadeIn(fudged_v4.label),
        )
        self.play(FadeOut(rect))
        self.wait(6)

        # Emphasize directions
        italy_vect = diff.get_vector()
        axis_vect = v1.get_end() - v3.get_end()
        for vect, color in [(italy_vect, RED), (axis_vect, GREY)]:
            lines = Line(ORIGIN, 2 * normalize(vect)).replicate(200)
            lines.insert_n_curves(20)
            lines.set_stroke(color, 3)
            for line in lines:
                line.move_to(np.random.uniform(-3, 3, 3))
            self.play(
                LaggedStartMap(
                    VShowPassingFlash, lines,
                    lag_ratio=1 / len(lines),
                    run_time=4
                )
            )


class SushiBratwurstExample(HitlerMussoliniExample):
    words = ["Sushi", "Germany", "Japan", "Bratwurst"]
    colors = [WHITE, "#FFCC00", "#BC002D", interpolate_color(GREY_BROWN, WHITE, 0.25)]
    interpolation_factor = -0.1
    default_frame_orientation = (-17, 80, 0)
    second_frame_orientation = (-24, 75, 0)
    diff_color = GREY_B

    def get_basis(self, model):
        basis = super().get_basis(model)
        basis = basis[[1, 2, 0]]
        basis[1] /= -2
        basis[2] /= 3
        return basis


class SizeDirection(Word2VecScene):
    def construct(self):
        # To illustrate "You could imagine many other directions in this space corresponding to semantic meaning"

        # Set up axes
        axes = self.axes
        frame = self.frame
        self.basis *= 1.5

        # Add vectors
        frame.reorient(35, 80, 0)
        colors = [BLUE_B, BLUE_C, BLUE_D]
        word_lists = [
            ["micrometer", "millimeter", "meter"],
            ["microgram", "milligram", "gram"],
            ["microliter", "milliliter", "liter"],
        ]
        vect_groups = VGroup(
            VGroup(
                self.get_labeled_vector(word, color=color, func_name=None)
                for word, color in zip(word_list, colors)
            )
            for word_list in word_lists
        )

        over_arrow = Arrow(2 * LEFT, 2 * RIGHT).shift(UP)
        over_arrow.set_stroke(YELLOW, width=10)
        over_words = Text("Size", font_size=72)
        over_words.set_color(YELLOW)
        over_words.set_backstroke(BLACK, 5)
        over_words.next_to(over_arrow, UP)
        annotation = VGroup(over_arrow, over_words)
        annotation.shift(LEFT)
        annotation.fix_in_frame()

        for vect_group in vect_groups:
            vect_group.labels = VGroup()
            for vect in vect_group:
                vect.label.rotate(45 * DEGREES, OUT)
                vect.label.next_to(vect.get_end(), normalize(vect.get_vector()), SMALL_BUFF)
                vect_group.labels.add(vect.label)

        self.play(
            frame.animate.reorient(49, 87, 0),
            LaggedStartMap(FadeIn, vect_groups[0], lag_ratio=0.25),
            LaggedStartMap(FadeIn, vect_groups[0].labels, lag_ratio=0.25),
            FadeIn(annotation, lag_ratio=0.1, time_span=(2, 3)),
            run_time=3
        )
        self.wait()
        for i in [0, 1]:
            self.play(
                ReplacementTransform(vect_groups[i], vect_groups[i + 1]),
                ReplacementTransform(vect_groups[i].labels, vect_groups[i + 1].labels),
            )
            self.wait()


class PluralityDirection(Word2VecScene):
    def construct(self):
        self.add_plane()
        self.axes.x_axis.set_stroke(opacity=0)
        self.axes.y_axis.set_stroke(opacity=0)

        # Test
        self.frame.reorient(-21, 77, 0, (1.97, -0.73, 0.54), 3.67)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * DEGREES))
        words = ["cat", "cats"]
        all_coords = 2 * np.array([self.basis @ self.model[word] for word in words])
        colors = [BLUE, RED]
        cat, cats = [
            self.get_labeled_vector(
                word,
                coords=coords,
                color=color,
                buff=0.05,
            )
            for word, color, coords in zip(words, colors, all_coords)
        ]
        diff = Arrow(cat.get_end(), cats.get_end(), buff=0)
        diff.set_color(YELLOW)

        self.add(cat, cats)
        self.add(cat.label, cats.label)

        self.wait(5)
        self.play(ShowCreation(diff))
        self.wait(10)


class ShowNearestNeighbors(Word2VecScene):
    seed_word = "tower"
    color = YELLOW
    n_shown = 10
    frame_height = 4
    frame_center = (2.18, 0.09, 0.72)
    frame_orientation = (-21, 87, 0)
    wait_time_per_example = 0.5

    def construct(self):
        # Setup
        frame = self.frame
        frame.reorient(*self.frame_orientation, self.frame_center, self.frame_height)
        frame.add_updater(lambda f, dt: f.increment_theta(dt * DEGREES))
        self.add_plane()

        # Add seed
        word = self.seed_word.lower()
        seed_vect = self.get_labeled_vector(word, color=self.color)
        seed_group = VGroup(seed_vect, seed_vect.label)
        self.add(seed_group)

        # Add neighbors
        nearest_words = self.get_nearest_words(word)
        neighbors = VGroup(*(
            self.get_labeled_vector(
                word,
                # coords=seed_vect.get_end() + np.random.uniform(-0.5, 0.5, 3),
                color=WHITE
            )
            for word in nearest_words
        ))
        for neighbor in neighbors:
            neighbor.label.scale(0.75, about_edge=LEFT)
            neighbor.label.set_fill(border_width=0)
            neighbor.add(neighbor.label)

        # Description
        title = Text(f"Embeddings closest to E({self.seed_word})")
        underline = Underline(title)
        items = VGroup(*(
            Text(f"E({word})", font_size=36)
            for word in nearest_words
        ))
        items.arrange(DOWN, aligned_edge=LEFT)
        items.next_to(underline, DOWN, buff=0.5)
        items.align_to(title["E"][-1], LEFT)
        items.set_backstroke(BLACK, 8)

        desc = VGroup(title, underline, items)
        desc.fix_in_frame()
        desc.to_corner(UR)

        self.add(title, underline)

        # Add them all
        last_neighbor = VectorizedPoint()
        for item, neighbor in zip(items, neighbors):
            faded_last_neighbor = last_neighbor.copy()
            faded_last_neighbor.set_opacity(0.2)
            self.add(faded_last_neighbor, seed_group, neighbor)
            self.play(
                FadeIn(item),
                FadeIn(neighbor),
                FadeOut(last_neighbor),
                FadeIn(faded_last_neighbor),
            )
            last_neighbor = neighbor
            self.wait(self.wait_time_per_example)
        self.play(last_neighbor.animate.set_opacity(0.2))

        self.wait(10)

    def get_nearest_words(self, word):
        return find_nearest_words(self.model, self.model[word], self.n_shown + 1)[1:]

    def animate_in_neighbors(self, neighbors):
        # Old
        to_fade = VGroup()
        for neighbor in neighbors:
            neighbor.label.set_fill(border_width=0)
            self.add(to_fade, neighbor.label, seed_vect, seed_vect.label)
            self.play(
                FadeIn(neighbor),
                FadeIn(neighbor.label),
                to_fade.animate.set_opacity(0.25),
            )
            to_fade = VGroup(neighbor, neighbor.label)
        self.add(to_fade, neighbor.label, seed_vect, seed_vect.label)
        self.play(to_fade.animate.set_opacity(0.2))
        self.wait(5)


class ShowNearestNeighborsToWikipedia(ShowNearestNeighbors):
    seed_word = "wikipedia"
    color = BLUE
    default_frame_orientation = (10, 70)


class ShowNearestNeighborsToCat(ShowNearestNeighbors):
    seed_word = "cat"
    color = YELLOW


class ShowNearestNeighborsToNavy(ShowNearestNeighbors):
    seed_word = "navy"
    color = RED


class ShowNearestNeighborsToJump(ShowNearestNeighbors):
    seed_word = "jump"
    color = BLUE
    wait_time_per_example = 1.0
    frame_center = (2.18, -2.0, 0.0)
    random_seed = 1

    def add_plane(self):
        return VGroup()

    def get_nearest_words(self, word):
        return ["hop", "skip", "leap", "bound", "bounce", "drop", "vault"]


class DotProducts(InteractiveScene):
    def construct(self):
        # Add vectors
        plane = NumberPlane(
            (-4, 4), (-4, 4),
            background_line_style=dict(
                stroke_width=2,
                stroke_opacity=0.5,
                stroke_color=BLUE,
            ),
            faded_line_ratio=1
        )
        plane.set_height(6)
        plane.to_edge(LEFT, buff=0)
        vects = VGroup(
            Vector(0.5 * RIGHT + 2 * UP).set_stroke(MAROON_B, 6),
            Vector(1.0 * RIGHT + 0.5 * UP).set_stroke(YELLOW, 6),
        )
        vects.shift(plane.get_center())

        def get_dot_product():
            coords = np.array([plane.p2c(v.get_end()) for v in vects])
            return np.dot(coords[0], coords[1])

        self.add(plane)
        self.add(vects)

        # Vector labels
        vect_labels = VGroup(*(
            Tex(Rf"\\vec{{\\textbf{{ {char} }} }}")
            for char in "vw"
        ))
        for label, vect in zip(vect_labels, vects):
            label.vect = vect
            label.match_color(vect)
            label.add_updater(lambda l: l.move_to(
                l.vect.get_end() + 0.25 * normalize(l.vect.get_vector())
            ))

        self.add(vect_labels)

        # Add coordinate expressions
        vect_coords = VGroup(*(
            TexMatrix(
                [
                    [char + f"_{{{str(n)}}}"]
                    for n in [1, 2, 3, 4, "n"]
                ],
                bracket_h_buff=0.1,
                ellipses_row=-2,
            )
            for char in "vw"
        ))
        vect_coords.arrange(RIGHT, buff=0.75)
        vect_coords.next_to(plane, RIGHT, buff=1)
        vect_coords.set_y(1)
        for coords, vect in zip(vect_coords, vects):
            coords.get_entries().match_color(vect)
        dot = Tex(R"\\cdot", font_size=72)
        dot.move_to(vect_coords)

        self.add(vect_coords, dot)

        # Add right hand side
        rhs = Tex("= +0.00", font_size=60)
        rhs.next_to(vect_coords, RIGHT)
        result = rhs.make_number_changeable("+0.00", include_sign=True)
        result.add_updater(lambda m: m.set_value(get_dot_product()))

        self.add(rhs)

        # Add dot product label
        brace = Brace(vect_coords, DOWN, buff=0.25)
        dp_label = brace.get_text("Dot product", buff=0.25)

        self.add(brace, dp_label)

        # Play around
        def dual_rotate(angle1, angle2, run_time=2):
            self.play(
                Rotate(vects[0], angle1 * DEGREES, about_point=plane.get_origin()),
                Rotate(vects[1], angle2 * DEGREES, about_point=plane.get_origin()),
                run_time=run_time
            )

        dual_rotate(-20, 20)
        dual_rotate(50, -60)
        dual_rotate(0, 80)
        dual_rotate(20, -80)

        # Show computation
        equals = rhs[0].copy()
        entry_pairs = VGroup(*(
            VGroup(*pair)
            for pair in zip(*[vc.get_columns()[0] for vc in vect_coords])
        ))
        prod_terms = entry_pairs.copy()
        for src_pair, trg_pair in zip(entry_pairs, prod_terms):
            trg_pair.arrange(RIGHT, buff=0.1)
            trg_pair.next_to(equals, RIGHT, buff=0.5)
            trg_pair.match_y(src_pair)
        prod_terms[-2].space_out_submobjects(1e-3)
        prod_terms[-2].match_x(prod_terms)
        prod_terms.target = prod_terms.generate_target()
        prod_terms.target.space_out_submobjects(1.5).match_y(vect_coords)
        plusses = VGroup(*(
            Tex("+", font_size=48).move_to(midpoint(m1.get_bottom(), m2.get_top()))
            for m1, m2 in zip(prod_terms.target, prod_terms.target[1:])
        ))

        rhs.target = rhs.generate_target()
        rhs.target[0].rotate(PI / 2)
        rhs.target.arrange(DOWN)
        rhs.target.next_to(prod_terms, DOWN)

        self.add(equals)
        self.play(
            LaggedStart(*(
                TransformFromCopy(m1, m2)
                for m1, m2 in zip(entry_pairs, prod_terms)
            ), lag_ratio=0.1, run_time=2),
            MoveToTarget(rhs)
        )
        self.wait()
        self.play(
            MoveToTarget(prod_terms),
            rhs.animate.next_to(prod_terms.target, DOWN),
            LaggedStartMap(Write, plusses),
        )
        self.wait()

        # Positive value
        dual_rotate(-65, 65)
        self.play(FlashAround(result, time_width=1.5, run_time=3))
        self.wait()

        # Orthogonal
        elbow = Elbow(width=0.25, angle=vects[0].get_angle())
        elbow.shift(plane.get_origin())
        zero = DecimalNumber(0)
        zero.replace(result, 1)
        dual_rotate(
            (vects[1].get_angle() + PI / 2 - vects[0].get_angle()) / DEGREES,
            0,
        )
        self.remove(result)
        self.add(zero)
        self.play(ShowCreation(elbow))
        self.wait()
        self.remove(elbow, zero)
        self.add(result)

        # Negative
        dual_rotate(20, -60)
        self.play(FlashAround(result, time_width=1.5, run_time=3))
        self.wait()

        # Play again
        dual_rotate(75, -95, run_time=8)


class DotProductWithPluralDirection(InteractiveScene):
    vec_tex = R"\\vec{\\text{plur}}"
    ref_words = ["cat", "cats"]
    word_groups = [
        ["puppy", "puppies"],
        ["octopus", "octopi", "octopuses", "octopodes"],
        ["student", "students"],
        ["one", "two", "three", "four"],
    ]
    x_range = (-4, 4 + 1e-4, 0.25)
    colors = [BLUE, RED]
    threshold = -1.0

    def construct(self):
        # Initialize equation
        self.model = get_word_to_vec_model()
        word_groups = self.word_groups
        words = list(it.chain(*word_groups))

        # Write plurality equation
        gen_lhs = self.get_equation_lhs(words[0])[0].copy()
        equals = Tex(":=")
        rf1, rf2 = self.ref_words
        rhs = Tex(
            Rf"E(\\text{{{rf2}}}) - E(\\text{{{rf1}}})",
            tex_to_color_map={
                Rf"\\text{{{ref_word}}}": color
                for ref_word, color in zip(self.ref_words, self.colors)
            }
        )
        top_eq = VGroup(gen_lhs, equals, rhs)
        top_eq.arrange(RIGHT)
        gen_lhs.align_to(rhs, DOWN)
        top_eq.center().to_edge(UP, buff=0.5)

        self.play(FadeIn(rhs, UP))
        self.play(LaggedStart(
            FadeIn(equals, 0.5 * LEFT),
            FadeIn(gen_lhs, 1.0 * LEFT),
        ))
        self.wait()

        # Show on number line
        x_range = self.x_range
        number_line = NumberLine(
            x_range,
            big_tick_numbers=list(np.arange(*x_range[:2])),
            tick_size=0.05,
            longer_tick_multiple=3.0,
            width=12
        )
        number_line.rotate(PI / 2)
        number_line.add_numbers(
            np.arange(*x_range[:2]),
            num_decimal_places=1,
            font_size=40,
            direction=LEFT,
        )
        number_line.numbers.shift(SMALL_BUFF * LEFT)
        number_line.set_max_height(FRAME_HEIGHT - 1)
        number_line.to_edge(LEFT, buff=1.0)

        eq_lhs = self.get_equation_lhs(words[0])
        eq_rhs = self.get_equation_rhs(eq_lhs, words[0])
        equation = VGroup(eq_lhs, eq_rhs)
        brace = Brace(eq_lhs[2], LEFT, buff=0.1)
        brace.next_to(equation, LEFT, SMALL_BUFF, DOWN)
        equation_group = VGroup(brace, equation)
        dp = eq_rhs.get_value()

        word = eq_lhs[2][2:-1]
        lil_word = word.copy().scale(0.25)
        dot = GlowDot(color=word[0].get_color())
        dot.move_to(number_line.n2p(dp))

        lil_word.next_to(dot, RIGHT, buff=0)
        equation_group.next_to(dot, RIGHT, buff=0, submobject_to_align=brace)

        self.play(
            top_eq.animate.scale(0.75).to_corner(UR),
            TransformFromCopy(gen_lhs, eq_lhs[0]),
            FadeIn(eq_lhs[1:], shift=DOWN),
            FadeIn(brace, shift=DOWN),
            UpdateFromAlphaFunc(
                eq_rhs,
                lambda m, a: m.set_value(a * dp).next_to(eq_lhs[-1], RIGHT),
                run_time=1,
            ),
            Write(number_line, run_time=1)
        )
        self.add_dot(word, dot)
        self.wait()

        # Show some alternate
        new_rhs = eq_rhs.copy()
        eq_rhs.set_opacity(0)
        new_rhs.f_always.set_value(lambda: number_line.p2n(brace.get_center()))
        new_rhs.always.next_to(eq_lhs[-1], RIGHT)
        self.add(new_rhs)

        to_fade = Group(dot)
        for word_group in self.word_groups:
            for new_word in word_group:
                new_dp = self.get_dot_with_key_word(new_word)
                nl_point = number_line.n2p(new_dp)
                color = self.colors[int(new_dp > self.threshold)]
                new_dot = GlowDot(number_line.n2p(new_dp), color=color)
                new_lhs = self.get_equation_lhs(new_word)
                new_rhs = self.get_equation_rhs(new_lhs, new_word)
                new_rhs.set_opacity(0)
                new_equation = VGroup(new_lhs, new_rhs)
                new_equation.move_to(equation, LEFT)
                new_brace = brace.copy()
                new_equation_group = VGroup(new_brace, new_equation)
                y_shift = new_dot.get_y() - brace.get_y()
                new_equation_group.shift(y_shift * UP)

                if new_word == word_group[0]:
                    added_anim = FadeOut(to_fade)
                    to_fade = Group()
                else:
                    ghost = equation_group.copy()
                    ghost.target = ghost.generate_target()
                    ghost.target.set_fill(opacity=0.75)
                    ghost.target.scale(0.5, about_point=ghost[0].get_left())
                    added_anim = MoveToTarget(ghost)
                    to_fade.add(ghost)
                self.play(
                    Transform(equation_group, new_equation_group),
                    added_anim,
                )
                self.add_dot(new_lhs[2][2:-1], new_dot)
                to_fade.add(new_dot)

    def add_dot(self, word, dot):
        self.play(
            FadeInFromPoint(dot, word.get_center()),
            LaggedStart(
                (FadeTransform(char.copy(), dot.copy().set_opacity(0))
                for char in word),
                lag_ratio=2e-2,
                group_type=Group
            ),
            run_time=1
        )

    def get_equation_lhs(self, word):
        tex_pieces = [
            self.vec_tex, R"\\cdot", Rf"E(\\text{{{word}}})", "="
        ]
        expression = Tex(
            " ".join(tex_pieces),
            tex_to_color_map={self.vec_tex: YELLOW}
        )
        parts = [
            expression[tex_piece][0]
            for tex_piece in tex_pieces
        ]
        gen_part = parts[0]
        gen_part[0].set_width(0.75 * gen_part.get_width(), about_edge=DOWN)
        gen_part[0].shift(SMALL_BUFF * DOWN)
        value = self.get_dot_with_key_word(word)
        parts[2][2:-1].set_color(self.colors[int(value > self.threshold)])
        return VGroup(*parts)

    def get_equation_rhs(self, equation_lhs, word):
        rhs = DecimalNumber(self.get_dot_with_key_word(word))
        rhs.next_to(equation_lhs[-1], RIGHT)
        return rhs

    def get_dot_with_key_word(self, word):
        if word == "octopodes":
            return 2.3  # Hack
        elif word == "four":
            return 1.80  # To make the spacing nicer
        rf1, rf2 = self.ref_words
        return np.dot(
            (self.model[rf2] - self.model[rf1]).flatten(),
            self.model[word].flatten(),
        )


class DotProductWithGenderDirection(DotProductWithPluralDirection):
    vec_tex = R"\\vec{\\text{gen}}"
    ref_words = ["man", "woman"]
    words = [
        "mother", "father",
        "aunt", "uncle",
        "sister", "brother",
        "mama", "papa",
    ]
    x_range = (-5, 7 + 1e-4, 0.25)
    colors = [BLUE, RED]
    threshold = 1.0


class RicherEmbedding(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("The King doth wake tonight and takes his rouse ...")
        phrase.to_edge(UP)
        words = break_into_words(phrase)
        rects = get_piece_rectangles(words)
        king_index = 1

        words.fix_in_frame()
        rects.fix_in_frame()

        self.add(words)

        # Setup axes
        self.set_floor_plane("xz")
        frame = self.frame
        frame.reorient(9, -6, 0)
        axes = ThreeDAxes((-5, 5), (-2, 2), (-5, 5))
        axes.shift(DOWN)
        plane = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_width=1, stroke_color=BLUE_E),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY)
        plane.set_flat_stroke(False)
        plane.rotate(PI / 2, RIGHT)
        plane.move_to(axes)

        self.add(axes)
        self.add(plane)

        # Embed the word
        king_rect = rects[king_index]
        vector = Vector([-1, 1, 1])
        vector.shift(axes.get_origin())
        vector.match_color(king_rect)
        vector.set_flat_stroke(False)
        label = Text("King", font_size=24)
        label.next_to(vector.get_end(), normalize(vector.get_vector()), buff=0.1)

        self.play(DrawBorderThenFill(king_rect))
        self.play(
            TransformFromCopy(words[king_index], label),
            GrowArrow(vector),
        )
        self.wait(3)

        # Mention position
        index_labels = VGroup(*(
            Integer(n + 1, font_size=36).next_to(rect, DOWN, buff=0.2)
            for n, rect in enumerate(rects)
        ))
        index_labels.fix_in_frame()
        idx_vect, idx_label = self.get_added_vector(
            vector.get_end(), 0.5 * (RIGHT + OUT), "Pos. 2", TEAL,
            next_to_direction=UP,
            font_size=16
        )
        idx_label.rotate(45 * DEGREES, DOWN)
        idx_label.set_backstroke(BLACK, 1)

        self.play(
            LaggedStartMap(FadeIn, index_labels, shift=0.5 * DOWN),
        )
        self.play(
            TransformFromCopy(index_labels[king_index].set_backstroke(), idx_label),
            GrowArrow(idx_vect),
            frame.animate.reorient(-28, -22, 0).set_anim_args(run_time=3)
        )
        self.play(
            frame.animate.reorient(-11, -4, 0),
            LaggedStartMap(FadeOut, index_labels, lag_ratio=0.05, shift=0.5 * DOWN, time_span=(6, 7)),
            run_time=7
        )

        # Show king ingesting context
        self.play(
            LaggedStart(*(
                ContextAnimation(
                    words[king_index],
                    [*words[:king_index], *words[king_index + 1:]],
                    direction=DOWN,
                    fix_in_frame=True,
                    time_width=3,
                    min_stroke_width=3,
                    lag_ratio=0.05,
                    path_arc=PI / 3,
                )
                for n in range(3)
            ), lag_ratio=0.5),
            frame.animate.reorient(-5, -12, 0),
            run_time=5,
        )

        # Knock in many directions
        new_labeled_vector_args = [
            ([2, 1, 0], "lived in Scotland", None, DR),
            ([0, -1, -1], "murdered predecessor", None, RIGHT),
            ([-1.5, 1, -2], "in Shakespearean language", None, RIGHT),
        ]
        new_labeled_vects = VGroup()
        last_vect = idx_vect
        for args in new_labeled_vector_args:
            new_labeled_vects.add(self.get_added_vector(
                last_vect.get_end(), *args
            ))
            last_vect = new_labeled_vects[-1][0]
            last_vect.apply_depth_test()


        (vect1, label1), (vect2, label2), (vect3, label3) = new_labeled_vects
        self.play(
            GrowArrow(vect1),
            FadeIn(label1, 0.5 * DOWN),
            frame.animate.reorient(2, -16, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=8),
        )
        self.play(
            GrowArrow(vect2),
            FadeIn(label2, 0.5 * DOWN),
            frame.animate.reorient(35, -23, 0, (0.6, -0.04, 0.02), 6.01).set_anim_args(run_time=5),
        )
        self.play(
            GrowArrow(vect3),
            FadeIn(label3, 0.5 * DOWN),
            frame.animate.reorient(20, -29, 0, (0.61, 0.01, 0.0), 6.10).set_anim_args(run_time=5),
        )
        self.play(
            frame.animate.reorient(-19, -25, 0, (0.61, 0.01, 0.0), 6.10),
            run_time=5
        )

    def get_added_vector(self, curr_tip, direction, label, color=None, next_to_direction=UP, buff=0.1, font_size=24):
        if color is None:
            color = random_bright_color(hue_range=(0.45, 0.65))
        vect = Vector(direction)
        vect.set_color(color)
        vect.set_flat_stroke(False)
        vect.shift(curr_tip)
        text = Text(label, font_size=font_size)
        text.set_backstroke(BLACK, 4)
        text.next_to(vect.get_center(), next_to_direction, buff=buff)
        text.set_fill(border_width=0)

        result = VGroup(vect, text)
        return result


# For chapter 6

class MultipleMoleEmbeddings(Word2VecScene):
    default_frame_orientation = (0, 0)
    label_rotation = 0

    def setup(self):
        super().setup()
        self.set_floor_plane("xz")
        self.frame.add_ambient_rotation()
        self.add_plane()
        for mob in [self.plane, self.axes]:
            mob.rotate(-90 * DEGREES, RIGHT)

    def construct(self):
        # Show generic mole embedding
        frame = self.frame
        frame.reorient(-6, -6, 0, (-0.73, 1.29, -0.57), 5.27)
        phrases = VGroup(map(Text, [
            "American shrew mole",
            "One mole of carbon dioxide",
            "Take a biopsy of the mole",
        ]))
        for phrase in phrases:
            phrases.fix_in_frame()
            phrases.to_corner(UL)
            phrase["mole"][0].set_color(YELLOW)

        gen_vector = self.get_labeled_vector("mole", coords=(-2, 1.0, 1.5))
        curr_phrase = phrases[1]
        mover = curr_phrase["mole"][0]
        mover.set_backstroke(BLACK, 4)

        self.add(curr_phrase)
        self.wait()
        self.play(
            GrowArrow(gen_vector),
            TransformFromCopy(mover, gen_vector.label),
        )
        self.wait(10)

        # Show three refined meanings
        images = Group(
            ImageMobject("ShrewMole"),
            Tex(R"6.02 \\times 10^{23}", font_size=24).set_color(BLUE),
            ImageMobject("LipMole"),
        )
        for image in images[::2]:
            image.set_height(0.5)
            image.set_opacity(0.75)

        colors = [GREY_BROWN, BLUE, ORANGE]
        ref_vects = VGroup(
            self.get_labeled_vector("", coords=coords)
            for coords in [
                (-1.0, -1.5, 1.5),
                (-4.0, 0.5, 1.0),
                (-0.5, 1.0, 2.5),
            ]
        )
        for vect, image, color in zip(ref_vects, images, colors):
            vect.set_color(color)
            image.next_to(vect.get_end(), UP, SMALL_BUFF)

        gen_vect_group = VGroup(gen_vector, gen_vector.label)

        self.play(
            frame.animate.reorient(-30, -5, 0, (-1.11, 1.35, -0.72), 5.27),
            LaggedStart(
                (TransformFromCopy(gen_vector, ref_vect)
                for ref_vect in ref_vects),
                lag_ratio=0.25,
                run_time=2,
            ),
            LaggedStart(
                (FadeInFromPoint(image, gen_vector.label.get_center())
                for image in images),
                lag_ratio=0.25,
                run_time=2,
                group_type=Group,
            ),
            gen_vect_group.animate.set_opacity(0.25).set_anim_args(run_time=2),
            run_time=2,
        )
        self.wait(3)

        ref_vect_groups = Group(
            Group(*pair) for pair in zip(ref_vects, images)
        )

        # Oscillate between meanings based on context
        diff_vects = VGroup(
            Arrow(gen_vector.get_end(), ref_vect.get_end(), buff=0)
            for ref_vect in ref_vects
        )
        diff_vects.set_color(GREY_B)

        last_phrase = curr_phrase
        last_diff = VGroup()
        for n, diff in enumerate(diff_vects):
            ref_vect_groups.target = ref_vect_groups.generate_target()
            ref_vect_groups.target.set_opacity(0.2)
            ref_vect_groups.target[n].set_opacity(1)
            if n != 2:
                ref_vect_groups.target[2][1].set_opacity(0.1)
            phrase = phrases[n]
            self.play(
                gen_vect_group.animate.set_opacity(1),
                MoveToTarget(ref_vect_groups),
                FadeOut(last_phrase, UP),
                FadeIn(phrase, UP),
                FadeOut(last_diff)
            )
            self.play(
                ShowCreation(diff, time_span=(1, 2)),
                TransformFromCopy(gen_vector, ref_vects[n], time_span=(1, 2)),
                ContextAnimation(
                    phrase["mole"][0], phrase,
                    direction=DOWN,
                    fix_in_frame=True,
                ),
            )
            self.wait(3)

            last_phrase = phrase
            last_diff = diff

        self.wait(5)

    def get_basis(self, model):
        basis = super().get_basis(model) * 2
        basis[2] *= -1
        return basis


class RefineTowerMeaning(MultipleMoleEmbeddings):
    def construct(self):
        # Set up vectors and images
        frame = self.frame
        frame.reorient(-26, -4, 0, (3.27, 1.57, 0.59), 5.28)
        frame.add_ambient_rotation(0.5 * DEGREES)

        words = VGroup(Text(word) for word in "Miniature Eiffel Tower".split(" "))
        words.scale(1.25)
        words.to_edge(UP)
        words.fix_in_frame()

        tower_images = Group(
            ImageMobject(f"Tower{n}")
            for n in range(1, 5)
        )
        eiffel_tower_images = Group(
            ImageMobject(f"EiffelTower{n}")
            for n in range(1, 4)
        )
        mini_eiffel_tower_images = Group(
            ImageMobject("MiniEiffelTower1")
        )
        image_groups = Group(
            tower_images,
            eiffel_tower_images,
            mini_eiffel_tower_images
        )

        vectors = VGroup(
            self.get_labeled_vector("", coords=coords)
            for coords in [
                (4, -1, 3.0),
                (5, -2, 1.5),
                (-3, -1, 2.5),
            ]
        )
        colors = [BLUE_D, GREY_B, GREY_C]
        for vector, color, image_group in zip(vectors, colors, image_groups):
            vector.set_color(color)
            for image in image_group:
                image.set_height(1.5)
                image.next_to(vector.get_end(), RIGHT * np.sign(vector.get_end()[0]))

        # Show tower
        tower = words[-1]
        tower.set_x(0)
        pre_tower_image = tower_images[0].copy()
        pre_tower_image.fix_in_frame()
        pre_tower_image.replace(tower, stretch=True)
        pre_tower_image.set_opacity(0)

        self.add(tower)
        self.wait()
        self.play(
            GrowArrow(vectors[0]),
            ReplacementTransform(pre_tower_image, tower_images[0]),
            run_time=2,
        )
        for ti1, ti2 in zip(tower_images, tower_images[1:]):
            self.play(
                FadeTransform(ti1, ti2),
                run_time=2
            )
        self.wait(2)

        # Eiffel tower
        words[:-1].set_opacity(0)
        eiffel_tower = words[-2:]

        self.play(
            frame.animate.reorient(-4, -7, 0, (2.95, 1.82, 0.49), 6.59),
            eiffel_tower.animate.set_opacity(1).arrange(RIGHT, aligned_edge=DOWN).to_edge(UP),
        )
        self.play(
            vectors[0].animate.set_opacity(0.25),
            tower_images[-1].animate.set_opacity(0.2),
            TransformFromCopy(vectors[0], vectors[1]),
            FadeTransform(tower_images[-1].copy(), eiffel_tower_images[0]),
            ContextAnimation(words[2], words[1], direction=DOWN, fix_in_frame=True),
            run_time=2,
        )
        for ti1, ti2 in zip(eiffel_tower_images, eiffel_tower_images[1:]):
            self.play(
                FadeTransform(ti1, ti2),
                run_time=2
            )
        self.wait(6)

        # Miniature eiffel tower
        self.play(
            frame.animate.reorient(-14, -2, 0, (-0.12, 2.21, 0.72), 7.05).set_anim_args(run_time=2),
            words.animate.set_opacity(1).arrange(RIGHT, aligned_edge=DOWN).to_edge(UP),
        )
        self.play(
            vectors[1].animate.set_opacity(0.25),
            eiffel_tower_images[-1].animate.set_opacity(0.2),
            TransformFromCopy(vectors[1], vectors[2]),
            FadeTransform(eiffel_tower_images[-1].copy(), mini_eiffel_tower_images[0]),
            ContextAnimation(words[2], words[0], direction=DOWN, fix_in_frame=True),
            run_time=2,
        )
        self.wait(10)


class UpdatingPoetryEmbedding(RicherEmbedding):
    def construct(self):
        # (Largely copied from RicherEmbedding, could factor better later)
        # Add phrase
        poem_str = "...\\nTwo roads diverged in a wood, and I—\\nI took the one less traveled by,"
        phrase = Text(poem_str, alignment="LEFT")
        phrase[:3].rotate(PI / 2).shift(SMALL_BUFF * UP)
        phrase.refresh_bounding_box()
        phrase.to_edge(UP, buff=SMALL_BUFF)
        words = break_into_words(phrase)
        rects = get_piece_rectangles(words)

        words.fix_in_frame()
        rects.fix_in_frame()

        self.add(words)

        # Setup axes
        self.set_floor_plane("xz")
        frame = self.frame
        frame.reorient(9, -6, 0)
        frame.reorient(9, -1, 0, 0.75 * UP)
        axes = ThreeDAxes((-5, 5), (-2, 2), (-5, 5))
        axes.shift(DOWN)
        plane = NumberPlane(
            (-5, 5), (-5, 5),
            background_line_style=dict(stroke_width=1, stroke_color=BLUE_E),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY)
        plane.set_flat_stroke(False)
        plane.rotate(PI / 2, RIGHT)
        plane.move_to(axes)

        self.add(axes)
        self.add(plane)

        # Embed the word
        one_index = len(words) - 4
        one_rect = SurroundingRectangle(words[one_index])
        one_rect.set_fill(GREEN, 0.2)
        one_rect.set_stroke(GREEN, 2)
        one_rect.fix_in_frame()
        vector = Vector([-3, 1, 2])
        vector.shift(axes.get_origin())
        vector.match_color(one_rect)
        vector.set_flat_stroke(False)
        label = Text("one", font_size=36)
        label.next_to(vector.get_end(), normalize(vector.get_vector()), buff=0.1)

        self.play(DrawBorderThenFill(one_rect))
        self.play(
            TransformFromCopy(words[one_index], label),
            GrowArrow(vector),
        )
        self.wait(3)

        # Knock in many directions
        new_labeled_vector_args = [
            ([2, 1, 0], "of two roads", None, UL),
            ([2, -1, -1], "symbolizing choice", None, UR),
            ([0.5, 1, -3], "contrasting the original\\nwith the familiar", None, DR),
        ]
        new_labeled_vects = VGroup()
        last_vect = vector
        for args in new_labeled_vector_args:
            new_labeled_vects.add(self.get_added_vector(
                last_vect.get_end(), *args
            ))
            last_vect = new_labeled_vects[-1][0]
            last_vect.apply_depth_test()
        orientation_args = [
            (-4, -12, 0, (-0.89, 0.03, -0.41), 8.10),
            (3, -9, 0, (-0.34, 0.49, -0.63), 8.60),
            (34, -14, 0, (-0.59, 0.49, -0.62), 9.20),
            (20, -29, 0, (0.61, 0.01, 0.0), 6.10),
        ]


        for (vect, label), orientation in zip(new_labeled_vects, orientation_args):
            self.play(
                GrowArrow(vect, time_span=(2, 3)),
                FadeIn(label, 0.5 * DOWN, time_span=(2, 3)),
                frame.animate.reorient(*orientation).set_anim_args(run_time=6),
                ContextAnimation(
                    one_rect, phrase[:-16],
                    run_time=4,
                    fix_in_frame=True,
                    path_arc=60 * DEGREES,
                    lag_ratio=1e-3,
                    direction=UP,
                ),
            )
        self.play(
            frame.animate.reorient(22, -23, 0, (-0.86, 0.4, -0.35), 7.15),
            run_time=5
        )


# For chapter 7

class SimpleSpaceExample(InteractiveScene):
    def construct(self):
        # Setup axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        frame.reorient(14, 77, 0, (2.23, 0.25, 1.13), 4.46)

        # Show an initial vector in the space
        frame.add_ambient_rotation()
        vect = Arrow(axes.c2p(0, 0, 0), axes.c2p(2, -1, 1), buff=0)
        vect.set_color(BLUE)
        vect.always.set_perpendicular_to_camera(self.frame)
        label = Text("you", font_size=24)
        # label = Text("Photo", font_size=24).set_backstroke(BLACK, 5)
        label.rotate(PI / 2, RIGHT)
        label.next_to(vect.get_center(), OUT + LEFT, buff=0)

        self.play(
            ShowCreation(vect),
            FadeIn(label, vect.get_vector())
        )
        self.wait(5)

        # Many directions -> Different kinds of meaning
        ideas = VGroup(
            Text("Part of a command"),
            Text("Affectionate"),
            Text("Sadness"),
        )
        ideas.set_backstroke(BLACK, 3)
        ideas.scale(0.35)
        ideas.rotate(PI / 2, RIGHT)

        last_idea = VGroup()
        last_direction = 1.0 * normalize(cross(RIGHT, vect.get_vector()))
        for idea in ideas:
            direction = rotate_vector(last_direction, PI / 3, vect.get_vector())
            new_vect = self.get_added_vector(vect, direction)
            new_vect.set_perpendicular_to_camera(self.frame)
            idea.next_to(new_vect.get_center(), buff=0.1)
            lines = get_direction_lines(axes, new_vect.get_vector(), color=new_vect.get_color())
            self.play(
                FadeOut(last_idea),
                ShowCreation(new_vect),
                FadeIn(idea, new_vect.get_vector()),
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=2 / len(lines), run_time=2)
            )
            self.wait(1)
            last_idea = VGroup(new_vect, idea)
            last_direction = direction
        self.play(FadeOut(last_idea))
        self.wait(5)

        # Specific ideas added onto "you"
        ideas = VGroup(
            # Text("Astronaut"),
            # Text("Riding a Horse"),
            # Text("On the moon"),
            #
            Text("needs an adjective next"),
            Text("preceded by \\"that which does not kill\\""),
            Text("related to growth and strength"),
            #
            # Text("River bank"),
            # Text("Beginning of a story"),
            # Text("Establishing a setting"),
        )
        ideas.scale(0.4)
        ideas.rotate(PI / 2, RIGHT)
        directions = [
            (-0.25, -1, 0.75),
            (-0.5, -0.25, 0.5),
            (1.0, -0.5, 1.0),
        ]
        orientations = [
            (11, 92, 0, (2.69, 0.55, 1.12), 6.25),
            (-8, 83, 0, (2.73, 0.56, 1.24), 6.80),
            (-14, 79, 0, (2.49, 0.61, 1.41), 7.64),
        ]

        vects = VGroup(vect)
        concepts = VGroup(label)
        for idea, direction, orientation in zip(ideas, directions, orientations):
            point = vects[-1].get_end()
            new_vect = self.get_added_vector(vects[-1], direction)
            new_vect.always.set_perpendicular_to_camera(self.frame)
            idea.next_to(new_vect.get_center())
            self.play(
                frame.animate.reorient(*orientation),
                GrowArrow(new_vect),
                FadeIn(idea, 0.5 * new_vect.get_vector())
            )
            self.wait(2)
            vects.add(new_vect)
        self.wait(15)

    def add_plane_and_axes(
        self,
        x_range=(-4, 4),
        y_range=(-4, 4),
        z_range=(-3, 3),
    ):
        axes = ThreeDAxes(x_range, y_range, z_range)
        plane = NumberPlane(
            x_range, y_range,
            background_line_style=dict(
                stroke_color=GREY_D,
                stroke_width=1
            ),
            faded_line_ratio=1,
        )
        plane.axes.set_stroke(GREY_D, 0)

        self.add(plane, axes)
        return plane, axes

    def get_added_vector(self, last_vect, direction):
        point = last_vect.get_end()
        new_vect = Arrow(point, point + direction, buff=0)
        new_vect.set_color(random_bright_color())
        new_vect.set_flat_stroke(False)
        return new_vect


class ManyIdeasManyDirections(SimpleSpaceExample):
    random_seed = 2

    def construct(self):
        # Axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        frame.reorient(-17, 73, 0, (-0.06, 0.11, 0.31), 6.03)
        frame.add_ambient_rotation()

        # Many directions -> Different kinds of meaning
        ideas = VGroup(
            Text(word)
            for word in [
                "Typewriter",
                "Paradigm",
                "Whimsical",
                "Gelatinous",
                "Rainbow",
                "Serendipitous",
                "Algorithm",
                "Nebulous",
                "Spatula",
                "Lethargic",
                "Effervescent",
                "Asteroid",
                "Pungent",
                "Daydream",
                "Mercurial",
                "Cactus",
                "Diaphanous",
                "Hiccup",
                "Viscous",
                "Thunderclap",
            ]
        )
        ideas.set_backstroke(BLACK, 3)
        ideas.scale(0.5)
        ideas.rotate(PI / 2, RIGHT)

        last_idea = VGroup()
        last_direction = RIGHT + OUT
        for idea in ideas:
            direction = normalize(cross(last_direction, np.random.uniform(-1, 1, 3)))
            new_vect = Vector(direction)
            new_vect.set_perpendicular_to_camera(self.frame)
            new_vect.set_color(random_bright_color())
            idea.next_to(new_vect.get_end(), direction, buff=0.1)
            lines = get_direction_lines(axes, direction, color=new_vect.get_color(), n_lines=250, stroke_width=2)
            idea.set_fill(interpolate_color(new_vect.get_color(), WHITE, 0.5))
            self.play(
                FadeOut(last_idea),
                GrowArrow(new_vect),
                FadeIn(idea, new_vect.get_vector()),
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=1 / len(lines), run_time=1.5)
            )
            self.wait()
            last_idea = VGroup(new_vect, idea)
            last_direction = direction
        self.play(FadeOut(last_idea))
        self.wait(5)


class MJSpace(SimpleSpaceExample):
    def construct(self):
        # Set up axes
        frame = self.frame
        plane, axes = self.add_plane_and_axes()
        axes.set_stroke(width=1)
        frame.add_ambient_rotation()

        # Show vectors landing in the space
        sentence = Text("Michael Jordan plays the sport of basketball", font_size=36)
        sentence.to_edge(UP)
        tokens = break_into_tokens(sentence)
        token_rects = get_piece_rectangles(tokens, leading_spaces=True, h_buff=0)
        arrs = VGroup(
            NumericEmbedding().scale(0.25).next_to(rect, DOWN, buff=1.0)
            for rect in token_rects
        )
        arrows = VGroup(Arrow(rect, arr, buff=0.1) for rect, arr in zip(token_rects, arrs))
        vects = VGroup(
            Vector(np.random.uniform(-3, 3, 3))
            for arr in arrs
        )
        vects.set_stroke(GREY_B)
        vects.fix_in_frame()

        VGroup(token_rects, tokens, arrows, arrs).fix_in_frame()

        frame.reorient(-18, 86, 0, (0.21, 0.12, 3.56), 11.65)
        self.add(token_rects, tokens)
        self.play(
            LaggedStartMap(FadeIn, arrs, shift=DOWN, lag_ratio=0.1),
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            frame.animate.reorient(11, 76, 0, ORIGIN, FRAME_HEIGHT),
            FadeOut(VGroup(token_rects, tokens, arrows), UP, time_span=(1, 2)),
            LaggedStart(
                (Transform(arrow, vect)
                for arrow, vect in zip(arrs, vects)),
                lag_ratio=0.05,
            ),
            run_time=3
        )
        self.remove(arrs)
        self.add(vects)
        self.wait()
        self.play(LaggedStart(
            (vect.animate.scale(0, about_point=vect.get_start())
            for vect in vects),
            lag_ratio=0.05,
            remover=True
        ))

        # Show three directions
        colors = [YELLOW, RED, "#F88158"]
        all_coords = [normalize([-1, -1, 1])]
        all_coords.append(normalize(cross(all_coords[0], IN)))
        all_coords.append(-normalize(cross(all_coords[0], all_coords[1])))
        all_coords = np.array(all_coords)[[0, 2, 1]]
        labels = VGroup(*map(Text, ["First Name Michael", "Last Name Jordan", "Basketball"]))
        label_directions = [LEFT + OUT, IN, RIGHT + OUT]

        vect_groups = VGroup()
        vects = VGroup()
        for coords, label, color, direction in zip(all_coords, labels, colors, label_directions):
            vect = Vector(2.0 * coords)
            vect.set_color(color)
            vect.always.set_perpendicular_to_camera(self.frame)
            label.scale(0.5)
            label.rotate(PI / 2, RIGHT)
            label.set_color(color)
            label.next_to(vect.get_end(), direction, buff=0.1)
            label.set_fill(border_width=0.5)
            label.set_backstroke(BLACK, 4)
            vects.add(vect)
            vect_groups.add(VGroup(vect, label))

        orientations = [
            (17, 76, 0),
            (17, 80, 0),
            (-16, 77, 0),
        ]

        for vect, label, orientation in zip(vects, labels, orientations):
            lines = get_direction_lines(axes, vect.get_vector(), color=vect.get_color())
            self.play(
                GrowArrow(vect),
                FadeIn(label, vect.get_vector()),
                frame.animate.reorient(*orientation),
            )
            self.play(
                LaggedStartMap(ShowCreationThenFadeOut, lines, lag_ratio=2 / len(lines))
            )
            self.wait(2)

        # Bring in "plucked out" vector
        emb_coords = 2.0 * all_coords[:2].sum(0)
        emb = Vector(emb_coords)
        emb.always.set_perpendicular_to_camera(self.frame)
        emb.set_flat_stroke(False)
        emb_label = Tex(R"\\vec{\\textbf{E}}", font_size=30)
        emb_label.rotate(89 * DEGREES, RIGHT)
        emb_label.add_updater(lambda m: m.move_to(1.1 * emb.get_end()))
        emb_label.suspend_updating()

        self.play(
            frame.animate.reorient(7, 66, 0).set_anim_args(run_time=2),
            FadeIn(emb, shift=2 * (IN + LEFT)),
            FadeIn(emb_label, shift=2 * (IN + LEFT)),
        )
        self.wait()

        # Set up dot product display
        def get_proj_point(vect1, vect2):
            v1 = vect1.get_end()
            v2 = vect2.get_end()
            return v2 * np.dot(v1, v2) / np.dot(v2, v2)

        def get_dot_product_lines(vect, proj_line_color=GREY_A):
            dashed_line = always_redraw(
                lambda: Line(emb.get_end(), get_proj_point(emb, vect)).set_stroke(WHITE, 2).set_anti_alias_width(10)
            )
            proj_line = always_redraw(
                lambda: Line(ORIGIN, get_proj_point(emb, vect)).set_stroke(proj_line_color, width=4, opacity=0.75)
            )
            return dashed_line, proj_line

        m_dashed_line, m_proj_line = get_dot_product_lines(vects[0])

        formula = Tex(R"\\vec{\\textbf{E}} \\cdot \\big(\\overrightarrow{\\text{First Name Michael}}\\big) = ", font_size=36)
        formula[3:-1].set_color(YELLOW)
        formula.to_corner(UL)
        formula.fix_in_frame()
        rhs = DecimalNumber(font_size=42)
        rhs.fix_in_frame()
        rhs.next_to(formula[-1], RIGHT, buff=0.15)
        rhs.target_vect = vects[0]
        rhs.add_updater(lambda m: m.set_value(np.dot(m.target_vect.get_end(), emb.get_end()) / 4.0))

        m_proj_line.suspend_updating()
        self.play(
            ShowCreation(m_dashed_line),
            TransformFromCopy(Line(ORIGIN, emb.get_end(), flat_stroke=False), m_proj_line),
            FadeIn(formula, UP),
            vect_groups[1:].animate.set_opacity(0.25),
        )
        m_proj_line.resume_updating()
        self.play(
            TransformFromCopy(rhs.copy().unfix_from_frame().set_opacity(0).move_to(m_proj_line), rhs),
        )
        emb_label.resume_updating()
        for _ in range(2):
            self.play(
                emb.animate.put_start_and_end_on(ORIGIN, [-2.5, -2.0, -0.5]),
                rate_func=wiggle,
                run_time=5
            )
        self.wait(2)
        self.play(emb.animate.put_start_and_end_on(axes.get_origin(), 1.5 * all_coords[1:3].sum(0)), run_time=3)
        self.play(frame.animate.reorient(26, 68, 0), run_time=2 )
        self.play(emb.animate.put_start_and_end_on(ORIGIN, [1.0, -1.5, -1.0]), run_time=3)
        self.wait(2)
        self.play(
            frame.animate.reorient(-4, 73, 0),
            emb.animate.put_start_and_end_on(ORIGIN, emb_coords),
            run_time=3
        )
        self.wait(5)

        # Dotting against L.N. Jordan
        j_dashed_line, j_proj_line = get_dot_product_lines(vects[1])
        j_paren = Tex(R"\\big(\\overrightarrow{\\text{Last Name Jordan}}\\big) = ", font_size=36)
        j_paren[:-1].set_color(RED)
        m_paren = formula[3:]
        m_paren.fix_in_frame()
        j_paren.move_to(m_paren, LEFT)
        j_paren.fix_in_frame()
        rhs.target_vect = vects[1]

        self.play(
            frame.animate.reorient(15, 97, 0),
            FadeOut(m_paren, UP, time_span=(1, 2)),
            FadeIn(j_paren, UP, time_span=(1, 2)),
            rhs.animate.next_to(j_paren, RIGHT, buff=0.15).set_anim_args(time_span=(1, 2)),
            LaggedStart(
                vect_groups[0].animate.set_opacity(0.25),
                vect_groups[1].animate.set_opacity(1),
                FadeOut(m_dashed_line),
                FadeOut(m_proj_line),
                lag_ratio=0.25,
                run_time=2
            )
        )
        j_proj_line.suspend_updating()
        self.play(
            ShowCreation(j_dashed_line),
            TransformFromCopy(Line(ORIGIN, emb.get_end(), flat_stroke=False), j_proj_line),
        )
        j_proj_line.resume_updating()
        self.play(
            emb.animate.put_start_and_end_on(ORIGIN, [-1.5, -1.5, 0]).set_anim_args(run_time=3, rate_func=there_and_back)
        )
        self.wait()

        # Dotting against basketball
        b_dashed_line, b_proj_line = get_dot_product_lines(vects[2])
        b_paren = Tex(R"\\big(\\overrightarrow{\\text{Basketball}}\\big) = ", font_size=36)
        b_paren[:-1].set_color(vects[2].get_color())
        b_paren.move_to(m_paren, LEFT)
        b_paren.fix_in_frame()
        rhs.suspend_updating()

        self.play(
            frame.animate.reorient(2, 65, 0),
            FadeOut(j_paren, UP),
            FadeIn(b_paren, UP),
            rhs.animate.next_to(b_paren[-1], RIGHT, buff=0.2).set_value(0),
            FadeOut(j_dashed_line),
            FadeOut(j_proj_line),
            vect_groups[1].animate.set_opacity(0.25),
            vect_groups[2].animate.set_opacity(1.0),
        )
        self.wait()

        rhs.target_vect = vects[2]
        rhs.resume_updating()
        self.add(b_dashed_line, b_proj_line)
        self.play(
            emb.animate.put_start_and_end_on(ORIGIN, [0.6, -2.2, 0]),
            rate_func=there_and_back,
            run_time=6,
        )
        self.wait(3)

        # Emphasize dot products with first two names
        self.play(
            frame.animate.reorient(5, 85, 0).set_anim_args(run_time=2),
            FadeOut(formula[:3]),
            FadeOut(b_paren),
            FadeOut(rhs),
            FadeOut(b_dashed_line),
            FadeOut(b_proj_line),
            vect_groups[:2].animate.set_opacity(1),
            vect_groups[2].animate.set_opacity(0.25),
        )
        self.wait()
        self.play(
            ShowCreation(m_dashed_line),
            ShowCreation(m_proj_line),
        )
        self.wait()
        self.play(
            ShowCreation(j_dashed_line),
            ShowCreation(j_proj_line),
        )
        self.wait(20)
        self.play(
            *map(FadeOut, [j_dashed_line, j_proj_line, m_dashed_line, m_proj_line, emb, emb_label]),
        )

        # Show sum of the first two names
        j_vect_copy, m_vect_copy = vect_copies = vects[:2].copy()
        vect_copies.clear_updaters()
        vect_copies.set_stroke(opacity=0.5)
        j_vect_copy.shift(vects[1].get_vector())
        m_vect_copy.shift(vects[0].get_vector())
        emb.put_start_and_end_on(axes.get_origin(), m_vect_copy.get_end())

        self.play(frame.animate.reorient(-6, 78, 0), run_time=2)
        self.play(LaggedStart(
            TransformFromCopy(vects[1], m_vect_copy),
            TransformFromCopy(vects[0], j_vect_copy),
            lag_ratio=0.5
        ))
        self.play(GrowArrow(emb))
        self.wait(4)

        # Show the basketball direction
        self.play(
            *map(FadeOut, [m_vect_copy, j_vect_copy, emb])
        )
        self.play(
            frame.animate.reorient(-19, 77, 0, (1.32, -0.22, -0.12), 3.75),
            vect_groups[:2].animate.set_opacity(0.25),
            vect_groups[2][0].animate.set_opacity(1.0),
            vect_groups[2][1].animate.set_opacity(1.0),
            run_time=2
        )
        self.wait(20)`,
    annotations: {
      5: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      6: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      104: "Subdivides existing bezier curves to increase point density. Needed before applying nonlinear transformations for smooth results.",
      108: "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation.",
      119: "LyingAboutTokens2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      120: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      122: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      128: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      135: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      140: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      141: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      142: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      143: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      146: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      147: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      148: "FadeOut transitions a mobject from opaque to transparent.",
      149: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      150: "DrawBorderThenFill first draws the stroke outline, then fills the interior.",
      153: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      154: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      157: "FadeOut transitions a mobject from opaque to transparent.",
      159: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      180: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      184: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      186: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      187: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      188: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      190: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      195: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      260: "DiscussTokenization extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      261: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      265: "ImageTokens extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      268: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      299: "SoundTokens extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      300: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      341: "IntroduceEmbeddingMatrix extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      342: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      612: "Word2VecScene extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      696: "Class AmbientWordEmbedding inherits from Word2VecScene.",
      697: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      778: "ThreeDSpaceExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      779: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      906: "HighDimensionalSpaceCompanion extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      907: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1019: "Class LearningEmbeddings inherits from Word2VecScene.",
      1020: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1107: "Class KingQueenExample inherits from Word2VecScene.",
      1115: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1385: "Class HitlerMussoliniExample inherits from KingQueenExample.",
      1400: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1495: "Class SushiBratwurstExample inherits from HitlerMussoliniExample.",
      1511: "Class SizeDirection inherits from Word2VecScene.",
      1512: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1569: "Class PluralityDirection inherits from Word2VecScene.",
      1570: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1601: "Class ShowNearestNeighbors inherits from Word2VecScene.",
      1610: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1694: "Class ShowNearestNeighborsToWikipedia inherits from ShowNearestNeighbors.",
      1700: "Class ShowNearestNeighborsToCat inherits from ShowNearestNeighbors.",
      1705: "Class ShowNearestNeighborsToNavy inherits from ShowNearestNeighbors.",
      1710: "Class ShowNearestNeighborsToJump inherits from ShowNearestNeighbors.",
      1724: "DotProducts extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1725: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1885: "DotProductWithPluralDirection extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1898: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2067: "Class DotProductWithGenderDirection inherits from DotProductWithPluralDirection.",
      2081: "RicherEmbedding extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2082: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2232: "Class MultipleMoleEmbeddings inherits from Word2VecScene.",
      2244: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2365: "Class RefineTowerMeaning inherits from MultipleMoleEmbeddings.",
      2366: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2470: "Class UpdatingPoetryEmbedding inherits from RicherEmbedding.",
      2471: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2571: "SimpleSpaceExample extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2572: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2695: "Class ManyIdeasManyDirections inherits from SimpleSpaceExample.",
      2698: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2758: "Class MJSpace inherits from SimpleSpaceExample.",
      2759: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/generation.py"] = {
    description: "Token generation and sampling scenes. Visualizes temperature, top-k, and nucleus sampling strategies for controlling transformer text output.",
    code: `from transformers.models.videomae import image_processing_videomae
from manim_imports_ext import *
from _2024.transformers.helpers import *

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import PreTrainedModel
import torch
import openai
import tiktoken


@lru_cache(maxsize=1)
def get_gpt2_tokenizer(model_name='gpt2'):
    return GPT2Tokenizer.from_pretrained(model_name)


@lru_cache(maxsize=1)
def get_gpt2_model(model_name='gpt2'):
    return GPT2LMHeadModel.from_pretrained(model_name)


def gpt2_predict_next_token(text, n_shown=7):
    tokenizer = get_gpt2_tokenizer()
    model = get_gpt2_model()
    # Encode the input text
    indexed_tokens = tokenizer.encode(
        text, add_special_tokens=False, return_tensors='pt'
    )

    # Predict all tokens
    with torch.no_grad():
        outputs = model(indexed_tokens)
        # Pull out the first batch, and the last token prediction
        predictions = outputs[0][0, -1, :]

    # Get the predicted next token
    indices = torch.argsort(predictions)
    top_indices = reversed(indices[-n_shown:])
    tokens = list(map(tokenizer.decode, top_indices))
    probs = softmax(predictions)[top_indices]

    return tokens, probs


def gpt3_predict_next_token(text, n_shown=10, random_seed=0):
    openai.api_key = os.getenv('OPENAI_KEY')
    response = openai.Completion.create(
        # Or another model version, adjust as necessary
        engine="gpt-3.5-turbo-instruct",
        prompt=text,
        max_tokens=1,
        n=1,
        temperature=1.0,
        user=str(random_seed),
        logprobs=50  # I think this is actually set to a max of 20?
    )
    top_logprob_dict = response.choices[0]["logprobs"]["top_logprobs"][0]
    tokens, logprobs = zip(*top_logprob_dict.items())
    probs = np.exp(logprobs)
    indices = np.argsort(-probs)
    shown_tokens = [tokens[i] for i in indices[:n_shown]]
    return shown_tokens, probs[indices[:n_shown]]


def clean_text(text):
    return " ".join(filter(lambda s: s.strip(), re.split(r"\\s", text)))


def next_token_bar_chart(
    words, probs,
    reference_point=ORIGIN,
    font_size=24,
    width_100p=1.0,
    prob_exp=0.75,
    bar_height=0.25,
    bar_space_factor=0.5,
    buff=1.2,
    show_ellipses=True,
    use_percent=True,
):
    labels = VGroup(Text(word, font_size=font_size) for word in words)
    bars = VGroup(
        Rectangle(prob**(prob_exp) * width_100p, bar_height)
        for prob, label in zip(probs, labels)
    )
    bars.arrange(DOWN, aligned_edge=LEFT, buff=bar_space_factor * bar_height)
    bars.set_fill(opacity=1)
    bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
    bars.set_stroke(WHITE, 1)

    bar_groups = VGroup()
    for label, bar, prob in zip(labels, bars, probs):
        if use_percent:
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
        else:
            prob_label = DecimalNumber(prob, font_size=0.75 * font_size)
        prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
        label.next_to(bar, LEFT)
        bar_groups.add(VGroup(label, bar, prob_label))

    if show_ellipses:
        ellipses = Tex(R"\\vdots", font_size=font_size)
        ellipses.next_to(bar_groups[-1][0], DOWN)
        bar_groups.add(ellipses)

    bar_groups.shift(reference_point - bars.get_left() + buff * RIGHT)

    return bar_groups


class SimpleAutogregression(InteractiveScene):
    text_corner = 3.5 * UP + 0.75 * RIGHT
    line_len = 31
    font_size = 35
    n_shown_predictions = 12
    seed_text = "Behold, a wild pi creature, foraging in its native"
    seed_text_color = BLUE_B
    machine_name = "Transformer"
    machine_phi = 10 * DEGREES
    machine_theta = 12 * DEGREES
    n_predictions = 120
    skip_through = False
    random_seed = 0
    model = "gpt2"

    def construct(self):
        # Repeatedly generate
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=(n > 10),
                skip_anims=self.skip_through,
            )

    def init_text_and_machine(self):
        # Set up active text
        self.cur_str = self.seed_text
        text_mob = self.string_to_mob(self.cur_str)
        text_mob.set_color(self.seed_text_color)
        next_word_line = self.get_next_word_line(text_mob)

        # Set up Transformer as some sort of machine
        machine = self.get_transformer_drawing()
        machine.set_y(0).to_edge(LEFT, buff=-0.6)

        self.add(text_mob)
        self.add(next_word_line)
        self.add(machine)

        return text_mob, next_word_line, machine

    def string_to_mob(self, text):
        text += " l"  # Dumb hack for alignment
        result = get_paragraph(
            text.replace("\\n", " ").split(" "),
            self.line_len,
            self.font_size
        )
        result.move_to(self.text_corner, UL)
        result[-1].set_fill(BLACK, 0)  # Continue dumb hack
        result[-1].stretch(0, 0, about_edge=LEFT)
        return result

    def get_next_word_line(self, text_mob, char_len=7):
        next_word_line = Underline(text_mob[:char_len])
        next_word_line.set_stroke(TEAL, 2)
        next_word_line.next_to(text_mob[-1], RIGHT, SMALL_BUFF, aligned_edge=DOWN)
        if self.skip_through:
            next_word_line.set_opacity(0)
        return next_word_line

    def get_transformer_drawing(self):
        self.camera.light_source.move_to([-5, 5, 10])
        self.frame.set_field_of_view(20 * DEGREES)
        blocks = VGroup(
            VPrism(3, 2, 0.2)
            for n in range(10)
        )
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(width=0)
        blocks.set_shading(0.25, 0.5, 0.2)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)
        blocks.rotate(self.machine_phi, RIGHT, about_edge=OUT)
        blocks.rotate(self.machine_theta, UP, about_edge=OUT)

        blocks.deactivate_depth_test()
        for block in blocks:
            block.sort(lambda p: p[2])

        word = Text(self.machine_name, alignment="LEFT")
        word.next_to(blocks[-1], UP)
        word.shift(0.1 * UP + 0.4 * LEFT)
        word.move_to(blocks[-1])
        word.set_backstroke(BLACK, 5)
        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow)
        return result

    def get_distribution(
        self, words, probs, machine,
        font_size=24,
        width_100p=1.8,
        bar_height=0.25,
        show_ellipses=True
    ):
        labels = VGroup(Text(word, font_size=font_size) for word in words)
        bars = VGroup(
            Rectangle(prob * width_100p, bar_height)
            for prob, label in zip(probs, labels)
        )
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.5 * bar_height)
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
        bars.set_stroke(WHITE, 1)

        bar_groups = VGroup()
        for label, bar, prob in zip(labels, bars, probs):
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
            prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
            label.next_to(bar, LEFT)
            bar_groups.add(VGroup(label, bar, prob_label))

        if show_ellipses:
            ellipses = Tex(R"\\vdots", font_size=font_size)
            ellipses.next_to(bar_groups[-1][0], DOWN)
            bar_groups.add(ellipses)

        arrow_point = machine[-1].get_right()
        bar_groups.shift(arrow_point - bars.get_left() + 1.5 * RIGHT)
        bar_groups.align_to(machine, UP)

        return bar_groups

    def animate_text_input(self, text_mob, machine, position_text_over_machine=True, added_anims=[], lag_ratio=0.02):
        blocks = machine[0]
        text_copy = text_mob.copy()
        if position_text_over_machine:
            text_copy.target = text_copy.generate_target()
            text_copy.target.set_max_width(4)
            text_copy.target.next_to(blocks[0], UP)
            text_copy.target.shift_onto_screen()
            self.play(MoveToTarget(text_copy, path_arc=-45 * DEGREES))
        self.play(LaggedStart(
            *added_anims,
            Transform(
                text_copy,
                VGroup(VectorizedPoint(machine.get_top())),
                lag_ratio=lag_ratio,
                run_time=1,
                path_arc=-45 * DEGREES,
                remover=True,
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(machine[1:]),
            lag_ratio=0.5
        ))

    def animate_prediction_ouptut(self, machine, cur_str):
        words, probs = self.predict_next_token(cur_str)
        bar_groups = self.get_distribution(words, probs, machine)
        self.play(
            LaggedStart(
                (FadeInFromPoint(bar_group, machine[0][-1].get_right())
                for bar_group in bar_groups),
                lag_ratio=0.025,
                group=bar_groups,
                run_time=1
            )
        )
        return bar_groups

    def animate_random_sample(self, bar_groups):
        widths = np.array([group[1].get_width() for group in bar_groups[:-1]])
        dist = widths / widths.sum()
        seed = random.randint(0, 1000)
        buff = 0.025
        highlight_rect = SurroundingRectangle(bar_groups[0], buff=buff)
        highlight_rect.set_stroke(YELLOW, 2)
        highlight_rect.set_fill(YELLOW, 0.25)

        def highlight_randomly(rect, dist, alpha):
            np.random.seed(seed + int(10 * alpha))
            index = np.random.choice(np.arange(len(dist)), p=dist)
            rect.surround(bar_groups[index], buff=buff)
            rect.stretch(1.1, 0)

        self.play(
            UpdateFromAlphaFunc(highlight_rect, lambda rect, a: highlight_randomly(rect, dist, a)),
            Animation(bar_groups)
        )

        bar_groups.add_to_back(highlight_rect)

    def animate_word_addition(self, bar_groups, text_mob, next_word_line, force_unskip=False):
        # Choose the highlighted_group
        bar_group = None
        if isinstance(bar_groups[0], Rectangle):
            # Use the highlight rect to find the group element
            bars = bar_groups[1:-1]
            diffs = [abs(bg.get_y() - bar_groups[0].get_y()) for bg in bars]
            bar_group = bar_groups[1:][np.argmin(diffs)]
        if bar_group is None:
            bar_group = bar_groups[0]

        # Animate selection
        word = bar_group[0].get_text()
        new_str = self.cur_str + word
        new_text_mob = self.string_to_mob(new_str)
        new_text_mob[:len(self.seed_text.replace(" ", ""))].set_color(self.seed_text_color)

        word_targets = new_text_mob[word.strip()]
        if len(word_targets) > 0:
            target = word_targets[-1]
        else:
            target = new_text_mob[-len(word) - 1:-1]

        # target = new_text_mob[-len(word):]

        self.add(bar_groups)
        self.play(
            FadeTransform(bar_group[0].copy(), target),
            Transform(
                next_word_line,
                self.get_next_word_line(new_text_mob),
            ),
        )
        if force_unskip:
            self.skip_animations = False
            target.save_state()
            target.set_fill(YELLOW)
            self.wait(0.5)
            target.restore()
            self.skip_animations = True
        self.play(
            FadeOut(bar_groups),
        )

        self.remove(text_mob)
        self.add(new_text_mob)

        self.cur_str = new_str

        return new_text_mob

    def new_selection_cycle(self, text_mob, next_word_line, machine, quick=False, skip_anims=False):
        if skip_anims:
            self.skip_animations = True

        if quick:
            words, probs = self.predict_next_token(self.cur_str)
            bar_groups = self.get_distribution(words, probs, machine)
            self.add(bar_groups)
        else:
            self.animate_text_input(text_mob, machine)
            bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
            force_unskip=skip_anims
        )
        return new_text_mob

    #

    def predict_next_token(self, text):
        result = None
        n_shown = self.n_shown_predictions
        if self.model == "gpt3":
            try:
                result = gpt3_predict_next_token(
                    text, n_shown, random_seed=self.random_seed
                )
            except Exception as e:
                pass
        if result is None:
            result = gpt2_predict_next_token(text, n_shown)
        return result


class AltSimpleAutoRegression(SimpleAutogregression):
    n_predictions = 1
    line_len = 25

    def reposition_transformer_drawing(self, machine):
        machine.move_to(0.5 * RIGHT)
        in_arrow = machine[-1].copy()
        in_arrow.rotate(-45 * DEGREES)
        in_arrow.next_to(machine, UL)
        self.add(in_arrow)
        return machine


class AnnotateNextWord(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()
        self.add(machine, *machine[1:])
        words, probs = self.predict_next_token(self.cur_str)
        bar_groups = self.get_distribution(words, probs, machine[-1].get_right())

        self.add(bar_groups)

        # Initial text
        from manimlib.mobject.boolean_ops import Union
        highlight = Union(
            SurroundingRectangle(text_mob["Behold, a wild pi creature,"]),
            SurroundingRectangle(text_mob["foraging in its native"]),
        )
        highlight.set_stroke(BLUE, 3)
        arrow = Vector(LEFT, stroke_width=10)
        arrow.next_to(highlight, RIGHT).match_y(text_mob[0])

        dist_rect = SurroundingRectangle(bar_groups)
        dist_rect.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(highlight),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(PI / 2).next_to(dist_rect, UP),
            ReplacementTransform(highlight, dist_rect),
        )
        self.wait()
        self.play(
            FadeOut(dist_rect),
            FadeOut(arrow),
        )

        # Flash through
        self.remove(bar_groups)
        text_mob = self.new_selection_cycle(
            text_mob, next_word_line, machine,
        )


class QuickerRegression(SimpleAutogregression):
    skip_through = True


class AutoregressionGPT3(SimpleAutogregression):
    model = "gpt3"


class QuickRegressionGPT3(SimpleAutogregression):
    skip_through = True
    model = "gpt3"


class GPT3CleverestAutocomplete(QuickRegressionGPT3):
    seed_text = "To date, the cleverest thinker of all time was"
    n_predictions = 70

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 2),
            )


class GPT3OnLearningSimpler(QuickRegressionGPT3):
    seed_text = "The most effective way to learn computer science is"
    text_corner = 3.5 * UP + 3 * LEFT
    line_len = 35
    font_size = 35
    n_predictions = 300
    time_per_prediction = 0.2
    random_seed = 313
    model = "gpt3"
    min_y = -3
    up_shift = 5 * UP
    show_dist = False

    def construct(self):
        # Test
        cur_str = self.seed_text
        text_mob = VGroup()
        for n in range(self.n_predictions):
            self.clear()
            words, probs = self.predict_next_token(cur_str, n_shown=20)
            index = np.random.choice(np.arange(len(words)), p=(probs / probs.sum()))
            new_word = words[index]
            cur_str += new_word
            text_mob = self.string_to_mob(cur_str)

            # Color seed
            if self.color_seed:
                text_mob[:len(self.seed_text.replace(" ", ""))].set_color(BLUE)

            # Add to text, shift if necessary
            text_mob[new_word.strip()][-1].set_color(YELLOW)
            if text_mob.get_bottom()[1] < self.min_y:
                text_mob.shift(self.up_shift)
                self.text_corner += self.up_shift
            self.add(text_mob)

            # Add the distribution
            if self.show_dist:
                dist = self.get_distribution(
                    words[:self.n_shown_predictions],
                    probs[:self.n_shown_predictions],
                    buff=0
                )
                dist.set_height(4)
                dist.to_edge(DOWN)
                rect = SurroundingRectangle(dist[min(index, len(dist) - 1)])
                self.add(dist, rect)

            self.wait(self.time_per_prediction)


class GPT3OnLongPassages(GPT3OnLearningSimpler):
    seed_text = "Writing long passages seems to involve more foresight and planning than what single-word prediction"
    n_predictions = 100
    color_seed = False


class GPT3CreaturePrediction(GPT3CleverestAutocomplete):
    seed_text = "the fluffy blue creature"
    n_predictions = 1


class GPT3CreaturePrediction2(GPT3CleverestAutocomplete):
    seed_text = "the fluffy blue creature roamed the"
    n_predictions = 1


class LowTempExample(GPT3OnLearningSimpler):
    seed_text = "Once upon a time, there was a"
    model = "gpt3"
    min_y = 1
    up_shift = 2 * UP
    show_dist = True
    temp = 0
    n_predictions = 200
    time_per_prediction = 0.25

    def predict_next_token(self, text, n_shown=None):
        words, probs = super().predict_next_token(text, n_shown)
        if self.temp == 0:
            probs = np.zeros_like(probs)
            probs[0] = 1
        else:
            probs = probs**(1 / self.temp)
            probs /= probs.sum()
        return words, probs


class HighTempExample(LowTempExample):
    temp = 5
    model = "gpt3"


class MidTempExample(LowTempExample):
    seed_text = "If you could see the underlying probability distributions a large language model uses when generating text, then"
    temp = 1
    model = "gpt3"


class ChatBotPrompt(SimpleAutogregression):
    system_prompt = """
        What follows is a conversation between a user and a helpful,
        very knowledgeable AI assistant.
    """
    user_prompt = "User: Give me some ideas for what to do when visiting Paris."
    ai_seed = "AI Assistant: "
    machine_name = "Large\\nLanguage\\nModel"

    line_len = 28
    font_size = 36
    color_seed = False

    n_predictions = 60
    model = "gpt3"
    random_seed = 12

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()

        all_strs = list(map(clean_text, [self.system_prompt, self.user_prompt, self.ai_seed]))

        system_prompt, user_prompt, ai_seed = all_text = VGroup(
            get_paragraph(
                s.split(" "),
                font_size=self.font_size,
                line_len=self.line_len
            )
            for s in all_strs
        )
        all_text.arrange(DOWN, aligned_edge=LEFT, buff=0.75)
        all_text.move_to(self.text_corner, UL)
        self.remove(text_mob)
        self.add(all_text)

        text_mob = ai_seed
        self.text_corner = text_mob.get_corner(UL)
        next_word_line.next_to(ai_seed, RIGHT, aligned_edge=DOWN)

        self.cur_str = "\\n\\n".join(all_strs)

        # Comment on system prompt
        sys_rect = SurroundingRectangle(system_prompt)
        sys_rect.set_stroke(GREEN, 2)

        self.play(
            ShowCreation(sys_rect),
            system_prompt.animate.set_color(GREEN_B)
        )
        self.wait()

        # Users prompt
        from manimlib.mobject.boolean_ops import Union

        top_line = user_prompt["Give me some ideas for what"]
        low_line = user_prompt["to do when visiting Santiago."]
        user_rect = Union(
            SurroundingRectangle(low_line),
            SurroundingRectangle(top_line),
        )
        user_rect.set_stroke(BLUE, 2)

        sys_rect.insert_n_curves(100)
        self.play(
            ReplacementTransform(sys_rect, user_rect),
            top_line.animate.set_color(BLUE_B),
            low_line.animate.set_color(BLUE_B),
        )
        self.wait()
        self.play(
            FadeOut(user_rect),
        )

        # Run predictions
        text_mob = all_text
        self.add(all_text.copy())
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 0),
            )

    def string_to_mob(self, text):
        seed = self.ai_seed.strip()
        if seed in text:
            text = text[text.index(seed):]
        return super().string_to_mob(text)


class ChatBotPrompt2(ChatBotPrompt):
    user_prompt = "User: Can you explain what temperature is, in the context of softmax?"


class ChatBotPrompt3(ChatBotPrompt):
    user_prompt = "User: Can you give me some ideas for what to do while visiting Munich?"


class VoiceToTextExample(SimpleAutogregression):
    model_name = "voice-to-text"

    def construct(self):
        # Add model
        box = Rectangle(4, 3)
        box.set_stroke(WHITE, 2)
        name = Text(self.model_name, font_size=60)
        name.set_max_width(box.get_width())
        name.next_to(box, UP)
        machine = self.get_transformer_drawing()
        machine.center()
        machine.set_max_width(0.75 * box.get_width())
        machine.move_to(box)
        arrows = Vector(0.75 * RIGHT, stroke_width=8).replicate(2)
        arrows[0].next_to(box, LEFT, SMALL_BUFF)
        arrows[1].next_to(box, RIGHT, SMALL_BUFF)
        model = Group(box, name, arrows, machine)

        self.add(*model)
        self.add(Point())

        # Process input
        max_width = 3.75
        in_mob = self.get_input().set_max_width(max_width)
        out_mob = self.get_output().set_max_width(max_width)
        in_mob.next_to(arrows, LEFT)
        out_mob.next_to(arrows, RIGHT)

        self.add(in_mob)
        self.play(LaggedStart(
            FadeOutToPoint(
                in_mob.copy(), machine.get_left(),
                path_arc=-45 * DEGREES,
                lag_ratio=0.01,
            ),
            LaggedStart(
                (block.animate.set_color(TEAL).set_anim_args(rate_func=there_and_back)
                for block in machine[0][:-1]),
                lag_ratio=0.1,
                run_time=1,
            ),
            FadeInFromPoint(
                out_mob.copy(), machine.get_right(),
                path_arc=45 * DEGREES,
                lag_ratio=0.02
            ),
            lag_ratio=0.7
        ))
        self.wait()

    def get_input(self) -> Mobject:
        result =ImageMobject("AudioSnippet").set_width(3.75)
        result.set_height(3, stretch=True)
        return result

    def get_output(self) -> Mobject:
        return Text("""
            Some models take
            in audio and
            produce a transcript
        """, alignment="LEFT")


class TextToVoiceExample(VoiceToTextExample):
    model_name = "text-to-voice"

    def get_input(self):
        return Text("""
            This sentence comes from
            a model going the other
            way around, producing
            synthetic speech just
            from text.
        """, alignment="LEFT")

    def get_output(self):
        return super().get_input()


class TextToImage(VoiceToTextExample):
    model_name = "text-to-image"
    prompt = """
        1960s photograph of a cute fluffy blue wild pi
        creature, a creature whose body is shaped like
        the symbol π, who is foraging in its native territory,
        staring back at the camera with an exotic scene
        in the background.
    """
    image_name = "PiCreatureDalle3_5"

    def get_clean_prompt(self):
        return clean_text(self.prompt)

    def get_input(self):
        return get_paragraph(self.get_clean_prompt().split(" "), line_len=25)

    def get_output(self):
        return ImageMobject(self.image_name)

    def generate_output(self):
        # Test
        self.prompt = """
            1960s photograph of a cute fluffy blue wild pi
            creature, a creature whose face bears a subtle resemblence
            to the shape of the symbol π, who is foraging in its native
            territory, staring back at the camera with an exotic scene
            in the background.
        """

        self.prompt = "abstract depiction of furry fluffiness"

        openai.api_key = os.getenv('OPENAI_KEY')
        prompt = self.get_clean_prompt()

        response = openai.Image.create(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        print(prompt)
        print(image_url)

        response = openai.Image.create_variation(
          image=open("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/images/raster/PiCreatureDalle3_17.png", "rb"),
          n=1,
          size="1024x1024"
        )


class TranslationExample(VoiceToTextExample):
    model_name = "machine translation"

    def get_input(self):
        return Text("Attention is all\\nyou need")

    def get_output(self):
        return Group(Point(), *Text("注意力就是你所需要的一切"))


class PredictionVsGeneration(SimpleAutogregression):
    model = "gpt2"

    def construct(self):
        # Setup
        self.add(FullScreenRectangle())
        morty = Mortimer()
        morty.to_edge(DOWN)
        morty.body.insert_n_curves(100)
        self.add(morty)

        # Words
        words = VGroup(Text("Prediction"), Text("Generation"))
        words.scale(1.5)
        for vect, word in zip([UL, UR], words):
            word.next_to(morty, vect)
            word.shift(0.5 * UP)

        # Create prediction object
        seed_text = "The goal of predicting the next"
        self.n_shown_predictions = 8
        tokens, probs = self.predict_next_token(seed_text)
        dist = self.get_distribution(tokens, probs)
        brace = Brace(dist, LEFT, SMALL_BUFF)
        words = Text(seed_text, font_size=36).next_to(brace, LEFT)
        prediction = VGroup(words, brace, dist)
        prediction.set_width(FRAME_WIDTH / 2 - 1)
        prediction.next_to(morty, UL)
        prediction.shift(0.5 * UP).shift_onto_screen()
        self.add(prediction)

        # Animations
        self.play(
            morty.change("raise_right_hand", prediction),
            FadeIn(prediction[0], UP),
            GrowFromCenter(prediction[1]),
            LaggedStart(
                (FadeInFromPoint(bar, prediction[1].get_center())
                for bar in prediction[2]),
                lag_ratio=0.05,
            )
        )
        self.play(Blink(morty))
        self.play(
            morty.change("raise_left_hand", 3 * UR),
        )
        self.wait()
        self.play(Blink(morty))
        self.wait()


class ManyParallelPredictions(SimpleAutogregression):
    line_len = 200
    n_shown_predictions = 8
    model = "gpt3"

    def construct(self):
        # Setup
        self.fake_machine = VectorizedPoint().replicate(3)
        full_string = "Harry Potter was a highly unusual boy"

        # Draw last layer vectors
        last_layer = VGroup(
            NumericEmbedding(length=10)
            for n in range(12)
        )
        last_layer.arrange(RIGHT, buff=0.35 * last_layer[0].get_width())
        last_layer.set_height(3)
        last_layer.to_edge(DOWN)
        # self.add(last_layer)

        rects = VGroup(map(SurroundingRectangle, last_layer))
        rects.set_stroke(YELLOW, 2)
        arrows = VGroup(Vector(0.5 * UP).next_to(rect, UP, buff=0.1) for rect in rects)
        arrows.set_stroke(YELLOW)

        # Show prediction groups
        words = full_string.split(" ")
        substrings = [
            " ".join(words[:n + 1])
            for n in range(len(words))
        ]

        predictions = VGroup(
            self.get_prediction_group(substring)
            for substring in substrings
        )
        predictions[0].to_edge(UP, buff=1.25).align_to(rects[1], LEFT)
        for prediction, arrow, rect in zip(predictions, arrows, rects):
            prediction.move_to(predictions[0], LEFT)
            arrow.become(Arrow(
                rect.get_top(),
                prediction[1].get_left(),
            ))
            arrow.set_stroke(YELLOW)

        last_group = VGroup(
            rects[0].copy().set_opacity(0),
            arrows[0].copy().set_opacity(0),
            predictions[0].copy().set_opacity(0),
        )
        for rect, arrow, prediction in zip(rects, arrows, predictions):
            self.remove(last_group)
            self.play(
                TransformFromCopy(last_group[0], rect),
                TransformFromCopy(last_group[1], arrow),
                TransformMatchingStrings(last_group[2][0].copy(), prediction[0], run_time=1),
                FadeTransform(last_group[2][1].copy(), prediction[1]),
                FadeTransform(last_group[2][2].copy(), prediction[2]),
            )
            self.wait()
            last_group = VGroup(rect, arrow, prediction)

    def get_prediction_group(self, text):
        words, probs = self.predict_next_token(text)
        dist = self.get_distribution(
            words, probs,
            width_100p=2.0
        )
        dist.set_max_height(2.5)
        brace = Brace(dist, LEFT)
        prefix = Text(text, font_size=30)
        prefix.next_to(brace, LEFT)

        result = VGroup(prefix, brace, dist)

        return result


class PeekUnderTheHood(SimpleAutogregression):
    def construct(self):
        # Add parts
        text_mob, next_word_line, machine = self.init_text_and_machine()
        blocks, label, arrow = machine
        self.remove(text_mob, next_word_line)

        # Zoom in
        self.camera.light_source.move_to([-15, 5, 10])
        self.set_floor_plane("xz")

        blocks.rotate(-5 * DEGREES, UP, about_edge=OUT)
        blocks.rotate(-10 * DEGREES, RIGHT, about_edge=OUT)
        blocks.target = blocks.generate_target()
        blocks.target.set_height(5)
        blocks.target.center()
        blocks.target[5:].set_opacity(0.3)

        self.play(
            self.frame.animate.reorient(-23, -12, 0, (1.79, -0.56, 1.27), 8.40).set_anim_args(run_time=3),
            MoveToTarget(blocks, run_time=3),
            FadeOut(arrow, RIGHT),
            FadeOut(label, 2 * OUT),
        )
        self.wait()

        blocks[5:].set_opacity(0.3)

        # Add matrices
        matrices = VGroup(WeightMatrix(shape=(8, 8)) for x in range(9))
        matrices.arrange_in_grid(h_buff_ratio=0.25, v_buff_ratio=0.4)
        matrices.match_width(blocks)
        index = 6
        matrices.move_to(blocks[index], OUT)
        self.add(matrices, blocks[index:])`,
    annotations: {
      2: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      3: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      60: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      82: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      95: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      97: "DecimalNumber displays a formatted decimal that can be animated. Tracks a value and auto-updates display.",
      103: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      112: "SimpleAutogregression extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      127: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      183: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      193: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      216: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      228: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      234: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      253: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      255: "Transform smoothly morphs one mobject into another by interpolating their points.",
      263: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      265: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      280: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      281: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      306: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      307: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      339: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      341: "Transform smoothly morphs one mobject into another by interpolating their points.",
      348: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      350: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      353: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      354: "FadeOut transitions a mobject from opaque to transparent.",
      399: "Class AltSimpleAutoRegression inherits from SimpleAutogregression.",
      412: "Class AnnotateNextWord inherits from SimpleAutogregression.",
      413: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      434: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      456: "Class QuickerRegression inherits from SimpleAutogregression.",
      460: "Class AutoregressionGPT3 inherits from SimpleAutogregression.",
      464: "Class QuickRegressionGPT3 inherits from SimpleAutogregression.",
      469: "Class GPT3CleverestAutocomplete inherits from QuickRegressionGPT3.",
      473: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      483: "Class GPT3OnLearningSimpler inherits from QuickRegressionGPT3.",
      496: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      534: "Class GPT3OnLongPassages inherits from GPT3OnLearningSimpler.",
      540: "Class GPT3CreaturePrediction inherits from GPT3CleverestAutocomplete.",
      545: "Class GPT3CreaturePrediction2 inherits from GPT3CleverestAutocomplete.",
      550: "Class LowTempExample inherits from GPT3OnLearningSimpler.",
      571: "Class HighTempExample inherits from LowTempExample.",
      576: "Class MidTempExample inherits from LowTempExample.",
      582: "Class ChatBotPrompt inherits from SimpleAutogregression.",
      599: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      672: "Class ChatBotPrompt2 inherits from ChatBotPrompt.",
      676: "Class ChatBotPrompt3 inherits from ChatBotPrompt.",
      680: "Class VoiceToTextExample inherits from SimpleAutogregression.",
      683: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      744: "Class TextToVoiceExample inherits from VoiceToTextExample.",
      760: "Class TextToImage inherits from VoiceToTextExample.",
      814: "Class TranslationExample inherits from VoiceToTextExample.",
      824: "Class PredictionVsGeneration inherits from SimpleAutogregression.",
      827: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      875: "Class ManyParallelPredictions inherits from SimpleAutogregression.",
      880: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      953: "Class PeekUnderTheHood inherits from SimpleAutogregression.",
      954: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/helpers.py"] = {
    description: "Shared helper functions and reusable components for the transformer video series. Includes network diagram builders, embedding visualizers, and common animation utilities.",
    code: `from __future__ import annotations

from manim_imports_ext import *

from typing import TYPE_CHECKING
import warnings
# import datasets

DATA_DIR = Path(get_output_dir(), "2024/transformers/data/")
WORD_FILE = Path(DATA_DIR, "OWL3_Dictionary.txt")


if TYPE_CHECKING:
    from typing import Optional
    from manimlib.typing import Vect3, ManimColor


def get_paragraph(words, line_len=40, font_size=48):
    """
    Handle word wrapping
    """
    words = list(map(str.strip, words))
    word_lens = list(map(len, words))
    lines = []
    lh, rh = 0, 0
    while rh < len(words):
        rh += 1
        if sum(word_lens[lh:rh]) > line_len:
            rh -= 1
            lines.append(words[lh:rh])
            lh = rh
    lines.append(words[lh:])
    text = "\\n".join([" ".join(line).strip() for line in lines])
    return Text(text, alignment="LEFT", font_size=font_size)


def softmax(logits, temperature=1.0):
    logits = np.array(logits)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')  # Ignore all warnings within this block
        logits = logits - np.max(logits)  # For numerical stability
        exps = np.exp(np.divide(logits, temperature, where=temperature != 0))
    
    if np.isinf(exps).any() or np.isnan(exps).any() or temperature == 0:
        result = np.zeros_like(logits)
        result[np.argmax(logits)] = 1
        return result
    return exps / np.sum(exps)


def value_to_color(
    value,
    low_positive_color=BLUE_E,
    high_positive_color=BLUE_B,
    low_negative_color=RED_E,
    high_negative_color=RED_B,
    min_value=0.0,
    max_value=10.0
):
    alpha = clip(float(inverse_interpolate(min_value, max_value, abs(value))), 0, 1)
    if value >= 0:
        colors = (low_positive_color, high_positive_color)
    else:
        colors = (low_negative_color, high_negative_color)
    return interpolate_color_by_hsl(*colors, alpha)


def read_in_book(name="tale_of_two_cities"):
    return Path(DATA_DIR, name).with_suffix(".txt").read_text()


def load_image_net_data(dataset_name="image_net_1k"):
    data_path = Path(Path.home(), "Documents", dataset_name)
    image_dir = Path(data_path, "images")
    label_category_path = Path(DATA_DIR, "image_categories.txt")
    image_label_path = Path(data_path, "image_labels.txt")

    if not os.path.exists(image_dir):
        os.makedirs(image_dir)
        image_data = datasets.load_from_disk(str(data_path))
        indices = range(len(image_data))
        categories = label_category_path.read_text().split("\\n")
        labels = [categories[image_data[index]['label']] for index in indices]
        image_label_path.write_text("\\n".join(labels))
        for index in ProgressDisplay(indices):
            image = image_data[index]['image']
            image.save(str(Path(image_dir, f"{index}.jpeg")))


    labels = image_label_path.read_text().split("\\n")
    return [
        (Path(image_dir, f"{index}.jpeg"), label)
        for index, label in enumerate(labels)
    ]


def show_matrix_vector_product(scene, matrix, vector, buff=0.25, x_max=999, fix_in_frame=False):
    # Show product
    eq = Tex("=")
    eq.set_width(0.5 * vector.get_width())
    shape = (matrix.shape[0], 1)
    rhs = NumericEmbedding(
        values=x_max * np.ones(shape),
        value_range=(-x_max, x_max),
        decimal_config=dict(include_sign=True, edge_to_fix=ORIGIN),
        ellipses_row=matrix.ellipses_row,
    )
    rhs.scale(vector.elements[0].get_height() / rhs.elements[0].get_height())
    eq.next_to(vector, RIGHT, buff=buff)
    rhs.next_to(eq, RIGHT, buff=buff)
    if fix_in_frame:
        eq.fix_in_frame()
        rhs.fix_in_frame()

    scene.play(FadeIn(eq), FadeIn(rhs.get_brackets()))

    last_rects = VGroup()
    n_rows = len(matrix.rows)
    for n, row, entry in zip(it.count(), matrix.get_rows(), rhs[:-2]):
        if matrix.ellipses_row is not None and n == (matrix.ellipses_row % n_rows):
            scene.add(entry)
        else:
            last_rects = matrix_row_vector_product(
                scene, row, vector, entry, last_rects,
                fix_in_frame=fix_in_frame
            )
    scene.play(FadeOut(last_rects))

    return eq, rhs


def matrix_row_vector_product(scene, row, vector, entry, to_fade, fix_in_frame=False):
    def get_rect(elem):
        return SurroundingRectangle(elem, buff=0.1, is_fixed_in_frame=fix_in_frame).set_stroke(YELLOW, 2)

    row_rects = VGroup(*map(get_rect, row))
    vect_rects = VGroup(*map(get_rect, vector[:-2]))
    partial_values = [0]
    for e1, e2 in zip(row, vector[:-2]):
        if not isinstance(e1, DecimalNumber) and isinstance(e2, DecimalNumber):
            increment = 0
        else:
            val1 = round(e1.get_value(), e1.num_decimal_places)
            val2 = round(e2.get_value(), e2.num_decimal_places)
            increment = val1 * val2
        partial_values.append(partial_values[-1] + increment)
    n_values = len(partial_values)

    scene.play(
        ShowIncreasingSubsets(row_rects),
        ShowIncreasingSubsets(vect_rects),
        UpdateFromAlphaFunc(entry, lambda m, a: m.set_value(
            partial_values[min(int(np.round(a * n_values)), n_values - 1)]
        )),
        FadeOut(to_fade),
        rate_func=linear,
    )

    return VGroup(row_rects, vect_rects)


def get_full_matrix_vector_product(
    mat_sym="w",
    vect_sym="x",
    n_rows=5,
    n_cols=5,
    mat_sym_color=BLUE,
    height=3.0,
    ellipses_row=-2,
    ellipses_col=-2,
):
    m_indices = list(map(str, [*range(1, n_cols), "m"]))
    n_indices = list(map(str, [*range(1, n_rows), "n"]))
    matrix = TexMatrix(
        [
            [Rf"{mat_sym}_{{{m}, {n}}}" for n in n_indices]
            for m in m_indices
        ],
        ellipses_row=ellipses_row,
        ellipses_col=ellipses_col,
    )
    matrix.set_height(height)
    matrix.get_entries().set_color(mat_sym_color)
    vector = TexMatrix(
        [[Rf"x_{{{n}}}"] for n in n_indices],
        ellipses_row=ellipses_row,
    )
    vector.match_height(matrix)
    vector.next_to(matrix, RIGHT)
    equals = Tex("=", font_size=72)
    equals.next_to(vector, RIGHT)

    result_terms = [
        [Rf"w_{{{m}, {n}}} x_{n}" for n in n_indices]
        for m in m_indices
    ]
    rhs = TexMatrix(
        result_terms,
        ellipses_row=ellipses_row,
        ellipses_col=ellipses_col,
    )
    rhs.match_height(matrix)
    rhs.next_to(equals, RIGHT)
    for m, row in enumerate(rhs.get_rows()):
        if m == (ellipses_row % len(m_indices)):
            continue
        for n, entry in enumerate(row):
            if n != (ellipses_col % len(n_indices)):
                entry[:4].set_color(mat_sym_color)
        for e1, e2 in zip(row, row[1:]):
            plus = Tex("+")
            plus.match_height(e1)
            points = [e1.get_right(), e2.get_left()]
            plus.move_to(midpoint(*points))
            plus.align_to(e1, UP)
            e2.add(plus)

    return matrix, vector, equals, rhs


def show_symbolic_matrix_vector_product(scene, matrix, vector, rhs, run_time_per_row=0.75):
    last_rects = VGroup()
    for mat_row, rhs_row in zip(matrix.get_rows(), rhs.get_rows()):
        mat_rects = VGroup(*map(SurroundingRectangle, mat_row))
        vect_rects = VGroup(*map(SurroundingRectangle, vector.get_columns()[0]))
        rect_group = VGroup(mat_rects, vect_rects)
        rect_group.set_stroke(YELLOW, 2)
        scene.play(
            FadeOut(last_rects),
            *(
                ShowIncreasingSubsets(group, rate_func=linear)
                for group in [mat_rects, vect_rects, rhs_row]
            ),
            run_time=run_time_per_row,
        )
        last_rects = rect_group
    scene.play(FadeOut(last_rects))


def data_flying_animation(
    point,
    vect=2 * DOWN + RIGHT,
    color=GREY_C,
    max_opacity=0.75,
    font_size=48,
    fix_in_frame=False
    ):
    word = Text("Data", color=color, font_size=font_size)
    if fix_in_frame:
        word.fix_in_frame()
    return UpdateFromAlphaFunc(
        word, lambda m, a: m.move_to(
            interpolate(point, point + vect, a)
        ).set_opacity(there_and_back(a) * max_opacity)
    )


def get_data_modifying_matrix_anims(
    matrix,
    word_shape=(5, 10),
    alpha_maxes=(0.7, 0.9),
    shift_vect=2 * DOWN + RIGHT,
    run_time=3,
    fix_in_frame=False,
    font_size=48,
):
    x_min, x_max = [matrix.get_x(LEFT), matrix.get_x(RIGHT)]
    y_min, y_max = [matrix.get_y(UP), matrix.get_y(DOWN)]
    z = matrix.get_z()
    points = np.array([
        [
            interpolate(x_min, x_max, a1),
            interpolate(y_min, y_max, a2),
            z,
        ]
        for a1 in np.linspace(0, alpha_maxes[1], word_shape[1])
        for a2 in np.linspace(0, alpha_maxes[0], word_shape[0])
    ])
    return [
        LaggedStart(
            (data_flying_animation(p, vect=shift_vect, fix_in_frame=fix_in_frame, font_size=font_size)
            for p in points),
            lag_ratio=1 / len(points),
            run_time=run_time
        ),
        RandomizeMatrixEntries(matrix, run_time=run_time),
    ]


def data_modifying_matrix(scene, matrix, *args, **kwargs):
    anims = get_data_modifying_matrix_anims(matrix, *args, **kwargs)
    scene.play(*anims)


def create_pixels(image_mob, pixel_width=0.1):
    x0, y0, z0 = image_mob.get_corner(UL)
    x1, y1, z1 = image_mob.get_corner(DR)
    points = np.array([
        [x, y, 0]
        for y in np.arange(y0, y1, -pixel_width)
        for x in np.arange(x0, x1, pixel_width)
    ])
    square = Square(pixel_width).set_fill(WHITE, 1).set_stroke(width=0)
    pixels = VGroup(
        square.copy().move_to(point, UL).set_color(
            Color(rgb=image_mob.point_to_rgb(point))
        )
        for point in points
    )
    return pixels


def get_network_connections(layer1, layer2, max_width=2.0, opacity_exp=1.0):
    radius = layer1[0].get_width() / 2
    return VGroup(
        Line(n1.get_center(), n2.get_center(), buff=radius).set_stroke(
            color=value_to_color(random.uniform(-10, 10)),
            width=max_width * random.random(),
            opacity=random.random()**opacity_exp,
        )
        for n1 in layer1
        for n2 in layer2
    )


def get_vector_pair(angle_in_degrees=90, length=1.0, colors=(BLUE, BLUE)):
    angle = angle_in_degrees * DEGREES
    v1 = Vector(length * RIGHT)
    v2 = v1.copy().rotate(angle, about_point=ORIGIN)
    v1.set_color(colors[0])
    v2.set_color(colors[1])
    arc = Arc(radius=0.2, angle=angle)
    arc.set_stroke(WHITE, 2)
    label = Tex(Rf"180^\\circ", font_size=24)
    num = label.make_number_changeable("180")
    num.set_value(angle_in_degrees)
    label.next_to(arc.pfp(0.5), normalize(arc.pfp(0.5)), buff=SMALL_BUFF)

    return VGroup(v1, v2, arc, label)


class NeuralNetwork(VGroup):
    def __init__(
        self,
        layer_sizes=[6, 12, 6],
        neuron_radius=0.1,
        v_buff_ratio=1.0,
        h_buff_ratio=7.0,
        max_stroke_width=2.0,
        stroke_decay=2.0,
    ):
        self.max_stroke_width = max_stroke_width
        self.stroke_decay = stroke_decay
        layers = VGroup(*(
            Dot(radius=neuron_radius).get_grid(n, 1, v_buff_ratio=v_buff_ratio)
            for n in layer_sizes
        ))
        layers.arrange(RIGHT, buff=h_buff_ratio * layers[0].get_width())

        lines = VGroup(*(
            VGroup(*(
                Line(
                    n1.get_center(),
                    n2.get_center(),
                    buff=n1.get_width() / 2,
                )
                for n1, n2 in it.product(l1, l2)
            ))
            for l1, l2 in zip(layers, layers[1:])
        ))

        super().__init__(layers, lines)
        self.layers = layers
        self.lines = lines

        self.randomize_layer_values()
        self.randomize_line_style()

    def randomize_layer_values(self):
        for group in self.lines:
            for line in group:
                line.set_stroke(
                    value_to_color(random.uniform(-10, 10)),
                    self.max_stroke_width * random.random()**self.stroke_decay,
                )
        return self

    def randomize_line_style(self):
        for layer in self.layers:
            for dot in layer:
                dot.set_stroke(WHITE, 1)
                dot.set_fill(WHITE, random.random())
        return self


class ContextAnimation(LaggedStart):
    def __init__(
        self,
        target,
        sources,
        direction=UP,
        hue_range=(0.1, 0.3),
        time_width=2,
        min_stroke_width=0,
        max_stroke_width=5,
        lag_ratio=None,
        strengths=None,
        run_time=3,
        fix_in_frame=False,
        path_arc=PI / 2,
        **kwargs,
    ):
        arcs = VGroup()
        if strengths is None:
            strengths = np.random.random(len(sources))**2
        for source, strength in zip(sources, strengths):
            sign = direction[1] * (-1)**int(source.get_x() < target.get_x())
            arcs.add(Line(
                source.get_edge_center(direction),
                target.get_edge_center(direction),
                path_arc=sign * path_arc,
                stroke_color=random_bright_color(hue_range=hue_range),
                stroke_width=interpolate(
                    min_stroke_width,
                    max_stroke_width,
                    strength,
                )
            ))
        if fix_in_frame:
            arcs.fix_in_frame()
        arcs.shuffle()
        lag_ratio = 0.5 / len(arcs) if lag_ratio is None else lag_ratio

        super().__init__(
            *(
                VShowPassingFlash(arc, time_width=time_width)
                for arc in arcs
            ),
            lag_ratio=lag_ratio,
            run_time=run_time,
            **kwargs,
        )


class LabeledArrow(Arrow):
    def __init__(
        self,
        *args,
        label_text: Optional[str] = None,
        font_size: float = 24,
        label_buff: float = 0.1,
        direction: Optional[Vect3] = None,
        label_rotation: float = PI / 2,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        if label_text is not None:
            start, end = self.get_start_and_end()
            label = Text(label_text, font_size=font_size)
            label.set_fill(self.get_color())
            label.set_backstroke()
            label.rotate(label_rotation, RIGHT)
            if direction is None:
                direction = normalize(end - start)
            label.next_to(end, direction, buff=label_buff)
            self.label = label
        else:
            self.label = None


class WeightMatrix(DecimalMatrix):
    def __init__(
        self,
        values: Optional[np.ndarray] = None,
        shape: tuple[int, int] = (6, 8),
        value_range: tuple[float, float] = (-9.9, 9.9),
        ellipses_row: Optional[int] = -2,
        ellipses_col: Optional[int] = -2,
        num_decimal_places: int = 1,
        bracket_h_buff: float = 0.1,
        decimal_config=dict(include_sign=True),
        low_positive_color: ManimColor = BLUE_E,
        high_positive_color: ManimColor = BLUE_B,
        low_negative_color: ManimColor = RED_E,
        high_negative_color: ManimColor = RED_B,
    ):
        if values is not None:
            shape = values.shape
        self.shape = shape
        self.value_range = value_range
        self.low_positive_color = low_positive_color
        self.high_positive_color = high_positive_color
        self.low_negative_color = low_negative_color
        self.high_negative_color = high_negative_color
        self.ellipses_row = ellipses_row
        self.ellipses_col = ellipses_col

        if values is None:
            values = np.random.uniform(*self.value_range, size=shape)

        super().__init__(
            values,
            num_decimal_places=num_decimal_places,
            bracket_h_buff=bracket_h_buff,
            decimal_config=decimal_config,
            ellipses_row=ellipses_row,
            ellipses_col=ellipses_col,
        )
        self.reset_entry_colors()

    def reset_entry_colors(self):
        for entry in self.get_entries():
            entry.set_fill(color=value_to_color(
                entry.get_value(),
                self.low_positive_color,
                self.high_positive_color,
                self.low_negative_color,
                self.high_negative_color,
                0, max(self.value_range),
            ))
        return self


class NumericEmbedding(WeightMatrix):
    def __init__(
        self,
        values: Optional[np.ndarray] = None,
        shape: Optional[Tuple[int, int]] = None,
        length: int = 7,
        num_decimal_places: int = 1,
        ellipses_row: int = -2,
        ellipses_col: int = -2,
        value_range: tuple[float, float] = (-9.9, 9.9),
        bracket_h_buff: float = 0.1,
        decimal_config=dict(include_sign=True),
        dark_color: ManimColor = GREY_C,
        light_color: ManimColor = WHITE,
        **kwargs,
    ):
        if values is not None:
            if len(values.shape) == 1:
                values = values.reshape((values.shape[0], 1))
            shape = values.shape
        if shape is None:
            shape = (length, 1)
        super().__init__(
            values,
            shape=shape,
            value_range=value_range,
            num_decimal_places=num_decimal_places,
            bracket_h_buff=bracket_h_buff,
            decimal_config=decimal_config,
            low_positive_color=dark_color,
            high_positive_color=light_color,
            low_negative_color=dark_color,
            high_negative_color=light_color,
            ellipses_row=ellipses_row,
            ellipses_col=ellipses_col,
            **kwargs,
        )

        # No sign on zeros
        for entry in self.get_entries():
            if entry.get_value() == 0:
                entry[0].set_opacity(0)


class EmbeddingArray(VGroup):
    def __init__(
        self,
        shape=(10, 9),
        height=4,
        dots_index=-4,
        buff_ratio=0.4,
        bracket_color=GREY_B,
        backstroke_width=3,
        add_background_rectangle=False,
    ):
        super().__init__()

        # Embeddings
        embeddings = VGroup(
            NumericEmbedding(length=shape[0])
            for n in range(shape[1])
        )
        embeddings.set_height(height)
        buff = buff_ratio * embeddings[0].get_width()
        embeddings.arrange(RIGHT, buff=buff)

        # Background rectangle
        if add_background_rectangle:
            for embedding in embeddings:
                embedding.add_background_rectangle()

        # Add brackets
        brackets = Tex("".join((
            R"\\left[\\begin{array}{c}",
            *(shape[1] // 3) * [R"\\quad \\\\"],
            R"\\end{array}\\right]",
        )))
        brackets.set_height(1.1 * embeddings.get_height())
        lb = brackets[:len(brackets) // 2]
        rb = brackets[len(brackets) // 2:]
        lb.next_to(embeddings, LEFT, buff=0)
        rb.next_to(embeddings, RIGHT, buff=0)
        brackets.set_fill(bracket_color)

        # Assemble result
        dots = VGroup()
        self.add(embeddings, dots, brackets)
        self.embeddings = embeddings
        self.dots = dots
        self.brackets = brackets
        self.set_backstroke(BLACK, backstroke_width)

        if dots_index is not None:
            self.swap_embedding_for_dots(dots_index)


    def swap_embedding_for_dots(self, dots_index=-4):
        to_replace = self.embeddings[dots_index]
        dots = Tex(R"\\dots", font_size=60)
        dots.set_width(0.75 * to_replace.get_width())
        dots.move_to(to_replace)
        self.embeddings.remove(to_replace)
        self.dots.add(dots)
        return self


class RandomizeMatrixEntries(Animation):
    def __init__(self, matrix, **kwargs):
        self.matrix = matrix
        self.entries = matrix.get_entries()
        self.start_values = [entry.get_value() for entry in self.entries]
        self.target_values = np.random.uniform(
            matrix.value_range[0],
            matrix.value_range[1],
            len(self.entries)
        )
        super().__init__(matrix, **kwargs)

    def interpolate_mobject(self, alpha: float) -> None:
        for index, entry in enumerate(self.entries):
            start = self.start_values[index]
            target = self.target_values[index]
            sub_alpha = self.get_sub_alpha(alpha, index, len(self.entries))
            entry.set_value(interpolate(start, target, sub_alpha))
        self.matrix.reset_entry_colors()


class AbstractEmbeddingSequence(MobjectMatrix):
    pass


class Dial(VGroup):
    def __init__(
        self,
        radius=0.5,
        relative_tick_size=0.2,
        value_range=(0, 1, 0.1),
        initial_value=0,
        arc_angle=270 * DEGREES,
        stroke_width=2,
        stroke_color=WHITE,
        needle_color=BLUE,
        needle_stroke_width=5.0,
        value_to_color_config=dict(),
        set_anim_streak_color=TEAL,
        set_anim_streak_width=4,
        set_value_anim_streak_density=6,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.value_range = value_range
        self.value_to_color_config = value_to_color_config
        self.set_anim_streak_color = set_anim_streak_color
        self.set_anim_streak_width = set_anim_streak_width
        self.set_value_anim_streak_density = set_value_anim_streak_density

        # Main dial
        self.arc = Arc(arc_angle / 2, -arc_angle, radius=radius)
        self.arc.rotate(90 * DEGREES, about_point=ORIGIN)

        low, high, step = value_range
        n_values = int(1 + (high - low) / step)
        tick_points = map(self.arc.pfp, np.linspace(0, 1, n_values))
        self.ticks = VGroup(*(
            Line((1.0 - relative_tick_size) * point, point)
            for point in tick_points
        ))
        self.bottom_point = VectorizedPoint(radius * DOWN)
        for mob in self.arc, self.ticks:
            mob.set_stroke(stroke_color, stroke_width)

        self.add(self.arc, self.ticks, self.bottom_point)

        # Needle
        self.needle = Line()
        self.needle.set_stroke(
            color=needle_color,
            width=[needle_stroke_width, 0]
        )
        self.add(self.needle)

        # Initialize
        self.set_value(initial_value)

    def value_to_point(self, value):
        low, high, step = self.value_range
        alpha = inverse_interpolate(low, high, value)
        return self.arc.pfp(alpha)

    def set_value(self, value):
        self.needle.put_start_and_end_on(
            self.get_center(),
            self.value_to_point(value)
        )
        self.needle.set_color(value_to_color(
            value,
            min_value=self.value_range[0],
            max_value=self.value_range[1],
            **self.value_to_color_config
        ))

    def animate_set_value(self, value, **kwargs):
        kwargs.pop("path_arc", None)
        center = self.get_center()
        points = [self.needle.get_end(), self.value_to_point(value)]
        vects = [point - center for point in points]
        angle1, angle2 = [
            (angle_of_vector(vect) + TAU / 4) % TAU - TAU / 4
            for vect in vects
        ]
        path_arc = angle2 - angle1

        density = self.set_value_anim_streak_density
        radii = np.linspace(0, 0.5 * self.get_width(), density + 1)[1:]
        diff_arcs = VGroup(*(
            Arc(
                angle1, angle2 - angle1,
                radius=radius,
                arc_center=center,
            )
            for radius in radii
        ))
        diff_arcs.set_stroke(self.set_anim_streak_color, self.set_anim_streak_width)

        return AnimationGroup(
            self.animate.set_value(value).set_anim_args(path_arc=path_arc, **kwargs),
            *(
                VShowPassingFlash(diff_arc, time_width=1.5, **kwargs)
                for diff_arc in diff_arcs
            )
        )

    def get_random_value(self):
        low, high, step = self.value_range
        return interpolate(low, high, random.random())


class MachineWithDials(VGroup):
    default_dial_config = dict(
        stroke_width=1.0,
        needle_stroke_width=5.0,
        relative_tick_size=0.25,
        set_anim_streak_width=2,
    )

    def __init__(
        self,
        width=5.0,
        height=4.0,
        n_rows=6,
        n_cols=8,
        dial_buff_ratio=0.5,
        stroke_color=WHITE,
        stroke_width=1,
        fill_color=GREY_D,
        fill_opacity=1.0,
        dial_config=dict(),
    ):
        super().__init__()
        box = Rectangle(width, height)
        box.set_stroke(stroke_color, stroke_width)
        box.set_fill(fill_color, fill_opacity)
        self.box = box

        dial_config = dict(**self.default_dial_config, **dial_config)
        dials = Dial(**dial_config).get_grid(n_rows, n_cols, buff_ratio=dial_buff_ratio)
        buff = dials[0].get_width() * dial_buff_ratio
        dials.set_width(box.get_width() - buff)
        dials.set_max_height(box.get_width() - buff)
        dials.move_to(box)
        for dial in dials:
            dial.set_value(dial.get_random_value())
        self.dials = dials

        self.add(box, dials)

    def random_change_animation(self, lag_factor=0.5, run_time=3.0, **kwargs):
        return LaggedStart(
            *(
                dial.animate_set_value(dial.get_random_value())
                for dial in self.dials
            ), lag_ratio=lag_factor / len(self.dials),
            run_time=run_time,
            **kwargs
        )

    def rotate_all_dials(self, run_time=2, lag_factor=1.0):
        shuffled_dials = list(self.dials)
        random.shuffle(shuffled_dials)
        return LaggedStart(
            *(
                Rotate(dial.needle, TAU, about_point=dial.get_center())
                for dial in shuffled_dials
            ),
            lag_ratio=lag_factor / len(self.dials)
        )`,
    annotations: {
      1: "Enables PEP 604 union types (X | Y) and postponed evaluation of annotations for cleaner type hints.",
      3: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      34: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      42: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      60: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      65: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      99: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      112: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      113: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      115: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      127: "FadeOut transitions a mobject from opaque to transparent.",
      152: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      155: "FadeOut transitions a mobject from opaque to transparent.",
      190: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      211: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      229: "FadeOut transitions a mobject from opaque to transparent.",
      237: "FadeOut transitions a mobject from opaque to transparent.",
      248: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      250: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      251: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      253: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      272: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      273: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      276: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      277: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      280: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      334: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      342: "NeuralNetwork extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      396: "Class ContextAnimation inherits from LaggedStart.",
      423: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      430: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      445: "Class LabeledArrow inherits from Arrow.",
      459: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      461: "Adds a dark outline behind text/LaTeX for readability over complex 3D backgrounds.",
      471: "Class WeightMatrix inherits from DecimalMatrix.",
      524: "Class NumericEmbedding inherits from WeightMatrix.",
      568: "EmbeddingArray extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      596: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      622: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      630: "RandomizeMatrixEntries extends Animation. Custom Animation subclass. Override interpolate(alpha) to define how the animation progresses from 0 to 1.",
      647: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      651: "Class AbstractEmbeddingSequence inherits from MobjectMatrix.",
      655: "Dial extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      686: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      710: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      737: "np.linspace creates evenly spaced values over an interval — essential for parametric sampling.",
      748: "AnimationGroup plays multiple animations together with individual timing control.",
      749: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      758: "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1.",
      761: "MachineWithDials extends VGroup. VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
      801: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      813: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
    }
  };

  files["_2024/transformers/ml_basics.py"] = {
    description: "Machine learning fundamentals: introduces neural networks, gradient descent, loss functions, and backpropagation concepts that underpin the transformer architecture.",
    code: `from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *
from _2024.transformers.generation import *


class DialTest(InteractiveScene):
    def construct(self):
        # Test
        dial = Dial(radius=0.5)
        self.add(dial)
        self.play(dial.animate_set_value(0.5, run_time=1))

        # Test
        machine = MachineWithDials()
        self.add(machine)
        self.play(machine.random_change_animation())


class MLWithinDeepL(InteractiveScene):
    def construct(self):
        # Organize boxes
        kw = dict(font_size=36, opacity=0.25)
        model_boxes = VGroup(
            self.get_titled_box("Multilayer Perceptrons", BLUE_D, **kw),
            self.get_titled_box("Convolutional Neural Networks", BLUE_D, **kw),
            self.get_titled_box("Transformers", BLUE, **kw),
        )
        for box in model_boxes:
            box.box.set_width(model_boxes.get_width(), stretch=True)
        dots = Tex(R"\\vdots", font_size=72)
        model_boxes.add(dots)
        model_boxes.arrange(DOWN, buff=0.1)
        dots.shift(0.2 * DOWN)
        transformer_box = model_boxes[2]

        dl_box = self.get_titled_box(
            "Deep Learning", TEAL,
            font_size=60,
            y_space=model_boxes.get_height() + 1.0,
            x_space=2.75,
            opacity=0.05
        )

        model_boxes.next_to(dl_box.title, DOWN)

        # Animate in word
        transformer_box.save_state()
        transformer_box.box.set_opacity(0)
        transformer_box.set_height(1)
        transformer_box.move_to(np.array([-1.58, -2.01, 0]))

        self.add(transformer_box)
        self.wait()
        self.add(dl_box, transformer_box)
        self.play(LaggedStart(
            FadeIn(dl_box, scale=1.2),
            Restore(transformer_box),
            *(FadeIn(model_boxes[i]) for i in [0, 1, 3]),
        ), lag_ratio=0.75, run_time=2)
        self.wait()

        dl_box.add(model_boxes)
        self.add(dl_box)

        # Place within ML box
        ml_box = self.get_titled_box(
            "Machine Learning",
            GREEN,
            opacity=0.1,
            font_size=72,
            x_space=6.0,
            y_space=5.0
        )
        dl_box.target = dl_box.generate_target()
        blank_boxes = dl_box.box.replicate(2)
        inner_boxes = VGroup(*blank_boxes, dl_box.target)
        reg_drawing = self.get_regression_drawing()
        bayes_net = self.get_bayes_net_drawing()
        for drawing, box in zip([reg_drawing, bayes_net], blank_boxes):
            drawing.set_height(0.8 * box.get_height())
            drawing.move_to(box)
            box.add(drawing)
        inner_boxes.set_height(3.5)
        inner_boxes.arrange(RIGHT)
        inner_boxes.set_max_width(ml_box.get_width() - 0.5)
        inner_boxes.next_to(ml_box.title, DOWN, buff=1.0)

        self.add(ml_box, dl_box, blank_boxes)
        self.play(
            FadeIn(ml_box),
            MoveToTarget(dl_box),
            LaggedStartMap(FadeIn, blank_boxes, scale=2.0, lag_ratio=0.5)
        )
        self.wait()

        ml_box.add(dl_box, blank_boxes)

        # Learn from data
        words = Text("Learn from data", font_size=72)
        words.to_edge(UP, buff=MED_SMALL_BUFF)
        learn = words["Learn"][0]
        learn.save_state()
        learn.set_x(0)
        words["data"].set_color(YELLOW)
        ml_box.target = ml_box.generate_target()
        ml_box.target.scale(0.75)
        ml_box.target.to_edge(DOWN)
        arrow = Arrow(ml_box.target, words)

        self.play(
            MoveToTarget(ml_box),
            GrowFromCenter(arrow),
            TransformFromCopy(ml_box.title["Learn"][0], learn),
        )
        self.play(
            Restore(learn),
            FadeIn(words["from data"][0], lag_ratio=0.1, shift=0.2 * RIGHT),
        )
        self.wait()
        self.play(
            FadeOut(ml_box),
            FadeOut(arrow),
        )
        self.wait()

        # Go back to the box
        self.clear()
        ml_box.center()
        self.add(ml_box)

        # Pop out
        ml_box.remove(dl_box)
        ml_box.add(dl_box.copy())
        ml_box.target = ml_box.generate_target()
        ml_box.target.scale(0.25).to_edge(LEFT)
        dl_box.target = dl_box.generate_target()
        dl_box.target.scale(2.0)
        dl_box.target.next_to(ml_box.target, RIGHT, buff=0.75),
        lines = VGroup(*(
            Line(
                ml_box.target[-1].get_corner(RIGHT + v),
                dl_box.target.get_corner(LEFT + v)
            )
            for v in [UP, DOWN]
        ))
        lines.set_stroke(TEAL, 2)

        self.play(
            MoveToTarget(ml_box),
            MoveToTarget(dl_box),
            GrowFromPoint(lines[0], dl_box.get_corner(UR)),
            GrowFromPoint(lines[1], dl_box.get_corner(DR)),
            run_time=1.5,
        )
        self.wait()

        # Show a neural network
        network = NeuralNetwork([5, 10, 5])
        network.next_to(dl_box, RIGHT, buff=1.0)

        self.play(
            FadeIn(network.layers[0]),
            ShowCreation(network.lines[0], lag_ratio=0.01),
            FadeIn(network.layers[1], lag_ratio=0.5),
            run_time=2
        )
        self.play(
            ShowCreation(network.lines[1], lag_ratio=0.01),
            FadeIn(network.layers[2], lag_ratio=0.5),
            run_time=2
        )

        # Ambiently change the network
        for _ in range(6):
            self.play(
                network.animate.randomize_line_style().randomize_layer_values(),
                run_time=3,
                lag_ratio=1e-4
            )

        # Pile of matrices
        pile_words = Text("Pile of matrices")
        pile_words.next_to(network, UP)
        path_arc = -60 * DEGREES
        arrow = Arrow(dl_box.get_top(), pile_words.get_corner(UL), path_arc=path_arc)
        matrices = VGroup(*(
            WeightMatrix(shape=(8, 6), ellipses_row=None, ellipses_col=None)
            for x in range(10)
        ))
        matrices.match_width(network)
        matrices.move_to(network, UP)
        matrices.shift(0.5 * DOWN)
        matrix_shift = 0.5 * (IN + RIGHT)

        matrices.arrange(OUT, buff=0.25)
        matrices.move_to(network)

        for matrix in matrices[:-1]:
            matrix.target = matrix.generate_target()
            for entry in matrix.target.get_entries():
                dot = Dot(radius=0.05)
                dot.set_fill(entry.get_fill_color(), opacity=0.25)
                dot.move_to(entry)
                entry.become(dot)
            matrix.target[-1].set_opacity(0.25)
        matrices[-1].get_entries().set_backstroke(BLACK, 8)

        self.play(
            FadeOut(network, 2 * DOWN),
            ShowCreation(arrow),
            FadeInFromPoint(pile_words, dl_box.title.get_center(), path_arc=path_arc),
            FadeOut(network, DOWN)
        )
        mat_shift = 0.5 * IN + 0.25 * DOWN
        self.play(
            LaggedStart(*(
                Succession(
                    FadeIn(matrix, shift=mat_shift),
                    MoveToTarget(matrix)
                )
                for matrix in matrices[:-1]
            ), lag_ratio=0.25, run_time=5),
            Animation(Point()),
            FadeIn(matrices[-1], shift=mat_shift, time_span=(3.75, 4.75))
        )
        self.wait()

    def get_titled_box(self, text, color, font_size=48, y_space=0.5, x_space=0.5, opacity=0.1):
        title = Text(text, font_size=font_size)
        box = Rectangle(
            title.get_width() + x_space,
            title.get_height() + y_space
        )
        box.set_fill(interpolate_color(BLACK, color, opacity), 1)
        box.set_stroke(color, 2)
        title.next_to(box.get_top(), DOWN, buff=MED_SMALL_BUFF)
        result = VGroup(box, title)
        result.box = box
        result.title = title
        return result

    def get_regression_drawing(self):
        axes = Axes((-1, 10), (-1, 10))
        m = 0.5
        y0 = 2
        line = axes.get_graph(lambda x: y0 + m * x)
        line.set_stroke(YELLOW, 2)
        dots = VGroup(
            Dot(axes.c2p(x, y0 + m * x + np.random.normal()))
            for x in np.random.uniform(0, 10, 15)
        )

        reg_drawing = VGroup(axes, dots, line)
        return reg_drawing

    def get_bayes_net_drawing(self):
        radius = MED_SMALL_BUFF
        node = Circle(radius=radius)
        node.set_stroke(GREY_B, 2)
        node.shift(2 * DOWN)
        nodes = VGroup(
            node.copy().shift(x * RIGHT + y * UP)
            for x, y in [
                (-1, 0),  
                (1, 0),
                (-2, 2),
                (0, 2),
                (2, 2),
                (-2, 4),
                (0, 4),
            ]
        )
        edge_index_pairs = [
            (2, 0),
            (3, 0),
            (3, 1),
            (4, 1),
            (5, 2),
            (6, 3),
        ]
        edges = VGroup()
        for i1, i2 in edge_index_pairs:
            n1, n2 = nodes[i1], nodes[i2]
            edge = Arrow(
                n1.get_center(), 
                n2.get_center(),
                buff=radius,
                color=WHITE,
                stroke_width=3
            )
            edges.add(edge)

        network = VGroup(nodes, edges)
        return network


class ShowCross(InteractiveScene):
    def construct(self):
        # Test
        cross = Cross(Square(side_length=5))
        cross.set_stroke(width=[0, 30, 0])
        self.play(ShowCreation(cross))
        self.wait()


class FlashThroughImageData(InteractiveScene):
    time_per_example = 0.1

    def construct(self):
        # Images
        image_data = load_image_net_data()
        arrow = Vector(RIGHT)

        for path, text in ProgressDisplay(image_data):
            image = ImageMobject(str(path))
            label = Text(text.split(",")[0])
            label.use_winding_fill(False)
            image.next_to(arrow, LEFT)
            label.next_to(arrow, RIGHT)
            self.add(image, arrow, label)
            self.wait(self.time_per_example)
            self.remove(image, label)

            if hasattr(image, "shader_wrapper"):
                for tid in image.shader_wrapper.texture_names_to_ids.values():
                    release_texture(tid)


class FlashThroughTextData2(InteractiveScene):
    n_examples = 200
    time_per_example = 0.1
    window_size = 50
    line_len = 35
    ul_point = 5 * LEFT + 3 * UP

    def construct(self):
        # Test
        totc = read_in_book(name="tale_of_two_cities")
        words = re.split(r"\\s", totc)
        words = list(filter(lambda s: s, words))

        for n in range(self.n_examples):
            index = random.randint(0, len(words) - self.window_size)
            window = words[index:index + self.window_size]
            phrase = get_paragraph(window, line_len=self.line_len)
            phrase.move_to(self.ul_point, UL)

            word = phrase[window[-1]][-1]
            rect = SurroundingRectangle(word, buff=0.1)
            rect.set_stroke(YELLOW, 2)
            rect.set_fill(YELLOW, 0.5)

            self.add(phrase)
            self.wait(self.time_per_example)
            self.remove(phrase)


class TweakedMachine(InteractiveScene):
    n_tweaks = 200
    time_per_example = 0.1

    def construct(self):
        # Test
        machine = MachineWithDials(
            dial_config=dict(
                value_to_color_config=dict(
                    low_negative_color=BLUE_E,
                    high_negative_color=BLUE_B,
                )
            )
        )
        machine.move_to(2 * DOWN)
        machine.set_width(4)
        arrow = Vector(DOWN, stroke_width=10)
        arrow.next_to(machine, UP)

        self.add(machine, arrow)

        values = np.array([d.get_random_value() for d in machine.dials])

        for n in range(self.n_tweaks):
            nudges = np.random.uniform(-1, 1, values.shape)
            values += 0.1 * nudges
            values[values > 1.0] = 0.9
            values[values < 0.0] = 0.1
            for dial, value in zip(machine.dials, values):
                dial.set_value(value)
            self.wait(self.time_per_example)


class PremiseOfML(InteractiveScene):
    box_center = RIGHT
    n_examples = 50
    random_seed = 316
    show_matrices = False

    def construct(self):
        self.init_data()

        # Set up input and output
        machine = self.get_machine()
        machine.set_width(4)
        machine.move_to(self.box_center)
        model_label = Text("Model", font_size=72)
        model_label.move_to(machine.box)
        in_arrow = Vector(RIGHT).next_to(machine, LEFT)
        out_arrow = Vector(RIGHT).next_to(machine, RIGHT)

        self.add(machine.box)
        self.add(in_arrow, out_arrow)
        self.add(model_label)

        # Show initial input and output
        in_data, out_data = self.new_input_output_example(in_arrow, out_arrow)

        in_word, out_word = [
            Text(word).next_to(machine, UP).match_x(mob).shift_onto_screen()
            for word, mob in [("Input", in_data), ("Output", out_data)]
        ]

        self.play(
            FadeIn(in_data, lag_ratio=0.001),
            FadeIn(in_word, 0.5 * UP),
        )
        self.play(FadeOutToPoint(in_data.copy(), machine.get_left(), lag_ratio=0.005, path_arc=-60 * DEGREES))
        self.play(
            FadeInFromPoint(out_data, machine.get_right(), lag_ratio=0.1, path_arc=60 * DEGREES),
            FadeIn(out_word, 0.5 * UP)
        )
        self.wait()

        # Show code
        model_label.target = model_label.generate_target()
        model_label.target.scale(in_word[0].get_height() / model_label[0].get_height())
        model_label.target.align_to(in_word, UP)
        code = self.get_code()
        code.set_height(machine.get_height() - MED_SMALL_BUFF)
        code.set_max_width(machine.get_width() - MED_SMALL_BUFF)
        code.move_to(machine, UP).shift(SMALL_BUFF * DOWN)

        self.play(
            MoveToTarget(model_label),
            ShowIncreasingSubsets(code, run_time=3),
        )
        self.wait()

        # Show tunable parameters
        param_label = Text("Tunable parameters")
        param_label.next_to(machine, UP)
        param_label.set_color(BLUE)

        self.play(
            FadeOut(code, 0.25 * DOWN, lag_ratio=0.01),
            Write(machine.dials, lag_ratio=0.001),
            FadeOut(model_label, 0.5 * UP),
            FadeIn(param_label, 0.5 * UP),
        )
        self.play(machine.rotate_all_dials())
        self.wait()

        # Show lots of new data
        for n in range(self.n_examples):
            new_in_data, new_out_data = self.new_input_output_example(in_arrow, out_arrow)
            self.add(in_data, out_data)
            time_span = (0, 0.35)
            self.play(
                machine.random_change_animation(run_time=0.5),
                FadeOut(in_data, time_span=time_span),
                FadeOut(out_data, time_span=time_span),
                FadeIn(new_in_data, time_span=time_span),
                FadeIn(new_out_data, time_span=time_span),
            )
            in_data, out_data = new_in_data, new_out_data

        if not self.show_matrices:
            return

        # Make room
        up_shift = 1.5 * UP
        down_shift = 1.75 * DOWN

        down_group = Group(in_arrow, machine, param_label, out_arrow, out_data, out_word)
        self.play(
            in_data.animate.scale(0.75).shift(up_shift + 0.5 * UP),
            UpdateFromFunc(out_data, lambda m: m.match_y(in_data)),
            in_word.animate.shift(up_shift),
            down_group.animate.shift(down_shift),
        )

        # Create pixels
        image = in_data
        pixels = create_pixels(in_data)

        # Show input array
        in_array = NumericEmbedding(shape=(10, 10), ellipses_col=-2)
        in_array.match_height(machine)
        in_array.next_to(in_arrow, LEFT)
        image.set_opacity(0.8)

        self.play(
            TransformFromCopy(
                pixels,
                VGroup(*(in_array.get_entries().family_members_with_points())),
                run_time=2,
                lag_ratio=1e-3
            ),
            FadeInFromPoint(in_array.get_brackets(), image.get_bottom()),
            Write(in_array.get_ellipses(), time_span=(1, 2))
        )
        self.play(image.animate.set_opacity(1))
        self.wait()

        # Show one dimensional array
        vector = NumericEmbedding(length=10)
        vector.replace(in_array, dim_to_match=1)
        vector.move_to(in_array, RIGHT)

        self.remove(in_array)
        self.play(
            TransformFromCopy(in_array.get_brackets(), vector.get_brackets()),
            TransformFromCopy(in_array.get_columns()[5], vector.get_columns()[0]),
            *map(FadeOut, in_array.get_columns()),
        )
        self.wait()
        self.remove(vector)
        self.play(LaggedStart(
            TransformFromCopy(vector.get_brackets(), in_array.get_brackets()),
            TransformFromCopy(vector.get_columns()[0], in_array.get_columns()[5]),
            *(
                FadeIn(col, shift=col.get_center() - vector.get_center())
                for col in in_array.get_columns()
            )
        ))
        self.wait()

        # Show 3d tensor
        self.frame.set_field_of_view(30 * DEGREES)
        dot_array = in_array.copy()
        for entry in (*dot_array.get_entries(), *dot_array.get_ellipses()):
            dot = Dot(entry.get_center(), radius=0.06)
            entry.set_submobjects([dot])

        tensor = VGroup(*(
            dot_array.copy()
            for n in range(5)
        ))
        for layer in tensor:
            for dot in (*layer.get_entries(), *layer.get_ellipses()):
                dot.set_fill(
                    interpolate_color(GREY_C, GREY_B, random.random()),
                    opacity=0.5,
                )
                dot.set_backstroke(BLACK, 2)
        tensor.arrange(OUT, buff=0.25)
        tensor.move_to(in_array, RIGHT)
        tensor.rotate(5 * DEGREES, RIGHT)
        tensor.rotate(5 * DEGREES, UP)

        self.remove(in_array)
        self.play(TransformFromCopy(VGroup(in_array), tensor))
        self.play(Rotate(tensor, 20 * DEGREES, axis=UP, run_time=4))
        self.play(Transform(tensor, VGroup(in_array), remover=True))
        self.add(in_array)

        # Express output as an array of numbers
        values = np.random.uniform(0, 1, (10, 1))
        values[5] = 9.7
        out_array = DecimalMatrix(values, ellipses_row=-2)
        out_array.match_height(machine)
        out_array.match_y(out_arrow)
        out_array.match_x(out_word)

        self.play(
            FadeInFromPoint(out_array, machine.get_right(), lag_ratio=1e-3),
            out_data.animate.scale(0.75).fade(0.5).rotate(-PI / 2).next_to(out_array, RIGHT, buff=0.25),
        )
        self.wait()

        # Describe parameters as weights
        weights_label = Text("Weights")
        weights_label.next_to(machine, UP, buff=0.5)
        weights_label.match_color(param_label)
        equiv = Tex(R"\\Updownarrow")
        equiv.next_to(weights_label, UP)

        top_dials = machine.dials[:8]
        dial_rects = VGroup(*map(SurroundingRectangle, top_dials))
        dial_rects.set_stroke(TEAL, 2)
        dial_arrows = VGroup(*(
            Arrow(weights_label.get_bottom(), rect.get_top(), buff=0.05)
            for rect in dial_rects
        ))
        dial_arrows.set_stroke(TEAL)

        self.play(
            FadeIn(weights_label, scale=2),
            param_label.animate.next_to(equiv, UP),
            Write(equiv),
        )
        self.play(
            LaggedStart(*(
                VFadeInThenOut(VGroup(arrow, rect))
                for arrow, rect in zip(dial_arrows, dial_rects)
            ), lag_ratio=0.25, run_time=3)
        )
        self.wait()

        # Show weighted sum
        machine.dials.save_state()
        weights_label.set_backstroke(BLACK, 5)
        weights_label.target = weights_label.generate_target()
        weights_label.target.next_to(top_dials, DOWN, buff=0.25)
        weighted_sum = Tex(
            R"w_1 x_1 + w_2 x_2 + w_3 x_3 + \\cdots + w_n x_n",
            font_size=42,
        )
        weighted_sum.next_to(machine, UP, buff=1.0)
        weight_parts = weighted_sum[re.compile(r"w_\\d|w_n")]
        weight_parts.set_color(BLUE)
        data_parts = weighted_sum[re.compile(r"x_\\d|x_n")]
        data_parts.set_color(GREY_A)

        indices = [0, 1, 2, -1]
        dial_lines = VGroup(*(
            Line(top_dials[n].get_top(), weight_parts[n].get_bottom(), buff=0.1)
            for n in indices
        ))
        ellipses = weighted_sum[R"\\cdots"]
        dial_lines.set_stroke(BLUE_B, 1)

        column = in_array.get_columns()[-1]
        col_rect = SurroundingRectangle(column)
        col_rect.set_stroke(YELLOW, 2)

        self.play(ShowCreation(col_rect))
        self.play(
            FadeOut(VGroup(param_label, equiv), UP),
            MoveToTarget(weights_label),
            machine.dials[8:].animate.fade(0.75),
            LaggedStart(*(
                TransformFromCopy(column[n], data_parts[n])
                for n in indices
            )),
            Group(in_data, in_word).animate.to_edge(LEFT, buff=0.25)
        )
        self.play(
            Write(weighted_sum["+"]),
            Write(weighted_sum[R"\\cdots"]),
            LaggedStart(*(
                FadeTransform(top_dials[n].copy(), weight_parts[n])
                for n in indices
            )),
            LaggedStartMap(ShowCreation, dial_lines),
            run_time=1
        )
        self.wait()
        for x in range(3):
            self.play(*(
                dial.animate_set_value(dial.get_random_value())
                for dial in top_dials
            ))

        # Wrap a function around it
        func_wrapper = Tex(R"f()")
        func_wrapper[:2].next_to(weighted_sum, LEFT, buff=SMALL_BUFF)
        func_wrapper[2].next_to(weighted_sum, RIGHT, buff=SMALL_BUFF)
        func_wrapper.set_color(PINK)

        nl_words = Text("Simple nonlinear\\nfunction", font_size=42, alignment="LEFT")
        nl_words.next_to(func_wrapper, UP, buff=1.5, aligned_edge=LEFT)
        nl_words.match_color(func_wrapper)
        nl_arrow = Arrow(nl_words, func_wrapper[0].get_top())
        nl_arrow.match_color(nl_words)

        self.play(
            FadeIn(func_wrapper),
            FadeIn(nl_words, lag_ratio=0.1),
            ShowCreation(nl_arrow),
        )
        self.wait()

        # Show next layer
        weights_label.target = weights_label.generate_target()
        weights_label.target.next_to(weighted_sum, UP, buff=1.0)
        dial_lines.target = VGroup(*(
            Line(
                weights_label.target, weight_parts[index].get_top(),
                buff=SMALL_BUFF
            )
            for index in indices
        ))
        dial_lines.target.match_style(dial_lines)

        layer1 = NumericEmbedding(shape=(10, 5), ellipses_col=-2)
        layer1.match_height(in_array)
        layer1.next_to(in_arrow, RIGHT)
        mid_arrow = in_arrow.copy()
        mid_arrow.next_to(layer1, RIGHT)
        dots = Tex(R"\\dots").next_to(mid_arrow, RIGHT)

        expr_rect = SurroundingRectangle(func_wrapper)
        expr_rect.set_stroke(PINK, 2)
        x01_rect = SurroundingRectangle(layer1.elements[0])
        x01_rect.match_style(expr_rect)
        rect_lines = VGroup(*(
            Line(expr_rect.get_corner(DOWN + v), x01_rect.get_corner(UP + v))
            for v in [LEFT, RIGHT]
        ))
        rect_lines.match_style(expr_rect)

        self.play(LaggedStart(
            FadeOut(weights_label),
            FadeOut(dial_lines),
            FadeOut(nl_words),
            FadeOut(nl_arrow),
            FadeOut(col_rect),
            FadeOut(machine),
            FadeIn(expr_rect),
        ))
        self.play(
            TransformFromCopy(in_array.get_brackets(), layer1.get_brackets()),
            TransformFromCopy(in_arrow, mid_arrow),
            out_arrow.animate.next_to(dots, RIGHT),
            Write(dots),
        )
        self.play(
            TransformFromCopy(expr_rect, x01_rect),
            ShowCreation(rect_lines, lag_ratio=0),
            FadeInFromPoint(layer1.elements[0], expr_rect.get_center()),
        )
        self.play(ShowIncreasingSubsets(layer1[1:-1]))
        self.add(layer1)
        self.wait()

        # Highlight a subset of the data
        in_subset = VGroup(*(
            elem
            for row in in_array.get_rows()[:3]
            for elem in row[:3]
        ))
        in_subset_rects = VGroup(*map(SurroundingRectangle, in_subset))
        data_part_rects = VGroup(*map(SurroundingRectangle, data_parts))
        self.play(
            LaggedStartMap(ShowCreationThenFadeOut, in_subset_rects, lag_ratio=0.02),
            LaggedStartMap(ShowCreationThenFadeOut, data_part_rects, lag_ratio=0.04),
            run_time=3
        )
        self.wait()

        # Show added layers
        to_fade = VGroup(
            func_wrapper, expr_rect, rect_lines, x01_rect,
            weighted_sum
        )

        self.play(
            LaggedStartMap(FadeOut, to_fade, run_time=1),
            in_arrow.animate.scale(0.5, about_edge=LEFT),
            layer1.animate.rotate(70 * DEGREES, UP).next_to(in_arrow, RIGHT, buff=-0.25),
            mid_arrow.animate.scale(0.5).next_to(in_arrow, RIGHT, buff=0.75),
        )

        layer1_group = VGroup(layer1, mid_arrow)
        layer2_group, layer3_group = layer1_group.replicate(2)
        layer2_group.next_to(layer1_group, RIGHT, buff=SMALL_BUFF)
        layer3_group.next_to(layer2_group, RIGHT, buff=SMALL_BUFF)
        self.play(TransformFromCopy(layer1_group, layer2_group))
        self.play(
            TransformFromCopy(layer2_group, layer3_group),
            VGroup(dots, out_arrow).animate.next_to(layer3_group, RIGHT),
        )
        self.play(
            LaggedStart(*(
                dot.animate.shift(0.1 * UP).set_anim_args(rate_func=there_and_back)
                for dot in dots
            ), lag_ratio=0.25)
        )
        self.wait()

        # Bring back machine
        layers = VGroup(layer1_group, layer2_group, layer3_group, dots)

        self.play(
            FadeIn(machine, scale=0.8),
            FadeIn(weights_label, shift=DOWN),
            ShowCreation(dial_lines, lag_ratio=0.1),
            FadeIn(weighted_sum, shift=UP),
            FadeOut(layers, scale=0.8),
        )
        self.wait()
        self.play(
            machine.random_change_animation()
        )
        self.wait()

        # Show a matrix
        frame = self.frame
        matrix, vector, equals, rhs = get_full_matrix_vector_product()
        mat_prod_group = VGroup(matrix, vector, equals, rhs)
        mat_prod_group.next_to(machine, UP, buff=2.0)
        mat_prod_group.shift(0.5 * LEFT)

        p0 = machine.get_corner(UL)
        p1 = matrix.get_corner(DL)
        p2 = machine.get_corner(UR)
        p3 = rhs.get_corner(DR)
        brace = VGroup(
            CubicBezier(p0, p0 + 2 * UP, p1 + 2 * DOWN, p1 + 0.1 * DOWN),
            CubicBezier(p2, p2 + 2 * UP, p3 + 2 * DOWN, p3 + 0.1 * DOWN),
        )
        brace.set_stroke(WHITE, 5)

        self.play(LaggedStart(
            TransformFromCopy(data_parts, vector.get_columns()[0]),
            TransformFromCopy(weight_parts, matrix.get_rows()[0]),
            FadeTransform(weighted_sum, rhs.get_rows()[0]),
            frame.animate.set_height(10, about_edge=DOWN),
            FadeOut(in_data, DOWN),
            FadeOut(out_data, DOWN),
            in_word.animate.next_to(in_array, UP),
            FadeIn(matrix, lag_ratio=0.1),
            ShowCreation(brace, lag_ratio=0),
            weights_label.animate.set_height(0.5).next_to(matrix, UP, buff=MED_SMALL_BUFF),
            Uncreate(dial_lines, lag_ratio=0.1),
            FadeOut(col_rect),
            machine.dials.animate.restore(),
            FadeIn(vector.get_brackets()),
            FadeIn(rhs.get_brackets()),
            FadeIn(equals),
            run_time=3,
            lag_ratio=0.1,
        ))
        self.wait()

        # Animate matrix vector product
        ghost_row = rhs.get_rows()[0].copy()
        ghost_row.set_opacity(0.25)
        self.add(ghost_row)
        show_symbolic_matrix_vector_product(
            self, matrix, vector, rhs,
            run_time_per_row=1.5
        )
        self.remove(ghost_row)
        self.wait()

        # Associate weights with dials
        w_elems = matrix.get_entries()
        moving_dials = machine.dials[:len(w_elems)].copy()
        moving_dials.target = moving_dials.generate_target()
        for dial, w_elem in zip(moving_dials.target, w_elems):
            dial.move_to(w_elem)
            dial.scale(2)

        self.play(
            w_elems.animate.set_opacity(0.25),
            MoveToTarget(moving_dials, run_time=2),
        )
        self.play(
            LaggedStart(*(
                dial.animate_set_value(dial.get_random_value())
                for dial in moving_dials
            ), lag_ratio=0.02, run_time=3)
        )
        self.wait()
        self.play(
            FadeOut(moving_dials),
            w_elems.animate.set_opacity(1),
        )

        # Vector an data slice
        v_rect = SurroundingRectangle(vector.get_entries())
        self.play(
            ShowCreation(v_rect),
            ShowCreation(col_rect),
        )
        self.wait()
        self.play(
            FadeOut(v_rect),
            FadeOut(col_rect),
        )
        self.wait()

        # Show many matrices
        lhs = VGroup(matrix, vector)
        small_mat_product = Tex(R"W_{10} v_{11}")
        small_mat_product[R"W_{10}"].set_color(BLUE)
        w_index = small_mat_product.make_number_changeable("10")
        v_index = small_mat_product.make_number_changeable("11")
        small_mat_products = VGroup()
        n_rows, n_cols = 16, 8
        for n in range(n_rows * n_cols):
            w_index.set_value(n + 1)
            v_index.set_value(n + 1)
            new_prod = small_mat_product.copy()
            new_prod.arrange(RIGHT, buff=SMALL_BUFF, aligned_edge=DOWN)
            small_mat_products.add(new_prod)
        small_mat_products.arrange_in_grid(n_rows, n_cols, v_buff_ratio=2.0)
        small_mat_products.replace(machine.dials)

        mv_label = Text("matrix-vector products")
        mv_label.next_to(machine, UP, buff=1.0)
        mv_label[-1].set_opacity(0)
        mv_top_label = Text("Many, many")
        mv_top_label.next_to(mv_label, UP)
        mv_arrows = VGroup(*(
            Arrow(mv_label.get_bottom(), smp.get_top(), buff=0.1)
            for smp in small_mat_products
        ))

        self.play(
            FadeTransform(mat_prod_group, small_mat_products[0]),
            Uncreate(brace, lag_ratio=0),
            FadeOut(machine.dials, run_time=0.5),
            FadeTransform(weights_label, mv_label),
            GrowFromPoint(mv_arrows[0], weights_label.get_bottom()),
            frame.animate.set_height(FRAME_HEIGHT).move_to(DOWN).set_anim_args(time_span=(1, 2)),
            run_time=2,
        )
        self.wait()
        self.remove(mv_arrows)
        self.play(
            FadeIn(mv_top_label, UP),
            mv_label[-1].animate.set_opacity(1),
            ShowIncreasingSubsets(small_mat_products, rate_func=linear, run_time=12, int_func=np.ceil),
            ShowSubmobjectsOneByOne(mv_arrows, rate_func=linear, run_time=12, int_func=np.ceil),
        )
        self.remove(mv_arrows)
        self.play(FadeOut(mv_arrows[-1]))
        self.wait()

    def init_data(self):
        self.image_data = load_image_net_data()

    def new_input_output_example(self, in_arrow, out_arrow) -> tuple[Mobject, Mobject]:
        path, label_text = random.choice(self.image_data)
        image = ImageMobject(str(path))
        image.set_width(4)
        image.next_to(in_arrow, LEFT)
        label = Text(label_text.split(",")[0])
        label.set_max_width(2.5)
        label.next_to(out_arrow, RIGHT)
        return image, label

    def get_machine(self):
        return MachineWithDials()

    def get_code(self):
        # Test
        src = """
            #include <opencv2/opencv.hpp>
            #include <iostream>

            using namespace cv;
            using namespace std;

            int main(int argc, char** argv) {
                Mat image = imread(argv[1], IMREAD_GRAYSCALE);
                if (image.empty()) {
                    cout << "Could not open image" << endl;
                    return -1;
                }

                // Blur the image to reduce noise
                Mat blurredImage;
                GaussianBlur(image, blurredImage, Size(5, 5), 0);

                // Detect edges with Canny
                Mat edges;
                Canny(blurredImage, edges, 100, 200);
        """
        return Code(src, language="C++", alignment="LEFT")


class PremiseOfMLWithText(PremiseOfML):
    random_seed = 316

    def init_data(self):
        totc = read_in_book(name="tale_of_two_cities")
        words = re.split(r"\\s", totc)
        words = list(filter(lambda s: s, words))
        self.all_words = words

    def new_input_output_example(self, in_arrow, out_arrow):
        words = self.all_words
        window_size = 25
        index = random.randint(0, len(words) - window_size)
        window = words[index:index + window_size]
        in_text = get_paragraph(window[:-1], line_len=25)
        in_text.set_max_width(4)
        in_text.next_to(in_arrow, LEFT)
        out_text = Text(window[-1])
        out_text.next_to(out_arrow, RIGHT)
        return in_text, out_text

    def get_machine(self):
        machine = super().get_machine()
        machine.add(VectorizedPoint().next_to(machine, DOWN, buff=0.5))
        return machine

    def get_code(self):
        # Test
        src = """
            using namespace std;

            vector<string> findCapitalizedWords(const string& text) {
                vector<string> capitalizedWords;
                stringstream ss(text);
                string word;

                while (ss >> word) {
                    // Check for uppercase
                    if (!word.empty() && isupper(word[0])) {
                        capitalizedWords.push_back(word);
                    }
                }

                return capitalizedWords;
            }

            int main() {
                string text;
                cout << "Enter text: ";
                getline(cin, text); // Using getline to read spaces
        """
        return Code(src, language="C++", alignment="LEFT")


class PremiseOfMLWithMatrices(PremiseOfML):
    # Skip to animation 9
    show_matrices = True
    n_examples = 0
    random_seed = 6


class LinearRegression(InteractiveScene):
    radom_seed = 1

    def construct(self):
        # Set up axes
        x_min, x_max = (-1, 12)
        y_min, y_max = (-1, 10)
        axes = Axes((x_min, x_max), (y_min, y_max), width=12, height=6)
        axes.to_edge(DOWN)
        self.add(axes)

        # Add data
        n_data_points = 30
        m = 0.75
        y0 = 1

        data = np.array([
            (x, y0 + m * x + 0.75 * np.random.normal(0, 1))
            for x in np.random.uniform(2, x_max, n_data_points)
        ])
        points = axes.c2p(data[:, 0], data[:, 1])
        dots = DotCloud(points)

        dots.set_color(YELLOW)
        dots.set_glow_factor(1)
        dots.set_radius(0.075)

        self.add(dots)

        # Make title
        title = Text("Linear Regression", font_size=72)
        title.to_edge(UP)

        # Show line
        m_tracker = ValueTracker(m)
        y0_tracker = ValueTracker(y0)
        line = Line()
        line.set_stroke(TEAL, 2)

        def update_line(line):
            curr_y0 = y0_tracker.get_value()
            curr_m = m_tracker.get_value()
            line.put_start_and_end_on(
                axes.c2p(0, curr_y0),
                axes.c2p(x_max, curr_y0 + curr_m * x_max),
            )

        line.add_updater(update_line)

        self.play(
            FadeIn(title, UP),
            ShowCreation(line),
        )
        self.wait()

        # Label inputs and outputs
        in_labels = VGroup(Text("Input"), Text("Square footage"))
        out_labels = VGroup(Text("Output"), Text("Price"))
        for in_label in in_labels:
            in_label.next_to(axes.x_axis, DOWN, buff=0.1, aligned_edge=RIGHT)
        for out_label in out_labels:
            out_label.rotate(90 * DEGREES)
            out_label.next_to(axes.y_axis, LEFT, aligned_edge=UP)

        self.play(LaggedStart(
            FadeIn(in_labels[0], lag_ratio=0.1),
            FadeIn(out_labels[0], lag_ratio=0.1),
            lag_ratio=0.5,
        ))
        self.wait()
        self.play(LaggedStart(
            FadeTransform(*in_labels),
            FadeTransform(*out_labels),
            lag_ratio=0.8,
        ))
        self.wait()

        # Emphasize line
        self.play(
            VShowPassingFlash(
                line.copy().set_stroke(BLUE, 8).scale(1.1).insert_n_curves(100),
                time_width=1.5,
                run_time=2
            ),
        )
        self.wait()

        # Add line parameter updaters
        words = ["slope", "y-intercept"]
        value_ranges = [(0, 2, 0.2), (-2, 3, 0.5)]
        m_label, y0_label = labels = VGroup(
            VGroup(
                Dial(value_range=value_range),
                Text(f"{text} = "),
                DecimalNumber(),
            )
            for text, value_range in zip(words, value_ranges)
        )
        for label, tracker in zip(labels, [m_tracker, y0_tracker]):
            label[0].set_height(2 * label[2].get_height())
            label.arrange(RIGHT)
            label[0].f_always.set_value(tracker.get_value)
            label[2].f_always.set_value(tracker.get_value)
        labels.arrange(DOWN, aligned_edge=LEFT)
        labels.next_to(axes.y_axis, RIGHT, buff=1.0)
        labels.to_edge(UP)

        self.play(
            FadeOut(title, UP),
            FadeIn(m_label, UP),
        )
        self.play(
            m_tracker.animate.set_value(1.5),
            run_time=2,
        )
        self.play(FadeIn(y0_label, UP))
        self.play(
            y0_tracker.animate.set_value(-2),
            run_time=2
        )
        self.wait()

        # Tweak line parameters
        for n in range(10):
            alpha = random.random()
            if alpha > 0.5:
                alpha += 1
            new_m = interpolate(m_tracker.get_value(), m, alpha)
            new_y0 = interpolate(y0_tracker.get_value(), y0, alpha)
            self.play(LaggedStart(
                m_tracker.animate.set_value(new_m),
                y0_tracker.animate.set_value(new_y0),
                run_time=1.5,
                lag_ratio=0.25,
            ))
            self.wait(0.5)


class ShowGPT3Numbers(InteractiveScene):
    def construct(self):
        # Title
        gpt3_label = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * gpt3_label.get_height())
        title = VGroup(openai_logo, gpt3_label)
        title.arrange(RIGHT)
        title.to_edge(UP)

        self.add(title)

        # 175b weights
        n_param = 175_181_291_520
        weights_count = Integer(n_param, color=BLUE)
        weights_text = VGroup(Text("Total parameters:"), weights_count)
        weights_text.arrange(RIGHT, buff=MED_SMALL_BUFF)
        weights_text.next_to(title, DOWN, buff=1.0)
        weights_arrow = Arrow(weights_count, gpt3_label, stroke_width=6, buff=0.2)

        param_shape = (8, 24)
        pre_dials = Dial().get_grid(*param_shape)
        dial_matrix = MobjectMatrix(
            pre_dials, *param_shape,
            ellipses_row=-2,
            ellipses_col=-2,
        )
        dial_matrix.set_width(FRAME_WIDTH)
        dial_matrix.next_to(weights_text, DOWN, buff=MED_SMALL_BUFF)

        dials = dial_matrix.get_entries()
        dots = dial_matrix.get_ellipses()

        self.play(
            FadeIn(weights_text[:-1], time_span=(0, 3)),
            CountInFrom(weights_count, 0),
            GrowArrow(weights_arrow, time_span=(0, 3)),
            LaggedStartMap(FadeIn, pre_dials, scale=3, lag_ratio=0.1),
            run_time=10,
        )
        self.play(
            LaggedStart(
                (dial.animate_set_value(dial.get_random_value())
                for dial in dials),
                lag_ratio=1.0 / len(dials),
                run_time=5
            )
        )
        self.wait()

        # Change name to weights
        new_name = Text("Total weights: ")
        new_name.move_to(weights_text[0], RIGHT)

        self.play(
            Transform(weights_text[0]["Total"][0], new_name["Total"][0]),
            Transform(weights_text[0]["parameters:"][0], new_name["weights:"][0]),
        )
        self.wait()

        # Organize dials into matrices
        mat_text = Text("Organized into 27,938 matrices")
        mat_text["27,938"].set_color(TEAL)
        mat_text.next_to(weights_text, DOWN, buff=MED_SMALL_BUFF)
        mat_text.shift((weights_count.get_x(LEFT) - mat_text["27,938"].get_x(LEFT)) * RIGHT)

        mat_grid_shape = n, m = (3, 7)
        matrices = VGroup(
            WeightMatrix(shape=(5, 5))
            for n in range(np.product(mat_grid_shape))
        )
        matrices.arrange_in_grid(
            *mat_grid_shape,
            v_buff_ratio=0.3,
            h_buff_ratio=0.2,
        )
        matrices.set_width(FRAME_WIDTH - 1)
        mat_dots = VGroup(
            *(
                Tex(R"\\dots").next_to(mat, RIGHT)
                for mat in matrices[m - 1::m]
            ),
            *(
                Tex(R"\\vdots").next_to(mat, DOWN)
                for mat in matrices[-m:]
            )
        )
        matrices_group = VGroup(matrices, mat_dots)
        matrices_group.set_width(FRAME_WIDTH - 1)
        matrices_group.next_to(mat_text, DOWN, buff=0.5)
        matrices_group.set_x(0)
        all_entries = VGroup(
            entry
            for mat in matrices
            for row in mat.get_rows()
            for entry in row
        )

        pre_entries = []
        height = all_entries[0].get_height()
        for n, entry in enumerate(all_entries):
            index = n * len(dials) // len(all_entries)
            dial = dials[min(index, len(dials) - 1)].copy()
            dial.target = dial.generate_target()
            dial.target.set_height(height)
            dial.target.move_to(entry)
            pre_entries.append(dial)
        pre_entries = VGroup(*pre_entries)

        self.remove(dial_matrix)
        lag_ratio = 1 / len(all_entries)
        self.play(
            Write(mat_text),
            LaggedStartMap(MoveToTarget, pre_entries, lag_ratio=lag_ratio),
            TransformFromCopy(dots, mat_dots),
            *(FadeIn(mat.get_brackets()) for mat in matrices)
        )
        self.play(
            FadeOut(pre_entries, lag_ratio=0.2 * lag_ratio),
            FadeIn(all_entries, lag_ratio=0.2 * lag_ratio),
            run_time=2
        )
        self.add(matrices)
        self.wait()

        # Show 8 different categories
        count_text = VGroup(weights_text, mat_text)
        title_scale_factor = 0.75
        count_text.target = count_text.generate_target()
        count_text.target.scale(title_scale_factor)
        count_text.target.to_edge(UP, MED_SMALL_BUFF).to_edge(LEFT)
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.next_to(count_text.target, DOWN).set_x(0)
        h_line.insert_n_curves(10)
        h_line.set_stroke(width=[0, 3, 3, 3, 0])

        category_names = VGroup(*map(TexText, [
            "Embedding",
            "Key",
            "Query",
            # "Value",  # Dumb alignment hack
            # "Output",
            R"Value$_\\downarrow$",
            R"Value$_\\uparrow$",
            "Up-projection",
            "Down-projection",
            "Unembedding",
        ]))
        # category_names[3][-1].set_fill(BLACK)  # Dumb alignment hack
        category_names.arrange(DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        category_names.set_height(5.5)
        category_names.next_to(h_line, DOWN, buff=MED_LARGE_BUFF)
        category_names.to_edge(LEFT, buff=0.5)
        category_names.set_fill(border_width=0.2)

        mat_index = 0
        counts = [1, * 6 * [3], 1]
        mat_groups = VGroup()
        for name, count, dots in zip(category_names, counts, mat_dots):
            new_mat_index = mat_index + count
            mat_group = matrices[mat_index:new_mat_index]
            mat_index = new_mat_index

            mat_group.target = mat_group.generate_target()
            if len(mat_group) > 1:
                mat_group.target.add(*mat_group.copy())
            mat_group.target.arrange(RIGHT, buff=LARGE_BUFF)
            mat_group.target.set_height(0.25)
            mat_group.target.next_to(category_names, RIGHT)
            mat_group.target.match_y(name)

            dots.target = dots.generate_target()
            if dots.get_width() < dots.get_height():
                dots.target.rotate(90 * DEGREES)
            dots.target.next_to(mat_group.target, RIGHT)
            mat_groups.add(mat_group)
        mat_dots[0].target.set_opacity(0)
        mat_dots[7].target.set_opacity(0)

        n_groups = len(category_names)
        self.play(LaggedStart(
            MoveToTarget(count_text),
            title.animate.scale(title_scale_factor).next_to(count_text.target, RIGHT, LARGE_BUFF),
            FadeOut(weights_arrow),
            GrowFromCenter(h_line),
            FadeIn(category_names),
            LaggedStart(map(MoveToTarget, mat_groups), lag_ratio=0.05),
            LaggedStart(map(MoveToTarget, mat_dots[:n_groups]), lag_ratio=0.05),
            LaggedStart(map(FadeOut, mat_dots[n_groups:]), lag_ratio=0.05),
            FadeOut(matrices[sum(counts):]),
        ))

        # Add lines
        h_lines = Line(LEFT, RIGHT).set_width(13).replicate(n_groups)
        h_lines.set_stroke(WHITE, 1, 0.5)
        for name, line in zip(category_names, h_lines):
            line.next_to(name, DOWN, buff=0.1, aligned_edge=LEFT)
            name.line = line
        v_line = Line(
            mat_groups.get_corner(DL) + 0.5 * DOWN,
            mat_groups.get_corner(UL) + 0.25 * UP,
        )
        v_line.shift(SMALL_BUFF * LEFT)
        v_line.match_style(h_lines)

        self.play(
            Write(h_lines),
            Write(v_line),
        )
        self.wait()

        # Prepare expressions for parameter counts
        const_to_value = {
            "n_vocab": 50_257,
            "d_embed": 12_288,
            "d_query": 128,
            "d_value": 128,
            "n_heads": 96,
            "n_layers": 96,
            "n_neurons": 4 * 12_288,
        }
        const_lists = [
            ["d_embed", "n_vocab"],
            ["d_query", "d_embed", "n_heads", "n_layers",],
            ["d_query", "d_embed", "n_heads", "n_layers",],
            ["d_value", "d_embed", "n_heads", "n_layers",],
            ["d_embed", "d_value", "n_heads", "n_layers"],
            ["n_neurons", "d_embed", "n_layers"],
            ["d_embed", "n_neurons", "n_layers"],
            ["n_vocab", "d_embed"],
        ]

        def get_product_expression(category, consts, font_size=30, suffix=None):
            values = [const_to_value[const] for const in consts]
            result = np.product(values)
            result_str = "{:,}".format(result)
            expr = VGroup()
            expr = Text(
                " * ".join(consts) + " = " + result_str,
                font_size=font_size,
            )
            expr.next_to(v_line, RIGHT)
            expr.align_to(category.line, DOWN)
            expr.shift(0.25 * expr.get_height() * UP)
            expr.rhs = expr[result_str]
            expr.rhs.set_color(BLUE)

            counts = VGroup(
                Integer(
                    const_to_value[const],
                    font_size=0.8 * font_size,
                )
                for const in consts
            )
            counts.next_to(expr, UP, buff=0.05)
            for count, const in zip(counts, consts):
                count.match_x(expr[const])
            counts.set_fill(GREY_B)

            result = VGroup(expr, counts)

            if suffix is not None:
                label = Text(suffix)
                label.match_height(expr)
                label.next_to(expr, RIGHT, buff=MED_SMALL_BUFF)
                result.add(label)

            return result

        product_expressions = VGroup(
            get_product_expression(category, consts)
            for category, consts in zip(category_names, const_lists)
        )
        exprs = [pe[0] for pe in product_expressions]
        counts = [pe[1] for pe in product_expressions]

        # Embedding
        def highlight_category(*indices):
            category_names.target = category_names.generate_target()
            category_names.target.set_fill(opacity=0.15, border_width=0)
            for index in indices:
                category_names.target[index].set_fill(opacity=1, border_width=0.5)
            return MoveToTarget(category_names)

        self.play(
            FadeOut(mat_groups),
            FadeOut(mat_dots[1:7]),
            highlight_category(0)
        )
        self.play(
            FadeIn(exprs[0]),
            FadeIn(counts[0], 0.25 * UP),
        )
        self.wait()

        # Unembedding
        total = Integer(2 * 12_288 * 50_257)
        total.to_edge(RIGHT, buff=1.0)
        total.set_color(BLUE)
        total_box = SurroundingRectangle(total, buff=0.25)
        total_box.set_fill(BLACK, 1)
        total_box.set_stroke(WHITE, 2)
        lines = VGroup(*(Line(exprs[i].get_right(), total_box) for i in [0, 7]))
        lines.set_stroke(BLUE, 2)

        self.play(
            highlight_category(0, 7),
            TransformMatchingStrings(exprs[0].copy(), exprs[7]),
            TransformFromCopy(counts[0][0].copy(), counts[7][1]),
            TransformFromCopy(counts[0][1].copy(), counts[7][0]),
            run_time=2
        )
        self.wait()
        self.play(
            ShowCreation(lines, lag_ratio=0),
            FadeIn(total_box),
            FadeTransform(exprs[0][-11:].copy(), total),
            FadeTransform(exprs[7][-11:].copy(), total),
        )
        self.wait()
        self.play(FlashAround(weights_count, time_width=1.5, run_time=2))
        self.wait()
        self.play(
            FadeOut(lines),
            FadeOut(total_box),
            FadeOut(total),
        )
        self.wait()

        # Attention matrices
        covered_categories = [0, 7]
        att_categories = [1, 2, 3, 4]
        per_head_factors = [
            ["d_query", "d_embed"],
            ["d_query", "d_embed"],
            ["d_value", "d_embed"],
            ["d_embed", "d_value"],
        ]
        per_head_exprs = VGroup(
            get_product_expression(name, factors, suffix="per head")
            for name, factors in zip(category_names[1:5], per_head_factors)
        )
        per_layer_exprs = VGroup(
            get_product_expression(name, factors + ["n_heads"], suffix="per layer")
            for name, factors in zip(category_names[1:5], per_head_factors)
        )
        full_att_exprs = product_expressions[1:5]
        for group in [per_head_exprs, per_layer_exprs, full_att_exprs]:
            sum_box = SurroundingRectangle(
                VGroup(expr[0].rhs for expr in group)
            )
            sum_box.set_stroke(BLUE, 2)
            sum_label = Integer(sum(
                np.product(list(count.get_value() for count in expr[1]))
                for expr in group
            ))
            sum_label.set_color(BLUE)
            sum_label.next_to(sum_box, DOWN)
            sum_box.add(sum_label)
            group.sum_box = sum_box

        self.play(
            *(
                product_expressions[i].animate.set_fill(opacity=0.25, border_width=0)
                for i in covered_categories
            ),
            highlight_category(att_categories[0]),
            FadeIn(per_head_exprs[0], shift=0.5 * RIGHT)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, per_head_exprs[1:], shift=0.5 *DOWN, lag_ratio=0.5),
            highlight_category(*att_categories),
        )
        self.wait()
        self.play(FadeIn(per_head_exprs.sum_box, run_time=3, rate_func=there_and_back_with_pause))
        self.wait()
        self.play(
            FadeOut(per_head_exprs),
            FadeIn(per_layer_exprs),
        )
        self.wait()
        self.play(FadeIn(per_layer_exprs.sum_box, run_time=3, rate_func=there_and_back_with_pause))
        self.wait()
        self.play(
            FadeOut(per_layer_exprs),
            FadeIn(full_att_exprs),
        )
        self.wait()
        self.play(FadeIn(full_att_exprs.sum_box))
        self.wait()

        # Compare with total weights
        total_weights_rect = SurroundingRectangle(weights_count)
        total_weights_rect.set_stroke(BLUE_B, 2)
        box = full_att_exprs.sum_box.copy()
        box.remove(box.submobjects[0])
        self.play(Transform(box, total_weights_rect))
        self.wait()
        self.play(
            FadeOut(box),
            FadeOut(full_att_exprs.sum_box),
        )
        self.wait()

        # MLP matrices
        mlp_categories = [5, 6]
        mlp_exprs = product_expressions[5:7]
        per_layer_exprs = VGroup(
            get_product_expression(category_names[i], const_lists[i][:2], suffix="per layer")
            for i in mlp_categories
        )

        self.play(
            full_att_exprs.animate.set_fill(opacity=0.25, border_width=0),
            highlight_category(*mlp_categories),
        )
        self.wait()
        self.play(FadeIn(per_layer_exprs[0]))
        self.wait()
        self.play(
            TransformMatchingStrings(per_layer_exprs[0][0].copy(), per_layer_exprs[1][0]),
            TransformFromCopy(per_layer_exprs[0][1][0], per_layer_exprs[1][1][1]),
            TransformFromCopy(per_layer_exprs[0][1][1], per_layer_exprs[1][1][0]),
            TransformFromCopy(per_layer_exprs[0][2], per_layer_exprs[1][2]),
            run_time=1
        )
        self.wait()
        self.play(
            FadeOut(per_layer_exprs),
            FadeIn(mlp_exprs),
        )
        self.wait()

        # Sum up MLP right hand sides
        rhs_rect = SurroundingRectangle(VGroup(expr[0].rhs for expr in mlp_exprs))
        rhs_rect.set_stroke(BLUE, 2)
        rhs_rect.stretch(1.2, 1, about_edge=DOWN)
        c2v = const_to_value
        mlp_total = Integer(2 * c2v["n_neurons"] * c2v["d_embed"] * c2v["n_layers"])
        mlp_total.next_to(rhs_rect)
        mlp_total.set_color(BLUE)
        mlp_total_rect = BackgroundRectangle(mlp_total)
        mlp_total_rect.set_fill(BLACK, 1)

        self.play(
            FadeIn(rhs_rect),
            FadeIn(mlp_total_rect),
            FadeTransform(mlp_exprs[0][0].rhs.copy(), mlp_total),
            FadeTransform(mlp_exprs[1][0].rhs.copy(), mlp_total),
        )
        self.wait()

        # Align all right hand sides
        self.play(
            category_names.animate.set_fill(opacity=1, border_width=0.5),
            product_expressions.animate.set_fill(opacity=1, border_width=0.5),
        )

        all_rhss = VGroup(
            VGroup(expr[0]["="][0], expr[0].rhs)
            for expr in product_expressions
        )
        all_rhss.target = all_rhss.generate_target()
        for mob in all_rhss.target:
            mob.align_to(product_expressions, RIGHT)
            mob.shift(0.5 * RIGHT)
        all_rhss_rect = SurroundingRectangle(all_rhss.target)
        all_rhss_rect.match_style(rhs_rect)

        self.play(
            FadeOut(mlp_total_rect, RIGHT),
            FadeOut(mlp_total, RIGHT),
            ReplacementTransform(rhs_rect, all_rhss_rect),
            MoveToTarget(all_rhss)
        )
        self.wait()

        # Move weights count
        self.play(LaggedStart(
            h_line.animate.scale(0.5, about_edge=LEFT),
            weights_text.animate.arrange(DOWN).scale(1.5).next_to(all_rhss_rect, UP),
            FadeOut(mat_text, LEFT),
            title.animate.to_edge(LEFT, buff=2.5),
            lag_ratio=0.2,
            run_time=2
        ))
        self.wait()


class DistinguishWeightsAndData(InteractiveScene):
    def construct(self):
        # Set up titles
        weights_title, data_title = titles = VGroup(
            Text(word, font_size=60)
            for word in ["Weights", "Data"]
        )
        weights_title.set_color(BLUE)
        data_title.set_color(GREY_B)

        for title, sign in zip(titles, [-1, 1]):
            title.set_x(sign * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=0.25)
            underline = Underline(title, stretch_factor=1.5)
            underline.match_color(title)
            underline.set_y(title[0].get_y(DOWN) - 0.1)
            title.add(underline)

        v_line = Line(UP, DOWN).set_height(4.5)
        v_line.to_edge(UP, buff=0)
        v_line.set_stroke(GREY_A, 2)

        # Set up matrices
        matrices = VGroup(
            WeightMatrix(
                shape=(6, 8),
                ellipses_row=None,
                ellipses_col=None,
            )
            for n in range(4)
        )
        matrices.arrange_in_grid(v_buff=1, h_buff=1)
        vectors = VGroup(
            NumericEmbedding(length=8, ellipses_row=None)
            for n in range(8)
        )
        vectors.arrange(RIGHT)

        tensors = VGroup(matrices, vectors)
        for group, title in zip(tensors, titles):
            group.set_height(2.5)
            group.next_to(title, DOWN, buff=0.5)

        # Mix up all the numbers
        mat_nums = VGroup(
            elem
            for matrix in matrices
            for elem in matrix.get_entries()
        )
        mat_braces = VGroup(
            brace
            for matrix in matrices
            for brace in matrix.get_brackets()
        )
        vec_nums = VGroup(
            elem
            for vector in vectors
            for elem in vector.get_entries()
        )
        vec_braces = VGroup(
            brace
            for vector in vectors
            for brace in vector.get_brackets()
        )

        def random_point(x_min, x_max, y_min, y_max):
            return np.array([
                random.uniform(x_min, x_max),
                random.uniform(y_min, y_max),
                0
            ])

        all_nums = VGroup(*mat_nums, *vec_nums)
        all_nums.shuffle()
        for num in all_nums:
            states = num.replicate(4)
            for state in states[1:]:
                state.set_height(0.15)
            sign = 1 if num in vec_nums else -1
            states[1].move_to(random_point(6.5 * sign, 1 * sign, 0, 3.5))
            states[2].move_to(random_point(-8, 8, -4, 4))
            states[3].move_to(random_point(-8, 8, -4, 4))
            states[3].set_opacity(0)
            num.states = states
            num.become(states[3])

        self.add(all_nums)

        # Animations
        lag_ratio = 1 / len(all_nums)
        self.play(
            LaggedStart(
                (Transform(num, num.states[2], path_arc=PI)
                for num in all_nums),
                lag_ratio=lag_ratio,
                run_time=3
            ),
        )
        self.wait()
        self.play(
            LaggedStart(
                (LaggedStart(
                    (Transform(num, num.states[1])
                    for num in group),
                    lag_ratio=lag_ratio,
                    run_time=2
                )
                for group in [mat_nums, vec_nums]),
                lag_ratio=0.5
            ),
            ShowCreation(v_line),
        )
        self.play(
            Write(weights_title),
            LaggedStart(
                (Transform(num, num.states[0])
                for num in mat_nums),
                lag_ratio=lag_ratio,
                run_time=2
            ),
            FadeIn(mat_braces, lag_ratio=0.1, time_span=(1, 2)),
        )
        self.play(
            Write(data_title),
            LaggedStart(
                (Transform(num, num.states[0])
                for num in vec_nums),
                lag_ratio=lag_ratio,
                run_time=2
            ),
            FadeIn(vec_braces, lag_ratio=0.1, time_span=(1, 2)),
        )
        self.wait()

        # Add subtitles
        subtitles = VGroup(
            Text("What defines the model", font_size=40),
            Text("What the model processes", font_size=40),
        )
        for subtitle, title, group in zip(subtitles, titles, tensors):
            subtitle.next_to(title, DOWN)
            self.play(
                FadeIn(subtitle, lag_ratio=0.1),
                group.animate.next_to(subtitle, DOWN, buff=0.5),
            )
            self.wait()


class SoftmaxBreakdown(InteractiveScene):
    def construct(self):
        # Show example probability distribution
        word_strs = ['Dumbledore', 'Flitwick', 'Mcgonagall', 'Quirrell', 'Snape', 'Sprout', 'Trelawney']
        words = VGroup(*(Text(word_str, font_size=30) for word_str in word_strs))
        values = np.array([-0.8, -5.0, 0.5, 1.5, 3.4, -2.3, 2.5])
        prob_values = softmax(values)
        chart = BarChart(prob_values, width=10)
        chart.bars.set_stroke(width=1)

        probs = VGroup(*(DecimalNumber(pv) for pv in prob_values))
        probs.arrange(DOWN, buff=0.25)
        probs.generate_target()
        for prob, bar in zip(probs.target, chart.bars):
            prob.scale(0.5)
            prob.next_to(bar, UP)

        for word, bar in zip(words, chart.bars):
            word.scale(0.75)
            height = word.get_height()
            word.move_to(bar.get_bottom(), LEFT)
            word.rotate(-45 * DEGREES, about_point=bar.get_bottom())
            word.shift(height * DOWN)

        chart.save_state()
        for bar in chart.bars:
            bar.stretch(0, 1, about_edge=DOWN)
        chart.set_opacity(0)

        seq_title = Text("Sequence of numbers", font_size=60)
        seq_title.next_to(probs, LEFT, buff=0.75)
        seq_title.set_color(YELLOW)
        prob_title = Text("Probability distribution", font_size=60)
        prob_title.set_color(chart.bars[3].get_color())
        prob_title.center().to_edge(UP)

        self.play(
            LaggedStartMap(FadeIn, probs, shift=0.25 * DOWN, lag_ratio=0.3),
            FadeIn(seq_title),
            run_time=1
        )
        self.wait()
        self.play(
            Restore(chart, lag_ratio=0.1),
            MoveToTarget(probs),
            FadeTransform(seq_title, prob_title),
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeIn, words),
        )
        self.wait()

        # Show constraint between 0 and 1
        index = 3
        bar = chart.bars[index]
        bar.save_state()
        prob = probs[index]
        prob.bar = bar
        max_height = chart.y_axis.get_y(UP) - chart.x_axis.get_y()
        prob.f_always.set_value(lambda: prob.bar.get_height() / max_height)
        prob.always.match_height(probs[1])
        prob.always.next_to(prob.bar, UP)

        one_line = DashedLine(*chart.x_axis.get_start_and_end())
        one_line.set_stroke(RED, 2)
        one_line.align_to(chart.y_axis, UP)

        low_line = one_line.copy()
        low_line.set_stroke(PINK, 5)
        low_line.match_y(chart.x_axis)

        self.play(FadeIn(low_line), FadeIn(one_line), FadeOut(prob_title))
        self.play(low_line.animate.match_y(one_line))
        self.play(FadeOut(low_line))
        self.wait()

        self.play(
            FadeIn(one_line, time_span=(0, 1)),
            bar.animate.set_height(max_height, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(
            bar.animate.set_height(1e-4, about_edge=DOWN, stretch=True),
            run_time=2,
        )
        self.play(Restore(bar))
        self.wait()
        prob.clear_updaters()

        # Show sum
        prob_copies = probs.copy()
        prob_copies.scale(1.5)
        prob_copies.arrange(RIGHT, buff=1.0)
        prob_copies.to_edge(UP)
        prob_copies.shift(LEFT)
        plusses = VGroup(*(
            Tex("+").move_to(VGroup(p1, p2))
            for p1, p2 in zip(prob_copies, prob_copies[1:])
        ))
        equals = Tex("=").next_to(prob_copies, RIGHT)
        rhs = DecimalNumber(1.00)
        rhs.next_to(equals, RIGHT)

        self.play(
            TransformFromCopy(probs, prob_copies),
            Write(plusses),
            Write(equals),
            FadeOut(one_line),
        )
        self.play(
            LaggedStart(*(
                FadeTransform(pc.copy(), rhs)
                for pc in prob_copies
            ), lag_ratio=0.07)
        )
        self.wait()

        sum_group = VGroup(*prob_copies, *plusses, equals, rhs)
        chart_group = VGroup(chart, probs, words)

        # Show example matrix vector output
        n = len(words)
        vector = NumericEmbedding(length=n, ellipses_row=None)
        in_values = np.array([e.get_value() for e in vector.elements])
        rows = []
        for value in values:
            row = np.random.uniform(-1, 1, len(in_values))
            row *= value / np.dot(row, in_values)
            rows.append(row)
        matrix_values = np.array(rows)

        matrix = WeightMatrix(
            values=matrix_values,
            ellipses_row=None,
            ellipses_col=None,
            num_decimal_places=2,
        )
        for mob in matrix, vector:
            mob.set_height(4)
        vector.to_edge(UP).set_x(2.5)
        matrix.next_to(vector, LEFT)

        self.play(LaggedStart(
            chart_group.animate.scale(0.35).to_corner(DL),
            FadeOut(sum_group, UP),
            FadeIn(matrix, UP),
            FadeIn(vector, UP),
        ))
        eq, rhs = show_matrix_vector_product(self, matrix, vector, x_max=9)
        self.wait()

        # Comment on output
        rhs_rect = SurroundingRectangle(rhs)
        rhs_words = Text("Not at all a\\nprobability distribution!")
        rhs_words.next_to(rhs_rect, DOWN)

        neg_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() < 0
        ))
        gt1_rects = VGroup(*(
            SurroundingRectangle(entry)
            for entry in rhs.get_entries()
            if entry.get_value() > 1
        ))
        VGroup(rhs_rect, neg_rects).set_stroke(RED, 4)
        gt1_rects.set_stroke(BLUE, 4)

        for rect in (*neg_rects, *gt1_rects):
            neg = rect in neg_rects
            rect.word = Text("Negative" if neg else "> 1", font_size=36)
            rect.word.match_color(rect)
            rect.word.next_to(rhs, RIGHT)
            rect.word.match_y(rect)
        neg_words = VGroup(*(r.word for r in neg_rects))
        gt1_words = VGroup(*(r.word for r in gt1_rects))

        sum_arrow = Vector(DOWN).next_to(rhs, DOWN)
        sum_sym = Tex(R"\\sum", font_size=36).next_to(sum_arrow, LEFT)
        sum_num = DecimalNumber(sum(e.get_value() for e in rhs.get_entries()))
        sum_num.next_to(sum_arrow, DOWN)

        self.play(
            ShowCreation(rhs_rect),
            FadeIn(rhs_words),
        )
        self.wait()
        self.play(
            ReplacementTransform(VGroup(rhs_rect), neg_rects),
            LaggedStart(*(FadeIn(rect.word, 0.5 * RIGHT) for rect in neg_rects)),
        )
        self.wait()
        self.play(
            ReplacementTransform(neg_rects, gt1_rects),
            FadeTransformPieces(neg_words, gt1_words),
        )
        self.wait()
        self.play(
            LaggedStart(
                FadeOut(rhs_words),
                FadeOut(gt1_rects),
                FadeOut(gt1_words),
            ),
            GrowArrow(sum_arrow),
            FadeIn(sum_num, DOWN),
            FadeIn(sum_sym),
        )
        self.wait()
        self.play(*map(FadeOut, [sum_arrow, sum_sym, sum_num]))

        # Preview softmax application
        rhs.generate_target()
        rhs.target.to_edge(LEFT, buff=1.5)
        rhs.target.set_y(0)

        softmax_box = Rectangle(width=5, height=6.5)
        softmax_box.set_stroke(BLUE, 2)
        softmax_box.set_fill(BLUE_E, 0.5)
        in_arrow, out_arrow = Vector(RIGHT).replicate(2)
        in_arrow.next_to(rhs.target, RIGHT)
        softmax_box.next_to(in_arrow, RIGHT)
        out_arrow.next_to(softmax_box, RIGHT)

        softmax_label = Text("softmax", font_size=60)
        softmax_label.move_to(softmax_box)

        rhs_values = np.array([e.get_value() for e in rhs.get_entries()])
        dist = softmax(rhs_values)
        output = DecimalMatrix(dist.reshape((dist.shape[0], 1)))
        output.match_height(rhs)
        output.next_to(out_arrow, RIGHT)

        bars = chart.bars.copy()
        for bar, entry in zip(bars, output.get_entries()):
            bar.rotate(-PI / 2)
            bar.stretch(2, 0)
            bar.next_to(output)
            bar.match_y(entry)

        self.play(LaggedStart(
            FadeOut(matrix, 2 * LEFT),
            FadeOut(vector, 3 * LEFT),
            FadeOut(eq, 3.5 * LEFT),
            FadeOut(chart_group, DL),
            GrowArrow(in_arrow),
            FadeIn(softmax_box, RIGHT),
            FadeIn(softmax_label, RIGHT),
            MoveToTarget(rhs),
            GrowArrow(out_arrow),
            FadeIn(output, RIGHT),
            TransformFromCopy(chart.bars, bars),
        ), lag_ratio=0.2, run_time=2)
        self.wait()

        # Highlight larger and smaller parts
        rhs_entries = rhs.get_entries()
        changer = VGroup(rhs_entries, output.get_entries(), bars)
        changer.save_state()
        for index in range(4, 0, -1):
            changer.target = changer.saved_state.copy()
            changer.target.set_fill(border_width=0)
            for group in changer.target:
                for j, elem in enumerate(group):
                    if j != index:
                        elem.fade(0.8)
            self.play(MoveToTarget(changer))
            self.wait()
        self.play(Restore(changer))
        self.remove(changer)
        self.add(rhs, output, bars)
        self.wait()

        # Swap out for variables
        variables = VGroup(*(
            Tex(f"x_{{{n}}}", font_size=48).move_to(elem)
            for n, elem in enumerate(rhs_entries, start=1)
        ))

        self.remove(rhs_entries)
        self.play(
            LaggedStart(*(
                TransformFromCopy(entry, variable, path_arc=PI / 2)
                for entry, variable in zip(rhs_entries, variables)
            ), lag_ratio=0.1, run_time=1.0)
        )
        self.wait()

        # Exponentiate each part
        exp_parts = VGroup(*(
            Tex(f"e^{{{var.get_tex()}}}", font_size=48).move_to(var)
            for var in variables
        ))
        exp_parts.align_to(softmax_box, LEFT)
        exp_parts.shift(0.75 * RIGHT)
        exp_parts.space_out_submobjects(1.5)
        gt0s = VGroup(
            Tex(R"> 0").next_to(exp_part, aligned_edge=DOWN)
            for exp_part in exp_parts
        )

        self.play(
            softmax_label.animate.next_to(softmax_box, UP, buff=0.15),
            LaggedStart(*(
                TransformMatchingStrings(var.copy(), exp_part)
                for var, exp_part in zip(variables, exp_parts)
            ), run_time=1, lag_ratio=0.01)
        )
        self.play(LaggedStartMap(FadeIn, gt0s, shift=0.5 * RIGHT, lag_ratio=0.25, run_time=1))
        self.wait()
        self.play(FadeOut(gt0s))

        # Compute the sum
        exp_sum = Tex(R"\\sum_{n=0}^{N-1} e^{x_{n}}", font_size=42)
        exp_sum[R"e^{x_{n}}"].scale(1.5, about_edge=LEFT)
        exp_sum.next_to(softmax_box.get_right(), LEFT, buff=0.75)

        lines = VGroup(*(Line(exp_part.get_right(), exp_sum.get_left(), buff=0.1) for exp_part in exp_parts))
        lines.set_stroke(TEAL, 2)

        self.play(
            LaggedStart(*(
                FadeTransform(exp_part.copy(), exp_sum)
                for exp_part in exp_parts
            ), lag_ratio=0.01),
            LaggedStartMap(ShowCreation, lines, lag_ratio=0.01),
            run_time=1
        )
        self.wait()
        self.play(FadeOut(lines))

        # Divide each part by the sum
        lil_denoms = VGroup()
        for exp_part in exp_parts:
            slash = Tex("/").match_height(exp_sum)
            slash.next_to(exp_sum, LEFT, buff=0)
            denom = VGroup(slash, exp_sum).copy()
            denom.set_height(exp_part.get_height() * 1.5)
            denom.next_to(exp_part, RIGHT, buff=0)
            lil_denoms.add(denom)
        lil_denoms.align_to(softmax_box.get_center(), LEFT)

        lines = VGroup(*(Line(exp_sum.get_left(), denom.get_center()) for denom in lil_denoms))
        lines.set_stroke(TEAL, 1)

        self.remove(exp_sum)
        self.play(
            exp_parts.animate.next_to(lil_denoms, LEFT, buff=0),
            LaggedStart(*(
                FadeTransform(exp_sum.copy(), denom)
                for denom in lil_denoms
            ), lag_ratio=0.01),
        )
        self.wait()

        # Resize box
        sm_terms = VGroup(*(
            VGroup(exp_part, denom)
            for exp_part, denom in zip(exp_parts, lil_denoms)
        ))
        sm_terms.generate_target()

        target_height = 5.0
        full_output = Group(output, bars)
        full_output.generate_target()
        full_output.target.set_height(target_height, about_edge=RIGHT)
        full_output.target.shift(1.5 * LEFT)
        equals = Tex("=")
        equals.next_to(full_output.target, LEFT)

        softmax_box.generate_target()
        softmax_box.target.set_width(3.0, stretch=True)
        VGroup(softmax_box.target, sm_terms.target).set_height(target_height + 0.5).next_to(equals, LEFT)

        rhs.generate_target()
        rhs_entries.become(variables)
        self.remove(variables)
        rhs.target.set_height(target_height)
        rhs.target.next_to(softmax_box.target, LEFT, buff=1.5)

        self.play(
            softmax_label.animate.next_to(softmax_box.target, UP),
            MoveToTarget(softmax_box),
            MoveToTarget(sm_terms),
            MoveToTarget(full_output),
            MoveToTarget(rhs),
            FadeTransform(out_arrow, equals),
            in_arrow.animate.become(
                Arrow(rhs.target, softmax_box.target).match_style(in_arrow)
            ),
        )
        self.wait()

        # Set up updaters
        output_entries = output.get_entries()
        bar_width_ratio = bars.get_width() / max(o.get_value() for o in output_entries)
        temp_tracker = ValueTracker(1)

        def update_outs(output_entries):
            inputs = [entry.get_value() for entry in rhs_entries]
            outputs = softmax(inputs, temp_tracker.get_value())
            for entry, output in zip(output_entries, outputs):
                entry.set_value(output)

        def update_bars(bars):
            for bar, entry in zip(bars, output_entries):
                width = max(bar_width_ratio * entry.get_value(), 1e-3)
                bar.set_width(width, about_edge=LEFT, stretch=True)

        output_entries.clear_updaters().save_state()
        bars.clear_updaters().save_state()
        output_entries.add_updater(update_outs)
        bars.add_updater(update_bars)

        self.add(bars, output_entries)

        # Tweak values
        index_value_pairs = [
            (6, 4.0),
            (4, 4.2),
            (2, 4.0),
            (0, 6.0),
            (4, 9.9)
        ]
        # index_value_pairs = [  # For emphasizing a max
        #     (3, 8.5),
        #     (6, 8.0),
        #     (2, 8.1),
        #     (0, 9.0),
        # ]
        for index, value in index_value_pairs:
            entry = rhs_entries[index]
            rect = SurroundingRectangle(entry)
            rect.set_stroke(BLUE if value > entry.get_value() else RED, 3)
            self.play(
                ChangeDecimalToValue(entry, value),
                FadeIn(rect, time_span=(0, 1)),
                run_time=4
            )
            self.play(FadeOut(rect))

        # Add temperature
        frame = self.frame
        temp_color = RED
        new_title = Text("softmax with temperature")
        new_title["temperature"].set_color(temp_color)
        get_t = temp_tracker.get_value
        t_line = NumberLine(
            (0, 10, 0.2),
            tick_size=0.025,
            big_tick_spacing=1,
            longer_tick_multiple=2.0,
            width=4
        )
        t_line.set_stroke(width=1.5)
        t_line.next_to(softmax_box, UP)
        t_tri = ArrowTip(angle=-90 * DEGREES)
        t_tri.set_color(temp_color)
        t_tri.set_height(0.2)
        t_label = Tex("T = 0.00", font_size=36)
        t_label.rhs = t_label.make_number_changeable("0.00")
        t_label["T"].set_color(temp_color)
        t_tri.add_updater(lambda m: m.move_to(t_line.n2p(get_t()), DOWN))
        t_label.add_updater(lambda m: m.rhs.set_value(get_t()))
        t_label.add_updater(lambda m: m.next_to(t_tri, UP, buff=0.1, aligned_edge=LEFT))
        t_label.update()

        new_title.next_to(t_label, UP, buff=0.5).match_x(softmax_box)

        self.play(
            frame.animate.move_to(0.75 * UP),
            TransformMatchingStrings(softmax_label, new_title),
            FadeIn(t_line),
            FadeIn(t_tri),
            FadeIn(t_label),
            run_time=1
        )

        # Change formula
        template = Tex(R"e^{x_{0} / T} / \\sum_{n=0}^{N - 1} e^{x_n / T}")
        template["T"].set_color(temp_color)
        template["/"][1].scale(1.9, about_edge=LEFT)
        template[R"\\sum_{n=0}^{N - 1}"][0].scale(0.7, about_edge=RIGHT)
        index_part = template.make_number_changeable("0")

        new_sm_terms = VGroup()
        all_Ts = VGroup()
        for n, term in enumerate(sm_terms, start=1):
            template.replace(term, dim_to_match=1)
            index_part.set_value(n)
            new_term = template.copy()
            all_Ts.add(*new_term["T"])
            new_sm_terms.add(new_term)

        self.play(
            LaggedStart(*(
                FadeTransform(old_term, new_term)
                for old_term, new_term in zip(sm_terms, new_sm_terms)
            )),
            LaggedStart(*(
                TransformFromCopy(t_label[0], t_mob[0])
                for t_mob in all_Ts
            )),
        )
        self.wait()

        # Oscilate between values
        for value in [4, 10, 2]:
            self.play(temp_tracker.animate.set_value(value), run_time=8)
            self.wait()
        self.play(temp_tracker.animate.set_value(0), run_time=3)
        max_rects = VGroup(
            SurroundingRectangle(rhs.get_entries()[4]),
            SurroundingRectangle(VGroup(output.get_entries()[4], bars[4])),
        )
        self.play(LaggedStartMap(ShowCreationThenFadeOut, max_rects))
        self.wait()
        for value in [5, 1, 7]:
            self.play(temp_tracker.animate.set_value(value), run_time=4)
            self.wait()

        # Describe logits
        prob_arrows, logit_arrows = (
            VGroup(*(
                Vector(-vect).next_to(entry, vect, buff=0.25)
                for entry in matrix.get_entries()
            ))
            for matrix, vect in [(output, RIGHT), (rhs, LEFT)]
        )
        prob_arrows.next_to(bars, RIGHT)
        prob_rects = VGroup(*map(SurroundingRectangle, output.get_entries()))
        logit_rects = VGroup(*map(SurroundingRectangle, rhs.get_entries()))
        VGroup(prob_rects, logit_rects).set_stroke(width=1)

        prob_words = Text("Probabilities")
        prob_words.next_to(output, UP, buff=0.25)
        logit_words = Text("Logits")
        logit_words.next_to(rhs, UP, buff=0.25)

        logit_group = VGroup(logit_arrows, logit_words, logit_rects)
        logit_group.set_color(TEAL)
        prob_group = VGroup(prob_arrows, prob_words, prob_rects)
        prob_group.set_color(YELLOW)

        for arrows, word, rects in [prob_group, logit_group]:
            self.play(
                t_line.animate.set_y(3.35),
                Write(word),
                Write(rects, stroke_width=5, stroke_color=rects[0].get_stroke_color(), lag_ratio=0.3, run_time=3),
            )
            self.wait()


class CostFunction(InteractiveScene):
    def construct(self):
        # Add graph
        axes = Axes((0, 1, 0.1), (0, 5, 1), width=10, height=6)
        axes.center().to_edge(LEFT)
        axes.x_axis.add_numbers(num_decimal_places=1)
        axes.y_axis.add_numbers(num_decimal_places=0, direction=LEFT)
        x_label = Tex("p")
        x_label.next_to(axes.x_axis.get_right(), UR)
        axes.add(x_label)

        graph = axes.get_graph(lambda x: -np.log(x), x_range=(0.001, 10, 0.01))
        graph.set_color(RED)

        expr = Tex(R"\\text{Cost} = -\\log(p)", font_size=60)
        expr.next_to(axes.i2gp(0.1, graph), UR, buff=0.1)

        self.add(axes, graph, expr)

        # Add sample phrase
        phrase = Text("Watching 3Blue1Brown makes you smarter")
        phrase.scale(0.75)
        phrase.to_edge(UP)
        phrase.align_to(axes.c2p(0.1, 0), LEFT)
        pieces = break_into_tokens(phrase)
        pieces[-1].set_opacity(0.0)
        rects = get_piece_rectangles(pieces, leading_spaces=True, h_buff=0)

        self.add(rects, pieces)

        # Add predictions
        arrow = Vector(0.5 * DOWN)
        arrow.next_to(rects[-1], DOWN, SMALL_BUFF)
        index = 0

        tokens, probs = gpt3_predict_next_token(phrase.get_text()[:-len(" smarter")])
        bar_chart = next_token_bar_chart(
            tokens[:8], probs[:8],
            width_100p=7.0,
            bar_space_factor=1.0,
            use_percent=False,
        )
        bar_chart.next_to(arrow, DOWN)
        bar_chart.shift(1.25 * RIGHT)
        bar_chart.set_opacity(0.5)
        bar_chart[index].set_opacity(1.0)
        rect = SurroundingRectangle(bar_chart[index])

        self.add(arrow, bar_chart, rect)

        # Animate in graph
        self.play(
            ShowCreation(graph, run_time=3),
            Write(expr, run_time=2),
        )
        self.wait()

        # Show point on the graph
        line = axes.get_line_from_axis_to_point(0, axes.i2gp(probs[index], graph), line_func=Line)
        line.set_stroke(YELLOW)

        self.play(FadeTransform(rect.copy(), line))
        self.wait()`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      3: "Imports * from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      4: "Imports * from the _2024.transformers.generation module within the 3b1b videos codebase.",
      7: "DialTest extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      8: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      20: "MLWithinDeepL extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      21: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      31: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      48: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      54: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      56: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      57: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      58: "Restore animates a mobject back to a previously saved state (from save_state()).",
      59: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      61: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      90: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      91: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      93: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      95: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      100: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      103: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      109: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      111: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      114: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      116: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      117: "Restore animates a mobject back to a previously saved state (from save_state()).",
      118: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      120: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      121: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      122: "FadeOut transitions a mobject from opaque to transparent.",
      123: "FadeOut transitions a mobject from opaque to transparent.",
      125: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      149: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      156: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      162: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      163: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      164: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      165: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      168: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      169: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      170: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      176: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      177: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      183: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      186: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      209: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      210: "FadeOut transitions a mobject from opaque to transparent.",
      211: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      213: "FadeOut transitions a mobject from opaque to transparent.",
      216: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      217: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      218: "Succession plays animations one after another in sequence.",
      219: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      225: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      227: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      230: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      235: "Interpolates between colors in HSL space for perceptually uniform gradients.",
      298: "ShowCross extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      299: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      307: "FlashThroughImageData extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      310: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      330: "FlashThroughTextData2 extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      337: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      359: "TweakedMachine extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      363: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      392: "PremiseOfML extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      398: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      975: "Class PremiseOfMLWithText inherits from PremiseOfML.",
      1029: "Class PremiseOfMLWithMatrices inherits from PremiseOfML.",
      1036: "LinearRegression extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1039: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1174: "ShowGPT3Numbers extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1175: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1666: "DistinguishWeightsAndData extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1667: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1814: "SoftmaxBreakdown extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1815: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2368: "CostFunction extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2369: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/mlp.py"] = {
    description: "Multi-layer perceptron (feed-forward network) scenes. Visualizes the MLP layers within a transformer block: linear transformations, ReLU/GELU activations, and how neurons learn features.",
    code: `import torch
from scipy.stats import norm

from _2024.transformers.helpers import *
from manim_imports_ext import *


class LastTwoChapters(InteractiveScene):
    def construct(self):
        # Show last two chapters
        frame = self.frame
        self.camera.light_source.set_z(15)
        self.set_floor_plane("xz")

        thumbnails = self.get_thumbnails()
        self.play(
            LaggedStartMap(FadeIn, thumbnails, shift=UP, lag_ratio=0.5)
        )
        self.wait()

        # Show transformer schematic
        blocks = Group(self.get_block() for x in range(10))
        blocks[1::2].stretch(2, 2).set_opacity(1)

        blocks.arrange(OUT, buff=0.5)
        blocks.set_depth(8, stretch=True)
        blocks.set_opacity(0.8)
        blocks.apply_depth_test()

        trans_title = Text("Transformer", font_size=96)
        trans_title.next_to(blocks, UP, buff=0.5)

        self.play(
            frame.animate.reorient(-32, 0, 0, (0.56, 2.48, 0.32), 12.75),
            thumbnails.animate.scale(0.5).arrange(RIGHT, buff=2.0).to_edge(UP, buff=0.25),
            LaggedStartMap(FadeIn, blocks, shift=0.25 * UP, scale=1.5, lag_ratio=0.1),
            FadeIn(trans_title, UP),
        )
        self.wait()

        # Break out transformer as sequence of blocks
        att_blocks = blocks[0::2]
        mlp_blocks = blocks[1::2]

        att_title = Text("Attention", font_size=72)
        mlp_title_full = Text("Multilayer Perceptron", font_size=72)
        mlp_title = Text("MLP", font_size=72)

        self.play(
            frame.animate.reorient(-3, -2, 0, (0.23, 2.57, 0.3), 12.75),
            trans_title.animate.shift(2 * UP),
            att_blocks.animate.shift(4 * LEFT),
            mlp_blocks.animate.shift(4 * RIGHT),
        )

        att_icon = self.get_att_icon(att_blocks[-1])
        mlp_icon = self.get_mlp_icon(mlp_blocks[-1])
        att_title.next_to(att_blocks[-1], UP, buff=0.75)
        for title in [mlp_title, mlp_title_full]:
            title.next_to(mlp_blocks[-1], UP, buff=0.75)
        self.play(
            FadeIn(att_icon, lag_ratio=1e-3),
            FadeIn(att_title, UP),
            trans_title.animate.scale(0.75).set_opacity(0.5)
        )
        self.wait()
        self.play(
            Write(mlp_icon),
            FadeIn(mlp_title_full, UP),
        )
        self.wait()
        self.play(
            TransformMatchingStrings(mlp_title_full, mlp_title)
        )
        self.wait()

        # Show sports facts
        sport_facts = VGroup(
            Text(line)
            for line in Path(DATA_DIR, "athlete_sports.txt").read_text().split("\\n")
        )
        for fact in sport_facts:
            fact.next_to(trans_title, UP)
            fact.shift(random.uniform(-3, 3) * RIGHT)
            fact.shift(random.uniform(0, 3) * UP)

        self.remove(mlp_icon, mlp_title)
        self.play(
            FadeOut(thumbnails),
            FadeOut(trans_title),
            LaggedStart(
                (Succession(FadeIn(fact), fact.animate.scale(0.5).set_opacity(0).move_to(mlp_blocks))
                for fact in sport_facts),
                lag_ratio=0.15,
            )
        )
        self.wait()

        # Ask what is the MLP
        rect = SurroundingRectangle(Group(mlp_blocks, mlp_title), buff=1.0)
        rect.stretch(0.8, 1)
        rect.match_z(mlp_blocks[-1])
        question = Text("What are these?", font_size=90)
        question.next_to(rect, UP, buff=3.0)
        question.match_color(rect)
        question.set_fill(border_width=0.5)
        arrow = Arrow(question, rect)
        arrow.match_color(rect)

        self.play(
            Group(att_blocks, att_title).animate.fade(0.5),
            ShowCreation(rect),
            Write(question),
            GrowArrow(arrow),
        )
        self.wait()

    def get_thumbnails(self):
        folder = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails"
        images = [
            ImageMobject(str(Path(folder, "Chapter5_TN5"))),
            ImageMobject(str(Path(folder, "Chapter6_TN4"))),
        ]
        thumbnails = Group(
            Group(
                SurroundingRectangle(image, buff=0).set_stroke(WHITE, 3),
                image
            )
            for n, image in zip([5, 6], images)
        )
        thumbnails.set_height(3.5)
        thumbnails.arrange(RIGHT, buff=1.0)
        thumbnails.fix_in_frame()
        return thumbnails

    def get_att_icon(self, block, n_rows=8):
        att_icon = Dot().get_grid(n_rows, n_rows)
        att_icon.set_height(block.get_height() * 0.9)
        att_icon.set_backstroke(BLACK, 0.5)
        for dot in att_icon:
            dot.set_fill(opacity=random.random()**5)
        att_icon.move_to(block, OUT)
        return att_icon

    def get_mlp_icon(self, block, dot_buff=0.15, layer_buff=1.5, layer0_size=5):
        layers = VGroup(
            Dot().get_grid(layer0_size, 1, buff=dot_buff),
            Dot().get_grid(2 * layer0_size, 1, buff=dot_buff),
            Dot().get_grid(layer0_size, 1, buff=dot_buff),
        )
        layers.set_height(block.get_height() * 0.9)
        layers.arrange(RIGHT, buff=layer_buff)
        for layer in layers:
            for dot in layer:
                dot.set_fill(opacity=random.random())
        layers.set_stroke(WHITE, 0.5)
        lines = VGroup(
            Line(dot1.get_center(), dot2.get_center(), buff=dot1.get_width() / 2)
            for l1, l2 in zip(layers, layers[1:])
            for dot1 in l1
            for dot2 in l2
        )
        for line in lines:
            line.set_stroke(
                color=value_to_color(random.uniform(-10, 10)),
                width=3 * random.random()**3
            )

        icon = VGroup(layers, lines)
        icon.move_to(block, OUT)
        return icon

    def get_block(self, width=5, height=3, depth=1, color=GREY_D, opacity=0.8):
        block = Cube(color=color, opacity=opacity)
        block.deactivate_depth_test()
        block.set_shape(width, height, depth)
        block.set_shading(0.5, 0.5, 0.0)
        block.sort(lambda p: np.dot(p, [-1, 1, 1]))
        return block


class AltLastTwoChapters(LastTwoChapters):
    def construct(self):
        # Show last two chapters
        thumbnails = self.get_thumbnails()
        thumbnails.set_height(2.0)
        thumbnails.arrange(RIGHT, buff=2.0)
        thumbnails.to_edge(UP)
        for n, thumbnail in zip([5, 6], thumbnails):
            label = Text(f"Chapter {n}")
            label.next_to(thumbnail, DOWN, SMALL_BUFF)
            thumbnail.add(label)

        self.play(
            LaggedStartMap(FadeIn, thumbnails, shift=UP, lag_ratio=0.5)
        )
        self.wait()

        # Focus on chapter 6
        for thumbnail in thumbnails:
            thumbnail.target = thumbnail.generate_target()
            thumbnail.target.scale(1.25)
            thumbnail.target[-1].scale(1.0 / 1.5).next_to(thumbnail.target[0], DOWN, SMALL_BUFF)
        thumbnails[1].target.set_x(-2.85)
        thumbnails[1].target.to_edge(UP, MED_SMALL_BUFF)
        thumbnails[0].target.next_to(thumbnails[1].target, LEFT, buff=2.5)

        self.play(
            LaggedStartMap(MoveToTarget, thumbnails)
        )
        self.wait()


class MLPIcon(LastTwoChapters):
    def construct(self):
        # Add network
        network = self.get_mlp_icon(Square(6), layer_buff=3.0, layer0_size=6)
        self.play(Write(network, stroke_width=0.5, lag_ratio=1e-2, run_time=5))
        self.wait()

        # Propagate through
        thick_layers = VGroup(network[1].family_members_with_points()).copy()
        for line in thick_layers:
            line.set_stroke(width=2 * line.get_width())
            line.insert_n_curves(20)
        self.play(LaggedStartMap(VShowPassingFlash, thick_layers, time_width=1.5, lag_ratio=5e-3, run_time=3))
        self.wait()


class MLPStepsPreview(InteractiveScene):
    def construct(self):
        # Setup framing
        background = FullScreenRectangle()
        top_frame, low_frame = frames = Rectangle(7, 3.25).replicate(2)
        frames.arrange(DOWN, buff=0.5)
        frames.to_edge(LEFT)
        frames.set_fill(BLACK, 1)
        frames.set_stroke(WHITE, 2)

        titles = VGroup(
            VGroup(Text("Structure:"), Text("Easy")),
            VGroup(Text("Emergent behavior:"), Text("Exceedingly challenging")),
        )
        for title, frame, color in zip(titles, frames, [GREEN, RED]):
            title.scale(2)
            for part in title:
                part.set_max_width(6)
            title.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
            title.next_to(frame, RIGHT, buff=0.5)
            title[1].set_color(color)

        titles[0].save_state()
        top_frame.save_state()
        top_frame.set_shape(8, 6).center().to_edge(LEFT)
        titles[0].next_to(top_frame, RIGHT, buff=0.5)

        self.add(background)
        self.add(top_frame)
        self.add(titles[0][0])

        # Add all steps
        arrows = Vector(2.2 * RIGHT).get_grid(1, 3, buff=0.25)
        arrows.move_to(top_frame)
        up_proj = WeightMatrix(shape=(10, 6))
        down_proj = WeightMatrix(shape=(6, 10))
        VGroup(up_proj, down_proj).match_width(arrows[0])
        up_proj.next_to(arrows[0], UP, buff=MED_SMALL_BUFF)
        down_proj.next_to(arrows[2], UP, buff=MED_SMALL_BUFF)

        axes = Axes((-4, 4), (0, 4))
        graph = axes.get_graph(lambda x: max(0, x))
        graph.set_stroke(YELLOW, 5)
        plot = VGroup(axes, graph)
        plot.set_width(arrows[0].get_width() * 0.75)
        plot.next_to(arrows[1], UP, buff=MED_SMALL_BUFF)

        labels = VGroup(*map(Text, ["Linear", "ReLU", "Linear"]))
        for label, arrow in zip(labels, arrows):
            label.next_to(arrow, DOWN)

        structure = VGroup(arrows, labels, VGroup(up_proj, plot, down_proj))

        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5),
            LaggedStartMap(FadeIn, labels, shift=0.5 * RIGHT, lag_ratio=0.5),
            Write(titles[0][1])
        )
        self.play(LaggedStart(
            FadeIn(up_proj, shift=0.5 * UP),
            FadeIn(down_proj, shift=0.5 * UP),
            lag_ratio=0.5
        ))
        self.play(FadeIn(plot, lag_ratio=1e-2))
        self.wait(3)

        # Reference emergent structure

        self.play(
            Restore(top_frame),
            Restore(titles[0]),
            structure.animate.set_width(0.9 * top_frame.saved_state.get_width()).move_to(top_frame.saved_state),
            FadeIn(low_frame, DOWN),
            FadeIn(titles[1][0], DOWN),
        )
        self.play(
            Write(titles[1][1], stroke_color=RED)
        )

        # Data flying
        kw = dict(font_size=16, shift_vect=0.5 * DOWN + 0.5 * RIGHT, word_shape=(5, 5))
        data_modifying_matrix(self, up_proj, **kw)
        data_modifying_matrix(self, down_proj, **kw)
        self.wait()

        # Swap out for toy example
        toy_example_title = Text("Motivating Toy Example", font_size=54)
        toy_example_title.next_to(titles[1][0], DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)
        strike = Line().replace(titles[1][0])
        strike.set_stroke(RED, 8)

        low_matrices = VGroup(up_proj, down_proj)
        top_matrices = low_matrices.copy()
        low_matrices.generate_target()
        low_matrices.target.scale(1.75).arrange(RIGHT, buff=0.5)
        low_matrices.target.move_to(low_frame, DOWN).shift(MED_SMALL_BUFF * UP)

        self.play(
            ShowCreation(strike),
            FadeOut(titles[1][1]),
            titles[1][0].animate.set_opacity(0.5)
        )
        self.add(top_matrices)
        self.play(
            MoveToTarget(low_matrices),
            FadeIn(toy_example_title, DOWN)
        )
        self.wait()

        # Write down fact
        row_rect = SurroundingRectangle(low_matrices[0].get_rows()[0], buff=0.1)
        col_rect = SurroundingRectangle(low_matrices[1].get_columns()[0], buff=0.1)
        VGroup(row_rect, col_rect).set_stroke(WHITE, 1)
        fact = Text("Michael Jordan plays Basketball", font_size=36)
        fact.next_to(frames[1].get_top(), DOWN)
        fact.align_to(low_matrices, LEFT)
        mj, bb = fact["Michael Jordan"], fact["plays Basketball"]
        mj_brace = Brace(mj, DOWN, buff=0.1)
        bb_brace = Brace(bb, DOWN).match_y(mj_brace)
        mj_arrow = Arrow(row_rect, mj_brace, buff=0.05)
        bb_arrow = Arrow(col_rect.get_top(), bb_brace, buff=0.05)

        row_cover = BackgroundRectangle(low_matrices[0].get_rows()[1:], buff=0.05)
        col_cover = BackgroundRectangle(low_matrices[1].get_columns()[1:], buff=0.05)
        VGroup(row_cover, col_cover).set_fill(BLACK, 0.75)

        self.play(LaggedStart(
            FadeIn(row_cover),
            FadeIn(row_rect),
            GrowFromCenter(mj_brace),
            FadeIn(mj, 0.5 * UP)
        ))
        self.play(
            FadeIn(col_cover),
            FadeIn(col_rect),
            GrowArrow(bb_arrow),
            GrowFromCenter(bb_brace),
            FadeIn(bb, 0.5 * UP)
        )
        self.add(*low_matrices, row_cover, col_cover, row_rect, col_rect)
        self.play(
            RandomizeMatrixEntries(low_matrices[0]),
            RandomizeMatrixEntries(low_matrices[1]),
        )
        self.wait()


class MatricesVsIntuition(InteractiveScene):
    def construct(self):
        # Add matrix
        matrix = WeightMatrix(shape=(15, 15))
        matrix.set_height(4)
        matrix.to_edge(LEFT)

        Text("Matrices filled with parameters\\nlearned during gradient descent")
        Text("Motivating examples which risk being\\noversimplifications of what true models do")

        self.add(matrix)


class BasicMLPWalkThrough(InteractiveScene):
    random_seed = 1

    def construct(self):
        # Init camera settings
        self.set_floor_plane("xz")
        frame = self.frame
        self.camera.light_source.set_z(15)

        # Sequence of embeddings comes in to an MLP block
        embedding_array = EmbeddingArray(shape=(6, 9))
        embedding_array.set_width(10)

        block = VCube(fill_color=GREY_D, fill_opacity=0.5)
        block.sort(lambda p: p[2])
        block[-1].set_fill(opacity=0)
        block.set_stroke(GREY_B, 2, 0.25, behind=False)
        block.set_shading(0.25, 0.25, 0.5)
        block.set_shape(11, 4, 4)
        block.move_to(0.5 * IN, IN)
        block_title = Text("MLP", font_size=90)
        block_title.next_to(block, UP)

        frame.reorient(-21, -12, 0, (0.34, -0.94, -0.18), 9.79)
        frame.set_field_of_view(30 * DEGREES)
        self.add(block, block_title)
        self.play(FadeIn(embedding_array, shift=2 * OUT))
        self.wait()

        # Highlight one vector
        index = 3
        emb = embedding_array.embeddings[index]
        highlight_rect = SurroundingRectangle(emb)
        embedding_array.target = embedding_array.generate_target()
        embedding_array.target.set_stroke(width=0)
        embedding_array.target.set_opacity(0.5)
        embedding_array.target[0][index].set_backstroke(BLACK, 2)
        embedding_array.target[0][index].set_opacity(1)

        self.play(
            MoveToTarget(embedding_array),
            ShowCreation(highlight_rect),
        )
        self.wait()

        # Reorient
        rot_about_up = 89 * DEGREES
        rot_about_left = 1 * DEGREES
        up_emb = emb.copy()  # For use down below
        full_block = Group(block, embedding_array, highlight_rect, block_title)
        full_block.target = full_block.generate_target()
        full_block.target[0].set_depth(16, about_edge=IN, stretch=True)
        full_block.target[0].set_height(5, about_edge=DOWN, stretch=True)
        full_block.target.rotate(rot_about_up, UP)
        full_block.target[:3].rotate(rot_about_left, LEFT)
        full_block.target.scale(0.5)
        full_block.target[3].rotate(90 * DEGREES, DOWN).next_to(full_block.target[0], UP, buff=0.5)
        full_block.target.center().to_edge(DOWN, buff=0.75)
        full_block.target[0][4].set_opacity(0.1)

        self.play(
            frame.animate.reorient(-3, -2, 0, (-0.0, -2.0, 0.01), 6.48),
            MoveToTarget(full_block),
            run_time=2
        )

        # Preview the sequence of operations
        values = np.random.uniform(-10, 10, 9)
        values[0] = 1.0
        vects = VGroup(
            NumericEmbedding(values=values, dark_color=GREY_B),
            NumericEmbedding(values=np.clip(values, 0, np.inf), dark_color=GREY_B),
            NumericEmbedding(length=6),
        )
        vects.set_width(emb.get_depth())
        vects.arrange(RIGHT, buff=2.0)
        vects.next_to(emb, RIGHT, buff=2.0)

        arrows = VGroup(
            Arrow(v1, v2)
            for v1, v2 in zip([emb, *vects[:-1]], vects)
        )
        arrow_labels = VGroup(Text("Linear"), Text("ReLU"), Text("Linear"))
        arrow_labels.scale(0.5)

        phases = VGroup()
        simple_phases = VGroup()
        for arrow, label, vect in zip(arrows, arrow_labels, vects):
            label.next_to(arrow, UP)
            phases.add(VGroup(arrow, label, vect))
            simple_phases.add(VGroup(arrow, vect))

        self.play(
            LaggedStartMap(FadeIn, vects, shift=RIGHT, lag_ratio=0.8),
            LaggedStartMap(ShowCreation, arrows, lag_ratio=0.8),
            LaggedStartMap(FadeIn, arrow_labels, lag_ratio=0.8),
        )
        self.wait()

        # Show the sum
        sum_circuit, output_emb = self.get_sum_circuit(emb, vects[-1])

        self.play(
            frame.animate.reorient(15, -4, 0, (0.82, -1.91, 0.04), 7.18),
            ShowCreation(sum_circuit, lag_ratio=0.1),
            run_time=2
        )
        self.play(
            TransformFromCopy(emb, output_emb, path_arc=-30 * DEGREES),
            TransformFromCopy(vects[2], output_emb, path_arc=-30 * DEGREES),
            run_time=2
        )
        self.wait()

        # Show all in parallel
        simple_phases.add_to_back(highlight_rect)
        simple_phases.add(VGroup(sum_circuit, output_emb))
        simple_phase_copies = VGroup(
            simple_phases.copy().match_z(emb)
            for emb in embedding_array.embeddings
        )
        for sp_copy in simple_phase_copies:
            for group in sp_copy[1:]:
                arrow, vect = group
                for entry in vect.get_entries():
                    dot = Dot().scale(0.5)
                    dot.match_color(entry)
                    dot.set_fill(opacity=0.5)
                    dot.move_to(entry)
                    entry.become(dot)
                group.fade(0.5)

        self.play(
            frame.animate.reorient(0, -48, 0, (0.55, -2.21, 0.18), 7.05),
            LaggedStart((
                TransformFromCopy(simple_phases, sp_copy)
                for sp_copy in simple_phase_copies
            ), lag_ratio=0.1),
            FadeOut(block_title, time_span=(0, 1)),
            run_time=3,
        )
        self.play(frame.animate.reorient(9, -15, 0, (0.55, -2.21, 0.18), 7.05), run_time=4)
        self.play(frame.animate.reorient(-24, -16, 0, (0.18, -2.13, 0.09), 7.63), run_time=12)
        block_title.next_to(block, UP)
        self.play(
            frame.animate.to_default_state(),
            LaggedStartMap(FadeOut, simple_phase_copies, lag_ratio=0.1),
            FadeIn(block_title),
            run_time=2,
        )
        self.wait()

        # Show MJ -> Basketball example
        example_fact = TexText("\`\`Michael Jordan plays Basketball''", font_size=60)
        example_fact.to_edge(UP)

        mj = TexText("Michael Jordan", font_size=36)
        mj.next_to(emb, UL)
        mj_lines = VGroup(
            Line(char.get_bottom(), emb.get_top(), buff=0.1, path_arc=10 * DEGREES)
            for char in mj
        )
        mj_lines.set_stroke(YELLOW, 1, 0.5)

        basketball = TexText("Basketball", font_size=24)
        basketball.next_to(vects[2], UP, buff=0.2)

        self.play(Write(example_fact))
        self.wait()
        self.play(FadeTransform(example_fact[mj.get_tex()].copy(), mj))
        self.play(Write(mj_lines, stroke_width=2, stroke_color=YELLOW_B, lag_ratio=1e-2))
        self.wait()

        mover = emb.copy()
        for vect in vects:
            self.play(Transform(mover, vect, rate_func=linear))
        self.remove(mover)
        self.wait()
        self.play(FadeTransform(example_fact[basketball.get_tex()].copy(), basketball))
        self.wait(2)

        # Multiply by the up-projection
        up_proj = WeightMatrix(shape=(9, 6))
        up_proj.set_height(3)
        up_proj.to_corner(UL)
        up_emb.set_height(2)
        up_emb.next_to(up_proj, RIGHT)
        up_emb[-2:].set_fill(YELLOW)  # Brackets

        self.play(
            phases[1:].animate.set_opacity(0.1),
            sum_circuit.animate.set_stroke(opacity=0.1),
            output_emb.animate.set_opacity(0.1),
            FadeOut(mj),
            FadeOut(mj_lines),
            FadeOut(basketball),
            FadeOut(example_fact),
        )
        self.wait()
        self.play(TransformFromCopy(emb, up_emb))
        self.play(FadeIn(up_proj, lag_ratio=0.01))
        eq, rhs = show_matrix_vector_product(self, up_proj, up_emb)
        self.wait()
        data_modifying_matrix(self, up_proj, word_shape=(4, 7), fix_in_frame=True)
        self.wait()

        # Show machine
        machine = MachineWithDials(
            width=up_proj.get_width() + SMALL_BUFF,
            height=up_proj.get_height() + SMALL_BUFF,
            n_rows=8,
            n_cols=9,
        )
        machine.move_to(up_proj)

        self.play(FadeIn(machine))
        self.play(machine.random_change_animation())
        self.wait()
        self.play(FadeOut(machine))

        # Emphasize dot product with rows
        n, m = up_proj.shape
        n_rows_shown = 5
        R_labels = VGroup(
            Tex(R"\\vec{\\textbf{R}}_" + f"{{{n}}}")
            for n in [*range(n_rows_shown - 1), "n"]
        )
        R_labels[-2].become(Tex(R"\\vdots").replace(R_labels[-2], dim_to_match=1))
        R_labels.arrange(DOWN, buff=0.5)
        R_labels.match_height(up_proj)
        R_labels.move_to(up_proj)
        h_lines = VGroup(
            Line(up_proj.get_brackets()[0], R_labels, buff=0.1),
            Line(R_labels, up_proj.get_brackets()[1], buff=0.1),
        )
        h_lines.set_stroke(GREY_A, 2)
        row_labels = VGroup(
            VGroup(R_label, h_lines.copy().match_y(R_label))
            for R_label in R_labels
        )
        row_matrix = VGroup(
            up_proj.get_brackets().copy(),
            row_labels
        )

        E_label = Tex(R"\\vec{\\textbf{E}}")
        E_label.match_height(R_labels[0])
        E_label.set_color(YELLOW)
        E_label.move_to(up_emb)
        E_col = VGroup(
            up_emb[-2:].copy(),
            Line(up_emb.get_top(), E_label, buff=0.1).set_stroke(GREY_A, 2),
            E_label,
            Line(E_label, up_emb.get_bottom(), buff=0.1).set_stroke(GREY_A, 2),
        )

        dot_prods = VGroup()
        for n, R_label in enumerate(R_labels):
            if n == len(R_labels) - 2:
                dot_prod = R_label.copy()
            else:
                dot_prod = VGroup(
                    R_label.copy(),
                    Tex(R"\\cdot"),
                    E_label.copy(),
                )
                dot_prod.arrange(RIGHT, buff=0.1)
                dot_prod[-1].align_to(dot_prod[0][1], DOWN)
                dot_prod.set_width(rhs.get_width() * 0.75)
            dot_prod.move_to(R_label)
            dot_prods.add(dot_prod)
        dot_prods.move_to(rhs)
        dot_prod_rhs = VGroup(
            rhs.get_brackets().copy(),
            dot_prods,
        )

        self.play(LaggedStart(
            FadeOut(up_proj, scale=1.1),
            FadeIn(row_matrix, scale=1.1),
            FadeOut(up_emb, scale=1.1),
            FadeIn(E_col, scale=1.1),
            FadeOut(rhs, scale=1.1),
            FadeIn(dot_prod_rhs[0], scale=1.1),
            lag_ratio=0.1
        ))
        self.wait()
        for row_label, dot_prod in zip(row_labels, dot_prods):
            R_label = row_label[0]
            self.play(
                TransformFromCopy(R_label, dot_prod[0]),
                TransformFromCopy(R_label, dot_prod[1]),
                TransformFromCopy(E_label, dot_prod[2]),
                VShowPassingFlash(
                    Line(row_label.get_left(), row_label.get_right()).set_stroke(YELLOW, 5).insert_n_curves(100),
                    time_width=1.5
                ),
                VShowPassingFlash(
                    Line(E_col.get_top(), E_col.get_bottom()).set_stroke(YELLOW, 5).insert_n_curves(100),
                    time_width=1.5
                ),
                run_time=1
            )
        self.wait()

        # First name Michael direction
        row_rect = SurroundingRectangle(row_labels[0])
        row_rect.set_stroke(GREY_BROWN, 2)
        row_rect.set_fill(GREY_BROWN, 0.25)
        row_eq = Tex("=").rotate(PI / 2)
        row_eq.next_to(row_rect, UP, SMALL_BUFF)
        first_name_label = Tex(R"\\overrightarrow{\\text{First Name Michael}}")
        first_name_label.set_stroke(WHITE, 1)
        first_name_label.match_width(row_rect)
        first_name_label.next_to(row_eq, UP)

        dot_prod = dot_prods[0]
        dp_rect = SurroundingRectangle(dot_prod, buff=0.2)
        dp_rect.set_stroke(RED)
        dp_eq = Tex("=")
        dp_eq.next_to(dp_rect, RIGHT, SMALL_BUFF)
        mde_rhs = VGroup(
            Tex(R"\\approx 1 \\quad \\text{If } \\vec{\\textbf{E}} \\text{ encodes \`\`First Name Michael''}"),
            Tex(R"\\le 0 \\quad \\text{If not}")
        )
        mde_rhs[0][R"\\vec{\\textbf{E}}"].set_color(YELLOW)
        mde_rhs.scale(0.75)
        mde_rhs.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        rhs_brace = Brace(mde_rhs, LEFT)
        rhs_brace.next_to(dp_eq, RIGHT, SMALL_BUFF)
        mde_rhs.next_to(rhs_brace, RIGHT, MED_SMALL_BUFF)

        self.play(
            FadeIn(row_rect, scale=2),
            FadeTransform(row_labels[0].copy(), first_name_label),
            GrowFromCenter(row_eq),
            frame.animate.reorient(0, 0, 0, (0.22, 0.54, 0.0), 9.27),
        )
        self.wait()

        self.play(TransformFromCopy(row_rect.copy().set_fill(opacity=0), dp_rect))
        self.play(
            Write(dp_eq),
            GrowFromCenter(rhs_brace),
            FadeIn(mde_rhs),
        )
        self.wait()

        # "First name Michael" + "Last name Jordan"
        fn_tex = R"\\overrightarrow{\\text{F.N. Michael}}"
        ln_tex = R"\\overrightarrow{\\text{L.N. Jordan}}"
        name_sum_label = Tex(f"{fn_tex} + {ln_tex}")
        name_sum_label.match_width(row_rect).scale(1.2)
        name_sum_label.next_to(row_eq, UP)

        self.play(
            FadeTransform(first_name_label, name_sum_label[:21]),
            FadeIn(name_sum_label[21:], shift=RIGHT, scale=2),
            FadeOut(mde_rhs),
            FadeOut(rhs_brace),
        )
        self.wait()

        dist_rhs = VGroup(
            Tex(R"(\\vec{\\textbf{M}} + \\vec{\\textbf{J}}) \\cdot \\vec{\\textbf{E}}"),
            Tex("="),
            Tex(R"\\vec{\\textbf{M}} \\cdot \\vec{\\textbf{E}} + \\vec{\\textbf{J}} \\cdot \\vec{\\textbf{E}}"),
        )
        dist_rhs.scale(0.75)
        dist_rhs.arrange(RIGHT, buff=0.2)
        dist_rhs.next_to(dp_eq, RIGHT)
        for part in dist_rhs:
            part[R"\\vec{\\textbf{M}}"].set_color(RED_B)
            part[R"\\vec{\\textbf{J}}"].set_color(RED)
            part[R"\\vec{\\textbf{E}}"].set_color(YELLOW)
        under_brace = Brace(dist_rhs[2])

        two_condition = TexText(R"$\\approx 2$ \\; if $\\vec{\\textbf{E}}$ encodes \`\`Michael Jordan''")
        two_condition[R"\\vec{\\textbf{E}}"].set_color(YELLOW)
        else_condition = TexText(R"$\\le 1$ \\; Otherwise")
        VGroup(two_condition, else_condition).scale(0.75)
        two_condition.next_to(under_brace, DOWN, aligned_edge=LEFT)
        else_condition.next_to(two_condition, DOWN, MED_LARGE_BUFF, aligned_edge=LEFT)

        self.play(LaggedStart(
            FadeTransformPieces(name_sum_label[:21].copy(), dist_rhs[0][1:3]),
            FadeTransformPieces(name_sum_label[21].copy(), dist_rhs[0][3]),
            FadeTransformPieces(name_sum_label[22:].copy(), dist_rhs[0][4:6]),
            FadeTransformPieces(dot_prod[1:].copy(), dist_rhs[0][7:]),
            FadeIn(dist_rhs[0][0]),
            FadeIn(dist_rhs[0][6]),
            lag_ratio=0.2
        ))
        self.wait()
        self.play(
            TransformMatchingStrings(dist_rhs[0].copy(), dist_rhs[2], lag_ratio=0.01, path_arc=-45 * DEGREES),
            Write(dist_rhs[1])
        )
        self.wait()
        self.play(
            frame.animate.set_y(0.5),
            GrowFromCenter(under_brace),
            FadeIn(two_condition, DOWN)
        )
        self.wait()
        self.play(FadeIn(else_condition, DOWN))
        self.wait(2)

        # Go back to the numbers
        for entry in rhs.get_entries():
            entry.set_value(np.random.uniform(-10, 10))
        rhs.get_entries()[0].set_value(2.0)
        self.play(
            LaggedStart(*map(FadeOut, [
                name_sum_label, row_eq, row_rect,
                dp_rect, dp_eq, dist_rhs, under_brace,
                two_condition, else_condition,
            ]), lag_ratio=0.1, run_time=1),
            frame.animate.reorient(0, 0, 0, (-0.06, -0.06, 0.0), 8.27),
        )
        self.play(
            FadeOut(row_matrix),
            FadeIn(up_proj),
            FadeOut(E_col),
            FadeIn(up_emb),
            FadeOut(dot_prod_rhs),
            FadeIn(rhs),
        )

        # Show other rows
        questions = VGroup(*map(Text, [
            "Blah",
            "Is it English?",
            "Part of source code?",
            "European country?",
            "In quotation marks?",
            "Something metallic?",
            "A four-legged animal?",
        ]))
        questions.scale(0.75)
        rows = up_proj.get_rows()
        rhs_entries = rhs.get_entries()
        last_question = VGroup()
        last_rect = VectorizedPoint(rows[1].get_top())
        for index in range(1, 7):
            for mob in [rows, rhs_entries]:
                mob.target = mob.generate_target()
                mob.target.set_opacity(0.25)
                mob.target[index].set_opacity(1)
            row_rect = SurroundingRectangle(rows[index])
            row_rect.set_stroke(PINK, 2)
            question = questions[index]
            question.next_to(rows[index], UP, buff=0.15)
            question.set_backstroke(BLACK, 3)
            self.play(
                MoveToTarget(rows),
                MoveToTarget(rhs_entries),
                FadeOut(last_question),
                FadeIn(question),
                FadeTransform(last_rect, row_rect, time_span=(0, 0.75)),
                run_time=1.0
            )
            self.wait(0.5)
            last_question = question
            last_rect = row_rect
        self.play(
            rows.animate.set_opacity(1),
            rhs.animate.set_opacity(1),
            FadeOut(last_question),
            FadeOut(last_rect),
        )
        self.wait()

        # Add a bias
        plus = Tex("+")
        plus.next_to(up_emb, RIGHT)
        bias = WeightMatrix(shape=(9, 1), ellipses_col=None)
        bias.get_entries()[0].set_value(-1).set_color(RED)
        bias.match_height(up_proj)
        bias.next_to(plus)
        bias_name = Text("Bias")
        bias_name.next_to(bias, UP)

        eq.target = eq.generate_target()
        eq.target.next_to(bias, RIGHT)
        rhs.target = vects[0].copy()
        rhs.target.replace(rhs, dim_to_match=1)
        rhs.target.next_to(eq.target, RIGHT)

        self.play(
            Write(plus),
            FadeIn(bias, lag_ratio=0.1),
            MoveToTarget(eq),
            MoveToTarget(rhs),
        )
        self.wait()
        self.play(
            frame.animate.scale(1.1, about_edge=DOWN),
            Write(bias_name),
        )
        self.wait()

        # Emphasize the parameters are learned from data
        data_modifying_matrix(self, bias, word_shape=(5, 1), alpha_maxes=(0.4, 0.9), fix_in_frame=True)
        bias.get_entries()[0].set_value(-1).set_color(RED)

        # Pull up the MJ example again
        fe_rect = SurroundingRectangle(rhs.get_entries()[0], buff=0.1)  # fe = First entry
        fe_rect.set_stroke(RED, 3)
        fe_eq = Tex("=")
        fe_eq.next_to(fe_rect, RIGHT, SMALL_BUFF)
        fe_expr = VGroup(dist_rhs[2].copy(), Tex("- 1"))
        fe_expr[1].set_height(fe_expr[0].get_height() * 0.8)
        fe_expr.arrange(RIGHT)
        fe_expr.next_to(fe_eq, RIGHT)

        bias_rect = SurroundingRectangle(bias.get_entries()[0])

        self.play(
            ShowCreation(fe_rect),
            FadeIn(fe_eq, RIGHT),
            Write(fe_expr)
        )
        self.wait()
        self.play(ShowCreation(bias_rect))
        self.wait()
        self.play(bias_rect.animate.surround(fe_expr[1]))
        self.wait()
        self.play(bias_rect.animate.surround(fe_expr))
        self.wait()

        # Show what it means, but now shifted
        conditions = VGroup(
            TexText(R"$\\approx 1$ \\; if $\\vec{\\textbf{E}}$ encodes \`\`Michael Jordan''"),
            TexText(R"$\\le 0$ \\; Otherwise"),
        )
        conditions[0][R"\\vec{\\textbf{E}}"].set_color(YELLOW)
        conditions.scale(0.75)
        conditions.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        under_brace = Brace(fe_expr, DOWN)
        conditions.next_to(under_brace, DOWN, aligned_edge=LEFT)

        self.play(
            FadeOut(bias_rect),
            GrowFromCenter(under_brace),
            FadeIn(conditions[0], DOWN)
        )
        self.wait()
        self.play(FadeIn(conditions[1], 0.25 * DOWN))
        self.wait(2)

        self.play(
            frame.animate.reorient(0, 0, 0, (-2.5, 0.44, 0.0), 9.33),
            LaggedStart(*map(FadeOut, [
                fe_rect, fe_eq, fe_expr,
                under_brace, *conditions
            ]))
        )

        # Show the matrix size
        up_proj.refresh_bounding_box()
        row_rects = VGroup(
            SurroundingRectangle(row, buff=0.1)
            for row in up_proj.get_rows()
        )
        row_rects.set_stroke(WHITE, 1)
        row_rects.set_fill(GREY_C, 0.25)
        row_rects[-2].match_width(row_rects, stretch=True)

        over_brace = Brace(row_rects[0], UP, buff=SMALL_BUFF)
        d_model = 12288
        row_size = Integer(d_model)
        row_size.next_to(over_brace, UP)
        side_brace = Brace(row_rects, LEFT)
        num_rows = Integer(4 * d_model)
        num_rows.next_to(side_brace, LEFT)
        num_rows_expr = Tex(R"4 \\times 12{,}288")
        num_rows_expr.next_to(side_brace, LEFT)

        self.play(
            FadeIn(row_rects, lag_ratio=0.5),
            GrowFromCenter(side_brace),
            CountInFrom(num_rows)
        )
        self.wait()
        self.play(FadeTransform(num_rows, num_rows_expr))
        self.wait()
        self.play(
            FadeTransform(num_rows_expr["12{,}288"].copy(), row_size),
            TransformFromCopy(side_brace, over_brace),
        )
        self.wait()
        self.play(FadeOut(row_rects, lag_ratio=0.1))

        # Calculate matrix size
        full_product = VGroup(
            num_rows_expr.copy(),
            Tex(R"\\times"),
            row_size.copy(),
            Tex(Rf"="),
            Integer(4 * d_model * d_model)
        )
        full_product.scale(1.5)
        full_product.arrange(RIGHT, buff=MED_SMALL_BUFF)
        full_product.next_to(row_rects, UP, buff=2.5)

        self.play(LaggedStart(
            frame.animate.reorient(0, 0, 0, (-3.88, 1.51, 0.0), 11.35),
            TransformFromCopy(num_rows_expr, full_product[0]),
            FadeIn(full_product[1], UP),
            TransformFromCopy(row_size, full_product[2]),
            lag_ratio=0.25,
            run_time=2
        ))
        self.play(
            TransformFromCopy(full_product[:3], full_product[3:])
        )
        self.wait()
        self.play(FlashAround(full_product[-1], run_time=2, time_width=1.5))

        # Count bias parameters
        bias_count = Tex(R"4 \\times 12{,}288")
        bias_count.match_height(full_product)
        bias_count.match_y(full_product)
        bias_count.match_x(bias)
        bias_rect = SurroundingRectangle(VGroup(bias, bias_name))
        bias_rect.set_stroke(BLUE_B)
        bias_arrow = Arrow(bias_rect.get_top(), bias_count.get_bottom())
        bias_arrow.match_color(bias_rect)
        bias_count.match_color(bias_rect)

        div_eq = Tex(R"{4 \\times 12{,}288 \\over 603{,}979{,}776} \\approx 0.00008 ")
        div_eq[R"{4 \\times 12{,}288"].match_color(bias_rect)
        div_eq.next_to(frame.get_corner(UR), DL, buff=MED_LARGE_BUFF)
        div_eq.shift(RIGHT)

        self.play(ShowCreation(bias_rect))
        self.play(
            GrowArrow(bias_arrow),
            FadeInFromPoint(bias_count, bias_arrow.get_start()),
            full_product.animate.scale(0.8).shift(3.5 * LEFT)
        )
        self.wait()
        self.play(
            frame.animate.set_x(-3.0),
            FadeTransform(bias_count.copy(), div_eq[R"4 \\times 12{,}288"]),
            Write(div_eq[R"\\over"]),
            FadeTransform(full_product[-1].copy(), div_eq[R"603{,}979{,}776}"]),
            Write(div_eq[R"\\approx 0.00008"]),
        )
        self.wait()

        self.play(
            frame.animate.reorient(0, 0, 0, (-2.5, 0.44, 0.0), 9.33),
            *map(FadeOut, [full_product, bias_rect, bias_arrow, bias_count, div_eq])
        )

        # Collapse
        substrs = [R"W_\\uparrow", R"\\vec{\\textbf{E}}_i", "+", R"\\vec{\\textbf{B}}_\\uparrow"]
        linear_expr = Tex(" ".join(substrs))
        W_up, E_i, plus2, B_up = [linear_expr[ss] for ss in substrs]
        VGroup(W_up, B_up).set_color(BLUE)
        E_i.set_color(YELLOW)
        linear_expr.move_to(plus).shift(0.6 * LEFT)

        low_emb_label = E_i.copy()
        low_emb_label.scale(0.5).next_to(emb, UP)

        self.play(
            frame.animate.reorient(0, 0, 0, (-0.03, 0.03, 0.0), 8.34),
            ReplacementTransform(up_proj, W_up, lag_ratio=1e-3),
            FadeOut(side_brace, RIGHT, scale=0.5),
            FadeOut(num_rows_expr, RIGHT, scale=0.5),
            FadeOut(over_brace, DR, scale=0.5),
            FadeOut(row_size, DR, scale=0.5),
        )
        self.wait()
        self.play(ReplacementTransform(up_emb, E_i, lag_ratio=1e-2))
        self.play(TransformFromCopy(E_i, low_emb_label))
        self.wait()
        self.play(
            ReplacementTransform(plus, plus2),
            ReplacementTransform(bias, B_up, lag_ratio=1e-2),
            FadeOut(bias_name, DL),
            VGroup(eq, rhs).animate.next_to(B_up, RIGHT).shift(0.1 * DOWN),
            run_time=2
        )
        self.wait()

        # Add parameters below first linear arrow
        self.play(
            linear_expr.animate.scale(0.5).next_to(arrows[0], DOWN, buff=0.1),
            ReplacementTransform(rhs, vects[0]),
            FadeOut(eq, 4 * DOWN + LEFT),
            run_time=2
        )
        self.wait()

        # Pull up ReLU
        self.play(phases[1].animate.set_opacity(1))
        phase1_copy = VGroup(vects[0], arrows[1], vects[1]).copy()
        phase1_copy.save_state()

        self.play(
            phase1_copy.animate.scale(2.0).next_to(full_block, UP, buff=0.5),
            frame.animate.reorient(0, 0, 0, (-0.26, 0.54, 0.0), 9.40)
        )
        self.wait()

        # Break down ReLU
        relu_arrow = phase1_copy[1]
        neg_arrows = VGroup()
        pos_arrows = VGroup()
        neg_left_rects = VGroup()
        zero_right_rects = VGroup()
        pos_left_rects = VGroup()
        pos_right_rects = VGroup()
        in_vect = phase1_copy[0]
        out_vect = phase1_copy[2]
        for e1, e2 in zip(in_vect.get_entries(), out_vect.get_entries()):
            arrow = Arrow(e1, e2, buff=0.3)
            if e1.get_value() > 0:
                arrow.set_color(BLUE)
                pos_arrows.add(arrow)
                pos_left_rects.add(SurroundingRectangle(e1, color=BLUE))
                pos_right_rects.add(SurroundingRectangle(e2, color=BLUE))
            else:
                arrow.set_color(RED)
                neg_arrows.add(arrow)
                neg_left_rects.add(SurroundingRectangle(e1, color=RED))
                zero_right_rects.add(SurroundingRectangle(e2, color=RED))
        VGroup(neg_left_rects, zero_right_rects, pos_left_rects, pos_right_rects).set_stroke(width=2)

        self.play(ShowCreation(neg_left_rects, lag_ratio=0.5))
        self.wait()
        self.play(
            TransformFromCopy(neg_left_rects, zero_right_rects, lag_ratio=0.5),
            ShowCreation(neg_arrows, lag_ratio=0.5),
            FadeOut(relu_arrow),
        )
        self.wait()
        self.play(
            FadeOut(neg_left_rects, lag_ratio=0.25),
            FadeOut(zero_right_rects, lag_ratio=0.25),
            FadeOut(neg_arrows, lag_ratio=0.25),
            ShowCreation(pos_left_rects)
        )
        self.wait()
        self.play(
            ShowCreation(pos_arrows, lag_ratio=0.5),
            TransformFromCopy(pos_left_rects, pos_right_rects, lag_ratio=0.5),
        )
        self.wait()

        # Graph ReLU
        relu_title_full = Text("Rectified\\nLinear\\nUnit", alignment="LEFT")
        relu_title_full.next_to(relu_arrow, UP)

        axes = Axes((-4, 4), (-1, 4))
        axes.set_width(6)
        axes.next_to(phase1_copy, RIGHT, buff=1.0)
        axes.add_coordinate_labels(font_size=16)
        relu_graph = axes.get_graph(lambda x: max(0, x), discontinuities=[0])
        relu_graph.set_stroke(YELLOW, 4)
        plot = VGroup(axes, relu_graph)

        relu_graph_label = Text("ReLU")
        relu_graph_label.match_color(relu_graph)
        relu_graph_label.move_to(axes, UL)

        self.play(
            frame.animate.set_x(2.7),
            FadeIn(relu_arrow),
            FadeIn(relu_title_full, 0.1 * UP, lag_ratio=0.1, run_time=2),
            FadeOut(pos_arrows, lag_ratio=0.25),
            FadeOut(pos_left_rects, lag_ratio=0.25),
            FadeOut(pos_right_rects, lag_ratio=0.25),
            FadeIn(plot, RIGHT),
        )
        self.wait()
        self.play(*(
            TransformFromCopy(relu_title_full[substr], relu_graph_label[substr])
            for substr in ["Re", "L", "U"]
        ))
        self.add(relu_graph_label)

        # Recall the meaning of the first entry
        mid_vect = phase1_copy[0]
        conditions_rect = SurroundingRectangle(conditions, buff=0.25)
        conditions_rect.set_stroke(YELLOW, 1)
        under_brace = Brace(conditions_rect, DOWN, buff=SMALL_BUFF)
        VGroup(conditions, conditions_rect, under_brace).next_to(mid_vect, UP)
        fe_rect = SurroundingRectangle(mid_vect.get_entries()[0])

        condition_group = VGroup(fe_rect, under_brace, conditions, conditions_rect)

        self.play(
            frame.animate.reorient(0, 0, 0, (2.61, 0.97, 0.0), 11.5),
            ShowCreation(fe_rect),
            GrowFromCenter(under_brace),
        )
        self.play(
            TransformFromCopy(fe_rect, conditions_rect),
            FadeInFromPoint(conditions, fe_rect.get_center()),
        )
        self.wait()
        self.play(condition_group.animate.match_x(phase1_copy[2]))

        equals = Tex("=")
        ineq = conditions[1][0]
        equals.replace(ineq, dim_to_match=0)
        self.play(
            FlashAround(equals, run_time=2, time_width=1.5),
            ineq.animate.become(equals)
        )
        self.wait()
        self.play(
            frame.animate.reorient(0, 0, 0, (2.48, 0.33, 0.0), 9.17),
            FadeOut(condition_group, lag_ratio=0.01)
        )

        # Graph GeLU
        gelu_title_full = Text("Gaussian\\nError\\nLinear\\nUnit", font_size=42, alignment="LEFT")
        gelu_title_full.next_to(relu_arrow, UP)
        gelu_graph = axes.get_graph(lambda x: x * norm.cdf(x))
        gelu_graph.set_stroke(GREEN, 4)

        gelu_graph_label = Text("GELU")
        gelu_graph_label.next_to(relu_graph_label, DOWN, buff=MED_LARGE_BUFF, aligned_edge=LEFT)
        gelu_graph_label.match_color(gelu_graph)

        self.play(
            FadeTransform(relu_title_full, gelu_title_full),
            relu_graph_label.animate.set_fill(opacity=0.25),
            relu_graph.animate.set_stroke(opacity=0.25),
            ShowCreation(gelu_graph),
            TransformFromCopy(relu_graph_label, gelu_graph_label)
        )
        self.wait(2)
        self.play(
            gelu_graph.animate.set_stroke(opacity=0.25),
            gelu_graph_label.animate.set_fill(opacity=0.25),
            relu_graph.animate.set_stroke(opacity=1),
            relu_graph_label.animate.set_fill(opacity=1),
            FadeTransform(gelu_title_full, relu_title_full),
        )
        self.wait()

        # Describe these as neurons
        neuron_word = Text("Neurons", font_size=72)
        neuron_word.next_to(phase1_copy, RIGHT, buff=2.5)
        neuron_arrows = VGroup(
            Arrow(neuron_word.get_left(), entry.get_right(), buff=0.4, stroke_width=3)
            for entry in phase1_copy[2].get_entries()
        )

        self.play(
            plot.animate.set_width(2).next_to(relu_arrow, DOWN),
            FadeOut(VGroup(relu_graph_label, gelu_graph_label, gelu_graph)),
            Write(neuron_word),
            ShowCreation(neuron_arrows, lag_ratio=0.2, run_time=3),
            LaggedStartMap(
                FlashAround, phase1_copy[2].get_entries(),
                time_width=3.0,
                lag_ratio=0.05,
                time_span=(1, 4),
                run_time=4
            )
        )
        self.wait()

        # Show the classic dots picture
        blocking_rect = BackgroundRectangle(VGroup(phase1_copy), buff=0.1)
        blocking_rect.set_fill(BLACK, 1)
        up_emb.move_to(blocking_rect, LEFT)
        dots = VGroup(
            Dot(radius=0.15).move_to(entry).set_fill(WHITE, opacity=clip(entry.get_value(), 0, 1))
            for entry in phase1_copy[2].get_entries()
        )
        dots.set_stroke(WHITE, 2)
        up_emb = emb.copy()
        up_emb.rotate(PI / 2, DOWN)
        up_emb.rotate(1 * DEGREES)
        up_emb.match_width(phase1_copy[0])
        up_emb.move_to(phase1_copy[0]).shift(RIGHT)
        up_emb[-2:].set_color(YELLOW)
        lines = VGroup(
            Line(entry.get_right() + 0.05 * RIGHT, dot).set_stroke(
                color=value_to_color(random.uniform(-10, 10)),
                width=3 * random.random()**2,
            )
            for entry in up_emb.get_entries()
            for dot in dots
        )

        self.play(
            FadeIn(blocking_rect),
            Write(dots),
        )
        self.play(TransformFromCopy(emb, up_emb))
        self.play(ShowCreation(lines, lag_ratio=3 / len(lines)))
        self.wait()
        self.play(
            LaggedStart(*map(FadeOut, [up_emb, *lines, blocking_rect, *dots]), lag_ratio=0.01)
        )

        # Discuss active and inactive
        entry = phase1_copy[2].get_entries()[0]
        entry_rect = SurroundingRectangle(entry)
        entry_rect.set_stroke(YELLOW, 2)
        active_words = TexText(R"\`\`Michael Jordan'' neuron is \\emph{active}")
        active = active_words["active"][0]
        active.set_color(BLUE_B)
        active_words.next_to(entry_rect, UP, aligned_edge=LEFT)
        active_words.shift(LEFT)
        inactive = TexText(R"\\emph{inactive}")
        inactive.set_color(RED)
        inactive.move_to(active, LEFT)

        self.play(
            frame.animate.reorient(0, 0, 0, (2.45, 0.58, 0.0), 9.65),
            ShowCreation(entry_rect),
            Write(active_words, run_time=1),
        )
        self.wait()
        self.play(
            ChangeDecimalToValue(entry, 0),
            ReplacementTransform(active, inactive[2:]),
            GrowFromCenter(inactive[:2]),
        )
        active_words.add(inactive)
        self.wait()

        # Replace the ReLU diagram portion
        self.play(
            Restore(phase1_copy),
            TransformMatchingStrings(relu_title_full, arrow_labels[1]),
            plot.animate.scale(0.5).next_to(arrows[1], DOWN, SMALL_BUFF),
            FadeOut(neuron_word, DOWN),
            FadeOut(neuron_arrows, DOWN, lag_ratio=0.1),
            FadeOut(entry_rect, DOWN),
            FadeOut(active_words, DOWN, lag_ratio=0.01),
            run_time=1.5
        )
        self.remove(phase1_copy)

        # Down projection
        neurons = vects[1].copy()
        neurons.target = neurons.generate_target()
        neurons.target.set_height(4)
        neurons.target.move_to(3 * RIGHT + 2.5 * UP)
        down_proj = WeightMatrix(shape=(6, 9))
        down_proj.set_height(2.75)
        down_proj.next_to(neurons.target, LEFT)

        plus = Tex("+")
        plus.next_to(neurons.target, RIGHT)
        bias = WeightMatrix(shape=(6, 1))
        bias.match_height(down_proj)
        bias.next_to(plus, RIGHT)

        equals = Tex("=")
        equals.next_to(bias, RIGHT)
        rhs = vects[2].copy()
        rhs.set_opacity(1)
        rhs.match_height(bias)
        rhs.next_to(equals, RIGHT)

        self.play(phases[2].animate.set_opacity(1))
        self.play(MoveToTarget(neurons))
        self.play(FadeTransform(arrows[2].copy(), down_proj))
        self.wait()
        temp_eq, temp_rhs = show_matrix_vector_product(self, down_proj, neurons)
        self.wait()
        self.play(
            FadeOut(temp_eq, DOWN),
            FadeOut(temp_rhs, DOWN),
            Write(plus),
            FadeIn(bias, RIGHT),
        )
        self.wait()
        self.play(
            Write(equals),
            TransformFromCopy(vects[2], rhs),
        )
        self.wait()

        # Name it as the down-projection
        over_brace = Brace(down_proj, UP)
        name = TexText("\`\`Down projection''")
        name.next_to(over_brace, UP)

        side_brace = Brace(rhs, RIGHT)
        dim_count = Integer(12288)
        dim_count.next_to(side_brace, RIGHT)

        self.play(
            CountInFrom(dim_count),
            GrowFromCenter(side_brace),
        )
        self.wait()
        self.play(
            Write(name),
            GrowFromCenter(over_brace),
        )
        self.wait()

        # Show column-by-column
        col_matrix = self.get_col_matrix(down_proj, 7)
        bias_as_col = self.get_col_matrix(bias, 1, dots_index=None, sym="B", top_index="", width_multiple=0.7)
        n_labels = VGroup(
            Tex(f"n_{{{m}}}")
            for m in [*range(6), "m"]
        )
        n_labels.arrange(DOWN, buff=0.5)
        n_labels.match_height(neurons.get_entries())
        n_labels.move_to(neurons.get_entries())
        n_labels.replace_submobject(-2, Tex(R"\\vdots").move_to(n_labels[-2]))
        n_labels.set_color(BLUE)
        n_vect = VGroup(neurons[-2:].copy(), n_labels)

        self.play(
            LaggedStart(*map(FadeOut, [over_brace, name, side_brace, dim_count])),
            LaggedStart(
                FadeOut(down_proj),
                FadeIn(col_matrix),
                FadeOut(neurons),
                FadeIn(n_vect),
                FadeOut(bias),
                FadeIn(bias_as_col),
            )
        )
        self.wait()

        # Expand the column interpretation
        over_brace = Brace(VGroup(col_matrix, n_vect), UP)
        scaled_cols = VGroup(
            VGroup(n_label, col_label[0]).copy()
            for n_label, col_label in zip(n_labels, col_matrix[1])
        )
        scaled_cols.target = VGroup()
        for pair in scaled_cols:
            pair.target = pair.generate_target()
            pair.target[0].scale(1.5)
            pair.target.arrange(RIGHT, buff=0.1, aligned_edge=DOWN)
            scaled_cols.target.add(pair.target)
        scaled_cols.target[-2].become(Tex(R"\\dots"))
        scaled_cols.target.arrange(RIGHT, buff=0.75)
        scaled_cols.target.set_width(1.25 * over_brace.get_width())
        scaled_cols.target.next_to(over_brace, UP, buff=0.5)

        plusses = VGroup(
            Tex("+").move_to(midpoint(m1.get_right(), m2.get_left()))
            for m1, m2 in zip(scaled_cols.target, scaled_cols.target[1:])
        )

        self.play(
            frame.animate.reorient(0, 0, 0, (-0.27, 1.04, 0.0), 11.06),
            GrowFromCenter(over_brace),
            LaggedStartMap(MoveToTarget, scaled_cols, lag_ratio=0.7, run_time=5),
            LaggedStartMap(FadeIn, plusses, lag_ratio=0.7, run_time=5),
        )
        self.wait()

        # Highlight each set
        last_rects = VGroup()
        all_rect_groups = VGroup()
        for tup in zip(col_matrix[1], n_labels, scaled_cols):
            rects = VGroup(SurroundingRectangle(mob) for mob in tup)
            rects.set_stroke(YELLOW, 2)
            self.play(
                FadeOut(last_rects),
                FadeIn(rects),
            )
            self.wait(0.5)
            all_rect_groups.add(rects)
            last_rects = rects
        self.play(FadeOut(last_rects))

        # First column as basketball
        col_rect, n_rect, prod_rect = rects = all_rect_groups[0]
        basketball = Text("Basketball", font_size=60)
        basketball.set_color("#F88158")
        basketball.next_to(col_rect, LEFT)
        basketball.save_state()
        basketball.rotate(-PI / 2)
        basketball.move_to(col_rect)
        basketball.set_opacity(0)

        n0_term = scaled_cols[0][0]
        n0_term.save_state()
        one = Tex("1", font_size=60).move_to(n0_term, DR).set_color(BLUE)
        zero = Tex("0", font_size=60).move_to(n0_term, DR).set_color(RED)

        self.play(
            ShowCreation(col_rect),
            col_matrix[1][1:].animate.set_opacity(0.5),
            n_labels[1:].animate.set_opacity(0.5),
            scaled_cols[1:].animate.set_opacity(0.5),
            plusses.animate.set_opacity(0.5)
        )
        self.play(Restore(basketball, path_arc=PI / 2))
        self.wait()
        self.play(TransformFromCopy(col_rect, n_rect))
        self.wait()
        self.play(
            TransformFromCopy(col_rect, prod_rect),
            TransformFromCopy(n_rect, prod_rect),
        )
        self.play(Transform(n0_term, one))
        self.wait()
        self.play(Transform(n0_term, zero))
        self.wait()
        self.play(Restore(n0_term))
        n0_term.restore()
        self.wait()

        # Cycle through columns one more time
        rects.add(basketball)
        for index in range(1, len(all_rect_groups)):
            self.play(
                FadeOut(all_rect_groups[index - 1]),
                FadeIn(all_rect_groups[index]),
                col_matrix[1][index].animate.set_opacity(1),
                n_labels[index].animate.set_opacity(1),
                scaled_cols[index].animate.set_opacity(1),
                plusses[index - 1].animate.set_opacity(1),
            )
            self.wait(0.5)
        self.play(FadeOut(all_rect_groups[-1]))

        # Highlight bias
        bias_rect = SurroundingRectangle(bias)
        bias_brace = Brace(bias_rect, UP)
        bias_word = Text("Bias")
        bias_word.next_to(bias_brace, UP, MED_SMALL_BUFF)

        self.play(
            ReplacementTransform(over_brace, bias_brace),
            FadeIn(bias_rect),
            FadeOut(plusses, lag_ratio=0.1),
            FadeOut(scaled_cols, lag_ratio=0.1),
        )
        self.play(FadeIn(bias_word, 0.5 * UP))
        self.wait()
        self.play(LaggedStart(*map(FadeOut, [bias_word, bias_brace, bias_rect])))

        # Collpase the down projection
        W_down = Tex(R"W_\\downarrow", font_size=60).set_color(BLUE)
        B_down = Tex(R"\\vec{\\textbf{B}}_\\downarrow", font_size=60).set_color(BLUE_B)
        W_down.next_to(neurons, LEFT)
        B_down.move_to(bias_as_col)
        WB_down = VGroup(W_down, B_down)
        n_rect = Rectangle(1, 1)
        n_rect.set_height(W_down.get_height())
        n_rect.move_to(n_vect)
        n_rect.set_fill(GREY_C)
        n_rect.set_stroke(WHITE, 1)

        down_proj_expr = VGroup(W_down, n_vect, plus, B_down)
        down_proj_expr.target = down_proj_expr.generate_target()
        down_proj_expr.target[1].become(VGroup(n_rect))
        down_proj_expr.target.arrange(RIGHT, buff=SMALL_BUFF)
        down_proj_expr.target.scale(0.4)
        down_proj_expr.target.next_to(arrows[2], DOWN)

        self.play(ReplacementTransform(col_matrix, W_down, lag_ratio=5e-3, run_time=2))
        self.play(ReplacementTransform(bias_as_col, B_down, lag_ratio=1e-2))
        self.wait()
        self.play(
            LaggedStart(
                MoveToTarget(down_proj_expr),
                FadeOut(equals, 2 * DOWN + 0.5 * LEFT),
                ReplacementTransform(rhs, vects[2]),
                lag_ratio=0.25,
                time_span=(0, 1.5),
            ),
            frame.animate.reorient(0, -14, 0, (-0.1, -2.03, 0.01), 6.31),
            run_time=2,
        )
        self.wait()

        # Add it to the original
        faded_sum_circuit = sum_circuit.copy()
        sum_circuit.set_stroke(opacity=1)
        sum_circuit.insert_n_curves(20)

        self.add(faded_sum_circuit)
        self.play(
            frame.animate.reorient(13, -8, 0, (0.15, -2.05, 0.0), 6.52),
            ShowCreation(sum_circuit, lag_ratio=0.5),
            low_emb_label.animate.shift(0.2 * LEFT).set_anim_args(time_span=(0, 1)),
            FadeOut(output_emb),
            run_time=2,
        )
        self.remove(faded_sum_circuit)
        output_emb.set_fill(opacity=1)
        self.play(LaggedStart(
            TransformFromCopy(emb, output_emb, path_arc=-45 * DEGREES),
            TransformFromCopy(vects[2], output_emb, path_arc=-45 * DEGREES),
            run_time=2,
            lag_ratio=0.2,
        ))
        self.wait()

        # Yet again, emphasize the MJ example
        m_color = interpolate_color_by_hsl(GREY_BROWN, WHITE, 0.5)
        j_color = RED_B
        b_color = basketball.get_color()
        m_tex = Tex(R"\\overrightarrow{\\text{F.N. Michael}}").set_color(m_color)
        j_tex = Tex(R"\\overrightarrow{\\text{L.N. Jordan}}").set_color(j_color)
        b_tex = Tex(R"\\overrightarrow{\\text{Basketball}}").set_color(b_color)
        mj = VGroup(m_tex, Tex("+"), j_tex).copy()
        mjb = VGroup(m_tex, Tex("+"), j_tex, Tex("+"), b_tex).copy()
        for tex_mob in [mj, mjb]:
            tex_mob.set_height(0.45)
            tex_mob.arrange(RIGHT, buff=SMALL_BUFF)
            tex_mob.set_fill(border_width=1)
        mj.next_to(low_emb_label, UP, buff=1.0).shift(0.5 * LEFT)
        mjb.next_to(output_emb, UP, buff=1.5).shift(1.0 * RIGHT)
        mj_arrow = Arrow(mj.get_bottom(), low_emb_label, buff=0.1)
        mjb_arrow = Arrow(output_emb.get_top(), mjb.get_bottom(), buff=0.15)

        self.play(
            frame.animate.reorient(4, -6, 0, (-0.29, -1.76, 0.02), 7.70),
            FadeIn(mj, lag_ratio=0.1),
            ShowCreation(mj_arrow)
        )
        self.play(Transform(mj.copy(), emb.copy().set_opacity(0), lag_ratio=0.005, remover=True, run_time=2))
        mover = emb.copy()
        for vect in [*vects, output_emb]:
            self.play(Transform(mover, vect, rate_func=linear))
        self.remove(mover)
        self.play(
            frame.animate.reorient(-3, -5, 0, (1.09, -1.48, -0.03), 9.61),
            FadeTransform(mj.copy(), mjb[:3]),
            FadeTransformPieces(mj.copy()[-1:], mjb[3:]),
            ShowCreation(mjb_arrow),
            run_time=2,
        )
        self.wait(2)
        self.play(
            frame.animate.reorient(21, -14, 0, (-0.13, -2.21, 0.11), 6.91).set_anim_args(run_time=5),
            LaggedStartMap(FadeOut, VGroup(mj, mj_arrow, mjb_arrow, mjb)),
        )

        # Show it done in parallel to all embeddings
        self.play(
            frame.animate.reorient(14, -12, 0, (0.55, -2.21, 0.18), 7.05),
            LaggedStart((
                TransformFromCopy(simple_phases, sp_copy)
                for sp_copy in simple_phase_copies
            ), lag_ratio=0.1),
            FadeOut(block_title, time_span=(0, 1)),
            run_time=5,
        )
        self.play(
            frame.animate.reorient(42, -23, 0, (0.55, -2.21, 0.18), 7.05),
            run_time=8
        )

        self.wait()

        # Show neurons?
        sum_circuits = VGroup(
            sum_circuit,
            *(sp[0] for sp in simple_phase_copies),
            *(sp[-1] for sp in simple_phase_copies),
        )
        n_vects = VGroup(vects[1], *(sp[2][1] for sp in simple_phase_copies))

        neuron_points = np.array([
            entry.get_center()
            for vect in n_vects[1:]
            for entry in vect.get_entries()
        ])
        neurons = DotCloud(neuron_points)
        neurons.set_radius(0.075)
        neurons.set_shading(0.25, 0.25, 0.5)
        neurons.apply_depth_test()
        rgbas = np.random.random(len(neuron_points))
        rgbas = rgbas.repeat(4).reshape((rgbas.size, 4))
        rgbas[:, 3] = 1
        neurons.set_rgba_array(rgbas)
        neuron_ellipses = VGroup(
            n_vect.get_ellipses()
            for n_vect in n_vects[1:]
        )

        self.play(
            frame.animate.reorient(11, -5, 0, (0.55, -2.21, 0.18), 7.05),
            sum_circuits.animate.set_stroke(width=1, opacity=0.2),
            FadeOut(block[4]),
            run_time=2
        )
        self.play(
            frame.animate.reorient(-11, -5, 0, (0.55, -2.21, 0.18), 7.05).set_anim_args(run_time=4),
            FadeOut(n_vects),
            ShowCreation(neurons, run_time=2),
            FadeIn(neuron_ellipses, time_span=(1, 2)),
        )
        self.add(neuron_ellipses)
        self.play(frame.animate.reorient(13, -7, 0, (0.55, -2.21, 0.18), 7.05), run_time=4)
        self.wait()

    def get_sum_circuit(
        self, in_vect, diff_vect,
        v_buff=0.15,
        h_buff=0.5,
        y_diff=0.65,
        color=YELLOW
    ):
        plus = VGroup(Line(UP, DOWN), Line(LEFT, RIGHT))
        plus.scale(0.6)
        circle = Circle(radius=1)
        oplus = VGroup(circle, plus)
        oplus.set_height(0.3)
        oplus.next_to(diff_vect, RIGHT, buff=h_buff)

        p0 = in_vect.get_top() + v_buff * UP
        p1 = in_vect.get_top() + y_diff * UP
        p2 = oplus.get_center()
        p2[1] = p1[1]
        p3 = oplus.get_top()
        top_line = VMobject()
        top_line.set_points_as_corners([p0, p1, p2, p3])

        oplus.refresh_bounding_box()  # Why?
        h_line1 = Line(diff_vect.get_right(), oplus.get_left())
        h_line2 = Line(oplus.get_right(), oplus.get_right() + h_buff * RIGHT)

        output = diff_vect.copy()
        output.next_to(h_line2, RIGHT, buff=0)
        for e1, e2, e3 in zip(in_vect.get_entries(), diff_vect.get_entries(), output.get_entries()):
            e3.set_value(e1.get_value() + e2.get_value())

        circuit = VGroup(top_line, oplus, h_line1, h_line2)
        circuit.set_stroke(color, 3)

        return circuit, output

    def get_col_matrix(self, matrix, n_cols_shown, dots_index=-2, sym="C", top_index="m-1", width_multiple=1.0):
        C_labels = VGroup(
            Tex(Rf"\\vec{{\\textbf{{{sym}}}}}_{{{n}}}")
            for n in [*range(n_cols_shown - 1), top_index]
        )
        C_labels.arrange(RIGHT, buff=0.5)
        C_labels.move_to(matrix.get_entries())
        C_labels.set_width(matrix.get_entries().get_width() * width_multiple)


        v_lines = VGroup(
            Line(matrix.get_bottom(), C_labels.get_bottom() + SMALL_BUFF * DOWN),
            Line(C_labels.get_top() + SMALL_BUFF * UP, matrix.get_top()),
        )
        v_lines.set_stroke(WHITE, 1)
        col_labels = VGroup(
            VGroup(C_label, v_lines.copy().match_x(C_label))
            for C_label in C_labels
        )
        if dots_index is not None:
            dots = Tex(R"\\hdots")
            dots.move_to(col_labels[dots_index])
            col_labels.replace_submobject(dots_index, dots)

        return VGroup(matrix.get_brackets().copy(), col_labels)


class NonlinearityOfLanguage(InteractiveScene):
    def construct(self):
        # Set up axes and M + J
        unit_size = 2.5

        plane = NumberPlane(
            axis_config=dict(
                stroke_width=1,
            ),
            background_line_style=dict(
                stroke_color=BLUE_D,
                stroke_width=1,
                stroke_opacity=0.75
            ),
            faded_line_ratio=1,
            unit_size=unit_size,
        )
        m_vect = Vector(unit_size * RIGHT).rotate(60 * DEGREES, about_point=ORIGIN)
        j_vect = m_vect.copy().rotate(-90 * DEGREES, about_point=ORIGIN)
        m_vect.set_color(YELLOW)
        j_vect.set_color(RED)
        m_ghost = m_vect.copy().shift(j_vect.get_vector())
        j_ghost = j_vect.copy().shift(m_vect.get_vector())
        VGroup(m_ghost, j_ghost).set_stroke(opacity=0.25)

        sum_point = m_ghost.get_end()
        span_line = Line(-sum_point, sum_point)
        span_line.set_length(2 * FRAME_WIDTH)
        span_line.set_stroke(WHITE, 2, opacity=0.5)

        self.add(plane)
        self.add(m_vect, m_ghost, j_vect, j_ghost)
        self.add(span_line)

        # Label vectors
        m_label = Text("First Name Michael")
        j_label = Text("Last Name Jordan")
        for label, vect in [(m_label, m_vect), (j_label, j_vect)]:
            label.scale(0.6)
            label.match_color(vect)
            direction = np.sign(vect.get_vector()[1]) * UP
            label.next_to(ORIGIN, direction, buff=0.2, aligned_edge=LEFT)
            label.rotate(vect.get_angle(), about_point=ORIGIN)
            label.set_backstroke(BLACK, 3)

        self.add(m_label)
        self.add(j_label)

        # Add dot product expression
        expr = Tex(R"(\\vec{\\textbf{M}} + \\vec{\\textbf{J}}) \\cdot \\textbf{E}")
        expr[1:3].match_color(m_vect)
        expr[4:6].match_color(j_vect)
        expr.to_corner(UL)
        self.add(expr)

        # Set up embedding with dot product tracker
        emb_point = VectorizedPoint(unit_size * UL)
        emb = Vector()
        emb.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, emb_point.get_center()))
        normalized_sum = normalize(sum_point)

        def get_line_point():
            return normalized_sum * np.dot(normalized_sum, emb_point.get_center())

        shadow = Line()
        shadow.set_stroke(PINK, 3)
        shadow.add_updater(lambda m: m.put_start_and_end_on(ORIGIN, get_line_point()))  # This is a long line

        dot = Dot()
        dot.set_fill(PINK, 1)
        dot.f_always.move_to(get_line_point)

        dashed_line = always_redraw(
            lambda: DashedLine(emb_point.get_center(), get_line_point()).set_stroke(PINK, 2)
        )

        dp_decimal = DecimalNumber(font_size=36)
        dp_decimal.match_color(dot)
        dp_decimal.f_always.set_value(lambda: np.dot(normalized_sum, emb_point.get_center()) * 2.0 / 3.535534)
        dp_decimal.always.next_to(dot, DR, buff=SMALL_BUFF)

        self.add(shadow, emb, dot, dashed_line, dp_decimal)

        emb_point.move_to(ORIGIN + 0.01 * UP)
        for point in [m_vect.get_end(), m_ghost.get_end(), j_vect.get_end(), m_ghost.get_end()]:
            self.play(emb_point.animate.move_to(point), run_time=3)

        # Set up names
        names = VGroup(
            Text(name, font_size=36)
            for name in [
                "Michael Jordan",
                "Michael Phelps",
                "Alexis Jordan",
            ]
        )
        name_points = [
            sum_point,
            m_vect.get_end(),
            j_vect.get_end(),
        ]
        for name, point in zip(names, name_points):
            name.set_backstroke(BLACK, 3)
            direction = RIGHT + np.sign(point[1]) * UP
            name.next_to(point, direction, buff=0.1)

        # Go through names
        name = names[0].copy()
        name_ghosts = names.copy().set_fill(opacity=0.75).set_stroke(width=0)

        self.play(
            FadeIn(name, 0.5 * UP),
            Rotate(emb_point, TAU, about_point=emb_point.get_center() + 0.15 * DL, run_time=4),
        )
        self.wait()
        self.add(name_ghosts[0])
        self.play(
            Transform(name, names[1]),
            emb_point.animate.move_to(m_vect.get_end()),
            run_time=2,
        )
        self.wait()
        self.add(name_ghosts[1])
        self.play(
            Transform(name, names[2]),
            emb_point.animate.move_to(j_vect.get_end()).set_anim_args(path_arc=30 * DEGREES),
            run_time=2,
        )
        self.add(name_ghosts[2])
        self.wait()

        # Show other names
        other_point = span_line.pfp(0.45)
        other_word = Text("(Other)", font_size=36)
        other_word.set_fill(GREY_B)
        other_word.next_to(other_point, UL, buff=0)

        self.play(
            emb_point.animate.move_to(other_point),
            LaggedStart(
                FadeOut(name),
                FadeIn(other_word),
                lag_ratio=0.5,
            ),
            run_time=3
        )
        self.wait()

        # Show "yes" vs. "no" regions
        regions = FullScreenRectangle().scale(2).replicate(2)
        regions.arrange(LEFT, buff=0)
        regions[0].set_fill(GREEN_B, 0.35)
        regions[1].set_fill(RED, 0.25)
        regions.rotate(span_line.get_angle(), about_point=ORIGIN)
        regions.shift(0.85 * sum_point)

        yes_no_words = VGroup(
            Text("Yes", font_size=72).set_fill(GREEN).to_corner(UR),
            Text("No", font_size=72).set_fill(RED).to_edge(UP).shift(LEFT),
        )

        for region, word in zip(regions, yes_no_words):
            self.play(FadeIn(region), FadeIn(word))
        self.wait()


class Superposition(InteractiveScene):
    def construct(self):
        # Add undulating bubble to encompass N-dimensional space
        frame = self.frame
        bubble = self.undulating_bubble()
        bubble_label = TexText(R"$N$-dimensional\\\\ Space")
        bubble_label.set_height(1)
        bubble_label["$N$"].set_color(YELLOW)
        bubble_label.next_to(bubble, LEFT)

        self.add(bubble)
        self.add(bubble_label)

        # Preview some ideas
        ideas = VGroup(Text("Latin"), Text("Microphone"), Text("Basketball"), Text("The 1920s"))
        ideas.scale(0.75)
        vectors = VGroup()
        idea_vects = VGroup()
        vect = DOWN
        colors = [PINK, GREEN, ORANGE, BLUE]
        for idea, color in zip(ideas, colors):
            vect = rotate_vector(vect, 80 * DEGREES)
            vector = Vector(1.25 * normalize(vect))
            idea.next_to(vector.get_end(), vector.get_vector(), buff=SMALL_BUFF)
            idea_vect = VGroup(vector, idea)
            idea_vect.set_color(color)
            idea_vect.shift(bubble.get_center())
            idea_vects.add(idea_vect)

        frame.save_state()
        frame.scale(0.75)
        frame.move_to(VGroup(bubble, bubble_label))
        self.play(
            Restore(frame, run_time=7),
            LaggedStartMap(VFadeInThenOut, idea_vects, lag_ratio=0.5, run_time=5)
        )

        # Written conditions and answer
        conditions = [
            R"$90^\\circ$ apart",
            R"between $89^\\circ$ and $91^\\circ$ apart"
        ]
        task1, task2 = tasks = VGroup(
            TexText(Rf"Choose multiple vectors,\\\\ each pair {phrase}", font_size=42, alignment="")
            for phrase in conditions
        )
        task1[R"90^\\circ"].set_color(RED)
        task2[R"$89^\\circ$ and $91^\\circ$"].set_color(BLUE)
        task1.center().to_edge(UP)
        task2.move_to(task1, UL)

        maximum1, maximum2 = maxima = VGroup(
            TexText(fR"Maximum \\# of vectors: {answer}", font_size=42)
            for answer in ["$N$", R"$\\approx \\exp(\\epsilon \\cdot N)$"]
        )
        for maximum in maxima:
            maximum.next_to(tasks, DOWN, buff=LARGE_BUFF, aligned_edge=LEFT)
        maximum1["N"].set_color(YELLOW)
        maximum2["N"].set_color(YELLOW)

        # Add 3 vectors such that each pair is 90-degrees
        perp_vectors = VGroup(*map(Vector, [RIGHT, UP, OUT]))
        perp_vectors.set_shading(0.25, 0.25, 0.25)
        perp_vectors.set_submobject_colors_by_gradient(RED, GREEN, BLUE)
        elbows = VGroup(
            Elbow(width=0.1).rotate(angle, axis, about_point=ORIGIN).set_stroke(WHITE, 2)
            for angle, axis in [(0, UP), (-PI / 2, UP), (PI / 2, RIGHT)]
        )
        elbows.set_stroke(GREY_A, 2)

        perp_group = VGroup(perp_vectors, elbows)
        perp_group.rotate(-10 * DEGREES, UP)
        perp_group.rotate(20 * DEGREES, RIGHT)
        perp_group.scale(2)
        perp_group.move_to(bubble)

        self.play(
            FadeIn(task1),
            LaggedStartMap(GrowArrow, perp_vectors[:2], lag_ratio=0.5)
        )
        self.play(ShowCreation(elbows[0]))
        self.play(
            GrowArrow(perp_vectors[2]),
            LaggedStartMap(ShowCreation, elbows[1:3], lag_ratio=0.5),
        )
        self.play(
            Rotate(perp_group, -50 * DEGREES, axis=perp_vectors[1].get_vector(), run_time=15),
            Write(maximum1, time_span=(2, 4)),
        )

        # Relax the assumption
        ninety_part = task1[conditions[0]]
        cross = Cross(ninety_part)
        crossed_part = VGroup(ninety_part, cross)
        new_cond = task2[conditions[1]]
        new_cond.align_to(ninety_part, LEFT)

        pairs = VGroup(get_vector_pair(89), get_vector_pair(91))
        pairs.arrange(RIGHT)
        pairs.to_corner(UL)

        self.play(
            FadeOut(maximum1),
            ShowCreation(cross),
        )
        self.play(
            crossed_part.animate.shift(0.5 * DOWN).set_fill(opacity=0.5),
            Write(new_cond),
            LaggedStartMap(FadeIn, pairs, lag_ratio=0.25),
        )
        self.play(
            Rotate(perp_group, 50 * DEGREES, axis=perp_vectors[1].get_vector(), run_time=10)
        )

        # Struggle with 3 vectors (Sub out the title)
        three_d_label = TexText(R"3-dimensional\\\\ Space")
        three_d_label["3"].set_color(BLUE)
        three_d_label.move_to(bubble_label, UL)
        bubble_label.save_state()

        pv = perp_vectors
        pv.save_state()
        alt_vects = pv.copy()
        origin = pv[0].get_start()
        for vect in alt_vects:
            vect.rotate(5 * DEGREES, axis=normalize(np.random.random(3)), about_point=origin)

        new_vects = VGroup()
        for (v1, v2) in it.combinations(pv, 2):
            new_vects.add(Arrow(ORIGIN, v1.get_length() * normalize(v1.get_vector() + v2.get_vector()), buff=0).shift(origin))
        new_vects.set_color(YELLOW)
        new_vect = new_vects[0]

        def shake(vect):
            self.play(
                vect.animate.rotate(5 * DEGREES, RIGHT, about_point=origin),
                rate_func=lambda t: wiggle(t, 9)
            )

        self.play(
            FadeIn(three_d_label, DOWN),
            bubble_label.animate.to_edge(DOWN).set_opacity(0.5)
        )
        self.play(
            GrowArrow(new_vect),
            Transform(perp_vectors, alt_vects)
        )
        shake(new_vect)
        self.play(
            Restore(perp_vectors),
            Transform(new_vect, new_vects[1])
        )
        shake(new_vect)
        self.play(
            Transform(perp_vectors, alt_vects),
            Transform(new_vect, new_vects[2])
        )
        shake(new_vect)
        self.wait()
        self.play(
            new_vect.animate.scale(0, about_point=origin),
            ApplyMethod(perp_group.scale, 0, dict(about_point=origin), lag_ratio=0.25),
            Restore(bubble_label),
            FadeOut(three_d_label, UP),
            run_time=2
        )
        self.remove(new_vect, perp_group)

        # Stack on many vectors
        dodec = Dodecahedron()
        vertices = [face.get_center() for face in dodec]
        vectors = VGroup(Vector(vert) for vert in vertices)
        vectors.set_flat_stroke(True)
        vectors.rotate(30 * DEGREES, UR)
        for vector in vectors:
            vector.always.set_perpendicular_to_camera(self.frame)
            vector.set_color(random_bright_color(hue_range=(0.5, 0.7)))
        vectors.move_to(bubble)

        self.wait(6)
        self.play(
            FadeOut(crossed_part),
            Write(maximum2),
            Rotating(vectors, TAU, axis=UP, run_time=20),
            LaggedStartMap(VFadeIn, vectors, lag_ratio=0.5, run_time=8)
        )
        self.wait()

        # Somehow communicate exponential scaling

    def undulating_bubble(self):
        bubble = ThoughtBubble(filler_shape=(6, 3))[0][-1]
        bubble.set_stroke(WHITE, 1)
        bubble.set_fill(GREY)
        bubble.set_shading(0.5, 0.5, 0)
        bubble.to_edge(DOWN)

        points = bubble.get_points().copy()
        points -= np.mean(points, 0)

        def update_bubble(bubble):
            center = bubble.get_center()
            angles = np.apply_along_axis(angle_of_vector, 1, points)
            stretch_factors = 1.0 + 0.05 * np.sin(6 * angles + self.time)
            bubble.set_points(points * stretch_factors[:, np.newaxis])
            # bubble.move_to(center)
            bubble.set_x(0).to_edge(DOWN)

        bubble.add_updater(update_bubble)
        return bubble


class StackOfVectors(InteractiveScene):
    def construct(self):
        # Set up the big matrix
        rows = VGroup(
            NumericEmbedding(shape=(1, 9), ellipses_col=-5, value_range=(-1, 1))
            for n in range(20)
        )
        rows.arrange(DOWN)
        for row in rows:
            row.brackets[0].align_to(rows, LEFT)
            row.brackets[1].align_to(rows, RIGHT)
        rows.set_height(6)
        rows.to_edge(DOWN)
        rows[-2].become(Tex(R"\\vdots").replace(rows[-2], dim_to_match=1))
        brackets = NumericEmbedding(shape=(20, 9)).brackets
        brackets.set_height(rows.get_height() + MED_SMALL_BUFF)
        brackets[0].next_to(rows, LEFT, SMALL_BUFF)
        brackets[1].next_to(rows, RIGHT, SMALL_BUFF)

        top_brace = Brace(rows[0], UP)
        top_label = top_brace.get_text("100-dimensional")
        side_brace = Brace(brackets, LEFT)
        side_label = side_brace.get_text("10,000\\nvectors")

        self.play(
            GrowFromCenter(top_brace),
            FadeIn(top_label, lag_ratio=0.1),
            LaggedStartMap(FadeIn, rows, shift=0.25 * DOWN, lag_ratio=0.1, run_time=3),
            *map(GrowFromCenter, brackets)
        )
        self.play(
            LaggedStart(
                (RandomizeMatrixEntries(row)
                for row in rows[:-2]),
                lag_ratio=0.05,
            )
        )
        self.wait()

        # Label first vector
        self.play(
            GrowFromCenter(side_brace),
            FadeIn(side_label, lag_ratio=0.1),
        )
        self.wait(4)


class ShowAngleRange(InteractiveScene):
    def construct(self):
        # Test
        angle_tracker = ValueTracker(10)
        vect_pair = always_redraw(lambda: get_vector_pair(angle_tracker.get_value(), length=3, colors=(RED, GREEN)))

        self.add(vect_pair)
        self.play(
            angle_tracker.animate.set_value(180),
            run_time=8,
        )
        self.wait()
        self.play(
            angle_tracker.animate.set_value(95),
            run_time=3
        )
        self.wait()


class MLPFeatures(InteractiveScene):
    def construct(self):
        # Add neurons
        radius = 0.15
        layer1, layer2 = layers = VGroup(
            Dot(radius=radius).get_grid(n, 1, buff=radius / 2)
            for n in [8, 16]
        )
        layer2.arrange(DOWN, buff=radius)
        layers.arrange(RIGHT, buff=3.0)
        layers.to_edge(LEFT, buff=LARGE_BUFF)
        layers.set_stroke(WHITE, 1)
        for neuron in layer1:
            neuron.set_fill(opacity=random.random())
        layer2.set_fill(opacity=0)

        self.add(layers)

        # Add connections
        connections = get_network_connections(layer1, layer2)
        self.add(connections)

        # Show single-neuron features
        features = iter([
            "Table",
            "Slang",
            "AM Radio",
            "Humble",
            "Notebook",
            "Transparent",
            "Duration",
            "Madonna",
            "Mirror",
            "Pole Vaulting",
            "Albert Einstein",
            "Authentic",
            "Scientific",
            "Passionate",
            "Bell Laboratories",
            "Uzbekistan",
            "Umbrella",
            "Immanuel Kant",
            "Baroque Music",
            "Intense",
            "Clock",
            "Water skiing",
            "Ancient Egypt",
            "Ambiguous",
            "Volume",
            "Alexander the Great",
            "Innovative",
            "Religious",
        ])

        last_neuron = VGroup()
        last_feature_label = VGroup()
        for neuron in layer2[:15]:
            feature_label = Text(next(features), font_size=36)
            feature_label.next_to(neuron, buff=SMALL_BUFF)

            self.play(
                FadeOut(last_feature_label),
                FadeIn(feature_label),
                last_neuron.animate.set_fill(opacity=0),
                neuron.animate.set_fill(opacity=1),
            )

            last_neuron = neuron
            last_feature_label = feature_label

        # Show polysemantic features
        brace = Brace(layer2, RIGHT)

        def to_random_state(layer):
            for dot in layer.generate_target():
                dot.set_fill(opacity=random.random())
            return MoveToTarget(layer)

        self.play(
            feature_label.animate.scale(48 / 36).next_to(brace, RIGHT),
            GrowFromCenter(brace),
            to_random_state(layer2),
        )
        self.wait()
        for n in range(12):
            feature_label = Text(next(features))
            feature_label.next_to(brace, RIGHT)
            self.play(
                FadeOut(last_feature_label),
                FadeIn(feature_label),
                to_random_state(layer2),
            )
            self.wait(0.5)

            last_feature_label = feature_label


class BreakDownThreeSteps(BasicMLPWalkThrough):
    def construct(self):
        # Add four vectors, spaced apart
        vectors = VGroup(
            NumericEmbedding(length=n)
            for n in [8, 16, 16, 8]
        )
        vectors.set_height(6)
        vectors.arrange(RIGHT, buff=3.5)
        vectors[2].shift(1.1 * LEFT)
        vectors[1].shift(0.2 * LEFT)
        vectors.shift(DOWN)
        for e1, e2 in zip(vectors[1].get_entries(), vectors[2].get_entries()):
            e2.set_value(max(e1.get_value(), 0))

        # Add arrows between them
        arrows = VGroup(
            Arrow(v1, v2)
            for v1, v2 in zip(vectors, vectors[1:])
        )
        arrows.shift(DOWN)

        E_sym = Tex(R"\\vec{\\textbf{E}}")
        E_sym.next_to(arrows[0], LEFT).shift(0.1 * UP)

        for vect in vectors:
            vect.scale(0.75)
            vect.shift(0.25 * UP)

        # Put matrices on outer two
        up_proj, down_proj = matrices = VGroup(
            WeightMatrix(shape=(12, 6)),
            WeightMatrix(shape=(6, 11)),
        )
        matrices.scale(0.25)
        for arrow, mat in zip(arrows[::2], matrices):
            mat.next_to(arrow, UP)

        # Put ReLU graph on the middle
        axes = Axes((-3, 3), (0, 3))
        graph = axes.get_graph(lambda x: max(x, 0))
        graph.set_color(BLUE)
        relu = VGroup(axes, graph)
        relu.match_width(arrows[1])
        relu.next_to(arrows[1], UP)

        # Full box
        box = SurroundingRectangle(VGroup(arrows, matrices), buff=1.0)
        box.set_stroke(WHITE, 2)
        box.set_fill(GREY_E, 1)
        title = Text("Multilayer Perceptron", font_size=60)
        title.next_to(box, UP, SMALL_BUFF)

        self.add(box, title)

        # Animate them all in
        for matrix in matrices:
            matrix.brackets.save_state()
            matrix.brackets.stretch(0, 0).set_opacity(0)

        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5),
            FadeIn(up_proj.get_rows(), lag_ratio=0.1, time_span=(0.0, 1.5)),
            FadeIn(down_proj.get_rows(), lag_ratio=0.1, time_span=(1.5, 3.0)),
            Restore(up_proj.brackets, time_span=(0.0, 1.5)),
            Restore(down_proj.brackets, time_span=(1.5, 3.0)),
            Write(relu, time_span=(1, 2)),
            run_time=3
        )
        self.wait()

        # Show row replacement on the first
        n, m = up_proj.shape
        n_rows_shown = 8
        R_labels = VGroup(
            Tex(R"\\vec{\\textbf{R}}_{" + str(n) + "}")
            for n in [*range(n_rows_shown - 1), "n-1"]
        )
        R_labels[-2].become(Tex(R"\\vdots").replace(R_labels[-2], dim_to_match=1))
        R_labels.arrange(DOWN, buff=0.5)
        R_labels.match_height(up_proj)
        R_labels.move_to(up_proj)
        h_lines = VGroup(
            Line(up_proj.get_brackets()[0], R_labels, buff=0.1),
            Line(R_labels, up_proj.get_brackets()[1], buff=0.1),
        )
        h_lines.set_stroke(GREY_A, 2)
        row_labels = VGroup(
            VGroup(R_label, h_lines.copy().match_y(R_label))
            for R_label in R_labels
        )
        row_labels.set_color(YELLOW)
        row_matrix = VGroup(
            up_proj.get_brackets().copy(),
            row_labels
        )

        self.play(
            FadeOut(up_proj.get_rows(), lag_ratio=0.1),
            FadeIn(row_labels, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            row_labels[0][0].copy().animate.scale(2).next_to(title, UL).shift(2 * LEFT).set_opacity(0),
        )
        self.wait()

        # Show the neurons
        dots = VGroup(
            Dot().set_fill(opacity=random.random()).move_to(entry)
            for entry in vectors[2].get_columns()[0]
        )
        for dot in dots:
            dot.match_x(dots[0])
        dots.set_stroke(WHITE, 1)
        self.play(Write(dots))
        self.wait()

        # Show column replacement on the second
        col_matrix = self.get_col_matrix(down_proj, 8)
        col_labels = col_matrix[1]
        col_labels.set_color(RED_B)

        self.play(
            FadeOut(down_proj.get_columns(), lag_ratio=0.1),
            FadeIn(col_labels, lag_ratio=0.1),
        )
        self.wait()
        self.play(
            col_labels[0][0].copy().animate.scale(2).next_to(title, UR).shift(2 * RIGHT).set_opacity(0),
        )
        self.wait()

        return
        #### Trash ####

        vectors[0].next_to(arrows[0], LEFT)
        vectors[0].align_to(vectors[1], DOWN)
        self.play(FadeIn(vectors[0]))
        for i in (0, 1):
            self.play(
                FadeTransform(vectors[i].copy(), vectors[i + 1]),
                rate_func=linear,
            )


class SuperpositionVectorBundle(InteractiveScene):
    def construct(self):
        # Setup
        frame = self.frame
        axes = ThreeDAxes(z_range=(-3, 3))
        axes.scale(0.5)
        vects = VGroup(
            self.get_new_vector(v)
            for v in np.identity(3)
        )

        frame.reorient(23, 71, 0, (0.0, 0.0, 0.5), 3.5)
        frame.add_ambient_rotation(4 * DEGREES)
        self.add(frame)
        self.add(axes)
        self.add(vects)
        self.wait(2)

        # Add a new vector
        n_vects = 10
        for n in range(n_vects):
            new_vect = self.get_new_vector(normalize(np.random.uniform(-1, 1, 3)))
            # self.play(GrowArrow(new_vect))
            vects.add(new_vect)
            self.space_out_vectors(vects, run_time=3 + 0.5 * n)
        self.wait(5)

        # Use tensor flow to repeatedly cram more vectors into a space
        pass

    def get_new_vector(self, coords, color=None, opacity=0.9):
        if color is None:
            color = random_bright_color(hue_range=(0.4, 0.6), luminance_range=(0.5, 0.9))
        vect = Vector(coords, thickness=2.0)
        vect.set_fill(color, opacity=opacity, border_width=2)
        vect.always.set_perpendicular_to_camera(self.frame)
        return vect

    def space_out_vectors(self, vects, run_time=4, learning_rate=0.01):
        num_vectors = len(vects)
        ends = np.array([v.get_end() for v in vects])
        matrix = torch.from_numpy(ends)
        matrix.requires_grad_(True)

        optimizer = torch.optim.Adam([matrix], lr=learning_rate)
        dot_diff_cutoff = 0.01
        id_mat = torch.eye(num_vectors, num_vectors)

        def update_vects(vects):
            optimizer.zero_grad()
            dot_products = matrix @ matrix.T
            # Punish deviation from orthogonal
            diff = dot_products - id_mat
            # loss = (diff.abs() - dot_diff_cutoff).relu().sum()
            loss = diff.pow(6).sum()

            # Extra incentive to keep rows normalized
            loss += num_vectors * diff.diag().pow(2).sum()
            loss.backward()
            optimizer.step()

            for vect, arr in zip(vects, matrix):
                vect.put_start_and_end_on(ORIGIN, arr.detach().numpy())

        self.play(UpdateFromFunc(vects, update_vects, run_time=run_time))


# Some old stubs


class ClassicNeuralNetworksPicture(InteractiveScene):
    def construct(self):
        pass


class ShowBiasBakedIntoWeightMatrix(LastTwoChapters):
    def construct(self):
        # Add initial blocks
        frame = self.frame
        square = Square(2.0)
        att_icon = self.get_att_icon(square)
        att_icon.set_stroke(WHITE, 1, 0.5)
        mlp_icon = self.get_mlp_icon(square, layer_buff=1.0)
        lnm_icon = self.get_layer_norm_icon()
        lnm_icon.match_height(mlp_icon)

        att_block = self.get_block(att_icon, "Attention", "604M Parameters", color=YELLOW)
        mlp_block = self.get_block(mlp_icon, "MLP", "1.2B Parameters", color=BLUE)
        lnm_block = self.get_block(lnm_icon, "Layer Norm", "49K Parameters", color=GREY_B)

        blocks = VGroup(att_block, mlp_block, lnm_block)
        blocks.arrange(RIGHT, buff=1.5)

        lil_wrapper = self.get_layer_wrapper(blocks[:2].copy())
        big_wrapper = self.get_layer_wrapper(blocks)

        self.add(lil_wrapper, blocks[:2])
        frame.match_x(blocks[:2])
        self.wait()
        self.play(
            frame.animate.match_x(blocks),
            ReplacementTransform(lil_wrapper, big_wrapper),
            FadeIn(lnm_block, RIGHT),
        )
        self.wait()
        self.play(FlashAround(lnm_block[2], run_time=3, time_width=2))
        self.wait()

    def get_layer_norm_icon(self):
        axes1, axes2 = all_axes = VGroup(
            Axes((-4, 4), (0, 1, 0.25))
            for x in range(2)
        )
        all_axes.set_shape(1.5, 0.5)
        all_axes.arrange(DOWN, buff=1.0)
        graph1 = axes1.get_graph(lambda x: 0.5 * norm.pdf(0.5 * x - 0.5))
        graph2 = axes2.get_graph(lambda x: 1.5 * norm.pdf(x))
        graph1.set_stroke(BLUE).set_fill(BLUE, 0.25)
        graph2.set_stroke(BLUE).set_fill(BLUE, 0.25)
        arrow = Arrow(axes1, axes2, buff=0.1)

        return VGroup(axes1, graph1, arrow, axes2, graph2)

    def get_layer_wrapper(self, blocks):
        beige = "#F5F5DC"
        rect = self.get_block(blocks, color=beige, buff=0.5, height=4)[0]
        wrapped_arrow = self.get_wrapped_arrow(rect)
        multiple = Tex(R"\\times 96")
        multiple.next_to(wrapped_arrow, UP)

        arrows = VGroup()
        for b1, b2 in zip(blocks, blocks[1:]):
            arrows.add(Arrow(b1[0], b2[0], buff=0.1))

        return VGroup(rect, arrows, wrapped_arrow, multiple)

    def get_block(
        self, content,
        upper_label="",
        lower_label="",
        upper_font_size=42,
        lower_font_size=36,
        buff=0.25,
        height=2,
        color=BLUE,
        stroke_width=3,
        fill_opacity=0.2
    ):
        block = SurroundingRectangle(content, buff=buff)
        block.set_height(height, stretch=True)
        block.round_corners(radius=0.25)
        block.set_stroke(color, 3)
        block.set_fill(color, fill_opacity)

        low_label = Text(lower_label, font_size=lower_font_size)
        low_label.next_to(block, DOWN, MED_SMALL_BUFF)
        top_label = Text(upper_label, font_size=upper_font_size)
        top_label.next_to(block, UP, MED_SMALL_BUFF)

        return VGroup(block, content, low_label, top_label)

    def get_wrapped_arrow(self, big_block, buff=0.75, color=GREY_B, stroke_width=4):
        vertices = [
            big_block.get_corner(RIGHT),
            big_block.get_corner(RIGHT) + buff * RIGHT,
            big_block.get_corner(UR) + buff * UR,
            big_block.get_corner(UL) + buff * UL,
            big_block.get_corner(LEFT) + buff * LEFT,
            big_block.get_corner(LEFT),
        ]
        line = Polygon(*vertices)
        line.round_corners()
        line.set_points(line.get_points()[:-2, :])
        line.set_stroke(color, stroke_width)
        tip = ArrowTip().move_to(line.get_end(), RIGHT)
        tip.set_color(color)
        line.add(tip)
        return line


class AlmostOrthogonal(InteractiveScene):
    def construct(self):
        pass`,
    annotations: {
      4: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      5: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      8: "LastTwoChapters extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      9: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      16: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      17: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      19: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      28: "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D.",
      30: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      33: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      34: "Smoothly animates the camera to a new orientation over the animation duration.",
      35: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      36: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      37: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      39: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      45: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      46: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      47: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      49: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      50: "Smoothly animates the camera to a new orientation over the animation duration.",
      51: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      52: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      53: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      61: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      62: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      63: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      64: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      66: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      67: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      68: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      69: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      71: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      72: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      75: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      79: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      88: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      89: "FadeOut transitions a mobject from opaque to transparent.",
      90: "FadeOut transitions a mobject from opaque to transparent.",
      91: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      92: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      97: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      103: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      107: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      110: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      111: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      112: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      113: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      114: "GrowArrow animates an arrow growing from its start point to full length.",
      116: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      133: "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves.",
      182: "Class AltLastTwoChapters inherits from LastTwoChapters.",
      183: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      214: "Class MLPIcon inherits from LastTwoChapters.",
      215: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      230: "MLPStepsPreview extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      231: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      377: "MatricesVsIntuition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      378: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      390: "BasicMLPWalkThrough extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      393: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1758: "NonlinearityOfLanguage extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1759: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1924: "Superposition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1925: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2136: "StackOfVectors extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2137: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2183: "ShowAngleRange extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2184: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2202: "MLPFeatures extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2203: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2299: "Class BreakDownThreeSteps inherits from BasicMLPWalkThrough.",
      2300: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2445: "SuperpositionVectorBundle extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2446: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2515: "ClassicNeuralNetworksPicture extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2516: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2520: "Class ShowBiasBakedIntoWeightMatrix inherits from LastTwoChapters.",
      2521: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2625: "AlmostOrthogonal extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2626: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/network_flow.py"] = {
    description: "Data flow through the full transformer network. Visualizes how information passes through embedding, attention, MLP, and layer norm stages end-to-end.",
    code: `import torch
from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *


class HighLevelNetworkFlow(InteractiveScene):
    # example_text = "To date, the cleverest thinker of all time was blank"
    use_words = False
    # possible_next_tokens = [
    #     (" the", 0.0882),
    #     (" probably", 0.0437),
    #     (" John", 0.0404),
    #     (" Sir", 0.0366),
    #     (" Albert", 0.0363),
    #     (" Ber", 0.0331),
    #     (" a", 0.029),
    #     (" Isaac", 0.0201),
    #     (" undoubtedly", 0.0158),
    #     (" arguably", 0.0133),
    #     (" Im", 0.0116),
    #     (" Einstein", 0.0113),
    #     (" Ludwig", 0.0104),
    # ]
    hide_block_labels = False
    block_to_title_direction = UP

    example_text = "The Computer History Museum is located in Mountain"
    possible_next_tokens = [
        ("Mountain", 0.706),
        ("the", 0.128),
        ("San", 0.078),
        ("Seattle", 0.006),
        ("New", 0.003),
        ("Palo", 0.002),
        ("Google", 0.002),
        ("southern", 0.002),
    ]

    def setup(self):
        super().setup()
        self.set_floor_plane("xz")
        self.camera.light_source.move_to([-10, 10, 15])
        self.camera.light_source.add_updater(lambda m: m.set_z(self.frame.get_z() + 10))
        self.layers = VGroup()
        self.blocks = Group()
        self.mlps = Group()

    def construct(self):
        frame = self.frame

        # Embedding
        self.show_initial_text_embedding()

        # Passing through some layers
        self.progress_through_attention_block(target_frame_x=-2)
        self.progress_through_mlp_block(
            show_one_by_one=True,
            sideview_orientation=(-78, -10, 0),
        )

        # # Temporary
        # block = self.blocks[-1]
        # nodes = self.mlps[-1][0]
        # lines = self.mlps[-1][1]
        # lines.save_state()
        # lines.set_stroke(width=0.5, opacity=0.5)
        # self.play(
        #     FadeOut(self.token_blocks),
        #     FadeOut(self.token_arrows),
        #     FadeOut(self.final_word_question),
        #     FadeOut(self.blocks[0]),
        #     FadeOut(self.layers),
        #     FadeOut(block[0]),
        # )
        # self.play(
        #     self.frame.animate.reorient(-65, -8, 0, (-0.77, 1.7, 7.26), 12.57),
        #     Restore(lines, lag_ratio=0.01),
        #     run_time=4
        # )
        # self.play(self.frame.animate.reorient(-15, -8, 0, (-0.77, 1.7, 7.26), 12.57), run_time=4)

        #
        orientation = frame.get_euler_angles() / DEGREES
        mlp_kw = dict(sideview_orientation=orientation, final_orientation=orientation)
        att_kw = dict(target_orientation=orientation)
        for x in [-3, -4, -5, -6, -7]:
            self.progress_through_attention_block(target_frame_x=x, **att_kw)
            self.progress_through_mlp_block(**mlp_kw)

        # Show how it ends
        self.remove_mlps()
        self.mention_repetitions()
        self.focus_on_last_layer()
        self.show_unembedding()

    def get_embedding_array(
        self,
        shape=(9, 10),
        height=4,
        dots_index=-4,
        buff_ratio=0.4,
        bracket_color=GREY_B,
        backstroke_width=3,
        add_background_rectangle=False,
    ):
        return EmbeddingArray(
            shape=shape,
            height=height,
            dots_index=dots_index,
            buff_ratio=buff_ratio,
            bracket_color=bracket_color,
            backstroke_width=backstroke_width,
            add_background_rectangle=add_background_rectangle,
        )

    def swap_embedding_for_dots(self, embedding_array, dots_index=-4):
        embedding_array.swap_embedding_for_dots(dots_index)

    def get_next_layer_array(self, embedding_array, z_buff=3.0):
        next_array = embedding_array.copy()
        embeddings = [part for part in next_array if isinstance(part, NumericEmbedding)]
        for embedding in embeddings:
            for entry in embedding.get_entries():
                entry.set_value(random.uniform(*embedding.value_range))
        next_array.shift(z_buff * OUT)
        return next_array

    def get_block(
        self,
        layer,
        depth=2.0,
        buff=1.0,
        size_buff=1.0,
        color=GREY_E,
        opacity=0.75,
        shading=(0.25, 0.1, 0.0),
        title="Attention",
        title_font_size=96,
        title_backstroke_width=3,
    ):
        # Block
        body = Cube(color=color, opacity=opacity)
        body.deactivate_depth_test()
        width, height = layer.get_shape()[:2]
        body.set_shape(width + size_buff, height + size_buff, depth)
        body.set_shading(0.5, 0.5, 0.0)
        body.next_to(layer, OUT, buff=buff)
        body.sort(lambda p: np.dot(p, [-1, 1, 1]))

        title = Text(title, font_size=title_font_size)
        title.set_backstroke(BLACK, title_backstroke_width)
        title.next_to(body, self.block_to_title_direction, buff=0.1)
        block = Group(body, title)
        block.body = body
        block.title = title
        if self.hide_block_labels:
            title.set_opacity(0)

        return block

    def show_initial_text_embedding(self, word_scale_factor=0.6, bump_first=False):
        # Mention next word prediction task
        phrase = Text(self.example_text)
        phrase.set_max_width(FRAME_WIDTH - 1)
        if self.use_words:
            words = break_into_words(phrase)
            rects = get_piece_rectangles(words)
        else:
            words = break_into_tokens(phrase)
            rects = get_piece_rectangles(
                words, leading_spaces=True, h_buff=0
            )

        words.remove(words[-1])
        q_marks = Text("???")
        rects[-1].set_color(YELLOW)
        q_marks.next_to(rects[-1], DOWN)

        big_rect = Rectangle()
        big_rect.replace(rects[:-1], stretch=True)
        big_rect.set_stroke(GREY_B, 2)
        arrow = Arrow(big_rect.get_top(), rects[-1].get_top(), path_arc=-120 * DEGREES)
        arrow.scale(0.5, about_edge=DR)

        self.play(ShowIncreasingSubsets(words, run_time=1))
        self.add(rects[-1])
        self.play(LaggedStart(
            FadeIn(big_rect),
            ShowCreation(arrow),
            Write(q_marks),
            lag_ratio=0.3,
        ))
        self.wait()
        self.play(
            FadeOut(big_rect),
            LaggedStart(*(
                DrawBorderThenFill(rect)
                for rect in rects[:-1]
            ), lag_ratio=0.02),
            LaggedStart(*(
                token.animate.match_color(rect)
                for token, rect in zip(words, rects)
            )),
            FadeOut(arrow)
        )
        self.wait()

        # Label the tokens
        token_label = Text("Tokens", font_size=72)
        token_label.to_edge(UP)
        arrows = VGroup(
            Arrow(token_label.get_bottom(), rect.get_top()).match_color(rect)
            for rect in rects[:-1]
        )

        self.play(FadeIn(token_label, UP))
        self.play(LaggedStartMap(VFadeInThenOut, arrows, lag_ratio=0.25, run_time=4))
        self.play(FadeOut(token_label, DOWN))

        # Show words into vectors
        layer = self.get_embedding_array(
            shape=(10, len(words)),
            dots_index=None,
        )
        vectors = layer.embeddings

        blocks = VGroup(*(VGroup(rect, token) for rect, token in zip(rects, words)))
        q_group = VGroup(rects[-1], q_marks)
        blocks.target = blocks.generate_target()
        for block, vector in zip(blocks.target, vectors):
            block.scale(word_scale_factor)
            block.next_to(layer, UP, buff=1.5)
            block.match_x(vector)

        arrows = VGroup(*(
            Arrow(block, vect, stroke_width=3)
            for block, vect in zip(blocks.target, vectors)
        ))
        word_to_index = dict(zip(self.example_text.split(" "), it.count()))
        # self.swap_embedding_for_dots(layer, word_to_index.get("...", -4))

        if bump_first:
            blocks.target[0].next_to(blocks.target[1], LEFT, buff=0.1)

        self.play(
            self.frame.animate.move_to(1.0 * UP),
            MoveToTarget(blocks),
            q_group.animate.scale(word_scale_factor).next_to(blocks.target, RIGHT, aligned_edge=UP),
            LaggedStartMap(FadeIn, vectors, shift=0.5 * DOWN),
            LaggedStartMap(GrowFromCenter, arrows),
            Write(layer.dots)
        )
        # self.play(Write(layer.brackets))
        self.wait()

        self.token_blocks = blocks
        self.token_arrows = arrows
        self.final_word_question = VGroup(rects[-1], q_marks)

        self.layers.add(layer)

    def progress_through_attention_block(
        self,
        depth=2.0,
        target_orientation=(-40, -15, 0),
        target_frame_height=14,
        target_frame_x=-2,
        target_frame_y=0,
        attention_anim_run_time=5,
        label_text="Attention"
    ):
        # self.embed()
        # # Test
        # self.frame.clear_updaters()
        # attention_anim_run_time = 16
        # target_orientation = (-53, -16, 0, (-1.08, -0.57, 1.12), 11.27)
        # self.camera.light_source.set_z(10)

        layer = self.layers[-1]
        layer.brackets.set_fill(opacity=1)
        layer.save_state()
        block = self.get_block(layer, title=label_text, depth=depth)
        z_diff = block.get_z() - layer.get_z()
        block_opacity = block.body[0].get_opacity()
        block.body[0].set_opacity(0)
        block.title.set_backstroke(BLACK, 5)
        new_layer = layer.copy()
        new_layer.match_z(block)
        new_layer.set_backstroke(BLACK, 1)

        self.frame.target = self.frame.generate_target()
        self.frame.target.reorient(*target_orientation)
        self.frame.target.set_height(target_frame_height)
        self.frame.target.set_x(target_frame_x)
        self.frame.target.set_y(target_frame_y)

        self.play(
            MoveToTarget(self.frame),
            LaggedStart(
                layer.animate.set_opacity(0.25),
                FadeIn(block),
                TransformFromCopy(layer, new_layer),
                (self.blocks[-1].title if len(self.blocks) > 0 else VGroup()).animate.set_opacity(0.25),
                lag_ratio=0.3
            ),
            self.token_arrows.animate.set_opacity(0.1),
            run_time=2
        )
        self.play_simple_attention_animation(
            new_layer,
            run_time=attention_anim_run_time,
            added_anims=[self.frame.animate.reorient(0, -15, 0, (-0.07, 0.45, 2.03), 9.25)]
        )

        # Take new layer out of block
        self.add(*block.body, block.title, new_layer)
        new_z = block.get_z() + z_diff
        self.play(
            block.body[0].animate.set_opacity(block_opacity),
            new_layer.animate.set_z(new_z),
            self.frame.animate.set_z(new_z),
            Restore(layer),
        )
        self.add(block, new_layer)

        if False:
            # Highlight example
            index = 4
            highlight = new_layer[0][index].copy()
            highlight.set_backstroke(BLACK, 3)
            highlight.set_fill(border_width=1)
            rect = SurroundingRectangle(highlight)
            rect.set_stroke(TEAL, 3)
            new_arrow = Arrow(self.token_blocks[index].get_bottom(), rect.get_top(), tip_angle=45 * DEGREES)
            new_arrow.set_fill(GREY_A, border_width=4)

            self.add(block.body, new_layer, block.title),
            self.play(LaggedStart(
                self.frame.animate.reorient(0, -14, 0, (-0.18, 0.73, 3.26), 8.31),
                block.body.animate.set_color(BLACK).set_shading(0.1, 0.1, 0.5),
                block.title.animate.set_opacity(0.1),
                new_layer.animate.fade(0.75).set_stroke(width=0),
                self.token_blocks[index].animate.scale(2, about_edge=DOWN),
                self.token_blocks[:index].animate.shift(0.35 * LEFT),
                FadeIn(highlight),
                ShowCreation(rect),
                FadeIn(new_arrow, scale=4),
                run_time=3
            ))
            self.wait()

        self.blocks.add(block)
        self.layers.add(new_layer)

    def play_simple_attention_animation(self, layer, run_time=5, added_anims=[]):
        arc_groups = VGroup()
        for _ in range(3):
            for n, e1 in enumerate(layer.embeddings):
                arc_group = VGroup()
                for e2 in layer.embeddings[n + 1:]:
                    sign = (-1)**int(e2.get_x() > e1.get_x())
                    arc_group.add(Line(
                        e1.get_top(), e2.get_top(),
                        path_arc=sign * PI / 3,
                        stroke_color=random_bright_color(hue_range=(0.1, 0.3)),
                        stroke_width=5 * random.random()**5,
                    ))
                arc_group.shuffle()
                if len(arc_group) > 0:
                    arc_groups.add(arc_group)

        self.play(
            LaggedStart(*(
                AnimationGroup(
                    LaggedStartMap(VShowPassingFlash, arc_group.copy(), time_width=2, lag_ratio=0.15),
                    LaggedStartMap(ShowCreationThenFadeOut, arc_group, lag_ratio=0.15),
                )
                for arc_group in arc_groups
            ), lag_ratio=0.0),
            LaggedStartMap(RandomizeMatrixEntries, layer.embeddings, lag_ratio=0.0),
            *added_anims,
            run_time=run_time
        )
        self.add(layer)

    def progress_through_mlp_block(
        self,
        n_neurons=20,
        # depth=3.0,
        depth=4.0,
        buff=1.0,
        dot_buff_ratio=0.2,
        neuron_color=GREY_C,
        neuron_shading=(0.25, 0.75, 0.2),
        sideview_orientation=(-60, -5, 0),
        final_orientation=(-51, -18, 0),
        show_one_by_one=False,
        # label_text="Multilayer\\nPerceptron",
        # title_font_size=72,
        label_text="Feedforward",
        title_font_size=90,
    ):
        # MLP Test
        layer = self.layers[-1]
        block = self.get_block(
            layer,
            depth=depth,
            buff=buff,
            size_buff=1.0,
            title=label_text,
            title_font_size=title_font_size,
        )

        # New layer
        new_layer = self.get_next_layer_array(layer)
        new_layer.next_to(block.body, OUT, buff)

        # Neurons
        def get_neurons(points):
            neurons = DotCloud(points, radius=0.1)
            neurons.make_3d()
            neurons.set_glow_factor(0)
            neurons.set_shading(*neuron_shading)
            neurons.set_color(neuron_color)
            neurons.set_opacity(np.random.uniform(0.5, 1, len(points)))
            return neurons

        all_neuron_points = np.zeros((0, 3))
        neuron_clusters = Group()
        connections = VGroup()
        y_min = block.body.get_y(DOWN) + SMALL_BUFF
        y_max = block.body.get_y(UP) - SMALL_BUFF
        block_z = block.body.get_z()
        for embedding in layer.embeddings:
            l1_points = np.array([e.get_center() for e in embedding.get_columns()[0]])
            l3_points = l1_points.copy()
            l1_points[:, 2] = block_z - 0.4 * depth
            l3_points[:, 2] = block_z + 0.4 * depth
            x0, y0, z0 = embedding.get_center()
            l2_points = np.array([
                [x0, y, block_z]
                for y in np.linspace(y_min, y_max, n_neurons)
            ])
            new_points = np.vstack([l1_points, l2_points, l3_points])
            all_neuron_points = np.vstack([all_neuron_points, new_points])
            neuron_clusters.add(get_neurons(new_points))
            # Lines
            weights = VGroup(*(
                Line(
                    p1, p2,
                    buff=0.1,
                    stroke_width=3,
                    stroke_opacity=random.random(),
                    stroke_color=value_to_color(random.uniform(-10, 10))
                )
                for points1, points2 in [
                    [l1_points, l2_points],
                    [l2_points, l3_points],
                ]
                for p1, p2 in it.product(points1, points2)
                if random.random() < 0.2
            ))
            connections.add(weights)
        neurons = get_neurons(all_neuron_points)
        connections.apply_depth_test()
        connections.set_flat_stroke(False)

        # Flow through layer
        if show_one_by_one:
            networks = Group(*(Group(cluster, lines.copy()) for cluster, lines in zip(neuron_clusters, connections)))
            last_network = VectorizedPoint()
            emb_pairs = list(zip(layer.embeddings, new_layer.embeddings))
            index = 4
            block.title.rotate(PI / 2, DOWN)
            self.play(
                self.blocks[-1].title.animate.set_opacity(0.25),
                FadeIn(block.title, time_span=(0, 1)),
                self.frame.animate.reorient(*sideview_orientation),
                Write(networks[index][1]),
                FadeIn(networks[index][0]),
                FadeTransform(emb_pairs[index][0].copy(), emb_pairs[index][1]),
                run_time=3
            )
            lag_kw = dict(lag_ratio=0.5, run_time=9)
            self.remove(block.title)
            self.play(
                LaggedStart(*(
                    FadeTransform(e1.copy(), e2)
                    for e1, e2 in [*emb_pairs[:index], *emb_pairs[index + 1:]]
                ), **lag_kw),
                LaggedStart(*(
                    Write(network[1])
                    for network in [*networks[:index], *networks[index + 1:]]
                ), **lag_kw),
                LaggedStart(*(
                    FadeIn(network[0])
                    for network in [*networks[:index], *networks[index + 1:]]
                ), **lag_kw),
                self.frame.animate.reorient(0, -37, 0, (-1.08, 2.29, 7.99), 12.27),
                block.title.animate.rotate(PI / 2, UP),
                run_time=15,
            )
            self.remove(networks)
            self.add(neurons, connections, block.body, block.title, new_layer)
            self.play(
                FadeIn(block.body),
                FadeIn(new_layer),
                FadeOut(new_layer.embeddings.copy()),
                # self.frame.animate.reorient(*final_orientation).match_z(new_layer),
                self.frame.animate.reorient(-48, -12, 0).match_z(new_layer),
                run_time=2
            )
        else:
            self.play(
                FadeIn(block.title, time_span=(0, 1)),
                self.blocks[-1].title.animate.set_opacity(0.25),
                self.frame.animate.reorient(*sideview_orientation),
                FadeIn(neurons, time_span=(0, 1)),
                Write(connections, stroke_width=3),
                TransformFromCopy(layer, new_layer),
                run_time=3
            )
            self.add(block.body, block.title, new_layer)
            self.play(
                self.frame.animate.reorient(*final_orientation).match_z(new_layer),
                FadeIn(block.body),
                run_time=2,
            )

        # Aggregate
        self.mlps.add(Group(neurons, connections))
        self.blocks.add(block)
        self.layers.add(new_layer)

    def remove_mlps(self):
        self.remove(self.mlps)

    def mention_repetitions(self, depth=8):
        # Mention repetition
        frame = self.frame
        layer = self.layers[-1]
        block = self.blocks[-1].body

        thin_blocks = block.replicate(2)
        thin_blocks.set_depth(1.0, stretch=True)

        dots = Tex(".....", font_size=250)
        brace = Brace(dots, UP)
        brace_text = brace.get_text("Many\\nrepetitions")
        rep_label = Group(dots, brace, brace_text)
        rep_label.set_width(depth)
        rep_label.rotate(PI / 2, DOWN)
        rep_label.next_to(layer, OUT, buff=3.0)
        rep_label.align_to(ORIGIN, DOWN)

        thin_blocks[0].align_to(brace, IN)
        thin_blocks[1].align_to(brace, OUT)
        dots.scale(0.5)
        VGroup(brace, brace_text).next_to(thin_blocks, UP)

        final_layer = self.get_next_layer_array(layer)
        final_layer.set_z(rep_label.get_z(OUT) + 1)
        final_layer.save_state()
        final_layer.become(layer)
        final_layer.set_opacity(0)

        self.play(
            frame.animate.reorient(-57, -8, 0, (-6.59, 1.2, 57.84), 30),
            FadeIn(thin_blocks, lag_ratio=0.1),
            GrowFromCenter(brace),
            Write(brace_text, time_span=(1, 2)),
            Write(dots),
            run_time=3
        )
        self.play(
            Restore(final_layer, run_time=2)
        )
        self.wait()

        self.rep_label = rep_label
        self.blocks.add(*thin_blocks)
        self.layers.add(final_layer)

    def focus_on_last_layer(self):
        # Last Layer
        layer = self.layers[-1]
        rect = BackgroundRectangle(layer)
        rect.set_fill(BLACK, 0.5)
        rect.scale(3)

        self.rep_label.target = self.rep_label.generate_target()
        self.rep_label.target[1:].set_opacity(0)
        self.rep_label.target[0].set_opacity(0.5)

        target_frame_center = layer.get_center() + 4 * RIGHT + UP

        self.add(self.rep_label, rect, layer)
        self.play(
            FadeIn(rect),
            MoveToTarget(self.rep_label),
            *map(FadeOut, [block.title for block in self.blocks if hasattr(block, "title")]),
            self.frame.animate.reorient(-3, -12, 0, target_frame_center, 12.00),
            run_time=3,
        )

    def show_unembedding(self):
        # Unembedding
        label_font_size = 30
        layer = self.layers[-1]
        last_embedding = layer.embeddings[-1]
        rect = SurroundingRectangle(last_embedding)
        rect.set_stroke(YELLOW, 3)

        words, dist = zip(*self.possible_next_tokens)
        bars = BarChart(dist).bars
        bars.rotate(-90 * DEGREES)
        bars.set_width(2.5, stretch=True)
        bars.next_to(layer, RIGHT, buff=4.0)
        bars.set_y(2)
        for bar, word, value in zip(bars, words, dist):
            percentage = DecimalNumber(
                100 * value,
                num_decimal_places=2,
                unit="%",
                font_size=label_font_size,
            )
            percentage.next_to(bar, RIGHT)
            text = Text(word, font_size=label_font_size)
            text.next_to(bar, LEFT)
            bar.push_self_into_submobjects()
            bar.add(text, percentage)

        dots = Tex(R"\\vdots")
        dots.next_to(bars[-1][1], DOWN)
        bars.add(dots)
        brace = Brace(bars, LEFT)

        arrow = Line()
        arrow.clear_points()
        arrow.start_new_path(rect.get_top())
        arrow.add_cubic_bezier_curve_to(
            rect.get_top() + UP,
            rect.get_top() + UR,
            rect.get_top() + 2 * RIGHT,
        )
        arrow.add_line_to(
            brace.get_left() + 0.1 * LEFT,
        )
        arrow.make_smooth(approx=False)
        arrow.add_tip()
        arrow.set_color(YELLOW)

        self.play(ShowCreation(rect))
        self.play(
            ShowCreation(arrow),
            GrowFromCenter(brace),
            LaggedStartMap(FadeIn, bars, shift=DOWN)
        )
        self.wait()
        # Test
        self.play(
            self.frame.animate.reorient(-5, -8, 0, (2.99, 2.67, 71.6), 13.60),
            run_time=8
        )


class SimplifiedFlow(HighLevelNetworkFlow):
    example_text = "Word vectors will be updated to encode more than mere words"
    attention_anim_run_time = 1.0
    orientation = (-55, -19, 0)
    target_frame_x = -2
    x_range = np.linspace(-2, -8, 5)
    frame_height_growth_rate = 0.15

    def construct(self):
        # Test
        self.camera.light_source.set_z(60)
        self.show_initial_text_embedding(word_scale_factor=0.6)
        self.show_simple_flow(self.x_range)

    def show_simple_flow(self, x_range, orientation=None):
        if orientation is None:
            orientation = self.orientation
        curr_time = float(self.time)
        curr_height = self.frame.get_height()
        self.frame.add_updater(lambda f: f.set_height(curr_height + self.frame_height_growth_rate * (self.time - curr_time)))
        for x in x_range:
            self.progress_through_attention_block(
                target_orientation=orientation,
                target_frame_x=self.target_frame_x,
                attention_anim_run_time=self.attention_anim_run_time,
            )
            self.progress_through_mlp_block(
                sideview_orientation=orientation,
                final_orientation=orientation,
            )


class AltIntro(HighLevelNetworkFlow):
    example_text = "Four score and seven years ago our fathers"

    def construct(self):
        self.show_initial_text_embedding(word_scale_factor=0.6)


class SimplifiedFlowAlternateAngle(SimplifiedFlow):
    example_text = "The goal of the network is to predict the next token"
    attention_anim_run_time = 1.0
    orientation = (-10, -20, 0)
    target_frame_x = 0
    hide_block_labels = True


class LongerFlowLongerAttention(SimplifiedFlow):
    example_text = "The goal of the network is to predict the next token"
    attention_anim_run_time = 3.0
    target_frame_x = 0
    hide_block_labels = True
    x_range = np.linspace(-2, -12, 7)


class MentionContextSizeAndUnembedding(SimplifiedFlow):
    example_text = "Harry Potter was a highly unusual boy ... least favourite teacher, Professor Snape"
    attention_anim_run_time = 5.0
    use_words = True

    def construct(self):
        # Initial flow
        self.show_initial_text_embedding(word_scale_factor=0.5)
        self.cycle_through_embeddings()
        self.show_simple_flow(
            np.linspace(-2, -5, 2),
            orientation=(-50, -19, 0)
        )
        self.show_context_size()

        # Skip to the end
        self.remove_mlps()
        self.mention_repetitions()
        self.focus_on_last_layer()
        self.remove(*self.layers[:-1])

        # Discuss unembedding
        self.show_desired_output()
        self.show_unembedding_matrix()

    def cycle_through_embeddings(self):
        layer = self.layers[0]
        blocks = self.token_blocks
        arrows = self.token_arrows
        embeddings = VGroup(*layer.embeddings, *layer.dots)
        embeddings.sort(lambda p: p[0])
        rect_groups = VGroup(*(
            VGroup(*(
                BackgroundRectangle(mob, buff=0.2 if mob in arrows else 0.1)
                for mob in tup
            ))
            for tup in zip(blocks, arrows, embeddings)
        ))
        rect_groups.set_opacity(0)
        self.add(rect_groups)

        for index in range(len(rect_groups)):
            rect_groups.generate_target()
            rect_groups.target.set_opacity(0.75)
            rect_groups.target[index].set_opacity(0)
            self.play(MoveToTarget(rect_groups))
        self.play(FadeOut(rect_groups))

    def show_context_size(self):
        # Zoom in
        frame = self.frame
        frame.save_state()
        block_titles = VGroup(*(block.title for block in self.blocks))
        layer = self.layers[-1]

        self.play(
            frame.animate.reorient(-27, -18, 0, (-0.84, -0.36, 18.15), 9.00),
            FadeOut(block_titles, lag_ratio=0.01),
            run_time=2,
        )

        # Show size
        font_size = 72
        brace = Brace(layer.embeddings, DOWN)
        label = brace.get_text("Context size", font_size=font_size)
        label.generate_target()
        rhs = Tex("= 2{,}048", font_size=font_size)
        full_label = VGroup(label.target, rhs)
        full_label.arrange(RIGHT, aligned_edge=UP)
        full_label.next_to(brace, DOWN)

        dim_brace = Brace(layer.embeddings, RIGHT)
        dim_label = dim_brace.get_text("12,288", font_size=font_size)

        self.play(
            GrowFromCenter(brace),
            FadeIn(label, 0.5 * DOWN),
        )
        self.wait()
        self.play(
            MoveToTarget(label),
            Write(rhs)
        )
        self.wait()
        self.play(
            FadeOut(layer.brackets),
            GrowFromCenter(dim_brace),
            FadeIn(dim_label, 0.5 * LEFT),
        )
        self.wait(3)

        # Return
        self.play(
            Restore(frame, run_time=3),
            FadeIn(block_titles, time_span=(2, 3), lag_ratio=0.01),
            FadeIn(layer.brackets),
            LaggedStartMap(FadeOut, VGroup(
                brace, label, rhs, dim_brace, dim_label
            ), shift=DOWN)
        )
        for layer, block in zip(self.layers, self.blocks):
            self.add(layer, block)
        self.add(self.layers[-1])

    def show_desired_output(self):
        # Show phrase
        frame = self.frame
        phrase = Text(self.example_text)
        phrase.set_max_width(FRAME_WIDTH - 1)
        phrase.to_corner(UL)
        phrase.fix_in_frame()
        last_word = phrase[self.example_text.split(" ")[-1]][0]
        last_word.set_opacity(0)
        rect = SurroundingRectangle(last_word, buff=0.1)
        rect.set_stroke(YELLOW, 2)
        rect.set_fill(YELLOW, 0.5)
        rect.align_to(last_word, LEFT)
        q_marks = Text("???")
        q_marks.next_to(rect, DOWN)
        q_group = VGroup(rect, q_marks)
        q_group.fix_in_frame()

        self.play(FadeIn(phrase, lag_ratio=0.1, run_time=2))
        self.play(FadeIn(q_group, 0.2 * RIGHT))
        self.wait()

        # Get all possible next words
        layer = self.layers[-1]
        word_strs = [
            "...",
            "Snake",
            "Snape",
            "Snare",
            "...",
            "Treks",
            "Trelawney",
            "Trellis",
            "...",
            "Quirky",
            "Quirrell",
            "Quirt",
            "..."
        ]
        words = VGroup()
        dots = VGroup()
        for word_str in word_strs:
            word = Text(word_str)
            words.add(word)
            if word_str == "...":
                dots.add(word)
                word.rotate(PI / 2)
        words.arrange(DOWN, aligned_edge=LEFT)
        dots.shift(0.25 * RIGHT)
        words.set_max_height(5.0)
        words.to_edge(DOWN, buff=1.0)
        words.to_edge(RIGHT, buff=0.1)
        words.fix_in_frame()

        # Add probability bars
        values = [0, 0.78, 0, 0, 0.16, 0, 0, 0.06, 0]
        bars = VGroup()
        probs = VGroup()
        value_iter = iter(values)
        bar_height = 0.8 * words[1].get_height()
        for word in words:
            if word in dots:
                continue
            value = next(value_iter)
            prob = DecimalNumber(value, font_size=24)
            bar = Rectangle(10 * value * bar_height, bar_height)
            bar.next_to(word, LEFT)
            hsl = list(Color(BLUE).get_hsl())
            hsl[0] = interpolate(0.5, 0.6, value)
            bar.set_fill(Color(hsl=hsl), 1)
            bar.set_stroke(BLUE, 1)
            prob.next_to(bar, LEFT)
            probs.add(prob)
            bars.add(bar)

        probs.fix_in_frame()
        bars.fix_in_frame()
        brace = Brace(VGroup(probs, dots), LEFT).fix_in_frame()
        arrow = Vector(RIGHT, stroke_width=8).fix_in_frame()
        arrow.set_color(YELLOW)
        arrow.next_to(brace, LEFT)

        # Show creation
        self.play(
            LaggedStartMap(FadeIn, words, shift=0.1 * DOWN),
            LaggedStartMap(FadeIn, bars),
            LaggedStartMap(FadeIn, probs, shift=0.2 * LEFT),
            GrowFromCenter(brace),
            GrowArrow(arrow),
        )
        self.wait()

        self.prob_group = VGroup(arrow, brace, words, bars, probs)
        self.phrase = VGroup(phrase, q_group)

    def show_unembedding_matrix(self, vector_index=-1):
        # Clear frame
        frame = self.frame
        prob_group = self.prob_group
        prob_group.save_state()
        prob_group.generate_target()
        prob_group.target.scale(0.5, about_edge=DR)
        prob_group.target.to_corner(DR)
        prob_group.target[0].set_opacity(0)

        self.play(
            FadeOut(self.phrase, UP),
            MoveToTarget(prob_group),
            frame.animate.reorient(0, 1, 0, (4.98, 3.34, 30.08), 12.00),
            run_time=2,
        )

        # Show the weight matrix
        layer = self.layers[-1]
        vector = layer.embeddings[vector_index].copy()
        matrix = WeightMatrix(
            shape=(15, vector.shape[0]),
            ellipses_row=8,
        )
        matrix.set_height(6)
        matrix.next_to(vector, UP, aligned_edge=RIGHT, buff=1.0)
        matrix.shift(LEFT)
        last_vector_rect = rect = SurroundingRectangle(vector, buff=0.1)
        rect.set_stroke(YELLOW, 3)
        vector.generate_target()
        vector.target.next_to(matrix, RIGHT)
        last_vect_arrow = Arrow(
            rect.get_top(), vector.target.get_bottom(),
        )
        last_vect_arrow.set_color(YELLOW)

        self.play(
            FadeIn(matrix, scale=0.8, shift=DR),
            *map(FadeOut, [self.token_blocks, self.token_arrows, self.final_word_question]),
        )
        self.play(ShowCreation(rect))
        self.play(MoveToTarget(vector), ShowCreation(last_vect_arrow))

        # Show matrix vector product
        eq, rhs = show_matrix_vector_product(self, matrix, vector)

        # Count values
        brace = Brace(rhs, RIGHT)
        brace_label = brace.get_tex(R"\\sim 50k \\text{ values}", font_size=60, buff=0.25)

        self.play(
            GrowFromCenter(brace),
            FadeIn(brace_label, lag_ratio=0.1),
        )
        brace_group = VGroup(brace, brace_label)

        # Show words
        word_strs = [
            "aah",
            "aardvark",
            "aardwolf",
            "aargh",
            "ab",
            "aback",
            "abacterial",
            "abacus",
            "...",
            "zygote",
            "zygotic",
            "zyme",
            "zymogen",
            "zymosis",
            "zzz",
        ]
        words = VGroup(*map(Text, word_strs))
        words[word_strs.index("...")].rotate(PI / 2)
        for word, entry in zip(words, rhs):
            word.set_max_height(entry.get_height())
            word.next_to(rhs, RIGHT, buff=0.25)
            word.match_y(entry)

        self.play(
            LaggedStartMap(FadeIn, words, shift=0.5 * RIGHT),
            brace_group.animate.next_to(words, RIGHT),
        )
        self.wait()

        # Mention softmax
        big_rect = SurroundingRectangle(VGroup(rhs, words), buff=0.25)
        big_rect.set_fill(TEAL, 0.1)
        big_rect.set_stroke(TEAL, 3)
        softmax_arrow = Vector(2.2 * RIGHT, stroke_width=8)
        softmax_arrow.next_to(big_rect, RIGHT, buff=0.1)
        softmax_label = Text("softmax", font_size=48)
        softmax_label.next_to(softmax_arrow, UP, buff=0.35)

        prob_group.generate_target()
        prob_group.target[0].scale(0).move_to(prob_group)
        prob_group.target.set_height(4)
        prob_group.target.to_edge(UP, buff=0.25).to_edge(RIGHT, buff=0.1)

        self.play(
            LaggedStart(
                FadeOut(brace_group),
                DrawBorderThenFill(big_rect),
                ShowCreation(softmax_arrow),
                FadeIn(softmax_label, lag_ratio=0.1),
            ),
            MoveToTarget(prob_group, time_span=(1, 2))
        )
        prob_group.unfix_from_frame()
        prob_group.match_height(rhs)
        prob_group.next_to(softmax_arrow, RIGHT, buff=0.25)
        self.wait()

        # Ask about other vectors
        rects = VGroup(*(
            SurroundingRectangle(emb)
            for emb in layer.embeddings[:-1]
        ))
        rects.set_stroke(PINK, 3)
        rects.set_fill(PINK, 0.25)
        question = Text("What about these?", font_size=90)
        question.next_to(layer, DOWN, buff=2.0)
        question_arrows = VGroup()
        for rect in rects:
            question_arrows.add(Arrow(
                question.get_top(), rect.get_bottom(),
                stroke_color=rect.get_color()
            ))

        self.play(
            frame.animate.reorient(-1, 0, 0, (1.69, 1.57, 29.79), 15.88),
            Write(question),
            run_time=2,
        )

        last_rect = VGroup()
        for rect, arrow in zip(rects, question_arrows):
            self.play(
                FadeOut(last_rect),
                FadeIn(rect),
                FadeIn(arrow),
            )
            last_rect = rect
        self.play(FadeOut(last_rect))
        self.wait()

        # Move back
        self.play(
            FadeOut(question_arrows, lag_ratio=0.1),
            FadeOut(question, lag_ratio=0.1),
            frame.animate.reorient(0, 1, 0, (4.27, 3.46, 29.83), 12.86),
            run_time=2,
        )
        # self.play(
        #     LaggedStartMap(FadeOut, VGroup(
        #         *question_arrows, question,
        #         matrix, vector, eq, rhs,
        #         big_rect, softmax_arrow, prob_group, words, softmax_label,
        #         last_vector_rect, last_vect_arrow,
        #     )),
        #     FadeOut(question_arrows, lag_ratio=0.1),
        #     FadeOut(question, lag_ratio=0.1),
        #     frame.animate.reorient(-2, 0, 0, (2.72, 1.82, 29.44), 10.31),
        #     run_time=2,
        # )
        self.wait()

        # Name the unembedding matrix
        matrix_rect = SurroundingRectangle(matrix, buff=0.1)
        matrix_rect.set_stroke(BLUE, 3)
        name = Text("Unembedding\\nmatrix", font_size=60)
        label = Tex("W_U", font_size=90)
        name.next_to(matrix_rect, LEFT)
        label.next_to(name, DOWN)

        self.play(
            Write(matrix_rect, stroke_width=5, stroke_color=BLUE),
            Write(name, run_time=2),
            frame.animate.reorient(0, 1, 0, (4.24, 3.15, 29.81), 13.23),
        )
        self.wait()
        self.play(
            FadeIn(label, DOWN),
            name.animate.next_to(label, UP, buff=1)
        )
        self.wait()

        # Data flying
        data_modifying_matrix(self, matrix)
        self.wait()

        # Count parameters
        label.set_backstroke(BLACK, 6)
        entries = VGroup(*matrix.elements, *matrix.ellipses)
        row_rects = VGroup(*map(SurroundingRectangle, matrix.get_rows()))
        col_rects = VGroup(*map(SurroundingRectangle, matrix.get_columns()))
        VGroup(row_rects, col_rects).set_stroke(GREY, 1).set_fill(GREY_B, 0.5)

        left_brace = Brace(matrix, LEFT)
        top_brace = Brace(matrix, UP)
        vocab_count = Integer(50257, font_size=90)
        vocab_count.next_to(left_brace, LEFT)
        dim_count = Integer(12288, font_size=90)
        dim_count.next_to(top_brace, UP)

        top_equation = VGroup(
            Text("Total parameters = "),
            Integer(vocab_count.get_value()),
            Tex(R"\\times"),
            Integer(dim_count.get_value()),
            Tex("="),
            Integer(vocab_count.get_value() * dim_count.get_value()).set_color(YELLOW),
        )
        top_equation.arrange(RIGHT)
        top_equation.scale(2)
        top_equation.next_to(dim_count, UP, buff=1)
        top_equation.align_to(vocab_count, LEFT)

        self.play(
            label.animate.move_to(matrix),
            FadeOut(name, lag_ratio=0.1),
            entries.animate.set_fill(opacity=0.5, border_width=0),
        )
        self.add(row_rects, label)
        self.play(
            GrowFromCenter(left_brace, time_span=(0, 1)),
            CountInFrom(vocab_count),
            LaggedStartMap(VFadeInThenOut, row_rects, lag_ratio=0.2),
            run_time=2.5,
        )
        self.add(col_rects, label)
        self.play(
            GrowFromCenter(top_brace, time_span=(0, 1)),
            CountInFrom(dim_count),
            LaggedStartMap(VFadeInThenOut, col_rects, lag_ratio=0.2),
            frame.animate.reorient(0, 0, 0, (5.66, 4.08, 29.89), 14.32),
            run_time=2.5,
        )
        self.wait()
        self.play(LaggedStart(
            Write(top_equation[0:6:2]),
            TransformFromCopy(vocab_count, top_equation[1]),
            TransformFromCopy(dim_count, top_equation[3]),
            frame.animate.reorient(0, -1, 0, (4.15, 5.07, 29.75), 17.21),
            run_time=3
        ))
        self.play(
            FadeTransform(top_equation[1:4].copy(), top_equation[-1])
        )
        self.wait()


class FlowForMLPIntroReview(SimplifiedFlow):
    example_text = "That which does not kill you only makes you stronger"
    x_range = np.linspace(-2, -4, 3)
    frame_height_growth_rate = 0.3
    possible_next_tokens = [
        ("stronger", 0.906),
        ("stranger", 0.028),
        ("more", 0.006),
        ("weaker", 0.003),
        ("...", 0.003),
        ("Strong", 0.002),
        ("wish", 0.002),
        ("STR", 0.002),
    ]

    def construct(self):
        # Initial flow
        self.camera.light_source.set_z(20)
        self.show_initial_text_embedding(word_scale_factor=0.6)
        self.play(self.frame.animate.scale(1.25))
        self.show_simple_flow(self.x_range)

        # Show how it ends
        self.remove_mlps()
        self.mention_repetitions()
        self.frame.clear_updaters()
        self.focus_on_last_layer()
        self.play(self.frame.animate.scale(1.15).shift(RIGHT))
        self.show_unembedding()


class FlowForCHM(SimplifiedFlow):
    example_text = "Down by the river bank ... until they jumped into the pond"
    use_words = True
    x_range = np.linspace(-2, -4, 3)
    frame_height_growth_rate = 0.3

    def construct(self):
        # Initial flow
        self.camera.light_source.set_z(20)
        self.show_initial_text_embedding(word_scale_factor=0.6)
        self.play(self.frame.animate.scale(1.25))
        self.show_simple_flow(self.x_range)

        # Show how it ends
        self.remove_mlps()
        self.mention_repetitions()
        self.frame.clear_updaters()
        self.focus_on_last_layer()
        self.play(self.frame.animate.scale(1.15).shift(RIGHT))
        self.show_unembedding()

    def progress_through_mlp_block(self, *args, **kwargs):
        kwargs.update(
            label_text="Feed Forward",
            title_font_size=92,
        )
        super().progress_through_mlp_block(*args, **kwargs)


class FlowForCHM2(FlowForCHM):
    example_text = "The Computer History Museum is located in Mountain"
    possible_next_tokens = [
        ("Mountain", 0.706),
        ("the", 0.128),
        ("San", 0.078),
        ("Seattle", 0.006),
        ("New", 0.003),
        ("Palo", 0.002),
        ("Google", 0.002),
        ("southern", 0.002),
    ]

    def show_initial_text_embedding(self, word_scale_factor=0.6, bump_first=False):
        super().show_initial_text_embedding(word_scale_factor=0.5)


class TextToNumerical(SimplifiedFlow):
    example_text = "Text must be encoded as numbers blah"
    use_words = True

    def construct(self):
        # Test
        self.show_initial_text_embedding(word_scale_factor=0.75)


class FlowForCHMNoText(FlowForCHM2):
    def get_block(self, *args, **kwargs):
        kwargs.update(title="")
        return super().get_block(*args, **kwargs)


class TextPassageIntro(InteractiveScene):
    example_text = MentionContextSizeAndUnembedding.example_text

    def construct(self):
        # Read in passage
        passage_str = Path(DATA_DIR, "harry_potter_3.txt").read_text()
        passage_str = passage_str.replace("\\n", "\\n\\\\\\\\")
        passage = TexText(passage_str, alignment="", additional_preamble=R"\\tiny")
        passage.set_height(FRAME_HEIGHT - 1)
        passage[-len("Snape"):].set_opacity(0)

        # Initial surroundings
        frame = self.frame
        lh, rh = (1176, 1540)
        section = passage[lh:rh].copy()
        section.save_state()
        section.set_width(FRAME_WIDTH - 1)
        section.center()

        word_lh, word_rh = (150, 155)
        word = section[word_lh:word_rh].copy()
        word.save_state()
        word.set_height(0.75).center()

        self.play(Write(word))
        self.wait()
        self.play(
            Restore(word, time_span=(0, 0.5), remover=True),
            ShowIncreasingSubsets(section),
            run_time=1
        )
        self.play(ContextAnimation(word, section))
        self.wait()
        self.play(
            Restore(section, remover=True),
            ShowIncreasingSubsets(VGroup(*passage[:lh], *passage[rh:]))
        )
        self.add(passage)
        self.play(ContextAnimation(
            passage[lh:rh][word_lh:word_rh],
            VGroup(
                *passage[0:11],
                *passage[211:217],
                # *passage[2366:2374],
            ),
            lag_ratio=0.01
        ))
        self.wait()

        # Compress
        start, end = self.example_text.split(" ... ")
        short_text = Text(self.example_text)
        short_text["Snape"].set_opacity(0)
        short_text.set_max_width(FRAME_WIDTH - 1)
        dots = short_text["..."][0]

        lh, rh = (31, 2474)
        self.play(
            FadeTransformPieces(
                passage[:lh],
                short_text[start][0],
            ),
            ReplacementTransform(passage[lh:rh], dots),
            FadeTransformPieces(
                passage[rh:],
                short_text[end][0],
            ),
            run_time=3
        )
        self.add(short_text)
        self.wait()


class ThumbnailBase(HighLevelNetworkFlow):
    block_to_title_direction = LEFT
    def construct(self):
        # Add blocks
        self.show_initial_text_embedding(word_scale_factor=0.6)
        for x in range(4):
            self.progress_through_attention_block(
                depth=1.5,
                label_text="Att"
            )
            self.progress_through_mlp_block(
                depth=2.0,
                label_text="MLP"
            )
        self.frame.reorient(-45, -10, 0, (-2.9, 0.95, 24.44), 14.34)

        # Fade title
        for n, block in enumerate(self.blocks):
            block.title.set_opacity(0)
            block.body.set_opacity(0.5)
        self.remove(*self.layers[:-1])

        self.mention_repetitions()
        self.rep_label[0].apply_depth_test()
        self.rep_label[1:].set_opacity(0)
        self.add(self.blocks)

    example_text = "The initials GPT stand for Generative Pre-trained Transformer"


class MoleExample1(HighLevelNetworkFlow):
    block_to_title_direction = LEFT
    highlighted_group_index = 1

    def construct(self):
        # Show three phrases
        phrase_strs = [
            "American shrew mole",
            "One mole of carbon dioxide",
            "Take a biopsy of the mole",
        ]
        phrases = VGroup(map(Text, phrase_strs))
        phrases.arrange(DOWN, buff=2.0)
        phrases.move_to(0.25 * DOWN)

        self.play(Write(phrases[0]), run_time=1)
        self.wait()
        for i in [1, 2]:
            self.play(
                Transform(phrases[i - 1]["mole"].copy(), phrases[i]["mole"].copy(), remover=True),
                FadeIn(phrases[i], lag_ratio=0.1)
            )
            self.wait()

        # Add mole images
        images = Group(
            ImageMobject("ShrewMole").set_height(1),
            Tex(R"6.02 \\times 10^{23}").set_color(TEAL),
            ImageMobject("LipMole").set_height(1),
        )
        braces = VGroup()
        mole_words = VGroup()
        for image, phrase in zip(images, phrases):
            mole_word = phrase["mole"][0]
            brace = Brace(mole_word, UP, SMALL_BUFF)
            image.next_to(brace, UP, SMALL_BUFF)
            braces.add(brace)
            mole_words.add(mole_word)

        self.play(
            LaggedStartMap(GrowFromCenter, braces, lag_ratio=0.5),
            LaggedStartMap(FadeIn, images, shift=UP, lag_ratio=0.5),
            mole_words.animate.set_color(YELLOW).set_anim_args(lag_ratio=0.1),
        )
        self.wait()

        # Subdivide
        word_groups = VGroup()
        for phrase in phrases:
            words = break_into_words(phrase.copy())
            rects = get_piece_rectangles(
                words, leading_spaces=False, h_buff=0.05
            )
            word_group = VGroup(VGroup(*pair) for pair in zip(rects, words))
            word_groups.add(word_group)

        self.play(
            FadeIn(word_groups),
            LaggedStartMap(FadeOut, braces, shift=0.25 * DOWN, lag_ratio=0.25),
            LaggedStartMap(FadeOut, images, shift=0.25 * DOWN, lag_ratio=0.25),
            run_time=1
        )
        self.remove(phrases)
        self.wait()

        # Divide into three regions
        for group, sign in zip(word_groups, [-1, 0, 1]):
            group.target = group.generate_target()
            group.target.scale(0.75)
            group.target.set_x(sign * FRAME_WIDTH / 3)
            group.target.to_edge(UP)

        v_lines = Line(UP, DOWN).replicate(2)
        v_lines.set_height(FRAME_HEIGHT)
        v_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        v_lines.center()
        v_lines.set_stroke(GREY_B, 1)

        self.play(
            LaggedStartMap(MoveToTarget, word_groups),
            ShowCreation(v_lines, lag_ratio=0.5, time_span=(1, 2))
        )

        # Show vector embeddings
        embs = VGroup()
        arrows = VGroup()
        seed_array = np.random.uniform(0, 10, 7)
        for group in word_groups:
            for word in group:
                arrow = Vector(0.5 * DOWN)
                arrow.next_to(word, DOWN, SMALL_BUFF)
                size = sum(len(m.get_points()) for m in  word.family_members_with_points())
                values = (seed_array * size % 10)
                emb = NumericEmbedding(values=values)
                emb.set_height(2)
                emb.next_to(arrow, DOWN, SMALL_BUFF)

                arrows.add(arrow)
                embs.add(emb)
        mole_indices = [2, 4, 13]
        non_mole_indices = [n for n in range(len(embs)) if n not in mole_indices]

        mole_vect_rects = VGroup(
            SurroundingRectangle(embs[index])
            for index in mole_indices
        )
        mole_vect_rects.set_stroke(YELLOW, 2)

        self.play(
            LaggedStartMap(GrowArrow, arrows),
            LaggedStartMap(FadeIn, embs, shift=0.25 * DOWN),
        )
        self.wait()
        self.play(
            LaggedStartMap(ShowCreation, mole_vect_rects),
            VGroup(arrows[j] for j in non_mole_indices).animate.set_fill(opacity=0.5),
            VGroup(embs[j] for j in non_mole_indices).animate.set_fill(opacity=0.5),
        )
        self.wait()
        self.play(
            FadeOut(mole_vect_rects)
        )

        # Prepare to pass through an attention block
        wg_lens = [len(wg) for wg in word_groups]
        indices = [0, *np.cumsum(wg_lens)]
        full_groups = VGroup(
            VGroup(wg, arrows[i:j], embs[i:j])
            for wg, i, j in zip(word_groups, indices, indices[1:])
        )
        highlighted_group = full_groups[self.highlighted_group_index]
        fade_groups = [fg for n, fg in enumerate(full_groups) if n != self.highlighted_group_index]
        highlighted_group.target = highlighted_group.generate_target()
        highlighted_group.target.scale(1.5, about_edge=UP)
        highlighted_group.target.space_out_submobjects(1.1)
        highlighted_group.target.center()
        highlighted_group.target[2].set_fill(opacity=1)

        self.play(
            FadeOut(v_lines, time_span=(0, 1)),
            MoveToTarget(highlighted_group, lag_ratio=5e-4),
            *(
                FadeOut(
                    fg,
                    shift=fg.get_center() - highlighted_group.get_center() + 2 * DOWN,
                    lag_ratio=1e-3
                )
                for fg in  fade_groups
            ),
            run_time=2
        )
        self.wait()

        # Pass through attention
        layer = VGroup(highlighted_group[2])
        layer.embeddings = highlighted_group[2]
        self.layers.set_submobjects([])
        self.layers.add(layer)

        self.progress_through_attention_block(target_frame_x=-2)
        self.wait()`,
    annotations: {
      2: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      3: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      4: "Imports * from the _2024.transformers.embedding module within the 3b1b videos codebase.",
      7: "HighLevelNetworkFlow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      40: "setup() runs before construct(). Used to initialize shared state and add persistent mobjects.",
      44: "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects.",
      49: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      147: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      149: "Dot product: measures alignment between two vectors. Zero means perpendicular.",
      151: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      164: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      176: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      183: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      188: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      189: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      190: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      191: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      194: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      195: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      196: "FadeOut transitions a mobject from opaque to transparent.",
      197: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      198: "DrawBorderThenFill first draws the stroke outline, then fills the interior.",
      201: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      202: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      205: "FadeOut transitions a mobject from opaque to transparent.",
      207: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      210: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      213: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      217: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      218: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      219: "FadeOut transitions a mobject from opaque to transparent.",
      237: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      246: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      247: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      249: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      250: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      251: "LaggedStartMap applies an animation to each element of a group with staggered start times.",
      252: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      255: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      282: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      298: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      300: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      301: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      302: "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
      303: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      304: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      307: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      313: "Smoothly animates the camera to a new orientation over the animation duration.",
      319: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      320: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      321: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      322: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      323: "Restore animates a mobject back to a previously saved state (from save_state()).",
      335: "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
      668: "Class SimplifiedFlow inherits from HighLevelNetworkFlow.",
      676: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      700: "Class AltIntro inherits from HighLevelNetworkFlow.",
      703: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      707: "Class SimplifiedFlowAlternateAngle inherits from SimplifiedFlow.",
      715: "Class LongerFlowLongerAttention inherits from SimplifiedFlow.",
      723: "Class MentionContextSizeAndUnembedding inherits from SimplifiedFlow.",
      728: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1176: "Class FlowForMLPIntroReview inherits from SimplifiedFlow.",
      1191: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1207: "Class FlowForCHM inherits from SimplifiedFlow.",
      1213: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1236: "Class FlowForCHM2 inherits from FlowForCHM.",
      1253: "Class TextToNumerical inherits from SimplifiedFlow.",
      1257: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1262: "Class FlowForCHMNoText inherits from FlowForCHM2.",
      1268: "TextPassageIntro extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1271: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1341: "Class ThumbnailBase inherits from HighLevelNetworkFlow.",
      1343: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1371: "Class MoleExample1 inherits from HighLevelNetworkFlow.",
      1375: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/old_auto_regression.py"] = {
    description: "Earlier version of the autoregression scenes, preserved for reference. Contains alternative visual approaches to explaining next-token prediction.",
    code: `from manim_imports_ext import *
from _2024.transformers.helpers import *

from transformers import GPT2Tokenizer
from transformers import GPT2LMHeadModel
from transformers import PreTrainedModel
import torch
import openai
import tiktoken


@lru_cache(maxsize=1)
def get_gpt2_tokenizer(model_name='gpt2'):
    return GPT2Tokenizer.from_pretrained(model_name)


@lru_cache(maxsize=1)
def get_gpt2_model(model_name='gpt2'):
    return GPT2LMHeadModel.from_pretrained(model_name)


def gpt2_predict_next_token(text, n_shown=7):
    tokenizer = get_gpt2_tokenizer()
    model = get_gpt2_model()
    # Encode the input text
    indexed_tokens = tokenizer.encode(
        text, add_special_tokens=False, return_tensors='pt'
    )

    # Predict all tokens
    with torch.no_grad():
        outputs = model(indexed_tokens)
        # Pull out the first batch, and the last token prediction
        predictions = outputs[0][0, -1, :]

    # Get the predicted next token
    indices = torch.argsort(predictions)
    top_indices = reversed(indices[-n_shown:])
    tokens = list(map(tokenizer.decode, top_indices))
    probs = softmax(predictions)[top_indices]

    return tokens, probs


def gpt3_predict_next_token(text, n_shown=10, random_seed=0):
    openai.api_key = os.getenv('OPENAI_KEY')
    response = openai.Completion.create(
        # Or another model version, adjust as necessary
        engine="gpt-3.5-turbo-instruct",
        prompt=text,
        max_tokens=1,
        n=1,
        temperature=1.0,
        user=str(random_seed),
        # Retrieve more than are shown
        logprobs=50
    )
    top_logprob_dict = response.choices[0]["logprobs"]["top_logprobs"][0]
    tokens, logprobs = zip(*top_logprob_dict.items())
    probs = np.exp(logprobs)
    indices = np.argsort(probs)
    top_indices = indices[-1:-n_shown:-1]
    top_tokens = [tokens[i] for i in top_indices]
    top_probs = [probs[i] for i in top_indices]
    return top_tokens, top_probs


class SimpleAutogregression(InteractiveScene):
    text_corner = 3.5 * UP + 0.75 * RIGHT
    line_len = 31
    font_size = 35
    n_shown_predictions = 12
    seed_text = "Behold, a wild pi creature, foraging in its native"
    seed_text_color = BLUE_B
    machine_name = "Transformer"
    machine_phi = 10 * DEGREES
    machine_theta = 12 * DEGREES
    n_predictions = 120
    skip_through = False
    random_seed = 0
    model = "gpt2"

    def construct(self):
        # Repeatedly generate
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                quick=(n > 10),
                skip_anims=self.skip_through,
            )

    def init_text_and_machine(self):
        # Set up active text
        self.cur_str = self.seed_text
        text_mob = self.string_to_mob(self.cur_str)
        text_mob.set_color(self.seed_text_color)
        next_word_line = self.get_next_word_line(text_mob)

        # Set up Transformer as some sort of machine
        machine = self.get_transformer_drawing()
        machine.set_y(0).to_edge(LEFT, buff=-0.6)

        self.add(text_mob)
        self.add(next_word_line)
        self.add(machine)

        return text_mob, next_word_line, machine

    def string_to_mob(self, text):
        text += " l"  # Dumb hack for alignment
        result = get_paragraph(
            text.replace("\\n", " ").split(" "),
            self.line_len,
            self.font_size
        )
        result.move_to(self.text_corner, UL)
        result[-1].set_fill(BLACK, 0)  # Continue dumb hack
        result[-1].stretch(0, 0, about_edge=LEFT)
        return result

    def get_next_word_line(self, text_mob, char_len=7):
        next_word_line = Underline(text_mob[:char_len])
        next_word_line.set_stroke(TEAL, 2)
        next_word_line.next_to(text_mob[-1], RIGHT, SMALL_BUFF, aligned_edge=DOWN)
        if self.skip_through:
            next_word_line.set_opacity(0)
        return next_word_line

    def get_transformer_drawing(self):
        self.camera.light_source.move_to([-5, 5, 10])
        self.frame.set_field_of_view(20 * DEGREES)
        blocks = VGroup(
            VPrism(3, 2, 0.2)
            for n in range(10)
        )
        blocks.set_fill(GREY_D, 1)
        blocks.set_stroke(width=0)
        blocks.set_shading(0.25, 0.5, 0.2)
        blocks.arrange(OUT)
        blocks.move_to(ORIGIN, OUT)
        blocks.rotate(self.machine_phi, RIGHT, about_edge=OUT)
        blocks.rotate(self.machine_theta, UP, about_edge=OUT)

        blocks.deactivate_depth_test()
        for block in blocks:
            block.sort(lambda p: p[2])

        word = Text(self.machine_name, alignment="LEFT")
        word.next_to(blocks[-1], UP)
        word.shift(0.1 * UP + 0.4 * LEFT)
        word.move_to(blocks[-1])
        word.set_backstroke(BLACK, 5)
        out_arrow = Vector(
            0.5 * RIGHT, stroke_width=10,
            max_tip_length_to_length_ratio=0.5,
            max_width_to_length_ratio=12
        )
        out_arrow.next_to(blocks[-1], RIGHT, buff=SMALL_BUFF)
        out_arrow.set_opacity(0)

        result = VGroup(blocks, word, out_arrow)
        return result

    def get_distribution(
        self, words, probs, machine,
        font_size=24,
        width_100p=1.8,
        bar_height=0.25,
        show_ellipses=True
    ):
        labels = VGroup(Text(word, font_size=font_size) for word in words)
        bars = VGroup(
            Rectangle(prob * width_100p, bar_height)
            for prob, label in zip(probs, labels)
        )
        bars.arrange(DOWN, aligned_edge=LEFT, buff=0.5 * bar_height)
        bars.set_fill(opacity=1)
        bars.set_submobject_colors_by_gradient(TEAL, YELLOW)
        bars.set_stroke(WHITE, 1)

        bar_groups = VGroup()
        for label, bar, prob in zip(labels, bars, probs):
            prob_label = Integer(int(100 * prob), unit="%", font_size=0.75 * font_size)
            prob_label.next_to(bar, RIGHT, buff=SMALL_BUFF)
            label.next_to(bar, LEFT)
            bar_groups.add(VGroup(label, bar, prob_label))

        if show_ellipses:
            ellipses = Tex(R"\\vdots", font_size=font_size)
            ellipses.next_to(bar_groups[-1][0], DOWN)
            bar_groups.add(ellipses)

        arrow_point = machine[-1].get_right()
        bar_groups.shift(arrow_point - bars.get_left() + 1.5 * RIGHT)
        bar_groups.align_to(machine, UP)

        return bar_groups

    def animate_text_input(self, text_mob, machine, position_text_over_machine=True, added_anims=[], lag_ratio=0.02):
        blocks = machine[0]
        text_copy = text_mob.copy()
        if position_text_over_machine:
            text_copy.target = text_copy.generate_target()
            text_copy.target.set_max_width(4)
            text_copy.target.next_to(blocks[0], UP)
            text_copy.target.shift_onto_screen()
            self.play(MoveToTarget(text_copy, path_arc=-45 * DEGREES))
        self.play(LaggedStart(
            *added_anims,
            Transform(
                text_copy,
                VGroup(VectorizedPoint(machine.get_top())),
                lag_ratio=lag_ratio,
                run_time=1,
                path_arc=-45 * DEGREES,
                remover=True,
            ),
            LaggedStart(
                (
                    block.animate.set_color(
                        block.get_color() if block is blocks[-1] else TEAL
                    ).set_anim_args(rate_func=there_and_back)
                    for block in blocks
                ),
                lag_ratio=0.1,
                run_time=1
            ),
            Animation(machine[1:]),
            lag_ratio=0.5
        ))

    def animate_prediction_ouptut(self, machine, cur_str):
        words, probs = self.predict_next_token(cur_str)
        bar_groups = self.get_distribution(words, probs, machine)
        self.play(
            LaggedStart(
                (FadeInFromPoint(bar_group, machine[0][-1].get_right())
                for bar_group in bar_groups),
                lag_ratio=0.025,
                group=bar_groups,
                run_time=1
            )
        )
        return bar_groups

    def animate_random_sample(self, bar_groups):
        widths = np.array([group[1].get_width() for group in bar_groups[:-1]])
        dist = widths / widths.sum()
        seed = random.randint(0, 1000)
        buff = 0.025
        highlight_rect = SurroundingRectangle(bar_groups[0], buff=buff)
        highlight_rect.set_stroke(YELLOW, 2)
        highlight_rect.set_fill(YELLOW, 0.25)

        def highlight_randomly(rect, dist, alpha):
            np.random.seed(seed + int(10 * alpha))
            index = np.random.choice(np.arange(len(dist)), p=dist)
            rect.surround(bar_groups[index], buff=buff)
            rect.stretch(1.1, 0)

        self.play(
            UpdateFromAlphaFunc(highlight_rect, lambda rect, a: highlight_randomly(rect, dist, a)),
            Animation(bar_groups)
        )

        bar_groups.add_to_back(highlight_rect)

    def animate_word_addition(self, bar_groups, text_mob, next_word_line, force_unskip=False):
        # Choose the highlighted_group
        bar_group = None
        if isinstance(bar_groups[0], Rectangle):
            # Use the highlight rect to find the group element
            bars = bar_groups[1:-1]
            diffs = [abs(bg.get_y() - bar_groups[0].get_y()) for bg in bars]
            bar_group = bar_groups[1:][np.argmin(diffs)]
        if bar_group is None:
            bar_group = bar_groups[0]

        # Animate selection
        word = bar_group[0].get_text()
        new_str = self.cur_str + word
        new_text_mob = self.string_to_mob(new_str)
        new_text_mob[:len(self.seed_text.replace(" ", ""))].set_color(self.seed_text_color)

        word_targets = new_text_mob[word.strip()]
        if len(word_targets) > 0:
            target = word_targets[-1]
        else:
            target = new_text_mob[-len(word) - 1:-1]

        # target = new_text_mob[-len(word):]

        self.add(bar_groups)
        self.play(
            FadeTransform(bar_group[0].copy(), target),
            Transform(
                next_word_line,
                self.get_next_word_line(new_text_mob),
            ),
        )
        if force_unskip:
            self.skip_animations = False
            target.save_state()
            target.set_fill(YELLOW)
            self.wait(0.5)
            target.restore()
            self.skip_animations = True
        self.play(
            FadeOut(bar_groups),
        )

        self.remove(text_mob)
        self.add(new_text_mob)

        self.cur_str = new_str

        return new_text_mob

    def new_selection_cycle(self, text_mob, next_word_line, machine, quick=False, skip_anims=False):
        if skip_anims:
            self.skip_animations = True

        if quick:
            words, probs = self.predict_next_token(self.cur_str)
            bar_groups = self.get_distribution(words, probs, machine)
            self.add(bar_groups)
        else:
            self.animate_text_input(text_mob, machine)
            bar_groups = self.animate_prediction_ouptut(machine, self.cur_str)
        self.animate_random_sample(bar_groups)
        new_text_mob = self.animate_word_addition(
            bar_groups, text_mob, next_word_line,
            force_unskip=skip_anims
        )
        return new_text_mob

    #

    def predict_next_token(self, text):
        result = None
        n_shown = self.n_shown_predictions
        if self.model == "gpt3":
            try:
                result = gpt3_predict_next_token(
                    text, n_shown, random_seed=self.random_seed
                )
            except Exception as e:
                pass
        if result is None:
            result = gpt2_predict_next_token(text, n_shown)
        return result


class AnnotateNextWord(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()
        self.add(machine, *machine[1:])
        words, probs = self.predict_next_token(self.cur_str)
        bar_groups = self.get_distribution(words, probs, machine)

        self.add(bar_groups)

        # Initial text
        from manimlib.mobject.boolean_ops import Union
        highlight = Union(
            SurroundingRectangle(text_mob["in its native"]),
            SurroundingRectangle(text_mob["Behold, a wild pi creature, foraging"]),
        )
        highlight.set_stroke(BLUE, 3)
        arrow = Vector(RIGHT, stroke_width=10)
        arrow.next_to(highlight, LEFT)

        dist_rect = SurroundingRectangle(bar_groups)
        dist_rect.set_stroke(YELLOW, 2)

        self.play(
            ShowCreation(highlight),
            GrowArrow(arrow)
        )
        self.wait()
        self.play(
            arrow.animate.rotate(-PI / 2).next_to(dist_rect, UP),
            ReplacementTransform(highlight, dist_rect),
        )
        self.wait()
        self.play(
            FadeOut(dist_rect),
            FadeOut(arrow),
        )


class QuickerRegression(SimpleAutogregression):
    skip_through = True


class AutoregressionGPT3(SimpleAutogregression):
    model = "gpt3"


class QuickRegressionGPT3(SimpleAutogregression):
    skip_through = True
    model = "gpt3"


class GPT3CleverestAutocomplete(QuickRegressionGPT3):
    seed_text = "To date, the cleverest thinker of all time was"
    n_predictions = 70

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        for n in range(self.n_predictions):
            text_mob = self.new_selection_cycle(
                text_mob, next_word_line, machine,
                skip_anims=(n > 2),
            )


class GPT3OnLearningSimpler(QuickRegressionGPT3):
    seed_text = "The most effective way to learn computer science is"
    text_corner = 3.5 * UP + 3 * LEFT
    line_len = 35
    font_size = 35
    n_predictions = 300
    time_per_prediction = 0.2
    random_seed = 313

    def construct(self):
        # Test
        cur_str = self.seed_text
        text_mob = VGroup()
        for n in range(self.n_predictions):
            self.remove(text_mob)
            words, probs = self.predict_next_token(cur_str)
            probs = probs / probs.sum()
            index = np.random.choice(np.arange(len(words)), p=probs)
            new_word = words[index]
            cur_str += new_word
            text_mob = self.string_to_mob(cur_str)
            text_mob[:len(self.seed_text.replace(" ", ""))].set_color(BLUE)
            text_mob[new_word.strip()][-1].set_color(YELLOW)
            if text_mob.get_bottom()[1] < -3:
                text_mob.shift(5 * UP)
                self.text_corner += 5 * UP
            self.add(text_mob)
            self.wait(self.time_per_prediction)


class ModelTakingInTextWithSurroundingPieces(SimpleAutogregression):
    def construct(self):
        text_mob, next_word_line, machine = self.init_text_and_machine()


class AthleteCompletion(SimpleAutogregression):
    seed_text = "Michael Jordan plays the sport of"
    text_corner = 3.5 * UP + 3.0 * LEFT
    machine_phi = 5 * DEGREES
    machine_theta = 12 * DEGREES
    model = "gpt3"

    def construct(self):
        # Initialize machine
        self.set_floor_plane("xz")
        frame = self.frame
        in_text, next_word_line, machine = self.init_text_and_machine()
        self.clear()
        machine = VGroup(*machine[0])
        machine.set_height(4)
        machine.next_to(in_text, DOWN, buff=LARGE_BUFF)

        dials = MachineWithDials(n_rows=10, n_cols=15).dials
        dials.set_stroke(opacity=0.25)
        dials.set_height(machine[-1].get_height() * 0.9)

        llm_title = Text("Large\\nLanguage\\nModel", alignment="LEFT", font_size=72)
        llm_title.set_backstroke(width=8)

        for mob in [dials, llm_title]:
            mob.rotate(self.machine_phi, RIGHT).rotate(self.machine_theta, UP)
            mob.move_to(machine[-1], OUT)

        last_block_copy = machine[-1].copy()
        self.add(last_block_copy)

        frame.reorient(-13, -6, 0)
        self.play(
            LaggedStart(
                (TransformFromCopy(last_block_copy.copy().set_opacity(0), block)
                for block in machine),
                lag_ratio=0.05,
            ),
            Write(dials),
            Write(llm_title),
            frame.animate.reorient(0, 0, 0),
            run_time=3
        )
        self.remove(last_block_copy)
        self.add(machine, dials, llm_title)

        # Feed in many facts
        facts = Path(DATA_DIR, "facts.txt").read_text().split("\\n")
        fact_mobs = VGroup(get_paragraph(fact.split(" "), line_len=20) for fact in facts)
        directions = compass_directions(12, start_vect=UR)
        for fact_mob, vect in zip(fact_mobs, it.cycle(directions)):
            fact_mob.set_max_width(2)
            fact_mob.move_to(5 * vect).shift_onto_screen(buff=0.25)

        self.play(
            LaggedStart(
                (Succession(
                    FadeIn(fact_mob),
                    fact_mob.animate.set_opacity(0).move_to(machine.get_center()),
                )
                for fact_mob in fact_mobs),
                lag_ratio=0.05,
                run_time=8
            )
        )
        self.remove(fact_mobs)
        self.wait()

        # Show MJ fact
        full_input = VGroup(in_text, next_word_line)
        full_input.set_height(0.4)
        full_input.to_edge(UP)

        in_arrow = Arrow(full_input, machine, buff=0.1)
        predictions, probs = self.predict_next_token(self.seed_text)

        bar_groups = self.get_distribution(predictions, probs, machine)
        bar_groups.next_to(machine[-1], RIGHT, buff=1.5)
        out_arrow = Arrow(machine[-1], bar_groups)

        top_rect = SurroundingRectangle(VGroup(bar_groups[0]))

        self.play(FadeIn(full_input, scale=2))
        self.play(
            GrowArrow(in_arrow),
            Transform(full_input.copy(), full_input.copy().scale(0.5).set_opacity(0).move_to(machine.get_top()))
        )
        self.play(
            frame.animate.reorient(-14, -2, 0, (1.83, 0.07, -0.38), 8.63),
            LaggedStart(
                (block.animate.set_color(TEAL).set_anim_args(rate_func=there_and_back)
                for block in machine[:-1]),
                lag_ratio=0.1,
                run_time=1
            ),
        )
        self.play(
            ShowCreation(out_arrow),
            FadeIn(bar_groups, lag_ratio=0.1)
        )
        self.wait()
        self.play(ShowCreation(top_rect))

        # Reshow parameters
        self.play(
            FadeOut(llm_title),
            dials.animate.set_stroke(opacity=1)
        )
        for _ in range(5):
            self.play(
                LaggedStart(
                    (dial.animate_set_value(dial.get_random_value())
                    for dial in dials),
                    lag_ratio=0.25 / len(dials),
                    run_time=1
                )
            )

        # Quetsions
        questions = VGroup(Text("How?"), Text("Where?"))
        questions.arrange(RIGHT, buff=1.0)
        questions.set_height(0.5)
        questions.next_to(machine[-1], DOWN)

        for question in questions:
            self.play(FadeIn(question, 0.5 * UP, scale=1.5))
        self.wait()


class ThatWhichDoesNotKillMe(SimpleAutogregression):
    text_corner = 3.5 * UP + 5.0 * LEFT
    line_len = 75
    # seed_text = "That which does not kill you only makes you"
    seed_text = "Down by the river bank"
    model = "gpt3"

    def construct(self):
        # Test
        text_mob, next_word_line, machine = self.init_text_and_machine()
        machine.set_x(0)
        text_mob = self.new_selection_cycle(
            text_mob, next_word_line, machine,
            quick=False,
            skip_anims=False,
        )`,
    annotations: {
      1: "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
      2: "Imports * from the _2024.transformers.helpers module within the 3b1b videos codebase.",
      60: "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models.",
      68: "SimpleAutogregression extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      83: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      139: "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance.",
      149: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      172: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      184: "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
      190: "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
      209: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      211: "Transform smoothly morphs one mobject into another by interpolating their points.",
      219: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      221: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      236: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      237: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      262: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      263: "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
      295: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      297: "Transform smoothly morphs one mobject into another by interpolating their points.",
      304: "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore().",
      306: "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run.",
      309: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      310: "FadeOut transitions a mobject from opaque to transparent.",
      355: "Class AnnotateNextWord inherits from SimpleAutogregression.",
      356: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      377: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      378: "ShowCreation draws a VMobject's stroke progressively from start to end.",
      379: "GrowArrow animates an arrow growing from its start point to full length.",
      381: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      382: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      383: "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call.",
      384: "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
      386: "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run.",
      387: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      388: "FadeOut transitions a mobject from opaque to transparent.",
      389: "FadeOut transitions a mobject from opaque to transparent.",
      393: "Class QuickerRegression inherits from SimpleAutogregression.",
      397: "Class AutoregressionGPT3 inherits from SimpleAutogregression.",
      401: "Class QuickRegressionGPT3 inherits from SimpleAutogregression.",
      406: "Class GPT3CleverestAutocomplete inherits from QuickRegressionGPT3.",
      410: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      420: "Class GPT3OnLearningSimpler inherits from QuickRegressionGPT3.",
      429: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      450: "Class ModelTakingInTextWithSurroundingPieces inherits from SimpleAutogregression.",
      451: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      455: "Class AthleteCompletion inherits from SimpleAutogregression.",
      462: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      476: "Text renders plain text with the default font. Supports substring indexing for partial styling.",
      486: "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level.",
      487: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      488: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      489: "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
      493: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      494: "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
      495: "Smoothly animates the camera to a new orientation over the animation duration.",
      509: "self.play() executes one or more animations simultaneously and waits for them to complete.",
      510: "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
      584: "Class ThatWhichDoesNotKillMe inherits from SimpleAutogregression.",
      591: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

  files["_2024/transformers/supplements.py"] = {
    description: "Supplementary scenes and corrections for the transformer series. Includes additional explanations, clarifications, and bonus visual demonstrations.",
    code: `from manim_imports_ext import *
from _2024.transformers.helpers import *
from _2024.transformers.embedding import *


# Intro chapter


class GPTInitials(InteractiveScene):
    def construct(self):
        # Write name
        frame = self.frame
        name_str = "Generative Pre-trained Transformer"
        name = Text(name_str, font_size=72)
        name.to_edge(UP)
        name.save_state()
        frame.move_to(name).shift(DOWN)
        words = VGroup(name[word][0] for word in name_str.split(" "))
        initials = Text("GPT")
        initials.replace(name, dim_to_match=1)
        t_target = initials["T"][0].generate_target()
        t_target.shift(3 * RIGHT)

        words[0].next_to(initials["P"], LEFT, aligned_edge=DOWN)
        words[1].next_to(t_target, LEFT, aligned_edge=DOWN)

        morty = Mortimer(mode='plain').flip()
        morty.next_to(initials, DL).shift(0.5 * DOWN)
        morty.body.insert_n_curves(100)
        self.add(morty)

        def letter_anim(letters, point):
            for letter in letters:
                letter.save_state()
                letter.set_opacity(0)
                letter.move_to(point)
            return LaggedStart(
                (Restore(letter) for letter in letters),
                lag_ratio=0.05,
                time_span=(0.25, 0.75)
            )

        self.play(
            LaggedStartMap(FadeIn, initials, scale=2, lag_ratio=0.25, run_time=1),
            morty.change("raise_right_hand", initials),
        )
        self.play(Blink(morty))
        self.play(
            ReplacementTransform(initials[0], words[0][0]),
            letter_anim(words[0][1:], initials[0].get_center()),
            morty.animate.look_at(words[0]),
            run_time=1
        )
        self.wait(0.5)
        self.play(
            words[0].animate.next_to(words[1], LEFT, aligned_edge=DOWN),
            Transform(initials[2], t_target),
            ReplacementTransform(initials[1], words[1][0]),
            letter_anim(words[1][1:], initials[1].get_center()),
            morty.change("well", words[1]),
            run_time=1
        )
        self.remove(initials)
        self.wait(0.5)
        self.play(
            Transform(name[:-len(words[2])], name.saved_state[:-len(words[2])]),
            ReplacementTransform(initials[2], words[2][0]),
            letter_anim(words[2][1:], initials[2].get_center()),
            morty.animate.look_at(words[2])
        )
        self.add(name)
        self.play(Blink(morty))
        self.wait()

        # Set up T structure
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(words, DOWN).set_x(0)
        v_lines = Line(UP, DOWN).set_height(FRAME_HEIGHT).replicate(2)
        v_lines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        v_lines.next_to(h_line, DOWN, buff=0)
        t_lines = VGroup(h_line, *v_lines)
        t_lines.set_stroke(GREY_B, 1)

        # Go through each word
        words.target = words.generate_target()
        words.target[0].set_fill(YELLOW)
        words.target[1:].set_fill(WHITE, 0.5, border_width=0)
        offset = FRAME_WIDTH / 3
        words.target[0].set_x(-offset)
        words.target[1].set_x(0)
        words.target[2].set_x(offset)
        line = Underline(words.target[0])
        line.set_stroke(YELLOW)
        self.play(LaggedStart(
            MoveToTarget(words),
            ShowCreation(line),
            frame.animate.center(),
            morty.change("thinking", words.target[0]).set_opacity(0),
            Write(t_lines, stroke_width=2),
            FlashAround(words.target[0].copy())
        ))
        self.remove(morty)
        self.wait()
        for i in [1, 2]:
            words.target = words.generate_target()
            words.target.set_fill(WHITE, 0.5, border_width=0)
            words.target[i].set_fill(YELLOW, 1, border_width=0.5)
            self.play(
                line.animate.become(Underline(words[i])).set_stroke(YELLOW).set_anim_args(run_time=0.75),
                FlashAround(words[i]),
                MoveToTarget(words),
            )
            self.wait()

        # Isolate just Transformer
        self.play(
            words[2].animate.set_x(0).set_color(WHITE).shift(0.25 * UP),
            line.animate.set_x(0).set_color(WHITE).set_width(6).shift(0.25 * UP),
            FadeOut(words[0], LEFT),
            FadeOut(words[1], 3 * LEFT),
            Uncreate(t_lines, lag_ratio=0),
        )
        self.wait()


class DifferentUsesOfModel(InteractiveScene):
    def construct(self):
        # Set up sentences
        sentences = VGroup(
            Text("A machine learning model ..."),
            Text("A fashion model ..."),
        )
        images = Group(
            NeuralNetwork([8, 6, 6, 8]),
            ImageMobject("Zoolander"),
        )
        for sent, image, sign in zip(sentences, images, [-1, 1]):
            sent.set_y(-2)
            sent.set_x(sign * FRAME_WIDTH / 4)
            image.set_width(4)
            image.next_to(sent, UP, buff=0.5)
        images[0].match_y(images[1])
        sentences[0]["model"].set_color(BLUE)
        sentences[1]["model"].set_color(YELLOW)

        # Put word in context
        word = Text("model", font_size=72)
        word.to_edge(UP, buff=0.25)

        self.play(FadeIn(word, UP))
        self.wait()
        self.play(
            FadeTransform(word.copy(), sentences[0]["model"]),
            LaggedStart(
                Write(sentences[0]),
                Write(images[0], lag_ratio=0.01, stroke_width=0.5),
                lag_ratio=0.5,
                run_time=2
            )
        )
        self.wait()
        self.play(
            FadeTransform(word.copy(), sentences[1]["model"]),
            LaggedStart(
                Write(sentences[1]),
                FadeIn(images[1], shift=0.5 * UP, scale=1.25),
                lag_ratio=0.2,
                run_time=2
            )
        )
        self.wait()

        # Show relevance
        s0, s1 = sentences
        path_arc = -0.65 * PI
        left_arrows = VGroup(
            Arrow(
                s0[word].get_top(),
                s0["model"].get_top(),
                path_arc=path_arc
            )
            for word in ["machine", "learning"]
        )
        right_arrow = Arrow(
            s1["fashion"].get_top(),
            s1["model"].get_top(),
            path_arc=path_arc
        )
        left_arrows[0].set_stroke(TEAL, opacity=0.9)
        left_arrows[1].set_stroke(TEAL_D, opacity=0.75)
        right_arrow.set_stroke(TEAL_C, opacity=0.8)

        self.play(
            LaggedStartMap(ShowCreation, left_arrows, lag_ratio=0.5),
            self.frame.animate.move_to(DOWN),
            images.animate.shift(UP),
        )
        self.play(ShowCreation(right_arrow))
        self.wait()

        # Show word vectors
        words = VGroup(
            *(s0[word] for word in s0.get_text().split(" ")[:-1]),
            *(s1[word] for word in s1.get_text().split(" ")[:-1]),
        )
        vectors = VGroup(
            NumericEmbedding().set_height(2).next_to(word, DOWN, buff=0.2)
            for word in words
        )

        self.play(
            LaggedStartMap(FadeIn, vectors, shift=0.25 * DOWN, lag_ratio=0.25, run_time=3)
        )
        self.play(
            LaggedStartMap(RandomizeMatrixEntries, vectors)
        )
        self.wait()


class BigMatrixMultiplication(InteractiveScene):
    mat_dims = (12, 12)
    random_seed = 9

    def construct(self):
        # Test
        matrix = WeightMatrix(shape=self.mat_dims)
        matrix.set_width(FRAME_WIDTH - 4)
        matrix.to_edge(LEFT, buff=0.5)
        vector = NumericEmbedding(length=self.mat_dims[0])
        vector.match_height(matrix)
        vector.next_to(matrix, RIGHT)

        self.add(matrix)
        self.add(vector)
        show_matrix_vector_product(self, matrix, vector)
        self.wait()


class LongListOFQuestions(InteractiveScene):
    def construct(self):
        # Add word and vector
        word = Text("Queen")
        arrow = Vector(0.75 * RIGHT)
        vector = NumericEmbedding(length=12)
        vector.set_height(5)
        word_group = VGroup(word, arrow, vector)
        word_group.arrange(RIGHT, buff=0.15)
        word_group.to_edge(LEFT, buff=2.0)

        self.add(word_group)

        # Add neurons and questions
        questions = VGroup(map(Text, [
            "Is it English?",
            "Is it a noun?",
            "Does it refer to a person?",
            "Is it an amount?",
            "Is A tone assertive",
            "Is it a piece of a bigger word?",
            "Is it part of a quote?",
            "Is it part of a lie?",
        ]))
        questions.scale(0.75)
        n_questions = len(questions)
        neurons = VGroup(Circle(radius=0.2) for n in range(n_questions))
        neurons.add(Tex(R"\\vdots", font_size=72))
        neurons.arrange_in_grid(n_questions, 1, buff_ratio=0.5)
        neurons.set_height(6)
        neurons.set_stroke(WHITE, 1)
        neurons.next_to(vector, RIGHT, buff=3.0)
        values = [0.9, 0.8, 0.85, 0.1, 0.5, 0.05, 0.2, 0.02]
        for neuron, question, value in zip(neurons, questions, values):
            neuron.set_fill(WHITE, value)
            question.next_to(neuron, RIGHT)

        # Add connections
        connections = VGroup(
            VGroup(
                Line(
                    elem.get_right(), neuron.get_center(),
                    buff=neuron.get_width() / 2
                ).set_stroke(
                    color=value_to_color(random.uniform(-10, 10)),
                    width=2 * random.random()
                )
                for elem in vector.get_entries()
            )
            for neuron in neurons[:-1]
        )

        # Animate
        lag_ratio = 0.3
        self.play(
            LaggedStart(
                (ShowCreation(line_group, lag_ratio=0)
                for line_group in connections),
                lag_ratio=lag_ratio,
            ),
            LaggedStartMap(FadeIn, neurons, lag_ratio=lag_ratio),
            LaggedStartMap(FadeIn, questions, lag_ratio=lag_ratio),
            run_time=4
        )
        self.wait()


class ChatBotIcon(InteractiveScene):
    def construct(self):
        # Add bot
        bot = SVGMobject("ChatBot")
        bot.set_fill(GREY_B)
        bot[0].set_stroke(WHITE, 3)
        bot.set_height(3)
        bot.to_edge(RIGHT)

        arrow = Vector(
            1.5 * RIGHT,
            max_tip_length_to_length_ratio=0.4,
            max_width_to_length_ratio=9.0,
        )
        arrow.set_stroke(width=20)
        arrow.next_to(bot, LEFT).match_y(bot[0])

        self.play(
            ShowCreation(arrow),
            Write(bot),
        )
        self.wait()


class GamePlan(InteractiveScene):
    screen_opacity = 0.0

    def construct(self):
        # Setup up icons
        self.add(FullScreenRectangle())
        videos = VideoIcon().get_grid(7, 1, buff_ratio=0.3)
        videos.set_fill(BLUE_B)
        videos.set_height(6.5)
        videos.to_corner(UL)
        column_x = videos.get_x()

        nn_vids = videos[:4]
        tr_vids = videos[4:]
        tr_vids.save_state()
        tr_vids.scale(1.25)
        tr_vids.space_out_submobjects(1.25)
        tr_vids.set_y(0).to_edge(LEFT)

        def highlight_video(video, group=videos):
            for vid in group:
                vid.target = vid.generate_target()
                if vid is video:
                    vid.target.set_x(column_x + 0.5)
                    vid.target.set_opacity(1)
                else:
                    vid.target.set_x(column_x)
                    vid.target.set_opacity(0.5)
            return LaggedStartMap(MoveToTarget, group, lag_ratio=0.01, run_time=1)

        self.add(tr_vids)

        # Here now
        here_arrow = Vector(0.75 * LEFT, stroke_width=10)
        here_arrow.set_color(RED).next_to(tr_vids[0], RIGHT)
        here_words = Text("You are\\nhere")
        here_words.next_to(here_arrow, RIGHT)
        here_words.set_color(RED)
        here_group = VGroup(here_arrow, here_words)

        self.play(
            highlight_video(tr_vids[0], tr_vids),
            MaintainPositionRelativeTo(here_group, tr_vids[0]),
            VFadeIn(here_group),
        )
        self.wait()

        # First chapter
        curly = self.get_curly_brace(tr_vids[0])

        topics = VGroup(
            Text("Beginning"),
            Text("Ending"),
            Text("Background material"),
            Text("Premise of deep learning"),
            Text("Word embeddings"),
            Text("Dot products"),
            Text("Softmax"),
        )
        topics.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        topics.next_to(curly.get_corner(UR), DR, buff=0.25)
        for topic in topics[-4:]:
            topic.scale(0.8, about_edge=LEFT)
            topic.shift(0.5 * RIGHT)
            dot = Dot(color=WHITE)
            dot.next_to(topic, LEFT)
            topic.add(dot)

        screen = ScreenRectangle()
        screen.set_fill(BLACK, 1)
        screen.set_stroke(WHITE, 2)
        screen.set_opacity(self.screen_opacity)
        screen.set_height(5)
        screen.next_to(topics[0], DOWN, aligned_edge=LEFT)

        self.play(
            FadeOut(here_group),
            ShowCreation(curly),
            FadeIn(screen, RIGHT),
            FadeInFromPoint(topics[0], here_group.get_center()),
        )
        self.wait()
        self.play(
            topics[0].animate.set_opacity(0.5),
            FadeIn(topics[1]),
            screen.animate.next_to(topics[1], DOWN, aligned_edge=LEFT)
        )
        self.wait()
        self.play(
            screen.animate.scale(0.5, about_edge=DR).to_edge(RIGHT),
            topics[1].animate.set_opacity(0.5),
            LaggedStartMap(FadeIn, topics[2:], shift=0.1 * DOWN, lag_ratio=0.5)
        )
        self.wait()

        # Second chapter
        new_curly = self.get_curly_brace(tr_vids[1].copy().shift(0.5 * RIGHT))
        screen.target = screen.generate_target()
        screen.target.set_height(5)
        screen.target.next_to(curly, RIGHT)
        att_title = Text("Attention")
        att_title.next_to(screen.target, UP, aligned_edge=LEFT)
        self.play(
            highlight_video(tr_vids[1], tr_vids),
            curly.animate.become(new_curly),
            FadeOut(topics),
            MoveToTarget(screen),
            FadeInFromPoint(att_title, tr_vids[1].get_center()),
        )
        self.wait()

        # Third chapter
        new_curly = self.get_curly_brace(tr_vids[2].copy().shift(0.5 * RIGHT))
        chapter3_topics = Text(
            "MLPs, Training, Positional encodings, ..."
        )
        chapter3_topics.next_to(screen, UP, aligned_edge=LEFT)

        self.play(
            highlight_video(tr_vids[2], tr_vids),
            curly.animate.become(new_curly),
            FadeOut(att_title),
            FadeIn(chapter3_topics, lag_ratio=0.1, time_span=(1, 3)),
        )
        self.wait()

        # Show earlier chapters
        prev_thumbnails = Group(
            ImageMobject(f"nn{k}_thumbnail.png")
            for k in range(1, 5)
        )
        prev_thumbnails.arrange(RIGHT, buff=1.0)
        prev_thumbnails.set_width(FRAME_WIDTH - 2)
        prev_thumbnails.move_to(2 * UP)

        tn_dir = "/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/Thumbnails/"
        new_thumbnails = Group(
            ImageMobject(os.path.join(tn_dir, f"Chapter{n}"))
            for n in range(5, 8)
        )
        for tn1, tn2 in zip(prev_thumbnails, new_thumbnails):
            tn2.replace(tn1, stretch=True)
            tn2.next_to(tn1, DOWN, buff=1.0)
        chapter_titles = VGroup(
            Text(f"Chapter {k}", font_size=30)
            for k in range(1, 8)
        )
        for title, rect in zip(chapter_titles, (*prev_thumbnails, *new_thumbnails)):
            title.next_to(rect, UP, buff=0.2, aligned_edge=LEFT)

        tr_rect = SurroundingRectangle(
            Group(new_thumbnails, chapter_titles[4:]),
            buff=0.25
        )
        tr_rect.set_stroke(BLUE, 2)
        tr_label = Text("Transformers")
        tr_label.next_to(tr_rect, DOWN)

        self.play(
            FadeOut(curly),
            FadeOut(screen),
            FadeOut(chapter3_topics),
            LaggedStartMap(FadeIn, chapter_titles,),
            FadeIn(prev_thumbnails, shift=0.5 * UP, lag_ratio=0.25),
            *(
                FadeTransform(vid, tn)
                for vid, tn in zip(tr_vids, new_thumbnails)
            ),
        )
        self.play(
            ShowCreation(tr_rect),
            FadeIn(tr_label),
        )
        self.wait()

    def get_curly_brace(self, video, width=2.0, height=6.5, buff=0.1):
        start = video.get_right() + buff * RIGHT
        top_point = np.array([start[0] + width, 0.5 * height, 0])
        low_point = np.array([start[0] + width, -0.5 * height, 0])
        result = VGroup(
            CubicBezier(
                start,
                start + width * RIGHT,
                point + width * LEFT,
                point,
            )
            for point in [top_point, low_point]
        )
        result.set_stroke(GREY_A, 2)
        return result


class SkipAhead(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)
        morty = self.teacher
        self.play(
            morty.change("hesitant", self.students),
            self.change_students("confused", "pondering", "pondering", look_at=self.screen),
        )
        self.wait(2)
        self.play(self.change_students("confused", "tease", "well", look_at=morty.eyes))
        self.wait(5)


class SeaOfNumbersUnderlay(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            morty.change("pleading"),
            self.change_students("surprised", "horrified", "droopy")
        )
        self.look_at(3 * LEFT + 2 *  UP)
        self.look_at(3 * RIGHT + 2 * UP)
        self.look_at(3 * LEFT + 2 * UP)

        self.play(
            morty.change("raise_right_hand", self.screen),
            self.change_students("hesitant", "pondering", "maybe", look_at=self.screen)
        )
        self.wait(2)
        self.play(self.change_students("erm", "pondering", "confused", look_at=self.screen))
        self.wait(2)

        self.look_at(5 * RIGHT + 2 * UP)
        self.play(self.change_students("hesitant", "pondering", "hesitant", look_at=5 * RIGHT + 2 * UP))
        self.wait(3)
        self.play(
            morty.change("well"),
            self.change_students("pondering", "pondering", "erm", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            morty.change("raise_left_hand", look_at=5 * RIGHT + 3 * UP),
            self.change_students("tease", "thinking", "pondering", look_at=5 * RIGHT + 3 * UP)
        )
        self.wait(8)


class Outdated(TeacherStudentsScene):
    def construct(self):
        # Add label
        text = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * text.get_height())
        gpt3_label = VGroup(openai_logo, text)
        gpt3_label.arrange(RIGHT)
        gpt3_label.scale(0.75)
        param_count = Text("175B Parameters")
        param_count.set_color(BLUE)
        param_count.next_to(gpt3_label, DOWN, aligned_edge=LEFT)
        gpt3_label.add(param_count)

        gpt3_label.move_to(self.hold_up_spot, DOWN)

        morty = self.teacher
        morty.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            FadeIn(gpt3_label, UP),
        )
        self.play(self.change_students("raise_left_hand", "hesitant", "sassy"))
        self.play(
            self.students[0].says(TexText("Isn't that outdated?"))
        )
        self.wait(3)


class ConvolutionComment(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer()
        morty.to_corner(DR)
        bubble = morty.get_bubble(Text("""
            In other models, the weighted
            sums can be grouped differently,
            e.g. as convolutions, but for
            Transformers it's always
            matrix-vector multiplication.
        """, font_size=36, alignment="LEFT"), bubble_type=SpeechBubble)

        self.add(bubble)
        self.play(morty.change("speaking"))
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


class ConfusionAtScreen(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("well"),
            self.change_students("maybe", "confused", "concentrating", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            self.teacher.change("tease"),
            self.change_students("hesitant", "plain", "erm", look_at=self.teacher.eyes)
        )
        self.wait(3)


class HoldUpExample(TeacherStudentsScene):
    def construct(self):
        self.background.set_fill(opacity=0.0)
        self.teacher.body.insert_n_curves(100)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("happy", "hooray", "well", look_at=4 * UR)
        )
        self.wait(5)


class ReactToWordVectors(InteractiveScene):
    def construct(self):
        # Test
        morty = Mortimer().flip()
        randy = Randolph().flip()
        morty, randy = pis = VGroup(morty, randy)
        pis.arrange(RIGHT, buff=2.0)
        pis.to_edge(DOWN)
        randy.make_eye_contact(morty)

        self.add(pis)
        self.play(
            PiCreatureSays(
                morty, "This is how search\\nworks you know!",
                target_mode="hooray",
                content_introduction_class=FadeIn,
                content_introduction_kwargs=dict(lag_ratio=0.1),
            ),
            randy.change("guilty"),
        )
        self.play(Blink(randy))
        self.wait()
        dots = Text(".....", font_size=120)
        dots[:1].set_opacity(0)
        dots[-1:].set_opacity(0)
        self.play(
            morty.debubble(),
            PiCreatureBubbleIntroduction(
                randy, dots, target_mode="confused",
                bubble_type=ThoughtBubble,
            ),
            morty.change("tease", look_at=6 * LEFT),
        )
        self.play(Blink(morty))
        self.wait()


class DimensionComparrison(InteractiveScene):
    def construct(self):
        titles = VGroup(
            Text("3d vectors"),
            Text("Word vectors"),
        )
        titles.scale(1.5)
        for title, vect in zip(titles, [LEFT, RIGHT]):
            title.move_to(vect * FRAME_WIDTH / 4)
            title.to_edge(UP, buff=MED_SMALL_BUFF)
        h_line = Line(LEFT, RIGHT)
        h_line.set_width(FRAME_WIDTH)
        h_line.next_to(titles, DOWN)
        h_line.set_x(0)
        v_line = Line(UP, DOWN)
        v_line.set_height(FRAME_HEIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(GREY_B, 2)

        self.play(
            ShowCreation(lines, lag_ratio=0.5),
            LaggedStartMap(Write, titles, lag_ratio=0.5)
        )
        self.wait()


class AtLeastKindOf(TeacherStudentsScene):
    def construct(self):
        # Test
        morty = self.teacher
        stds = self.students

        self.play(
            morty.says("...kind of", mode="hesitant"),
            self.change_students("hesitant", "sassy", "erm", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.change_students("sassy", "hesitant", "hesitant", look_at=morty.eyes),
            morty.change("guilty"),
        )
        self.wait(4)


class NetworkEndAnnotation(InteractiveScene):
    opacity = 0.5

    def construct(self):
        im = ImageMobject("NetworkEnd")
        im.set_height(FRAME_HEIGHT)
        self.add(im)

        # word by word
        prof = Text("Professor").set_height(0.25).move_to(np.array([4.77, 3.36, 0.]))
        hp = Text("Harry Potter").set_height(0.33).move_to(np.array([-5.58, 3.33, 0.]))
        lf = Text("least favourite").set_height(0.26).move_to(np.array([1.39, 3.35, 0]))
        snape = Rectangle(3.5, 0.3).move_to(np.array([5.0, 1.11, 0]))

        def get_inverse_rect(mob):
            big_rect = FullScreenFadeRectangle()
            big_rect.scale(1.1)
            lil_rect = SurroundingRectangle(mob)
            big_rect.start_new_path(lil_rect.get_points()[-1])
            big_rect.append_points(lil_rect.get_points()[-2::-1])
            big_rect.set_stroke(WHITE, 1)
            big_rect.set_fill(BLACK, self.opacity)
            return big_rect

        rects = VGroup(map(get_inverse_rect, [prof, hp, lf, snape]))

        rect = rects[0].copy()
        self.play(FadeIn(rect))
        self.wait()
        for rect2 in rects[1:]:
            self.play(Transform(rect, rect2))
            self.wait()
        self.play(FadeOut(rect))


class LowTempHighTempContrast(InteractiveScene):
    def construct(self):
        # Test
        titles = VGroup(
            Text("Temp = 0", font_size=72).set_x(-FRAME_WIDTH / 4),
            Text("Temp = 5", font_size=72).set_x(FRAME_WIDTH / 4),
        )
        titles.to_edge(UP, buff=0.25)
        h_line = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        h_line.next_to(titles, DOWN, buff=0.1)
        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        lines = VGroup(h_line, v_line)
        lines.set_stroke(GREY_B, 2)

        self.play(
            LaggedStartMap(FadeIn, titles, shift=0.25 * UP, lag_ratio=0.25),
            LaggedStartMap(Write, lines, lag_ratio=0.5),
            run_time=1
        )
        self.wait()


class Intuitions(TeacherStudentsScene):
    def construct(self):
        # Add words
        words = VGroup(
            Text("Structure of Deep Learning"),
            Text("Word embeddings"),
            Text("Dot products"),
            Text("Softmax"),
        )
        words.arrange(DOWN, buff=0.5, aligned_edge=LEFT)
        words.move_to(self.hold_up_spot, DOWN)
        checks = VGroup(
            Checkmark(font_size=72).next_to(word, LEFT)
            for word in words
        )
        checks.set_color(GREEN)

        morty = self.teacher
        self.play(
            LaggedStartMap(FadeIn, words, shift=UP, lag_ratio=0.1),
            morty.change("raise_right_hand"),
            self.change_students("thinking", "pondering", "well", look_at=words),
            run_time=1,
        )
        self.play(
            LaggedStartMap(Write, checks, lag_ratio=0.25, stroke_color=GREEN),
        )
        for pi in self.students:
            pi.body.insert_n_curves(100)
        self.play(
            self.change_students("tease", "thinking", "well")
        )
        self.wait(4)


class PiGesturingAtEarlyView(PiCreatureScene):
    def construct(self):
        morty = self.pi_creature.flip()
        morty.to_corner(DR)
        morty.shift(0.5 * LEFT)
        morty.set_color(GREY_BROWN)
        morty.body.insert_n_curves(100)
        for mode in ["raise_right_hand", "well", "gracious", "well", "tease"]:
            self.play(morty.change(mode, ORIGIN + 2 * random.random() * UP))
            self.wait(3)


class EndScreen(PatreonEndScreen):
    pass


# Attention chapter


class HighlightAttentionTitle(TeacherStudentsScene):
    def construct(self):
        # Add image
        im = ImageMobject("AttentionPaper")
        im.set_height(FRAME_HEIGHT)
        title = Text("Attention is All You Need")
        title.set_height(0.219)
        title.move_to(np.array([-0.037, 3.28, 0.0]))
        title.set_fill(BLACK, 1)
        self.clear()
        self.background.set_opacity(0)
        self.add(self.background, im)
        
        self.wait()
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            im.animate.set_opacity(0.1),
            title.animate.set_fill(WHITE).scale(2).next_to(morty, UP, MED_LARGE_BUFF).to_edge(RIGHT),
            LaggedStartMap(VFadeIn, self.pi_creatures),
            morty.change("raise_right_hand"),
            self.change_students("pondering", "well", "thinking", look_at=self.hold_up_spot)
        )
        self.wait()

        # # Small transition
        # alt_title = Text("Attention is all\\nyou need")
        # alt_title.move_to(4.68 * LEFT)

        # self.play(
        #     # TransformMatchingStrings(title, alt_title, run_time=1),
        #     FadeTransformPieces(title, alt_title, run_time=1),
        #     FadeOut(im, scale=2),
        #     self.change_students("pondering", "pondering", "pondering", look_at=alt_title),
        # )
        # self.wait()

        # Highlight attention
        att = title["Attention"][0]
        rest = title["is All You Need"][0]
        self.play(
            FlashAround(att, run_time=2),
            att.animate.set_color(YELLOW),
        )
        self.wait(2)
        self.play(
            att.animate.center().to_edge(UP),
            FadeOut(rest, DR),
            FadeOut(im, scale=1.5),
            self.background.animate.set_opacity(0.75),
            morty.change("tease", 3 * UP),
            self.change_students(None, None, "pondering", look_at=3 * UP)
        )
        self.look_at(3 * UL)
        self.wait()
        self.look_at(3 * UR)
        self.wait(2)

        # Key property
        sentence = Text("What makes Attention powerful is that it's parallelizable")
        sentence.move_to(UP)
        sent_att = sentence["Attention"]
        sent_par = sentence["parallelizable"]
        sent_att.set_opacity(0)
        sent_par.set_opacity(0)
        par_box = SurroundingRectangle(sent_par, buff=0)
        par_box.stretch(1.2, 1, about_edge=DOWN)
        par_box.set_stroke(width=0)
        par_box.set_fill(RED, 0.2)
        par_line = Underline(sent_par, stretch_factor=1)
        par_line.set_stroke(RED, 2)

        self.play(
            att.animate.replace(sentence["Attention"]),
            FadeIn(sentence, lag_ratio=0.1),
            morty.change("raise_right_hand", sentence),
            self.change_students("sassy", "confused", "pondering", look_at=sentence)
        )
        self.play(
            ShowCreation(par_line),
            morty.animate.look_at(par_line),
            FadeIn(par_box),
            self.change_students("pondering", look_at=par_line),
        )
        self.wait(5)


class ThinkOfMoreExamples(TeacherStudentsScene):
    def construct(self):
        # Show general confusion
        morty = self.teacher
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)

        self.play(
            morty.change("raise_right_hand"),
            self.change_students("confused", "maybe", "confused", look_at=3 *UP, run_time=2, lag_ratio=0.25),
        )
        self.wait(2)
        self.play(morty.change("guilty"))
        self.play(
            
            self.change_students("confused", "pleading", "concentrating", look_at=3 * UP, run_time=2, lag_ratio=0.25)
        )
        self.wait(3)
        self.play(
            self.change_students("maybe", "confused", "dejected", look_at=morty.eyes, lag_ratio=0),
            morty.change("well")
        )
        self.wait(2)

        # Ask about the goal
        self.wait()
        self.play(LaggedStart(
            self.students[2].says("What is attention\\nsupposed to do?"),
            self.students[0].change("maybe"),
            self.students[1].change("pondering"),
            morty.change("tease"),
            lag_ratio=0.1
        ))
        self.wait(5)


class SimplerExample(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.change_students("pondering", "thinking", "pondering", look_at=self.screen)
        )
        self.play(
            self.teacher.says("Take a simpler\\nexample"),
            self.change_students("pondering", look_at=self.teacher.eyes)
        )
        self.play(self.change_students("thinking", "well", "tease"))
        self.wait(6)


class NotQuiteTrue(InteractiveScene):
    def construct(self):
        morty = Mortimer()
        morty.to_corner(DR)
        self.play(
            morty.says("Actually, that's not\\nquite true!"),
            run_time=1
        )
        for x in range(2):
            self.play(Blink(morty))
            self.wait()


class ThisIsMadeUp(TeacherStudentsScene):
    def construct(self):
        for pi in self.students:
            pi.change_mode("pondering").look_at(self.screen)
        self.play(
            self.teacher.says("This is a made-up\\nmotivating example"),
            self.change_students("pondering", look_at=self.teacher.eyes)
        )
        self.play(self.change_students("well", "sassy", "guilty", look_at=self.teacher.eyes))
        self.wait(4)


class AskAboutOtherEmbeddings(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[1].says(
                TexText(R"What does $W_Q$ do \\\\ to the non-nouns?"),
                mode="raise_left_hand"
            ),
            self.teacher.change("guilty"),
        )
        self.play(
            self.change_students("confused", None, "pondering", look_at=self.screen)
        )
        self.wait()
        self.play(self.teacher.change("shruggie"))
        self.play(
            self.change_students("sassy", "maybe", "sassy"),
        )
        self.wait(3)


class ShoutSoftmax(TeacherStudentsScene):
    def construct(self):
        self.play(LaggedStart(
            self.students[0].change("happy"),
            self.students[1].change("hooray"),
            self.students[2].says("Softmax!", mode="surprised", bubble_config=dict(buff=0.5, direction=LEFT)),
            self.teacher.change("well")
        ))
        self.wait(5)


class LeftArcSmaller(InteractiveScene):
    def construct(self):
        # Test
        arrow = Arrow(RIGHT, LEFT, path_arc=1.0 * PI, stroke_color=RED, stroke_width=8)
        self.play(ShowCreation(arrow))
        self.wait()


class SetThemToZero(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.students[0].says("Set them to 0?", mode="maybe"),
            self.students[1].change("pondering", look_at=self.screen),
            self.students[1].change("pondering", look_at=self.screen),

        )
        self.wait()
        self.play(
            self.teacher.says("Then they wouldn't\\nbe normalized", mode="tease"),
        )
        self.wait(3)


class CalledMasking(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says(TexText(R"This is called\\\\\`\`masking''")),
            self.change_students(
                "pondering", "confused", "erm", look_at=self.screen,
            )
        )
        self.wait(5)


class ReferenceLargerContextTechnologies(InteractiveScene):
    def construct(self):
        # Test
        words = VGroup(
            Text("Sparse Attention Mechanisms"),
            Text("Blockwise Attention"),
            Text("Linformer"),
            Text("Reformer"),
            Text("Ring attention"),
            Text("Longformer"),
            Text("Adaptive Attention Span"),
            Tex(R"\\vdots")
        )
        words.arrange(DOWN, aligned_edge=LEFT, buff=MED_LARGE_BUFF)
        words[-1].shift(0.5 * RIGHT)

        self.play(
            LaggedStartMap(FadeIn, words, shift=0.5 * DOWN, lag_ratio=0.5, run_time=4)
        )
        self.wait()
        self.play(
            LaggedStartMap(FadeOut, words, shift=RIGHT, lag_ratio=0.1)
        )
        self.wait()


class AskAboutCrossAttention(TeacherStudentsScene):
    def construct(self):
        stds = self.students
        self.play(
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[1].says("What about\\ncross-attention?", bubble_config=dict(buff=0.5), mode="raise_left_hand"),
            stds[2].change("pondering", look_at=stds[1].eyes),
            self.teacher.change("well", look_at=stds[1].eyes)
        )
        self.wait(5)


class SelfVsCrossFrames(InteractiveScene):
    def construct(self):
        # Add screens
        self.add(FullScreenRectangle())
        screens = ScreenRectangle().replicate(2)
        screens.set_fill(BLACK, 1)
        screens.set_stroke(WHITE, 2)
        screens.set_height(0.45 * FRAME_HEIGHT)
        screens.arrange(RIGHT, buff=0.5)
        self.add(screens)

        # Add titles
        titles = VGroup(
            Text("Self-attention", font_size=60),
            Text("Cross-attention", font_size=60),
        )
        for title, screen in zip(titles, screens):
            title.next_to(screen, UP, buff=MED_LARGE_BUFF)

        self.play(Write(titles[0]))
        self.wait()
        self.play(TransformMatchingStrings(titles[0].copy(), titles[1]))
        self.wait()


class OngoingTranscription(InteractiveScene):
    def construct(self):
        phrase = Text("or maybe audio input of speech, and an ongoing transcription")
        words = break_into_words(phrase)
        for word in words:
            self.add(word)
            self.wait(0.1 * len(word))
        self.wait()


class ReferenceStraightforwardValueMatrix(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        morty.body.insert_n_curves(100)
        self.play(
            morty.change("raise_right_hand"),
            self.change_students("happy", "well", "tease", look_at=3 * UR)
        )
        self.wait(3)
        self.play(
            morty.change("hesitant"),
            self.change_students("erm", "hesitant", "guilty", look_at=3 * UR)
        )
        self.wait(5)


class SeekingMatchedParameters(TeacherStudentsScene):
    def construct(self):
        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        equation = VGroup(
            Text("# Value params").set_color(RED),
            Tex("=", font_size=72).rotate(PI / 2),
            Text("(# Query params) + (# Key params)"),
        )
        equation[2].scale(0.75)
        equation[2]["# Query params"].set_color(YELLOW)
        equation[2]["# Key params"].set_color(TEAL)
        equation.arrange(DOWN, buff=MED_LARGE_BUFF)
        equation.move_to(self.hold_up_spot, DOWN)

        self.play(
            self.teacher.change("raise_right_hand", equation),
            FadeIn(equation, UP),
            self.change_students("erm", "confused", "sassy", look_at=equation),
        )
        self.wait(2)
        self.play(
            self.change_students("pondering", "confused", "hesitant", look_at=self.screen)
        )
        self.wait(4)
        self.play(
            self.change_students("erm", "confused", "sassy", look_at=equation)
        )
        self.wait(4)


class HeadName(InteractiveScene):
    def construct(self):
        # Test
        title = Text("One head of attention", font_size=72)
        title.to_edge(UP)
        head = title["head"][0]
        self.play(
            Write(title, run_time=1)
        )
        self.play(
            FlashAround(head, time_width=2, run_time=2),
            head.animate.set_color(YELLOW),
        )
        self.wait()


class DInputAndOutputOfValue(InteractiveScene):
    def construct(self):
        # Test
        d_embed = 12_288
        in_label, out_label = [
            VGroup(Text(text), Integer(d_embed))
            for text in ["d_input", "d_output"]
        ]
        for label, shift in [(in_label, LEFT), (out_label, RIGHT)]:
            label.arrange(DOWN)
            label.scale(0.65)
            label.next_to(ORIGIN, UP, buff=LARGE_BUFF)
            label.shift(1.0 * shift)
            arrow = Arrow(label, 0.5 * shift)
            label.add(arrow)

        self.play(FadeIn(in_label, lag_ratio=0.1))
        self.wait()
        self.play(FadeIn(out_label, lag_ratio=0.1))
        self.wait()


class NowRepeatManyTimes(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("pondering", "pondering", "pondering", look_at=self.screen),
        )
        self.wait()
        self.play(
            self.teacher.says("Now do that about\\n10,000 times"),
            self.change_students("droopy", "erm", "well", look_at=self.teacher.eyes)
        )
        self.wait(5)


class ALotToHoldInYouHead(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.says("It's a lot to\\nhold in your head!", mode="surprised"),
            self.change_students("confused", "erm", "dejected", look_at=self.screen),
        )
        self.wait(5)


class ReactToMHSA(TeacherStudentsScene):
    def construct(self):
        self.play(
            self.teacher.change("hesitant"),
            self.change_students("sad", "confused", "dejected", look_at=self.screen)
        )
        self.wait(3)
        self.play(
            self.change_students("guilty", "maybe", "erm")
        )
        self.wait(3)


class AskAboutOutput(TeacherStudentsScene):
    random_seed = 3
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.play(
            stds[0].change("hesitant", look_at=stds[1].eyes),
            stds[1].says("What about the\\nOutput matrix?", mode="raise_left_hand"),
            stds[2].change("hesitant", look_at=stds[1].eyes),
        )
        self.play(
            morty.change("concentrating")
        )
        self.play(Blink(morty))
        self.wait(5)


class OneThirdOfWhatYouNeed(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle().fix_in_frame())
        title = Text("Attention is All You Need", font_size=72)
        all_word = title["All"][0]
        cross = Line(all_word.get_left(), all_word.get_right())
        cross.set_stroke(RED, 8)
        correction = Text("About 1/3 of What", font_size=60)
        correction.set_color(RED)
        correction.next_to(all_word, UP, MED_LARGE_BUFF)
        lines = VGroup(
            CubicBezier(
                all_word.get_corner(UP + v),
                all_word.get_corner(UP + v) + 0.5 * UP,
                correction.get_corner(DOWN + v) + 0.5 * DOWN,
                correction.get_corner(DOWN + v),
            )
            for v in [LEFT, RIGHT]
        )
        lines.set_stroke(RED, 2)

        self.add(title)
        self.wait()
        self.add(all_word, cross)
        self.play(ShowCreation(cross), all_word.animate.set_fill(opacity=0.5))
        self.play(
            FadeTransform(all_word.copy(), correction),
            ShowCreation(lines, lag_ratio=0),
        )
        self.wait()
        self.play(self.frame.animate.set_y(-3.75).set_height(11), run_time=2)
        self.wait()


class MoreResourcesBelow(InteractiveScene):
    def construct(self):
        # Test
        self.add(FullScreenRectangle())
        words = Text("More resources below", font_size=72)
        words.move_to(UP)
        arrows = Vector(1.5 * DOWN, stroke_width=10).get_grid(1, 3, buff=1.5)
        arrows.next_to(words, DOWN, buff=MED_LARGE_BUFF)
        morty = Mortimer()
        morty.body.insert_n_curves(100)
        morty.to_corner(DR)

        self.add(words)
        self.play(
            LaggedStartMap(GrowArrow, arrows, lag_ratio=0.5),
            morty.change("thinking", look_at=4 * DOWN)
        )
        self.play(Blink(morty))
        self.wait()


class PatreonEndScreen(EndScreen):
    pass


# MLP Chapter


class HowAndWhere(TeacherStudentsScene):
    def construct(self):
        # Test
        self.remove(self.background)
        self.play(
            self.students[1].says("How?", mode="raise_left_hand", look_at=self.screen),
            self.teacher.change("tease"),
            self.students[2].change("pondering", look_at=self.screen),
        )
        self.play(
            self.students[0].says("Where?", mode="maybe", look_at=self.screen),
        )
        self.wait(3)


class IntroducingMLPs(TeacherStudentsScene):
    def construct(self):
        # Look at screen
        morty = self.teacher
        screen = self.screen
        self.play(
            morty.change("raise_right_hand", screen),
            self.change_students("pondering", "confused", "pondering", look_at=screen),
        )
        self.wait(2)

        # Computation vs. interpretation
        words = VGroup(Text("Computation"), Text("Interpretation"))
        words.scale(1.5)
        words.arrange(DOWN, buff=1.0, aligned_edge=LEFT)
        words.to_corner(UR).shift(LEFT)
        check = Checkmark()
        check.match_height(words[0])
        check.next_to(words[0], RIGHT)
        check.set_color(GREEN)
        warning = SVGMobject("warning")
        warning.set_color(RED)
        warning.match_height(words[1])
        warning.next_to(words[1], RIGHT)

        self.play(
            FadeIn(words[0], UP),
            self.change_students("tease", "happy", "thinking", look_at=words[0]),
            morty.change("raise_left_hand", words[0]),
        )
        self.play(Write(check, stroke_color=GREEN))
        self.wait(3)
        self.play(
            FadeIn(words[1], UP),
            self.change_students("erm", "confused", "pondering", words[1]),
            morty.change("maybe", words[1])
        )
        self.play(Write(warning, stroke_color=RED))
        self.wait(5)


class ReferenceFactStorage(TeacherStudentsScene):
    def construct(self):
        # Look at screen
        morty = self.teacher
        screen = self.screen
        self.play(
            morty.change("raise_right_hand", screen),
            self.change_students("pondering", "confused", "pondering", look_at=screen),
        )
        self.wait(4)

        # Hold up words
        words = Text("Store a fact", font_size=72)
        words.next_to(morty, UP, LARGE_BUFF).shift_onto_screen()

        self.play(
            morty.change("raise_left_hand"),
            FadeIn(words, UP),
            self.change_students("erm", "maybe", "sassy", look_at=morty.eyes),
        )
        self.look_at(self.screen)
        self.wait(3)
        self.play(morty.change("tease"))
        self.wait(2)

        # Relax
        self.play(
            FadeOut(words, DOWN),
            self.change_students("pondering", "tease", "happy", look_at=self.screen),
        )
        self.wait(3)


class LookingAtPreview(TeacherStudentsScene):
    def construct(self):
        # Test
        bubble = ThoughtBubble(filler_shape=(5, 2.5))
        bubble.flip()
        bubble.pin_to(self.students[2])
        bubble.to_edge(LEFT)
        self.play(
            self.teacher.change("raise_right_hand"),
            self.change_students("pondering", "confused", "pondering", look_at=bubble),
        )
        self.play(FadeIn(bubble, lag_ratio=0.1))
        self.play(self.teacher.change("tease"))
        self.wait(2)
        self.play(self.change_students("erm", "pondering", "thinking", look_at=bubble))
        self.wait(3)


class RefreshersNeverHurt(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.change_students("confused", "horrified", "sad", look_at=self.screen),
        )
        self.wait(2)
        self.play(
            self.teacher.says(TexText(R"Let's do a\\\\quick refresher"), mode="tease"),
            self.change_students("pondering", "hesitant", "erm", look_at=self.teacher.eyes)
        )
        self.wait(3)


class EmbeddingLabel(InteractiveScene):
    def construct(self):
        # Background
        bg = ImageMobject("/Users/grant/3Blue1Brown Dropbox/3Blue1Brown/videos/2024/transformers/images/EmbeddingStill.jpg")
        bg.set_height(FRAME_HEIGHT)
        # self.add(bg)

        # Label
        ghost_vect = Rectangle()
        ghost_vect.set_shape(0.8, 4)
        ghost_vect.move_to([4.25, -1.0, 0])

        brace = Brace(ghost_vect, LEFT)
        name = brace.get_text("Embedding")
        length = Integer(12288)
        length.next_to(brace, LEFT, buff=0.5).shift(0.25 * UP)
        numbers_label = Text("Numbers")
        numbers_label.next_to(length, DOWN)
        gpt3_label = Text("(Length in GPT-3)", font_size=24)
        gpt3_label.next_to(length, UP, buff=1.0),
        gpt3_label.set_color(YELLOW)
        arrow = Arrow(gpt3_label.get_bottom(), length.get_top(), buff=0.1)

        self.play(
            GrowFromCenter(brace),
            Write(name)
        )
        self.wait()
        self.play(
            FadeTransform(name, numbers_label),
            CountInFrom(length, 0, run_time=1.5),
        )
        self.play(
            FadeIn(gpt3_label, lag_ratio=0.1),
            GrowFromCenter(arrow),
        )
        self.wait(1.5)


class ThatWhichDoesntKillHeader(InteractiveScene):
    def construct(self):
        # Test
        words = Text("That which does not kill you only makes you")
        words.to_edge(UP)
        rect = SurroundingRectangle(words["you"][-1], buff=0.1)
        rect.set_stroke(BLUE, 3)
        rect.set_fill(BLUE, 0.5)
        arrow = Arrow(rect.get_bottom(), rect.get_bottom() + 2 * DL)
        arrow.match_color(rect)

        brace = Brace(rect, DOWN, buff=0.1)

        self.add(rect, words, brace)


class QuickAttentionDescription(InteractiveScene):
    def construct(self):
        # To be added standing on an Attention block
        morty = Mortimer(height=2)
        morty.move_to(DOWN + LEFT)
        morty.flip()
        self.play(morty.says("Incorporate context", look_at=4 * DOWN))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class QuickMLPDescription(InteractiveScene):
    def construct(self):
        # To be added standing on an MLP block
        morty = Mortimer(height=2, color=GREY_C)
        morty.move_to(DOWN + RIGHT)
        self.play(morty.says("More\\ncomputation", mode="maybe", look_at=4 * DOWN))
        for x in range(2):
            self.play(Blink(morty))
            self.wait(2)


class ContrastBetweenSimpleComputationDifficultInterpretation(InteractiveScene):
    def construct(self):
        morty = Mortimer(height=2).to_edge(DOWN, buff=1.0)
        morty.body.insert_n_curves(100)
        items = VGroup(
            VGroup(Text("Computation"), Checkmark().set_height(0.5).set_color(GREEN)),
            VGroup(Text("Interpretation"), SVGMobject("warning").set_color(RED).set_height(0.5)),
        )
        for item, vect in zip(items, [LEFT, RIGHT]):
            item.scale(0.75)
            item.arrange(RIGHT)
            item.next_to(morty, UP + vect, buff=0.5)
            item.shift(-1.0 * vect * RIGHT)


        self.play(
            morty.change("raise_right_hand", items[0]),
            FadeIn(items[0], UP)
        )
        self.play(Blink(morty))
        self.wait()
        self.play(
            morty.change("raise_left_hand", items[1]),
            FadeIn(items[1], UP),
            items[0].animate.fade(0.5),
        )
        self.play(Blink(morty))
        self.wait()


class AmbientChangingDots(InteractiveScene):
    def construct(self):
        # Test
        dots = Dot().get_grid(20, 30)
        dots.set_height(8)
        dots.set_fill(opacity=0.5)
        dots.phases = np.random.uniform(0, TAU, len(dots))
        dots.freqs = np.random.uniform(0.3, 0.8, len(dots))

        def update_dots(dots):
            for dot, phase, freq in zip(dots, dots.phases, dots.freqs):
                dot.set_fill(opacity=np.cos(phase + freq * self.time)**2)
            return dots

        dots.add_updater(update_dots)
        self.add(dots)
        self.wait(30)


class MakeSomeAssumptions(TeacherStudentsScene):
    def construct(self):
        # Test
        self.play(
            self.teacher.says("We need a\\nfew assumptions"),
            self.change_students("pondering", "sassy", "tease", look_at=self.screen)
        )
        self.play(self.teacher.change("raise_right_hand"))
        self.wait(6)


class MovingToSecondToken(InteractiveScene):
    def construct(self):
        # Add phrase
        phrase = Text("Michael Jordan plays the sport of")
        phrase.to_edge(UP)
        tokens = break_into_tokens(phrase)
        rects = get_piece_rectangles(tokens, h_buff=0, leading_spaces=True)

        self.add(tokens, rects)

        # Add vectors
        embeddings = VGroup(
            NumericEmbedding().scale(0.5).next_to(rect, DOWN, LARGE_BUFF)
            for rect in rects
        )
        arrows = VGroup(Arrow(rect, emb, buff=0.1) for rect, emb in zip(rects, embeddings))

        self.add(arrows)
        self.add(embeddings)

        # Animate in
        self.play(
            Write(rects),
            LaggedStartMap(GrowArrow, arrows),
            LaggedStartMap(FadeIn, embeddings, shift=0.5 * DOWN),
        )
        self.wait()

        # Highlight two
        highlight_rect = SurroundingRectangle(VGroup(rects[:2], embeddings[:2]))

        self.play(
            ShowCreation(highlight_rect),
            tokens[2:].animate.fade(0.5),
            rects[2:].animate.fade(0.5),
            arrows[2:].animate.fade(0.5),
            embeddings[2:].animate.fade(0.5),
        )
        self.wait()

        # Attention
        self.play(
            LaggedStart(
                (ContextAnimation(e2, embeddings[0].get_entries(), path_arc=90 * DEGREES, lag_ratio=0.1, min_stroke_width=2)
                for e2 in embeddings[1].get_entries()),
                lag_ratio=0.1
            ),
            RandomizeMatrixEntries(embeddings[1]),
            run_time=4
        )
        self.play(
            highlight_rect.animate.surround(VGroup(rects[1], embeddings[1]), buff=0)
        )
        self.wait()


class WhatAboutBiggerThanOne(TeacherStudentsScene):
    def construct(self):
        # Test
        self.screen.set_x(0)
        self.play(
            self.students[0].change("pondering", self.screen),
            self.students[1].says("And if it's\\nbigger than 1?", mode="sassy", bubble_direction=RIGHT),
            self.students[2].change("erm", self.screen),
            self.teacher.change("guilty"),
        )
        self.wait(2)
        self.play(
            # self.teacher.says("Don't worry\\nabout it", mode="maybe")
            self.teacher.change("maybe")
        )
        self.play(
            self.change_students("hesitant", "sassy", "angry")
        )
        self.wait(3)


class HighlightRect(InteractiveScene):
    def construct(self):
        rect = Rectangle(1, 3)
        rect.scale(0.5)
        rect.set_stroke(MAROON_B, 3)
        self.play(ShowCreation(rect))
        self.wait()


class AskWhy(TeacherStudentsScene):
    def construct(self):
        self.remove(self.background)

        # Test
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.teacher.change("guilty"),
            self.students[2].says(Text("Um, Why?", font_size=72), mode="maybe", look_at=self.screen),
            self.students[0].change("confused", self.screen),
            self.students[1].change("erm", self.screen),
        )
        self.wait()
        self.play(self.teacher.change("tease"))
        self.wait(2)


class GPT3Logo(InteractiveScene):
    def construct(self):
        # Test
        gpt3_label = Text("GPT-3", font="Consolas", font_size=72)
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(2.0 * gpt3_label.get_height())
        title = VGroup(openai_logo, gpt3_label)
        title.arrange(RIGHT)
        title.to_edge(UP)

        self.play(
            FadeIn(gpt3_label, lag_ratio=0.1),
            Write(openai_logo, stroke_color=BLUE, stroke_width=0.5),
        )
        self.wait()


class AndGate(InteractiveScene):
    def construct(self):
        # Test
        gate = SVGMobject("and_gate")
        gate.set_fill(WHITE).set_stroke(width=0)
        name = Text("AND\\nGate", font_size=96, alignment="LEFT")
        name.next_to(gate, RIGHT, LARGE_BUFF)
        self.play(
            Write(gate),
            FadeIn(name, lag_ratio=0.1, time_span=(0, 2)),
            run_time=3
        )
        self.wait()


class MJFactsAsVectorSum(InteractiveScene):
    def construct(self):
        # Test
        facts = VGroup(
            Tex(Rf"\\overrightarrow{{\\text{{{fact}}}}}")
            for fact in [
                "Basketball",
                "Chicago Bulls",
                "Number 23",
                "Born 1963",
            ]
        )
        facts.add(Tex(R"\\vdots"))
        facts.arrange(DOWN, buff=0.75)
        colors = ["#F88158", "#CE1141", YELLOW, GREY, WHITE]
        for fact, color in zip(facts, colors):
            fact.set_color(color)

        plusses = Tex(R"+").replicate(len(facts) - 1)
        for f1, f2, plus in zip(facts, facts[1:], plusses):
            plus.move_to(midpoint(f1.get_bottom(), f2.get_top()))

        self.add(facts[0])
        for fact, plus in zip(facts[1:], plusses):
            self.play(
                FadeIn(fact, shift=0.5 * DOWN),
                Write(plus),
                run_time=1,
            )
            self.wait()


class AskAboutBias(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.students[0].change("erm", look_at=self.screen),
            self.students[1].change("confused", look_at=self.screen),
            self.students[2].says("What's that\\nbias doing?", look_at=self.screen, bubble_direction=LEFT),
        )
        self.play(
            self.teacher.change("maybe")
        )
        self.wait(4)
        self.play(
            self.change_students("sassy", "maybe", "pondering", look_at=self.screen)
        )
        self.wait(4)


class ThatsIt(TeacherStudentsScene):
    def construct(self):
        for pi in self.pi_creatures:
            pi.body.insert_n_curves(100)
        self.play(
            self.teacher.says("That's it!", mode="hooray", look_at=self.students),
            self.change_students("happy", "thinking", "well", look_at=self.screen)
        )
        self.wait()
        self.play(self.teacher.debubble(mode="raise_right_hand", look_at=self.screen))
        self.wait()
        self.play(
            self.change_students("thinking", "tease", "happy", look_at=self.screen)
        )
        self.wait(3)


class AddTwoMatrixSizes(InteractiveScene):
    def construct(self):
        # Test
        rect = Rectangle(3.0, 1.0)
        rect.set_stroke(BLUE, 3)
        total = Integer(2 * 4 * (12288**2))
        total.set_color(BLUE)
        total.next_to(rect, UP)

        self.play(
            ShowCreation(rect),
            Write(total)
        )
        self.wait()


class ReflectOnTwoThings(TeacherStudentsScene):
    def construct(self):
        # Initial reactions
        morty = self.teacher
        screen = self.screen
        stds = self.students

        morty.change_mode("raise_right_hand").look_at(self.screen)
        for std in stds:
            std.change_mode("happy")
        self.play(
            self.change_students("pondering", "thinking", "happy", look_at=screen)
        )
        self.wait(2)

        # Reflection points
        points = VGroup(
            Text("Two points of reflection"),
            Text("1."),
            Text("2."),
        )

        points[0].add(Underline(points[0], buff=-0.05))
        points[0].scale(1.25)
        points[0].set_color(YELLOW)

        dials = VGroup(Dial(initial_value=random.random()) for n in range(10))
        dials.set_height(0.5)
        dials.arrange(RIGHT)
        dials.set_flat_stroke(True)
        dials[-2].become(Tex(R"\\dots").replace(dials[-2], dim_to_match=0))
        dials.next_to(points[1], RIGHT)
        points[1].add(dials)

        vectors = self.get_vectors()
        points[2].add(vectors)

        points.arrange(DOWN, aligned_edge=LEFT, buff=0.5)
        points.to_edge(UP)
        points[1:].shift(MED_SMALL_BUFF * RIGHT + 0.5 * DOWN)

        self.play(
            morty.change("tease", points[0]),
            self.change_students("erm", "plain", "hesitant", look_at=points[0]),
            Write(points[0], stroke_color=YELLOW_B),
        )
        self.wait(2)
        self.play(
            Write(points[1][:2]),
            LaggedStartMap(FadeIn, dials, lag_ratio=0.25),
            self.change_students("tease", "plain", "erm", look_at=points[1]),
            morty.change("raise_right_hand", points[1]),
        )
        self.wait(2)

        # Show vector clump
        self.play(
            VFadeIn(points[2]),
            Rotate(vectors, PI, axis=UP, run_time=8),
            self.change_students("confused", "hesitant", "erm", look_at=points[2]),
            morty.change("surprised", points[2]),
        )
        self.wait(3)

    def get_vectors(self):
        dodec = Dodecahedron()
        vectors = VGroup()
        for face in dodec:
            # for vert in face.get_anchors():
            for vert in [face.get_center()]:
                if not any([np.isclose(vert, v.get_end()).all() for v in vectors]):
                    vect = Vector(vert)
                    vect.set_color(random_bright_color(hue_range=(0.5, 0.7)))
                    vect.always.set_perpendicular_to_camera(self.frame)
                    vectors.add(vect)
        vectors.rotate(25 * DEGREES, axis=UR)
        vectors.set_height(1.5)
        return vectors


class RotatingVectors(ReflectOnTwoThings):
    def construct(self):
        self.clear()
        # Test
        vectors = self.get_vectors()
        vectors.set_height(4)

        self.play(
            Rotate(vectors, TAU, axis=UP, run_time=25, rate_func=linear),
        )


class AskIfThisIsReal(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students

        # Test
        self.play(
            stds[0].says("Is this how ChatGPT stores facts?"),
            morty.change("well"),
        )
        self.look_at(self.screen)
        self.wait()
        self.play(
            self.change_students("speaking", "pondering", "skeptical", look_at=self.screen)
        )
        self.wait(2)
        self.play(
            morty.says("Almost certainly\\nnot quite...", mode="guilty", bubble_direction=RIGHT),
            stds[1].change("angry"),
            stds[2].change("erm"),
        )
        self.wait(4)


class SingleNeuronVsMultiple(InteractiveScene):
    def construct(self):
        # Add network
        radius = 0.1
        layers = VGroup(
            Dot(radius=radius).get_grid(n, 1, buff=radius)
            for n in [8, 16, 8]
        )
        layers.arrange(RIGHT, buff=2.0)
        layers.set_stroke(WHITE, 1)
        for layer in layers:
            for dot in layer:
                dot.set_fill(opacity=random.random())

        connections = VGroup(
            get_network_connections(layers[i], layers[i + 1])
            for i in (0, 1)
        )

        network = VGroup(layers, connections)
        network.set_height(5)
        network.center()

        self.add(network)

        # Show first neuron light up
        rect = SurroundingRectangle(layers[1][0])
        name = Text("Michael Jordan")
        name.next_to(rect, UP, SMALL_BUFF)
        name.save_state()
        for letter, dot in zip(*make_even(name, layers[0])):
            letter.move_to(dot)
            letter.set_opacity(0)

        thick_connections = connections.copy()
        for group in thick_connections:
            for line in group:
                line.set_stroke(width=2 * line.get_stroke_width(), opacity=1)
                line.insert_n_curves(20)
        self.play(
            LaggedStartMap(
                VShowPassingFlash,
                thick_connections[0],
                lag_ratio=1 / len(thick_connections[0]),
                time_width=2.0,
            ),
            layers[1][0].animate.set_fill(opacity=1),
            layers[1][1:].animate.set_fill(opacity=0),
            Restore(name, lag_ratio=0.05),
            run_time=2
        )
        self.play(ShowCreation(rect))
        self.wait()
        network.add(rect, name)

        # Split the image
        network_copy = network.copy()
        for dot in network_copy[0][1]:
            dot.set_fill(opacity=random.random())
        network_copy.to_edge(RIGHT)
        network_copy[-2].become(SurroundingRectangle(network_copy[0][1]))

        v_line = Line(UP, DOWN).set_height(FRAME_HEIGHT)
        v_line.set_stroke(WHITE, 2)
        check = Checkmark().set_fill(GREEN).scale(2)
        ex = Exmark().set_fill(RED).scale(2)
        check.next_to(network_copy[-1], RIGHT)
        ex.move_to(check).shift(0.5 * FRAME_WIDTH * LEFT)

        self.play(
            network.animate.to_edge(LEFT),
            TransformFromCopy(network, network_copy),
            ShowCreation(v_line),
        )
        self.play(LaggedStart(
            Write(ex, stroke_color=RED),
            Write(check, stroke_color=GREEN),
            lag_ratio=0.5
        ))
        self.wait()


class WriteSuperposition(InteractiveScene):
    def construct(self):
        # Test
        word = Text("Superposition", font_size=120)
        outline = word.copy()
        outline.set_stroke(TEAL, 3)
        outline.set_fill(opacity=0)

        self.play(
            FadeIn(word, lag_ratio=0.1),
            LaggedStartMap(
                VShowPassingFlash,
                outline,
                time_width=2,
                run_time=5,
                lag_ratio=0.01
            )
        )
        self.wait()


class JohnsonLindenstraussName(InteractiveScene):
    def construct(self):
        # Test
        text = VGroup(
            Text("Johnson–Lindenstrauss\\nLemma"),
            Tex(R"\\Rightarrow", font_size=120),
        )
        text[0].set_color(RED_B)
        text.arrange(RIGHT, buff=MED_LARGE_BUFF)
        self.play(
            FadeIn(text[0], lag_ratio=0.1),
            Write(text[1], run_time=1)
        )


class ContrastGPTDimensionSizes(InteractiveScene):
    def construct(self):
        # Setup
        openai_logo = SVGMobject("OpenAI.svg")
        openai_logo.set_fill(WHITE)
        openai_logo.set_height(1.0)

        model_names = VGroup(
            Text("GPT-2"),
            Text("GPT-3"),
            Text("GPT-4"),
        )
        model_names.scale(1.25)
        model_names.arrange(RIGHT, buff=2.0)
        model_names.set_color(GREY_A)
        arrows = VGroup(
            Arrow(n1, n2, buff=0.25)
            for n1, n2 in zip(model_names, model_names[1:])
        )
        dim_counts = VGroup(
            Text(f"Model dim: {dim}", font_size=36)
            for dim in ["768", "12,288", "???"]
        )
        for model, count in zip(model_names, dim_counts):
            count.next_to(model, DOWN)

        arrows.add_to_back(Arrow().set_opacity(0))
        for name, count, arrow in zip(model_names, dim_counts, arrows):
            self.play(
                FadeIn(name),
                FadeIn(count, 0.5 * DOWN),
                GrowArrow(arrow)
            )
        self.wait()


class ReferenceSAP(TeacherStudentsScene):
    def construct(self):
        morty = self.teacher
        stds = self.students
        self.background.scale(2)
        self.frame.scale(1.25, about_edge=DR)

        # Test
        bubble = stds[0].get_bubble("How would you\\ntest this?", bubble_type=SpeechBubble)
        bubble.shift(0.5 * LEFT)
        self.play(LaggedStart(
            FadeIn(bubble, lag_ratio=0.1),
            stds[0].change("raise_left_hand"),
            stds[1].change("confused"),
            stds[2].change("maybe"),
            morty.change("tease")
        ))
        self.wait(2)
        self.play(
            morty.says(TexText(R"There's nice\\\\research using\\\\Sparse Autoencoders"), mode="hooray")
        )
        self.play(
            self.change_students(None, "pondering", "erm", look_at=morty.bubble),
        )
        self.wait()
        self.look_at(self.screen)
        self.wait(5)


class DetailsNotDiscussed(InteractiveScene):
    def construct(self):
        # Test
        title = Text("Details not discussed")
        title.add(Underline(title))
        title.scale(1.25)
        title.set_color(RED)
        title.to_edge(UP).to_edge(RIGHT, buff=0)

        details = VGroup(
            Text("Tokenization"),
            Text("Positional encoding"),
            Text("Layer normalization"),
            Text("Training"),
        )
        dots = Dot().get_grid(len(details), 1, buff=0.75)
        dots.next_to(title, DOWN, buff=MED_LARGE_BUFF).shift(2 * LEFT)
        for detail, dot in zip(details, dots):
            detail.next_to(dot, RIGHT)
            detail.add_to_back(dot)
        vdots = Tex(R"\\vdots")
        vdots.next_to(details, DOWN, MED_LARGE_BUFF).shift(LEFT)
        details.add(vdots)
        details.set_color(GREY_A)

        self.add(title)
        self.play(
            LaggedStartMap(FadeIn, details, shift=0.25 * DOWN, lag_ratio=0.5),
            run_time=4
        )
        self.wait()

        # Highlight training
        self.play(
            details[-2][1:].animate.scale(2, about_edge=LEFT).set_color(WHITE),
            details[-1].animate.shift(0.1 * DOWN),
            details[:-2].animate.set_opacity(0.5).scale(0.9, about_edge=UL),
        )
        self.wait()


class TriPanelWithPi(InteractiveScene):
    def construct(self):
        vlines = Line(UP, DOWN).replicate(2)
        vlines.set_height(FRAME_HEIGHT / 2)
        vlines.arrange(RIGHT, buff=FRAME_WIDTH / 3)
        vlines.to_edge(UP, buff=0)
        hline = Line(LEFT, RIGHT).set_width(FRAME_WIDTH)
        hline.move_to(vlines, DOWN)
        lines = VGroup(*vlines, hline)

        lines.set_stroke(WHITE, 2)
        self.add(lines)

        # Test
        morty = Mortimer(mode="happy")
        morty.to_corner(DR).shift(3 * LEFT)

        self.play(morty.change("tease", 4 * UL))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("coin_flip_2", 3 * UP))
        self.play(Blink(morty))
        self.wait()
        self.play(morty.change("hooray", 5 * UR).set_anim_args(path_arc=10 * DEGREES))
        self.play(Blink(morty))
        self.wait()


class WriteRLHF(InteractiveScene):
    def construct(self):
        words = VGroup(
            Text("Reinforcement"),
            Text("Learning with"),
            Text("Human"),
            Text("Feedback"),
        )
        words.scale(1.5)
        words.arrange(DOWN, aligned_edge=LEFT)
        words.set_color(GREY_A)
        self.play(LaggedStartMap(FadeIn, words, shift=0.25 * DOWN, lag_ratio=0.25))
        self.play(*(
            word[1:].animate.set_opacity(0.75)
            for word in words
        ))
        self.wait()


class ListOfFacts(InteractiveScene):
    def construct(self):
        # Title
        title = Text("Facts", font_size=120)
        title.to_corner(UL, buff=0.75)
        underline = Underline(title)
        underline.scale(1.3)
        underline.set_stroke(width=(0, 5, 5, 5, 0))
        title.add(underline)
        self.add(title)

        # List of facts
        n_facts = 15
        facts = VGroup(
            Text(line)
            for line in Path(DATA_DIR, "facts.txt").read_text().split("\\n")[:n_facts]
        )
        facts.set_color(GREY_A)
        facts.arrange(DOWN, buff=0.35, aligned_edge=LEFT)
        facts.set_height(5.5)
        facts.next_to(title, DOWN, buff=0.5).align_to(title[0], LEFT)
        self.add(facts)

        # Add line to LLM
        vline = Line(UP, DOWN).replace(facts, dim_to_match=1)
        vline.next_to(facts, RIGHT, buff=2.0)
        vline.scale(0.7)

        lines = VGroup(
            Line(
                fact.get_right(),
                vline.pfp(a),
                path_arc=interpolate(-20, 20, a) * DEGREES,
                color=random_bright_color(hue_range=(0.1, 0.2))
            ).insert_n_curves(20).set_stroke(width=(0, 5, 5, 0))
            for fact, a in zip(
                facts,
                np.linspace(0, 1, len(facts))
            )
        )
        self.add(lines)`,
    annotations: {
      9: "GPTInitials extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      10: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      126: "DifferentUsesOfModel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      127: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      220: "BigMatrixMultiplication extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      224: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      239: "LongListOFQuestions extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      240: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      306: "ChatBotIcon extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      307: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      330: "GamePlan extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      333: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      522: "Class SkipAhead inherits from TeacherStudentsScene.",
      523: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      535: "Class SeaOfNumbersUnderlay inherits from TeacherStudentsScene.",
      536: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      573: "Class Outdated inherits from TeacherStudentsScene.",
      574: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      604: "ConvolutionComment extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      605: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      624: "Class ConfusionAtScreen inherits from TeacherStudentsScene.",
      625: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      638: "Class HoldUpExample inherits from TeacherStudentsScene.",
      639: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      649: "ReactToWordVectors extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      650: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      686: "DimensionComparrison extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      687: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      712: "Class AtLeastKindOf inherits from TeacherStudentsScene.",
      713: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      730: "NetworkEndAnnotation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      733: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      765: "LowTempHighTempContrast extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      766: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      787: "Class Intuitions inherits from TeacherStudentsScene.",
      788: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      822: "Class PiGesturingAtEarlyView inherits from PiCreatureScene.",
      823: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      834: "Class EndScreen inherits from PatreonEndScreen.",
      841: "Class HighlightAttentionTitle inherits from TeacherStudentsScene.",
      842: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      929: "Class ThinkOfMoreExamples inherits from TeacherStudentsScene.",
      930: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      965: "Class SimplerExample inherits from TeacherStudentsScene.",
      966: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      978: "NotQuiteTrue extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      979: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      991: "Class ThisIsMadeUp inherits from TeacherStudentsScene.",
      992: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1003: "Class AskAboutOtherEmbeddings inherits from TeacherStudentsScene.",
      1004: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1024: "Class ShoutSoftmax inherits from TeacherStudentsScene.",
      1025: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1035: "LeftArcSmaller extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1036: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1043: "Class SetThemToZero inherits from TeacherStudentsScene.",
      1044: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1059: "Class CalledMasking inherits from TeacherStudentsScene.",
      1060: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1070: "ReferenceLargerContextTechnologies extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1071: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1096: "Class AskAboutCrossAttention inherits from TeacherStudentsScene.",
      1097: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1108: "SelfVsCrossFrames extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1109: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1133: "OngoingTranscription extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1134: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1143: "Class ReferenceStraightforwardValueMatrix inherits from TeacherStudentsScene.",
      1144: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1159: "Class SeekingMatchedParameters inherits from TeacherStudentsScene.",
      1160: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1191: "HeadName extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1192: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1207: "DInputAndOutputOfValue extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1208: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1229: "Class NowRepeatManyTimes inherits from TeacherStudentsScene.",
      1230: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1243: "Class ALotToHoldInYouHead inherits from TeacherStudentsScene.",
      1244: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1252: "Class ReactToMHSA inherits from TeacherStudentsScene.",
      1253: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1265: "Class AskAboutOutput inherits from TeacherStudentsScene.",
      1267: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1282: "OneThirdOfWhatYouNeed extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1283: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1317: "MoreResourcesBelow extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1318: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1338: "Class PatreonEndScreen inherits from EndScreen.",
      1345: "Class HowAndWhere inherits from TeacherStudentsScene.",
      1346: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1360: "Class IntroducingMLPs inherits from TeacherStudentsScene.",
      1361: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1401: "Class ReferenceFactStorage inherits from TeacherStudentsScene.",
      1402: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1434: "Class LookingAtPreview inherits from TeacherStudentsScene.",
      1435: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1452: "Class RefreshersNeverHurt inherits from TeacherStudentsScene.",
      1453: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1466: "EmbeddingLabel extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1467: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1505: "ThatWhichDoesntKillHeader extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1506: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1521: "QuickAttentionDescription extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1522: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1533: "QuickMLPDescription extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1534: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1544: "ContrastBetweenSimpleComputationDifficultInterpretation extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1545: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1574: "AmbientChangingDots extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1575: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1593: "Class MakeSomeAssumptions inherits from TeacherStudentsScene.",
      1594: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1604: "MovingToSecondToken extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1605: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1660: "Class WhatAboutBiggerThanOne inherits from TeacherStudentsScene.",
      1661: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1681: "HighlightRect extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1682: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1690: "Class AskWhy inherits from TeacherStudentsScene.",
      1691: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1708: "GPT3Logo extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1709: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1726: "AndGate extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1727: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1741: "MJFactsAsVectorSum extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1742: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1773: "Class AskAboutBias inherits from TeacherStudentsScene.",
      1774: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1792: "Class ThatsIt inherits from TeacherStudentsScene.",
      1793: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1809: "AddTwoMatrixSizes extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1810: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1825: "Class ReflectOnTwoThings inherits from TeacherStudentsScene.",
      1826: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1905: "Class RotatingVectors inherits from ReflectOnTwoThings.",
      1906: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1917: "Class AskIfThisIsReal inherits from TeacherStudentsScene.",
      1918: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      1941: "SingleNeuronVsMultiple extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      1942: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2023: "WriteSuperposition extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2024: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2044: "JohnsonLindenstraussName extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2045: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2059: "ContrastGPTDimensionSizes extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2060: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2095: "Class ReferenceSAP inherits from TeacherStudentsScene.",
      2096: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2124: "DetailsNotDiscussed extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2125: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2165: "TriPanelWithPi extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2166: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2193: "WriteRLHF extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2194: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
      2212: "ListOfFacts extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      2213: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();