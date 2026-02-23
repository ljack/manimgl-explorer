#!/usr/bin/env python3
"""
Generate ManimGL Explorer data files.

Reads all Python source files from 3b1b/videos (2024-2026),
generates annotated chunk JS files and a data.js manifest.
"""

import os
import re
import json
from pathlib import Path
from collections import defaultdict

VIDEOS = Path("/tmp/3b1b-videos")
OUTPUT = Path("/Users/jarkko/_dev/manimgl-explorer")
DATA_DIR = OUTPUT / "data"

# ─── Project metadata ─────────────────────────────────────────────
TOPICS = {
    "_2024/antp": ("ANTP", "Ask a new teacher personal — animations for an advice column-style video."),
    "_2024/holograms": ("Holograms", "Holographic optics, diffraction patterns, and wave interference physics for the 'But what is a hologram?' video."),
    "_2024/inscribed_rect": ("Inscribed Rectangle", "The inscribed rectangle problem: proving every closed curve has an inscribed rectangle using topology (Möbius strips and Klein bottles)."),
    "_2024/linalg": ("Linear Algebra", "Eigenvector/eigenvalue lecture — extending the Essence of Linear Algebra series."),
    "_2024/manim_demo": ("ManimGL Demo", "Demonstration scenes showcasing ManimGL capabilities, including the Lorenz attractor."),
    "_2024/puzzles": ("Puzzles", "Mathematical puzzles including higher-dimensional geometry and randomness."),
    "_2024/transformers": ("Transformers", "Deep dive into transformer neural network architecture: attention, embeddings, MLPs, and autoregression for the GPT explainer series."),
    "_2025/colliding_blocks_v2": ("Colliding Blocks v2", "Elastic block collisions that compute digits of pi, with connections to Grover's quantum search algorithm."),
    "_2025/cosmic_distance": ("Cosmic Distance Ladder", "Astronomical distance measurement: parallax, standard candles, and the cosmic distance ladder."),
    "_2025/grover": ("Grover's Algorithm", "Grover's quantum search algorithm: qubits, superposition, state vectors, and quantum speedup."),
    "_2025/guest_videos": ("Guest Videos", "Animations for guest collaborations: Burnside's lemma, Euclid's elements, and miscellaneous."),
    "_2025/laplace": ("Laplace Transforms", "Comprehensive Laplace transform series: differential equations, exponentials, spring-mass systems, and the s-domain."),
    "_2025/spheres": ("Sphere Geometry", "Surface area, volumes, and random geometric puzzles on spheres."),
    "_2025/zeta": ("Riemann Zeta", "Visualizations of the Riemann zeta function and analytic continuation."),
    "_2026/hairy_ball": ("Hairy Ball Theorem", "Vector fields on spheres, stereographic projection, and the hairy ball theorem proof."),
    "_2026/monthly_mindbenders": ("Monthly Mindbenders", "Monthly mathematical puzzle series — probability, combinatorics, and geometry puzzles."),
}

# ─── Existing hand-written annotations to preserve ────────────────
# These override auto-generated annotations for files that already had them
EXISTING_ANNOTATIONS = {
    "_2024/manim_demo/lorenz.py": {
        1: "Imports all of ManimGL plus common math/utility extensions. This single import gives access to Scene, Mobject, Tex, animation classes, numpy (as np), and constants like PI, TAU, RIGHT, UP, etc.",
        2: "SciPy's solve_ivp is a numerical ODE integrator. It supports multiple methods (RK45, RK23, etc.) for solving initial value problems of the form dy/dt = f(t, y).",
        5: "The Lorenz system: a set of three coupled nonlinear ODEs discovered by Edward Lorenz in 1963. The default parameters (sigma=10, rho=28, beta=8/3) produce chaotic behavior — the famous 'butterfly' attractor.",
        6: "Unpacks the 3D state vector into x, y, z components.",
        7: "dx/dt = sigma*(y - x): The sigma parameter controls the rate of convective overturning. This term couples x to y.",
        8: "dy/dt = x*(rho - z) - y: The rho parameter relates to the temperature difference driving convection. When rho > 1, the origin becomes unstable.",
        9: "dz/dt = x*y - beta*z: The beta parameter is a geometric factor of the convection cell. This nonlinear x*y term is what makes the system chaotic.",
        13: "Helper function that wraps solve_ivp to return an array of solution points sampled at regular intervals dt. Returns shape (N, 3) array of [x, y, z] states.",
        14: "solve_ivp solves the ODE numerically using adaptive Runge-Kutta by default. t_span gives the integration interval, y0 is the initial state, t_eval specifies output times.",
        20: "solution.y has shape (3, N) — transpose to get (N, 3) so each row is a [x, y, z] point.",
        23: "A placeholder function showing TracingTail usage. VGroup(...) with a generator expression is a ManimGL pattern for creating groups of objects from iterables.",
        25: "TracingTail creates a fading trail behind a moving Mobject. time_traced=3 means the tail shows the last 3 seconds of movement. match_color copies the dot's color.",
        30: "InteractiveScene is ManimGL's primary scene class for 3D. It gives access to self.frame (the camera frame), mouse interaction, and the full 3D rendering pipeline.",
        31: "construct() is the main entry point for any Scene. ManimGL calls this method when the scene is run. All animation logic goes here.",
        33: "ThreeDAxes creates a 3D coordinate system with x, y, z axes. Each range tuple is (min, max, step). The width/height/depth control the visual size in scene units.",
        41: "FRAME_WIDTH is a ManimGL constant (default ~14.2 units). set_width scales the axes to fill the frame horizontally.",
        44: "self.frame is the camera's Frame mobject in InteractiveScene. reorient(theta, phi, gamma, center, height) sets the camera angle: theta=43deg rotation, phi=76deg elevation, centered at IN (into screen), zoomed to height 10.",
        45: "add_updater attaches a function called every frame. Here it slowly rotates the camera by 3 degrees per second, giving a gentle orbiting effect. dt is the time delta between frames.",
        49: "Tex renders LaTeX equations. The R prefix makes it a raw string so backslashes don't need double-escaping in Python. t2c (text-to-color) maps variable names to colors for visual clarity.",
        64: "fix_in_frame() pins a Mobject to the camera frame so it stays in place as the 3D camera moves. Essential for HUD/overlay elements in 3D scenes.",
        65: "to_corner(UL) positions the mobject in the upper-left corner with default padding. UL = UP + LEFT.",
        66: "set_backstroke() adds a dark outline behind the text so it remains readable against the 3D background.",
        67: "Write is an animation that draws text/LaTeX stroke by stroke. self.play() runs one or more animations and blocks until they complete.",
        70: "epsilon = 1e-5: A tiny perturbation. Starting 10 trajectories with z differing by only 0.00001 demonstrates sensitive dependence on initial conditions — the hallmark of chaos.",
        73: "Creates 10 initial states: all start at (10, 10, z) where z ranges from 10 to 10+9*epsilon. These nearly identical starting points will diverge dramatically.",
        77: "color_gradient creates a smooth interpolation between two colors across N steps. BLUE_E (dark blue) to BLUE_A (light blue) gives a subtle depth effect.",
        82: "axes.c2p(*points.T) converts coordinates to scene points. c2p = 'coords to point'. *points.T unpacks the transposed array so x, y, z arrays are passed as separate arguments.",
        83: "set_stroke(color, width, opacity) configures the curve's visual appearance. Low opacity (0.25) so the curves look ethereal.",
        89: "GlowDot is a radial gradient dot that creates a soft glowing effect. It's a Group (not VMobject) because it uses a special shader for the glow.",
        91: "The update_dots function runs every frame. It moves each dot to the current end of its curve. As ShowCreation extends the curves, the dots follow along the growing tip.",
        95: "add_updater(update_dots) registers the function to be called on every frame render. The curves=curves default argument captures the reference at definition time.",
        97: "TracingTail for each dot: creates colored trailing lines that fade over 3 seconds. This produces the beautiful 'light painting' effect as dots move through the attractor.",
        104: "Setting curves to 0 opacity hides them visually while ShowCreation still extends their points. The dots and tails provide the visual — the curves are invisible guides.",
        105: "Unpacking with * passes each ShowCreation as a separate argument to play(), running them all simultaneously. rate_func=linear ensures constant-speed drawing over 30 seconds.",
        114: "PatreonEndScreen is a pre-built end card template from 3Blue1Brown's codebase that displays supporter credits.",
    },
    "_2026/monthly_mindbenders/ladybug.py": {
        1: "Standard ManimGL import: provides InteractiveScene, all Mobject types, animation classes, mathematical constants, and numpy.",
        4: "InteractiveScene subclass: the main scene for the ladybug puzzle. InteractiveScene provides mouse/keyboard interaction and 3D camera (self.frame).",
        5: "random_seed = 0: ManimGL's Scene class uses this to seed Python's random module in setup(), ensuring the random walk is reproducible across runs.",
        9: "get_clock() is a helper method defined below that creates and returns a Clock mobject with number labels.",
        10: "clock.ticks is a VGroup of line segments at each hour position. get_start() gives the inner endpoint of each tick mark — used as positions for the ladybug to visit.",
        11: "cycle_animation wraps an animation to loop forever. ClockPassesTime animates the clock hands moving through 12*60 minutes over 12 seconds of real time, then repeats.",
        15: "SVGMobject loads and parses an SVG file into ManimGL vector paths. The 'ladybug' string references a file in the assets directory.",
        18: "set_shading(ambient, diffuse, specular) adds 3D-style lighting. (0.5, 0.5, 0) means moderate ambient and diffuse light, no specular highlights.",
        19: "Creates a red circle dot positioned at the bottom of the ladybug SVG to represent its body. Radius is proportional to the bug's height for consistent scaling.",
        21: "Group (not VGroup) holds heterogeneous mobjects: a Dot, a Point (invisible anchor), and the SVG. Group transforms all children together without requiring them to be VMobjects.",
        24: "Builds a random curved flight path for the ladybug's entrance. start_new_path begins a new bezier path at the origin.",
        26: "rotate_vector(RIGHT, PI*random.random()) creates a random unit vector in the upper half-plane. This produces a natural-looking zigzag flight path.",
        28: "make_smooth() converts the piecewise linear path into a smooth bezier curve by computing appropriate control points at each joint.",
        29: "put_start_and_end_on rescales and repositions the entire path so it starts at 7*LEFT (off screen) and ends at the 12 o'clock position.",
        31: "MoveAlongPath animates the bug following the curved path over 3 seconds. The bug's position is interpolated along the path's arc length.",
        32: "The animate builder: clock.numbers[0].animate.set_color(RED) creates an animation that transitions the '12' label to red. This is ManimGL's declarative animation syntax.",
        38: "A set tracks which clock numbers have been visited. The while loop runs until all 12 numbers are painted — this is the 'coupon collector' / random walk on a cycle.",
        40: "random.choice([+1, -1]) gives an unbiased coin flip: move clockwise or counterclockwise. This is a simple symmetric random walk on Z/12Z (integers mod 12).",
        42: "path_arc controls the curved path of the Arrow. Negative step * TAU/12 makes the arc follow the clock's circular geometry in the correct direction.",
        43: "Arrow creates a curved arrow between clock positions. The 1.2 multiplier places arrows slightly outside the clock face. path_arc bends the arrow along the clock's circle.",
        53: "When only one number remains unvisited and we're about to visit it, color it TEAL (success color) instead of RED. This highlights the 'last number painted' for the puzzle.",
        55: "Three simultaneous animations: (1) arrow fades in then out, (2) bug moves along an arc to the next number, (3) that number's color changes. self.play runs them all at once.",
        56: "The animate builder chain: bug.animate.move_to(...).set_anim_args(path_arc=..., time_span=...) creates a curved movement animation that completes in the first half second.",
        64: "get_clock builds a custom clock face. Clock() is a built-in ManimGL mobject with hour_hand, minute_hand, and ticks submobjects.",
        67: "Scales all clock hands and ticks to 75% of their default size, pivoting around each line's start point (the center of the clock). This creates a more compact face.",
        70: "VGroup(Integer(n) for n in [12, *range(1, 12)]): creates number labels 12, 1, 2, ..., 11. Integer renders each as a formatted number mobject.",
        71: "Places each number at 75% of the radius around the clock, using rotate_vector(UP, -theta) to go clockwise from 12 o'clock. The negative theta matches standard clock direction.",
        79: "A simple companion scene that displays the puzzle question as text. Text() renders with the default font; Write animates it appearing stroke by stroke.",
    },
}

# ─── File description templates ───────────────────────────────────
FILE_DESCRIPTIONS = {
    "_2024/manim_demo/lorenz.py": "A ManimGL demo scene that visualizes the Lorenz strange attractor in 3D. Computes trajectories from nearby initial conditions to show sensitive dependence on initial conditions (chaos), then animates glowing dots tracing those paths with colored tails.",
    "_2026/monthly_mindbenders/ladybug.py": "A random walk puzzle animation: a ladybug lands on a clock face and performs a random walk, stepping +1 or -1 each turn, painting each number as it visits. The puzzle asks: what is the probability that the last number painted is 6?",
    "_2024/transformers/attention.py": "Core attention mechanism scenes for the transformer explainer series. Demonstrates how attention patterns work: queries, keys, values, softmax scoring, and multi-head attention. Includes visual demonstrations of word embeddings interacting through attention weights.",
    "_2024/transformers/embedding.py": "Word and token embedding visualizations. Shows how text is broken into tokens, mapped to high-dimensional vectors, and how positional encoding works. Foundational to the transformer series.",
    "_2024/transformers/mlp.py": "Multi-layer perceptron (feed-forward network) scenes. Visualizes the MLP layers within a transformer block: linear transformations, ReLU/GELU activations, and how neurons learn features.",
    "_2024/transformers/ml_basics.py": "Machine learning fundamentals: introduces neural networks, gradient descent, loss functions, and backpropagation concepts that underpin the transformer architecture.",
    "_2024/transformers/auto_regression.py": "Autoregressive text generation: demonstrates how transformers generate text one token at a time, with each prediction conditioned on all previous tokens.",
    "_2024/transformers/generation.py": "Token generation and sampling scenes. Visualizes temperature, top-k, and nucleus sampling strategies for controlling transformer text output.",
    "_2024/transformers/helpers.py": "Shared helper functions and reusable components for the transformer video series. Includes network diagram builders, embedding visualizers, and common animation utilities.",
    "_2024/transformers/chm.py": "Chapter marker scenes and visual transitions for the transformer series. Contains title cards, section headers, and bridging animations between concepts.",
    "_2024/transformers/network_flow.py": "Data flow through the full transformer network. Visualizes how information passes through embedding, attention, MLP, and layer norm stages end-to-end.",
    "_2024/transformers/almost_orthogonal.py": "Explores the geometry of high-dimensional embeddings: why random vectors in high dimensions are nearly orthogonal, and implications for transformer representations.",
    "_2024/transformers/old_auto_regression.py": "Earlier version of the autoregression scenes, preserved for reference. Contains alternative visual approaches to explaining next-token prediction.",
    "_2024/transformers/supplements.py": "Supplementary scenes and corrections for the transformer series. Includes additional explanations, clarifications, and bonus visual demonstrations.",
    "_2024/holograms/diffraction.py": "Diffraction and interference pattern scenes for the hologram video. Simulates wave propagation, single/double slit experiments, and how interference creates holographic images.",
    "_2024/holograms/model.py": "Core hologram model: simulates the recording and playback of holograms by modeling reference beams, object waves, and their interference patterns on a photographic plate.",
    "_2024/holograms/supplements.py": "Supplementary scenes for the hologram video: additional demonstrations of wave optics, Fourier analysis of diffraction, and bonus visualizations.",
    "_2024/inscribed_rect/helpers.py": "Helper functions for the inscribed rectangle problem: parametric curve generators, Möbius strip constructions, and geometric utility functions.",
    "_2024/inscribed_rect/loops.py": "Main scenes for the inscribed rectangle theorem proof. Visualizes how pairs of points on a closed curve map to a Möbius strip, and why an inscribed rectangle must exist.",
    "_2024/inscribed_rect/supplements.py": "Supplementary scenes exploring edge cases, alternative proofs, and deeper topology of the inscribed rectangle problem.",
    "_2024/linalg/eigenlecture.py": "Eigenvector and eigenvalue lecture scenes for the linear algebra series. Visualizes how matrices transform space, what eigenvectors represent geometrically, and characteristic polynomial computation.",
    "_2024/antp/main.py": "Animation scenes for an 'ask a new teacher personal' advice-style video segment.",
    "_2024/puzzles/added_dimension.py": "Explores how adding dimensions reveals hidden structure in mathematical puzzles. Uses 3D visualizations to solve problems that seem impossible in 2D.",
    "_2024/puzzles/max_rand.py": "A puzzle about maximizing randomness: explores probability distributions, entropy, and information-theoretic concepts through visual demonstrations.",
    "_2024/puzzles/supplements.py": "Supplementary puzzle scenes with additional brain teasers and mathematical curiosities.",
    "_2025/laplace/shm.py": "Spring-mass system for the Laplace transform series. The SrpingMassSystem class (typo in original) combines visual springs, mass blocks, and real-time physics ODE integration via updaters. Supports spring constant k, damping mu, external forcing, and force/velocity visualization.",
    "_2025/laplace/derivatives.py": "Visualizes derivatives in the context of Laplace transforms. Shows how differentiation in the time domain corresponds to multiplication by s in the frequency domain.",
    "_2025/laplace/exponentials.py": "Exponential functions and their Laplace transforms. Demonstrates the fundamental relationship between e^(st) and the s-domain, building intuition for the transform.",
    "_2025/laplace/integration.py": "Integration and the Laplace transform: visualizes how integration in time maps to division by s, with applications to solving ODEs.",
    "_2025/laplace/main_equations.py": "Core equation scenes for the Laplace series. Presents the key Laplace transform formulas, inverse transforms, and their derivations with step-by-step animations.",
    "_2025/laplace/main_supplements.py": "Supplementary material for the main Laplace transform video, including alternative derivations and extended examples.",
    "_2025/laplace/prequel_equations.py": "Equation scenes for the Laplace transform prequel, introducing the motivation and historical context for Laplace transforms.",
    "_2025/laplace/derivative_supplements.py": "Additional derivative-related scenes for the Laplace series, covering higher-order derivatives and the general differentiation property.",
    "_2025/laplace/supplements.py": "General supplementary scenes for the Laplace transform series: extra examples, edge cases, and bonus demonstrations.",
    "_2025/cosmic_distance/paralax.py": "Parallax measurement scenes: demonstrates how astronomers use Earth's orbital motion to triangulate distances to nearby stars via apparent angular shift.",
    "_2025/cosmic_distance/part2.py": "Part 2 of the cosmic distance ladder: extending beyond parallax to standard candles, Cepheid variables, and Type Ia supernovae.",
    "_2025/cosmic_distance/planets.py": "Planetary system scenes: solar system scale models, orbital mechanics, and the astronomical unit as the first rung of the distance ladder.",
    "_2025/cosmic_distance/supplements.py": "Supplementary cosmic distance scenes with additional distance measurement techniques and astronomical visualizations.",
    "_2025/cosmic_distance/supplements2.py": "Extended supplementary material for the cosmic distance series, covering redshift, Hubble's law, and the expanding universe.",
    "_2025/grover/clarification.py": "Clarification scenes for the Grover's algorithm video: addressing common misconceptions about quantum speedup and measurement.",
    "_2025/grover/polarization.py": "Polarization analogy for quantum mechanics: uses light polarization to build intuition for qubit states and measurement before introducing Grover's algorithm.",
    "_2025/grover/qc_supplements.py": "Quantum computing supplementary scenes: additional qubit visualizations, gate operations, and quantum circuit diagrams.",
    "_2025/grover/runtime.py": "Runtime analysis of Grover's algorithm: visualizes the O(√N) speedup over classical search and why you can't do better.",
    "_2025/grover/state_vectors.py": "Quantum state vector visualizations for Grover's algorithm. Includes custom Ket notation, amplitude bar charts, and the geometric interpretation of Grover iterations as rotations.",
    "_2025/colliding_blocks_v2/blocks.py": "Main colliding blocks simulation: elastic collisions between blocks of mass ratio 100^n compute digits of pi. Includes physics engine, phase space visualization, and the connection to billiards in a circle.",
    "_2025/colliding_blocks_v2/grover.py": "Bridge scenes connecting colliding blocks to Grover's quantum search algorithm: both involve counting collisions/reflections that approximate pi.",
    "_2025/colliding_blocks_v2/supplements.py": "Supplementary scenes for the colliding blocks video: additional phase space diagrams, conservation of energy/momentum, and limiting behavior.",
    "_2025/guest_videos/burnside.py": "Burnside's lemma animation: counting distinct objects under group symmetry, with colorful examples of necklace counting and rotational equivalence.",
    "_2025/guest_videos/euclid.py": "Euclid's Elements animations: visualizing classical geometric constructions, axioms, and proofs from the foundational mathematics text.",
    "_2025/guest_videos/misc_animations.py": "Miscellaneous animations produced for various guest collaborations and one-off projects.",
    "_2025/spheres/random_puzzles.py": "Random geometric puzzles on spheres: probability of random points forming specific configurations, great circle distributions, and spherical geometry.",
    "_2025/spheres/supplements.py": "Supplementary sphere geometry scenes: additional surface area derivations, solid angle concepts, and spherical cap calculations.",
    "_2025/spheres/volumes.py": "Sphere volume derivation scenes: visualizing the integral approach to computing sphere volume using disk slicing and Cavalieri's principle.",
    "_2025/zeta/play.py": "Riemann zeta function playground: interactive visualizations of zeta(s) in the complex plane, analytic continuation, and the critical strip.",
    "_2026/hairy_ball/spheres.py": "Core scenes for the Hairy Ball Theorem video. Includes Fibonacci sphere point generation, stereographic projection (and inverse), vector field transformations via the Jacobian, tangent vector fields on spheres, and animated stream lines demonstrating why every continuous tangent field on S² must have a zero.",
    "_2026/hairy_ball/model3d.py": "3D model scenes for the hairy ball video: sphere meshes, surface rendering, and tangent plane visualizations.",
    "_2026/hairy_ball/old_functions.py": "Earlier versions of hairy ball functions, preserved for reference. Contains alternative approaches to vector field visualization.",
    "_2026/hairy_ball/supplements.py": "Supplementary scenes for the hairy ball theorem: additional examples of vector fields with zeros, counterexamples on tori, and topological context.",
    "_2026/hairy_ball/talent.py": "Talent/presentation scenes for the hairy ball video: intro sequences, visual hooks, and narrative transitions.",
}


# ─── Annotation Engine ────────────────────────────────────────────

IMPORT_ANNOTATIONS = {
    "from manim_imports_ext import *": "Standard ManimGL wildcard import: provides Scene, Mobject hierarchy, animations, constants (PI, TAU, UP, RIGHT, etc.), numpy as np, and utility functions.",
    "from __future__ import annotations": "Enables PEP 604 union types (X | Y) and postponed evaluation of annotations for cleaner type hints.",
}

CLASS_BASES = {
    "InteractiveScene": "InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
    "Scene": "Base Scene class — construct() is called when the scene runs. Manages mobjects and animations.",
    "VGroup": "VGroup is a container for VMobjects that transforms, colors, and animates them together as a unit.",
    "Group": "Group holds heterogeneous mobjects (mix of VMobjects, Surfaces, Images) and transforms them together.",
    "VMobject": "VMobject (Vector Mobject) is defined by bezier curves. Supports stroke, fill, and smooth transformations.",
    "Animation": "Custom Animation subclass. Override interpolate(alpha) to define how the animation progresses from 0 to 1.",
    "Mobject": "Base Mobject class — everything visible on screen inherits from this. Provides position, color, and transformation methods.",
    "Surface": "3D parametric surface rendered as a triangle mesh via OpenGL.",
    "Tex": "Custom Tex subclass for specialized LaTeX rendering with additional formatting.",
    "StreamLines": "StreamLines traces curves that follow a vector field flow. Often wrapped in AnimatedStreamLines for dynamic visualization.",
    "ThreeDScene": "3D-optimized Scene with built-in camera controls and depth rendering.",
    "MovingCameraScene": "Scene with an animatable camera frame for panning, zooming, and tracking.",
}

MANIM_OBJECTS = {
    "ThreeDAxes": "ThreeDAxes creates a 3D coordinate system. Each range tuple is (min, max, step). width/height/depth set visual size.",
    "NumberPlane": "NumberPlane creates an infinite-looking 2D coordinate grid with major and minor gridlines.",
    "Sphere": "Sphere creates a parametric surface mesh. Can be textured, made transparent, and depth-tested for 3D rendering.",
    "Surface": "Parametric Surface rendered as a triangle mesh. Define via a uv-function mapping (u,v) → (x,y,z).",
    "SurfaceMesh": "SurfaceMesh draws wireframe grid lines on a Surface for spatial reference.",
    "VectorField": "VectorField samples a vector function at grid points and renders arrows. Useful for visualizing flows and forces.",
    "StreamLines": "StreamLines traces curves following a vector field. Combined with AnimatedStreamLines for flowing animation.",
    "AnimatedStreamLines": "Wraps StreamLines in a continuous animation that spawns, flows, and fades line segments.",
    "ParametricCurve": "ParametricCurve traces a function f(t) → (x,y,z) over a parameter range, producing a smooth 3D curve.",
    "ValueTracker": "ValueTracker holds a numeric value that can be animated. Other mobjects read it via get_value() in updaters.",
    "ComplexPlane": "ComplexPlane extends NumberPlane for complex number visualization. Points map to complex numbers directly.",
    "DotCloud": "DotCloud efficiently renders many small dots using a point-based shader — much faster than individual Dot mobjects.",
    "GlowDot": "GlowDot is a radial gradient dot with a soft glow effect, rendered via a special shader.",
    "TracingTail": "TracingTail creates a fading trail behind a moving mobject, showing its recent trajectory.",
    "Arrow": "Arrow creates a line with an arrowhead. path_arc parameter curves the arrow along a circular arc.",
    "Axes": "2D Axes with configurable ranges, labels, and tick marks. Use c2p/p2c for coordinate conversion.",
    "Tex": "Tex renders LaTeX mathematical expressions. Use raw strings (r\"...\") to avoid backslash escaping issues.",
    "Text": "Text renders plain text with the default font. Supports substring indexing for partial styling.",
    "Integer": "Integer displays a formatted integer that can be animated with set_value() and CountInFrom.",
    "DecimalNumber": "DecimalNumber displays a formatted decimal that can be animated. Tracks a value and auto-updates display.",
}

API_PATTERNS = [
    (r'^\s*self\.play\(\s*$', "self.play() executes one or more animations simultaneously and waits for them to complete."),
    (r'self\.wait\(\d', "self.wait(n) pauses the scene for n seconds, allowing updaters and animations to run."),
    (r'self\.wait\(\)', "self.wait() pauses for 1 second (default), allowing updaters and ambient animations to run."),
    (r'frame\.reorient\(', "Reorients the 3D camera: (theta, phi, gamma, center, height) sets horizontal rotation, elevation, roll, look-at point, and zoom level."),
    (r'frame\.animate\.reorient', "Smoothly animates the camera to a new orientation over the animation duration."),
    (r'\.add_updater\(lambda\s+\w+,\s*dt', "Time-based updater: called every frame with the mobject and time delta (dt). Used for physics simulations and continuous motion."),
    (r'\.add_updater\(lambda\s+\w+:', "State-based updater: called every frame with the mobject. Used for reactive positioning and following other objects."),
    (r'\.animate\.', "The animate builder pattern: creates an animation interpolating from the current state to the state after the chained method call."),
    (r'fix_in_frame\(\)', "Pins this mobject to the camera frame (HUD-style) so it stays fixed when the 3D camera moves."),
    (r'apply_depth_test\(\)', "Enables OpenGL depth testing so objects behind others are correctly occluded in 3D."),
    (r'always_sort_to_camera', "Reorders transparent faces each frame for correct alpha blending from the current camera angle."),
    (r'set_clip_plane\(', "Clips geometry to a half-space defined by a normal vector and offset. Used for cross-section reveals."),
    (r'set_backstroke\(\)', "Adds a dark outline behind text/LaTeX for readability over complex 3D backgrounds."),
    (r'\.c2p\(', "c2p (coords to point) converts mathematical coordinates to scene positions through the axes' transformation."),
    (r'\.p2c\(', "p2c (point to coords) converts scene positions back to mathematical coordinates."),
    (r'color_gradient\(', "Creates a list of colors smoothly interpolated between the given endpoints."),
    (r'cycle_animation\(', "Wraps an animation to loop indefinitely. The animation restarts seamlessly when it completes."),
    (r'save_state\(\)', "Saves the mobject's current state (position, color, etc.) so it can be restored later with Restore()."),
    (r'apply_points_function\(', "Applies a transformation function to all bezier control points, enabling nonlinear warping of shapes."),
    (r'insert_n_curves\(', "Subdivides existing bezier curves to increase point density. Needed before applying nonlinear transformations for smooth results."),
    (r'make_smooth\(\)', "Recalculates bezier control points for smooth curves. Call after nonlinear transformations to fix jagged artifacts."),
    (r'set_shading\(', "Sets Phong lighting parameters: (ambient, diffuse, specular). Gives 2D shapes a subtle 3D appearance."),
    (r'add_ambient_rotation\(', "Makes the camera slowly rotate at the given rate (in radians/second), providing 3D depth perception."),
    (r'set_scale_stroke_with_zoom\(', "When True, stroke width scales with camera zoom. When False, strokes maintain constant screen-space width."),
]

ANIMATION_PATTERNS = {
    "ShowCreation": "ShowCreation draws a VMobject's stroke progressively from start to end.",
    "Write": "Write animates text/LaTeX appearing stroke by stroke, simulating handwriting.",
    "FadeIn": "FadeIn transitions a mobject from transparent to opaque, optionally with a directional shift.",
    "FadeOut": "FadeOut transitions a mobject from opaque to transparent.",
    "Transform": "Transform smoothly morphs one mobject into another by interpolating their points.",
    "TransformFromCopy": "TransformFromCopy creates a copy of the source, then transforms it into the target. Leaves the original unchanged.",
    "ReplacementTransform": "ReplacementTransform morphs source into target AND replaces source in the scene with target.",
    "MoveAlongPath": "MoveAlongPath animates a mobject following a curve, interpolated by arc length.",
    "LaggedStartMap": "LaggedStartMap applies an animation to each element of a group with staggered start times.",
    "LaggedStart": "LaggedStart runs multiple animations with staggered starts, controlled by lag_ratio.",
    "DrawBorderThenFill": "DrawBorderThenFill first draws the stroke outline, then fills the interior.",
    "GrowArrow": "GrowArrow animates an arrow growing from its start point to full length.",
    "VFadeIn": "VFadeIn fades in a VMobject by animating stroke width and fill opacity.",
    "VFadeInThenOut": "VFadeInThenOut fades a mobject in then back out within a single animation.",
    "Indicate": "Indicate briefly scales up and highlights a mobject to draw attention to it.",
    "Flash": "Flash creates a burst of light/lines radiating outward from a point.",
    "Restore": "Restore animates a mobject back to a previously saved state (from save_state()).",
    "CountInFrom": "CountInFrom animates a number counting up from a starting value.",
    "Succession": "Succession plays animations one after another in sequence.",
    "AnimationGroup": "AnimationGroup plays multiple animations together with individual timing control.",
    "UpdateFromFunc": "UpdateFromFunc calls a function on each frame to update a mobject's state.",
    "UpdateFromAlphaFunc": "UpdateFromAlphaFunc calls a function with the interpolation alpha (0→1) each frame.",
}

MATH_PATTERNS = [
    (r'np\.cross\(', "Cross product: produces a vector perpendicular to both inputs. For points on a sphere, cross(p, axis) gives a tangent vector."),
    (r'np\.dot\(', "Dot product: measures alignment between two vectors. Zero means perpendicular."),
    (r'np\.linalg\.norm\(', "Vector norm (magnitude/length). Used for normalization and distance calculations."),
    (r'np\.linspace\(', "np.linspace creates evenly spaced values over an interval — essential for parametric sampling."),
    (r'np\.exp\(', "Exponential function: fundamental to Laplace transforms, signal processing, and growth/decay models."),
    (r'np\.sin\(|np\.cos\(', "Trigonometric functions: used for circular/spherical geometry, wave physics, and periodic motion."),
    (r'np\.arctan2\(', "Two-argument arctangent: returns the angle in the correct quadrant, unlike np.arctan."),
    (r'solve_ivp\(', "SciPy's numerical ODE integrator: solves dy/dt = f(t,y) using adaptive Runge-Kutta methods."),
    (r'np\.outer\(', "Outer product: creates a matrix from two vectors. Fundamental in quantum mechanics and tensor operations."),
    (r'np\.einsum\(', "Einstein summation: flexible and efficient tensor contraction notation for multi-dimensional array operations."),
    (r'normalize_along_axis\(', "Normalizes each vector (row or column) to unit length. Projects points back onto the unit sphere."),
    (r'interpolate\(', "Linearly interpolates between two values: result = a + alpha*(b-a), where alpha ranges 0 to 1."),
    (r'interpolate_color', "Interpolates between colors in HSL space for perceptually uniform gradients."),
    (r'rotate_vector\(', "Rotates a 2D vector by the given angle. Equivalent to multiplying by a rotation matrix."),
    (r'get_norm\(', "Returns the Euclidean length of a vector. ManimGL utility wrapping np.linalg.norm."),
    (r'angle_between_vectors\(', "Computes the angle between two vectors using the dot product formula: cos(θ) = (a·b)/(|a||b|)."),
    (r'stereographic_proj\(', "Stereographic projection maps sphere points to the plane: (x,y,z) → (x/(1-z), y/(1-z)). Preserves angles (conformal)."),
    (r'fibonacci_sphere\(', "Fibonacci sphere: generates approximately uniformly distributed points on S² using the golden angle. Much better than latitude/longitude grids."),
]


def escape_template_literal(code):
    """Escape a string for embedding in a JavaScript template literal."""
    code = code.replace('\\', '\\\\')
    code = code.replace('`', '\\`')
    code = code.replace('${', '\\${')
    return code


def escape_js_string(s):
    """Escape a string for embedding in a JavaScript string literal."""
    s = s.replace('\\', '\\\\')
    s = s.replace('"', '\\"')
    s = s.replace('\n', '\\n')
    return s


def discover_projects():
    """Discover all Python files organized by project (2024-2026 only)."""
    projects = defaultdict(list)
    YEAR_DIRS = ["_2024", "_2025", "_2026"]
    # _2025/hairy_ball is a symlink to _2026/hairy_ball — skip it to avoid duplicates
    SKIP_PROJECTS = {"_2025/hairy_ball"}

    for year_name in YEAR_DIRS:
        year_dir = VIDEOS / year_name
        if not year_dir.is_dir():
            continue
        for project_dir in sorted(year_dir.iterdir()):
            if not project_dir.is_dir():
                continue
            project_key = f"{year_dir.name}/{project_dir.name}"
            if project_key in SKIP_PROJECTS:
                continue
            py_files = sorted(project_dir.glob("*.py"))
            for f in py_files:
                if f.name == "__init__.py":
                    continue
                rel_path = f"{year_dir.name}/{project_dir.name}/{f.name}"
                code = f.read_text(encoding="utf-8", errors="replace")
                projects[project_key].append({
                    "path": rel_path,
                    "name": f.name,
                    "code": code,
                    "abs_path": str(f),
                })

    return dict(projects)


def extract_classes_and_functions(code):
    """Extract class names, their base classes, and top-level function names."""
    classes = []
    functions = []

    for match in re.finditer(r'^class\s+(\w+)(?:\(([^)]*)\))?:', code, re.MULTILINE):
        name = match.group(1)
        bases = match.group(2) or ""
        classes.append({"name": name, "bases": bases, "line": code[:match.start()].count('\n') + 1})

    for match in re.finditer(r'^(\s*)def\s+(\w+)\s*\(', code, re.MULTILINE):
        indent = len(match.group(1))
        name = match.group(2)
        line = code[:match.start()].count('\n') + 1
        functions.append({"name": name, "line": line, "indent": indent})

    return classes, functions


def generate_description(file_info, project_key):
    """Generate a description for a file."""
    path = file_info["path"]

    # Check for hand-written description
    if path in FILE_DESCRIPTIONS:
        return FILE_DESCRIPTIONS[path]

    # Auto-generate from code analysis
    code = file_info["code"]
    classes, functions = extract_classes_and_functions(code)
    topic_name, topic_desc = TOPICS.get(project_key, ("Unknown", ""))

    scene_classes = [c for c in classes if any(b in c["bases"] for b in ["Scene", "InteractiveScene", "ThreeDScene", "MovingCameraScene"])]
    other_classes = [c for c in classes if c not in scene_classes]

    parts = []

    if scene_classes:
        names = ", ".join(c["name"] for c in scene_classes[:4])
        suffix = f" and {len(scene_classes) - 4} more" if len(scene_classes) > 4 else ""
        parts.append(f"Contains scene classes: {names}{suffix}.")

    if other_classes:
        names = ", ".join(c["name"] for c in other_classes[:3])
        suffix = f" and {len(other_classes) - 3} more" if len(other_classes) > 3 else ""
        parts.append(f"Defines helper classes: {names}{suffix}.")

    top_funcs = [f for f in functions if f["indent"] == 0 and f["name"] != "__init__"]
    if top_funcs and not scene_classes:
        names = ", ".join(f["name"] for f in top_funcs[:4])
        parts.append(f"Utility functions: {names}.")

    if not parts:
        parts.append(f"Source file for the {topic_name} project.")

    # Add topic context
    parts.append(f"Part of the {topic_name} series — {topic_desc.rstrip('.')}" + "." if topic_desc else "")

    return " ".join(parts).strip()


def generate_annotations(file_info, project_key):
    """Generate annotations for a file."""
    path = file_info["path"]
    code = file_info["code"]
    lines = code.split('\n')
    total_lines = len(lines)

    # Start with existing hand-written annotations
    annotations = dict(EXISTING_ANNOTATIONS.get(path, {}))

    # Determine target annotation count based on file size
    if total_lines < 100:
        target = min(25, total_lines // 3)
    elif total_lines < 300:
        target = min(40, total_lines // 5)
    elif total_lines < 1000:
        target = min(60, total_lines // 10)
    else:
        target = min(80, total_lines // 20)

    auto_annotations = {}

    for i, line in enumerate(lines):
        line_num = i + 1
        stripped = line.strip()

        if not stripped or stripped.startswith('#'):
            # Annotate significant section comments
            if stripped.startswith('# ') and len(stripped) > 10 and line_num > 1:
                prev = lines[i-1].strip() if i > 0 else ""
                next_line = lines[i+1].strip() if i + 1 < total_lines else ""
                if prev == "" or prev.startswith("#"):
                    # This is a section header comment — don't annotate, but note it
                    pass
            continue

        # ── Import annotations ──
        for pattern, text in IMPORT_ANNOTATIONS.items():
            if stripped == pattern:
                auto_annotations[line_num] = text
                break

        # Cross-project imports
        if re.match(r'from _20\d\d\.', stripped):
            parts = stripped.split('import')
            if len(parts) == 2:
                module = parts[0].replace('from ', '').strip()
                imported = parts[1].strip()
                auto_annotations[line_num] = f"Imports {imported} from the {module} module within the 3b1b videos codebase."

        # ── Class definitions ──
        class_match = re.match(r'^class\s+(\w+)\(([^)]+)\):', stripped)
        if class_match:
            cls_name = class_match.group(1)
            bases_str = class_match.group(2)
            bases = [b.strip() for b in bases_str.split(',')]
            for base in bases:
                if base in CLASS_BASES:
                    auto_annotations[line_num] = f"{cls_name} extends {base}. {CLASS_BASES[base]}"
                    break
            else:
                auto_annotations[line_num] = f"Class {cls_name} inherits from {bases_str}."

        # Simple class (no bases or just object)
        simple_class = re.match(r'^class\s+(\w+)\s*:', stripped)
        if simple_class and not class_match:
            auto_annotations[line_num] = f"Class {simple_class.group(1)} — standalone class definition."

        # ── Method/function definitions ──
        def_match = re.match(r'^(\s*)def\s+(\w+)\s*\(([^)]*)\)', stripped if not stripped.startswith('def') else line)
        if not def_match:
            def_match = re.match(r'^(\s*)def\s+(\w+)\s*\(', line)
        if def_match:
            indent = len(line) - len(line.lstrip())
            func_name = def_match.group(2) if def_match.lastindex >= 2 else re.search(r'def\s+(\w+)', stripped).group(1)

            if func_name == "construct" and indent > 0:
                auto_annotations[line_num] = "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here."
            elif func_name == "setup" and indent > 0:
                auto_annotations[line_num] = "setup() runs before construct(). Used to initialize shared state and add persistent mobjects."
            elif func_name == "__init__":
                pass  # Don't annotate every __init__
            elif func_name == "interpolate" and indent > 0:
                auto_annotations[line_num] = "interpolate(alpha) is called each frame with alpha from 0→1. Defines the custom animation's behavior over its duration."

        # ── ManimGL object creation ──
        for obj_name, desc in MANIM_OBJECTS.items():
            # Match both `Foo()` and `Foo(args)`
            if re.search(rf'\b{obj_name}\s*\(', stripped) and obj_name not in stripped.split('#')[0].split('class '):
                if line_num not in auto_annotations:
                    auto_annotations[line_num] = desc
                    break

        # ── API call patterns ──
        for pattern, text in API_PATTERNS:
            if re.search(pattern, stripped):
                if line_num not in auto_annotations:
                    auto_annotations[line_num] = text
                    break

        # ── Animation patterns ──
        for anim_name, desc in ANIMATION_PATTERNS.items():
            if re.search(rf'\b{anim_name}\b', stripped) and line_num not in auto_annotations:
                # Only annotate if this is a significant use (not just a reference)
                if re.search(rf'{anim_name}\s*\(', stripped) or re.search(rf'self\.play\(.*{anim_name}', stripped):
                    auto_annotations[line_num] = desc
                    break

        # ── Math patterns ──
        for pattern, text in MATH_PATTERNS:
            if re.search(pattern, stripped) and line_num not in auto_annotations:
                auto_annotations[line_num] = text
                break

        # ── Special value annotations ──
        if re.match(r'^[A-Z_]+\s*=\s*', stripped) and line_num not in auto_annotations:
            # Module-level constants
            const_match = re.match(r'^([A-Z_]+)\s*=\s*(.+)', stripped)
            if const_match and i < 40:  # Only near top of file
                name = const_match.group(1)
                val = const_match.group(2)
                if any(kw in name for kw in ['RADIUS', 'ORBIT', 'PERIOD', 'SPEED', 'ANGLE', 'TILT']):
                    auto_annotations[line_num] = f"Physical constant: {name} = {val.rstrip(',')}."

    # Merge: hand-written annotations take priority
    for line_num, text in auto_annotations.items():
        if line_num not in annotations:
            annotations[line_num] = text

    # Trim to target count, keeping hand-written and most important ones
    if len(annotations) > target:
        hand_written_lines = set(EXISTING_ANNOTATIONS.get(path, {}).keys())
        # Prioritize: hand-written > class defs > construct > imports > everything else
        priority_lines = set()
        for line_num in annotations:
            line = lines[line_num - 1] if line_num <= total_lines else ""
            stripped = line.strip()
            if line_num in hand_written_lines:
                priority_lines.add(line_num)
            elif re.match(r'^class\s+', stripped):
                priority_lines.add(line_num)
            elif 'def construct' in stripped:
                priority_lines.add(line_num)

        # Keep priority lines + fill up to target with remaining
        remaining = sorted(set(annotations.keys()) - priority_lines)
        keep = priority_lines | set(remaining[:max(0, target - len(priority_lines))])
        annotations = {k: v for k, v in annotations.items() if k in keep}

    return annotations


def generate_chunk_js(project_key, files_data):
    """Generate the JavaScript content for a chunk file."""
    parts = []
    parts.append("(function() {")
    parts.append("  const files = window.MANIM_DATA.files;")
    parts.append("")

    for file_info in files_data:
        path = file_info["path"]
        desc = file_info["description"]
        code = file_info["code"]
        anns = file_info["annotations"]

        escaped_code = escape_template_literal(code.rstrip('\n'))
        escaped_desc = escape_js_string(desc)

        parts.append(f'  files["{path}"] = {{')
        parts.append(f'    description: "{escaped_desc}",')
        parts.append(f'    code: `{escaped_code}`,')

        # Annotations
        if anns:
            parts.append('    annotations: {')
            for line_num in sorted(anns.keys()):
                text = anns[line_num]
                escaped_text = escape_js_string(text)
                parts.append(f'      {line_num}: "{escaped_text}",')
            parts.append('    }')
        else:
            parts.append('    annotations: {}')

        parts.append('  };')
        parts.append('')

    parts.append("})();")
    return '\n'.join(parts)


def chunk_filename(project_key):
    """Convert project key to chunk filename: _2024/transformers → _2024_transformers.js"""
    return project_key.replace('/', '_') + '.js'


def build_tree(projects):
    """Build the file tree structure for data.js."""
    years = defaultdict(lambda: defaultdict(list))

    for project_key, files in projects.items():
        year_dir, project_name = project_key.split('/')
        year_label = year_dir.replace('_', '')
        for f in files:
            years[year_label][project_name].append({
                "type": "file",
                "label": f["name"],
                "path": f["path"],
                "hasCode": True,
            })

    tree = []
    for year_label in sorted(years.keys()):
        year_node = {
            "type": "year",
            "label": year_label,
            "children": []
        }
        for project_name in sorted(years[year_label].keys()):
            folder = {
                "type": "folder",
                "label": project_name,
                "children": sorted(years[year_label][project_name], key=lambda x: x["label"])
            }
            year_node["children"].append(folder)
        tree.append(year_node)

    return tree


def build_file_index(projects):
    """Build the fileIndex mapping file paths to chunk URLs."""
    index = {}
    for project_key, files in projects.items():
        chunk = "data/" + chunk_filename(project_key)
        for f in files:
            index[f["path"]] = chunk
    return index


def build_search_index(projects):
    """Build a search index of class/function names per file."""
    index = {}
    for project_key, files in projects.items():
        for f in files:
            classes, functions = extract_classes_and_functions(f["code"])
            index[f["path"]] = {
                "classes": [c["name"] for c in classes],
                "functions": [fn["name"] for fn in functions if fn["indent"] == 0],
                "methods": [fn["name"] for fn in functions if fn["indent"] > 0 and fn["name"] not in ("__init__", "setup", "construct")],
            }
    return index


def count_scenes(projects):
    """Count total scene classes across all files."""
    count = 0
    for project_key, files in projects.items():
        for f in files:
            classes, _ = extract_classes_and_functions(f["code"])
            for c in classes:
                if any(b in c["bases"] for b in ["Scene", "InteractiveScene", "ThreeDScene", "MovingCameraScene"]):
                    count += 1
    return count


# ─── Architecture Context ─────────────────────────────────────────
ARCHITECTURE_CONTEXT = r"""You are an expert on ManimGL (3Blue1Brown's fork of Manim) and the 3b1b/videos repository. You help users understand how mathematical animations are built using this framework. Here is a comprehensive overview of ManimGL's architecture:

## Scene Hierarchy

ManimGL scenes inherit from `Scene`, with `InteractiveScene` being the most commonly used subclass. InteractiveScene adds mouse/keyboard interaction and provides `self.frame` (a `Frame` mobject controlling the camera). The main entry point is the `construct()` method, which is called when a scene runs. Scenes manage a list of active mobjects and orchestrate animations via `self.play()`, `self.wait()`, `self.add()`, and `self.remove()`.

## Mobject System

Everything visible on screen is a Mobject (mathematical object). The hierarchy is:
- **Mobject**: Base class with position, color, and transformation methods
- **VMobject**: Vector mobject defined by bezier curves (lines, arcs, polygons, Tex, Text)
- **VGroup**: A VMobject that contains other VMobjects and transforms them together
- **Group**: Like VGroup but for heterogeneous mobjects (can mix VMobjects with ImageMobjects, etc.)
- **Surface / Sphere**: 3D parametric surfaces rendered as triangle meshes
- **Dot / TrueDot / GlowDot**: Point-like objects with different rendering strategies
- **VectorField**: A collection of Arrow mobjects sampled from a vector-valued function
- **StreamLines**: Curves that follow a vector field flow, often wrapped in AnimatedStreamLines

Key Mobject methods: `move_to()`, `shift()`, `scale()`, `rotate()`, `set_color()`, `set_stroke()`, `set_fill()`, `set_opacity()`, `next_to()`, `to_corner()`, `to_edge()`, `get_center()`, `get_width()`, `copy()`, `add()`.

## Animation Pipeline

Animations in ManimGL work through several mechanisms:

1. **self.play(*animations)**: Runs one or more Animation objects simultaneously. Common animations include `FadeIn`, `FadeOut`, `Write`, `ShowCreation`, `Transform`, `MoveAlongPath`, `TransformFromCopy`. Each accepts `run_time`, `rate_func`, and `time_span` parameters.

2. **The Animate Builder**: `mobject.animate.method(args)` creates an animation that interpolates from the current state to the state after calling `method`. You can chain calls: `mob.animate.shift(UP).set_color(RED)`. Use `.set_anim_args()` to control timing.

3. **Updaters**: Functions attached via `mobject.add_updater(func)` that run every frame. Two signatures: `lambda m: ...` (state-based) and `lambda m, dt: ...` (time-based). Updaters are the primary mechanism for continuous animation, physics simulation, and reactive positioning.

4. **cycle_animation()**: Wraps an animation to loop indefinitely, useful for background effects.

## Camera and Frame Control

In InteractiveScene, `self.frame` is a Frame mobject that controls the virtual camera. Key methods:
- `frame.reorient(theta, phi, gamma, center, height)`: Set camera angles (theta = horizontal rotation, phi = elevation)
- `frame.animate.reorient(...)`: Smoothly animate camera movement
- `frame.add_updater()`: Attach continuous camera motion (e.g., slow rotation)
- `frame.add_ambient_rotation(rate)`: Shorthand for constant rotation
- `mobject.fix_in_frame()`: Pin a mobject to the screen (HUD-style) so it doesn't move with the 3D camera

## Coordinate Systems

- `ThreeDAxes`: 3D coordinate axes with configurable ranges and visual dimensions
- `NumberPlane`: 2D grid with background_lines and faded_lines
- `axes.c2p(x, y, z)` (coords to point): Converts mathematical coordinates to scene points
- `axes.p2c(point)` (point to coords): The inverse transformation

## Rendering Pipeline

ManimGL uses OpenGL for rendering (unlike the original Manim which renders to Cairo/FFmpeg). Key aspects:
- All geometry is rendered as OpenGL primitives (triangles, lines, points)
- `apply_depth_test()` enables z-buffer testing for correct 3D occlusion
- `always_sort_to_camera(camera)` reorders transparent faces each frame
- `set_clip_plane(normal, offset)` clips geometry to a half-space
- `set_shading(ambient, diffuse, specular)` enables Phong-style lighting
- Textures and images are rendered as textured quads

## Common Patterns in 3b1b Videos

1. **Physics simulations via updaters**: Attach an updater that does Euler integration each frame (see SpringMassSystem). State is stored as attributes, visuals are updated in sync.

2. **Parametric curves for visual elements**: `ParametricCurve` traces a function over a parameter range. Used for springs, spirals, and mathematical curves.

3. **TracingTail**: Creates a fading trail behind a moving mobject, commonly used to visualize trajectories.

4. **VGroup with generator expressions**: `VGroup(Mobject() for x in items)` is a concise pattern for creating collections.

5. **save_state() / Restore()**: Snapshot a mobject's state and animate back to it later.

6. **apply_points_function()**: Apply an arbitrary transformation to all bezier control points, enabling nonlinear transformations (like stereographic projection of an entire plane).

7. **LaTeX rendering**: `Tex(r"...")` renders LaTeX, `t2c` maps substrings to colors. `set_backstroke()` adds outlines for readability over 3D scenes.

When answering questions about this code, explain both the ManimGL API patterns and the underlying mathematics. Reference specific line numbers and methods when relevant. Be precise about the distinction between VMobject (bezier-based) and Surface (mesh-based) rendering paths."""


def write_data_js(tree, file_index, search_index, scene_count):
    """Write the data.js manifest file."""
    lines = []
    lines.append("// ManimGL Explorer - Data Layer (Manifest)")
    lines.append("// File tree, search index, and architecture context.")
    lines.append("// Source code is lazy-loaded from data/ chunk files.")
    lines.append("")
    lines.append("window.MANIM_DATA = {")
    lines.append("")
    lines.append("  // File storage — populated by lazy-loaded chunks")
    lines.append("  files: {},")
    lines.append("")

    # Tree
    lines.append("  // File tree structure")
    tree_json = json.dumps(tree, indent=4)
    # Indent tree JSON to match nesting
    tree_lines = tree_json.split('\n')
    lines.append("  tree: " + tree_lines[0])
    for tl in tree_lines[1:]:
        lines.append("  " + tl)
    lines.append(",")
    lines.append("")

    # File index
    lines.append("  // Maps file path → chunk URL for lazy loading")
    fi_json = json.dumps(file_index, indent=4, sort_keys=True)
    fi_lines = fi_json.split('\n')
    lines.append("  fileIndex: " + fi_lines[0])
    for fl in fi_lines[1:]:
        lines.append("  " + fl)
    lines.append(",")
    lines.append("")

    # Search index
    lines.append("  // Pre-extracted class/function names for search across unloaded files")
    si_json = json.dumps(search_index, indent=4, sort_keys=True)
    si_lines = si_json.split('\n')
    lines.append("  searchIndex: " + si_lines[0])
    for sl in si_lines[1:]:
        lines.append("  " + sl)
    lines.append(",")
    lines.append("")

    # Architecture context
    lines.append("  // Architecture context for AI chat")
    escaped_arch = ARCHITECTURE_CONTEXT.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    lines.append(f"  architectureContext: `{escaped_arch}`")
    lines.append("")
    lines.append("};")

    return '\n'.join(lines)


def main():
    print("Discovering projects...")
    projects = discover_projects()

    total_files = sum(len(files) for files in projects.values())
    total_lines = sum(len(f["code"].split('\n')) for files in projects.values() for f in files)
    print(f"Found {len(projects)} projects, {total_files} files, {total_lines} total lines")

    # Generate descriptions and annotations for each file
    print("Generating annotations...")
    for project_key, files in projects.items():
        for f in files:
            f["description"] = generate_description(f, project_key)
            f["annotations"] = generate_annotations(f, project_key)
            ann_count = len(f["annotations"])
            line_count = len(f["code"].split('\n'))
            density = ann_count / max(line_count, 1) * 100
            print(f"  {f['path']}: {line_count} lines, {ann_count} annotations ({density:.1f}%)")

    # Write chunk files
    print("\nWriting chunk files...")
    DATA_DIR.mkdir(exist_ok=True)
    for project_key, files in projects.items():
        chunk_name = chunk_filename(project_key)
        chunk_content = generate_chunk_js(project_key, files)
        chunk_path = DATA_DIR / chunk_name
        chunk_path.write_text(chunk_content, encoding="utf-8")
        size_kb = len(chunk_content) / 1024
        print(f"  {chunk_name}: {size_kb:.1f} KB")

    # Build manifest data
    print("\nBuilding manifest...")
    tree = build_tree(projects)
    file_index = build_file_index(projects)
    search_index = build_search_index(projects)
    scene_count = count_scenes(projects)

    # Write data.js
    manifest = write_data_js(tree, file_index, search_index, scene_count)
    (OUTPUT / "data.js").write_text(manifest, encoding="utf-8")
    print(f"  data.js: {len(manifest) / 1024:.1f} KB")

    print(f"\nDone! {total_files} files across {len(projects)} projects.")
    print(f"Scene classes found: {scene_count}")
    print(f"Total annotations: {sum(len(f['annotations']) for files in projects.values() for f in files)}")


if __name__ == "__main__":
    main()
