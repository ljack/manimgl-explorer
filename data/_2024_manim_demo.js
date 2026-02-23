(function() {
  const files = window.MANIM_DATA.files;

  files["_2024/manim_demo/lorenz.py"] = {
    description: "A ManimGL demo scene that visualizes the Lorenz strange attractor in 3D. Computes trajectories from nearby initial conditions to show sensitive dependence on initial conditions (chaos), then animates glowing dots tracing those paths with colored tails.",
    code: `from manim_imports_ext import *
from scipy.integrate import solve_ivp


def lorenz_system(t, state, sigma=10, rho=28, beta=8 / 3):
    x, y, z = state
    dxdt = sigma * (y - x)
    dydt = x * (rho - z) - y
    dzdt = x * y - beta * z
    return [dxdt, dydt, dzdt]


def ode_solution_points(function, state0, time, dt=0.01):
    solution = solve_ivp(
        function,
        t_span=(0, time),
        y0=state0,
        t_eval=np.arange(0, time, dt)
    )
    return solution.y.T


def for_later():
    tail = VGroup(
        TracingTail(dot, time_traced=3).match_color(dot)
        for dot in dots
    )


class LorenzAttractor(InteractiveScene):
    def construct(self):
        # Set up axes
        axes = ThreeDAxes(
            x_range=(-50, 50, 5),
            y_range=(-50, 50, 5),
            z_range=(-0, 50, 5),
            width=16,
            height=16,
            depth=8,
        )
        axes.set_width(FRAME_WIDTH)
        axes.center()

        self.frame.reorient(43, 76, 1, IN, 10)
        self.frame.add_updater(lambda m, dt: m.increment_theta(dt * 3 * DEGREES))
        self.add(axes)

        # Add the equations
        equations = Tex(
            R"""
            \\begin{aligned}
            \\frac{\\mathrm{d} x}{\\mathrm{~d} t} & =\\sigma(y-x) \\\\
            \\frac{\\mathrm{d} y}{\\mathrm{~d} t} & =x(\\rho-z)-y \\\\
            \\frac{\\mathrm{d} z}{\\mathrm{~d} t} & =x y-\\beta z
            \\end{aligned}
            """,
            t2c={
                "x": RED,
                "y": GREEN,
                "z": BLUE,
            },
            font_size=30
        )
        equations.fix_in_frame()
        equations.to_corner(UL)
        equations.set_backstroke()
        self.play(Write(equations))

        # Compute a set of solutions
        epsilon = 1e-5
        evolution_time = 30
        n_points = 10
        states = [
            [10, 10, 10 + n * epsilon]
            for n in range(n_points)
        ]
        colors = color_gradient([BLUE_E, BLUE_A], len(states))

        curves = VGroup()
        for state, color in zip(states, colors):
            points = ode_solution_points(lorenz_system, state, evolution_time)
            curve = VMobject().set_points_smoothly(axes.c2p(*points.T))
            curve.set_stroke(color, 1, opacity=0.25)
            curves.add(curve)

        curves.set_stroke(width=2, opacity=1)

        # Display dots moving along those trajectories
        dots = Group(GlowDot(color=color, radius=0.25) for color in colors)

        def update_dots(dots, curves=curves):
            for dot, curve in zip(dots, curves):
                dot.move_to(curve.get_end())

        dots.add_updater(update_dots)

        tail = VGroup(
            TracingTail(dot, time_traced=3).match_color(dot)
            for dot in dots
        )

        self.add(dots)
        self.add(tail)
        curves.set_opacity(0)
        self.play(
            *(
                ShowCreation(curve, rate_func=linear)
                for curve in curves
            ),
            run_time=evolution_time,
        )


class EndScreen(PatreonEndScreen):
    pass`,
    annotations: {
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
    }
  };

})();