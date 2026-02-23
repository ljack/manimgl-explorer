(function() {
  const files = window.MANIM_DATA.files;

  files["_2026/monthly_mindbenders/ladybug.py"] = {
    description: "A random walk puzzle animation: a ladybug lands on a clock face and performs a random walk, stepping +1 or -1 each turn, painting each number as it visits. The puzzle asks: what is the probability that the last number painted is 6?",
    code: `from manim_imports_ext import *


class Ladybug(InteractiveScene):
    random_seed = 0

    def construct(self):
        # Add clock
        clock = self.get_clock()
        clock_points = [tick.get_start() for tick in clock.ticks]
        clock_anim = cycle_animation(ClockPassesTime(clock, 12 * 60, 12))
        self.add(clock_anim)

        # Lady bug lands on it
        ladybug = SVGMobject("ladybug")
        ladybug.set_height(0.7)
        ladybug.set_color(GREY_A)
        ladybug.set_shading(0.5, 0.5, 0)
        circle = Dot(fill_color=RED_E, radius=0.36 * ladybug.get_height())
        circle.move_to(ladybug, DOWN)
        bug = Group(circle, Point(), ladybug)
        bug.move_to(clock.ticks[0].get_start())

        path = VMobject()
        path.start_new_path(ORIGIN)
        for n in range(5):
            step = rotate_vector(RIGHT, PI * random.random())
            path.add_line_to(path.get_end() + step)
        path.make_smooth()
        path.put_start_and_end_on(7 * LEFT, clock_points[0])

        self.play(MoveAlongPath(bug, path, run_time=3))
        self.play(clock.numbers[0].animate.set_color(RED))

        bug.shift(UP)

        # Run simulation
        curr_number = 0
        covered_numbers = {0}
        while len(covered_numbers) < 12:
            step = random.choice([+1, -1])
            next_number = curr_number + step
            path_arc = -step * TAU / 12
            arrow = Arrow(
                1.2 * clock_points[curr_number],
                1.2 * clock_points[next_number],
                buff=0,
                fill_color=YELLOW,
                path_arc=path_arc,
                thickness=5,
            )

            end_color = RED
            if len(covered_numbers) == 11 and next_number not in covered_numbers:
                end_color = TEAL
            self.play(
                VFadeInThenOut(arrow),
                bug.animate.move_to(clock_points[next_number]).set_anim_args(path_arc=path_arc, time_span=(0, 0.5)),
                clock.numbers[next_number].animate.set_color(end_color)
            )
            curr_number = next_number
            covered_numbers.add(curr_number)

    def get_clock(self, radius=2):
        # Add clock (Todo, add these modifications as options to the Clock class)
        clock = Clock()
        clock.set_height(2 * radius)
        for line in [clock.hour_hand, clock.minute_hand, *clock.ticks]:
            line.scale(0.75, about_point=line.get_start())

        numbers = VGroup(Integer(n) for n in [12, *range(1, 12)])
        for number, theta in zip(numbers, np.arange(0, TAU, TAU / 12)):
            number.move_to(0.75 * radius * rotate_vector(UP, -theta))

        clock.numbers = numbers
        clock.add(numbers)
        return clock


class Question(InteractiveScene):
    def construct(self):
        text = Text("""
            What is the probability that
            the last number painted is 6?
        """)
        text.to_edge(UP)
        self.play(Write(text))
        self.wait()`,
    annotations: {
      1: "Standard ManimGL import: provides InteractiveScene, all Mobject types, animation classes, mathematical constants, and numpy.",
      4: "InteractiveScene subclass: the main scene for the ladybug puzzle. InteractiveScene provides mouse/keyboard interaction and 3D camera (self.frame).",
      5: "random_seed = 0: ManimGL's Scene class uses this to seed Python's random module in setup(), ensuring the random walk is reproducible across runs.",
      7: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
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
      80: "Question extends InteractiveScene. InteractiveScene provides self.frame (3D camera), mouse/keyboard interaction, and the full OpenGL rendering pipeline.",
      81: "construct() is the main entry point for this scene. ManimGL calls it when the scene runs — all animation logic goes here.",
    }
  };

})();