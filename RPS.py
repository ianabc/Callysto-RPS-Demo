import numpy as np
import random as rand
import ipywidgets as widgets

from collections import Counter
from ipycanvas import Canvas


class Solver:
    def __init__(self, dim1, dim2=None):
        """Generate solver for RPS of specific size

        Args:
            dim1 (int): size in x direction
            dim2 (int, optional): size in y direction. Defaults to dim1 if None.
        """
        self.dim1 = dim1
        if dim2 == None:
            dim2 = dim1
        self.dim2 = dim2
        # List of dictionaries that show which states are solvable
        # Each dictionary corresponds to the number of zeros possible for states of the given size
        # For example, self.solve_list[0] is for states with no zeros, and self.solve_list[-1] is for states with only 2 non-zeros
        # States with only 1 non-zero are trivially solved.
        # The keys in each dict are the relevant states (represented as str) and the values are a move to reduce the state while keeping it solvable
        # If a state is not solvable, the value is an empty string
        self.solve_list = [{} for _ in range(dim1 * dim2 - 1)]

    def add_solution(self, ans_list):
        """Adds list of states from generate_state as a possible solution

        Args:
            ans_list (list): List of states as np arrays
        """
        prev_str = None
        for state in ans_list:
            state_str = self.arr_to_str(state)
            if prev_str != None:
                num_zeros = state_str.count("0")
                self.solve_list[num_zeros][state_str] = prev_str
            prev_str = state_str

    def solve_all_full(self):
        """Solve all states, including partial states (contains 0s). Too slow for sizes 4x4 and above"""
        for n in range(2, self.dim1 * self.dim2 + 1):
            arrs = self.generate_all_states_with_n_elements(n)
            num_zeros = self.dim1 * self.dim2 - n
            for a in arrs:
                if not self.check_state_count(a) or not self.check_fully_connected(a):
                    self.solve_list[num_zeros][a] = ""
                    continue
                moves = self.generate_moves(a)
                found = False
                for m in moves:
                    if m.count("0") == self.dim1 * self.dim2 - 1:
                        self.solve_list[num_zeros][a] = m
                        found = True
                        break
                    elif self.solve_list[num_zeros + 1][m]:
                        self.solve_list[num_zeros][a] = m
                        found = True
                        break
                if not found:
                    self.solve_list[num_zeros][a] = ""
            print(n)
        return

    def solve_all(self):
        """Solve all complete states (all entries non-zero). Too slow for sizes 4x4 and above"""
        for i in range(3 ** (self.dim1 * self.dim2)):
            ternary = str(np.base_repr(i, base=3))
            ternary = "0" * (self.dim1 * self.dim2 - len(ternary)) + ternary
            s = ""
            for t in ternary:
                s += str(int(t) + 1)
            self.solve_state(s)

            # Might want progress bar & stop action
            #if i % 1000 == 0:
            #    print(i)

    def solve_state(self, state):
        """Solve next move of state and add result to self.solve_list

        Args:
            state (str): string description of state, given by arr_to_str

        Returns:
            str: Next move. Empty string if not solvable
        """
        num_zeros = state.count("0")
        if num_zeros == self.dim1 * self.dim2 - 1:
            return True
        if state in self.solve_list[num_zeros]:
            return self.solve_list[num_zeros][state]
        if not self.check_state_count(state) or not self.check_fully_connected(state):
            self.solve_list[num_zeros][state] = ""
            return ""
        moves = self.generate_moves(state)
        for m in moves:
            result = self.solve_state(m)
            if result:
                self.solve_list[num_zeros][state] = m
                return m
        self.solve_list[num_zeros][state] = ""
        return ""

    def arr_to_str(self, arr):
        """Turn state array into string

        Args:
            arr (np 2d array): 2d np array representing RPS state

        Returns:
            str: String representation of array
        """
        ans_str = ""
        for x in np.nditer(arr):
            ans_str = ans_str + str(int(x))
        return ans_str

    def str_to_arr(self, st):
        """Get np array from string

        Args:
            st (str): String representation of RPS state

        Returns:
            ndarray: 2D array for state
        """
        arr = np.zeros((self.dim1, self.dim2))
        for i, s in enumerate(st):
            arr[i // self.dim2][i % self.dim2] = int(s)
        return arr

    def generate_moves(self, st):
        """Generate all possible moves from initial state

        Args:
            st (str): String representation of state

        Returns:
            list[str]: List of strings that correspond to possible moves from initial state
        """
        moves = []
        for i, s in enumerate(st):
            if s == "0":
                continue
            if i % self.dim2 < self.dim2 - 1:
                s2 = st[i + 1]
                if s2 != "0" and s != s2:
                    if (int(s) - int(s2)) % 3 == 1:
                        move = st[:i] + "0" + s + st[i + 2 :]
                    else:
                        move = st[:i] + s2 + "0" + st[i + 2 :]
                    moves.append(move)
            if i < self.dim2 * (self.dim1 - 1):
                s2 = st[i + self.dim2]
                if s2 != "0" and s != s2:
                    if (int(s) - int(s2)) % 3 == 1:
                        move = (
                            st[:i]
                            + "0"
                            + st[i + 1 : i + self.dim2]
                            + s
                            + st[i + self.dim2 + 1 :]
                        )
                    else:
                        move = (
                            st[:i]
                            + s2
                            + st[i + 1 : i + self.dim2]
                            + "0"
                            + st[i + self.dim2 + 1 :]
                        )
                    moves.append(move)
        return moves

    def generate_all_states_with_n_elements(self, n):
        """Generate all states with n non-zero elements. Used for solve_all_full

        Args:
            n (int): Number of non-zero elements

        Returns:
            list[str]: All possible states with n non-zero elements
        """
        return self.recursive_generate_states(
            nonzeros=n, zeros=self.dim1 * self.dim2 - n
        )

    def recursive_generate_states(self, nonzeros, zeros, _prefix=""):
        """Recursively generate all states for generate_all_states_with_n_elements

        Args:
            nonzeros (int): Number of zeros
            zeros (int): Number of non-zeros
            _prefix (str, optional): Previously generated states. Defaults to ''.

        Returns:
            list[str]: List of generated states
        """
        if nonzeros == 0:
            return [_prefix + "0" * (zeros)]
        ans = []
        for i in range(zeros + 1):
            ans += (
                self.recursive_generate_states(
                    nonzeros - 1, zeros - i, _prefix + "0" * i + "1"
                )
                + self.recursive_generate_states(
                    nonzeros - 1, zeros - i, _prefix + "0" * i + "2"
                )
                + self.recursive_generate_states(
                    nonzeros - 1, zeros - i, _prefix + "0" * i + "3"
                )
            )
        return ans

    def check_state_count(self, st):
        """If a state only has two types of elements, check that the 'larger' occurs exactly once. Tried to improve computation time, doesn't seem to do much

        Args:
            st (str): current state

        Returns:
            bool: If the state satisfies the solvability criteria
        """
        num_hist = Counter(st)
        if num_hist["1"] == 0:
            if num_hist["3"] != 1:
                return False
        elif num_hist["2"] == 0:
            if num_hist["1"] != 1:
                return False
        elif num_hist["3"] == 0:
            if num_hist["2"] != 1:
                return False
        return True

    def check_fully_connected(self, st):
        """Check that all non-zero elements in the state are connected. If two parts are not connected the state cannot be solved. Tried to improve computation time, doesn't seem to do much

        Args:
            st (str): current state

        Returns:
            bool: If the state satisfies the solvability criteria
        """
        set_list = []
        for i, s in enumerate(st):
            if s == "0":
                continue
            found_idx1 = -1
            found_idx2 = -1
            if i % self.dim2 > 0:
                prev_square = i - 1
                for j, set1 in enumerate(set_list):
                    if prev_square in set1:
                        found_idx1 = j
                        set1.add(i)
                        break
            if i // self.dim2 > 0:
                prev_square = i - self.dim2
                for j, set2 in enumerate(set_list):
                    if prev_square in set2:
                        found_idx2 = j
                        set2.add(i)
                        break
            if found_idx1 == -1 and found_idx2 == -1:
                set_list.append({i})
            elif found_idx1 != found_idx2 and found_idx1 != -1 and found_idx2 != -1:
                set_list[found_idx1] = set1.union(set2)
                set_list.pop(found_idx2)
        return len(set_list) == 1


def generate_state(num_x, num_y=None):
    """Generate an initial array for rock paper scissors with a valid solution

    Args:
        num_x (int): dimension of array in the x direction
        num_y (int, optional): dimension of array in the y direction. Defaults to num_x if None

    Returns:
        array: 2D numpy array of a rock paper scissors state
    """
    if not num_y:
        num_y = num_x

    state = np.zeros([num_x, num_y])
    start_square = rand.randrange(num_x * num_y)
    start_obj = rand.randint(1, 3)

    state[start_square % num_x][start_square // num_x] = start_obj

    solution_list = [np.copy(state)]

    for _ in range(num_x * num_y - 1):
        # Generate list of possible moves
        moves = []
        for square in range(num_x * num_y):
            x = square % num_x
            y = square // num_x
            obj = state[x][y]

            # Check square below and to the right
            if x < num_x - 1:
                obj1 = state[x + 1][y]

                # Check if this square empty and other square full, or vice versa
                if obj == 0 and obj1 != 0:
                    moves.append((square, square + 1))
                elif obj != 0 and obj1 == 0:
                    moves.append((square + 1, square))

            if y < num_y - 1:
                obj1 = state[x][y + 1]
                if obj == 0 and obj1 != 0:
                    moves.append((square, square + num_x))
                elif obj != 0 and obj1 == 0:
                    moves.append((square + num_x, square))

        # Choose random move and execute
        move = rand.choice(moves)
        state[move[0] % num_x][move[0] // num_x] = state[move[1] % num_x][
            move[1] // num_x
        ]
        state[move[1] % num_x][move[1] // num_x] = (
            state[move[1] % num_x][move[1] // num_x] - 2
        ) % 3 + 1

        solution_list.append(np.copy(state))

    return state, solution_list


class RPSCanvas:
    def __init__(self, initial_state, square_size=200, solver=None):
        """Initialize rock paper scissors canvas

        Args:
            initial_state (ndarray): 2D numpy array
            square_size (int, optional): Size of each image in px. Defaults to 200.
            solver (Solver): Solver class. Defaults to None
        """
        self.initial_state = initial_state
        self.state = np.copy(initial_state)
        self.prev_states = []
        self.square_size = square_size
        self.solver = solver

        # Number of tiles in x and y direction
        self.num_x = self.initial_state.shape[0]
        self.num_y = self.initial_state.shape[1]

        # Menu params
        self.menu_offset = self.square_size // 2
        self.button_offset = self.square_size // 4
        self.button_length = self.square_size * 1.5
        self.button_height = self.square_size // 3
        self.button1_x = self.square_size * self.num_x + self.menu_offset
        self.button1_y = 0
        self.button2_x = self.button1_x
        self.button2_y = self.button_height + self.button_offset
        self.button3_x = self.button1_x
        self.button3_y = 2 * (self.button_height + self.button_offset)
        self.button4_x = self.button1_x
        self.button4_y = 3 * (self.button_height + self.button_offset)
        self.font_size = self.square_size // 4
        self.font_offset = self.font_size // 4

        # Total length in x and y directions
        x_length = self.square_size * self.num_x + self.menu_offset + self.button_length
        y_length = self.square_size * self.num_y

        # Import display images
        self.imgs = {
            1: widgets.Image.from_file("./images/rock.png"),
            2: widgets.Image.from_file("./images/paper.png"),
            3: widgets.Image.from_file("./images/scissors.png"),
        }

        # Store the index of the previous clicked square (-1 for no previous square)
        self.prev_square = -1

        # Make canvas
        self.canvas = Canvas(width=x_length, height=y_length)

        # show images
        for square in range(self.num_x * self.num_y):
            self.draw_img(square)

        self.canvas.on_mouse_down(self.handle_mouse_down)
        self.canvas.on_mouse_up(self.handle_mouse_up)

        # Draw menu buttons
        self.canvas.fill_style = "gray"
        self.canvas.fill_rect(
            self.button1_x, self.button1_y, self.button_length, self.button_height
        )
        self.canvas.fill_rect(
            self.button2_x, self.button2_y, self.button_length, self.button_height
        )
        self.canvas.fill_rect(
            self.button3_x, self.button3_y, self.button_length, self.button_height
        )
        self.canvas.fill_rect(
            self.button4_x, self.button4_y, self.button_length, self.button_height
        )

        self.canvas.fill_style = "black"
        self.canvas.text_align = "center"
        self.canvas.font = str(self.font_size) + "px serif"
        self.canvas.fill_text(
            "Reset",
            self.button1_x + self.button_length // 2,
            self.button1_y + self.button_height // 2 + self.font_offset,
        )
        self.canvas.fill_text(
            "Undo",
            self.button2_x + self.button_length // 2,
            self.button2_y + self.button_height // 2 + self.font_offset,
        )
        self.canvas.fill_text(
            "New Game",
            self.button3_x + self.button_length // 2,
            self.button3_y + self.button_height // 2 + self.font_offset,
        )
        self.canvas.fill_text(
            "Get Hint",
            self.button4_x + self.button_length // 2,
            self.button4_y + self.button_height // 2 + self.font_offset,
        )

    def square_to_xy(self, square):
        """Transform square number into x,y coordinates for top right corner of square

        Args:
            square (int): Number of square, from 0 to num_x*num_y

        Returns:
            tuple: (x,y) coordinates of top right corner
        """
        x = square % self.num_x * self.square_size
        y = square // self.num_x * self.square_size
        return (x, y)

    def xy_to_square(self, x, y):
        """Get square from coordinates

        Args:
            x (int): x coordinate (px)
            y (int): y coordinate (px)

        Returns:
            int: square number from 0 to num_x*num_y
        """
        return y // self.square_size * self.num_x + x // self.square_size

    def draw_img(self, square):
        """Draw image on square based on current state

        Args:
            square (int): square on which to draw
        """
        x, y = self.square_to_xy(square)
        img_idx = self.state[square % self.num_x][square // self.num_x]
        if img_idx == 0:
            return
        else:
            self.canvas.draw_image(
                self.imgs[img_idx],
                x=x,
                y=y,
                width=self.square_size,
                height=self.square_size,
            )

    def change_square_color(self, square, color):
        """Change square background color

        Args:
            square (int): number of square to color
            color (string): color - either HEX code or generic name
        """
        if square == -1:
            return
        self.canvas.fill_style = color
        x, y = self.square_to_xy(square)
        self.canvas.fill_rect(x, y, self.square_size)
        self.draw_img(square)

    def change_state(self, square):
        """Attempts to change state by moving the object on prev_square onto the current square

        Args:
            square (int): square that was clicked on

        Returns:
            bool: is the state change successful/valid
        """
        x1 = self.prev_square % self.num_x
        y1 = self.prev_square // self.num_x

        x2 = square % self.num_x
        y2 = square // self.num_x

        if abs(x1 - x2) + abs(y1 - y2) == 1:
            obj1 = self.state[x1][y1]
            obj2 = self.state[x2][y2]

            if (obj1 - obj2) % 3 == 1:
                self.prev_states.append(np.copy(self.state))
                self.state[x1][y1] = 0
                self.state[x2][y2] = obj1
                return True
        return False

    def is_done(self):
        """If there is only one object left, color the grid green"""
        if np.count_nonzero(self.state) == 1:
            for square in range(self.num_x * self.num_y):
                self.change_square_color(square, "green")
                self.draw_img(square)

    def reset(self):
        """Set the current state to the inital state"""
        self.prev_states.append(np.copy(self.state))
        self.state = np.copy(self.initial_state)
        for square in range(self.num_x * self.num_y):
            self.change_square_color(square, "white")
            self.draw_img(square)
        self.prev_states = []
        self.prev_square = -1

    def undo(self):
        """Undo the previous move"""
        self.state = self.prev_states.pop()
        for square in range(self.num_x * self.num_y):
            self.change_square_color(square, "white")
            self.draw_img(square)
        self.prev_square = -1

    def new_game(self):
        """Generate a new game with the same dimension"""
        self.initial_state, solution = generate_state(self.num_x, self.num_y)
        if self.solver is not None:
            self.solver.add_solution(solution)
        self.state = np.copy(self.initial_state)
        self.prev_states = []
        self.prev_square = -1
        for square in range(self.num_x * self.num_y):
            self.change_square_color(square, "white")
            self.draw_img(square)

    def get_hint(self):
        """Get a hint for the next move using the Solver class. If no solution avalible, turns grid red"""
        state_str = self.solver.arr_to_str(self.state)
        move_str = self.solver.solve_state(state_str)
        move = self.solver.str_to_arr(move_str)

        if move_str:
            for i in range(self.num_x):
                for j in range(self.num_y):
                    if self.state[i][j] != move[i][j]:
                        square = i + self.num_x * j
                        self.change_square_color(square, "yellow")
        else:
            for square in range(self.num_x * self.num_y):
                self.change_square_color(square, "red")

    def click_grid(self, x, y):
        """Handle clicking on the grid of squares

        Args:
            x (int): x position of the click
            y (int): y position of the click
        """
        square = self.xy_to_square(x, y)
        self.change_square_color(square, "#8ED6FF")
        if self.state[square % self.num_x][square // self.num_x] == 0:
            self.change_square_color(self.prev_square, "white")
            self.prev_square = -1
        elif self.prev_square == -1:
            self.change_square_color(square, "#8ED6FF")
            self.prev_square = square
        elif square == self.prev_square:
            self.change_square_color(self.prev_square, "white")
            self.prev_square = -1
        elif self.change_state(square):
            self.change_square_color(square, "white")
            self.change_square_color(self.prev_square, "white")
            self.prev_square = -1
            self.is_done()
        else:
            self.change_square_color(square, "#8ED6FF")
            self.change_square_color(self.prev_square, "white")
            self.prev_square = square

    def click_menu(self, x, y):
        """Handle clicking on the menu

        Args:
            x (int): x position of the click
            y (int): y position of the click
        """
        if (
            x > self.button1_x
            and y > self.button1_y
            and y < self.button1_y + self.button_height
        ):
            self.reset()
        elif (
            x > self.button2_x
            and y > self.button2_y
            and y < self.button2_y + self.button_height
        ):
            self.undo()
        elif (
            x > self.button3_x
            and y > self.button3_y
            and y < self.button3_y + self.button_height
        ):
            self.new_game()
        elif (
            x > self.button4_x
            and y > self.button4_y
            and y < self.button4_y + self.button_height
        ):
            self.get_hint()

    def handle_mouse_down(self, x, y):
        """Handle clicking down on the canvas

        Args:
            x (int): x position of the click
            y (int): y position of the click
        """
        if x < self.num_x * self.square_size:
            self.click_grid(x, y)
        else:
            self.click_menu(x, y)

    def handle_mouse_up(self, x, y):
        """Handle releasing the mouse button. Used to implement dragging as clicking the destination square if applicable

        Args:
            x (int): x position of the click
            y (int): y position of the click
        """
        if x < self.num_x * self.square_size:
            square = self.xy_to_square(x, y)
            if self.prev_square != -1 and square != self.prev_square:
                self.click_grid(x, y)
