import numpy as np
import random
from math import dist
from PIL import Image
import imageio.v3 as iio

# Размер поля
GRID_SIZE = 5

# Действия
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3
ACTIONS = [UP, DOWN, LEFT, RIGHT]

# Ограничения
MAX_STEPS = 200

class SnakeEnv:
    def __init__(self):
        self.actions = ACTIONS
        self.grid_size = GRID_SIZE
        self.win_counter = 0
        self.steps = 0
        self.frames = []
        self.record_gif_on_win = True  # вкл/выкл запись побед
        self.reset()

    def get_state(self):
        head = self.snake[0]

        def next_pos(pos, dir):
            x, y = pos
            return {
                UP: (x, y - 1),
                DOWN: (x, y + 1),
                LEFT: (x - 1, y),
                RIGHT: (x + 1, y)
            }[dir]

        def is_danger(pos):
            x, y = pos
            return (
                    x < 0 or x >= self.grid_size or
                    y < 0 or y >= self.grid_size or
                    pos in self.snake
            )

        def turn_left(d): return [LEFT, DOWN, RIGHT, UP][d]

        def turn_right(d): return [RIGHT, UP, LEFT, DOWN][d]

        # Опасности
        danger_straight = int(is_danger(next_pos(head, self.direction)))
        danger_left = int(is_danger(next_pos(head, turn_left(self.direction))))
        danger_right = int(is_danger(next_pos(head, turn_right(self.direction))))

        # Текущие направление змеи
        dir_v = np.zeros(4, dtype=int)
        dir_v[self.direction] = 1

        # Положение яблока относительно головы
        apple_x, apple_y = self.apple
        head_x, head_y = head
        apple_up = int(apple_y < head_y)
        apple_down = int(apple_y > head_y)
        apple_left = int(apple_x < head_x)
        apple_right = int(apple_x > head_x)

        # Расстояния до стен (нормированные)
        dist_left = head_x / (self.grid_size - 1)
        dist_right = (self.grid_size - 1 - head_x) / (self.grid_size - 1)
        dist_up = head_y / (self.grid_size - 1)
        dist_down = (self.grid_size - 1 - head_y) / (self.grid_size - 1)

        # Евклидово расстояние до яблока
        max_dist = ((self.grid_size - 1) ** 2 * 2) ** 0.5
        euclid_dist = dist(self.apple, head) / max_dist

        # длина змеи
        length = len(self.snake)

        return [
            # Позиция головы
            head_x, head_y,
            # Опасности
            danger_straight, danger_right, danger_left,
            # Направления
            *dir_v,
            # Яблоко
            apple_up, apple_down, apple_left, apple_right,
            # Расстояния до стен
            dist_left, dist_right, dist_up, dist_down,
            # Евклидово расстояние до яблока
            euclid_dist,
            # Обычная длинна змеи
            length
        ]

    def reset(self):
        self.snake = [(3, 3), (3, 2), (3, 1)]
        self.direction = UP
        self.spawn_apple()
        self.done = False
        self.score = 0
        self.frames = []
        self.steps = 0
        return self.get_state()

    def spawn_apple(self):
        while True:
            self.apple = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if self.apple not in self.snake:
                break

    def _render_frame(self):
        grid = np.zeros((self.grid_size, self.grid_size, 3), dtype=np.uint8)
        for x, y in self.snake:
            grid[y][x] = [0, 255, 0]
        head_x, head_y = self.snake[0]
        grid[head_y][head_x] = [0, 0, 255]
        ax, ay = self.apple
        grid[ay][ax] = [255, 0, 0]
        return grid

    def step(self, action):
        if self.done:
            return self.get_state(), 0, "done"

        head_x, head_y = self.snake[0]

        if (action == UP and self.direction != DOWN) or \
                (action == DOWN and self.direction != UP) or \
                (action == LEFT and self.direction != RIGHT) or \
                (action == RIGHT and self.direction != LEFT):
            self.direction = action
        else:
            self.done = True
            return self.get_state(), -2, "incorrect_move"

        if self.direction == UP:
            head_y -= 1
        elif self.direction == DOWN:
            head_y += 1
        elif self.direction == LEFT:
            head_x -= 1
        elif self.direction == RIGHT:
            head_x += 1

        new_head = (head_x, head_y)

        if (
                head_x < 0 or head_x >= self.grid_size or
                head_y < 0 or head_y >= self.grid_size or
                new_head in self.snake
        ):
            self.done = True
            return self.get_state(), -3, "death"

        old_head = self.snake[0]
        self.snake.insert(0, new_head)

        if self.record_gif_on_win:
            frame = self._render_frame()
            frame = Image.fromarray(frame).resize((200, 200), resample=Image.NEAREST)
            self.frames.append(np.array(frame))

        self.steps += 1

        if self.steps >= MAX_STEPS:
            return self.get_state(), -3, "timeout"

        if len(self.snake) == self.grid_size * self.grid_size:
            self.done = True
            if self.record_gif_on_win:
                self.win_counter += 1
                iio.imwrite(f"gif/snake_win_{self.win_counter}.gif", self.frames, fps=5)
                print(f"Победный эпизод сохранён: snake_win_{self.win_counter}.gif")
            return self.get_state(), 2000.0, "win"

        could_eat = (abs(self.apple[0] - old_head[0]) + abs(self.apple[1] - old_head[1])) == 1
        chose_not_to = new_head != self.apple
        if could_eat and chose_not_to:
            def is_dead_after(pos, snake_body):
                for act in ACTIONS:
                    x, y = pos
                    if act == UP:
                        y -= 1
                    elif act == DOWN:
                        y += 1
                    elif act == LEFT:
                        x -= 1
                    elif act == RIGHT:
                        x += 1
                    next_pos = (x, y)
                    if 0 <= x < self.grid_size and 0 <= y < self.grid_size and next_pos not in snake_body:
                        return False  # есть безопасный ход
                return True  # все ходы ведут к смерти

            future_snake_body = self.snake.copy()
            future_snake_body.pop()

            if not is_dead_after(new_head, future_snake_body):
                self.snake.pop()
                return self.get_state(), -1.5, False

        if new_head == self.apple:
            self.score += 1
            reward = 35 + len(self.snake)
            self.spawn_apple()
        else:
            self.snake.pop()
            reward = 0

        return self.get_state(), reward, False
