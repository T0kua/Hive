# hive_full_ru.py
import pygame
import math
from collections import deque

pygame.init()

# --- Настройки окна и поля ---
WIDTH, HEIGHT = 1200, 820
PANEL_W = 260
HEX_SIZE = 36
COLS, ROWS = 13, 11
FPS = 60

# --- Цвета ---
BG = (24, 24, 24)
UI_BG = (38, 38, 44)
HEX_BORDER = (90, 90, 96)
HEX_BG = (60, 60, 66)
HIGHLIGHT = (110, 190, 110)
SELECT = (200, 90, 110)
WHITE = (245, 245, 245)
BLACK = (18, 18, 18)

PLAYER_COLOR = {"Белый": (235, 235, 235), "Чёрный": (50, 50, 54)}

SQRT3 = math.sqrt(3)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hive — полная версия (RU)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("arial", 16)
title_font = pygame.font.SysFont("arial", 22, bold=True)

# --- Геометрия (axial pointy-top) ---
DIRECTIONS = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]


def axial_to_pixel(q, r, size, ox=0, oy=0):
    x = ox + size * 1.5 * q
    y = oy + size * SQRT3 * (r + q / 2)
    return (x, y)


def polygon_points(center, r):
    x0, y0 = center
    return [(x0 + r * math.cos(math.radians(60 * i)),
             y0 + r * math.sin(math.radians(60 * i))) for i in range(6)]


# --- Классы игры ---
class Piece:
    def __init__(self, owner, kind):
        self.owner = owner  # "Белый"/"Чёрный"
        self.kind = kind  # "Пчела","Жук","Муравей","Кузнечик","Паук"
        self.color = PLAYER_COLOR[owner]

    def short(self):
        return {"Пчела": "Q", "Жук": "B", "Муравей": "A", "Кузнечик": "G", "Паук": "S"}[self.kind]


class HexCell:
    def __init__(self, q, r, center, size):
        self.q, self.r = q, r
        self.center = center
        self.size = size
        self.points = polygon_points(center, size - 2)
        self.stack = []  # стек фишек; верхний элемент - активная фишка
        self.highlight = False

    def occupied(self):
        return len(self.stack) > 0

    def top_piece(self):
        return self.stack[-1] if self.stack else None

    def draw(self, surf):
        # фон: подсветка > верхняя фишка > обычная
        if self.highlight:
            fill = HIGHLIGHT
        elif self.occupied():
            fill = self.top_piece().color
        else:
            fill = HEX_BG

        pygame.draw.polygon(surf, fill, self.points)
        pygame.draw.aalines(surf, HEX_BORDER, True, self.points)

        if self.occupied():
            top = self.top_piece()
            text_color = BLACK if sum(top.color) > 400 else WHITE
            txt = font.render(top.short(), True, text_color)
            rect = txt.get_rect(center=self.center)
            surf.blit(txt, rect)
            if len(self.stack) > 1:
                small = font.render(str(len(self.stack)), True, (200, 160, 80))
                surf.blit(small, (self.center[0] + self.size // 2 - 8, self.center[1] - self.size // 2 + 6))

    def inside(self, pos):
        x, y = pos
        cx, cy = self.center
        return (x - cx) ** 2 + (y - cy) ** 2 <= (self.size * 0.95) ** 2


class Board:
    def __init__(self, cols, rows, size):
        self.cols, self.rows = cols, rows
        self.size = size
        self.cells = {}
        self.make_grid()

    def make_grid(self):
        total_w = self.size * (1.5 * (self.cols - 1) + 1)
        total_h = self.size * SQRT3 * (self.rows + 0.5)
        ox = PANEL_W + (WIDTH - PANEL_W - total_w) / 2 + self.size
        oy = (HEIGHT - total_h) / 2 + self.size

        for col in range(self.cols):
            for row in range(self.rows):
                # центрируем axial coords
                q = col - self.cols // 2
                r = row - self.rows // 2 - (col // 2)
                x = ox + col * self.size * 1.5
                y = oy + (row + 0.5 * (col % 2)) * SQRT3 * self.size
                center = (x, y)
                self.cells[(q, r)] = HexCell(q, r, center, self.size)

    def draw(self, surf):
        for c in self.cells.values():
            c.draw(surf)

    def neighbors_coords(self, q, r):
        res = []
        for dq, dr in DIRECTIONS:
            coord = (q + dq, r + dr)
            if coord in self.cells:
                res.append(coord)
        return res

    def neighbors(self, coord):
        q, r = coord
        return [self.cells[c] for c in self.neighbors_coords(q, r)]

    def cell_at_pos(self, pos):
        for c in self.cells.values():
            if c.inside(pos):
                return c
        return None

    def occupied_coords(self):
        return [coord for coord, cell in self.cells.items() if cell.occupied()]

    def clear_highlights(self):
        for c in self.cells.values():
            c.highlight = False


# --- Утилиты правил ---
def is_connected_after_change(board: Board, removed_coord=None, added_coord=None):
    occ = set(board.occupied_coords())
    if removed_coord and removed_coord in occ:
        occ.remove(removed_coord)
    if added_coord:
        occ.add(added_coord)
    if not occ:
        return True
    start = next(iter(occ))
    q = deque([start])
    seen = {start}
    while q:
        cur = q.popleft()
        for nb in board.neighbors_coords(*cur):
            if nb in occ and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(occ)


def dir_index(from_coord, to_coord):
    fq, fr = from_coord
    tq, tr = to_coord
    dq = tq - fq
    dr = tr - fr
    for i, (mdq, mdr) in enumerate(DIRECTIONS):
        if (dq, dr) == (mdq, mdr):
            return i
    return None


def can_slide_between(board: Board, from_coord, to_coord, treat_from_as_empty=False):
    """
    Проверка базового 'sliding' между соседними клетками:
    нельзя протиснуться, если обе клетки по обе стороны от ребра заняты.
    treat_from_as_empty - если планируем временно убрать верхнюю фишку из from_coord при проверке.
    """
    idx = dir_index(from_coord, to_coord)
    if idx is None:
        return False
    fq, fr = from_coord
    left_idx = (idx - 1) % 6
    right_idx = (idx + 1) % 6
    a = (fq + DIRECTIONS[left_idx][0], fr + DIRECTIONS[left_idx][1])
    b = (fq + DIRECTIONS[right_idx][0], fr + DIRECTIONS[right_idx][1])
    occ_a = board.cells[a].occupied() if a in board.cells else False
    occ_b = board.cells[b].occupied() if b in board.cells else False
    # Если from_coord временно считается пустым, это не влияет на a/b (они не равны from)
    # Протиснуться нельзя, если оба соседних места заняты.
    return not (occ_a and occ_b)


# --- Ходы фигур ---
def moves_for_queen(board: Board, coord):
    q, r = coord
    possible = []
    for nb in board.neighbors_coords(q, r):
        cell = board.cells[nb]
        if not cell.occupied():
            if can_slide_between(board, coord, nb, treat_from_as_empty=True):
                possible.append(nb)
    return possible


def moves_for_beetle(board: Board, coord):
    q, r = coord
    possible = []
    for nb in board.neighbors_coords(q, r):
        # Beetle может перемещаться на любую соседнюю клетку (включая занятую)
        possible.append(nb)
    return possible


def moves_for_grasshopper(board: Board, coord):
    q, r = coord
    result = []
    for (dq, dr) in DIRECTIONS:
        cur_q, cur_r = q + dq, r + dr
        jumped = False
        while (cur_q, cur_r) in board.cells and board.cells[(cur_q, cur_r)].occupied():
            jumped = True
            cur_q += dq
            cur_r += dr
        if jumped and (cur_q, cur_r) in board.cells and not board.cells[(cur_q, cur_r)].occupied():
            result.append((cur_q, cur_r))
    return result


def moves_for_ant(board: Board, coord):
    # Ant: может достичь любой пустой клетки, достижимой sliding-путём
    q0, r0 = coord
    visited = set()
    q = deque()
    # Рассматриваем первый шаг: соседние пустые клетки, доступные при снятии с источника
    for nb in board.neighbors_coords(q0, r0):
        if not board.cells[nb].occupied() and can_slide_between(board, coord, nb, treat_from_as_empty=True):
            visited.add(nb)
            q.append(nb)
    while q:
        cur = q.popleft()
        for nb in board.neighbors_coords(*cur):
            if nb in visited:
                continue
            if board.cells[nb].occupied():
                continue
            if can_slide_between(board, cur, nb, treat_from_as_empty=False):
                visited.add(nb)
                q.append(nb)
    return list(visited)


def moves_for_spider(board: Board, coord):
    # Паук: ровно 3 шага sliding, нельзя возвращаться на ранее посещённую клетку
    results = set()
    start = coord

    def dfs(path, depth):
        if depth == 3:
            results.add(path[-1])
            return
        cur = path[-1]
        for nb in board.neighbors_coords(*cur):
            if nb in path:
                continue
            if board.cells[nb].occupied():
                continue
            # treat_from_as_empty = True только когда текущая клетка == start и мы сняли верхнюю
            treat = True if cur == start else False
            if can_slide_between(board, cur, nb, treat_from_as_empty=treat):
                dfs(path + [nb], depth + 1)

    for nb in board.neighbors_coords(*start):
        if not board.cells[nb].occupied() and can_slide_between(board, start, nb, treat_from_as_empty=True):
            dfs([start, nb], 1)
    return list(results)


# --- Правила постановки (исправлены) ---
def valid_placement(board: Board, coord, player):
    # нельзя ставить на занятую клетку
    if board.cells[coord].occupied():
        return False, "Клетка занята."

    occ = set(board.occupied_coords())

    # если поле пустое — можно ставить куда угодно
    if not occ:
        return True, ""

    # проверяем, есть ли у игрока уже фишки на поле
    has_own_on_board = any(board.cells[c].top_piece().owner == player for c in occ)

    neigh = board.neighbors_coords(*coord)

    # если у игрока нет фишек (его первый ход) — можно ставить рядом с любыми существующими
    if not has_own_on_board:
        if not any(board.cells[n].occupied() for n in neigh):
            return False, "Первую фишку нужно ставить рядом с существующим ульем."
        # и нельзя разрывать улей
        if not is_connected_after_change(board, removed_coord=None, added_coord=coord):
            return False, "Нельзя разрывать улей."
        return True, ""

    # если у игрока уже есть фишки:
    # должна касаться хотя бы одной своей фишки
    has_own_neigh = any(board.cells[n].occupied() and board.cells[n].top_piece().owner == player for n in neigh)
    if not has_own_neigh:
        return False, "Нужно ставить рядом с вашей частью улья."

    # и не должна касаться чужих
    has_enemy_neigh = any(board.cells[n].occupied() and board.cells[n].top_piece().owner != player for n in neigh)
    if has_enemy_neigh:
        return False, "Нельзя ставить рядом с вражескими фишками."

    # и поле после размещения должно оставаться связным
    if not is_connected_after_change(board, removed_coord=None, added_coord=coord):
        return False, "Нельзя разрывать улей."

    return True, ""


# --- Основная логика ---
class Game:
    def __init__(self):
        self.board = Board(COLS, ROWS, HEX_SIZE)
        self.turn = "Белый"
        self.selected_piece_type = None  # для постановки
        self.selected_cell = None  # координата выбранной клетки для перемещения
        self.inventory = {
            "Белый": {"Пчела": 1, "Жук": 2, "Муравей": 3, "Кузнечик": 2, "Паук": 2},
            "Чёрный": {"Пчела": 1, "Жук": 2, "Муравей": 3, "Кузнечик": 2, "Паук": 2}
        }
        # сколько ходов уже сделал каждый игрок (нужно для правила пчелы к 4-му ходу)
        self.moves_made = {"Белый": 0, "Чёрный": 0}
        self.message = "Выберите фишку слева или кликните по своей фишке."
        self.game_over = False

    def draw_ui(self, surf):
        panel = pygame.Rect(0, 0, PANEL_W, HEIGHT)
        pygame.draw.rect(surf, UI_BG, panel)
        surf.blit(title_font.render("Hive", True, WHITE), (20, 12))
        surf.blit(font.render(f"Ход: {self.turn if not self.game_over else '—'}", True, WHITE), (20, 46))
        surf.blit(font.render(self.message, True, WHITE), (20, 74))

        y = 120
        btn_h = 36
        gap = 10
        for kind in ["Пчела", "Жук", "Муравей", "Кузнечик", "Паук"]:
            rect = pygame.Rect(20, y, PANEL_W - 40, btn_h)
            color = SELECT if self.selected_piece_type == kind else (64, 64, 72)
            pygame.draw.rect(surf, color, rect, border_radius=8)
            txt = font.render(f"{kind}  x{self.inventory[self.turn][kind]}", True, WHITE)
            surf.blit(txt, (32, y + 8))
            y += btn_h + gap

        instr = [
            "Клики:",
            "- ЛКМ по кнопке: выбрать фишку для постановки",
            "- ЛКМ по пустой клетке: поставить (если разрешено)",
            "- ЛКМ по вашей фишке: выбрать -> подсветить ходы",
            "- ЛКМ по подсвеченной клетке: выполнить ход",
        ]
        y2 = HEIGHT - 140
        for i, ln in enumerate(instr):
            surf.blit(font.render(ln, True, WHITE), (18, y2 + i * 20))

    def handle_panel_click(self, pos):
        if pos[0] > PANEL_W:
            return False
        y = 120
        btn_h = 36
        gap = 10
        for kind in ["Пчела", "Жук", "Муравей", "Кузнечик", "Паук"]:
            rect = pygame.Rect(20, y, PANEL_W - 40, btn_h)
            if rect.collidepoint(pos):
                if self.inventory[self.turn][kind] > 0:
                    self.selected_piece_type = kind
                    self.selected_cell = None
                    self.board.clear_highlights()
                    self.message = f"Выбрана фишка {kind}. Кликните по клетке для постановки."
                else:
                    self.message = f"Фишки {kind} закончились."
                return True
            y += btn_h + gap
        return True

    def place_piece(self, coord):
        if self.game_over:
            return False
        if not self.selected_piece_type:
            self.message = "Сначала выберите тип фишки."
            return False

        # правило: если игрок уже сделал 3 свои хода (т.е. на очереди 4-й ход),
        # он обязан в этот ход поставить Пчелу (если ещё не поставил).
        if self.moves_made[self.turn] >= 3:
            # проверим, есть ли уже пчела этого игрока на поле
            queen_on_board = any(
                c.top_piece() and c.top_piece().owner == self.turn and c.top_piece().kind == "Пчела"
                for c in self.board.cells.values()
            )
            if not queen_on_board and self.selected_piece_type != "Пчела":
                self.message = "Вы обязаны поставить Пчелу не позже 4-го хода."
                return False

        ok, reason = valid_placement(self.board, coord, self.turn)
        if not ok:
            self.message = reason
            return False

        cell = self.board.cells[coord]
        # ставим: добавляем в стек
        piece = Piece(self.turn, self.selected_piece_type)
        cell.stack.append(piece)
        self.inventory[self.turn][self.selected_piece_type] -= 1

        # отмечаем ход
        self.moves_made[self.turn] += 1

        # проверка победы
        winner = self.check_victory()
        if winner:
            self.message = f"Победа! {winner} окружил(а) вражескую Пчелу."
            self.game_over = True
            return True

        # смена хода
        self.selected_piece_type = None
        self.board.clear_highlights()
        self.turn = "Чёрный" if self.turn == "Белый" else "Белый"
        self.message = f"Фишка поставлена. Ход: {self.turn}"
        return True

    def select_existing(self, coord):
        if self.game_over:
            return False
        cell = self.board.cells[coord]
        if not cell.occupied():
            self.message = "В этой клетке нет фишки."
            return False
        top = cell.top_piece()
        if top.owner != self.turn:
            self.message = "Это не ваша фишка."
            return False

        # верхняя фишка выбирается
        self.selected_cell = coord
        self.board.clear_highlights()

        kind = top.kind
        if kind == "Пчела":
            moves = moves_for_queen(self.board, coord)
        elif kind == "Жук":
            moves = moves_for_beetle(self.board, coord)
        elif kind == "Кузнечик":
            moves = moves_for_grasshopper(self.board, coord)
        elif kind == "Муравей":
            moves = moves_for_ant(self.board, coord)
        elif kind == "Паук":
            moves = moves_for_spider(self.board, coord)
        else:
            moves = []

        legal = []
        for tgt in moves:
            # Проверяем: если исходная клетка была единственной занятостью, удаление может разорвать
            src_cell = self.board.cells[coord]
            src_was_single = (len(src_cell.stack) == 1)
            remove_coord_for_check = coord if src_was_single else None
            # Beetle может перемещаться на занятые клетки; остальные нет
            tgt_cell = self.board.cells[tgt]
            if tgt_cell.occupied() and top.kind != "Жук":
                continue
            # проверка связности после перемещения (удаляем, затем добавляем)
            if not is_connected_after_change(self.board, removed_coord=remove_coord_for_check, added_coord=tgt):
                continue
            # для неползучих фигур с sliding мы уже проверяли sliding в генераторах
            legal.append(tgt)

        for c in legal:
            self.board.cells[c].highlight = True

        if legal:
            self.message = "Выбрана фишка. Подсвечены допустимые ходы."
        else:
            self.message = "У этой фишки нет допустимых ходов."
        return True

    def move_selected_to(self, coord):
        if self.game_over:
            return False
        if self.selected_cell is None:
            self.message = "Нет выбранной фишки."
            return False
        if not self.board.cells[coord].highlight:
            self.message = "Это не допустимый ход."
            return False

        src = self.selected_cell
        src_cell = self.board.cells[src]
        piece = src_cell.stack.pop()  # снимаем верхнюю фишку
        self.board.cells[coord].stack.append(piece)

        # отмечаем ход
        self.moves_made[self.turn] += 1

        # проверка победы
        winner = self.check_victory()
        if winner:
            self.message = f"Победа! {winner} окружил(а) вражескую Пчелу."
            self.game_over = True
            return True

        self.selected_cell = None
        self.board.clear_highlights()
        self.turn = "Чёрный" if self.turn == "Белый" else "Белый"
        self.message = f"Ход выполнен. Ход: {self.turn}"
        return True

    def check_victory(self):
        # ищем обе пчелы; если одна полностью окружена (все 6 соседей заняты), то победитель — противоположный игрок
        bees = []
        for coord, c in self.board.cells.items():
            if c.occupied() and c.top_piece().kind == "Пчела":
                bees.append((coord, c.top_piece().owner))
        for coord, owner in bees:
            neigh_coords = self.board.neighbors_coords(*coord)
            # Нужно проверить ровно 6 соседей внутри доски: если меньше 6 (граница), считать только существующие
            if neigh_coords and all(self.board.cells[n].occupied() for n in neigh_coords):
                # победитель — оппонент владельца окружённой пчелы
                return "Белый" if owner == "Чёрный" else "Чёрный"
        return None

    def click(self, pos):
        if self.game_over:
            return
        # Панель слева?
        if pos[0] <= PANEL_W:
            self.handle_panel_click(pos)
            return

        cell = self.board.cell_at_pos(pos)
        if not cell:
            self.message = "Клик вне поля."
            return
        coord = (cell.q, cell.r)

        # если есть подсвеченная клетка и выбрана фишка — выполнить ход
        if cell.highlight and self.selected_cell is not None:
            self.move_selected_to(coord)
            return

        # если выбрана фишка для постановки — поставить
        if self.selected_piece_type:
            self.place_piece(coord)
            return

        # иначе, если на клетке есть фишка — попробовать выбрать её
        if cell.occupied():
            self.select_existing(coord)
            return

        self.message = "Пустая клетка. Выберите фишку слева для постановки."

# --- Главный цикл ---
def main():
    game = Game()
    running = True
    while running:
        clock.tick(FPS)
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type == pygame.MOUSEBUTTONDOWN and ev.button == 1:
                game.click(ev.pos)

        screen.fill(BG)
        game.board.draw(screen)
        game.draw_ui(screen)

        # маркер выбранной клетки
        if game.selected_cell:
            c = game.board.cells[game.selected_cell]
            pygame.draw.circle(screen, SELECT, c.center, 6)

        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
