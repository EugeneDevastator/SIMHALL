import pyray as rl
import random
import math

TILE_SIZE = 64
GRID_WIDTH = 24
SCREEN_WIDTH = TILE_SIZE * GRID_WIDTH
SCREEN_HEIGHT = 1080
LEVEL_HEIGHT_TILES = 10000
SEED = 42
SCROLL_SPEED_DEFAULT = 200.0
SCROLL_SPEED_STEP = 1500.0
FONT_SIZE = 32
NOISE_RADIUS = 15.0

CHUNK_HEIGHT = 32
CHUNK_CACHE_RADIUS = 4

STONE_W = 24
STONE_H = 12
STONE_PROB = 0.30
STONE_COLOR = rl.Color(140, 140, 140, 255)

def _fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def _lerp(a, b, t):
    return a + t * (b - a)

def _grad(h, x, y):
    h = h & 3
    if h == 0: return  x + y
    if h == 1: return -x + y
    if h == 2: return  x - y
    return             -x - y

def perlin2(x, y, perm):
    xi = int(math.floor(x)) & 255
    yi = int(math.floor(y)) & 255
    xf = x - math.floor(x)
    yf = y - math.floor(y)
    u = _fade(xf)
    v = _fade(yf)
    aa = perm[perm[xi    ] + yi    ]
    ab = perm[perm[xi    ] + yi + 1]
    ba = perm[perm[xi + 1] + yi    ]
    bb = perm[perm[xi + 1] + yi + 1]
    x1 = _lerp(_grad(aa, xf,     yf    ), _grad(ba, xf - 1, yf    ), u)
    x2 = _lerp(_grad(ab, xf,     yf - 1), _grad(bb, xf - 1, yf - 1), u)
    return _lerp(x1, x2, v)

def make_perm(seed):
    rng = random.Random(seed)
    p = list(range(256))
    rng.shuffle(p)
    return (p + p) * 2

# deterministic per-cell rng — no object, just inline hash
def cell_rng(col, row, salt=0):
    return random.Random((col * 73856093) ^ (row * 19349663) ^ (salt * 83492791) ^ SEED)

class TileDef:
    def __init__(self, name, color, prop_prob=0.0, prop_color=None):
        self.name = name
        self.color = color
        self.prop_prob = prop_prob          # 0.0-1.0
        self.prop_color = prop_color        # rl.Color or None

class WeightKey:
    def __init__(self, row, weight):
        self.row = row
        self.weight = weight

class TileTrack:
    def __init__(self, tile_def, perm, ox, oy):
        self.tile_def = tile_def
        self.keys = []
        self.perm = perm
        self.ox = ox
        self.oy = oy

    def set_weight(self, row, weight):
        self.keys.append(WeightKey(row, weight))
        self.keys.sort(key=lambda k: k.row)

    def get_base_weight(self, row):
        if not self.keys:
            return 0.0
        if row <= self.keys[0].row:
            return self.keys[0].weight
        if row >= self.keys[-1].row:
            return self.keys[-1].weight
        for i in range(len(self.keys) - 1):
            a = self.keys[i]
            b = self.keys[i + 1]
            if a.row <= row <= b.row:
                t = (row - a.row) / (b.row - a.row)
                return a.weight + t * (b.weight - a.weight)
        return 0.0

    def get_score(self, col, row):
        base = self.get_base_weight(row)
        if base <= 0.0:
            return 0.0
        nx = (col + self.ox) / NOISE_RADIUS
        ny = (row + self.oy) / NOISE_RADIUS
        n = perlin2(nx, ny, self.perm)
        return base * (n * 0.5 + 0.5)

class LineEntry:
    def __init__(self, tile_def, col_start, col_end, row_start, row_end):
        self.tile_def = tile_def
        self.col_start = col_start
        self.col_end = col_end
        self.row_start = row_start
        self.row_end = row_end

class LevelGenerator:
    def __init__(self, seed, width_tiles, height_tiles):
        self.seed = seed
        self.width = width_tiles
        self.height = height_tiles
        self.tile_defs = {}
        self.tracks = {}
        self.line_entries = []
        self._chunks = {}
        self._master_rng = random.Random(seed)

    def define_tile(self, name, color, prop_prob=0.0, prop_color=None):
        td = TileDef(name, color, prop_prob, prop_color)
        self.tile_defs[name] = td
        perm = make_perm(self._master_rng.randint(0, 2**31))
        ox = self._master_rng.uniform(0, 1000)
        oy = self._master_rng.uniform(0, 1000)
        self.tracks[name] = TileTrack(td, perm, ox, oy)

    def settile(self, name, row, prob):
        self.tracks[name].set_weight(row, prob)

    def growto(self, name, row_from, row_to, prob_from, prob_to):
        self.tracks[name].set_weight(row_from, prob_from)
        self.tracks[name].set_weight(row_to, prob_to)

    def longline(self, name, col_start, col_end, row_start, row_end):
        self.line_entries.append(LineEntry(self.tile_defs[name], col_start, col_end, row_start, row_end))

    def _build_chunk(self, chunk_idx):
        row_start = chunk_idx * CHUNK_HEIGHT
        row_end = row_start + CHUNK_HEIGHT
        rows = []
        for row in range(row_start, row_end):
            actual_row = row % self.height
            best_scores = [-1.0] * self.width
            best_tiles = [None] * self.width
            for track in self.tracks.values():
                for col in range(self.width):
                    s = track.get_score(col, actual_row)
                    if s > best_scores[col]:
                        best_scores[col] = s
                        best_tiles[col] = track.tile_def
            rows.append(best_tiles)

        for ln in self.line_entries:
            dr = ln.row_end - ln.row_start
            dc = ln.col_end - ln.col_start
            steps = max(abs(dr), abs(dc), 1)
            for i in range(steps + 1):
                t = i / steps
                r = round(ln.row_start + t * dr)
                c = round(ln.col_start + t * dc)
                if row_start <= r < row_end and 0 <= c < self.width:
                    rows[r - row_start][c] = ln.tile_def

        return rows

    def _ensure_chunk(self, chunk_idx):
        if chunk_idx not in self._chunks:
            self._chunks[chunk_idx] = self._build_chunk(chunk_idx)

    def evict_chunks(self, current_chunk):
        to_del = [k for k in self._chunks if abs(k - current_chunk) > CHUNK_CACHE_RADIUS]
        for k in to_del:
            del self._chunks[k]

    def ensure_chunks_around(self, current_chunk):
        for ci in range(current_chunk - 1, current_chunk + CHUNK_CACHE_RADIUS + 1):
            if ci >= 0:
                self._ensure_chunk(ci)
        self.evict_chunks(current_chunk)

    def get_tile(self, row, col):
        row = row % self.height
        chunk_idx = row // CHUNK_HEIGHT
        local_row = row % CHUNK_HEIGHT
        if chunk_idx not in self._chunks:
            self._ensure_chunk(chunk_idx)
        if 0 <= col < self.width:
            return self._chunks[chunk_idx][local_row][col]
        return None


def draw_prop(tile, draw_x, draw_y, col, row):
    if tile is None or tile.prop_prob <= 0.0 or tile.prop_color is None:
        return
    rng = cell_rng(col, row)
    if rng.random() > tile.prop_prob:
        return
    # random position inside tile, keeping stone fully inside
    max_ox = TILE_SIZE - STONE_W
    max_oy = TILE_SIZE - STONE_H
    ox = rng.randint(0, max_ox)
    oy = rng.randint(0, max_oy)
    rl.draw_rectangle(draw_x + ox, draw_y + oy, STONE_W, STONE_H, tile.prop_color)


def main():
    gen = LevelGenerator(SEED, GRID_WIDTH, LEVEL_HEIGHT_TILES)

    gen.define_tile("desert1", rl.Color(210, 180, 100, 255), prop_prob=STONE_PROB, prop_color=STONE_COLOR)
    gen.define_tile("desert2", rl.Color(190, 160,  80, 255))
    gen.define_tile("forest",  rl.Color( 34, 120,  34, 255))
    gen.define_tile("canyon",  rl.Color(139,  69,  19, 255))
    gen.define_tile("water",   rl.Color( 30, 100, 200, 255))

    gen.settile("desert1", 0, 10)
    gen.settile("desert2", 0, 10)
    gen.settile("forest",  0,  0)
    gen.settile("water",   0,  0)

    gen.growto("forest",  100, 300,   0, 20)
    gen.growto("forest",  400, 900,  20,  0)
    gen.growto("desert1", 100, 300,  10,  1)
    gen.growto("desert2", 100, 300,  10,  1)
    gen.growto("desert2", 5800, 10000, 0, 100)
    gen.growto("water",   400,  600,   0, 30)
    gen.growto("water",   600, 5800,  30,  2)
    gen.growto("water",  5800, 10000, 30,  2)

    gen.longline("canyon",  4, 28,  50, 300)
    gen.longline("canyon",  2, 10, 500, 7000)

    gen.ensure_chunks_around(0)

    rl.init_window(SCREEN_WIDTH, SCREEN_HEIGHT, b"Top-Down Scroller Demo")
    rl.set_target_fps(60)

    scroll_offset = 0.0
    scroll_speed = SCROLL_SPEED_DEFAULT
    total_level_px = LEVEL_HEIGHT_TILES * TILE_SIZE

    while not rl.window_should_close():
        dt = rl.get_frame_time()

        if rl.is_key_down(rl.KEY_W):
            scroll_speed += SCROLL_SPEED_STEP * dt
        if rl.is_key_down(rl.KEY_S):
            scroll_speed -= SCROLL_SPEED_STEP * dt
            if scroll_speed < 0.0:
                scroll_speed = 0.0

        scroll_offset += scroll_speed * dt
        scroll_offset = math.fmod(scroll_offset, total_level_px)

        bottom_level_px = scroll_offset
        first_row = int(bottom_level_px // TILE_SIZE)
        current_chunk = (first_row % LEVEL_HEIGHT_TILES) // CHUNK_HEIGHT
        gen.ensure_chunks_around(current_chunk)

        rl.begin_drawing()
        rl.clear_background(rl.BLACK)

        pixel_offset_y = bottom_level_px % TILE_SIZE
        rows_on_screen = (SCREEN_HEIGHT // TILE_SIZE) + 2

        # --- tile pass ---
        for screen_row in range(rows_on_screen):
            level_row = first_row + screen_row
            draw_y = SCREEN_HEIGHT - (screen_row + 1) * TILE_SIZE + int(pixel_offset_y)
            for col in range(GRID_WIDTH):
                tile = gen.get_tile(level_row, col)
                draw_x = col * TILE_SIZE
                color = tile.color if tile else rl.DARKGRAY
                rl.draw_rectangle(draw_x, draw_y, TILE_SIZE, TILE_SIZE, color)
                rl.draw_rectangle_lines(draw_x, draw_y, TILE_SIZE, TILE_SIZE, rl.Color(0, 0, 0, 40))

        # --- prop pass ---
        for screen_row in range(rows_on_screen):
            level_row = first_row + screen_row
            draw_y = SCREEN_HEIGHT - (screen_row + 1) * TILE_SIZE + int(pixel_offset_y)
            for col in range(GRID_WIDTH):
                tile = gen.get_tile(level_row, col)
                draw_x = col * TILE_SIZE
                draw_prop(tile, draw_x, draw_y, col, level_row % LEVEL_HEIGHT_TILES)

        # --- ship ---
        ship_x = SCREEN_WIDTH // 2
        ship_y = SCREEN_HEIGHT - 120
        rl.draw_triangle(
            rl.Vector2(ship_x,      ship_y - 60),
            rl.Vector2(ship_x - 40, ship_y),
            rl.Vector2(ship_x + 40, ship_y),
            rl.RED
        )

        current_row = int(scroll_offset // TILE_SIZE) % LEVEL_HEIGHT_TILES
        cached = len(gen._chunks)
        rl.draw_text(f"Speed: {scroll_speed:.0f} px/s  [W/S]".encode(), 10, 10, FONT_SIZE, rl.WHITE)
        rl.draw_text(f"Row: {current_row}/{LEVEL_HEIGHT_TILES}  Chunks cached: {cached}".encode(), 10, 50, FONT_SIZE, rl.WHITE)

        rl.end_drawing()

    rl.close_window()

if __name__ == "__main__":
    main()
