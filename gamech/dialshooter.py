import pyray as rl
import random
import math
import time

# Window
W, H = 800, 600
rl.init_window(W, H, b"Dial Shmup")
rl.set_target_fps(60)

# Colors
WHITE = rl.WHITE
BLACK = rl.BLACK
BLUE = rl.BLUE
ORANGE = rl.Color(255, 140, 0, 255)
GREEN = rl.GREEN
YELLOW = rl.YELLOW
RED = rl.RED
GRAY = rl.GRAY
DARKGRAY = rl.Color(80, 80, 80, 255)
PURPLE = rl.PURPLE
SKYBLUE = rl.SKYBLUE

# --- Game State ---

player = {
    "x": 100, "y": H // 2,
    "w": 30, "h": 20,
    "speed": 4,
    "hp": 5,
}

bullets = []       # {x, y, vx, vy, color, w, h, owner}
enemies = []       # {x, y, w, h, hp, vx}
particles = []     # {x, y, vx, vy, life, max_life, color, w, h}

# Dial system
# Shot sequences produce letters: B->blue shot, O->orange shot
# Combos: BBO->L, BOBO->S, OOOOB->C
# Buffer of 5 letters, combos on buffer: SSC->spinning shield, LSC->?, etc.

shot_sequence = []   # tracks last 5 shots as 'B' or 'O'
letter_buffer = []   # up to 5 letters: L, S, C
active_effects = []  # {type, timer}

COMBO_SEQUENCES = {
    ("B","B","O"): "L",
    ("B","O","B","O"): "S",
    ("O","O","O","O","B"): "C",
}

LETTER_COMBOS = {
    ("S","S","C"): "spinning_shield",
    ("L","S","C"): "triple_laser",
    ("L","L","L"): "laser_storm",
    ("C","C","C"): "big_circle",
    ("S","S","S"): "scatter_burst",
}

spinning_shield_active = False
spinning_shield_timer = 0
spinning_shield_angle = 0

# Enemy spawning
spawn_timer = 0
spawn_interval = 2.0  # seconds, decreases over time
game_time = 0

score = 0

# Cooldowns
shoot_cooldown_b = 0
shoot_cooldown_o = 0
SHOOT_CD = 0.15

dt = 0

def fire_blue():
    bullets.append({
        "x": player["x"] + player["w"],
        "y": player["y"] + player["h"] // 2 - 3,
        "vx": 10, "vy": 0,
        "color": BLUE, "w": 12, "h": 6,
        "owner": "player", "dmg": 1
    })

def fire_orange():
    bullets.append({
        "x": player["x"] + player["w"],
        "y": player["y"] + player["h"] // 2 - 3,
        "vx": 9, "vy": 1,
        "color": ORANGE, "w": 10, "h": 8,
        "owner": "player", "dmg": 1
    })
    bullets.append({
        "x": player["x"] + player["w"],
        "y": player["y"] + player["h"] // 2 - 3,
        "vx": 9, "vy": -1,
        "color": ORANGE, "w": 10, "h": 8,
        "owner": "player", "dmg": 1
    })

def fire_green_laser():
    # Instant laser - wide fast projectile
    bullets.append({
        "x": player["x"] + player["w"],
        "y": player["y"] + player["h"] // 2 - 5,
        "vx": 25, "vy": 0,
        "color": GREEN, "w": 60, "h": 10,
        "owner": "player", "dmg": 3
    })

def fire_scatter():
    # 8 directions
    for i in range(8):
        angle = i * math.pi / 4
        bullets.append({
            "x": player["x"] + player["w"] // 2,
            "y": player["y"] + player["h"] // 2,
            "vx": math.cos(angle) * 8,
            "vy": math.sin(angle) * 8,
            "color": YELLOW, "w": 8, "h": 8,
            "owner": "player", "dmg": 1
        })

def fire_circle():
    # Large circle forward - big slow projectile
    bullets.append({
        "x": player["x"] + player["w"],
        "y": player["y"] + player["h"] // 2 - 20,
        "vx": 6, "vy": 0,
        "color": PURPLE, "w": 40, "h": 40,
        "owner": "player", "dmg": 5
    })

def check_shot_combo():
    global shot_sequence, letter_buffer
    # Check longest match first
    for length in [5, 4, 3]:
        if len(shot_sequence) >= length:
            tail = tuple(shot_sequence[-length:])
            if tail in COMBO_SEQUENCES:
                letter = COMBO_SEQUENCES[tail]
                shot_sequence = []
                add_letter(letter)
                trigger_shot_effect(letter)
                return
    # Keep buffer trimmed
    if len(shot_sequence) > 5:
        shot_sequence = shot_sequence[-5:]

def add_letter(letter):
    global letter_buffer
    letter_buffer.append(letter)
    if len(letter_buffer) > 5:
        letter_buffer = letter_buffer[-5:]
    check_letter_combo()

def check_letter_combo():
    global letter_buffer
    for length in [3]:
        if len(letter_buffer) >= length:
            tail = tuple(letter_buffer[-length:])
            if tail in LETTER_COMBOS:
                effect = LETTER_COMBOS[tail]
                letter_buffer = letter_buffer[:-length]
                activate_effect(effect)
                return

def trigger_shot_effect(letter):
    if letter == "L":
        fire_green_laser()
    elif letter == "S":
        fire_scatter()
    elif letter == "C":
        fire_circle()

def activate_effect(effect):
    global spinning_shield_active, spinning_shield_timer
    if effect == "spinning_shield":
        spinning_shield_active = True
        spinning_shield_timer = 30.0
    elif effect == "triple_laser":
        for dy in [-15, 0, 15]:
            bullets.append({
                "x": player["x"] + player["w"],
                "y": player["y"] + player["h"] // 2 + dy,
                "vx": 20, "vy": 0,
                "color": GREEN, "w": 50, "h": 8,
                "owner": "player", "dmg": 4
            })
    elif effect == "laser_storm":
        for _ in range(10):
            bullets.append({
                "x": player["x"] + player["w"],
                "y": player["y"] + player["h"] // 2,
                "vx": 15, "vy": random.uniform(-3, 3),
                "color": GREEN, "w": 30, "h": 6,
                "owner": "player", "dmg": 2
            })
    elif effect == "big_circle":
        bullets.append({
            "x": player["x"] + player["w"],
            "y": player["y"] + player["h"] // 2 - 35,
            "vx": 5, "vy": 0,
            "color": PURPLE, "w": 70, "h": 70,
            "owner": "player", "dmg": 10
        })
    elif effect == "scatter_burst":
        for i in range(16):
            angle = i * math.pi / 8
            bullets.append({
                "x": player["x"] + player["w"] // 2,
                "y": player["y"] + player["h"] // 2,
                "vx": math.cos(angle) * 10,
                "vy": math.sin(angle) * 10,
                "color": YELLOW, "w": 10, "h": 10,
                "owner": "player", "dmg": 2
            })

def spawn_enemy():
    y = random.randint(20, H - 60)
    hp = random.randint(1, 3)
    enemies.append({
        "x": W + 10, "y": y,
        "w": 28, "h": 20,
        "hp": hp, "max_hp": hp,
        "vx": random.uniform(-2.5, -1.5)
    })

def rect_overlap(ax, ay, aw, ah, bx, by, bw, bh):
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by

# Main loop
while not rl.window_should_close():
    dt = rl.get_frame_time()
    game_time += dt

    # Spawn rate: starts at 2s, decreases to min 0.3s over 120s
    spawn_interval = max(0.03, 2.0 - (game_time / 120.0) * 1.7)
    spawn_timer += dt
    if spawn_timer >= spawn_interval:
        spawn_timer = 0
        spawn_enemy()

    # Cooldowns
    shoot_cooldown_b = max(0, shoot_cooldown_b - dt)
    shoot_cooldown_o = max(0, shoot_cooldown_o - dt)

    # Player movement
    if rl.is_key_down(rl.KeyboardKey.KEY_W): player["y"] -= player["speed"]
    if rl.is_key_down(rl.KeyboardKey.KEY_S): player["y"] += player["speed"]
    if rl.is_key_down(rl.KeyboardKey.KEY_A): player["x"] -= player["speed"]
    if rl.is_key_down(rl.KeyboardKey.KEY_D): player["x"] += player["speed"]
    player["x"] = max(0, min(W - player["w"], player["x"]))
    player["y"] = max(0, min(H - player["h"], player["y"]))

    # Shooting
    if rl.is_key_pressed(rl.KeyboardKey.KEY_SPACE) and shoot_cooldown_b <= 0:
        fire_blue()
        shoot_cooldown_b = SHOOT_CD
        shot_sequence.append("B")
        check_shot_combo()

    if rl.is_key_pressed(rl.KeyboardKey.KEY_LEFT_SHIFT) and shoot_cooldown_o <= 0:
        fire_orange()
        shoot_cooldown_o = SHOOT_CD
        shot_sequence.append("O")
        check_shot_combo()

    # Spinning shield
    if spinning_shield_active:
        spinning_shield_timer -= dt
        spinning_shield_angle += 120 * dt  # degrees per second
        if spinning_shield_timer <= 0:
            spinning_shield_active = False
        # Shield orbs collision with enemies
        for orb_i in range(3):
            angle_rad = math.radians(spinning_shield_angle + orb_i * 120)
            ox = player["x"] + player["w"]//2 + math.cos(angle_rad) * 40 - 8
            oy = player["y"] + player["h"]//2 + math.sin(angle_rad) * 40 - 8
            for e in enemies[:]:
                if rect_overlap(ox, oy, 16, 16, e["x"], e["y"], e["w"], e["h"]):
                    e["hp"] -= 1
                    if e["hp"] <= 0:
                        enemies.remove(e)
                        score += 10

    # Update bullets
    for b in bullets[:]:
        b["x"] += b["vx"]
        b["y"] += b["vy"]
        if b["x"] > W + 100 or b["x"] < -100 or b["y"] < -100 or b["y"] > H + 100:
            bullets.remove(b)
            continue
        if b["owner"] == "player":
            for e in enemies[:]:
                if rect_overlap(b["x"], b["y"], b["w"], b["h"], e["x"], e["y"], e["w"], e["h"]):
                    e["hp"] -= b.get("dmg", 1)
                    if b in bullets:
                        bullets.remove(b)
                    if e["hp"] <= 0 and e in enemies:
                        enemies.remove(e)
                        score += 10
                    break
        elif b["owner"] == "enemy":
            if rect_overlap(b["x"], b["y"], b["w"], b["h"],
                            player["x"], player["y"], player["w"], player["h"]):
                player["hp"] -= 1
                if b in bullets:
                    bullets.remove(b)

    # Update enemies
    for e in enemies[:]:
        e["x"] += e["vx"]
        if e["x"] < -50:
            enemies.remove(e)
            continue
        # Enemy collision with player
        if rect_overlap(e["x"], e["y"], e["w"], e["h"],
                        player["x"], player["y"], player["w"], player["h"]):
            player["hp"] -= 1
            enemies.remove(e)

    # --- DRAW ---
    rl.begin_drawing()
    rl.clear_background(rl.Color(10, 10, 30, 255))

    # Enemies
    for e in enemies:
        rl.draw_rectangle(int(e["x"]), int(e["y"]), e["w"], e["h"], RED)
        # HP bar
        bar_w = int((e["hp"] / e["max_hp"]) * e["w"])
        rl.draw_rectangle(int(e["x"]), int(e["y"]) - 5, bar_w, 3, GREEN)

    # Bullets
    for b in bullets:
        rl.draw_rectangle(int(b["x"]), int(b["y"]), b["w"], b["h"], b["color"])

    # Spinning shield orbs
    if spinning_shield_active:
        for orb_i in range(3):
            angle_rad = math.radians(spinning_shield_angle + orb_i * 120)
            ox = int(player["x"] + player["w"]//2 + math.cos(angle_rad) * 40 - 8)
            oy = int(player["y"] + player["h"]//2 + math.sin(angle_rad) * 40 - 8)
            rl.draw_rectangle(ox, oy, 16, 16, SKYBLUE)

    # Player
    rl.draw_rectangle(int(player["x"]), int(player["y"]), player["w"], player["h"], WHITE)

    # HUD
    rl.draw_text(b"HP: " + str(player["hp"]).encode(), 10, 10, 20, GREEN)
    rl.draw_text(b"Score: " + str(score).encode(), 10, 35, 20, WHITE)
    rl.draw_text(b"Time: " + f"{game_time:.1f}".encode(), 10, 60, 20, GRAY)

    # Shot sequence display
    seq_str = "Seq: " + "".join(shot_sequence)
    rl.draw_text(seq_str.encode(), 10, H - 60, 18, YELLOW)

    # Letter buffer
    buf_str = "Buf: " + " ".join(letter_buffer)
    rl.draw_text(buf_str.encode(), 10, H - 35, 18, SKYBLUE)

    # Shield timer
    if spinning_shield_active:
        shield_str = f"SHIELD: {spinning_shield_timer:.1f}s"
        rl.draw_text(shield_str.encode(), W - 180, 10, 20, SKYBLUE)

    # Controls hint
    hint = b"SPACE=Blue  SHIFT=Orange | BBO=Laser BOBO=Scatter OOOOB=Circle"
    rl.draw_text(hint, 10, H - 80, 14, DARKGRAY)

    # Spawn rate indicator
    sr_str = f"Spawn: {spawn_interval:.2f}s"
    rl.draw_text(sr_str.encode(), W - 160, H - 30, 16, GRAY)

    if player["hp"] <= 0:
        rl.draw_text(b"GAME OVER", W//2 - 80, H//2 - 20, 40, RED)
        rl.draw_text(b"Close window to exit", W//2 - 100, H//2 + 30, 20, WHITE)

    rl.end_drawing()

rl.close_window()
