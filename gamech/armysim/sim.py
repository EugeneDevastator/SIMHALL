# sim.py
import random
import math
from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class UnitTemplate:
    hp: float
    attack: float
    defense: float
    regen: float
    attack_speed: int
    regen_speed: int
    count: int
    hit_chance: float
    splash_damage: float
    splash_count: int
    unit_size: int

@dataclass
class Unit:
    max_hp: float
    hp: float
    attack: float
    defense: float
    regen: float
    attack_speed: int
    regen_speed: int
    hit_chance: float
    splash_damage: float
    splash_count: int
    unit_size: int

    def is_alive(self) -> bool:
        return self.hp > 0.0

class Army:
    def __init__(self, tmpl: UnitTemplate):
        self.tmpl = tmpl
        self.units: List[float] = []  # hp values only for display
        self._units: List[Unit] = []

    def build(self):
        self.units = []
        self._units = []
        t = self.tmpl
        for _ in range(t.count):
            u = Unit(
                max_hp=t.hp, hp=t.hp,
                attack=t.attack, defense=t.defense,
                regen=t.regen, attack_speed=t.attack_speed,
                regen_speed=t.regen_speed, hit_chance=t.hit_chance,
                splash_damage=t.splash_damage, splash_count=t.splash_count,
                unit_size=t.unit_size
            )
            self._units.append(u)
            self.units.append(u.hp)

    def sync_display(self):
        self.units = [u.hp for u in self._units]

    def alive_units(self) -> List[Unit]:
        return [u for u in self._units if u.is_alive()]

    def alive_count(self) -> int:
        return sum(1 for u in self._units if u.is_alive())

    def alive_indices(self) -> List[int]:
        return [i for i, u in enumerate(self._units) if u.is_alive()]


def compute_interaction_slots(army: Army, area: int) -> List[int]:
    """
    Returns indices of units filling the interaction area.
    Each unit occupies unit_size slots.
    """
    alive = army.alive_indices()
    slots: List[int] = []
    used = 0
    for idx in alive:
        u = army._units[idx]
        if used + u.unit_size > area:
            break
        slots.append(idx)
        used += u.unit_size
    return slots


def perform_attack(attacker: Unit, targets_active: List[Unit],
                   targets_pool: List[Unit], log: List[str],
                   attacker_label: str) -> None:
    """
    Attacker hits primary target (first active).
    Splash hits splash_count additional units: active first, then pool.
    """
    if not targets_active:
        return

    primary = targets_active[0]
    if random.random() <= attacker.hit_chance:
        dmg = max(0.0, attacker.attack - primary.defense)
        primary.hp -= dmg
        if primary.hp < 0:
            primary.hp = 0.0
        log.append(f"{attacker_label} hits for {dmg:.1f} (target hp={primary.hp:.1f})")
    else:
        log.append(f"{attacker_label} misses")

    if attacker.splash_count > 0 and attacker.splash_damage > 0:
        splash_targets: List[Unit] = []
        for u in targets_active[1:]:
            if len(splash_targets) >= attacker.splash_count:
                break
            splash_targets.append(u)
        for u in targets_pool:
            if len(splash_targets) >= attacker.splash_count:
                break
            splash_targets.append(u)
        for u in splash_targets:
            sdmg = max(0.0, attacker.splash_damage - u.defense)
            u.hp -= sdmg
            if u.hp < 0:
                u.hp = 0.0
        if splash_targets:
            log.append(f"  splash {attacker.splash_damage:.1f} to {len(splash_targets)} units")


def perform_regen(unit: Unit, tick: int) -> None:
    if tick % unit.regen_speed == 0 and unit.is_alive():
        unit.hp = min(unit.max_hp, unit.hp + unit.regen)


class SimState:
    def __init__(self, left: Army, right: Army, interaction_area: int):
        self.left = left
        self.right = right
        self.interaction_area = interaction_area
        self.tick = 0
        self.running = True
        self.winner: Optional[str] = None
        self.log: List[str] = []

    def reset(self):
        self.left.build()
        self.right.build()
        self.tick = 0
        self.running = True
        self.winner = None
        self.log = []

    def next_attack_tick(self) -> int:
        """Returns how many ticks until the next attack event."""
        la = self.left.alive_units()
        ra = self.right.alive_units()
        if not la or not ra:
            return 1
        l_spd = la[0].attack_speed
        r_spd = ra[0].attack_speed
        t = self.tick
        l_next = l_spd - (t % l_spd) if t % l_spd != 0 else l_spd
        r_next = r_spd - (t % r_spd) if t % r_spd != 0 else r_spd
        return min(l_next, r_next)

    def step(self):
        if not self.running:
            return

        t = self.tick
        la = self.left.alive_units()
        ra = self.right.alive_units()

        if not la or not ra:
            self._check_winner()
            return

        # interaction slots
        l_active_idx = compute_interaction_slots(self.left,  self.interaction_area)
        r_active_idx = compute_interaction_slots(self.right, self.interaction_area)

        l_active = [self.left._units[i]  for i in l_active_idx]
        r_active = [self.right._units[i] for i in r_active_idx]

        l_alive_idx = self.left.alive_indices()
        r_alive_idx = self.right.alive_indices()

        l_pool = [self.left._units[i]  for i in l_alive_idx if i not in l_active_idx]
        r_pool = [self.right._units[i] for i in r_alive_idx if i not in r_active_idx]

        # attacks
        if l_active and r_active:
            l_spd = l_active[0].attack_speed
            if t % l_spd == 0:
                # each left active unit attacks
                for atk in l_active:
                    if atk.is_alive():
                        perform_attack(atk, r_active, r_pool, self.log, "LEFT")

        if r_active and l_active:
            r_spd = r_active[0].attack_speed
            if t % r_spd == 0:
                for atk in r_active:
                    if atk.is_alive():
                        perform_attack(atk, l_active, l_pool, self.log, "RIGHT")

        # regen all alive
        for u in self.left._units:
            if u.is_alive():
                perform_regen(u, t)
        for u in self.right._units:
            if u.is_alive():
                perform_regen(u, t)

        self.left.sync_display()
        self.right.sync_display()

        self.tick += 1
        self._check_winner()

        # keep log bounded
        if len(self.log) > 120:
            self.log = self.log[-120:]

    def _check_winner(self):
        la = self.left.alive_count()
        ra = self.right.alive_count()
        if la == 0 and ra == 0:
            self.winner = "DRAW"
            self.running = False
        elif la == 0:
            self.winner = "RIGHT"
            self.running = False
        elif ra == 0:
            self.winner = "LEFT"
            self.running = False
