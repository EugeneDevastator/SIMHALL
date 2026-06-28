"""
4-dimensional 24-cell structure explorer.
letters are cells (3d octahedral faces of the structure)
"""
import tkinter as tk
from tkinter import ttk
from itertools import combinations
from collections import deque

def dist2(a, b):
    return sum((x-y)**2 for x,y in zip(a,b))

vertices = set()
for signs in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    for positions in combinations(range(4), 2):
        v = [0,0,0,0]
        v[positions[0]] = signs[0]
        v[positions[1]] = signs[1]
        vertices.add(tuple(v))
vertices = list(vertices)

oct_sets = set()
for combo in combinations(range(len(vertices)), 6):
    pairs = list(combinations(combo, 2))
    dists = [dist2(vertices[a], vertices[b]) for a,b in pairs]
    if dists.count(2) == 12 and dists.count(4) == 3:
        oct_sets.add(frozenset(combo))
octahedra = list(oct_sets)
n = len(octahedra)

cell_adj = {i: set() for i in range(n)}
for i,j in combinations(range(n), 2):
    if len(octahedra[i] & octahedra[j]) == 3:
        cell_adj[i].add(j)
        cell_adj[j].add(i)

dist_bfs = {0: 0}
queue = deque([0])
while queue:
    cur = queue.popleft()
    for nb in cell_adj[cur]:
        if nb not in dist_bfs:
            dist_bfs[nb] = dist_bfs[cur] + 1
            queue.append(nb)

letters = list('ABCDEFGHIJKLMNOPQRSTUVWX')

def letter_to_idx(l):
    return letters.index(l)

def make_mask(center_idx, highlight_set):
    chars = []
    for i in range(24):
        if i == center_idx:
            chars.append('@')
        elif i in highlight_set:
            chars.append(letters[i])
        else:
            chars.append('.')
    return ''.join(chars)

def build_lines(center_idx, other_idxs):
    neighbors = set(cell_adj[center_idx])
    lines = []

    ruler = 'ABCDEFGHIJKLMNOPQRSTUVWX'
    desc = f"@ = {letters[center_idx]}  depth={dist_bfs[center_idx]}  nbrs={len(neighbors)}"
    lines.append((ruler, desc, 'ruler'))

    mask = make_mask(center_idx, neighbors)
    lines.append((mask, f"neighbors of {letters[center_idx]}", 'main'))

    for ov in other_idxs:
        o_neighbors = cell_adj[ov]
        shared = neighbors & o_neighbors
        star = '★' if ov in neighbors else ' '
        mask = make_mask(center_idx, shared)
        lines.append((mask, f"shared with {letters[ov]}{star} ({len(shared)})", 'shared'))

    return lines

root = tk.Tk()
root.title("24-cell Explorer")
root.configure(bg='#f0f0f0')

FONT = ('Courier', 18, 'bold')
FONT_MONO = ('Courier', 18, 'bold')

NUM_VIEWS = 6
default_starts = [0, 6, 12, 1, 7, 13]
selected = [tk.StringVar(value=letters[default_starts[i]]) for i in range(NUM_VIEWS)]

main_frame = tk.Frame(root, bg='#f0f0f0')
main_frame.pack(fill='both', expand=True, padx=12, pady=12)

tk.Label(main_frame,
         text="24-cell Explorer   @ = selected   . = absent   ★ = adjacent to this view",
         font=FONT, fg='#222222', bg='#f0f0f0').pack(pady=(0, 8))

grid_frame = tk.Frame(main_frame, bg='#f0f0f0')
grid_frame.pack(fill='both', expand=True)

text_areas = []
dropdowns = []

TEXT_WIDTH = 58
TEXT_HEIGHT = 8

def update_views(*_):
    idxs = [letter_to_idx(selected[v].get()) for v in range(NUM_VIEWS)]
    for v in range(NUM_VIEWS):
        center = idxs[v]
        others = [idxs[ov] for ov in range(NUM_VIEWS) if ov != v]
        lines = build_lines(center, others)

        txt = text_areas[v]
        txt.config(state='normal')
        txt.delete('1.0', 'end')

        for mask, desc, tag in lines:
            combined = mask + '  ' + desc + '\n'
            txt.insert('end', combined, tag)

        txt.config(state='disabled')

style = ttk.Style()
style.theme_use('clam')
style.configure('TCombobox', font=('Courier', 18, 'bold'))

for v in range(NUM_VIEWS):
    row, col = divmod(v, 2)

    outer = tk.Frame(grid_frame, bg='#ffffff', relief='ridge', bd=2)
    outer.grid(row=row, column=col, padx=6, pady=6, sticky='nsew')
    grid_frame.rowconfigure(row, weight=1)
    grid_frame.columnconfigure(col, weight=1)

    top = tk.Frame(outer, bg='#ffffff')
    top.pack(fill='x', padx=8, pady=(6, 2))

    tk.Label(top, text=f"View {v+1}", font=FONT,
             fg='#333399', bg='#ffffff').pack(side='left')

    dd = ttk.Combobox(top, textvariable=selected[v], values=letters, width=4, font=FONT)
    dd.pack(side='left', padx=(10, 0))
    dd.bind('<<ComboboxSelected>>', update_views)
    dropdowns.append(dd)

    txt = tk.Text(outer, font=FONT_MONO, bg='#fafafa', fg='#222222',
                  relief='flat', state='disabled', wrap='none',
                  width=TEXT_WIDTH, height=TEXT_HEIGHT)
    txt.pack(fill='both', expand=True, padx=4, pady=(2, 6))

    txt.tag_configure('ruler',  foreground='#aaaaaa')
    txt.tag_configure('main',   foreground='#0055cc')
    txt.tag_configure('shared', foreground='#cc6600')

    text_areas.append(txt)

update_views()
root.mainloop()
