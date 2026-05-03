import pyray as rl
import numpy as np
from PIL import Image
import sys
import os

# --- Fast WHT using numpy (Cooley-Tukey style, vectorized) ---

def fwht(a):
    """Fast Walsh-Hadamard Transform, in-place, unnormalized."""
    x = a.copy()
    h = 1
    while h < x.shape[0]:
        x = x.reshape(-1, h * 2)
        top = x[:, :h].copy()
        bot = x[:, h:].copy()
        x[:, :h] = top + bot
        x[:, h:] = top - bot
        x = x.reshape(-1)
        h *= 2
    return x

def wht_2d(img):
    h, w = img.shape
    result = img.astype(np.float64)
    # rows
    for r in range(h):
        result[r] = fwht(result[r])
    # cols
    for c in range(w):
        result[:, c] = fwht(result[:, c])
    return result

def iwht_2d(w_data):
    # WHT is self-inverse up to scale: IWHT(x) = WHT(x) / N
    ph, pw = w_data.shape
    result = wht_2d(w_data)
    return result / (ph * pw)

def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p

def encode_image(img_gray):
    h, w = img_gray.shape
    ph = next_pow2(h)
    pw = next_pow2(w)
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:h, :w] = img_gray.astype(np.float64)
    wht_data = wht_2d(padded)
    return wht_data, (h, w)

def make_sequency_map(ph, pw):
    def seq1d(n):
        idx = np.arange(n, dtype=np.uint32)
        gray = idx ^ (idx >> 1)
        s = np.zeros(n, dtype=np.float32)
        g = gray.copy()
        while np.any(g > 0):
            s += (g & 1).astype(np.float32)
            g >>= 1
        return s
    row_seq = seq1d(ph)
    col_seq = seq1d(pw)
    seq_map = row_seq[:, None] + col_seq[None, :]
    seq_map = seq_map / seq_map.max()
    return seq_map

def apply_filter_and_decode(wht_data, orig_shape, seq_map, min_seq, max_seq):
    mn = min(min_seq, max_seq)
    mx = max(min_seq, max_seq)
    mask = ((seq_map >= mn) & (seq_map <= mx)).astype(np.float64)
    filtered = wht_data * mask
    reconstructed = iwht_2d(filtered)
    h, w = orig_shape
    crop = reconstructed[:h, :w]
    # normalize to 0-255
    cmin, cmax = crop.min(), crop.max()
    if cmax > cmin:
        crop = (crop - cmin) / (cmax - cmin) * 255.0
    else:
        crop = np.zeros_like(crop)
    return filtered, crop.astype(np.uint8)

def wht_to_visual(wht_data):
    vis = np.abs(wht_data)
    vis = np.log1p(vis)
    vmax = vis.max()
    if vmax > 0:
        vis = vis / vmax
    return (vis * 255).astype(np.uint8)

def numpy_gray_to_texture(arr):
    h, w = arr.shape
    rgba = np.stack([arr, arr, arr, np.full((h, w), 255, dtype=np.uint8)], axis=-1)
    flat = rgba.flatten().astype(np.uint8)
    img = rl.Image()
    img.data = rl.ffi.cast("void *", rl.ffi.from_buffer(flat))
    img.width = w
    img.height = h
    img.mipmaps = 1
    img.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    return rl.load_texture_from_image(img)

class Slider:
    def __init__(self, x, y, w, h, label, val, lo, hi):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.label = label
        self.val = val
        self.lo, self.hi = lo, hi
        self.dragging = False

    def update(self):
        mx = rl.get_mouse_x()
        my = rl.get_mouse_y()
        if rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT):
            if (self.x <= mx <= self.x + self.w and
                    self.y <= my <= self.y + self.h):
                self.dragging = True
        if rl.is_mouse_button_released(rl.MouseButton.MOUSE_BUTTON_LEFT):
            self.dragging = False
        if self.dragging:
            t = (mx - self.x) / self.w
            self.val = self.lo + max(0.0, min(1.0, t)) * (self.hi - self.lo)

    def draw(self, font):
        rl.draw_rectangle(self.x, self.y + self.h // 2 - 4, self.w, 8, rl.GRAY)
        t = (self.val - self.lo) / (self.hi - self.lo)
        tx = int(self.x + t * self.w)
        rl.draw_circle(tx, self.y + self.h // 2, 14, rl.DARKBLUE)
        txt = f"{self.label}: {self.val:.3f}"
        rl.draw_text_ex(font, txt, rl.Vector2(self.x, self.y - 38), 32, 1, rl.BLACK)

def main():
    img_path = "image.png"
    if not os.path.exists(img_path):
        print("No image.png found.")
        sys.exit(1)

    pil_img = Image.open(img_path).convert("L")
    MAX_DIM = 256
    iw, ih = pil_img.size
    if iw > MAX_DIM or ih > MAX_DIM:
        scale = MAX_DIM / max(iw, ih)
        pil_img = pil_img.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
    img_gray = np.array(pil_img)
    print(f"Image shape: {img_gray.shape}, min={img_gray.min()}, max={img_gray.max()}")

    print("Encoding WHT...")
    wht_data, orig_shape = encode_image(img_gray)
    ph, pw = wht_data.shape
    print(f"WHT shape: ({ph},{pw}), min={wht_data.min():.1f}, max={wht_data.max():.1f}")

    print("Building sequency map...")
    seq_map = make_sequency_map(ph, pw)
    print("Done.")

    SCREEN_W, SCREEN_H = 1920, 1080
    rl.init_window(SCREEN_W, SCREEN_H, "WHT Square Wave Codec")
    rl.set_target_fps(60)

    font = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", 32, rl.ffi.NULL, 0)

    s_min = Slider(100, SCREEN_H - 160, 1700, 50, "Min Seq", 0.0, 0.0, 1.0)
    s_max = Slider(100, SCREEN_H -  80, 1700, 50, "Max Seq", 1.0, 0.0, 1.0)

    orig_tex = numpy_gray_to_texture(img_gray)

    filtered_wht, decoded_gray = apply_filter_and_decode(
        wht_data, orig_shape, seq_map, 0.0, 1.0)
    wht_tex     = numpy_gray_to_texture(wht_to_visual(filtered_wht))
    decoded_tex = numpy_gray_to_texture(decoded_gray)

    prev_min, prev_max = -1.0, -1.0

    IMG_DISPLAY = 512
    GAP = 60
    total_w = 3 * IMG_DISPLAY + 2 * GAP
    start_x = (SCREEN_W - total_w) // 2
    IMG_Y = 80
    labels = ["Original", "WHT (filtered)", "Decoded"]
    xs = [start_x + i * (IMG_DISPLAY + GAP) for i in range(3)]

    while not rl.window_should_close():
        s_min.update()
        s_max.update()

        if (abs(s_min.val - prev_min) > 0.001 or
                abs(s_max.val - prev_max) > 0.001):
            filtered_wht, decoded_gray = apply_filter_and_decode(
                wht_data, orig_shape, seq_map, s_min.val, s_max.val)
            rl.unload_texture(wht_tex)
            rl.unload_texture(decoded_tex)
            wht_tex     = numpy_gray_to_texture(wht_to_visual(filtered_wht))
            decoded_tex = numpy_gray_to_texture(decoded_gray)
            prev_min, prev_max = s_min.val, s_max.val

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        for i, (tex, lbl, px) in enumerate(zip(
                [orig_tex, wht_tex, decoded_tex], labels, xs)):
            src = rl.Rectangle(0, 0, tex.width, tex.height)
            dst = rl.Rectangle(px, IMG_Y, IMG_DISPLAY, IMG_DISPLAY)
            rl.draw_texture_pro(tex, src, dst, rl.Vector2(0, 0), 0, rl.WHITE)
            rl.draw_text_ex(font, lbl, rl.Vector2(px, IMG_Y - 42), 32, 1, rl.BLACK)

        s_min.draw(font)
        s_max.draw(font)

        rl.end_drawing()

    rl.unload_texture(orig_tex)
    rl.unload_texture(wht_tex)
    rl.unload_texture(decoded_tex)
    rl.unload_font(font)
    rl.close_window()

if __name__ == "__main__":
    main()
