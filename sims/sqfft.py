import pyray as rl
import numpy as np
from PIL import Image
import sys, os

def fwht_1d(a):
    x = a.copy()
    h = 1
    n = x.shape[0]
    while h < n:
        x = x.reshape(-1, h * 2)
        top = x[:, :h].copy()
        bot = x[:, h:].copy()
        x[:, :h] = top + bot
        x[:, h:] = top - bot
        x = x.reshape(-1)
        h *= 2
    return x

def wht_2d(img):
    result = img.astype(np.float64)
    for r in range(result.shape[0]):
        result[r] = fwht_1d(result[r])
    for c in range(result.shape[1]):
        result[:, c] = fwht_1d(result[:, c])
    return result

def iwht_2d(w):
    return wht_2d(w) / (w.shape[0] * w.shape[1])

def next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p

def make_seq_map(ph, pw):
    def seq1d(n):
        idx = np.arange(n, dtype=np.uint32)
        gray = idx ^ (idx >> 1)
        s = np.zeros(n, dtype=np.float32)
        g = gray.copy()
        while np.any(g > 0):
            s += (g & 1).astype(np.float32)
            g >>= 1
        return s
    r = seq1d(ph)
    c = seq1d(pw)
    sm = r[:, None] + c[None, :]
    mx = sm.max()
    if mx > 0:
        sm = sm / mx
    return sm

NUM_BANDS = 8
BAND_EDGES = np.linspace(0.0, 1.0, NUM_BANDS + 1)

def encode_channel(ch, ph, pw):
    padded = np.zeros((ph, pw), dtype=np.float64)
    padded[:ch.shape[0], :ch.shape[1]] = ch.astype(np.float64)
    return wht_2d(padded)

def encode_rgb(rgb, ph, pw):
    return [encode_channel(rgb[:, :, c], ph, pw) for c in range(3)]

def blend_and_decode_channel(wht1, wht2, orig_shape, seq_map, use_img2):
    blended = np.zeros_like(wht1)
    for i in range(NUM_BANDS):
        lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
        if i == NUM_BANDS - 1:
            mask = ((seq_map >= lo) & (seq_map <= hi)).astype(np.float64)
        else:
            mask = ((seq_map >= lo) & (seq_map < hi)).astype(np.float64)
        src = wht2 if use_img2[i] else wht1
        blended += src * mask
    rec = iwht_2d(blended)
    h, w = orig_shape
    crop = rec[:h, :w]
    cmin, cmax = crop.min(), crop.max()
    if cmax > cmin:
        crop = (crop - cmin) / (cmax - cmin) * 255.0
    else:
        crop = np.zeros_like(crop)
    return crop.astype(np.uint8)

def blend_and_decode_rgb(wht1_rgb, wht2_rgb, orig_shape, seq_map, use_img2):
    channels = [blend_and_decode_channel(wht1_rgb[c], wht2_rgb[c], orig_shape, seq_map, use_img2)
                for c in range(3)]
    h, w = orig_shape
    out = np.stack(channels, axis=-1)
    return out

def numpy_rgb_to_texture(arr):
    h, w = arr.shape[:2]
    alpha = np.full((h, w, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([arr, alpha], axis=-1)
    flat = rgba.flatten().astype(np.uint8)
    buf = rl.ffi.from_buffer(flat)
    img = rl.Image()
    img.data = rl.ffi.cast("void *", buf)
    img.width = w
    img.height = h
    img.mipmaps = 1
    img.format = rl.PixelFormat.PIXELFORMAT_UNCOMPRESSED_R8G8B8A8
    return rl.load_texture_from_image(img)

def numpy_gray_to_texture(arr):
    return numpy_rgb_to_texture(np.stack([arr, arr, arr], axis=-1))

def load_rgb(path, max_dim=None):
    pil = Image.open(path).convert("RGB")
    if max_dim is not None:
        iw, ih = pil.size
        if iw > max_dim or ih > max_dim:
            scale = max_dim / max(iw, ih)
            pil = pil.resize((int(iw * scale), int(ih * scale)), Image.LANCZOS)
    return np.array(pil)

def upsample_nearest(rgb, target_h, target_w):
    pil = Image.fromarray(rgb)
    pil = pil.resize((target_w, target_h), Image.NEAREST)
    return np.array(pil)

def draw_checkbox(font, x, y, size, label, checked):
    rl.draw_rectangle(x, y, size, size, rl.LIGHTGRAY)
    rl.draw_rectangle_lines(x, y, size, size, rl.DARKGRAY)
    if checked:
        rl.draw_rectangle(x + 6, y + 6, size - 12, size - 12, rl.DARKBLUE)
    rl.draw_text_ex(font, label, rl.Vector2(x + size + 6, y + 4), 24, 1, rl.BLACK)

def checkbox_clicked(x, y, size):
    mx, my = rl.get_mouse_x(), rl.get_mouse_y()
    return (x <= mx <= x + size and y <= my <= y + size and
            rl.is_mouse_button_pressed(rl.MouseButton.MOUSE_BUTTON_LEFT))

def main():
    if not os.path.exists("image1.png") or not os.path.exists("image2.png"):
        print("Need image1.png and image2.png")
        sys.exit(1)

    rgb1 = load_rgb("image1.png", max_dim=512)
    rgb2_small = load_rgb("image2.png", max_dim=128)

    print(f"Image1: {rgb1.shape}  Image2: {rgb2_small.shape}")

    rgb2_up = upsample_nearest(rgb2_small, rgb1.shape[0], rgb1.shape[1])

    ph = next_pow2(rgb1.shape[0])
    pw = next_pow2(rgb1.shape[1])
    orig_shape = (rgb1.shape[0], rgb1.shape[1])

    print(f"WHT space: {ph}x{pw}")
    print("Encoding RGB channels...")
    wht1_rgb = encode_rgb(rgb1, ph, pw)
    wht2_rgb = encode_rgb(rgb2_up, ph, pw)
    print("Building seq map...")
    seq_map = make_seq_map(ph, pw)
    print("Done.")

    meaningful_seq = rgb2_small.shape[0] / rgb1.shape[0]

    use_img2 = [False] * NUM_BANDS

    SCREEN_W, SCREEN_H = 1920, 1080
    rl.init_window(SCREEN_W, SCREEN_H, "WHT RGB Blender")
    rl.set_target_fps(60)

    font = rl.load_font_ex("C:/Windows/Fonts/arialbd.ttf", 32, rl.ffi.NULL, 0)

    IMG_SIZE = 380
    GAP = 40
    IMG_Y = 80

    orig1_tex = numpy_rgb_to_texture(rgb1)
    orig2_small_tex = numpy_rgb_to_texture(rgb2_small)
    orig2_up_tex = numpy_rgb_to_texture(rgb2_up)

    decoded = blend_and_decode_rgb(wht1_rgb, wht2_rgb, orig_shape, seq_map, use_img2)
    decoded_tex = numpy_rgb_to_texture(decoded)
    dirty = False

    total_w = 4 * IMG_SIZE + 3 * GAP
    sx = (SCREEN_W - total_w) // 2
    x1 = sx
    x2 = sx + IMG_SIZE + GAP
    x3 = sx + 2 * (IMG_SIZE + GAP)
    x4 = sx + 3 * (IMG_SIZE + GAP)

    CB_SIZE = 32
    CB_Y = IMG_Y + IMG_SIZE + 60
    CB_SPACING = (2 * IMG_SIZE + GAP) // NUM_BANDS

    while not rl.window_should_close():
        for i in range(NUM_BANDS):
            cx = x1 + i * CB_SPACING
            if checkbox_clicked(cx, CB_Y, CB_SIZE):
                use_img2[i] = not use_img2[i]
                dirty = True

        if dirty:
            decoded = blend_and_decode_rgb(wht1_rgb, wht2_rgb, orig_shape, seq_map, use_img2)
            rl.unload_texture(decoded_tex)
            decoded_tex = numpy_rgb_to_texture(decoded)
            dirty = False

        rl.begin_drawing()
        rl.clear_background(rl.RAYWHITE)

        def draw_img(tex, px, label, disp=IMG_SIZE):
            src = rl.Rectangle(0, 0, tex.width, tex.height)
            dst = rl.Rectangle(px, IMG_Y, disp, disp)
            rl.draw_texture_pro(tex, src, dst, rl.Vector2(0, 0), 0, rl.WHITE)
            rl.draw_text_ex(font, label, rl.Vector2(px, IMG_Y - 44), 28, 1, rl.BLACK)

        draw_img(orig1_tex, x1, f"Img1  {rgb1.shape[1]}x{rgb1.shape[0]}")
        s2d = max(int(IMG_SIZE * rgb2_small.shape[0] / rgb1.shape[0]), 64)
        draw_img(orig2_small_tex, x2, f"Img2  {rgb2_small.shape[1]}x{rgb2_small.shape[0]}", s2d)
        draw_img(orig2_up_tex, x3, "Img2 upsampled (nearest)")
        draw_img(decoded_tex, x4, "Result (blended)")

        rl.draw_text_ex(font, "Bands  unchecked=Img1  checked=Img2  (all 3 channels)",
                        rl.Vector2(x1, CB_Y - 40), 26, 1, rl.DARKGRAY)

        for i in range(NUM_BANDS):
            cx = x1 + i * CB_SPACING
            lo, hi = BAND_EDGES[i], BAND_EDGES[i + 1]
            draw_checkbox(font, cx, CB_Y, CB_SIZE, f"B{i+1}", use_img2[i])
            rl.draw_text_ex(font, f"{lo:.2f}", rl.Vector2(cx, CB_Y + CB_SIZE + 6), 20, 1, rl.DARKGRAY)
            if hi <= meaningful_seq + 1.0 / NUM_BANDS:
                rl.draw_rectangle(cx, CB_Y - 6, CB_SPACING - 4, 4, rl.GREEN)

        rl.draw_text_ex(font,
                        f"Green = img2 meaningful bands (~{meaningful_seq*100:.0f}% of freq range)",
                        rl.Vector2(x1, CB_Y + CB_SIZE + 36), 24, 1, rl.DARKGREEN)

        rl.end_drawing()

    rl.unload_texture(orig1_tex)
    rl.unload_texture(orig2_small_tex)
    rl.unload_texture(orig2_up_tex)
    rl.unload_texture(decoded_tex)
    rl.unload_font(font)
    rl.close_window()

if __name__ == "__main__":
    main()
