// y[r] = sum_c W[r][c] * x[c]   W: int8 (rows x cols, row-major), x: int16 (|x| <= 2047), y: int32. cols % 16 == 0.
// Freestanding WebAssembly SIMD: 16 weights per step: widen to 2 x i16x8, i32x4.dot_i16x8_s with x, add.
#include <wasm_simd128.h>
#include <stdint.h>
__attribute__((export_name("gemv")))
void gemv(const int8_t *W, const int16_t *x, int32_t *y, int rows, int cols) {
  for (int r = 0; r + 1 < rows; r += 2) {          // two rows at a time: x is loaded once for both
    const int8_t *a = W + (long)r * cols, *b = a + cols;
    v128_t s0 = wasm_i32x4_splat(0), s1 = s0, t0 = s0, t1 = s0;
    for (int c = 0; c < cols; c += 16) {
      v128_t xl = wasm_v128_load(x + c), xh = wasm_v128_load(x + c + 8);
      v128_t wa = wasm_v128_load(a + c), wb = wasm_v128_load(b + c);
      s0 = wasm_i32x4_add(s0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(wa), xl));
      s1 = wasm_i32x4_add(s1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(wa), xh));
      t0 = wasm_i32x4_add(t0, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_low_i8x16(wb), xl));
      t1 = wasm_i32x4_add(t1, wasm_i32x4_dot_i16x8(wasm_i16x8_extend_high_i8x16(wb), xh));
    }
    s0 = wasm_i32x4_add(s0, s1); t0 = wasm_i32x4_add(t0, t1);
    y[r] = wasm_i32x4_extract_lane(s0, 0) + wasm_i32x4_extract_lane(s0, 1) + wasm_i32x4_extract_lane(s0, 2) + wasm_i32x4_extract_lane(s0, 3);
    y[r + 1] = wasm_i32x4_extract_lane(t0, 0) + wasm_i32x4_extract_lane(t0, 1) + wasm_i32x4_extract_lane(t0, 2) + wasm_i32x4_extract_lane(t0, 3);
  }
  if (rows & 1) { int r = rows - 1; const int8_t *a = W + (long)r * cols; v128_t s0 = wasm_i32x4_splat(0);
    for (int c = 0; c < cols; c += 8) s0 = wasm_i32x4_add(s0, wasm_i32x4_dot_i16x8(wasm_i16x8_load8x8(a + c), wasm_v128_load(x + c)));
    y[r] = wasm_i32x4_extract_lane(s0, 0) + wasm_i32x4_extract_lane(s0, 1) + wasm_i32x4_extract_lane(s0, 2) + wasm_i32x4_extract_lane(s0, 3); }
}
