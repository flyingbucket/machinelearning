// lbp.c
// 依赖: stb_image.h (https://github.com/nothings/stb)
// 编译: gcc -O2 -o lbp lbp.c -lm
// 用法: ./lbp path/to/image.png [pad] [mode]
// mode: reflect(默认) | edge | constant

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static inline int clamp(int x, int lo, int hi) {
  return x < lo ? lo : (x > hi ? hi : x);
}

static inline int reflect_index(int i, int n) {
  // numpy "reflect" 风格（不重复边界值）
  if (n <= 1)
    return 0;
  // 将 i 映射到 [0, n-1]，镜像来回
  while (i < 0 || i >= n) {
    if (i < 0)
      i = -i - 1;
    else
      i = 2 * n - i - 1;
  }
  return i;
}

// 将任意 (r, c) 根据 pad 和 mode 映射到原图坐标，或指示常数 0
static inline uint8_t sample_with_pad(const uint8_t *src, int H, int W, int r,
                                      int c, int pad, const char *mode) {
  // r, c 为 padded 坐标; 先换算到原图坐标系
  int rr = r - pad;
  int cc = c - pad;

  if (strcmp(mode, "reflect") == 0) {
    int r0 = reflect_index(rr, H);
    int c0 = reflect_index(cc, W);
    return src[r0 * W + c0];
  } else if (strcmp(mode, "edge") == 0) { // clamp
    int r0 = clamp(rr, 0, H - 1);
    int c0 = clamp(cc, 0, W - 1);
    return src[r0 * W + c0];
  } else if (strcmp(mode, "constant") == 0) {
    if (rr < 0 || rr >= H || cc < 0 || cc >= W)
      return 0;
    return src[rr * W + cc];
  } else {
    // 未知模式，默认 reflect
    int r0 = reflect_index(rr, H);
    int c0 = reflect_index(cc, W);
    return src[r0 * W + c0];
  }
}

// 直接一次性生成 padded 图像 (H+2p) x (W+2p)
static uint8_t *make_padded(const uint8_t *src, int H, int W, int pad,
                            const char *mode, int *outH, int *outW) {
  int Hp = H + 2 * pad;
  int Wp = W + 2 * pad;
  uint8_t *dst = (uint8_t *)malloc((size_t)Hp * Wp);
  if (!dst)
    return NULL;

  for (int r = 0; r < Hp; ++r) {
    for (int c = 0; c < Wp; ++c) {
      dst[r * Wp + c] = sample_with_pad(src, H, W, r, c, pad, mode);
    }
  }
  *outH = Hp;
  *outW = Wp;
  return dst;
}

// 计算以 (r, c) 为左上角的 3x3 区域的 LBP 值（在 padded 图上）
static inline uint8_t lbp_kernel_3x3(const uint8_t *img, int Wp, int r, int c) {
  // 拉平成 9 个元素的索引（行主序）:
  // [0 1 2
  //  3 4 5
  //  6 7 8]
  const int off0 = r * Wp + c;
  uint8_t p[9];
  p[0] = img[off0 + 0];
  p[1] = img[off0 + 1];
  p[2] = img[off0 + 2];
  p[3] = img[off0 + Wp + 0];
  p[4] = img[off0 + Wp + 1]; // center
  p[5] = img[off0 + Wp + 2];
  p[6] = img[off0 + 2 * Wp + 0];
  p[7] = img[off0 + 2 * Wp + 1];
  p[8] = img[off0 + 2 * Wp + 2];

  const uint8_t center = p[4];

  // 顺时针 8 邻域（不含中心）
  // Python 顺序: [0, 1, 2, 5, 8, 7, 6, 3]
  const int order[8] = {0, 1, 2, 5, 8, 7, 6, 3};

  uint8_t code = 0;
  for (int i = 0; i < 8; ++i) {
    code <<= 1;
    code |= (p[order[i]] >= center) ? 1 : 0;
  }
  return code;
}

// 主流程: 读取灰度图 -> padding -> 扫描 3x3 -> 统计 256-bin 直方图

// === Histogram drawing (PPM) ===
// 以 PPM (P6) 写出 256-bin 直方图图像，不依赖任何第三方库。
static void write_hist_ppm(const unsigned int counts[256],
                           const char *out_path,
                           int img_w, int img_h,
                           int bar_gap /*像素间隔，建议=1或2*/) {
  if (img_w <= 0) img_w = 1024;
  if (img_h <= 0) img_h = 256;
  if (bar_gap < 0) bar_gap = 0;

  // 计算每个柱子的宽度（尽量整除）
  int total_gap = bar_gap * 255; // 256根柱子之间仅有255个间隔
  int bar_w = (img_w - total_gap) / 256;
  if (bar_w < 1) { bar_w = 1; } // 至少1px
  int used_w = bar_w * 256 + total_gap;
  int margin_left = (img_w - used_w) / 2; // 居中一些

  // 找最大计数做归一化
  unsigned int maxc = 0;
  for (int i = 0; i < 256; ++i) if (counts[i] > maxc) maxc = counts[i];
  if (maxc == 0) maxc = 1;

  // 分配画布（RGB）
  size_t buf_sz = (size_t)img_w * img_h * 3;
  unsigned char *buf = (unsigned char*)malloc(buf_sz);
  if (!buf) {
    fprintf(stderr, "alloc ppm buffer failed\n");
    return;
  }
  // 背景白色
  memset(buf, 255, buf_sz);

  // 画坐标轴（可选，这里画底边）
  for (int x = 0; x < img_w; ++x) {
    int y = img_h - 1;
    size_t off = ((size_t)y * img_w + x) * 3;
    buf[off + 0] = 0; buf[off + 1] = 0; buf[off + 2] = 0;
  }

  // 画柱子（黑色）
  int x = margin_left;
  for (int i = 0; i < 256; ++i) {
    // 高度：按比例映射到 [0, img_h-1)
    double h = (double)counts[i] / (double)maxc;
    int hpx = (int)(h * (img_h - 1));
    if (hpx < 0) hpx = 0;
    if (hpx > img_h - 1) hpx = img_h - 1;

    // 从底部往上填充 bar_w 像素宽
    for (int bx = 0; bx < bar_w; ++bx) {
      int cx = x + bx;
      if (cx < 0 || cx >= img_w) continue;
      for (int dy = 0; dy < hpx; ++dy) {
        int cy = (img_h - 2) - dy; // 留一像素给底边轴
        if (cy < 0) break;
        size_t off = ((size_t)cy * img_w + cx) * 3;
        buf[off + 0] = 0;
        buf[off + 1] = 0;
        buf[off + 2] = 0;
      }
    }
    x += bar_w + bar_gap;
  }

  // 写 PPM(P6) 文件
  FILE *fp = fopen(out_path, "wb");
  if (!fp) {
    fprintf(stderr, "open %s failed\n", out_path);
    free(buf);
    return;
  }
  fprintf(fp, "P6\n%d %d\n255\n", img_w, img_h);
  fwrite(buf, 1, buf_sz, fp);
  fclose(fp);
  free(buf);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "Usage: %s <image_path> [pad] [mode]\n", argv[0]);
    fprintf(stderr, "  mode: reflect(default) | edge | constant\n");
    return 1;
  }
  const char *path = argv[1];
  int pad = (argc >= 3) ? atoi(argv[2]) : 1;
  const char *mode = (argc >= 4) ? argv[3] : "reflect";

  int W = 0, H = 0, C = 0;
  // 要求输出单通道；stb_image 会帮你转灰度
  uint8_t *src = stbi_load(path, &W, &H, &C, 1);
  if (!src) {
    fprintf(stderr, "Failed to load image: %s\n", path);
    return 1;
  }
  C = 1;

  int Hp = 0, Wp = 0;
  uint8_t *padded = NULL;
  if (pad > 0) {
    padded = make_padded(src, H, W, pad, mode, &Hp, &Wp);
    if (!padded) {
      fprintf(stderr, "Out of memory for padded image.\n");
      stbi_image_free(src);
      return 1;
    }
  } else {
    // 不 padding，直接用原图
    padded = (uint8_t *)malloc((size_t)H * W);
    if (!padded) {
      fprintf(stderr, "Out of memory.\n");
      stbi_image_free(src);
      return 1;
    }
    memcpy(padded, src, (size_t)H * W);
    Hp = H;
    Wp = W;
  }

  // 统计直方图
  uint32_t counts[256] = {0};

  // 在 padded 图上以 (r,c) 为左上角，窗口 [r:r+3, c:c+3]
  // 合法起点是 0..Hp-3, 0..Wp-3
  for (int r = 0; r <= Hp - 3; ++r) {
    for (int c = 0; c <= Wp - 3; ++c) {
      uint8_t code = lbp_kernel_3x3(padded, Wp, r, c);
      counts[code] += 1;
    }
  }

  // 输出结果（可按需改为写文件）
  printf("# LBP histogram (code : count)\n");
  for (int k = 0; k < 256; ++k) {
    if (counts[k] != 0) {
      printf("%3d : %u\n", k, counts[k]);
    }
  }

  // === 写出PPM直方图图像 ===
  write_hist_ppm(counts, "lbp_hist.ppm", 1024, 256, 1);
  fprintf(stderr, "wrote histogram image: lbp_hist.ppm\n");

  free(padded);
  stbi_image_free(src);
  return 0;
}
