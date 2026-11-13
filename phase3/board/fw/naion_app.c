/* Phase 3 board step F (v5 Na-ion, NA-E032 split 2021): on-board streaming EOL prediction.
 * For every VAL cell and every new cycle n = 1..nmax:
 *   E  (ST INT8 C-model "enc"):  curve[3][300] -> h_n[32]                 (once per new cycle, cached)
 *   state row n: [h_n, h_n - h_1, (qd_n/qd_1 - 1)*fs, (qc_n/qc_1 - 1)*fs]
 *   H  (ST INT8 C-model "head"): states[100][66] + mask[100] -> y_net
 *   C  (this file): masked line fit on the last NA_WINDOW q_d values -> est; gate MLP 4->16->1 (exact GELU);
 *      y = (1-g) y_net + g z_ext;  EOL = y * NA_LABEL_SCALE + NA_LABEL_MEAN
 * Timing with the DWT cycle counter; one line per prefix over the ST-LINK VCP UART:
 *   P,<cell>,<n>,<eol_board>,<eol_host_ref>,<eol_true>,<cyc_E>,<cyc_H>,<cyc_C>
 * and a summary line S,<n_prefix>,<mape_board>,<mape_ref>,<max_abs_diff>,<mean ms E>,<mean ms H>,<mean ms C>.
 * Mirrors c_part() in phase3/board/st_split_export.py. */
#include <math.h>
#include <stdio.h>
#include <string.h>
#include "stm32h7rsxx.h"
#include "enc.h"
#include "head.h"
#include "naion_params.h"

STAI_ENC_CONTEXT_DECLARE(net_enc, STAI_ENC_CONTEXT_SIZE)
STAI_HEAD_CONTEXT_DECLARE(net_head, STAI_HEAD_CONTEXT_SIZE)
STAI_ALIGNED(STAI_ENC_ACTIVATION_1_ALIGNMENT) static uint8_t act_enc[STAI_ENC_ACTIVATION_1_SIZE_BYTES];
STAI_ALIGNED(STAI_HEAD_ACTIVATION_1_ALIGNMENT) static uint8_t act_head[STAI_HEAD_ACTIVATION_1_SIZE_BYTES];
static stai_ptr in_enc[STAI_ENC_IN_NUM], out_enc[STAI_ENC_OUT_NUM], in_head[STAI_HEAD_IN_NUM], out_head[STAI_HEAD_OUT_NUM];

static float h_cache[100][32];
static float qd_hist[100], qc_hist[100];

static int ai_init(void)
{
  stai_ptr a[1]; stai_size n;
  if (stai_enc_init(net_enc) != STAI_SUCCESS || stai_head_init(net_head) != STAI_SUCCESS) return -1;
  a[0] = (stai_ptr)act_enc;  stai_enc_set_activations(net_enc, a, 1);
  a[0] = (stai_ptr)act_head; stai_head_set_activations(net_head, a, 1);
  stai_enc_get_inputs(net_enc, in_enc, &n);   stai_enc_get_outputs(net_enc, out_enc, &n);
  stai_head_get_inputs(net_head, in_head, &n); stai_head_get_outputs(net_head, out_head, &n);
  return 0;
}

#include "naion_cpart.c"

static float max3(const float *p, int len) { float m = p[0]; for (int i = 1; i < len; i++) if (p[i] > m) m = p[i]; return m; }

void naion_app_run(void)
{
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; DWT->CYCCNT = 0; DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  if (ai_init()) { printf("E,ai_init failed\r\n"); return; }
  printf("I,NA-E032 split2021 streaming E->H->C, SystemCoreClock=%lu\r\n", (unsigned long)SystemCoreClock);
  double ape_b = 0, ape_r = 0, maxd = 0, cyc_e = 0, cyc_h = 0, cyc_c = 0; int np = 0;
  float *states = (float *)in_head[0], *mask = (float *)in_head[1];
  for (int c = 0; c < na_ncells; c++) {
    const na_cell_t *cell = &na_cells[c];
    for (int n = 1; n <= cell->nmax; n++) {
      int i = n - 1; const float *curve = cell->curves + (size_t)i * 900;
      /* E: encode the new cycle */
      uint32_t t0 = DWT->CYCCNT;
      memcpy(in_enc[0], curve, 900 * sizeof(float));
      stai_enc_run(net_enc, STAI_MODE_SYNC);
      memcpy(h_cache[i], out_enc[0], 32 * sizeof(float));
      uint32_t t1 = DWT->CYCCNT;
      qd_hist[i] = max3(curve + 600 + 150, 150); qc_hist[i] = max3(curve + 600, 150);   /* channel 2: [qc | qd] */
      /* H: build the state window and run the TCN + head */
      memset(states, 0, 100 * 66 * sizeof(float)); memset(mask, 0, 100 * sizeof(float));
      float qd1 = qd_hist[0] > 1e-3f ? qd_hist[0] : 1e-3f, qc1 = qc_hist[0] > 1e-3f ? qc_hist[0] : 1e-3f;
      for (int r = 0; r < n; r++) {
        float *row = states + r * 66;
        for (int k = 0; k < 32; k++) { row[k] = h_cache[r][k]; row[32 + k] = h_cache[r][k] - h_cache[0][k]; }
        row[64] = (qd_hist[r] / qd1 - 1.0f) * (float)NA_FADE_SCALE; row[65] = (qc_hist[r] / qc1 - 1.0f) * (float)NA_FADE_SCALE;
        mask[r] = 1.0f;
      }
      stai_head_run(net_head, STAI_MODE_SYNC);
      double y_net = ((float *)out_head[0])[0];
      uint32_t t2 = DWT->CYCCNT;
      /* C: extrapolation + gate */
      double y = c_part(qd_hist, n, y_net);
      uint32_t t3 = DWT->CYCCNT;
      double eol = y * NA_LABEL_SCALE + NA_LABEL_MEAN, ref = cell->ref_eol[i];
      ape_b += fabs(eol - cell->eol) / cell->eol; ape_r += fabs(ref - cell->eol) / cell->eol;
      if (fabs(eol - ref) > maxd) maxd = fabs(eol - ref);
      cyc_e += t1 - t0; cyc_h += t2 - t1; cyc_c += t3 - t2; np++;
      printf("P,%d,%d,%.4f,%.4f,%.2f,%lu,%lu,%lu\r\n", c, n, eol, ref, cell->eol,
             (unsigned long)(t1 - t0), (unsigned long)(t2 - t1), (unsigned long)(t3 - t2));
    }
  }
  double k = 1000.0 / SystemCoreClock / np;
  printf("S,%d,%.6f,%.6f,%.6f,%.4f,%.4f,%.4f\r\n", np, ape_b / np, ape_r / np, maxd, cyc_e * k, cyc_h * k, cyc_c * k);
}
