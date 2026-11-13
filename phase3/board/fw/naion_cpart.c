/* C part of NaFadeNet+ExtrapGate (mirrors c_part() in st_split_export.py); included by naion_app.c */
#include <math.h>
#include "naion_params.h"
static double gelu(double v) { return 0.5 * v * (1.0 + erf(v / sqrt(2.0))); }
static double clampd(double v, double lo, double hi) { return v < lo ? lo : (v > hi ? hi : v); }

/* C part: identical to c_part(qd, n, y_net, G) in st_split_export.py */
static double c_part(const float *qd, int n, double y_net)
{
  double sw = 0, st = 0, sq = 0;
  for (int i = 0; i < n; i++) { int t = i + 1; if (t > n - NA_WINDOW) { sw += 1; st += t; sq += qd[i]; } }
  if (sw < 1) sw = 1;
  double mt = st / sw, mq = sq / sw, var = 0, cov = 0;
  for (int i = 0; i < n; i++) { int t = i + 1; if (t > n - NA_WINDOW) { var += (t - mt) * (t - mt); cov += (t - mt) * (qd[i] - mq); } }
  double slope = cov / (var > 1e-6 ? var : 1e-6), inter = mq - slope * mt, r2 = 0;
  for (int i = 0; i < n; i++) { int t = i + 1; if (t > n - NA_WINDOW) { double e = qd[i] - inter - slope * t; r2 += e * e; } }
  double resid = sqrt(r2 / sw);
  double est = (slope < 0 && n >= 3) ? (NA_THR - inter) / (slope < -1e-8 ? slope : -1e-8) : NA_CAP;
  if (est > NA_CAP) est = NA_CAP;
  if (est < n + 1) est = n + 1;
  double f[4] = { n / 100.0, est / 100.0, clampd(slope * 1000.0, -10, 10), clampd(resid * 100.0, 0, 10) };
  double logit = NA_B2;
  for (int j = 0; j < NA_HID; j++) {
    double a = na_b1[j];
    for (int k = 0; k < 4; k++) a += na_W1[j * 4 + k] * f[k];
    logit += na_W2[j] * gelu(a);
  }
  double g = 1.0 / (1.0 + exp(-(-2.0 + NA_S * logit)));
  double z = (2.0 + NA_S * NA_DA) * est / 100.0 + (-3.6 + NA_S * NA_DC);
  return (1 - g) * y_net + g * z;
}

