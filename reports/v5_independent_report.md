# Báo Cáo Độc Lập Dự Án v5_auto: Dự Đoán Tuổi Thọ Pin Na-ion (BatteryLife)

**Người lập báo cáo:** Gemini 4 Argon (Independent Analyst) | **Ngày:** 2026-11-09  
**Phạm vi:** Đối chiếu độc lập toàn bộ 35 thí nghiệm huấn luyện (`NA-E000`..`NA-E034`), kiểm thử nén Phase 3 (`P0`–`P3`) và kết quả kiểm thử một lần trên tập Test (`TEST-*`) từ các tệp hồ sơ gốc.

---

## 1. Tóm Tắt Điều Hành & Ba Đóng Góp Luận Văn

Dự án **v5_auto** giải bài toán dự đoán tuổi thọ (EOL) cho bộ dữ liệu **Pin Na-ion của BatteryLife** (31 cell; 3 phép chia chuẩn của bài báo gốc: seeds `2021`, `42`, `2024`, mỗi phép chia gồm 20 cell train / 6 cell val / 5 cell test, tạo 100 tiền tố chu kỳ/cell). Mô hình đóng băng cuối cùng (**NaFadeNet + ExtrapGate(s=50) + N1 physics loss**, 39.236 tham số — nhỏ hơn 13 lần so với CPMLP 510.593 tham số) giảm sai số MAPE trên tập Test từ **0,2656 ± 0,0376** (CPMLP tái lập) xuống **0,1751 ± 0,0610** (−34,1% tương đối), vượt cả mô hình tốt nhất trong bài báo BatteryLife trên Na-ion là **CPTransformer (0,255 ± 0,036)**.

1. **Đóng góp 1 — Kiến trúc & Hàm mất mát có định hướng vật lý (Physics-informed Architecture & Loss):** Đưa trực tiếp đặc trưng suy giảm dung lượng tương đối ($q_d/q_{d,1}, q_c/q_{c,1}$) và nhánh ngoại suy tuyến tính có cổng thích nghi (adaptive extrapolation gate) vào mạng, kết hợp hàm mất mát vật lý **N1** (`degradation_law_fixed_b`, bậc $b=1$, ngưỡng EOL $0{,}8$ chuẩn BatteryLife). Riêng hàm mất mát N1 giúp giảm MAPE cùng kiến trúc trên cả 3 phép chia: Val **0,1087 → 0,0976**, Test **0,1910 → 0,1751** (Acc@15 Test tăng **0,394 → 0,515**).
2. **Đóng góp 2 — Chẩn đoán và kiểm tra độ dốc (Gradient Diagnostics Protocol):** Thiết lập quy trình tự động cân bằng trọng số $\lambda$ từ tỷ số chuẩn gradient ở bước 0 (float64 trên 3 batch cố định, $\rho \in \{0{,}5; 1{,}0\}$), giám sát dải cân bằng $R = \lambda\|\nabla L_{\text{phys}}\|/\|\nabla L_{\text{main}}\| \in [0{,}05; 2{,}0]$, góc cosine, tỷ lệ mẫu kích hoạt (active fraction), phép thử nhiễu bằng 0 (zero-perturbation check) và tiền kiểm độ nhạy (sensitivity pre-check). Quy trình này phát hiện 2 lỗi ngầm nghiêm trọng (nhiễu mặt nạ dropout độc lập ở `NA-E021` và suy biến đẩy hằng số do mô hình bất biến với mức điện áp ở `NA-E022`), đồng thời loại sớm 4 ứng viên vật lý không khả thi trước khi chạy GPU.
3. **Đóng góp 3 — Triển khai biên / TinyML (Edge Deployment):** Chuyển đổi sang cơ chế suy luận dòng (streaming P1) với bộ đệm trạng thái TCN nhân quả + lịch sử $q_d$, khớp tuyệt đối mô hình toàn cửa sổ (sai lệch tương đối tối đa $\le 2{,}7 \times 10^{-7}$ chu kỳ) và giảm RAM đầu vào từ **352 KB xuống ~98 KB fp32**. Lượng tử hóa sau huấn luyện (P3) với **fp16 weights (76,6 KB)** và **INT8 weight-only dynamic (38 KB trọng số / 104,9 KB tệp ONNX)** giữ nguyên độ chính xác trên cả Val lẫn Test ($\Delta\text{MAPE Test} = +0{,}0007$ với INT8).

---

## 2. Kiến Trúc Mô Hình Đóng Băng & Hàm Mất Mát (Tham Số Chi Tiết)

* **Kiến trúc (`NaFadeNet + ExtrapGate(s=50)`, mã cấu hình `NA-E009` / `NA-E032..E034`, 39.236 tham số):**
  * **Bộ mã hóa chu kỳ (Per-cycle CNN):** Trích xuất đặc trưng $h_t$ từ đường cong $(V, I, Q) \in \mathbb{R}^{100 \times 3 \times 300}$ của từng chu kỳ.
  * **Đặc trưng quỹ đạo suy giảm:** Ghép $[h_t,\, h_t - h_1,\, r_{d,t} = q_{d,t}/q_{d,1},\, r_{c,t} = q_{c,t}/q_{c,1}]$ đưa qua mạng **Causal Masked TCN** (dilation $1, 2, 4, 8$), tổng hợp `mean + last` qua MLP dự đoán $\hat{y}_{\text{net}}$.
  * **Đầu ngoại suy có cổng tái tham số hóa (`ExtrapGate`, +99 tham số, hệ số khuếch đại học $s=50$):** Khớp đường thẳng suy giảm trên 20 chu kỳ quan sát cuối của $q_d$ tới ngưỡng EOL $0{,}8$ để lấy $\hat{y}_{\text{lin}}$ (hiệu chỉnh qua $a = 2 + \text{da}\cdot s$, $c = -3{,}6 + \text{dc}\cdot s$), trộn với $\hat{y}_{\text{net}}$ qua cổng $g \in (0,1)$ có trọng số tăng dần theo độ dài tiền tố.
* **Hàm mất mát huấn luyện (`loss.md §1, §4.8, §5.2`):**
  $$\mathcal{L} = \mathcal{L}_{\text{RelHuber}}(\hat{y}, y; \delta=0{,}15) + \lambda \cdot \mathcal{L}_{N1}, \quad \mathcal{L}_{N1} = \frac{1}{|S|}\sum_{s\in S}\frac{1}{32}\sum_{k=1}^{32}\frac{|q_{\text{ref},s} - (q_{\text{ref},s}-0{,}8)n_k/\hat{y}_s - q_s(n_k)|}{q_{\text{ref},s}-0{,}8}$$
  * Tham số huấn luyện khóa chặt: Optimizer Adam $\text{lr}=5\times 10^{-5}$, $\text{batch}=128$, tối đa 100 epochs (tối thiểu 5, patience 5 — `A2-NA`).
  * Tham số N1 (`NA-E032/E033/E034`): $b=1$ cố định, $K=32$ điểm lấy mẫu đều trên $[1, n_{0{,}8}]$ của quỹ đạo $q_s(n)$ làm mượt median-9 (chỉ dùng cell tập train), $\rho = 1{,}0 \Rightarrow \lambda$ tự hiệu chỉnh theo split lần lượt là **0,481 / 0,241 / 0,337**; tỷ số độ dốc trung vị $R$ đạt **1,55 / 0,81 / 0,61** (đều đạt chuẩn `PASS` $\in [0{,}05; 2{,}0]$).

---

## 3. Bảng Tổng Hợp Kết Quả Validation, Test và Nén TinyML

### 3.1. Kết quả Validation (3 splits × 6 cells = 18 cell lượt đánh giá, 600 tiền tố/split)
| Nhóm mô hình | Thí nghiệm | Tham số | Val MAPE (2021 / 42 / 2024) | Val MAPE (TB ± SD) | Acc@15 | RMSE (chu kỳ) | Quyết định |
|---|---|---:|---|---|---|---|---|
| **CPMLP (Baseline nội bộ)** | `NA-E000..E002` | 510.593 | 0,1261 / 0,2337 / 0,1737 | **0,1778 ± 0,0539** | 0,499 | 41,4 | Baseline |
| **NaFadeNet (TCN cơ sở)** | `NA-E003..E005` | 39.137 | 0,1234 / 0,2062 / 0,1772 | **0,1690 ± 0,0420** | 0,498 | 40,3 | KEEP (gain 0,0089) |
| **+ ExtrapGate (chưa sửa s)** | `NA-E006..E008` | 39.236 | 0,1053 / 0,1819 / 0,1550 | **0,1474 ± 0,0389** | 0,554 | 35,6 | KEEP (g kẹt ở ~0,12) |
| **+ GateRepair (s=50, no phys)** | `NA-E009..E011` | 39.236 | 0,0999 / 0,1170 / 0,1091 | **0,1087 ± 0,0085** | 0,729 | 29,4 | Đạt đích Phase 1 |
| *Ablation: bỏ anchor ($h-h_1$)* | `NA-E012..E014` | 37.700 | 0,0854 / 0,1213 / 0,1154 | 0,1074 ± 0,0193 | 0,757 | 28,1 | Anchor không đóng góp |
| *Ablation: bỏ fade ($r_d, r_c$)* | `NA-E015..E017` | 39.140 | 0,0835 / 0,1897 / 0,1156 | 0,1296 ± 0,0545 | 0,679 | 31,9 | Xác nhận đóng góp (+0,0209) |
| *Ablation: cổng cố định $g=0{,}12$* | `NA-E018..E020` | 39.139 | 0,1046 / 0,1805 / 0,1541 | 0,1464 ± 0,0385 | 0,556 | 35,5 | Xác nhận đóng góp (+0,0378) |
| *Degradation law (learned $b$, $\rho=1$)* | `NA-E026..E028` | 39.334 | 0,0908 / 0,1121 / 0,1116 | 0,1049 ± 0,0121 | 0,791 | 28,3 | DISCARD (law-fit −15,6%) |
| *Ablation hàm chính: MSE loss* | `NA-E029..E031` | 39.236 | 0,0727 / 0,1203 / 0,1256 | 0,1062 ± 0,0292 | 0,765 | 27,7 | TIED (SD cao gấp 3,4×) |
| **FROZEN: GateRepair + N1 ($b=1$)** | `NA-E032..E034` | **39.236** | **0,0767 / 0,1076 / 0,1085** | **0,0976 ± 0,0181** | **0,793** | **27,0** | **KEEP (theo A7-NA)** |

### 3.2. Kết quả Test Một Lần (One-Time Test, 3 splits × 5 cells) & Nén TinyML
| Mô hình / Phiên bản lượng tử hóa | Dung lượng trọng số | Test MAPE (2021 / 42 / 2024) | Test MAPE (TB ± SD) | Test Acc@15 | Test RMSE | Tham chiếu bài báo BatteryLife |
|---|---:|---|---|---|---|---|
| **Dummy (trung bình tập train)** | — | 0,4452 / 0,3796 / 0,3866 | **0,4038 ± 0,0360** | 0,067 | 64,3 | Paper Dummy: `0,404 ± 0,029` |
| **CPMLP (nội bộ, Relative Huber)** | ~1.995 KB | 0,3075 / 0,2542 / 0,2350 | **0,2656 ± 0,0376** | 0,275 | 65,7 | Paper CPMLP: `0,274 ± 0,026` |
| **CPTransformer (số liệu bài báo)** | — | — | **0,255 ± 0,036** | 0,406 | — | Tốt nhất trên Na-ion trong bài báo |
| **NaFadeNet+ExtrapGate(s=50), no phys** | 153,3 KB | 0,2536 / 0,1715 / 0,1479 | **0,1910 ± 0,0555** | 0,394 | 47,8 | — |
| **FROZEN + N1 physics loss (fp32)** | **153,3 KB** | **0,2452 / 0,1467 / 0,1334** | **0,1751 ± 0,0610** | **0,515** | **45,3** | Giảm 34,1% MAPE so với CPMLP |
| **FROZEN N1 — fp16 weights (fp32 compute)** | **76,6 KB** | 0,2451 / 0,1467 / 0,1334 | **0,1751 ± 0,0610** | **0,515** | **45,3** | PASS ($\Delta\text{Val} \le +0{,}0001$) |
| **FROZEN N1 — INT8 dynamic (weight-only)** | **~38 KB** (104,9 KB ONNX) | 0,2454 / 0,1474 / 0,1346 | **0,1758 ± 0,0606** | **0,513** | **45,4** | PASS ($\Delta\text{Test} = +0{,}0007$) |
| *FROZEN N1 — INT8 static QDQ toàn phần* | 182,7 KB ONNX | *(Val: 0,0889 / 0,1388 / 0,1313)* | *(Val $\Delta$: +0,012..+0,031)* | — | — | **FAIL** (vỡ số học cổng/ngoại suy) |
| *FROZEN N1 — INT8 static mixed (cổng fp32)*| 111,2 KB ONNX | *(Val: 0,0782 / 0,1126 / 0,1085)* | *(Val $\Delta$: +0,0015/+0,0050/0)*| — | — | **FAIL** trên 2 splits (2021, 42) |

---

## 4. Phân Tích Nguyên Nhân Thành Công và Thất Bại

* **Những gì hiệu quả và lý do:**
  1. **Cổng ngoại suy tái tham số hóa ($s=50$, `NA-E009..E011`) + tỷ số dung lượng ($r_d, r_c$):** Ở tiền tố ngắn (1–10 chu kỳ), ngoại suy tuyến tính có sai số cực lớn (MAPE 2,5–4,7), nhưng ở tiền tố dài (51–100 chu kỳ) lại rất chuẩn (MAPE 0,04–0,11). Khi chưa có hệ số $s=50$ (`NA-E006`), với $\text{lr}=5\times 10^{-5}$ và ~180 bước cập nhật, cổng $g$ bị đóng băng quanh giá trị khởi tạo $0{,}12$. Nhờ nhân hệ số học $s=50$, $g$ tăng thích nghi theo độ dài tiền tố (0,158 → 0,203 ở split 2021; 0,259 → 0,629 ở split 42), tạo bước nhảy lớn nhất dự án (Val MAPE giảm **0,1474 → 0,1087**).
  2. **Hàm mất mát quy luật suy giảm bậc cố định $b=1$ (`N1`, `NA-E032..E034`):** Quỹ đạo xả thực tế của Na-ion gần như tuyến tính (bậc trung vị $b=1{,}05$, hệ số tương quan giữa tốc độ suy giảm chu kỳ 10–100 và tốc độ tới ngưỡng 0,8 đạt 0,95–0,96). Ràng buộc trực tiếp dự đoán $\hat{y}$ vào quỹ đạo $q_s(n)$ với $b=1$ (không qua đầu phụ) tạo gradient đồng hướng với hàm chính vào cuối quá trình học ($\cos \in [+0{,}87; +0{,}95]$ trên split 2021/2024), giúp giảm sai số quy luật (law-fit) từ 0,0539 xuống 0,0445 (−17,4%) và cải thiện cả Val lẫn Test trên mọi phép chia.
  3. **Tương đối Huber ($\delta=0{,}15$) so với MSE (`A6-NA`):** Dù MSE đạt MAPE trung bình tương đương (0,1062 so với 0,1087), độ lệch chuẩn giữa các phép chia của MSE cao gấp **3,4 lần** (0,0292 so với 0,0085) và độ chệch (bias) đảo dấu mạnh (+4,2 / −20,3 / +6,4 chu kỳ).
* **Những gì thất bại và lý do:**
  1. **Anchor chu kỳ 1 ($h_t - h_1$, `NA-E012..E014`):** Khi bỏ nhánh này, Val MAPE giảm nhẹ (0,1087 → 0,1074) do thông tin suy giảm đã nằm đủ trong $r_d, r_c$.
  2. **Các hàm mất mát What-if (`P2`, `P0/P0c/P4b`, `P-CE`, `P-H`) và Prefix-Drift (`NA-E021..E024`):** Mô hình cha đã tự thỏa mãn hoàn toàn các phép thử giảm dung lượng (`P0/P0c/P4b` có 0 vi phạm $>10^{-3}$ chu kỳ, dịch chuyển đúng −13..−70 chu kỳ). Ngược lại, mạng hầu như không nhạy với mức điện áp chuẩn hóa (`P2` dịch chuyển $\le 0{,}005$ chu kỳ); khi dùng `stop-gradient` đúng chuẩn (`NA-E022`), đạo hàm biến thành lực đẩy giảm dự đoán vô điều kiện ($\cos = -0{,}996$), làm sập độ chính xác (MAPE 0,1760). Còn `prefix_drift` hai chiều và một chiều (`NA-E023/E024`) đều có gradient ngược hướng hàm chính ($\cos \to -0{,}94..-0{,}97$), làm hỏng dự đoán ở tiền tố dài.
  3. **Quy luật suy giảm học số mũ $b$ (`NA-E025..E028`, `A5-NA`):** Đầu phụ học tự do $b$ làm yếu lực ràng buộc vật lý lên $\hat{y}$, chỉ giảm phần dư law-fit được 15,6% (yêu cầu $\ge 30\%$), nên chỉ đóng vai trò điều chuẩn thống kê.
  4. **Lượng tử hóa tĩnh toàn phần INT8 QDQ (`P3`):** Phép chia tỷ số $q_d/q_{d,1}$ (nhân $\times 10$) và độ dốc ngoại suy (nhân $\times 1000$) bị bão hòa/lỗi làm tròn nghiêm trọng khi ép kích hoạt sang INT8 tĩnh.

---

## 5. Đối Chiếu Chéo Dữ Liệu (Cross-Check) & Các Điểm Không Khớp Giữa Các Tệp

1. **Lệch nhỏ giữa `results.tsv` (CUDA A100) và `phase3/results_tinyml.tsv` (CPU PyTorch P0) trên `NA-E032..E034`:** `results.tsv` ghi `val_mape` là `0,076681 / 0,107602 / 0,108473` và `val_acc15` của `NA-E032` là **0,8600** (516/600), trong khi `results_tinyml.tsv` ghi `0,076678 / 0,107611 / 0,108466` và `val_acc15` của `NA-E032` là **0,8617** (517/600). Nguyên nhân: sai số dấu phẩy động $\le 9\times 10^{-6}$ giữa GPU A100 và CPU làm đúng 1 mẫu tiền tố nằm sát biên 15% đổi trạng thái đạt/trượt.
2. **Làm tròn độ chênh lệch của phép thử `fixed_gate=0.12` (`NA-E018..E020`):** `research_log.md` ghi mức tăng MAPE so với mô hình cha là **+0,0377** (lấy từ số đã làm tròn $0{,}1464 - 0{,}1087$), `handoff/reviews/ablations.md` ghi **+0,0378** (tính từ số chưa làm tròn $0{,}146409 - 0{,}108654 = +0{,}037755$), còn `state.json` làm tròn thành **+0,038**.
3. **Nhãn `gradient_health` trong `results.tsv` mâu thuẫn bề mặt với `reports/gradient_diagnostics_v5.md`:** Trong `results.tsv`, các lượt chạy `NA-E022, E026, E027, E032, E033, E034` bị gán nhãn `UNHEALTHY` (và `NA-E009..E021` là `WARNING`), nhưng trong `gradient_diagnostics_v5.md` tất cả 11 lượt chạy vật lý đều đạt `§5.3 balance = PASS`. Nguyên nhân: bộ kiểm tra cũ trong `train.py` dùng ngưỡng lỗi thời ($R \le 1{,}0$, $\cos \ge -0{,}3$) và báo động giả `EXPLODING` (tỷ số grad/param $6\times 10^{10}$) đối với hai tham số khởi tạo bằng 0 là `ext_da, ext_dc`.
4. **Độ bất định của tỷ lệ vi phạm P2 (`NA-E021`) giữa các thiết bị:** `results.tsv` và `research_log.md` ghi tỷ lệ vi phạm P2 trên tập Val là **0,480** (−42,6%), nhưng `handoff/reviews/NA-E021.md` ghi nhận khi chạy lại trên cùng checkpoint ở máy cục bộ thì ra **0,387** (−53,7%), vì độ dịch chuyển dự đoán chỉ $\le 0{,}0013$ chu kỳ (nhiễu dấu phẩy động giữa các thiết bị).
5. **Trường trạng thái tồn đọng trong `state.json`:** `state.json` (dòng 39 và 287) vẫn lưu chuỗi ghi chú cũ `"attribution ablations owed"` / `"ablations planned"`, dù 9 lượt chạy ablation `NA-E012..E020` đã hoàn tất và được duyệt trong `handoff/reviews/ablations.md`. Ngoài ra, `budget.usage.training_seconds = 28` chỉ đếm thời gian chạy smoke cục bộ (do 35 lượt huấn luyện chạy trên Colab A100).

---

## 6. Các Lưu Ý và Hạn Chế Quan Trọng (Caveats) Khi Viết Luận Văn

* **Quyết định hậu nghiệm `A7-NA` (Post-hoc rule relaxation):** Hàm mất mát N1 được thiết kế trước với ngưỡng vượt qua cổng vật lý là giảm $\ge 20\%$ sai số law-fit trên tập Val, nhưng thực tế chỉ đạt **−17,4%** (0,0539 → 0,0445). Việc giữ lại N1 (`KEEP`) dựa trên sửa đổi **A7-NA** của người dùng *sau khi đã xem kết quả* (chấp nhận mọi mức giảm $>0\%$). Luận văn bắt buộc phải trình bày trung thực đây là quyết định hậu nghiệm của người dùng và ghi nhận trên split 42 góc cosine giữa $\nabla L_{N1}$ và $\nabla L_{\text{main}}$ bị âm ở giữa quá trình học ($-0{,}64 \dots -0{,}80$) trước khi hồi phục ($+0{,}09$).
* **Tập Val dễ hơn tập Test đáng kể & cỡ mẫu rất nhỏ:** Bộ dữ liệu Na-ion chỉ có tổng cộng **31 cell** (6 cell Val và 5 cell Test mỗi phép chia). Phân phối cell của tập Val dễ hơn tập Test rất nhiều ở mọi mô hình (CPMLP: Val 0,1778 vs Test 0,2656; Mô hình đóng băng N1: Val **0,0976** vs Test **0,1751**). Chỉ một cell có tuổi thọ dài trên tập Test (`270040-1-3-62.p`, EOL 268 chu kỳ, xuất hiện ở cả 3 splits với MAPE 0,276–0,405) đã kéo toàn bộ sai số Test lên cao.
* **Thiên lệch chọn lọc trên tập Val (Selection bias qua 35 lượt chạy):** Đã có 35 lượt huấn luyện (`NA-E000`..`NA-E034`) và 7 lần sửa đổi quy tắc (`A1-NA`..`A7-NA`, bao gồm việc mở lại Phase 2 sau `NA-E024`) được chọn lọc dựa trên cùng 6 cell Val/split; hơn nữa nhánh `anchor` ($h_t - h_1$) dù được chứng minh là dư thừa trong ablation (`NA-E012..E014`) vẫn còn nằm lại trong cấu hình đóng băng (`NA-E009` config).
* **Chưa đo đạc trực tiếp trên phần cứng vi điều khiển STM32:** Các kết quả Phase 3 (streaming RAM ~98 KB, trọng số fp16 76,6 KB / INT8 dynamic ~38 KB) mới được xác thực trên host CPU (`onnxruntime`). Độ trễ suy luận thực tế (ms/chu kỳ), năng lượng tiêu thụ và bộ nhớ Flash/SRAM thực tế trên bo mạch STM32 / X-CUBE-AI chưa được đo đạc.
