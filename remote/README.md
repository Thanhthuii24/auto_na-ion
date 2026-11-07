# v5_auto chạy trên Colab (qua Google Drive for Desktop)

Thư mục Drive: `My Drive/v5_auto_remote/` (code.tar.gz = code + 3 cache, queue/, out/).

**Người dùng (1 lần mỗi phiên Colab, ~12 h):**
1. Mở `My Drive/v5_auto_remote/colab_v5_worker.ipynb` bằng Colab (trình duyệt hoặc extension
   "Google Colab" trong VS Code, đăng nhập thanhthuyytltt@gmail.com).
2. Runtime → Change runtime type → GPU → Run all. Để tab mở; worker tự lấy job trong `queue/`.
3. Phiên bị ngắt → Run all lại (job đã DONE được bỏ qua).

**Agent (trên Mac):** `python remote/remote.py submit <ID> <proposal> --split S` →
`status` → `fetch <ID>`. Job FAILED: xem `out/<ID>/run.log`, sửa, xoá `out/<ID>/` rồi submit lại.
