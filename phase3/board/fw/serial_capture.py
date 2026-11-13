"""Read the on-board streaming app output (ST-LINK VCP, 115200 8N1), save it, and summarise.
Lines: I,... (info)  P,cell,n,eol_board*1e4,eol_ref*1e4,eol_true*1e4,cyc_E,cyc_H,cyc_C  S,np,mape_board,mape_ref,maxdiff,msE,msH,msC  D,done
Usage: python serial_capture.py /dev/cu.usbmodem11302 out.log [timeout_s]"""
import sys, time
import numpy as np, serial

port, out = sys.argv[1], sys.argv[2]; tmo = float(sys.argv[3]) if len(sys.argv) > 3 else 180
lines, t0 = [], time.time()
with serial.Serial(port, 115200, timeout=1) as s, open(out, "w") as f:
    while time.time() - t0 < tmo:
        l = s.readline().decode(errors="replace").strip()
        if not l: continue
        f.write(l + "\n"); f.flush(); lines.append(l)
        if l.startswith(("I,", "S,", "E,", "D,")): print(l)
        if l.startswith("D,"): break
P = np.array([[float(x) for x in l.split(",")[1:]] for l in lines if l.startswith("P,") and l.count(",") == 8])
if len(P):
    eb, er, et, ce, ch, cc = P[:, 2] / 1e4, P[:, 3] / 1e4, P[:, 4] / 1e4, P[:, 5], P[:, 6], P[:, 7]
    clk = 600e6
    print(f"prefixes {len(P)}  board MAPE {np.mean(abs(eb - et) / et):.6f}  host-ref MAPE {np.mean(abs(er - et) / et):.6f}  "
          f"max|board-ref| {np.abs(eb - er).max():.2e} cycles  Acc@15 {np.mean(abs(eb - et) / et <= 0.15):.4f}")
    for nm, c in (("E", ce), ("H", ch), ("C", cc), ("E+H+C", ce + ch + cc)):
        print(f"  {nm}: mean {c.mean() / clk * 1e3:.3f} ms  min {c.min() / clk * 1e3:.3f}  max {c.max() / clk * 1e3:.3f}")
