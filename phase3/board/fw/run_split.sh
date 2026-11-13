#!/bin/zsh
# Phase 3 board step F for one split: generate C-models + val data, build, flash Appli (XSPI2), reset, capture UART.
# Usage: phase3/board/fw/run_split.sh 2021|42|2024      (Boot is unchanged; flash it only once)
set -e
SPLIT=$1; typeset -A EID; EID=(2021 NA-E032 42 NA-E033 2024 NA-E034); E=${EID[$SPLIT]}
FW=${0:A:h}; B=$FW/..; PY=/opt/homebrew/Caskroom/miniforge/base/envs/battery/bin/python
ST=/Applications/ST/STEdgeAI/4.1/Utilities/macarm/stedgeai; BUN="$HOME/Library/Application Support/stm32cube/bundles"
PRG="$BUN/programmer/2.23.0/bin/STM32_Programmer_CLI"; CM="$BUN/cmake/4.3.1+st.1/CMake.app/Contents/bin/cmake"
export PATH="$BUN/gnu-tools-for-stm32/14.3.1+st.2/bin:$BUN/ninja/$(ls $BUN/ninja)/bin:$PATH"
rm -rf $FW/net_enc $FW/net_head $FW/ws_enc $FW/ws_head
$ST generate -m $B/${E}_E_int8.onnx --name enc  --target stm32h7 --input-data-type float32 --output-data-type float32 --workspace $FW/ws_enc  --output $FW/net_enc  >/dev/null
$ST generate -m $B/${E}_H_int8.onnx --name head --target stm32h7 --input-data-type float32 --output-data-type float32 --workspace $FW/ws_head --output $FW/net_head >/dev/null
(cd $B/../.. && BL_SPLIT=$SPLIT $PY $FW/gen_fw_data.py | tail -1)
(cd $FW/proj && "$CM" --build --preset Release | grep -E "error|FLASH:|FAILED" || true)
ELF=$(ls $FW/proj/Appli/build/*Appli.elf)
(cd $FW/proj && "$PRG" -c port=SWD mode=UR -d $ELF --extload flasher/MX25UW25645G_NUCLEO-H7S3L8.stldr -q | grep -E "complete|Error")
mkdir -p $B/onboard
($PY $FW/serial_capture.py /dev/cu.usbmodem11302 $B/onboard/stream_$SPLIT.log 240) &
sleep 2; "$PRG" -c port=SWD mode=HOTPLUG -hardRst -q | grep -E "Hard reset|Error"; wait
