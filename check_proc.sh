#!/bin/bash
# 用法: ./check_proc.sh <PID>

PID=$1
if [ -z "$PID" ]; then
    echo "❌ 请提供要排查的进程 PID"
    exit 1
fi

echo "=== 进程基本信息 ==="
ps -fp $PID

echo -e "\n=== 命令行参数 ==="
tr '\0' ' ' < /proc/$PID/cmdline; echo

echo -e "\n=== 可执行文件路径 ==="
readlink -f /proc/$PID/exe

echo -e "\n=== 当前工作目录 ==="
readlink -f /proc/$PID/cwd

echo -e "\n=== 父进程信息 ==="
PPID=$(awk '/^PPid:/ {print $2}' /proc/$PID/status)
ps -fp $PPID

echo -e "\n=== 打开的文件句柄 (前20条) ==="
ls -l /proc/$PID/fd | head -n 20

echo -e "\n=== 网络连接 (需权限可见) ==="
grep -H "" /proc/$PID/net/{tcp,tcp6,udp,udp6} 2>/dev/null | head -n 20

echo -e "\n=== 环境变量 ==="
tr '\0' '\n' < /proc/$PID/environ | head -n 20

echo -e "\n=== 内存映射中加载的库 (前20条) ==="
grep -E "site-packages|\.so" /proc/$PID/maps | head -n 20

echo -e "\n=== 结束 ==="
