# gpu-hunter

守望 NVIDIA 显卡,一旦出现满足条件的**稳定空闲**卡,立即自动启动你的任务。

典型场景:`python train.py` 需要 2 张卡,但现在都被占着 —— 挂上 gpu-hunter,它每几秒轮询一次,凑齐卡的那一瞬间替你拉起训练。

## 安装

```bash
pip install -e .        # 依赖仅 nvidia-ml-py,Linux + NVIDIA 驱动
```

## 快速上手

```bash
# 等 1 张完全空闲的卡,空闲就跑 train.py
gpu-hunter -- python train.py

# 要 2 张卡,优先 GPU2、GPU3;输出写日志
gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py --epochs 100

# 每张卡需要 40GB 可用显存(不要求"完全空闲",但要求无计算进程)
gpu-hunter --mem-per-gpu 40G -- bash run.sh

# 只肯用 4/5/6 三张卡
gpu-hunter --allow 4,5,6 -- ./long_job.sh
```

单个 `xx.py` / `xx.sh` 会自动补解释器;`--` 之后的内容原样作为子命令执行。

## 参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gpu-num N` | 1 | 需要几张**同时**空闲的卡 |
| `--mem-per-gpu SIZE` | - | 单卡显存需求,如 `40G` / `24000M`;缺省表示要求卡完全空闲 |
| `--prefer LIST` | - | 排序偏好(软约束),如 `2,3`:先挑 2 再挑 3,**不够时用别的卡兜底** |
| `--allow LIST` | 全部 | 白名单(硬约束):只在列出的卡里挑,宁可死等 |
| `--interval SEC` | 5 | 轮询间隔 |
| `--stable-rounds K` | 3 | 连续 K 轮都空闲才算"稳定空闲",防止抓到两个任务间的间隙 |
| `--idle-mem SIZE` | 500M | 占用显存低于该值视为空闲(配合利用率 < 5%、无计算进程) |
| `--grace SEC` | 45 | 启动后的确认宽限期(见下文"抢占回滚") |
| `--heartbeat SEC` | 120 | 任务运行中定期打印各卡状态 |
| `--log FILE` | 终端透传 | 任务 stdout/stderr 写入文件(父进程日志始终走 stderr) |
| `--cwd DIR` | 当前目录 | 子进程工作目录 |

### prefer 与 allow 的区别

```
全部 GPU → allow 过滤(硬) → 稳定空闲筛选 → prefer 排序(软) → 取前 N 张
```

- `--prefer 3,2`:排序时 3 最优先、2 次之;它们没空就拿别的,不死等。
- `--allow 2,3`:只准用 2/3,别的空着也不碰。
- `--allow 2,3 --prefer 3`:在 2/3 里挑,两张都空时先拿 3。

多卡按队首贪婪取用:`--prefer 3 --gpu-num 2` 且空闲 {0,2,3} → 先拿 3,再拿编号最小的 0。

## 工作机制

1. **探测**:走 NVML(`nvidia-smi` 的底层库),不解析文本输出;"空闲" = 无计算进程 + 利用率低 + 显存占用低。
2. **稳定窗**:滑动窗口连续 K 轮空闲才出手;多卡要求同时满足(原子凑卡)。
3. **启动**:设置 `CUDA_VISIBLE_DEVICES` 后以独立会话拉起子进程。
4. **抢占校验**:宽限期内回查目标卡上的计算进程 PID。见到自家进程即放行;陌生进程持续出现且始终不见自家进程 → 判定被抢,杀掉本次启动并退避重试(超时未确认则带警告继续,兼容容器内 pid 不可见的情况)。多进程任务(torchrun worker)不会误杀。
5. **信号**:SIGINT/SIGTERM 转发给整个子进程组;5 秒内连按两次直接 SIGKILL。
6. **单实例锁**:flock 保证同一机器同时只有一个 hunter 在守候。

## 后台运行建议

```bash
nohup gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py &
# 或 tmux 里直接前台跑,任务输出实时可见
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # 无需真卡
```
