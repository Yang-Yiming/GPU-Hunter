# gpu-hunter

守望 NVIDIA 显卡,一旦出现满足条件的**稳定空闲**卡,立即自动启动你的任务。

支持两种用法:

1. **单任务**:一条命令 + 卡数/偏好,等到卡就跑(经典场景);
2. **工作流**:一个 YAML 定义多任务队列,贪心抢空档并发执行,或严格按顺序执行。

## 安装

```bash
pip install -e .        # 依赖:nvidia-ml-py、PyYAML;Linux + NVIDIA 驱动
```

## 快速上手

### 单任务模式

```bash
# 等 1 张完全空闲的卡,空闲就跑 train.py
gpu-hunter -- python train.py

# 要 2 张卡,优先 GPU2、GPU3
gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py --epochs 100

# 每张卡需要 40GB 可用显存(不要求"完全空闲",但要求无计算进程)
gpu-hunter --mem-per-gpu 40G -- bash run.sh
```

单个 `xx.py` / `xx.sh` 会自动补解释器;`--` 之后的内容原样作为子命令执行。

### 工作流模式

```bash
gpu-hunter workflow.yaml
```

```yaml
# workflow.yaml
defaults:                # 全局缺省,任务里可逐项覆盖
  mem-per-gpu: 40G

schedule: greedy         # greedy(默认):贪心抢空档 | sequential:严格按列出顺序

interval: 5              # 轮询节拍秒
stable-rounds: 3         # 连续几轮空闲才出手
grace: 45                # 启动确认宽限期秒

jobs:
  - name: pretrain       # name 用于日志与汇总报告,缺省 job-1..N
    cmd: python train.py --epochs 100   # 字符串按 shlex 切分;也接受列表 [python, ...]
    gpu-num: 4           # 需要几张同时空闲的卡
    prefer: [0, 1]       # 排序偏好(软约束,不够就用别的卡)

  - name: eval-a
    cmd: bash eval.sh --dataset a
    allow: [7]           # 白名单(硬约束,宁可死等)
    env: { DEBUG: "1" }  # 注入环境变量
    log: logs/eval_a.log # 缺省自动生成 logs/<name>-<时间戳>.log;"-" 表示透传终端

  - name: quick-check
    cmd: ./check.sh
    gpu-num: 1           # 不写的字段继承 defaults
```

#### 调度语义

| 模式 | 行为 |
|---|---|
| `greedy`(默认) | 每个节拍扫描全部待跑任务:**凑不齐的跳过继续扫后面的**(大任务不阻塞小任务);资源够时同一节拍可连续发射多个,各占互不重叠的卡 |
| `sequential` | 只盯队首,跑完(含失败)才放行下一个 |

失败处理:记录后继续队列;工作流结束时打印汇总表,**退出码 = 失败任务数**(全成功为 0)。被抢占的任务回滚并重新排队,`try` 列可看到尝试次数。

## 单任务参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--gpu-num N` | 1 | 需要几张同时空闲的卡 |
| `--mem-per-gpu SIZE` | - | 单卡显存需求,如 `40G`;缺省要求卡完全空闲 |
| `--prefer LIST` | - | 软偏好:`2,3` 先挑 2 再挑 3,不够用别的兜底 |
| `--allow LIST` | 全部 | 硬白名单:只在列出的卡里挑 |
| `--interval SEC` / `--stable-rounds K` | 5 / 3 | 轮询间隔 / 稳定窗口 |
| `--idle-mem SIZE` | 500M | 占用低于该值视为空闲(还需利用率 < 5%、无计算进程) |
| `--grace SEC` / `--heartbeat SEC` | 45 / 120 | 启动确认宽限期 / 运行状态心跳 |
| `--log FILE` / `--cwd DIR` | 终端透传 / 当前目录 | 任务输出重定向 / 子进程工作目录 |

prefer 与 allow 的区别:

```
全部 GPU → allow 过滤(硬) → 稳定空闲筛选 → prefer 排序(软) → 取前 N 张
```

- `--prefer 3,2`:3 最优先、2 次之,它们没空就拿别的;
- `--allow 2,3`:只准用 2/3,别的空着也不碰;
- 多卡按队首贪婪取用:`--prefer 3 --gpu-num 2` 且空闲 {0,2,3} → 拿 {3,0}。

## 工作机制

1. **探测**:走 NVML(`nvidia-smi` 底层库),不解析文本输出。
2. **稳定窗**:滑动窗口连续 K 轮空闲才出手;多卡原子凑齐。
3. **启动**:注入 `CUDA_VISIBLE_DEVICES`,独立会话拉起子进程;工作流内由资源账本保证并发任务卡不重叠。
4. **抢占校验**:宽限期内回查目标卡上的计算进程。见到自家 pid 即放行(torchrun worker 不会误杀);陌生进程持续出现且始终不见自家 → 回滚重排队;子进程提前退出则立即结束校验;容器内 pid 不可见时降级为带警告继续。
5. **信号**:SIGINT/SIGTERM 转发给全部子进程组;5 秒内连按两次直接 SIGKILL;Ctrl-C 会带走整个工作流的所有子任务。
6. **单实例锁**:flock 保证一台机器同时只有一个 hunter。

## 后台运行建议

```bash
nohup gpu-hunter --gpu-num 2 --prefer 2,3 --log runs/train.log -- python train.py &
nohup gpu-hunter workflow.yaml > wf.out 2>&1 &   # 任务输出在各自 log 文件里
```

## 测试

```bash
PYTHONPATH=src python3 -m unittest discover -s tests   # 无需真卡
```
