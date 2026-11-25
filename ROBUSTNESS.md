# Qwen2.5-1.5B 签收文件的变换实验

历史 12 个格子的入口仍为 `stages/stage4_robustness.py`。新的扫描使用相同的 Python 3.12、torch 2.8/cu128 和项目 requirements；在仓库根目录运行。准备已经通过阶段 3 的整个目录，包含 journal.jsonl、report.json、owner_model.zip 和三份 received/U*/model.zip；默认的基座阴性对照还需要 assets.py 准备好的官方模型目录。旧扫描使用 calib；新扫描使用 ordinary_splits 的 eval，在同一批文本上重测基线。

## 单项和组合

| 案例 | 操作 |
|---|---|
| original、bf16 | 原样另存及重载；整模型 BF16 转存及重载 |
| prune5/10/20/30/40 | 每个块的注意力及 MLP 线性矩阵分别按幅值剪枝；包含 down_proj。阈值相等时实测稀疏率可能略高于目标 |
| int8、int4 | 所有块的注意力及 MLP 线性权重，每 64 个输入列共享一个 FP32 scale，对称权重量化。INT4 每字节打包两值，保留偏置、词嵌入与输出头。参考推理后端从磁盘加载整数码，逐层反量化后做浮点乘法。请注明该实现；吞吐不代表 GPTQ/AWQ 等部署内核 |
| sft-r8、sft-r16 | 所有块的 down_proj 上做 LoRA 回答监督训练，然后合并 |
| sft-1、sft-3、sft-7、sft-full | 最后 1、3、7 个块或全模型更新。表中记录真实可训练比例。默认 5 万输入 token，样本完整，末个样本可超预算 |
| data1/5/10/25/100 | 相同种子的打乱顺序取嵌套的 1%、5%、10%、25%、100% 指令样本，各一轮，固定 rank 16。使用足量不同样本使各比例可区分 |
| cpt10000/50000/200000/1000000 | 最后一块普通语言模型继续训练，相同种子沿同一文本顺序独立跑到各自目标。默认使用普通训练切分循环采样；可用 --cpt 传入独立 UTF-8 领域语料。记录实际输入和监督 token 数 |
| C1 / C2 | sft-r16 后接 int8 / int4 |
| C3 / C4 | prune20 后接 sft-1 并固定剪枝掩码 / 同预算 sft-1 后接 prune20 |
| C5 / C6 | cpt200000 → sft-r16 / 再接 int4 |

训练数据不含探针、码字及来源标签。SFT 使用自备 UTF-8 JSONL，每行包含字符串 instruction 和 response，训练仅对回答 token 计算损失。相同实验系列使用同一份文件并保留摘要。各训练样本最多 256 token；LoRA 学习率 1e-4，直接更新 1e-5。训练过程中不使用指纹挑选检查点。

## 云端执行

以下路径为占位路径。扫描使用数据盘并保留临时文件空间。每个变换步骤保存到磁盘，计算各文件 SHA-256，重新加载后评估，再清理临时模型。

```bash
python stages/stage4_matrix.py runs/lora-delivery-001 runs/scan-precision
python stages/stage4_matrix.py runs/lora-delivery-001 runs/scan-sft \
  --sft /root/autodl-tmp/instructions.jsonl --cases original,data1,data5,data10,data25,data100,sft-1,sft-r16 --seeds 17,29
python stages/stage4_matrix.py runs/lora-delivery-001 runs/scan-combined \
  --sft /root/autodl-tmp/instructions.jsonl --cases C1,C2,C3,C4,C5,C6 --seeds 17,29
```

可用 --sources U1,U2,U3,owner,base 选择来源，默认包含两个阴性对照。--cases all 会包含成本较高的全参训练和百万 token 训练。建议先分组运行；RTX 5090 32 GB 上尚未实测全参模式的峰值，按需单独调度。每个输出目录锁定一个计划；中断后使用原参数、原提交和原始证据重启，已完成的来源/种子/案例会跳过。不要把同一确定性变换重复运行的结果计作独立样本。

- plan.json 保存批次账本摘要、环境、代码版本、数据摘要和计划。
- measurements.csv 和 rows/*.json 保存每一步相对 owner 及相对本副本的质量、读回、位裕量、权重文件摘要、候选整数分数、尺度和实际整数阈值。可用 bit_sums 对照跟踪 U2 擦除位或 U3 翻转位。
- summary.csv 按终点统计，包含全部文件分母及质量通过子集的分母。confusion.csv 为真实来源与裁决结果。阴性样本以筛查拦截为预期结果。
- 位参考来自原始签收文件读回；入口要求三份交付误码均为零，检查签收文件与账本摘要，并拒绝带擦除的参考字。扫描复用已经披露的探针做配对分析。阈值命中采用与阶段 3 相同的定点公式。分数除以 scale 得到显示分数。

## 冻结方案后的新批次确认

每个正式确认案例启动一次全新发行，随机抽取探针池样本、生成新码字并随机选择 U1–U3 之一。变换阶段不读取秘密探针。原文件签收后生成 challenge.zip，计算摘要并承诺来源标签，再进入筛查和一次性门限归因。裁决后公开来源和盐，由 verify.py 在新进程核对承诺及时间顺序。各角色仍在单台云机上运行。筛查拦截会提前结束并记录失败；质量超限、错指和漏判都应计入确认试验分母。

```bash
python stages/stage3_delivery.py runs/confirm-c2-001 --challenge C2 \
  --sft /root/autodl-tmp/instructions.jsonl --seed 17
python verify.py runs/confirm-c2-001 --challenge --trust-sha256 "$(cat runs/confirm-c2-001/trust.sha256)"
```

外部复核使用事先另存的可信摘要。查看 challenge.json、report.json、delivery_audit.json 和 verification.json；保留失败运行的目录。每个条件建议先做至少三批，按实际抽到的来源报告计数。一个新批次此处仅调查一个文件。离线扫描三个来源和正式确认批次数分别计数。

可选出图：在含 matplotlib 的分析环境运行 `python stages/plot_robustness.py runs/scan-combined`。
工具按目录内已有数据生成强度曲线、组合步骤图及来源与裁决计数矩阵 PNG。部分完成的目录只包含已完成测量。

## INT4 双达标与双人合谋（Qwen2.5-1.5B-Instruct）

在项目根目录、原 CUDA 环境和数据盘内执行。先扫描同一批通过的签收文件，
同一份留出文本对比 owner 与三个副本的量化前后质量、存活位、分数、阈值及联合通过。
保持相同的可量化矩阵集合，缩小组会增加 scale 存储开销，比较 measurements.csv 的 artifact_bytes。
int4 是原始每组 64 列的对称 INT4；g32/g16 缩小分组；cal16 在相同的 16 列组上，
用独立普通训练切分的前 32 个片段估计输入二阶矩，从 5 个范围中逐组选取加权误差最小的 scale。
该算法是本项目的校准基线。全部低位条件均需保存整数码、重载后计分。

```bash
python stages/stage4_matrix.py runs/lora-delivery-001 runs/int4-study-001 \
  --sources owner,U1,U2,U3 --cases original,int8,int4,int4-g32,int4-g16,int4-cal16
```

若需要强化写入，在新发行中使用 --int4-aware 16（也支持 32、64）。
FP32 与模拟 INT4 的训练步交替，均使用位级损失和相对 owner 的 KL 约束；
模拟时其他投影为量化后的冻结权重，载体投影先合并 LoRA 再量化，并用直通梯度优化。
校准检查点要求两种精度都满足质量预算、模拟 INT4 误码率不高于 2%，优先保证干净副本零误码，然后最小化最差读回误码。
在 int4-calibration-U*.json 查看检查点记录。若无合格检查点则停止并保留失败记录。
模拟器针对对称最大值量化；cal16 的校准裁剪及未知外部量化器需另测。
以下命令在新批次上用真正打包的 g16 做一次新调查：

```bash
python stages/stage3_delivery.py runs/confirm-int4-aware-001 --int4-aware 16 --challenge int4-g16
```

评估未知再量化时也要对新接收文件重跑原始 int4（64 列）条件。stage4_matrix 的终点
joint_pass 要求质量达标及唯一正确归因；若研究目标还要求位存活率达到给定水平，请在跑前另外声明阈值。
不同发行批次有不同码字，不应作为同码字配对消融。新实验显存/耗时未经实测。

### 合谋边界

已有合法签收文件可直接跑以下探索实验，无需重新训练。取 U1/U2/U3 全部三种两两组合，
在相同 owner 和架构上对对应参数做线性混合；percent 为编号较小接收方的权重百分比。
默认 9 个混合文件，每个文件打包、重载、读回。从全体三个登记候选者中按原筛查与整数阈值输出 hits。
保存 coalitions.csv 和 summary.json；临时混合文件在测量后清理，CSV 保留文件哈希。

```bash
python stages/stage5_coalition.py runs/lora-delivery-001 runs/coalition-sweep-001 --percent 25 50 75
```

严格成功要求命中集合等于真实贡献者集合。分别查看 all_identified、any_identified、innocent_hit、
missed、recall、screen_matches 及 quality_ok。any_identified 可以与 innocent_hit 同时为真。
质量达标且完整找回的分母是所有预定混合文件。此扫描使用旧批次已披露探针，仅作离线诊断。

正式复核为每个混合文件建立新发行及新的调查组。如下模拟 U1 与 U2 平均混合，
保留 challenge.zip，先承诺其摘要及加盐来源标签，筛查放行后门限归因，裁决后打开标签。
用独立进程检查双方是否都被正确命中且没有 U3。裁决证明有效与完整找回是两项指标；
核验文件中的 proof_verified 和 verified 分别表示证明通过及结果与预期集合相符。
筛查拦截则在披露归因探针前终止。U1/U2 是登记副本来源，结果仅涉及混合模型的贡献来源。

```bash
python stages/stage3_delivery.py runs/confirm-U1-U2-001 --coalition 1 2 --first-percent 50
```

建议每次仅运行一个进程。旧批次扫描和新批次确认的分母需分别报告。
