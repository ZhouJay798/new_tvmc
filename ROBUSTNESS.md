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
