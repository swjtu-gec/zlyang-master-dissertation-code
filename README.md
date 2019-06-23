# 硕士学位论文：基于深度学习的中文文本自动校对研究与实现
## 内容
1. 论文：[基于深度学习的中文文本自动校对研究与实现](https://github.com/young-zonglin/zlyang-master-thesis-codes/blob/master/docs/基于深度学习的中文文本自动校对研究与实现.pdf)
2. [环境设置](#环境设置)
3. [数据预处理](#数据预处理)
4. [基于卷积编解码网络的中文校对模型](#基于卷积编解码网络的中文校对模型)
5. [基于集成解码和重排序的中文文本自动校对](#基于集成解码和重排序的中文文本自动校对)
6. [基于多通道融合与重排序的中文文本自动校对](#基于多通道融合与重排序的中文文本自动校对)
7. [引用](#引用)
8. [License](#License)
9. [致谢](#致谢)


## 环境设置
1. 克隆本项目：`git clone https://github.com/young-zonglin/zlyang-master-thesis-codes.git`
2. 安装所需依赖：
    * [Anaconda3-4.4.0-Linux-x86_64（Python 3.6.1）](https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh)
    * [fairseq-py-extended](https://github.com/young-zonglin/fairseq-extended.git)：这是本文`fork`的`fairseq-py`，也是本项目所使用的具体版本（`fairseq 0.6.0`）。编译安装完成后，还需修改`path.sh`中的`FAIRSEQPY`，使其指向fairseq项目目录。
    * [subword-nmt](https://github.com/rsennrich/subword-nmt)：`training/preprocess_yangzl.sh`使用它训练BPE模型。
    * [nbest-reranker](https://github.com/nusnlp/nbest-reranker/)：用于重排序（本项目已包含该软件包，位于`software/`目录）；还需要安装 [KenLM](https://github.com/kpu/kenlm) Python模块，用于获取句子的LM分数；还需要安装 [Moses SMT工具包](https://github.com/moses-smt/mosesdecoder)，基于MERT算法训练重打分器。
    * [m2scorer](https://github.com/nusnlp/m2scorer)：`software/`目录下的`m2scorer/scripts/edit_creator.py`用于将平行数据转为m2格式；`eval/`目录下的`m2scorer/scripts/m2scorer.py`（本项目包含该软件包）用于计算标准性能评估指标`M^2 F_0.5`。
    
      **注意**：进入`software/`目录，运行`download.sh`即可下载指定版本的`subword-nmt`和`m2scorer`软件包。
    * [libgrass-ui工具包](http://59.108.48.12/lcwm/pkunlp/downloads/libgrass-ui.tar.gz)：即pkunlp分词工具，m2格式的官方参考编辑基于pkunlp切分句子。
    * 词向量训练工具包（可选+按需安装）：
      - [word2vec官方训练工具包](https://code.google.com/archive/p/word2vec/)
      - [wang2vec官方训练工具包](https://github.com/wlin12/wang2vec)
      - [cw2vec开源实现工具包](https://github.com/bamtercelboo/cw2vec)
    * [KenLM语言建模工具包](https://github.com/kpu/kenlm)：用于训练N-gram语言模型。
    * 数据预处理环节所需依赖：
      - [OpenCC](https://github.com/BYVoid/OpenCC)：用于将繁体中文转为简体中文。
      - [结巴分词工具](https://github.com/fxsjy/jieba) 和 [jieba_fast分词工具包](https://github.com/deepcs233/jieba_fast)。
      - [click命令行工具包](https://github.com/pallets/click)、[tqdm进度条工具包](https://github.com/tqdm/tqdm) 和 [smart_open](https://github.com/RaRe-Technologies/smart_open)。


## 数据预处理
### NLPCC 2018 GEC训练集和数据处理
1. 下载 [NLPCC2018-GEC](http://tcci.ccf.org.cn/conference/2018/taskdata.php) 训练平行语料。
2. 解压后将`data.train`放到`${PROJECT_ROOT}/data/nlpcc-2018-traindata/`目录中。
3. 通过`data/nlpcc2018_gec_process.py`预处理NLPCC 2018 GEC训练数据，进入`data/`目录，执行命令`python nlpcc2018_gec_process.py --help`获取详细信息。
4. `training/one_script_to_run_all.sh`会调用`data/nlpcc2018_gec_process.py`自动完成NLPCC 2018 GEC平行数据的预处理工作。
### 汉语水平考试（HSK）平行语料
1. 下载 [HSK](https://pan.baidu.com/s/18JXm1KGmRu3Pe45jt2sYBQ) 训练平行语料，用于数据扩增。
2. 将`hsk.src`和`hsk.trg`放到`${PROJECT_ROOT}/data/zh-hsk/`目录中。
3. HSK语料需先移除空格，再按当前的建模等级进行重切分，进入`data/`目录，执行命令`python preprocessing.py segment-src-trg --help`获取详细信息。
4. `training/one_script_to_run_all.sh`会调用`data/preprocessing.py`自动完成HSK平行数据的预处理工作。
### 维基百科中文语料和数据处理
1. 下载 [wiki-zh](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2) 中文维基百科语料，用于训练词向量和统计N-gram语言模型。
2. 使用 [gensim工具包](https://github.com/RaRe-Technologies/gensim) 的`gensim.scripts.segment_wiki`模块将中文维基百科语料从XML格式转为json格式（wiki.json.gz）：
    ```bash
    year=2018 && month=11 && day=14
    python -m gensim.scripts.segment_wiki \
        -i -f ${PROJECT_ROOT}/data/lm-emb-traindata/zhwiki-latest-pages-articles.xml.bz2 \
        -o ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.json.gz
    ```
3. 按字切分维基百科中文语料，用于训练**中文字向量**和**字级别的语言模型**。
    ```bash
    python preprocessing.py segment-wiki \
        --input-file=${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.json.gz \
        --output-file=${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.char.seg.txt \
        --char-level=True
    ```
4. 按词切分，用于训练**词级别的N-gram LM**。
    ```bash
    python preprocessing.py segment-wiki \
        --input-file=${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.json.gz \
        --output-file=${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.word.txt
    ```
5. 应用BPE切分，用于预训练**中文BPE token的嵌入表示**。
    1. `training/one_script_to_run_all.sh`训练BPE级别的校对模型时会生成BPE模型，位于`training/models/zh_bpe_model_${fusion_mode}_${how_to_remove}/train.bpe.model`。
    2. 将中文维基语料按词切分，命令如前所述。
    3. 应用子词切分，将词序列转为BPE序列。
        ```bash
        ${PROJECT_ROOT}/scripts/apply_bpe.py -c ${BPE_MODEL} \
            < ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.word.txt \
            > ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.bpe.txt
        ```


## 基于卷积编解码网络的中文校对模型
### 预训练不同级别的嵌入表示
1. 基于word2vec训练字和BPE的嵌入表示
    1. 训练字和BPE表示的命令除了训练语料与结果向量存储文件不同，其余完全相同。
    2. 如前所述，预处理维基百科中文语料，将其切分为字和BPE级别。
    3. 字向量训练命令为：
        ```bash
        ./word2vec -train ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.char.seg.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.char.word2vec.skipgram.ns.500d.txt \
            -size 500 -window 10 -sample 1e-4 -hs 0 -negative 10 -threads 10 -iter 5 -binary 0 -cbow 0
        ```
    4. BPE向量训练命令为：
         ```bash
        ./word2vec -train ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.bpe.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.jieba.seg.bpe.word2vec.skipgram.ns.500d.txt \
            -size 500 -window 10 -sample 1e-4 -hs 0 -negative 10 -threads 10 -iter 5 -binary 0 -cbow 0
        ```
2. 基于wang2vec训练字和BPE的嵌入表示
    1. 训练字和BPE表示的命令除了训练语料与结果向量存储文件不同，其余完全相同。
    2. 字向量训练命令为：
        ```bash
        ./word2vec -train ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.char.seg.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.char.structured.skipngram.500d.txt \
            -size 500 -window 10 -sample 1e-4 -hs 0 -negative 10 -nce 0 -threads 10 -iter 5 -binary 0 -type 3 -cap 0
        ```
    3. BPE向量训练命令为：
        ```bash
        ./word2vec -train ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.bpe.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.jieba.seg.bpe.structured.skipngram.500d.txt \
            -size 500 -window 10 -sample 1e-4 -hs 0 -negative 10 -nce 0 -threads 10 -iter 5 -binary 0 -type 3 -cap 0
        ```
3. 基于cw2vec训练字和BPE的嵌入表示
    1. 基于cw2vec训练字和BPE表示的命令除了训练语料、结果向量存储文件和**笔画n-gram最大长度**不同，其余完全相同。
    2. 字向量训练命令为：
        ```bash
        ./word2vec substoke -input ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.char.seg.txt \
            -infeature ${cw2vec_project}/Simplified_Chinese_Feature/sin_chinese_feature.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.char.cw2vec.500d.txt \
            -lr 0.025 -dim 500 -ws 10 -epoch 5 -minCount 10 -neg 10 -loss ns \
            -minn 3 -maxn 6 -thread 10 -t 1e-4 -lrUpdateRate 100
        ```
    3. BPE向量训练命令为：
        ```bash
        ./word2vec substoke -input ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.bpe.txt \
            -infeature ${cw2vec_project}/Simplified_Chinese_Feature/sin_chinese_feature.txt \
            -output ${PROJECT_ROOT}/data/embeddings/wiki.zh.jian.jieba.seg.bpe.cw2vec.500d.txt \
            -lr 0.025 -dim 500 -ws 10 -epoch 5 -minCount 10 -neg 10 -loss ns \
            -minn 3 -maxn 9 -thread 10 -t 1e-4 -lrUpdateRate 100
        ```
### 训练单模型
进入`training/`目录，运行`one_script_to_run_all.sh`：
```
./one_script_to_run_all.sh <model_arch> <model_level> <which_pretrained_embed> <fusion_mode> \
    <short> <long> <low> <high> \
    <src_vocab_size> <trg_vocab_size> <GPU_used_training> <GPU_used_inference> \
    <MAX_TOKENS> <MAX_SENS> <random_seed> <want_ensemble> <force_redo_remove_same_and_seg>
```
- `<model_arch>`：支持三种seq2seq模型，lstm即LSTM seq2seq模型，fconv即ConvS2S模型，transformer即Transformer模型。
- `<model_level>`：支持三种建模级别，word即词级别建模，bpe即BPE级别，char即字级别。
- `<which_pretrained_embed>`：支持三种预训练词向量和嵌入矩阵随机初始化：
    1. 若`<model_level>` == word，则可取`random`和`wang2vec`。
    2. 若`<model_level>` == bpe，则可取`blcu-wang2vec`（来自于 [BLCU-GEC-2018](https://github.com/blcu-nlp/NLPCC_2018_TASK2_GEC) 的 [Zlbnlp_data](https://pan.baidu.com/s/18JXm1KGmRu3Pe45jt2sYBQ)）、`wiki-wang2vec`（训练语料为中文维基百科语料）、`cw2vec-vec`（从笔画n-gram和词混合嵌入矩阵中直接拿到token的嵌入表示）、`cw2vec-avg`（将笔画n-gram向量平均的结果作为词向量）、`word2vec`和`random`。
    3. 若`<model_level>` == char，则可取`wang2vec`（训练语料为中文维基百科语料）、`cw2vec-vec`、`cw2vec-avg`、`word2vec`和`random`。
    4. 需确保相应的预训练词向量模型存在，否则`one_script_to_run_all.sh`脚本会退出。
- `<fusion_mode>`：支持三种模式，1表示仅使用NLPCC 2018 GEC训练数据，2表示NLPCC+HSK，模式3不使用。
- `<short>`：源端或目标端长度小于阈值`short`的句子对会被移除，本文设置`short` = 1，即不移除任何短句。
- `<long>`：源端或目标端长度大于阈值`long`的句子对会被移除，本文设置`long` >= 1000，即不移除任何长句。
- `<low>`：目标端与源端长度比小于阈值`low`的句子对会被移除，本文通过“数据清洗实验”确定`low`的最佳取值为0.1。
- `<high>`：目标端与源端长度比大于阈值`high`的句子对会被移除，本文设置`high` = 200.0（需为浮点数），即不移除目标端长度远大于源端的句子对。
- `<src_vocab_size>`：源端词汇表规模，词级别模型设为69677，BPE级别模型设为37000，字级别模型设为6000。
- `<trg_vocab_size>`：目标端词汇表规模，词级别模型设为65591，BPE级别模型设为37000，字级别模型设为6000。
- `<GPU_used_training>`：本文基于单卡训练单模型，取值为GPU卡的ID。
- `<GPU_used_inference>`：一般同`<GPU_used_training>`。
- `<MAX_TOKENS>`：一个批的数据至多包含多少token。词级别的模型设为3000，字和BPE级别的模型均设为6000。
- `<MAX_SENS>`：一个批的数据至多包含多少句子对，即batch size。词级别的模型设为30，字和BPE级别的模型均设为60。
- `<random_seed>`：随机数种子，用于网络权重的初始化。一般可设为1。
- `<want_ensemble>`：是否基于各检出点进行集成解码。由于最初几轮迭代模型还不够好，这种集成方式性能可能会掉，故一般设置为false。
- `<force_redo_remove_same_and_seg>`：如果预处理流程有变，则设为true，否则设为false，可以节省时间。
### 数据清洗实验
1. 进入`training/`目录，运行`tune_long_low_high.sh`：
    ```
    ./tune_long_low_high.sh <model_arch> <model_level> <which_pretrained_embed> <fusion_mode> \
        <params_file> <src_vocab_size> <trg_vocab_size> <GPU_used_training> <GPU_used_inference> \
        <MAX_TOKENS> <MAX_SENS> <random_seed> <want_ensemble> <force_redo_remove_same_and_seg>
    ```
    - `<params_file>`：存放`short`、`long`、`low`和`high`取值的文件。
    每一行都包括四个字段，分别为`short`、`long`、`low`和`high`的取值，字段分隔符可以为","、"\t"或者" "。
    `training/all_params_to_try_short_long_low_high`存储了本文"remove-long"、"remove-low"和"remove-high"三个实验所搜索的参数值。
    - 其他参数同`one_script_to_run_all.sh`。
2. 本文进行的数据清洗实验表明：
- "remove-long"和"remove-high"未能提升性能。
- "remove-low"的最佳过滤阈值为`low`=0.1，基准测试集的性能获得了有效提升。 


## 基于集成解码和重排序的中文文本自动校对
### 训练不同种子初始化的单模型
进入`training/`目录，运行`tune_random_seed.sh`：
```
./tune_random_seed.sh <model_arch> <model_level> <which_pretrained_embed> <fusion_mode> \
    <short> <long> <low> <high> \
    <src_vocab_size> <trg_vocab_size> <GPU_used_training> <GPU_used_inference> \
    <MAX_TOKENS> <MAX_SENS> <try_random_seed> <want_ensemble> <force_redo_remove_same_and_seg>
```
- `<try_random_seed>`：一系列随机数种子，空格分隔，例如：'1 2 3 4'。
会遍历这些随机数种子，依次训练不同初始化的各单模型。
- 其他参数同`one_script_to_run_all.sh`。
### 训练5-gram语言模型
1. 用途
- 词级别的语言模型仅用于BPE级别校对模型N-best输出的重排序。
- 字级别的LM用于字级别校对模型的重排序和多通道融合与重排序架构。
2. 训练
    1. 如前所述，预处理维基百科中文语料，将其切分为字和BPE级别。
    2. 进入`KenLM`项目，估计语言模型参数：
        1. 字级别的模型
            ```bash
            bin/lmplz -o 5 -S 60% -T /tmp \
                < ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.char.seg.txt \
                > ${PROJECT_ROOT}/training/models/lm/wiki_zh.char.5gram.arpa
            ```
        2. 词级别的模型
            ```bash
            bin/lmplz -o 5 -S 60% -T /tmp \
                < ${PROJECT_ROOT}/data/lm-emb-traindata/wiki${year}-${month}-${day}.jieba.seg.word.txt \
                > ${PROJECT_ROOT}/training/models/lm/wiki_zh.word.5gram.arpa
            ```
    3. 进入`KenLM`项目，将ARPA格式文件转为trie结构的二进制数据：
        1. 字级别的模型
            ```bash
            bin/build_binary -T /tmp/trie -S 6G trie \
                ${PROJECT_ROOT}/training/models/lm/wiki_zh.char.5gram.arpa \
                ${PROJECT_ROOT}/training/models/lm/wiki_zh.char.5gram.binary.trie
            ```
        2. 词级别的模型
            ```bash
            bin/build_binary -T /tmp/trie -S 6G trie \
                ${PROJECT_ROOT}/training/models/lm/wiki_zh.word.5gram.arpa \
                ${PROJECT_ROOT}/training/models/lm/wiki_zh.word.5gram.binary.trie
            ```
### 模型推断+重排序组件
1. 训练重打分器。进入`training/`目录，运行`train_reranker.sh`：
    ```
    ./train_reranker.sh <dev_data_dir> <train_reranker_output_dir> <device> \
        <model_path> <reranker_feats> <moses_path> \
        <DATA_BIN_DIR> <BPE_MODEL_DIR> <lm_url>
    ```
    - `<dev_data_dir>`：验证集的源端输入（`dev.input.txt`）和m2格式的参考编辑（`dev.m2`）所在的目录，
    为`training/`目录下的`processed_dir`，（`processed_dir`由`one_script_to_run_all.sh`自动生成）。
    - `<train_reranker_output_dir>`：用于存放训练重打分器时生成的文件。
    - `<device>`：指定模型推断时使用的GPU卡的ID。
    - `<model_path>`：支持单模型推断与集成解码：
        1. 若为单模型（`xxx.pt`）的url，则为单模型推断。
        2. 若为目录，则使用该目录下的所有模型进行集成解码。
    - `<reranker_feats>`：支持三种重排序方法：
        1. `eo`：为编辑操作特征重排序。
        2. `lm`：为语言模型特征重排序。
        3. `eolm`：为EO特征+LM特征重排序。
    - `<moses_path>`：[Moses](https://github.com/moses-smt/mosesdecoder) 项目的路径。
    - `<DATA_BIN_DIR>`：`fairseq`的`preprocess.py`的输出目录，`one_script_to_run_all.sh`会自动生成，为`processed_dir/bin`。
    - `<BPE_MODEL_DIR>`：指定BPE模型的所在目录：
        1. 若为BPE级别的校对模型，设为`training/models/zh_bpe_model_${fusion_mode}_${how_to_remove}/`，`one_script_to_run_all.sh`会自动生成。
        2. 若为词级别或字级别的校对模型，设为`None`即可。
    - `<lm_url>`：指定5-gram语言模型的URL：
        1. 若为词级别或BPE级别的校对模型，则为词级别的5-gram LM。
        2. 若为字级别的校对模型，则为字级别的5-gram LM。
2. 模型推断。进入`training/`目录，运行`run_trained_model.sh`：
    ```
    ./run_trained_model.sh <test_input> <run_models_output_dir> <device> \
        <model_path> <DATA_BIN_DIR> <BPE_MODEL_DIR> \
    ```
    - `<test_input>`：NLPCC 2018 GEC基准测试集（`data/test/nlpcc2018-test/source.txt`）或其他源端输入的URL。
    - `<run_models_output_dir>`：用于存放模型推断时生成的文件。
    - 其他参数同`train_reranker.sh`。
3. 模型推断+应用重排序机制。进入`training/`目录，运行`run_trained_model.sh`：
    ```
    ./run_trained_model.sh <test_input> <run_models_output_dir> <device> \
        <model_path> <DATA_BIN_DIR> <BPE_MODEL_DIR> \
        <reranker_weights> <reranker_feats> <lm_url>
    ```
    - `<reranker_weights>`：存放特征权重的文件，为`<train_reranker_output_dir>/weights.${reranker_feats}.txt`。
    - 其他参数同`train_reranker.sh`和`run_trained_model.sh`。
4. 调用`training/rerank_experiment.sh`，可一步到位完成：重打分器的训练 => 模型推断+重排序 => 基于基准测试集计算校对系统的标准性能指标。
但需要修改一些变量的取值。


## 基于多通道融合与重排序的中文文本自动校对
### 多通道融合与重排序架构
Multi-channel fusion and re-ranking using `training/multi_channel_fusion_experiment.sh`. The `multi_channel_fusion_experiment.sh` bash script trains the re-ranker components calling `training/train_multi_channel_fusion_reranker.sh` firstly, and then multi-channel fusion and applies the re-ranking mechanism via `training/multi_channel_fusion.sh`. Go to `training/` directory and run `multi_channel_fusion_experiment.sh` script directly in the terminal for more details.
### 复现学位论文第六章的实验
Reproduce experiments of chapter 6 of my master's dissertation with `training/all_experiments_multi_channel_fusion.sh`.


## 引用
```
杨宗霖. 基于深度学习的中文文本自动校对研究与实现[D]. 西南交通大学, 2019.
```


## License
与 [nusnlp/mlconvgec2018](https://github.com/nusnlp/mlconvgec2018) 一致，均为GPL-3.0。


## 致谢
- [nusnlp/mlconvgec2018](https://github.com/nusnlp/mlconvgec2018)
- [pytorch/fairseq](https://github.com/pytorch/fairseq)
- [blcu-nlp/NLPCC_2018_TASK2_GEC](https://github.com/blcu-nlp/NLPCC_2018_TASK2_GEC)

