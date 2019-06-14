# 硕士学位论文：基于深度学习的中文文本自动校对研究与实现
1. 论文：[基于深度学习的中文文本自动校对研究与实现](https://github.com/young-zonglin/zlyang-master-thesis-codes/blob/master/docs/)
2. [环境设置](https://github.com/young-zonglin/zlyang-master-thesis-codes#环境设置)
3. [数据预处理](https://github.com/young-zonglin/zlyang-master-thesis-codes#数据预处理)
4. [基于卷积编解码网络的中文校对模型](https://github.com/young-zonglin/zlyang-master-thesis-codes#基于卷积编解码网络的中文校对模型)
5. [基于集成解码和重排序的中文文本自动校对](https://github.com/young-zonglin/zlyang-master-thesis-codes#基于集成解码和重排序的中文文本自动校对)
6. [基于多通道融合与重排序的中文文本自动校对](https://github.com/young-zonglin/zlyang-master-thesis-codes#基于多通道融合与重排序的中文文本自动校对)
7. [致谢](https://github.com/young-zonglin/zlyang-master-thesis-codes#致谢)
8. [License](https://github.com/young-zonglin/zlyang-master-thesis-codes#License)


## 环境设置
1. 克隆本项目：`git clone https://github.com/young-zonglin/zlyang-master-thesis-codes.git`
2. 安装所需依赖：
    * [Anaconda3-4.4.0-Linux-x86_64（Python 3.6.1）](https://repo.continuum.io/archive/Anaconda3-4.4.0-Linux-x86_64.sh)
    * [fairseq-py-extended](https://github.com/young-zonglin/fairseq-extended.git)：这是我`fork`的`fairseq-py`，这也是本项目所使用的具体版本（`fairseq 0.6.0`）。
    * [subword-nmt](https://github.com/rsennrich/subword-nmt)：`training/preprocess_yangzl.sh`使用它训练BPE模型。
    * [nbest-reranker](https://github.com/nusnlp/nbest-reranker/)：用于重排序（本项目已包含该软件包，位于`software/`目录）；还需要安装 [KenLM](https://github.com/kpu/kenlm) Python模块，用于获取句子的LM分数；还需要安装 [Moses SMT工具包](https://github.com/moses-smt/mosesdecoder)，基于MERT算法训练重打分器。
    * [m2scorer](https://github.com/nusnlp/m2scorer)：`software/`目录下的`m2scorer/scripts/edit_creator.py`用于将平行数据转为m2格式；`eval/`目录下的`m2scorer/scripts/m2scorer.py`（本项目包含该软件包）用于计算标准性能评估指标`M^2 F_0.5`。
    
      **注意**：进入`software/`目录，运行`download.sh`下载指定版本的`subword-nmt`和`m2scorer`软件包。
    * [libgrass-ui工具包](http://59.108.48.12/lcwm/pkunlp/downloads/libgrass-ui.tar.gz)：即pkunlp分词工具，m2格式的官方参考编辑基于pkunlp切分句子。
    * 词向量训练工具包（可选+按需安装）：
      - [word2vec官方训练工具包](https://code.google.com/archive/p/word2vec/)
      - [wang2vec官方训练工具包](https://github.com/wlin12/wang2vec)
      - [cw2vec开源实现工具包](https://github.com/bamtercelboo/cw2vec)
    * 数据预处理环节所需依赖：
      - [OpenCC](https://github.com/BYVoid/OpenCC)：用于将繁体中文转为简体中文。
      - [结巴分词工具](https://github.com/fxsjy/jieba) 和 [jieba_fast分词工具包](https://github.com/deepcs233/jieba_fast)。
      - [click命令行工具包](https://github.com/pallets/click)、[tqdm进度条工具包](https://github.com/tqdm/tqdm) 和 [smart_open](https://github.com/RaRe-Technologies/smart_open)。
      
      
## 数据预处理
1. NLPCC 2018 GEC训练集和数据处理
    1. 下载 [NLPCC2018-GEC](http://tcci.ccf.org.cn/conference/2018/taskdata.php) 训练平行语料。
    2. 解压后将`data.train`放到`${PROJECT_ROOT}/data/nlpcc-2018-traindata/`目录中。
    3. 通过`data/nlpcc2018_gec_process.py`预处理NLPCC 2018 GEC训练数据，进入`data/`目录，运行`python nlpcc2018_gec_process.py --help`获取详细信息。
    4. `training/one_script_to_run_all.sh`会调用`data/nlpcc2018_gec_process.py`自动完成NLPCC 2018 GEC平行数据的预处理工作。
2. 汉语水平考试（HSK）平行语料
    1. 下载 [HSK](https://pan.baidu.com/s/18JXm1KGmRu3Pe45jt2sYBQ) 训练平行语料，用于数据扩增。
    2. 将`hsk.src`和`hsk.trg`放到`${PROJECT_ROOT}/data/zh-hsk/`目录中。
    3. HSK语料需先移除空格，再按当前的建模等级进行重切分，进入`data/`目录，运行`python preprocessing.py segment-src-trg --help`获取详细信息。
    4. `training/one_script_to_run_all.sh`会调用`data/preprocessing.py`自动完成HSK平行数据的预处理工作。
3. 维基百科中文语料和数据处理
    1. 下载 [wiki-zh](https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2) 中文维基百科语料，用于预训练词向量和统计N-gram语言模型。
    2. 使用 [gensim工具包](https://github.com/RaRe-Technologies/gensim) 的`gensim.scripts.segment_wiki`模块将中文维基百科语料从XML格式转为json格式（wiki.json.gz）：
        ```bash
        python -m gensim.scripts.segment_wiki -i -f ${PROJECT_ROOT}/data/lm-emb-traindata/zhwiki-latest-pages-articles.xml.bz2 -o ${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.json.gz
        ```
    3. 按字切分维基百科中文语料，用于训练中文字向量和字级别的语言模型。
        ```bash
        python preprocessing.py segment_wiki --input-file=${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.json.gz --output-file=${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.char.seg.txt --char-level=True
        ```
    4. 按词切分，用于预训练词级别的N-gram LM。
        ```bash
        python preprocessing.py segment_wiki --input-file=${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.json.gz --output-file=${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.jieba.seg.word.txt
        ```
    5. 应用BPE切分，用于预训练中文BPE token的嵌入表示。
        1. `training/one_script_to_run_all.sh`训练BPE级别的校对模型时会生成BPE模型，位于`training/models/zh_bpe_model_${fusion_mode}_${how_to_remove}/train.bpe.model`。
        2. 将中文维基语料按词切分，命令如上所示。
        3. 应用子词切分，将词序列转为BPE序列。
            ```bash
            ${PROJECT_ROOT}/scripts/apply_bpe.py -c ${BPE_MODEL} < ${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.jieba.seg.word.txt > ${PROJECT_ROOT}/data/lm-emb-traindata/wikiyear-month-day.jieba.seg.bpe.txt
            ```


## 基于卷积编解码网络的中文校对模型

### Pre-training Word Embeddings

#### Pre-training Chinese Word2vec Embeddings
TODO

#### Pre-training Chinese Wang2vec Embeddings
TODO

#### Pre-training Chinese Cw2vec Embeddings
TODO

### Training a Single Model
Go to `training/` directory and use `one_script_to_run_all.sh` to train the model by specifying model architecture, model level, which pre-trained embeddings to use and data fusion mode.

### Data Cleaning Experiments
Go to `training/` directory and use `tune_long_low_high.sh` to determine filtering thresholds including `long`, `low` and `high` parameters.


## 基于集成解码和重排序的中文文本自动校对

### Training Single Models with Different Random Seeds
Go to `training/` directory and use `tune_random_seed.sh` to train single models with different random number seeds.

### Pre-training 5-Gram Language Models
TODO

### Ensemble Decoding + Re-ranking Mechanism
Reproduce experiments of chapter 5 of my master's dissertation with `training/rerank_experiment.sh`. The `rerank_experiment.sh` shell script trains the re-ranker calling `training/train_reranker.sh` firstly, and then applies the re-ranking mechanism via `training/run_trained_model.sh`. Go to `training/` directory and run `rerank_experiment.sh` script directly in the terminal for more details.


## 基于多通道融合与重排序的中文文本自动校对

### Multi-channel Fusion Framework + Re-ranking Mechanism
Multi-channel fusion and re-ranking using `training/multi_channel_fusion_experiment.sh`. The `multi_channel_fusion_experiment.sh` bash script trains the re-ranker components calling `training/train_multi_channel_fusion_reranker.sh` firstly, and then multi-channel fusion and applies the re-ranking mechanism via `training/multi_channel_fusion.sh`. Go to `training/` directory and run `multi_channel_fusion_experiment.sh` script directly in the terminal for more details.

### Reproduce Experiments of Chapter 6 of My Master's Dissertation
Reproduce experiments of chapter 6 of my master's dissertation with `training/all_experiments_multi_channel_fusion.sh`.


## 致谢


## License

