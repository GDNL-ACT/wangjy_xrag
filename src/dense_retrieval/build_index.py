from colbert.infra import Run, RunConfig, ColBERTConfig
from colbert import Indexer

if __name__=='__main__':
    with Run().context(RunConfig(nranks=1, experiment="msmarco")):

        config = ColBERTConfig(
            nbits=2,
            root="./experiments",
        )
        indexer = Indexer(checkpoint="colbert-ir/colbertv2.0", config=config)
        indexer.index(name="msmarco.nbits=2", collection="/home/supie2/data/wangjy/xRAG/xRAG_old/data/pretrain/wikipedia/dev.tsv")